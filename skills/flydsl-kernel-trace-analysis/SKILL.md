---
name: flydsl-kernel-trace-analysis
description: >
  Profile GPU kernels using rocprofv3 to collect ATT instruction-level traces, then
  analyze the trace data using hotspot_analyzer.py to identify top-K stall hotspots
  (VMEM-load, VMEM-wait, LDS/SMEM-wait, barrier, MFMA stalls) mapped back to source
  lines, and produce an actionable optimization plan.
  Also analyzes rocprofv3 PMC counter CSVs (L2 hit rate, partial reads, over-fetch).
  Usage: /kernel-trace-analysis <cmd>
  Can also analyze an existing dispatch dir directly: /kernel-trace-analysis --dir <path>
allowed-tools: Read,Edit,Bash,Grep,Glob,Agent,Write
---

# Kernel Trace Analysis

Profile and analyze GPU kernel ATT traces to identify stall hotspots and produce
an optimization plan.

All analysis is done by running the bundled `hotspot_analyzer.py` (ATT) and
`pmc_l2_analyzer.py` (PMC CSVs); never recompute their metrics yourself.
Do **not** use GUI tools.

> If the question is only "did my change increase register pressure, spills, or
> LDS?", use `/isa-resource-diff` first — it is compile-only, needs no GPU or
> profiler run, and answers in seconds. Come here when you need to know *why* a
> kernel is slow rather than *what resources it uses*.
>
> To pick an LLVM knob that changes what the compiler emits — a compile hint, a
> function attribute, an `-mllvm` flag — and to prove it took effect, use
> `/llvm`. That is also compile-only.

## Arguments

| Argument | Description |
|----------|-------------|
| `<CMD>` | Command to profile. Example: `python bench_pa.py --batch 32` |
| `--dir <path>` | Skip collection; analyze existing `ui_output_agent_*_dispatch_*` directory |
| `--topk N` | Show top-N hotspots (default: 15) |

---

## Analyzer Scripts

- `${CLAUDE_SKILL_DIR}/scripts/hotspot_analyzer.py` — reads a `ui_output_agent_*_dispatch_*` ATT
  directory; reports top-K stall hotspots, stall-type breakdown, and occupancy
  (combined-VGPR-pool model, reads accum/LDS/SGPR from `out_kernel_trace.csv`).
- `${CLAUDE_SKILL_DIR}/scripts/pmc_l2_analyzer.py` — reads rocprofv3 PMC counter CSV(s); reports
  L2 hit rate, HBM 32B-partial fraction, and over-fetch ratio. Use when a
  kernel is memory-bound and you need to know *why* (ATT has no cache counters).
  See "L2 / HBM efficiency analysis" under Step 5.

---

## Workflow

### Mode A: Analyze existing dispatch directory

If the user provides `--dir <path>` or already has a `ui_output_agent_*_dispatch_*` directory:

```bash
# Both analyzers ship with this skill -- do not reimplement them.
python ${CLAUDE_SKILL_DIR}/scripts/hotspot_analyzer.py <dispatch_dir> --kernel <kernel> --topk 15 --mode both --detail --context 4
```

Skip to **Step 5: Interpret Results**.

---

### Mode B: Full collection workflow

#### Step 1: Kernel Discovery

```bash
touch /tmp/trace_ts
rocprofv3 --stats --kernel-trace -f csv -- <CMD> 2>&1
find . -maxdepth 3 -name "*stats*" -newer /tmp/trace_ts -type f 2>/dev/null
```

Parse the stats CSV and present a kernel table:

| Rank | Kernel Name | Calls | Total (us) | Avg (us) | % GPU Time |
|------|-------------|-------|------------|----------|------------|

Ask the user which kernel to trace if not obvious.

**Prefer `results.db`** if available — use sqlite3 for structured queries:
```bash
sqlite3 results.db "
SELECT ks.KernelName, COUNT(*) calls,
       ROUND(AVG(kd.end-kd.start)/1000.0,1) avg_us
FROM rocpd_kernel_dispatch kd
JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id=ks.id
GROUP BY ks.KernelName ORDER BY avg_us DESC LIMIT 20;"
```

#### Step 2: Configure input.yaml

Write `/tmp/trace_input.yaml` with the job below, setting
`kernel_include_regex` to the kernel found in Step 1:

```yaml
jobs:
   -
       kernel_include_regex: <KERNEL_NAME_PATTERN>
       kernel_iteration_range: "[1, [3-4]]"
       output_file: out
       output_directory: kernel_trace_output
       output_format: [csv]
       truncate_kernels: true
       sys_trace: true
       advanced_thread_trace: true
       att_target_cu: 1
       att_shader_engine_mask: "0xf"
       att_simd_select: "0xf"
       att_buffer_size: "0x6000000"
```

Key notes:
- `kernel_iteration_range`: `"[1, [3-4]]"` skips warmup, traces dispatches 3-4
- `att_buffer_size`: 96MB per SE; increase to `"0xC000000"` if truncated
- `att_target_cu: 1`: single CU keeps output manageable

#### Step 3: Collect ATT Trace

```bash
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 rocprofv3 -i /tmp/trace_input.yaml -- <CMD> 2>&1
find . -type d -name "ui_output_agent_*" -newer /tmp/trace_ts 2>/dev/null
```

If the `rocprof-trace-decoder` library is missing, install the release matching
the ROCm version in `/opt/rocm/.info/version`. Set `RTD_VERSION` to that release
before running the block below. **build-rocm-image** hardcodes its own
`RTD_VERSION` for the image it builds and does not derive it either; if the two
disagree, the ROCm version on the machine you are profiling wins:
```bash
RTD_VERSION=0.1.6   # <-- set to the release matching your ROCm version
RTD_INSTALLER="rocprof-trace-decoder-manylinux-2.28-${RTD_VERSION}-Linux.sh"
wget -q "https://github.com/ROCm/rocprof-trace-decoder/releases/download/${RTD_VERSION}/${RTD_INSTALLER}"
chmod +x "${RTD_INSTALLER}"
"./${RTD_INSTALLER}" --skip-license --prefix=/tmp/rtd-install
find /tmp/rtd-install -name '*.so*' -exec cp -a {} /opt/rocm/lib/ \;
ldconfig
```

**Output structure:**
```
ui_output_agent_<PID>_dispatch_<N>/
├── code.json          ← PRIMARY: per-instruction stall/cycle data
├── snapshots.json     ← source file path mapping (virtual → local filename)
├── source_0_*.py      ← embedded source files
├── filenames.json     ← wave file index
├── occupancy.json     ← occupancy timeline
└── se*_sm*_sl*_wv*.json  ← per-wave raw traces
```

---

## Step 4: Run hotspot_analyzer.py

The analyzer ships with this skill — do not reimplement it. Run:

```bash
# Full report
python ${CLAUDE_SKILL_DIR}/scripts/hotspot_analyzer.py <dispatch_dir> --topk 15 --mode both

# Source-level with code context (best for optimization)
python ${CLAUDE_SKILL_DIR}/scripts/hotspot_analyzer.py <dispatch_dir> --topk 5 --mode src --detail --context 4

# ASM-only for instruction-level detail
python ${CLAUDE_SKILL_DIR}/scripts/hotspot_analyzer.py <dispatch_dir> --mode asm --topk 20
```

---

## Step 5: Interpret Results

### code.json field reference

Each row in `code["code"]` is:
```
[asm, _, pc_index, source_loc, _, pc_addr, exec_count, total_cycles, stall_cycles, issue_cycles]
  0   1     2          3       4     5          6            7              8             9
```

- **col[8] `stall_cycles`**: cycles the instruction was blocked from issuing — **primary hotspot metric**
- **col[7] `total_cycles`**: total cycles charged to this instruction across all waves
- **col[3] `source_loc`**: `"/path/to/file.py:LINE"` — virtual path resolved via `snapshots.json`
- **col[6] `exec_count`**: number of wave-threads that executed this instruction

### snapshots.json: resolving source paths

`snapshots.json` encodes a nested dict tree mapping virtual paths to local filenames:
```json
{"/": {"FlyDSL": {"kernels": {"pa_decode_sw_fp8_ps.py": "source_0_pa_decode_sw_fp8_ps.py"}}}}
```
Flatten recursively: `/FlyDSL/kernels/pa_decode_sw_fp8_ps.py` → `source_0_pa_decode_sw_fp8_ps.py`

### Stall type classification

| Type | Instructions | Root Cause |
|------|-------------|------------|
| `VMEM-load` | `buffer_load_*`, `global_load_*` | Load itself stalled (VMEM queue full or back-pressure from no compute to hide behind) |
| `VMEM-wait` | `s_waitcnt vmcnt(N)` | Waiting for outstanding VMEM loads to complete |
| `LDS/SMEM-wait` | `s_waitcnt lgkmcnt(N)` | Waiting for LDS or SMEM ops |
| `barrier` | `s_barrier` | Cross-wave sync — slowest wave dominates |
| `MFMA/FMA` | `v_mfma_*` | MFMA dependency chain (RAW hazard) |
| `LDS` | `ds_read_*`, `ds_write_*` | LDS access latency |

### Common hotspot patterns

The snippets below are schematic — they illustrate *instruction scheduling*, so
loads are written in the abbreviated raw-intrinsic form. On the current surface
these are `fx.copy` from a `make_buffer_tensor` view; the scheduling advice is
unchanged either way.

#### Pattern 1: V/K loads inside MFMA loop → very high stall rate (80–95%)

```python
# BAD: load and MFMA alternate — only 1 MFMA of hiding time
for k_step in range_constexpr(QKHELOOP * 2):
    if k_step % 2 == 0:
        v_data = buffer_ops.buffer_load(...)   # stall_rate ~92%
    acc = rocdl.mfma_f32_16x16x32_fp8_fp8(...)

# GOOD: batch all loads before the MFMA loop
for td in range_constexpr(TLOOP):
    v_prefetch[td] = [buffer_ops.buffer_load(...) for _ in range_constexpr(QKHELOOP)]

for td in range_constexpr(TLOOP):
    for k_step in range_constexpr(QKHELOOP * 2):
        acc = rocdl.mfma_f32_16x16x32_fp8_fp8(...)   # entire QK MFMA hides VMEM latency
    v_results[td] = v_prefetch[td]   # already in registers
```

#### Pattern 2: Sequential loads with no compute → VMEM queue saturation

```python
# BAD: all loads back-to-back, no compute interleaved
for td in range_constexpr(TLOOP):
    for qkhe in range_constexpr(QKHELOOP):
        k4 = buffer_ops.buffer_load(k_rsrc, ka_dw, ...)   # queue fills up

# GOOD: prefetch next tile's K loads during current tile's MFMA computation
```

#### Pattern 3: LDS prob reads immediately before PV MFMA → lgkmcnt stall

```python
# BAD: LDS reads and MFMA in same loop
for vhe in ...:
    for vt in ...:
        p_i64 = lds_read(...)    # issued here
        tmp = mfma(v_i64, p_i64, ...)   # immediately consumed → lgkmcnt stall

# GOOD: batch all LDS reads first, then all MFMAs
for vhe in ...:
    for vt in ...:
        p_i64s.append(lds_read(...))    # all LDS reads issued first

for vhe in ...:
    for vt in ...:
        tmp = mfma(v_i64s[...], p_i64s[...], ...)   # LDS data already ready
```

#### Pattern 4: Scale loads too close to usage

```python
# BAD: scale load and usage separated by only TLOOP MFMAs
for td in range_constexpr(TLOOP):
    k_scale = buffer_ops.buffer_load(ks_rsrc, ...)   # issued here
# ... small compute gap ...
    result = acc * k_scale   # used too soon → stall

# GOOD: issue scale loads at the very beginning of the block,
# before K loads, to maximise latency hiding distance
```

#### Pattern 5: Hotspot attributed to kernel entry line

When `@flyc.kernel` / kernel decorator line appears as the top hotspot with a mix of
VMEM-wait + barrier stall types — this is a **debug info aggregation artifact**.
MLIR/compiler-generated instructions (address arithmetic, cndmask, prologue setup) map
to the outermost scope line. Ignore this line; focus on lines with explicit user ops.

### Register pressure check (architecture-aware)

`hotspot_analyzer.py` auto-detects the GPU architecture from ISA instruction patterns
and computes occupancy (waves/SIMD) as the **minimum across every resource limiter**:

```
occupancy = min(vgpr_limit, lds_limit, sgpr_limit, hw_max=8)
  vgpr_limit = 512 // (arch_vgpr_alloc + accum_vgpr_alloc)              # per SIMD
  lds_limit  = (LDS_total // lds_per_wg) * waves_per_wg // 4_SIMDs      # per SIMD
  sgpr_limit = 800 // sgpr_alloc                                        # per SIMD
```

**VGPR is a combined 512-entry pool on BOTH gfx942 and gfx950.** CDNA2 (gfx90a)
unified the arch (256) and accum (256) VGPR files into one 512 budget per SIMD,
and gfx942/gfx950 inherit that. Occupancy from VGPR is `512 / (arch + accum)` on
both — NOT `256 / max(arch, accum)`. (The separate-pool `256/max` model only
applied to gfx908 / CDNA1, where accum VGPRs were a distinct file accessible
only by MFMA.)

| Property | CDNA3 (gfx942) | CDNA4 (gfx950) |
|---|---|---|
| VGPR pool | 512 combined (256 arch + 256 accum, unified budget) | 512 combined (same) |
| Occupancy formula (VGPR) | `512 / (arch_alloc + accum_alloc)` | `512 / (arch_alloc + accum_alloc)` |
| Alloc granularity | 8 VGPRs | 8 VGPRs |
| LDS size | 64 KB | 160 KB |
| LDS alloc block | 256 bytes | 1280 bytes |
| VMCNT width | 6 bits (max 63 in-flight) | 6 bits (max 63 in-flight) |
| LGKMCNT width | 4 bits (max 15 in-flight) | 4 bits (max 15 in-flight) |

What actually changed in CDNA4 vs CDNA3 is the LDS size (64KB→160KB) and the LDS
alloc granularity — not the VGPR pooling model.

**Reading the real counts.** `code.json` only holds the (often single-CU,
often vgpr-form) disassembly, so it cannot reveal accum_vgpr / LDS / SGPR /
workgroup size — an AGPR-form-blind ISA scan reports `accum=0` and gets
occupancy badly wrong. The analyzer reads `out_kernel_trace.csv` (staged next to
the dispatch dir) for the authoritative `Accum_VGPR_Count` / `LDS_Block_Size` /
`SGPR_Count` / `Workgroup_Size_*`. arch_vgpr is taken as `max(ISA_scan, CSV)` so
a bogus-low CSV `VGPR_Count` field can't under-report. If no CSV is found it
falls back to ISA-only and prints a warning.

**Auto-detection**: gfx950-specific instructions (`v_mfma_scale_f32_*`, `v_mfma_f32_16x16x128_*`,
`v_mfma_f32_32x32x64_*`) indicate CDNA4. Absence indicates CDNA3.

```bash
sqlite3 results.db "
SELECT ks.KernelName, ki.arch_vgpr_count, ki.accum_vgpr_count, ki.lds_size
FROM rocpd_kernel_dispatch kd
JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id=ks.id
JOIN rocpd_info_kernel ki ON kd.kernel_id=ki.id LIMIT 5;"
```

Worked example (PA decode, gfx942): arch 144 + accum 136 = 280 combined → `512//280 = 1`
wave/SIMD, VGPR-bound (LDS allows 5, SGPR allows 7). Reaching 2 waves needs
combined ≤ 256, e.g. freeing ~24 VGPRs.

**Note**: `maxnreg` has been removed (it never reached LLVM). Forcing
`accum_vgpr=0` with it doubled occupancy but caused MFMA spills through
arch_vgpr — measured 4.5x GPU slowdown. Use `waves_per_eu` instead.

### L2 / HBM efficiency analysis (PMC, not ATT)

When the ATT hotspots are dominated by `VMEM-load` at high stall rate (e.g.
40-50% of stall, ~94% per-load), the kernel is memory-bound and the next
question is **why** — and ATT cannot answer it (it has no cache counters).
Capture PMC counters (see capture-kernel-trace "PMC Mode") and analyze with
`${CLAUDE_SKILL_DIR}/scripts/pmc_l2_analyzer.py`:

```bash
python ${CLAUDE_SKILL_DIR}/scripts/pmc_l2_analyzer.py \
    /tmp/pmc_out/pass_1/pmc_l2_counter_collection.csv \
    /tmp/pmc_ea_out/pass_1/pmc_ea_counter_collection.csv \
    --kernel <kernel> --ideal-gb <bytes_per_dispatch_GB> --ea-channels 2
```

Three metrics, three decisions:

| Metric | Formula | What it tells you |
|---|---|---|
| **L2 hit rate** | `TCC_HIT/(TCC_HIT+TCC_MISS)` | Is there temporal reuse to exploit? |
| **32B fraction** | `TCC_EA0_RDREQ_32B/TCC_EA0_RDREQ` | Spatial locality / cache-line waste |
| **over-fetch** | `est_HBM_bytes / (ideal_GB × dispatches)` | Redundant fetching |

**Decision tree** for a memory-bound decode kernel:

1. **L2 hit rate < 5%** → pure streaming, no reuse. This is **expected and
   correct** for decode with independent per-sequence paged KV — each KV byte
   is read once; the GQA (×heads) and MTP (×seq) reuse is captured in
   registers/LDS, never re-reads L2. *"Improving L2 hit rate" is a non-goal.*
   The only thing that raises it is real KV reuse = **shared-prefix serving**
   (a workload/scheduling property, not a kernel change).
2. **32B fraction ≈ 0%** → full 64B cache lines, no spatial-locality waste.
   Nothing to fix at the line level. (High 32B% would point to scattered/
   misaligned access worth restructuring.)
3. **over-fetch ≈ 1.0x** → the kernel reads exactly the data it needs. The
   achieved bandwidth (compute as `ideal_bytes / kernel_time`) is then the
   real ceiling for this access pattern. **50-60% of theoretical HBM peak is
   normal** even for clean streaming; paged-gather decode living at ~54% with
   0% partial + ~1.0x over-fetch is healthy, not a defect.

**Worked example (PA decode, gfx942, bs=16, ctx=131072, batch=256):**
L2 hit 1.7%, 32B 0%, over-fetch 1.04x, 2.85 TB/s = 54% peak. Conclusion: the
memory subsystem is clean; there is **no KV-load optimization left** — verified
by also testing block_size 16→64 (regressed +7.8%) and confirming dwordx8
doesn't exist on CDNA3 (dwordx4 / 16B is the max single vector load).

**Counter-capture caveat**: keep each PMC job to ≤ ~4 TCC counters (single
hardware pass). Multi-pass collection has triggered a GPU Hang on gfx942 — see
capture-kernel-trace.

### MFMA latency reference (cycles = pipeline depth)

| Instruction | Variant | Cycles | Notes |
|---|---|---|---|
| `v_mfma_f32_*_f16` / `_bf16` | 16x16x16 | 16 | |
| `v_mfma_f32_*_f16` / `_bf16` | 32x32x8 | 32 | |
| `v_mfma_f32_*_fp8_fp8` | 16x16x32 | 16 | CDNA3+CDNA4 |
| `v_mfma_f32_*_fp8_fp8` | 32x32x16 | 32 | CDNA3+CDNA4 |
| `v_mfma_f32_16x16x128_f8f6f4` | 16x16x128 | 16 or 32 | CDNA4 only; 32 if either A or B is FP8 |
| `v_mfma_f32_32x32x64_f8f6f4` | 32x32x64 | 32 or 64 | CDNA4 only; 64 if either A or B is FP8 |
| `v_mfma_scale_f32_16x16x128_f8f6f4` | 16x16x128 | 16 or 32 | CDNA4 only; with block exponent scaling |
| `v_mfma_scale_f32_32x32x64_f8f6f4` | 32x32x64 | 32 or 64 | CDNA4 only; with block exponent scaling |
| `v_mfma_f32_*_f32` | 16x16x4 | 32 | |
| `v_mfma_f32_*_f32` | 32x32x2 | 64 | |
| `v_mfma_f64_16x16x4_f64` | 16x16x4 | 64 | |

### MFMA dependency NOPs (CDNA4, from ISA reference Table 38)

These are the minimum independent instructions (or s_nop counts) required between
MFMA result production and consumption. The values vary by MFMA variant:

| Dependency pattern | Required waits | Comment |
|---|---|---|
| XDL write -> same XDL read SrcC (accumulate, exact same vDst) | 0-2 | Forwarding path; back-to-back accumulation OK |
| XDL write -> VALU/VM/LDS/FLAT read result (RAW) | 5, 8, 12, or 20 | No forwarding; must wait for MFMA commit to VGPR |
| XDL write -> MFMA read as SrcA or SrcB | 5, 8, 12, or 20 | No forwarding path |
| Non-DLops VALU write -> MFMA read | 2 | No 4/8 cycle forwarding path |
| VALU writes SGPR -> VMEM reads that SGPR | 5 | **HW does NOT check this** — user must add waits |
| V_CMPX* writes EXEC -> V_MFMA* | 4 | No EXEC forwarding with MFMA |

Wait counts for "5, 8, 12, or 20" depend on MFMA variant:
- 5 waits: 16x16 4-block variants (8 cycle MFMAs)
- 8 waits: 16x16x16 F16/BF16 etc. (16 cycle MFMAs)
- 12 waits: 32x32x8, 16x16x4 F32, etc. (32 cycle MFMAs)
- 20 waits: 32x32x4 F32, 32x32x2 F32 (64 cycle MFMAs)

---

## Step 6: Optimization Plan

After running `hotspot_analyzer.py --detail`, produce a prioritized plan:

```
## Stall Summary
- Total stalls: X cycles (Y% of kernel)
- Top stall type: VMEM-load (Z%)

## Hotspot Analysis

### #1 :LINE  stall=XK (N%)  VMEM-load  stall_rate=92%
Root cause: buffer_load inside QK MFMA loop — only 1 MFMA of hiding time.
Fix: Move all V loads before the QK MFMA loop.
Estimated gain: ~20% kernel cycle reduction (hypothesis; verify correctness and re-time).

### #2 :LINE  stall=XK (N%)  VMEM-load  stall_rate=80%
Root cause: K loads sequential with no compute interleaved.
Fix: Prefetch next tile's K during current tile's MFMA (double-buffer pattern).
See /prefetch-data-load skill.

### #3 ...

## Priority Order
1. [HIGH]  Fix V-load position (24% of all stalls, easy refactor)
2. [HIGH]  K-load cross-tile prefetch (8% of stalls, needs _process_block restructure)
3. [MED]   Move scale loads earlier (8% of stalls, trivial move)
4. [LOW]   Batch LDS reads before PV MFMA (4% of stalls, loop split)
```

---

## Error Handling

| Error | Fix |
|-------|-----|
| `rocprof-trace-decoder library path not found` | Install decoder .so (see Step 3) |
| Trace output empty | Check `kernel_include_regex` matches exactly |
| Trace truncated | Increase `att_buffer_size` to `"0xC000000"` |
| `kernel_iteration_range` mismatch | Adjust range; try `"[0, [1-2]]"` |
| `INVALID_SHADER_DATA` | aqlprofile/decoder version mismatch — update both |
| Source loc all `""` | Set `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`; check `-g` flag in compile pipeline |
| Top hotspot is kernel decorator line | Debug info artifact — skip it, focus on op lines |
| `--att` flag error | `--att` is boolean, no value; use `-i input.yaml` for full config |
| GPU Hang / HW Exception during PMC | Too many counters → multi-pass. Split into single-pass jobs of ≤ ~4 TCC counters |
| PMC `accum_vgpr=0` but kernel uses MFMA | vgpr-form MFMA: accumulators are in the arch VGPR file; read total from `VGPR_Count + Accum_VGPR_Count` |
