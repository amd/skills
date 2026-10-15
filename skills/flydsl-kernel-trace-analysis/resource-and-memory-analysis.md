# Resource and memory analysis



### Register pressure check (architecture-aware)

`hotspot_analyzer.py` auto-detects the GPU architecture from ISA instruction patterns
and estimates occupancy (waves/SIMD) as the minimum of the modeled resource limits:

```
occupancy = min(vgpr_limit, lds_limit, sgpr_limit, hw_max=8)
  vgpr_limit = 512 // round_up(arch_vgpr + accum_vgpr, 8)             # per SIMD
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
| Occupancy formula (VGPR) | `512 // round_up(arch + accum, 8)` | `512 // round_up(arch + accum, 8)` |
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
`SGPR_Count` / `Workgroup_Size_*`. For vgpr-form MFMA, the combined count is `max(ISA_scan, VGPR_Count +
Accum_VGPR_Count)`; do not add accumulator counts a second time. For agpr-form
MFMA, each sub-count is checked against its corresponding ISA scan. The script
rounds the combined count to the allocation granularity. If no CSV is found it
falls back to ISA-only and prints a warning.

**Auto-detection**: gfx950-specific instructions (`v_mfma_scale_f32_*`, `v_mfma_f32_16x16x128_*`,
`v_mfma_f32_32x32x64_*`) indicate CDNA4. Absence is inconclusive: the script defaults to CDNA3, so a gfx950 kernel without
these instructions will have an incorrect LDS limit. Verify the GPU independently
and qualify the occupancy estimate when this heuristic disagrees.

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

**Note**: `maxnreg` has been removed from FlyDSL because it never reached LLVM.
Use `waves_per_eu` for an occupancy hint, verify the emitted attribute, and
measure spills and timing rather than assuming higher occupancy is faster.

### L2 / HBM efficiency analysis (PMC, not ATT)

When the ATT hotspots are dominated by `VMEM-load` at high stall rate (e.g.
40-50% of stall, ~94% per-load), investigate memory latency, queue pressure, and
insufficient overlap. This alone does not prove HBM bandwidth saturation;
ATT has no cache counters.
Capture matching PMC counter groups and analyze with
`${SKILL_DIR}/scripts/pmc_l2_analyzer.py`:

```bash
python "$SKILL_DIR/scripts/pmc_l2_analyzer.py" \
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
   real ceiling for this access pattern. **50-60% of theoretical HBM peak has been observed** even for clean streaming; paged-gather decode living at ~54% with
   0% partial + ~1.0x over-fetch does not by itself establish a memory-access defect or rule out other optimizations.

**Worked example (PA decode, gfx942, bs=16, ctx=131072, batch=256):**
L2 hit 1.7%, 32B 0%, over-fetch 1.04x, 2.85 TB/s = 54% peak. Conclusion: the
memory subsystem is clean; these counters show no obvious redundant KV-load traffic in this measurement — verified
by also testing block_size 16→64 (regressed +7.8%) and confirming dwordx8
doesn't exist on CDNA3 (dwordx4 / 16B is the max single vector load).

**Counter-capture caveat**: keep each PMC job to ≤ ~4 TCC counters (single
hardware pass). Multi-pass collection has triggered a GPU Hang on gfx942 — see
the collection reference linked from SKILL.md.

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
