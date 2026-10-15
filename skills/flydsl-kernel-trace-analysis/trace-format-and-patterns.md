# Trace format and stall patterns



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
to the outermost scope line. Inspect its instructions before attributing a cause. Report aggregation separately
from hotspots that map to explicit user operations; do not discard its stall cost.
