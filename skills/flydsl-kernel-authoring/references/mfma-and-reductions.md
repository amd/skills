## 6. MFMA Integration (Matrix Math)

### Available MFMA Instructions

Prefer the atom form — it picks the intrinsic from shape + dtype, handles
fragment packing. Pick the atom family by target yourself -- `fx.rocdl.MFMA` for
CDNA3/CDNA4 (it always builds the CDNA3 MFMA type, which covers both), 
`fx.rocdl.cdna4.MFMA_Scale` for CDNA4 scaled, `fx.rocdl.WMMA` for gfx11/gfx1250:

```python
mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.Float16))   # -> f32 acc
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C)                    # d, a, b, c
```

The raw intrinsics take `(result_type, operands)` — note the exact spellings,
which are easy to get wrong:

```python
from flydsl.expr import rocdl

acc = rocdl.mfma_f32_16x16x16f16(T.vec(4, T.f32), [a, b, acc, 0, 0, 0])    # f16
acc = rocdl.mfma_f32_16x16x32_fp8_fp8(T.vec(4, T.f32), [a, b, acc, 0, 0, 0])  # fp8
acc = rocdl.mfma_i32_16x16x32_i8(T.vec(4, T.i32), [a, b, acc, 0, 0, 0])    # i8
```

### gfx1250 WMMA (wave32)

gfx1250 uses **WMMA** (not MFMA), M=N=16. Build the atom with `rocdl.WMMA` (arch
dispatched: gfx11 v16 ABI, gfx12 / gfx1250 v8 ABI) and issue it via
`fx.make_mma_atom` + `fx.gemm` / `fx.mma_atom_call`:

| dtype (A,B → Acc) | K | notes |
|---|---|---|
| f32 → f32 | 4 | |
| f16 / bf16 → f32 or same | 32 | |
| fp8 / bf8 (OCP E4M3FN / E5M2, any mix) → f32 or f16 | 64, 128 | |
| i8 → i32 | 64 | `sign_a` / `sign_b` / `clamp` kwargs |
| i4 → i32 | 32 | `sign_a` / `sign_b` / `clamp` kwargs |

```python
mma = fx.make_mma_atom(rocdl.WMMA(16, 16, 128, fx.Float8E4M3FN))       # fp8 → f32
mma = fx.make_mma_atom(rocdl.WMMA(16, 16, 32, fx.Int4, fx.Int32, sign_a=True, sign_b=True, clamp=True))
```

**MX-scaled WMMA** — `rocdl.WMMAScale(m, n, k, elem_ty_a, ..., block_size=32)` for
the E8M0 block-scaled f8/f6/f4 format (`16x16x128`, or `32x16x128` fp4-only).
Per-operand scales are atom state; `block_size` 32 → i32 scale (i64 for 16):

```python
mma = fx.make_mma_atom(rocdl.WMMAScale(16, 16, 128, fx.Float8E4M3FN))
mma = fx.atom_set_value(mma, "scale_a", fx.Int32(scale_a))
mma = fx.atom_set_value(mma, "scale_b", fx.Int32(scale_b))
fx.gemm(mma, frag_C, frag_A, frag_B, frag_C)
```

### GEMM Pattern (Preshuffle)
The preshuffle GEMM pattern in `kernels/gemm/preshuffle_gemm.py`:
1. B matrix is pre-shuffled to layout: (N/16, K/64, 4, 16, kpack_bytes)
2. A tiles loaded from global to LDS with XOR16 swizzle for bank-conflict avoidance
3. K64-byte micro-steps: each step issues 2x K32 MFMA operations
4. Ping-pong LDS (lds_stage=2) for overlapping loads with compute
5. Epilogue: either direct row-major store or CShuffle via LDS for packing

## 7. Reduction Patterns

### Warp Reduction (AMD wave64)
XOR-shuffle-based intra-wave reduction:
```python
width_i32 = fx.Int32(64)
val = fx.Float32(val)                                        # typed once, not per-iter
for sh in [32, 16, 8, 4, 2, 1]:
    peer = gpu.shuffle_xor(val, sh, width_i32)   # or gpu.shuffle(val, sh, w, mode="xor")
    val = val + fx.Float32(peer)  # typed add; explicit FOp only for fastmath flags
```

### Block Reduction
1. Intra-wave XOR shuffle (shifts: 32, 16, 8, 4, 2, 1)
2. Lane 0 writes per-wave partial to LDS
3. `gpu.barrier()`
4. Wave 0 reads and reduces NUM_WAVES partials from LDS

See the norm kernels in `kernels/norm/` (e.g. `rmsnorm_kernel.py`, which defines
local `wave_reduce_add` / `block_reduce_add` closures) for worked reductions.
