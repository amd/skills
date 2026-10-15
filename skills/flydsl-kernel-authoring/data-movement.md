# Data movement, LDS, matrix math, and reductions

- [Data Movement Patterns](#data-movement-patterns)
- [Shared Memory (LDS)](#shared-memory-lds)
- [MFMA Integration (Matrix Math)](#mfma-integration-matrix-math)
- [Reduction Patterns](#reduction-patterns)

## Data Movement Patterns

### Layout-Based Copy (Preferred for Element-wise Kernels)

The standard pattern: divide tensor by tile size, slice by block/thread, copy via atoms.

```python
@flyc.kernel
def my_kernel(A: fx.Tensor, B: fx.Tensor, BLOCK_DIM: fx.Constexpr[int]):
    bid = gpu.block_id("x")
    tid = gpu.thread_id("x")

    # 1. Divide tensor into blocks
    tA = fx.logical_divide(A, fx.make_layout(BLOCK_DIM, 1))
    tB = fx.logical_divide(B, fx.make_layout(BLOCK_DIM, 1))

    # 2. Select this block's tile
    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))

    # 3. Further divide for per-thread access
    tA = fx.logical_divide(tA, fx.make_layout(1, 1))  # 1 element per thread
    tB = fx.logical_divide(tB, fx.make_layout(1, 1))

    # 4. Allocate registers
    copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    rA = fx.make_rmem_tensor(1, fx.Float32)

    # 5. Copy: global -> register -> compute -> global
    fx.copy(copyAtom, fx.slice(tA, (None, tid)), rA)
    # ... compute on register values ...
    fx.copy(copyAtom, rA, fx.slice(tB, (None, tid)))
```

### Vectorized Loads (Wide Copies)

```python
VEC_WIDTH = 4
copy_bits = VEC_WIDTH * 32   # 128 bits
copyAtom = fx.make_copy_atom(fx.UniversalCopy(copy_bits), fx.Float32)

rA = fx.make_rmem_tensor(VEC_WIDTH, fx.Float32)

# Divide for VEC_WIDTH elements per thread

tA = fx.logical_divide(tA, fx.make_layout(VEC_WIDTH, 1))
fx.copy(copyAtom, fx.slice(tA, (None, tid)), rA)

# Load/store as vectors

vec = fx.memref_load_vec(rA)     # Load vector from register memref
fx.memref_store_vec(vec, rA)     # Store vector to register memref
```

### TiledCopy Abstraction (for 2D Copies)

```python
# Define thread and value layouts

thr_layout = fx.make_layout((4, 1), (1, 1))    # 4 threads
val_layout = fx.make_layout((1, 8), (1, 1))    # 8 values per thread

# Create copy atom

copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

# Build tiled copy from the TV layout. Use make_layout_tv -- a bare

# raked_product is NOT equivalent: make_layout_tv additionally derives the

# TV mapping via composition(right_inverse(layout_mn), ...).

tile_mn, tv_layout = fx.make_layout_tv(thr_layout, val_layout)
tiled_copy = fx.make_tiled_copy(copy_atom, tv_layout, tile_mn)
# fx.make_tiled_copy_tv(copy_atom, thr_layout, val_layout) is the one-call form.

# Get this thread's slice and partition

thr_copy = tiled_copy.get_slice(tid)
partition_src = thr_copy.partition_S(src_tensor)
partition_dst = thr_copy.partition_D(dst_tensor)
frag = fx.make_fragment_like(partition_src)

# Execute copy: src -> fragment -> dst

fx.copy(copy_atom, partition_src, frag)
fx.copy(copy_atom, frag, partition_dst)
```

### Buffer Load/Store

Preferred: build a buffer-resource view with `make_buffer_tensor` and move data
through copy atoms — the OOB-checked V# descriptor is built for you.
```python
buf = fx.rocdl.make_buffer_tensor(tensor)
tA  = fx.make_view(fx.get_iter(buf), fx.make_layout((M, N), (N, 1)))
copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
fx.copy(copy, fx.slice(tA, (None, tid)), rA)   # after partitioning tA
```

Legacy raw intrinsics (`create_buffer_resource` / `buffer_load` / `buffer_store`,
`offset` in **elements**) remain for un-migrated kernels. They now live in
`kernels/common/buffer_ops.py` (moved out of `flydsl.expr` in #880); use the buffer-resource view above when migrating them.

### Copy Atom Types

| Type | Bits | Usage |
|------|------|-------|
| `fx.UniversalCopy32b()` | 32 | 1x f32 element copy |
| `fx.UniversalCopy(64)` | 64 | 2x f32 elements |
| `fx.UniversalCopy(128)` | 128 | 4x f32 elements |
| `fx.rocdl.BufferCopy128b()` | 128 | AMD buffer load 4xf32 (CDNA) |
| `fx.rocdl.make_tdm_atom(...)` | whole tile | gfx1250 TDM async Global↔LDS DMA (1–5D) — see below |

**gfx1250 TDM async copy** (`fx.rocdl.make_tdm_atom`): a whole-tile DMA whose
descriptor (base pointer, per-dim extent for HW OOB handling, per-dim stride) is
carried as **atom state**. The global operand of the copy is a
shape/direction token only — its layout gives the compile-time N-D tile shape and
its address space picks load vs store; its *pointer is unused* (base comes from
state). Needs a **raw VA** (not `make_buffer_tensor`).

```python
lds = fx.SharedAllocator().allocate(fx.Array[fx.Float16, M * N]).peek()
lds2d = fx.make_view(lds.ptr, fx.make_layout((M, N), (N, 1)))       # note: lds.ptr
g2d = fx.make_view(fx.get_iter(A), fx.make_layout((M, N), (N, 1)))
atom = fx.rocdl.make_tdm_atom(g2d, [M, N], num_warps=4)            # rank = len(extents), 1–5D
fx.copy(atom, g2d, lds2d)                                        # Global → LDS
fx.rocdl.tdm_ops.tensor_wait(0)                                  # await async DMA
fx.copy(atom, g2d, lds2d, imm_offset=k_tile * k_stride_bytes)    # K-loop tile bump
# imm_offset is atom state, so it must go through fx.copy -- copy_atom_call

# takes no **kwargs and cannot carry it.

```

---

## Shared Memory (LDS)

### SharedAllocator Pattern (preferred for new kernels)

Declare the LDS storage as an ``@fx.struct`` of ``fx.Array`` fields, allocate it
inside the kernel with ``fx.SharedAllocator``, then ``.view()`` each field as a
layout. In the default ``static=True`` mode the compiler sizes a per-leaf static
LDS global, so ``launch(smem=...)`` is left unset.

```python
import flydsl.expr as fx

@fx.struct
class SharedStorage:
    a: fx.Array[fx.Float16, 8192]
    b: fx.Array[fx.Float16, 8192]

@flyc.kernel
def my_kernel(A: fx.Tensor, ...):
    lds = fx.SharedAllocator().allocate(SharedStorage).peek()
    lds_a = lds.a.view(fx.make_layout((64, 128), (128, 1)))
    lds_b = lds.b.view(fx.make_layout((64, 128), (128, 1)))
    # ds_write / ds_read happen through copy atoms or view loads/stores
```

For the dynamic mode (``static=False``), the launch wrapper auto-infers
``smem`` from ``SharedAllocator.allocated_bytes`` when ``smem=None``.

`SharedAllocator` is the allocator for new kernels. The legacy
`flydsl.utils.smem_allocator` module (`SmemAllocator` / `SmemPtr`, with its
`finalize()` step) is kept so existing kernels keep working, but it is not
recommended and warns when used — allocate through `fx.SharedAllocator()` over a
`@fx.struct` storage layout and build each `.view(...)` at the top of the kernel.

### LDS Capacity

| Architecture | GPU | LDS per CU |
|---|---|---|
| gfx942 | MI300X | 64 KB |
| gfx950 | MI350 | 160 KB |

---

## MFMA Integration (Matrix Math)

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

---

## Reduction Patterns

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
