## 4. Data Movement Patterns

### Layout-Based Copy (Preferred for Element-wise Kernels)

The standard pattern: divide tensor by tile size, slice by block/thread, copy via atoms.

```python
@flyc.kernel
def my_kernel(A: fx.Tensor, B: fx.Tensor, BLOCK_DIM: fx.Constexpr[int]):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

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
`kernels/common/buffer_ops.py` (moved out of `flydsl.expr` in #880); see the
**kernel-code-cleanup** skill to migrate them.

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

## 5. Shared Memory (LDS)

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
