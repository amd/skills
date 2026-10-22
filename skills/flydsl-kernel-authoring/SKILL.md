---
name: flydsl-kernel-authoring
description: >
  Comprehensive reference for authoring FlyDSL GPU kernels on AMD GPUs.
  Covers the layout algebra, tiled copy/MMA, buffer ops, loop-carried range loops,
  SharedAllocator (LDS), autotuning, and common patterns. Use when writing,
  reviewing, or understanding FlyDSL kernel code.
allowed-tools: Read Edit Bash Grep Glob Agent
---

# FlyDSL Kernel Authoring Skill

## Overview

FlyDSL is a Python DSL and MLIR-based compiler for writing high-performance GPU kernels on AMD GPUs (MI300X/MI350). It provides explicit layout algebra for controlling data movement, tiling, and memory access patterns. The layout system is the core abstraction that distinguishes FlyDSL from Triton/Gluon.

**Repository**: this checkout (examples below assume it is importable — see 12)
**Target GPU**: gfx942 (MI300X, CDNA3), gfx950 (MI350, CDNA4)
**Python**: 3.12, ROCm 7.2

**Scope (read this first)**: This skill is the **reference** — the full layout-algebra API
surface, per-op tables, MFMA/copy-atom catalogs, environment variables, and an exhaustive
troubleshooting list. Reach for it to *look something up* while writing or reviewing kernel
code. If instead you want a *guided, step-by-step procedure* that turns a kernel requirement
into a finished, tested kernel (classify -> skeleton -> compute -> control flow -> test), use
the **flydsl-tile-programming** skill, which is the wizard companion to this reference. For
diagnosing a kernel that already compiles but produces NaN/inf/wrong results, use the
**debug-flydsl-kernel** skill.

---

## 1. Architecture and Compilation

### Pipeline
```
Python (@flyc.kernel/@flyc.jit)
  -> AST Rewriting (for/if -> scf.for/scf.if)
  -> MLIR Tracing (generates Fly dialect + gpu/arith/scf/memref/vector ops)
  -> MlirCompiler.compile() (Fly -> ROCDL -> LLVM -> HSACO binary)
  -> JITCFunction (ExecutionEngine wrapper)
```

### Key Passes
Pipeline is built by `RocmBackend._pipeline_parts()` and split into three stages — see `docs/architecture_guide.md` §3 for the per-pass table. Highlights:
1. `fly-rewrite-func-signature` - Rewrite DSL types at function / SCF boundaries to packed LLVM structs
2. `fly-layout-lowering` - Lower layout algebra (`fly.crd2idx`, partitions, divides) to arithmetic
3. `fly-convert-atom-call-to-ssa-form` + `fly-promote-regmem-to-vectorssa` - Lift copy/MMA atom calls and register memory to vector SSA
4. `convert-fly-to-rocdl` - Fly ops -> ROCDL intrinsics
5. `gpu-module-to-binary{format=fatbin}` - Emit HSACO binary via LLVM AMDGPU backend

### Key Source Paths
- `python/flydsl/compiler/` - JIT compilation (jit_function.py, kernel_function.py)
- `python/flydsl/expr/` - DSL expression API (primitive.py, derived.py, typing.py)
- `python/flydsl/expr/primitive.py` - All layout algebra functions
- `python/flydsl/expr/derived.py` - CopyAtom, MmaAtom, TiledCopy, TiledMma wrappers
- `python/flydsl/expr/gpu.py` - GPU operations (thread_idx, block_idx, barrier)
- `python/flydsl/expr/rocdl/` - MFMA/WMMA and other ROCm intrinsics
  (package: cdna3, cdna4, cdna5, rdna3, rdna4, cluster, inline_asm, tdm_ops, universal;
  plus `utils.py` / `enum.py` helpers)
- `python/flydsl/expr/gpu.py` - `SharedAllocator` for LDS (shared memory), `thread_id`/`block_id`, `barrier`
- `kernels/common/buffer_ops.py` - legacy raw AMD buffer load/store intrinsics
  (moved out of `flydsl.expr` in #880; prefer `fx.rocdl.make_buffer_tensor`)
- `kernels/` - Pre-built kernels, organized into subpackages: `gemm/` (preshuffle_gemm.py, mxfp4_preshuffle.py, ...), `norm/` (layernorm/softmax/rmsnorm), `attention/`, `moe/`, `mega_moe/`, `common/` (incl. `common/mma/`), `comm/`, `conv/`

---

## 2. Layout System (Core Abstraction)

See [references/layout-algebra.md](references/layout-algebra.md): core types, construction, coordinate mapping, layout algebra, products, and divides.

---

## 3. Writing Kernels

### Basic Pattern
```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl

@flyc.kernel
def my_kernel(
    A: fx.Tensor,         # GPU tensor (memref via DLPack)
    B: fx.Tensor,
    N: fx.Constexpr[int], # Compile-time constant
):
    tid = gpu.thread_id("x")
    bid = gpu.block_id("x")
    # ... kernel body ...

@flyc.jit
def launch(
    A: fx.Tensor,
    B: fx.Tensor,
    N: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    my_kernel(A, B, N).launch(
        grid=(N // 256,), block=(256,), stream=stream
    )

# Usage:
import torch
A = torch.randn(1024, device="cuda", dtype=torch.float32)
B = torch.empty(1024, device="cuda", dtype=torch.float32)
launch(A, B, 1024)
```

### Current Syntax Quick Reference

Use the current public FlyDSL surface from `kernels/gemm/preshuffle_gemm.py` when writing new kernels:

```python
Vec = fx.Vector

tx = gpu.thread_id("x")
bx = gpu.block_id("x")
by = gpu.block_id("y")

i32_m: fx.Int32
c_m = fx.Int64(i32_m)
c4 = fx.Int64(4)
zero_f = fx.Float32(0.0)

layout = fx.make_layout((4, 64), (64, 1))
coord = fx.idx2crd(tx, layout)
wave_id = fx.get(coord, 0)
lane_id = fx.get(coord, 1)

acc = Vec.filled(4, 0.0, fx.Float32)
v_i64 = Vec(raw_vec).bitcast(fx.Int64)
elem0 = v_i64[0]

buf = fx.rocdl.make_buffer_tensor(tensor)          # preferred buffer-resource view
tA  = fx.make_view(fx.get_iter(buf), fx.make_layout((M, N), (N, 1)))
# load/store tA via copy atoms (fx.copy); the raw buffer intrinsics are legacy
```

Older code may use `gpu.thread_idx.x`, `gpu.block_idx.x`, `arith.constant(...)`, `T.i32`, raw `vector.*` helpers, the `ArithValue` wrapper, and `buffer_ops`. Keep those when editing existing code that already uses them heavily, but prefer `gpu.thread_id/block_id`, `fx.Int64`/`fx.Int32`/`fx.Float32`, `fx.Vector`, and `fx.rocdl.make_buffer_tensor` for new code. `ArithValue`, `fx.Index`, and `buffer_ops` are deprecated/legacy — for migrating an existing kernel see the **kernel-code-cleanup** skill.

### Parameter Types
| Type | Description | At host boundary |
|------|-------------|-----------------|
| `fx.Tensor` | GPU tensor (memref) | Auto-converted from torch.Tensor via DLPack |
| `fx.Constexpr[int]` | Compile-time constant | Different values -> different compiled kernels |
| `fx.Int32` | Runtime i32 | Auto-converted from Python int |
| `fx.Stream` | CUDA/HIP stream | `fx.Stream(None)` for default stream |

### Thread/Block Hierarchy
```python
from flydsl.expr import gpu

tid_x = gpu.thread_id("x")   # Preferred current spelling
bid_x = gpu.block_id("x")
bid_y = gpu.block_id("y")

# Legacy spelling still appears in older kernels:
tid_x = gpu.thread_idx.x
bid_x = gpu.block_idx.x

gpu.barrier()                # Workgroup synchronization
```

### Control Flow
```python
from flydsl.expr import range_constexpr

# Compile-time unrolled loop (emitted inline in IR)
for i in range_constexpr(N):
    ...

# Runtime loop (lowered by AST rewriting)
for i in range(runtime_value):
    ...
```

### Runtime vs Compile-Time Conditions (Current Style)

Use Python/DSL operators for runtime SSA comparisons. The AST rewriter lowers dynamic `if` conditions to `scf.IfOp`, and comparison operators like `==`, `<`, `>=` generate the needed MLIR predicates.

```python
tid = gpu.thread_id("x")
lane = tid % fx.Int64(64)
c_zero = fx.Int64(0)
c_limit = fx.Int64(8)

# Preferred: readable DSL comparisons
if lane == c_zero:
    ...

in_range = lane < c_limit
val = fx.arith.select(in_range, good_val, zero_val)

# Avoid for simple integer comparisons
in_range = arith.cmpi(arith.CmpIPredicate.slt, lane, c_limit)
```

Use `const_expr(...)` only for values known at trace/compile time, such as Python booleans, constexpr arguments, loop-unroll choices, or type/layout branches:

```python
if const_expr(trans_v):
    ...

if const_expr(max_context_partition_num <= WARP_SIZE):
    ...
```

Do **not** wrap GPU runtime values in `const_expr`. Even with `@flyc.kernel(known_block_size=(256, 1, 1))`, `gpu.thread_id("x")`, `lane`, and `warp_id` are runtime SSA values; the compiler knows their range, not the current lane instance.

```python
# Wrong: lane depends on gpu.thread_id("x")
if const_expr(lane == c_zero):
    ...

# Correct
if lane == c_zero:
    ...
```

Keep explicit `arith.cmpi(...)` / `arith.unwrap(...)` for low-level manual MLIR construction, such as passing a raw condition to `scf.IfOp` directly:

```python
cond = arith.unwrap(partition_idx >= visible_tile_count)
if_op = scf.IfOp(cond, has_else=False)
```

### Frontend Semantic Restrictions
When writing or reviewing `@flyc.kernel` / `@flyc.jit` code, proactively avoid these patterns because they can conflict with MLIR construction even if they look valid in plain Python.

1. **Do not define values inside `if/else` and use them later outside the branch.** Keep a single explicit definition path.
   ```python
   if cond:
       dst = a
   else:
       dst = b
   use(dst)  # avoid this pattern
   ```

2. **Do not mutate captured outer variables inside nested helper functions.** Read-only closure capture is acceptable, but writes should go through explicit parameters and return values.
   ```python
   def kernel():
       acc = fx.Float32(0.0)

       def helper(acc):
           acc = acc + fx.Float32(1.0)
           return acc

       acc = helper(acc)
   ```

3. **Avoid early `return`, and do not place `return` / `yield` inside `if/else` branches.** Prefer a single explicit exit so the frontend can determine result types.
   ```python
   if cond:
       out = v0
   else:
       out = v1
   return out
   ```

4. **Compile-time conditions must use `const_expr(...)`.** Use `if const_expr(flag): ...` for constexpr flags and other static decisions. A plain Python `if` is only safe when the condition is already a Python `bool`.

5. **Runtime branches inside helper functions should be dispatched via local `@flyc.jit`.** When a branch body has side effects, loop-carried values, or branch-local definitions, split the branch bodies into local helpers and wrap the `if` in a local JIT helper:
   ```python
   def then_path():
       ...

   def else_path():
       ...

   @flyc.jit
   def dispatch():
       if runtime_cond:
           then_path()
       else:
           else_path()

   dispatch()
   ```

### Runtime Loops with Loop-Carried Values (Software Pipelining)

Use `init=` on `range()` to create a runtime loop with explicit SSA phi nodes for loop-carried state. This is required for software pipelining (prefetch patterns) where data must flow across iterations.

**Pattern** (from `preshuffle_gemm.py`):
```python
# Prologue: load first tile
tile_0 = prefetch(0)
init_state = [acc_init, tile_0_flat_val1, tile_0_flat_val2, ...]

# Runtime loop with loop-carried state
# Use typed DSL integer bounds (fx.Int64) so the AST rewriter does not treat this as a Python unrolled range.
_start = fx.Int64(0)
_stop = fx.Int64(N - 1)
_step = fx.Int64(1)
for iv, state in range(_start, _stop, _step, init=init_state):
    acc_in = state[0]
    tile_in = state[1:]

    next_tile = prefetch(iv + 1)      # load NEXT data
    acc_in = compute(acc_in, tile_in)  # compute CURRENT

    results = yield [acc_in] + next_tile  # carry to next iter

# Epilogue: process last tile from results
acc_final = results[0]
tile_final = results[1:]
compute(acc_final, tile_final)
```

**How it works in MLIR:**
| Element | Meaning |
|---|---|
| `init=init_state` | List of SSA values that seed the runtime loop block arguments for iteration 0 |
| `state` | The loop-carried block arguments (phi nodes) for THIS iteration |
| `yield [...]` | Feeds values back as next iteration's `state` |
| `results` | After loop exits, holds the last yielded values |

**Three critical pitfalls (all verified by debugging):**

1. **Loop bounds must be a typed DSL integer such as `fx.Int64(...)`, NOT a plain Python int.** A plain int makes the AST rewriter unroll the loop and silently ignore `init=`. If you write `range(0, 15, 1, init=...)`, the AST rewriter treats the constant bounds as a Python `range` and unrolls; only plain Python-int bounds are unrolled, so a typed bound still produces a runtime `scf.for`. Use `fx.Int64(0)`, `fx.Int64(15)`, `fx.Int64(1)` instead.

2. **Prefer internal types, but unwrap at hard boundaries.** Most `range(..., init=...)` uses accept DSL numeric/vector values. If a lower-level helper explicitly expects raw `ir.Value`, unwrap with `v.ir_value()` / `_raw(v)` at that boundary only.

3. **Build LDS views at the top of the kernel, not inside the runtime loop.** Allocate shared memory with `fx.SharedAllocator().allocate(...).peek()` and build each `.view()` once up front. A view created inside the `scf.for` body is defined in the loop scope; using it in the epilogue (outside the loop) causes an SSA dominance error. Building it once at the top makes it dominate both the loop and the epilogue.

### Arithmetic Operations
```python
c42 = fx.Int64(42)                              # typed integer constant (preferred)
c3_14 = fx.Float32(3.14)                        # f32 constant (preferred)
mask = fx.Int32(0xFF)                            # i32 constant (preferred)

# Prefer operators / Numeric methods
result = a + b
result = a * scale
result = cond.select(true_val, false_val)
largest = fx.max(a, b)
smallest = fx.min(a, b)
tiles = fx.ceildiv(count, tile_size)  # signed/unsigned dispatch from typed operands

# Keep direct arith.*FOp only when explicit fastmath flags are required.
```

### Internal Types: Vector and Numeric (PREFERRED)

Use FlyDSL's internal typed system instead of raw MLIR ops. The `Vector` class wraps `vector<NxTy>` with operator overloading and type-safe methods.

```python
Vec = fx.Vector

# Wrap raw vector values
acc = Vec(frag_C.load())      # vector<Nxf32> → Vector with * / + operators

# Indexing (replaces vector.extract)
val = acc[idx]                 # returns Float32 scalar

# Bitcast (replaces vector.bitcast)
v_f32 = Vec(raw_vec).bitcast(fx.Float32)  # vector<Nxi32> → vector<Nxf32>

# Type conversion (replaces arith.trunc_f / arith.ext_f)
bf16_val = f32_val.to(fx.BFloat16)     # f32 → bf16

# Arithmetic — use Python operators, not arith.mulf/addf
result = (val * scale_a) * scale_b

# Splat constant vector
zeros = Vec.filled(N, 0.0, fx.Float32)

# Index cast — use fx.Int32 instead of arith.index_cast
idx = fx.Int32(gpu.block_id("x") * tile_m)
```

**Prefer internal types over raw ops:**
| Raw MLIR op | Internal type equivalent |
|-------------|------------------------|
| `vector.extract(v, static_position=[i], ...)` | `Vec(v)[i]` |
| `vector.bitcast(target_ty, v)` | `Vec(v).bitcast(Float32)` |
| `arith.trunc_f(ty, v)` | `v.to(BFloat16)` |
| `arith.mulf(a, b)` | `a * b` |
| `arith.addf(a, b)` | `a + b` |
| `arith.index_cast(T.i32, v)` | `fx.Int32(v)` |

Use `Vec.filled(...)` for splats and `Vec.from_elements(...)` for vectors from scalars.

### Arith Ops Availability Table
| Operation | Function | Works on Vectors | Notes |
|-----------|----------|-----------------|-------|
| Add | `a + b` | Yes | Use direct FOp only for explicit fastmath |
| Multiply | `a * b` | Yes | Use direct FOp only for explicit fastmath |
| Negate | `-a` | Yes | |
| Max / Min | `fx.max(a, b)` / `fx.min(a, b)` | Yes | Float forms propagate NaN; `fx.maxnumf` / `fx.minnumf` do not |
| Integer ceil-div | `fx.ceildiv(a, b)` | Yes | Direct signed/unsigned op; distinct from layout `fx.ceil_div` |
| Compare | `arith.cmpf(pred, a, b)` | Yes | predicate FIRST; returns i1/vec<i1> |
| Select | `cond.select(t, f)` | Yes | |
| Abs | no direct helper | Use `-v`, comparison, and `cond.select(...)` |
| FMA | `a * b + c` | Yes | Use direct FOp only when explicit fastmath is needed |
| Splat const | `Vec.filled(width, val, dtype)` | Creates vector | For scalar broadcast |

### Printf Debugging
```python
fx.printf("tid={} bid={} val={}", tid, bid, value)
```

---

## 4. Data Movement Patterns

See [references/data-movement.md](references/data-movement.md): layout-based copy, vectorized loads, TiledCopy, buffer load/store, copy atom types, and gfx1250 TDM.

## 5. Shared Memory (LDS)

See [references/data-movement.md](references/data-movement.md): `SharedAllocator` pattern and LDS capacity per architecture.

## 6. MFMA Integration (Matrix Math)

See [references/mfma-and-reductions.md](references/mfma-and-reductions.md): MFMA and gfx1250 WMMA atoms, raw intrinsics, and the preshuffle GEMM pattern.

## 7. Reduction Patterns

See [references/mfma-and-reductions.md](references/mfma-and-reductions.md): warp and block reductions.

## 8. Common Patterns and Recipes

See [references/recipes.md](references/recipes.md): element-wise template and cookbook, and a naive GEMM template.

## 9. Environment and Debugging

See [references/environment-and-debugging.md](references/environment-and-debugging.md): IR dumps, environment variables, disk cache, source-to-assembly debug info, and autotune.

---

## 10. Troubleshooting

### Common Issues

1. **Constants/casts**: Prefer `fx.Int32(...)`, `fx.Int64(...)`, and `fx.Float32(...)` (`fx.Index(...)` is being deprecated in favor of `fx.Int64(...)`). Use `arith.constant(...)` only at low-level boundaries.

2. **Raw `buffer_load` offset**: the `offset` parameter is in ELEMENTS, not bytes.
   The raw intrinsics live in `kernels/common/buffer_ops.py`, not `flydsl.expr`.

3. **Cache stale after code changes**: The disk cache auto-invalidates on source/closure changes. Only set `FLYDSL_RUNTIME_ENABLE_CACHE=0` or clear `~/.flydsl/cache/` if you changed C++ passes or non-closure helpers.

4. **LDS overflow**: Check capacity (64KB on gfx942, 160KB on gfx950). `SharedAllocator` sizes the LDS global for you; the compiler errors if a static allocation exceeds the arch limit.

5. **Dynamic vs Constexpr**: `Constexpr[int]` values are baked into IR -- different values produce different compiled kernels. Use `Int32` for truly dynamic values.

6. **Tensor layout marking**: For dynamic shapes or alignment, use `flyc.from_dlpack(tensor).mark_layout_dynamic(leading_dim=0, divisibility=4)`.

7. **AMD wavefront size**: Always 64 on gfx9xx. Use shifts [32, 16, 8, 4, 2, 1] for full-wave reduction.

8. **tile_k alignment for GEMM**: `tile_k * elem_bytes` must be divisible by 64 (K64-byte micro-step).

9. **INT4 (W4A8)**: A matrix is int8, B matrix is packed int4 (2 values/byte), unpacked to int8 in-kernel.

10. **Absolute value**: the *arith dialect* has no `absf`, but FlyDSL exports one — use `abs(v)` or `fx.absf(v)` rather than a negate/compare/select sequence.

11. **Scalar broadcast to vector**: Use `Vec.filled(width, value, fx.Float32)` to create a splat constant vector. Do NOT use raw vector ops for ordinary arithmetic.

---

## 11. Comparison with Triton/Gluon

| Aspect | FlyDSL | Triton | Gluon |
|--------|--------|--------|-------|
| Layout control | Explicit layout algebra (Shape, Stride, Layout) | Implicit via block pointers | Implicit |
| Tiling | Manual via divide/product operations | Auto-tiling with `tl.program_id` | Auto-tiling |
| Memory access | Copy atoms, buffer load/store, TiledCopy | `tl.load`/`tl.store` | `gluon.load`/`gluon.store` |
| MFMA | Direct `rocdl.mfma_*` intrinsics | `tl.dot` | `gluon.dot` |
| Shared memory | SharedAllocator + `@fx.struct` explicit management | Implicit scratchpad | Implicit |
| Abstraction level | Low (near hardware) | Medium | Medium-High |
| Compilation | MLIR (Fly dialect -> LLVM -> HSACO) | MLIR (Triton dialect -> LLVM) | MLIR |
| Control | Maximum control over data layout and movement | Less control, more automation | Least control |

FlyDSL gives maximum control at the cost of verbosity. The layout algebra is the key differentiator -- it enables precise control over how data is arranged in registers, shared memory, and global memory, and how threads map to data.

---

## 12. Running Kernels

### Locally

With an editable install (`pip install -e .`) the repo root is enough. From a
plain source checkout, put the built bindings on the path as well (CLAUDE.md):

```bash
export PYTHONPATH="${PWD}/build-fly/python_packages:${PWD}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${PWD}/build-fly/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH}"

python3 my_kernel.py
python3 -m pytest tests/kernels/test_vec_add.py -v
bash scripts/run_benchmark.sh
```

### On a remote host or container

Ask the user for the host, container, and checkout path; do not assume one.
With `$HOST`, `$CONTAINER`, and `$FLYDSL_ROOT` set:

```bash
ssh -o LogLevel=ERROR "$HOST" \
  "docker exec $CONTAINER bash -c 'cd $FLYDSL_ROOT && python3 my_kernel.py'"
```

Drop the `docker exec` wrapper when the checkout is on the host itself.
To build FlyDSL on such a host first, use the **build-flydsl** skill.
