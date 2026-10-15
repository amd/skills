# Kernel language and control flow

- [Basic Pattern](#basic-pattern)
- [Current Syntax Quick Reference](#current-syntax-quick-reference)
- [Parameter Types](#parameter-types)
- [Thread/Block Hierarchy](#threadblock-hierarchy)
- [Control Flow](#control-flow)
- [Runtime vs Compile-Time Conditions (Current Style)](#runtime-vs-compile-time-conditions-current-style)
- [Frontend Semantic Restrictions](#frontend-semantic-restrictions)
- [Runtime Loops with Loop-Carried Values (Software Pipelining)](#runtime-loops-with-loop-carried-values-software-pipelining)
- [Arithmetic Operations](#arithmetic-operations)
- [Internal Types: Vector and Numeric (PREFERRED)](#internal-types-vector-and-numeric-preferred)
- [Arith Ops Availability Table](#arith-ops-availability-table)
- [Printf Debugging](#printf-debugging)

## Basic Pattern

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

## Current Syntax Quick Reference

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

Older code may use `gpu.thread_idx.x`, `gpu.block_idx.x`, `arith.constant(...)`, `T.i32`, raw `vector.*` helpers, the `ArithValue` wrapper, and `buffer_ops`. Keep those when editing existing code that already uses them heavily, but prefer `gpu.thread_id/block_id`, `fx.Int64`/`fx.Int32`/`fx.Float32`, `fx.Vector`, and `fx.rocdl.make_buffer_tensor` for new code. `ArithValue`, `fx.Index`, and `buffer_ops` are deprecated/legacy — migrate incrementally and validate numerical results after each change.

## Parameter Types

| Type | Description | At host boundary |
|------|-------------|-----------------|
| `fx.Tensor` | GPU tensor (memref) | Auto-converted from torch.Tensor via DLPack |
| `fx.Constexpr[int]` | Compile-time constant | Different values -> different compiled kernels |
| `fx.Int32` | Runtime i32 | Auto-converted from Python int |
| `fx.Stream` | CUDA/HIP stream | `fx.Stream(None)` for default stream |

## Thread/Block Hierarchy

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

## Control Flow

```python
from flydsl.expr import range_constexpr

# Compile-time unrolled loop (emitted inline in IR)

for i in range_constexpr(N):
    ...

# Runtime loop (lowered by AST rewriting)

for i in range(runtime_value):
    ...
```

## Runtime vs Compile-Time Conditions (Current Style)

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

## Frontend Semantic Restrictions

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
   out = cond.select(v0, v1)
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

## Runtime Loops with Loop-Carried Values (Software Pipelining)

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

## Arithmetic Operations

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

## Internal Types: Vector and Numeric (PREFERRED)

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

## Arith Ops Availability Table

| Operation | Function | Works on Vectors | Notes |
|-----------|----------|-----------------|-------|
| Add | `a + b` | Yes | Use direct FOp only for explicit fastmath |
| Multiply | `a * b` | Yes | Use direct FOp only for explicit fastmath |
| Negate | `-a` | Yes | |
| Max / Min | `fx.max(a, b)` / `fx.min(a, b)` | Yes | Float forms propagate NaN; `fx.maxnumf` / `fx.minnumf` do not |
| Integer ceil-div | `fx.ceildiv(a, b)` | Yes | Direct signed/unsigned op; distinct from layout `fx.ceil_div` |
| Compare | `arith.cmpf(pred, a, b)` | Yes | predicate FIRST; returns i1/vec<i1> |
| Select | `cond.select(t, f)` | Yes | |
| Abs | `abs(v)` or `fx.absf(v)` | Yes | FlyDSL helper; not an arith dialect op |
| FMA | `a * b + c` | Yes | Use direct FOp only when explicit fastmath is needed |
| Splat const | `Vec.filled(width, val, dtype)` | Creates vector | For scalar broadcast |

## Printf Debugging

```python
fx.printf("tid={} bid={} val={}", tid, bid, value)
```
