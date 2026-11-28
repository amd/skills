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

**Repository**: this checkout (examples below assume it is importable — see 11)
**Target GPU**: gfx942 (MI300X, CDNA3), gfx950 (MI350, CDNA4)
**Python**: 3.12, ROCm 7.2

**Scope**: this skill is the **reference** (layout-algebra API, per-op tables, MFMA/copy-atom catalogs, environment variables, troubleshooting) to look things up while writing or reviewing kernel code. For a guided requirement-to-tested-kernel procedure use **flydsl-tile-programming**; for a kernel that compiles but produces NaN/inf/wrong results use **debug-flydsl-kernel**; to migrate legacy constructs use **kernel-code-cleanup**.

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
1. `fly-canonicalize` + `fly-rocdl-expand-ops` - Canonicalize Fly ops and expand target-specific operations
2. `fly-layout-lowering` - Lower layout algebra (`fly.crd2idx`, partitions, divides) to arithmetic
3. `fly-convert-atom-call-to-ssa-form` + `fly-promote-regmem-to-vectorssa` - Lift copy/MMA atom calls and register memory to vector SSA
4. `convert-fly-to-rocdl` - Convert Fly values (signatures, calls, launches and control flow alike) to the nested packed ABI, and Fly ops to ROCDL intrinsics
5. `gpu-module-to-binary{format=fatbin}` - Emit HSACO binary via LLVM AMDGPU backend

### Key Source Paths
- `python/flydsl/compiler/` - JIT compilation (jit_function.py, kernel_function.py)
- `python/flydsl/expr/` - DSL expression API: `primitive.py` (all layout algebra functions), `derived.py` (CopyAtom, MmaAtom, TiledCopy, TiledMma wrappers), `typing.py`
- `python/flydsl/expr/gpu.py` - `thread_id`/`block_id`, `barrier`, `SharedAllocator` for LDS (shared memory)
- `python/flydsl/expr/rocdl/` - MFMA/WMMA and other ROCm intrinsics
  (package: cdna3, cdna4, cdna5, rdna3, rdna4, cluster, inline_asm, tdm_ops, universal;
  plus `utils.py` / `enum.py` helpers)
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
tx = gpu.thread_id("x")
bx = gpu.block_id("x")
gpu.barrier()                                      # workgroup synchronization

layout = fx.make_layout((4, 64), (64, 1))
coord = fx.idx2crd(tx, layout)
wave_id = fx.get(coord, 0)
lane_id = fx.get(coord, 1)

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

### Control Flow
```python
for i in range_constexpr(N):              # compile-time unrolled; i is a Python int
    ...
for i in range(runtime_value):            # runtime scf.for (Python-int bounds too)
    ...
for k in fx.range(N, unroll_full=True):   # runtime loop with an LLVM unroll hint; or unroll=4; unroll=1 disables
    ...
```

A Python `if` on a DSL value is lowered by the AST rewriter to `scf.if`, and comparison operators (`==`, `<`, `>=`, ...) on `fx` values build the predicate. Compile-time conditions (Python booleans, `Constexpr` arguments, loop-unroll choices, type/layout branches) must use `const_expr(...)`: a plain `if` on a Python `bool` still picks one branch at trace time, but it is rewritten like a runtime `if`, so names first assigned inside it are undefined after it.

```python
lane = gpu.thread_id("x") % fx.Int64(64)
if lane == fx.Int64(0):                       # runtime -> scf.if
    ...
val = (lane < fx.Int64(8)).select(good_val, zero_val)
if const_expr(max_context_partition_num <= WARP_SIZE):   # compile-time
    ...
```

Do **not** wrap GPU runtime values in `const_expr`: `if const_expr(lane == c_zero)` is wrong. Even with `@flyc.kernel(known_block_size=(256, 1, 1))`, `gpu.thread_id("x")`, `lane`, and `warp_id` are runtime SSA values; the compiler knows their range, not the current lane instance. Unwrap a condition only for hand-built MLIR, e.g. `scf.IfOp((partition_idx >= visible_tile_count).ir_value(), has_else=False)`.

### Frontend Semantic Restrictions
When writing or reviewing `@flyc.kernel` / `@flyc.jit` code, proactively avoid these patterns because they can conflict with MLIR construction even if they look valid in plain Python.

1. **Before a runtime `if`, bind every name that any of its branches assigns, to an IR value of the assigned type.** Only names bound before the `if` become `scf.if` results; a name first defined inside the branches raises `NameError` when used after the `if`. The rewriter picks the results by syntax, before `const_expr` is evaluated, so `const_expr` branches count too: a name assigned in an `elif const_expr(...)` branch becomes a result once it is bound anywhere earlier, even inside an `if const_expr(...)` block. Where that earlier binding is `None` or was skipped, tracing fails with `Variable(s) [...] initialized as None before a dynamic if/else (scf.if)`. Bind a placeholder first (e.g. `nxt = fx.Int32(0)`, not `None` and not only under `if const_expr(PREFETCH):`), or move the `const_expr`-gated assignment out of the chain.

2. **Do not mutate captured outer variables inside nested helper functions.** Read-only closure capture is acceptable; writes go through parameters and return values (`acc = helper(acc)`).

3. **Avoid early `return`, and do not place `return` / `yield` inside `if/else` branches.** Bind the result before the `if`, assign it in each branch, and return once after, so the frontend can determine result types. A branch `return` behaves as in Python only when the condition is a Python value at trace time; under a dynamic condition, or inside a `range(...)` / dynamic `while` loop body, tracing raises `NotImplementedError`.

4. **Runtime branches inside helper functions should be dispatched via local `@flyc.jit`.** When a branch body has side effects, loop-carried values, or branch-local definitions, split the branch bodies into local helpers and wrap the `if` in a local JIT helper:
   ```python
   @flyc.jit
   def dispatch():
       if runtime_cond:
           then_path()
       else:
           else_path()

   dispatch()
   ```

### Runtime Loops with Loop-Carried Values (Software Pipelining)

`init=` on `range()` gives a runtime `scf.for` with explicit loop-carried state (block arguments, i.e. phi nodes), which software pipelining uses to carry prefetched data across iterations.

**Pattern** (from `preshuffle_gemm.py`):
```python
tile_0 = prefetch(0)                                        # prologue: load first tile
for iv, state in range(0, N - 1, 1, init=[acc_init] + tile_0):
    acc_in, tile_in = state[0], state[1:]
    next_tile = prefetch(iv + 1)                            # load NEXT data
    acc_in = compute(acc_in, tile_in)                       # compute CURRENT
    results = yield [acc_in] + next_tile                    # carry to next iteration
compute(results[0], results[1:])                            # epilogue: last tile
```

| Element | Meaning |
|---|---|
| `init=[...]` | Values that seed the loop block arguments for iteration 0 |
| `state` | The loop-carried block arguments (phi nodes) for THIS iteration |
| `yield [...]` | Feeds values back as next iteration's `state` |
| `results` | After loop exits, holds the last yielded values |

**Pitfalls:**

1. **Every `range(...)` in a kernel is a runtime `scf.for`, Python-int bounds included; only `range_constexpr` unrolls.** With `init=`, the bounds become `index` values and `init` is always kept. Without `init=`, the rewriter infers the carried state itself and uses `i32` bounds (`i64` if any bound is `i64`). Use `range_constexpr` when the body needs a Python int, e.g. to index a Python list.

2. **`init` and `yield` take DSL values** (`fx.Int32`, `fx.Float32`, `fx.Vector`) and convert them to `ir.Value` for you. Unwrap with `v.ir_value()` only where a lower-level helper requires a raw `ir.Value`.

3. **Build LDS views at the top of the kernel, not inside the runtime loop.** Allocate shared memory with `fx.SharedAllocator().allocate(...).peek()` and build each `.view()` once up front. A view created inside the `scf.for` body is defined in the loop scope; using it in the epilogue (outside the loop) causes an SSA dominance error. Building it once at the top makes it dominate both the loop and the epilogue.

### Arithmetic and `fx.Vector`

Use typed values and Python operators instead of raw MLIR ops. Keep a direct `arith.*FOp` only when explicit fastmath flags are required.

```python
Vec = fx.Vector
c42 = fx.Int64(42); c3_14 = fx.Float32(3.14)    # typed constants
idx = fx.Int32(gpu.block_id("x") * tile_m)      # cast; replaces arith.index_cast
acc = Vec(frag_C.load())                        # wrap a vector<Nxf32>
val = acc[i]                                    # replaces vector.extract
v_f32 = Vec(raw_vec).bitcast(fx.Float32)        # replaces vector.bitcast
bf16_val = f32_val.to(fx.BFloat16)              # replaces arith.trunc_f / ext_f
zeros = Vec.filled(N, 0.0, fx.Float32)          # splat; Vec.from_elements(...) builds from scalars
result = (val * scale_a) * scale_b              # replaces arith.mulf / addf
```

| Operation | Spelling | Notes |
|-----------|----------|-------|
| Add / Mul / Negate / FMA | `a + b`, `a * b`, `-a`, `a * b + c` | Work on vectors too |
| Max / Min | `fx.max(a, b)` / `fx.min(a, b)` | Float forms propagate NaN; `fx.maxnumf` / `fx.minnumf` do not |
| Integer ceil-div | `fx.ceildiv(a, b)` | Signed/unsigned from the operand types; distinct from layout `fx.ceil_div` |
| Compare | `a < b`, `a == b`, ... | Integer and float operands |
| Select | `cond.select(t, f)` | |
| Abs | `abs(v)` or `fx.absf(v)` | The arith dialect has no `absf`; FlyDSL provides one |

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

1. **Raw `buffer_load` offset**: the `offset` parameter is in ELEMENTS, not bytes.
   The raw intrinsics live in `kernels/common/buffer_ops.py`, not `flydsl.expr`.

2. **LDS overflow**: Check capacity (64KB on gfx942, 160KB on gfx950). `SharedAllocator` sizes the LDS global for you, but by default nothing rejects a total over the arch limit at compile time: on FlyDSL 0.5.0, a gfx942 kernel whose static `@fx.struct` storage came to 69248 bytes compiled and cached, and its launch failed with `hipErrorIllegalState` from `hipModuleLaunchKernel`. Since 0.5.1, launching under `with CompilationContext.compile_hints({"check_resource_limits": True}):` turns that into a compile error. When the size depends on host parameters, also call `check_smem_capacity(nbytes, arch)` from `flydsl.runtime.device` before compiling; it raises with both byte counts.

3. **Tensor layout marking**: For dynamic shapes or alignment, use `flyc.from_dlpack(tensor).mark_layout_dynamic(leading_dim=0, divisibility=4)`.

4. **AMD wavefront size**: Always 64 on gfx9xx. Use shifts [32, 16, 8, 4, 2, 1] for full-wave reduction.

5. **tile_k alignment for GEMM**: `tile_k * elem_bytes` must be divisible by 64 (K64-byte micro-step).

6. **INT4 (W4A8)**: A matrix is int8, B matrix is packed int4 (2 values/byte), unpacked to int8 in-kernel.

A suspected stale compile: see "Disk Cache Invalidation" in [references/environment-and-debugging.md](references/environment-and-debugging.md).

---

## 11. Running Kernels

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
