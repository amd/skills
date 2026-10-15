# Architecture and compilation

## Architecture and Compilation

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

## Comparison with Triton/Gluon

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
