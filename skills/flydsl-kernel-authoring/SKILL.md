---
name: flydsl-kernel-authoring
description: >-
  Authors and reviews FlyDSL GPU kernels using explicit layout algebra, copy
  atoms, tiled MMA, LDS allocation, and loop-carried state. Applies when writing
  or understanding FlyDSL kernel code, mapping tensors to threads, choosing
  MFMA or WMMA atoms, or fixing FlyDSL frontend and compilation errors.
  Runtime trace interpretation belongs to kernel trace analysis.
---

# FlyDSL Kernel Authoring

Write and review kernels with FlyDSL's explicit tensor layouts and MLIR-based
compiler. Use the reference matching the operation being changed; the examples
assume imports of `flydsl.compiler as flyc`, `flydsl.expr as fx`, and
`gpu`, `rocdl`, `const_expr`, `range_constexpr` from `flydsl.expr`.

## Prerequisites

- Default execution environment: Linux, Python 3.12, ROCm 7.2, and a compatible
  ROCm PyTorch build. The reference targets MI300X (`gfx942`, CDNA3) and MI350
  (`gfx950`, CDNA4). The separately marked `gfx1250` WMMA/TDM examples require
  that target and a toolchain that supports it; do not apply them to CDNA.
- Use the user's existing FlyDSL installation and checkout. Repository paths
  such as `python/`, `kernels/`, and `tests/` below are relative to the FlyDSL
  checkout, not to this installed skill. If no checkout exists, obtain
  [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) for examples and tests; follow its
  [installation guide](https://github.com/ROCm/FlyDSL/blob/1c29367b414457a78167e92b465208b0b84255fc/docs/installation.rst) for a wheel or source build.
- No container image is required or pinned by this skill. Prefer the existing
  compatible environment; record the image and ROCm version if using a container.
  GPU execution needs access to `/dev/kfd` and `/dev/dri` and a compatible host driver.
- Keep `ARCH` consistent with the actual GPU. Check `HIP_VISIBLE_DEVICES` and
  `ROCR_VISIBLE_DEVICES` before benchmarking. Do not use `HSA_OVERRIDE_GFX_VERSION`
  to make an unsupported GPU appear supported. Debug and cache variables are in
  [runtime.md](runtime.md).
- Reading or reviewing code needs no GPU. Mark correctness and performance as
  unverified when a compatible execution environment is unavailable.

## Authoring workflow

1. Establish the operation, shapes, strides, dtypes, numerical tolerance, and
   target architecture. Reuse a matching example from the user's checkout.
2. Choose the layout before the instruction: map logical coordinates to memory,
   divide into tiles, then partition those tiles across threads and values.
   See [layout-algebra.md](layout-algebra.md).
3. Define a `@flyc.kernel` and a `@flyc.jit` launch wrapper. Use `fx.Tensor`,
   typed runtime scalars, and `fx.Constexpr` parameters deliberately. See
   [kernel-language.md](kernel-language.md) for syntax and frontend restrictions.
4. Move data with copy atoms and `fx.rocdl.make_buffer_tensor` where applicable.
   Allocate LDS and build views before runtime loops so they dominate loop and
   epilogue uses. Match MMA/WMMA atoms to the architecture and dtype. See
   [data-movement.md](data-movement.md).
5. Handle tails explicitly or state and enforce divisibility requirements.
   The compact [examples](examples.md) show the main data path; they are not
   general masked-tail implementations.
6. Compare with a PyTorch or CPU reference on representative and boundary shapes,
   including supported dtypes and non-contiguous inputs if promised. Run the
   affected tests in the FlyDSL checkout. Benchmark only after correctness passes,
   and report target, shapes, timing method, and whether measurements were run.

## Decisions that matter

- Use `gpu.thread_id/block_id`, `fx.Int32/Int64/Float32`, and `fx.Vector` for new
  code. Preserve established legacy style when a small edit does not need migration.
- A GPU thread ID is a runtime SSA value even with `known_block_size`. Use a
  runtime `if` for it; reserve `const_expr` for compile-time decisions.
- A pipelined `range(..., init=...)` needs typed DSL bounds such as `fx.Int64`.
  Plain Python integer bounds unroll and can silently discard loop-carried state.
- Use `fx.make_layout_tv` or `fx.make_tiled_copy_tv` for thread/value layouts;
  a bare `raked_product` does not construct the same mapping.
- CDNA wavefronts contain 64 lanes. Do not carry that assumption into the
  explicitly marked wave32 WMMA examples.
- `waves_per_eu` is a hint conveyed through LLVM attributes. `maxnreg` was removed;
  passing flags through `gpu-module-to-binary opts=` does not control AMD codegen.

## Reference map

| Reference | Read when |
| --- | --- |
| [architecture.md](architecture.md) | Locating compiler passes or comparing the layout model with Triton/Gluon |
| [layout-algebra.md](layout-algebra.md) | Constructing, composing, dividing, or querying layouts |
| [kernel-language.md](kernel-language.md) | Choosing types, control flow, vectors, or loop-carried values |
| [data-movement.md](data-movement.md) | Partitioning copies, allocating LDS, selecting matrix atoms, or reducing |
| [examples.md](examples.md) | Adapting an elementwise or educational GEMM skeleton |
| [runtime.md](runtime.md) | Running tests, inspecting IR/debug info, autotuning, or diagnosing compiler errors |
