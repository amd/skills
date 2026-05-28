---
name: pytorch-kernel-optimization
description: This skill should be used when optimizing PyTorch models and kernels, including efficient tensor operations, torch.compile, custom autograd/CUDA/Triton extensions, mixed precision, memory and data pipeline tuning, model optimization techniques, CUDA graphs, and profiling.
---

# PyTorch Kernel Optimization

## Purpose
- Equip PyTorch workflows with concrete optimization patterns from high-level APIs to custom kernels.
- Provide practical snippets for compilation, extensions, mixed precision, memory efficiency, and profiling.

## When to Use
- Tuning PyTorch models for throughput/latency on GPU.
- Deciding between compiler-level optimizations and custom kernels (C++/CUDA/Triton).
- Profiling and addressing bottlenecks in compute or input pipelines.

## How to Use
- **Efficient tensor ops**: favor contiguous layouts (`.contiguous()` when needed); use `channels_last` for convs; replace Python loops with vectorized ops; prefer in-place ops (`add_`, `mul_`, `out=`) when autograd-safe.
- **torch.compile**: wrap functions or models with `@torch.compile`; choose modes:
  - `"default"` balanced, `"reduce-overhead"` for small batches/CUDA graphs, `"max-autotune"` for peak perf, `"max-autotune-no-cudagraphs"` when graphs undesirable.
  - Use `fullgraph=True` for whole-graph capture; set `dynamic=False` when shapes are static.
- **Custom autograd**: implement `torch.autograd.Function` saving minimal tensors; recompute in backward when memory-bound (e.g., checkpointed attention); use custom backward formulas for fused ops (e.g., SiLU).
- **CUDA extensions**: build with `CUDAExtension` (`-O3`, `--use_fast_math`, `-arch=sm_80`); enforce input checks in C++ bindings; expose kernels via `PYBIND11_MODULE`.
- **Mixed precision**: train with `torch.cuda.amp` + `GradScaler`; mix dtypes per op if needed; leverage `bfloat16` when supported.
- **Memory optimization**: apply gradient checkpointing (`checkpoint`, `checkpoint_sequential`); use memory-efficient attention via `scaled_dot_product_attention`; consider activation offloading (CPU swap) when memory-bound.
- **Data loading**: configure `DataLoader` with `num_workers`, `pin_memory`, `prefetch_factor`, `persistent_workers`, `drop_last`; implement fast collate; prefetch to GPU with custom loader using streams and non-blocking copies.
- **Model optimization**: fuse Conv+BN (`fuse_conv_bn`), apply quantization (`quant.fuse_modules`, `prepare`, `convert`), prune weights via `torch.nn.utils.prune`; ensure evaluation mode during quantization calibration.
- **CUDA graphs**: capture steady workloads via `torch.cuda.CUDAGraph`; warm up then capture forward/backward; reuse static input/output buffers; note `torch.compile(mode=\"reduce-overhead\")` can leverage graphs automatically.
- **Profiling**:
  - Use `torch.profiler.profile` with CPU/CUDA activities, schedules, and `tensorboard_trace_handler`; enable `record_shapes`, `profile_memory`, `with_stack`.
  - Review `prof.key_averages().table(sort_by=\"cuda_time_total\")`; iterate on hotspots.

## Validation Checklist
- Tensor layouts contiguous/channels_last as appropriate; Python loops eliminated; in-place ops safe for autograd.
- `torch.compile` mode chosen for workload; warmup complete; performance measured post-compilation.
- Custom ops (autograd or CUDA) validate device/contiguity; register usage and block sizes tuned for kernels.
- AMP scaling stable (no inf/nan); dtype choices align with numerical sensitivity.
- Data loader keeps GPU fed (no data starvation); streams overlap transfers where applicable.
- Profiling reviewed after each major change; bottlenecks addressed or noted.
