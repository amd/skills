---
name: gpu-architecture-fundamentals
description: This skill should be used when reasoning about GPU architecture fundamentals to guide kernel optimization choices such as memory hierarchy usage, execution model mapping, block sizing, and latency-aware tuning across HIP, Triton, and PyTorch.
---

# GPU Architecture Fundamentals

## Purpose
- Reference core GPU concepts (memory hierarchy, execution model) and typical bandwidth/latency numbers to ground optimization choices.
- Provide block size heuristics and ready-to-use checklists before writing or tuning kernels.
- Map common optimization patterns across HIP, Triton, and PyTorch to pick framework-specific tactics quickly.

## When to Use
- Planning or reviewing kernel designs where occupancy, memory bandwidth, or latency hiding are concerns.
- Selecting grid/block shapes, deciding on shared memory usage, or checking for coalesced accesses.
- Comparing optimization levers across frameworks when porting kernels.

## How to Use
- Recall memory hierarchy: prefer registers > shared/L1 > L2 > HBM; treat HBM as ~400–800 cycle latency, registers ~0, shared ~20–30 cycles.
- Anchor bandwidth sense-checks with table values (e.g., MI300X HBM3 ~5.3 TB/s, A100 HBM2e ~2.0 TB/s).
- Choose block sizes by operation: element-wise 256–1024 threads, reduction 256–512, matmul tiles 128x128 or 256x128, conv 32x32 or 64x64.
- Apply execution model mapping: thread ↔ element/partial tile, warp/wavefront ↔ contiguous data segments, block/workgroup ↔ tiles sharing shared memory, grid ↔ full problem coverage.
- Run the optimization checklist before finalizing kernels:
  - Ensure coalesced and vectorized memory access; avoid shared memory bank conflicts.
  - Target occupancy >50%; watch register pressure and shared memory usage to avoid spilling.
  - Fuse operations where possible; leverage mixed precision when valid.
  - Overlap transfers with compute; tune block/grid dimensions; unroll small loops.
- Use pattern summaries to pick tactics per framework:
  - Memory: HIP manual strides/shared, Triton `tl.arange`/implicit tiling, PyTorch `.contiguous()`/compiler.
  - Compute: HIP manual fusion/unroll, Triton `@triton.jit` + `tl.constexpr`, PyTorch `torch.compile`/FlashAttention.
  - Parallelism: HIP block/grid + occupancy APIs, Triton autotune + constexpr block sizes, PyTorch compiler/automatic launch config.

## Quick Checks
- If performance regresses, compare achieved block size and occupancy to table heuristics.
- If L2/HBM traffic is high, add tiling or fusion; if shared memory stalls, check bank conflicts and tile padding.
- When switching hardware, re-evaluate bandwidth and latency assumptions and retune block sizes accordingly.
