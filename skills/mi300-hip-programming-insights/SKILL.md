---
name: mi300-hip-programming-insights
description: CDNA3/MI300 HIP programming insights—chiplet/cache model, Infinity Cache, memory coherency, matrix cores, sparsity, and best practices.
---

# MI300 HIP Programming Insights

Use when tuning HIP kernels with CDNA3 architectural context (chiplets, caches, matrix cores).

Highlights:
- Memory hierarchy: 128B cache lines; leverage 256MB Infinity Cache (temporal locality); explicit sync across XCDs (relaxed coherency).
- Workgroups: size for 4 ACEs per XCD; balance across 38 CUs; exploit shared I-cache locality; LDS 64KB per CU.
- Matrix cores: align data; overlap matrix + vector + memory; choose FP8/TF32 for throughput vs precision; schedule for concurrency.
- Sparsity: 2:4 structured sparsity (INT8/FP8/FP16/BF16); weigh reordering overhead vs gains; good for attention/conv.
- Cross-platform: HIP differences vs CUDA—explicit fences, data-type fallbacks, platform-specific tuning.
- Debug/profiling: use ROCm tools to analyze cache misses, bandwidth, sync overhead; focus on memory-side cache behavior.

References:
- `references/AMD MI300 HIP Kernel Programming Guide_ CDNA3 Architecture Insights.md`
