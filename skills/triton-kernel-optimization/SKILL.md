---
name: triton-kernel-optimization
description: This skill should be used when writing or tuning Triton GPU kernels, including autotuning block sizes, coalesced accesses, tiled matmul, fused ops, reductions, flash-attention style kernels, quantization, custom gradients, and profiling.
---

# Triton Kernel Optimization

## Purpose
Provide production-validated patterns and tuning tactics for performant Triton kernels on AMD MI-series GPUs.

## When to Use
- Authoring new Triton kernels for PyTorch or standalone use
- Porting CUDA/HIP concepts into Triton with equivalent performance
- Profiling and benchmarking Triton kernels

## Optimization Priority

**Phase 1: Foundation** (correct and basic performance)
1. Use `@triton.autotune` with configs covering key block sizes (64/128/256)
2. Use `@triton.heuristics` for compile-time optimizations (e.g., `EVEN_K`)
3. Apply `tl.assume` for stride positivity to help compiler optimize
4. Separate boundary handling from main computation path
5. Use `tl.constexpr` for all compile-time constants

**Phase 2: Memory Optimization**
6. Implement cache modifiers (`.ca`, `.cg`) for L2 cache control
7. Use split-K for improved L2 reuse on large K dimensions
8. Apply XCD remapping (`remap_xcd`) for multi-die GPUs (MI250X, MI300)
9. Optimize GROUP_SIZE_M for better L2 locality
10. Pre-shuffle weight layouts for better vectorization

**Phase 3: Advanced Techniques**
11. Implement persistent kernels for repeated operations
12. Use attention sink for stable long-context attention
13. Fuse quantization with GEMM (e.g., blockscale + matmul)
14. Apply per-token or per-tensor quantization strategies
15. Use grouped GEMM for mixture-of-experts workloads

**Anti-patterns**:
- Hardcoding block sizes without autotune
- Ignoring tail handling (non-divisible shapes)
- Not using `tl.assume` for known constraints
- Excessive register pressure from large tile sizes
- Unnecessary synchronization or atomic operations

## Core Optimization Patterns

### 1. Autotuning and Heuristics

**Autotune configuration**:
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64,
                       'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32,
                       'GROUP_SIZE_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32,
                       'GROUP_SIZE_M': 4}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],  # Tune based on problem dimensions
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0,
    'GRID_MN': lambda args: triton.cdiv(args['M'], args['BLOCK_SIZE_M'])
                          * triton.cdiv(args['N'], args['BLOCK_SIZE_N']),
})
@triton.jit
def gemm_kernel(..., EVEN_K: tl.constexpr, GRID_MN: tl.constexpr):
    # Use EVEN_K to skip boundary checks in hot loop
    if EVEN_K:
        a = tl.load(a_ptrs)  # No mask needed
    else:
        a = tl.load(a_ptrs, mask=mask_k)
```

### 2. Split-K for Large K Dimensions

**Production pattern - Split-K GEMM**:
```python
# Split K dimension across multiple thread blocks
pid_unified = tl.program_id(axis=0)
pid_k = pid_unified % NUM_KSPLIT
pid = pid_unified // NUM_KSPLIT

# Each block processes SPLITK_BLOCK_SIZE elements
SPLITK_BLOCK_SIZE = tl.cdiv(K, NUM_KSPLIT)
num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)

# Offset K dimension by split index
offs_k_split = pid_k * SPLITK_BLOCK_SIZE + tl.arange(0, BLOCK_SIZE_K)

# Accumulate partial results
for k in range(num_k_iter):
    a = tl.load(a_ptrs + k * BLOCK_SIZE_K * stride_ak)
    b = tl.load(b_ptrs + k * BLOCK_SIZE_K * stride_bk)
    accumulator += tl.dot(a, b)

# Write partial result to temporary buffer
tl.store(c_ptr + pid_k * stride_ck, accumulator)

# Separate reduction kernel combines splits
```

### 3. XCD Remapping for Multi-Die GPUs

**Production pattern - XCD-aware PID mapping**:
```python
from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd, pid_grid

# For MI250X/MI300 with multiple chiplets
if NUM_KSPLIT == 1:
    remap_xcd(pid, GRID_MN)  # Remap PIDs for balanced die utilization
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
```
- **Purpose**: Balance work across chiplets on multi-die GPUs
- **When**: Single-K-split kernels on MI250X/MI300
- **Impact**: Better L2 cache locality and die utilization

### 4. Stride Assumptions and Cache Modifiers

**Production pattern - Compiler hints**:
```python
# Help compiler optimize by asserting stride properties
tl.assume(stride_am > 0)
tl.assume(stride_ak > 0)
tl.assume(stride_bk > 0)
tl.assume(stride_bn > 0)

# Cache modifiers for L2 control
a = tl.load(a_ptrs, cache_modifier=".ca")  # Cache all levels
b = tl.load(b_ptrs, cache_modifier=".cg")  # Cache global only
```

### 5. Blockscale Quantization Patterns

**Production pattern - FP8 blockscale GEMM**:
```python
# Load quantized int8 data
a_int8 = tl.load(a_ptr + offs)
b_int8 = tl.load(b_ptr + offs)

# Load per-block scales
GROUP_K = BLOCK_SIZE_K  # Typically block size = group size
a_scale = tl.load(a_scale_ptr + offs_m * stride_ascale_m + k_block * stride_ascale_k)
b_scale = tl.load(b_scale_ptr + k_block * stride_bscale_k + offs_n * stride_bscale_n)

# Compute in int32, then dequantize
accumulator_int32 += tl.dot(a_int8, b_int8, out_dtype=tl.int32)

# Dequantize with scales (broadcasting)
result_fp = accumulator_int32.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
```

### 6. Weight Preshuffling

**Production pattern - Optimized weight layout**:
```python
# Instead of loading weights in standard layout:
# b_ptr shape: [K, N]

# Preshuffle weights offline for better vectorization:
# b_preshuffled shape: [K // BLOCK_K, N // BLOCK_N, BLOCK_K, BLOCK_N]
# Allows full BLOCK_K x BLOCK_N tiles to be loaded contiguously

# In kernel, simplified loading:
b_block = tl.load(b_ptr + block_idx * (BLOCK_K * BLOCK_N))
# Reshape and use directly
```
- **Benefit**: Better memory coalescing and vectorization
- **Trade-off**: Requires offline weight preprocessing

### 7. Attention Sink Support

**Production pattern - Stable long-context attention**:
```python
# Standard attention: softmax over all keys
# Problem: Numerical instability for long sequences

# Attention sink: Keep first few tokens' attention stable
sink_size: tl.constexpr  # e.g., 4 or 8 tokens

# Separate handling for sink tokens
if qk_idx < sink_size:
    # Always keep sink tokens in attention
    qk_scale = 1.0
else:
    # Apply causal masking to non-sink tokens
    qk_scale = (qk_idx <= q_idx)

# Compute attention with sink preservation
attn_weight = tl.where(qk_scale > 0, tl.exp(qk - m_ij), 0.0)
```
- **Purpose**: Prevent attention collapse in long-context scenarios
- **When**: Prefill/decode with context > 4K tokens

### 8. Grouped GEMM for MOE

**Production pattern - MOE expert routing**:
```python
# Instead of separate GEMM per expert:
# for each expert: C[expert] = A[tokens_for_expert] @ W[expert]

# Batched approach with routing:
expert_ids = tl.load(expert_id_ptr + token_idx)
token_offset = tl.load(token_offset_ptr + token_idx)

# Load weight for selected expert
w_ptr = weight_base_ptr + expert_ids * expert_stride
w = tl.load(w_ptr + offs)

# Accumulate with proper indexing
output_ptr = out_base_ptr + token_offset * output_stride
tl.atomic_add(output_ptr, result)  # Multiple tokens may target same expert
```

### 9. Fused Operations

**Production pattern - Fused GEMM + activation**:
```python
# Fuse matmul with gating (common in FFN)
# Instead of: hidden = silu(linear1(x)) * linear2(x)  # Two kernels

# Fused single kernel:
@triton.jit
def fused_ff_gated_kernel(...):
    # Load input once
    x = tl.load(x_ptr + offs)

    # Compute both branches
    gate_result = tl.dot(x, w_gate)
    up_result = tl.dot(x, w_up)

    # Apply activation and multiply in registers
    gate_activated = gate_result / (1.0 + tl.exp(-gate_result))  # SiLU
    result = gate_activated * up_result

    tl.store(out_ptr, result)
```
- **Benefit**: Single load of `x`, reduced memory bandwidth
- **Common fusions**: GEMM + bias, GEMM + ReLU/GELU/SiLU, GEMM + residual

### 10. Per-Token Quantization

**Production pattern - Dynamic quantization**:
```python
# Quantize activations per-token at runtime
@triton.jit
def per_token_quant_gemm(...):
    # Load FP input
    a_fp = tl.load(a_ptr + offs)

    # Compute per-token (per-row) scale
    a_max = tl.max(tl.abs(a_fp), axis=1)  # Max per row
    a_scale = a_max / 127.0  # FP8 range

    # Quantize
    a_int8 = (a_fp / a_scale[:, None]).to(tl.int8)

    # Standard int8 matmul
    b_int8 = tl.load(b_ptr + offs)
    acc = tl.dot(a_int8, b_int8, out_dtype=tl.int32)

    # Dequantize with per-token scale
    b_scale = tl.load(b_scale_ptr + offs_n)
    result = acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]
```

### 11. Memory Access Optimization

**Production pattern - Coalesced loads**:
```python
# Ensure fastest-changing dimension matches memory layout
offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

# For row-major A[M, K]: stride_am > stride_ak
a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

# For column-major B[K, N]: stride_bn > stride_bk
b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

# Mask for boundary handling
mask_m = offs_m < M
mask_n = offs_n < N
mask_k = offs_k < K

a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :])
```

### 12. Persistent Kernel Pattern

**Production pattern - Reduce launch overhead**:
```python
# For operations called repeatedly with same shape
@triton.jit
def persistent_kernel(..., NUM_ITERATIONS: tl.constexpr):
    # Process multiple iterations without re-launching
    for iter in range(NUM_ITERATIONS):
        # Load iteration-specific data
        data = tl.load(data_ptr + iter * stride_iter)

        # Process
        result = compute(data)

        # Store
        tl.store(out_ptr + iter * stride_iter, result)
```

## Quick Reference

**Kernel structure**:
```python
@triton.autotune(configs=[...], key=[...])
@triton.heuristics({...})
@triton.jit
def kernel(ptr_args, scalar_args, COMPILE_TIME: tl.constexpr):
    # 1. Get program ID
    pid = tl.program_id(axis=0)

    # 2. Compute offsets
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. Load with masks
    data = tl.load(ptr + offs, mask=offs < SIZE)

    # 4. Compute
    result = tl.dot(data, weights) if matmul else compute(data)

    # 5. Store with masks
    tl.store(out_ptr + offs, result, mask=offs < SIZE)
```

**Core operations**:
- Program IDs: `tl.program_id(axis=0/1/2)`
- Offsets: `tl.arange(0, BLOCK_SIZE)`
- Memory: `tl.load(ptr, mask=..., cache_modifier=...)`, `tl.store(ptr, val, mask=...)`
- Math: `tl.dot(a, b)`, `tl.sum(x, axis=...)`, `tl.max(x)`, `tl.exp(x)`
- Atomics: `tl.atomic_add(ptr, val, mask=...)`

## Profiling

```python
# Benchmarking
from triton.testing import do_bench

latency_ms = do_bench(lambda: kernel[grid](...))

# Profiling with triton profiler
import triton.profiler as profilr

with profiler.profile():
    kernel[grid](...)
print(profiler.key_averages().table())
```

## Validation Checklist

- [ ] Autotune covers block sizes 64/128/256 with varying num_warps (2/4/8)
- [ ] Heuristics optimize for EVEN_K or other compile-time conditions
- [ ] `tl.assume` assertions for stride positivity
- [ ] Masks guard all boundary conditions (tail M, N, K)
- [ ] Cache modifiers applied for L2 optimization
- [ ] Split-K used for large K (>4096) GEMMs
- [ ] XCD remapping for MI250X/MI300 multi-die GPUs
- [ ] Fused operations reduce memory traffic
- [ ] Quantization scales properly broadcast
- [ ] Per-token vs per-tensor quantization chosen appropriately

## Performance Impact (Production-Validated)

| Optimization | Use Case | Typical Impact |
|-------------|----------|----------------|
| Autotune block sizes | All kernels | 1.5-3x vs default |
| `EVEN_K` heuristic | Divisible shapes | +5-10% (skip masks) |
| Split-K GEMM | Large K (>4K) | +20-40% throughput |
| XCD remapping | MI250X/MI300 | +10-15% utilization |
| Weight preshuffle | GEMM | +5-15% memory efficiency |
| Fused GEMM+activation | FFN layers | -30-50% memory traffic |
| Blockscale quant | INT8 GEMM | 2-3x vs FP16 |
| Per-token quant | Dynamic ranges | Better accuracy vs per-tensor |
| Attention sink | Long context (>8K) | Prevent collapse |
| Cache modifiers | Large tensors | +5-10% L2 hit rate |
| Grouped GEMM | MOE (8+ experts) | -50% vs sequential |
| Persistent kernels | Repeated calls | -20-40% launch overhead |
