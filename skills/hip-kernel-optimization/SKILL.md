---
name: hip-kernel-optimization
description: This skill should be used when writing or tuning HIP kernels on AMD/NVIDIA GPUs, covering memory coalescing, shared-memory tiling, bank conflict avoidance, warp primitives, occupancy, vectorization, async ops, loop unrolling, and profiling.
---

# HIP Kernel Optimization

## Purpose
Provide ready patterns for efficient HIP kernels and guide diagnosis of memory throughput, occupancy, and synchronization bottlenecks.

## When to Use
- Implementing or reviewing HIP kernels for AMD MI/CDNA architectures or CUDA-portable code
- Porting CUDA code to HIP while retaining performance
- Preparing profiling runs with `rocprof`

## Optimization Priority

**Phase 1: Low-hanging fruit** (try first, low risk)
1. `#pragma unroll` on hot loops with small, fixed trip counts
2. Enable `-ffast-math` compiler flag for floating-point kernels
3. Use 32B vectorized loads/stores instead of 16B
4. Add `__launch_bounds__(maxThreads, minBlocks)` to guarantee occupancy
5. Add `const` qualifiers on read-only pointers
6. Verify memory coalescing (consecutive threads → consecutive addresses)

**Phase 2: Targeted improvements** (profile first)
7. Profile with `rocprof` to confirm bottleneck
8. If memory-bound: CK-Tile buffer views with vectorization
9. If compute-bound: Shared memory tiling
10. Dynamically calculate block size based on problem dimensions
11. Replace large 2D shared arrays with atomicAdd for sparse patterns
12. Provide multiple block size configurations to avoid register spill
13. Add explicit rounding mode control for numerical correctness
14. Pre-compute workspace size to avoid dynamic allocation
15. Implement CSV-based tuning cache for repeated GEMM shapes

**Phase 3: Complex transformations** (high effort)
16. Algorithm changes (e.g., Top-K-only softmax)
17. gfx950: Use 16x16x32 MFMA instead of 2x 16x16x16
18. Kernel fusion (multi-op in single kernel)
19. Persistent kernels for repeatedly executed operations
20. Shape-based heuristic dispatching

**Anti-patterns**:
- Optimizing everything at once
- Manual loop unrolling (use `#pragma unroll` instead)
- Over-unrolling (factor > 8)
- Premature vectorization without alignment check
- Unnecessary buffer coherence flags (e.g., `glc`)

## Core Optimization Patterns

### 1. Memory Access
- **Coalescing**: Map consecutive threads to consecutive addresses; prefer SoA over AoS
- **Vectorization**: Use CK-Tile buffer views for efficient I/O; prefer 32B loads over 16B
- **Boundary handling**: Separate fast vectorized path from slow boundary path
  ```cpp
  if(idx + VEC_SIZE <= d) {
      vec_o out_vec;
      #pragma unroll
      for(size_t j = 0; j < VEC_SIZE; j++) {
          out_vec[j] = compute(x[j], y[j]);
      }
      buffer_out.template set(idx, 0, true, out_vec);  // Fast path
  } else {
      for(size_t j = 0; j < VEC_SIZE; j++) {          // Boundary path
          if(idx + j < d) ptr_out[idx + j] = compute(...);
      }
  }
  ```

### 2. Shared Memory
- **Tiling**: Load tiles once, reuse; balance TILE_SIZE vs occupancy
- **Bank conflicts**: Pad shared arrays (e.g., `[32][33]`) or rotate access
- **Sparse patterns**: Use atomicAdd to 1D counters (O(N)) instead of 2D arrays (O(N²))
  ```cpp
  // Three-pass pattern for sparse bucketing
  // Pass 1: Count items per category
  for(int i = start_idx; i < end_idx; ++i) {
      int32_t category_id = input_ids[i];
      atomicAdd(&category_counts[category_id], 1);
  }
  __syncthreads();

  // Pass 2: Compute prefix sum for offsets
  if(threadIdx.x == 0) {
      for(int i = 0; i < num_categories; ++i)
          cumsum[i+1] = cumsum[i] + category_counts[i];
  }
  __syncthreads();

  // Pass 3: Assign items using atomic write positions
  for(int i = start_idx; i < end_idx; ++i) {
      int32_t position = atomicAdd(&write_positions[input_ids[i]], 1);
      sorted_output[position] = i;
  }
  ```

### 3. Warp/Wavefront Primitives
- Use `__shfl_*`, ballots, and warp reductions to reduce shared memory
- Pattern for warp-level argmax:
  ```cpp
  auto arg_max = [](const kvp& a, const kvp& b) {
      return (a.value > b.value || (a.value == b.value && a.key < b.key)) ? a : b;
  };
  kvp thread_kvp = {item_id, max_val};
  thread_kvp = warp_reduce(thread_kvp, arg_max, WARP_SIZE);
  ```

### 4. Occupancy Tuning
- **Dynamic block sizing**: Calculate based on problem dimensions
  ```cpp
  int vec_size = nextPow2(d / 64);
  vec_size = min(vec_size, max_vec_size);
  int num_wave = min(nextPow2(d / 64 / vec_size), max_wave_num);
  dim3 block(max(num_wave, 1) * 64);
  ```

- **Guaranteed occupancy**: Use `__launch_bounds__` for predictable performance
  ```cpp
  __launch_bounds__(256, 8) __global__  // 256 threads, min 8 blocks per CU
  void kernel(scalar_t* __restrict__ output, ...) { }
  ```

- **Register spill prevention**: Provide multiple block size options
  ```cpp
  if (MPerBlock == 64)
      gemm_kernel<..., 64, ...>(...);
  else if (MPerBlock == 128)
      gemm_kernel<..., 128, ...>(...);
  else if (MPerBlock == 256)
      gemm_kernel<..., 256, ...>(...);
  ```

- **Adaptive grid sizing**: Don't use fixed grid for variable problem sizes; adapt to small dimensions

### 5. Loop Unrolling
- Apply `#pragma unroll` for small, fixed trip counts
- Unroll vector processing: `#pragma unroll` before `for(size_t j = 0; j < VEC_SIZE; j++)`

### 6. Async Memory Operations
- Overlap H2D/D2H with compute using multiple streams and `hipMemcpyAsync`

## AMD-Specific Optimizations

### 7. MFMA Instructions (gfx940/942/950)
- **gfx950**: Use single 16x16x32 MFMA instead of 2x 16x16x16
  ```cpp
  #if defined(__gfx950__)
  dout = gcn_mfma16x16x32_instr<scalar_t, 0, 0, 0>(K, Q, dout);
  #else
  for(int i = 0; i < 2; i++) {
      dout = gcn_mfma16x16x16_instr<scalar_t, 0, 0, 0>(K.xy[i], Q.xy[i], dout);
  }
  #endif
  ```
- Use `__builtin_shufflevector` to reorganize data for larger MFMA variants

### 8. Inline Assembly for Packed Operations
- **v_pk_mul_f32**: Process two floats at once
  ```cpp
  float2 result;
  asm volatile("v_pk_mul_f32 %0, %1, %2\n\t"
               "v_pk_mul_f32 %0, %0, %3"
               : "=v"(result) : "v"(act_vals), "v"(y_vals), "v"(scale_vals));
  ```

### 9. Compiler Flags
- **-ffast-math**: Enables aggressive FP optimizations
- **Avoid unnecessary coherence**: Don't use `ck_tile::amd_buffer_coherence_enum::glc` unless required

## Advanced Optimization Strategies

### 10. Algorithm-Level Optimizations
- **Top-K-only softmax**: Only compute exp on top-K values, not entire row
  ```cpp
  float thread_max = find_max_in_row();
  for(int k_idx = 0; k_idx < k; ++k_idx) {
      kvp top = find_argmax_in_remaining();
      output[k_idx] = expf(top.value - thread_max);
      renorm_value += output[k_idx];
      row_chunk[top.index] = -INFINITY;
  }
  float row_sum_rest = compute_sum_of_remaining_exp(thread_max);
  normalize_top_k(renorm_value + row_sum_rest);
  ```

- **Kernel fusion**: Combine operations to reduce launches (e.g., norm+RoPE+cache+quant)
- **Persistent kernels**: Keep kernels resident on GPU for repeated operations

### 11. Numerical Precision
- **Explicit rounding**: Add rounding mode parameters for attention kernels
- **FP8 descale**: Apply descaling during computation to avoid separate kernel

### 12. Kernel Selection and Dispatching
- **CSV tuning cache**: Cache optimal configs to eliminate repeated tuning
  ```cpp
  int get_algoIdx_from_csv(const std::string filename, ...) {
      // Parse CSV and match (trans_a, trans_b, m, n, k, dtypes)
      for each line:
          if (all_params_match) return algo_index;
      return -1;  // Not found
  }
  ```

- **Shape-based dispatch**: Use heuristics for kernel selection
  ```cpp
  Kernel select_kernel(int M, int N, int K) {
      if (M < 128) return gemm_small_m<...>;
      else         return gemm_large_m<...>;
  }
  ```

- **Workspace pre-calculation**: Compute exact size before allocation
  ```cpp
  int64_t ws_size = topkValue * (sizeof(T) + sizeof(IdxT)) * numRows;
  auto workspace = allocate_device_memory(ws_size);
  ```

## Quick Reference
- Kernel launch: `hipLaunchKernelGGL(kernel, dim3(grid), dim3(block), sharedMem, stream, args...)`
- Memory: `hipMalloc`, `hipMemcpy`, `hipFree`
- Sync: `__syncthreads()`, `hipDeviceSynchronize()`
- Atomics: `atomicAdd`, `atomicCAS`
- CK-Tile: `ck_tile::make_buffer_view<ck_tile::address_space_enum::global>(ptr, oob)`

## Profiling
- Summary: `rocprof --stats program`
- Detailed: `rocprof --hip-trace --hsa-trace program`
- Metrics: `rocprof -i metrics.txt program`

## Validation Checklist
- [ ] Coalesced loads/stores; bank conflicts minimized
- [ ] Vectorized I/O aligned and beneficial
- [ ] Occupancy >50%; no register spilling
- [ ] Shared memory: atomicAdd for sparse patterns (O(N) not O(N²))
- [ ] Loops unrolled for small fixed trips
- [ ] `-ffast-math` enabled for FP kernels
- [ ] No unnecessary coherence flags
- [ ] gfx950: Using 16x16x32 MFMA

## Performance Impact (Production-Validated)

| Optimization | Use Case | Typical Impact |
|-------------|----------|----------------|
| `#pragma unroll` | Memory kernels | +3-5% |
| AtomicAdd sparse | MOE, sorting | +15-20%, O(N²)→O(N) |
| 32B vectors | Memory-bound | Better throughput |
| `-ffast-math` | Math-heavy | +5-10% FP |
| Top-K softmax | Gating | Reduce exp by 50-90% |
| 16x16x32 MFMA | Attention | 2x→1x calls |
| `__launch_bounds__` | Position encoding | Guaranteed occupancy |
| Multiple MPerBlock | GEMM stages | Fix register spill |
| Persistent kernels | Paged attention | -50-80% launch overhead |
| CSV cache | GEMM tuning | Eliminate repeat tuning |
