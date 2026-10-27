## 8. Common Patterns and Recipes

### Element-wise Kernel Template
```python
@flyc.kernel
def elementwise_kernel(In: fx.Tensor, Out: fx.Tensor, BLOCK: fx.Constexpr[int], VEC: fx.Constexpr[int]):
    bid, tid = fx.block_idx.x, fx.thread_idx.x
    tile = BLOCK * VEC
    tIn = fx.logical_divide(In, fx.make_layout(tile, 1))
    tOut = fx.logical_divide(Out, fx.make_layout(tile, 1))
    tIn = fx.slice(tIn, (None, bid))
    tOut = fx.slice(tOut, (None, bid))
    tIn = fx.logical_divide(tIn, fx.make_layout(VEC, 1))
    tOut = fx.logical_divide(tOut, fx.make_layout(VEC, 1))
    copy = fx.make_copy_atom(fx.UniversalCopy(VEC * 32), fx.Float32)
    rIn = fx.make_rmem_tensor(VEC, fx.Float32)
    rOut = fx.make_rmem_tensor(VEC, fx.Float32)
    fx.copy(copy, fx.slice(tIn, (None, tid)), rIn)
    # Transform
    v = Vec(fx.memref_load_vec(rIn))
    v = v * v  # example: square
    fx.memref_store_vec(v, rOut)
    fx.copy(copy, rOut, fx.slice(tOut, (None, tid)))
```

### Element-wise Kernel Cookbook (GPU-Verified)
All recipes below follow the same vectorized copy_atom pattern (256 threads, vec_width=4, 128-bit loads).
Only the compute section between `memref_load_vec` and `memref_store_vec` differs.

```python
# --- Scale: C = A * scalar ---
vA = Vec(fx.memref_load_vec(rA))
scale = Vec.filled(vec_width, 2.0, fx.Float32)
vC = vA * scale

# --- Multiply: C = A * B ---
vC = Vec(fx.memref_load_vec(rA)) * Vec(fx.memref_load_vec(rB))

# --- FMA: D = A * B + C ---
vAB = Vec(fx.memref_load_vec(rA)) * Vec(fx.memref_load_vec(rB))
vD = vAB + Vec(fx.memref_load_vec(rC))

# --- ReLU: C = max(A, 0) ---
vA = Vec(fx.memref_load_vec(rA))
zero_vec = Vec.filled(vec_width, 0.0, fx.Float32)
vC = fx.max(vA, zero_vec)

# --- Abs: C = |A| ---
vC = abs(fx.memref_load_vec(rA))        # or fx.absf(...); memref_load_vec already
                                        # returns a Vector, so no Vec() wrap needed
```

### Naive GEMM Template (for understanding, not performance)
```python
@flyc.kernel
def naive_gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
               M: fx.Constexpr[int], N: fx.Constexpr[int], K: fx.Constexpr[int],
               BM: fx.Constexpr[int], BN: fx.Constexpr[int]):
    tid, bid = gpu.thread_id("x"), gpu.block_id("x")
    bm, bn = bid // (N // BN), bid % (N // BN)
    tm, tn = tid // BN, tid % BN
    row, col = bm * BM + tm, bn * BN + tn

    # Buffer-resource views with logical row-major layouts
    tA = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(A)), fx.make_layout((M, K), (K, 1)))
    tB = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(B)), fx.make_layout((K, N), (N, 1)))
    tC = fx.make_view(fx.get_iter(fx.rocdl.make_buffer_tensor(C)), fx.make_layout((M, N), (N, 1)))

    copy = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    rA, rB, rC = (fx.make_rmem_tensor(1, fx.Float32) for _ in range(3))

    acc = fx.Float32(0.0)
    for k in range_constexpr(K):
        fx.copy(copy, fx.slice(tA, (row, k)), rA)
        fx.copy(copy, fx.slice(tB, (k, col)), rB)
        acc = acc + fx.Vector(fx.memref_load_vec(rA))[0] * fx.Vector(fx.memref_load_vec(rB))[0]

    fx.memref_store_vec(fx.Vector.from_elements([acc], fx.Float32), rC)
    fx.copy(copy, rC, fx.slice(tC, (row, col)))
```
