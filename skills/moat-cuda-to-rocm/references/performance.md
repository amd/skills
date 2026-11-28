# Performance: what a correct port can still lose

A port whose results match CUDA can still be many times slower on AMD: how to find the cost, and the causes measured so far.

<!-- toc:begin -->
Contents:

- Finding the cost
- Measured causes
<!-- toc:end -->

## Finding the cost

### Profile the hot kernels and read their ISA before blaming tile tables or wavefront size

A port that is byte-identical in output can still be an order of magnitude slower. The
causes in this file were measured, not guessed: each was swapped in and out one at a time
on the same binary and input. Before profiling, rule out an unoptimized build: a CMake HIP
build with no build type compiles every kernel at `-O0` (build-cmake-link.md, "An unset CMake
build type runs every HIP kernel unoptimized"), and a kernel launched with far fewer than
1024 threads may need `__launch_bounds__` to get its registers (platform-limits.md). Then an
hour is enough to find the cost:

1. `rocprofv3 --kernel-trace --stats -d <dir> -o run --output-format csv -- <cmd>` gives
   per-kernel time and call counts (`run_kernel_stats.csv`). rocprofv3 aborts if
   `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are both set, so unset the CUDA one.
2. Pull the code objects out of the binary (`llvm-objdump --offloading <bin>`, which writes
   `<bin>.N.hip-amdgcn-amd-amdhsa--gfxXXXX` next to it) and read `llvm-readelf --notes` for
   each hot kernel: `.vgpr_count`, `.vgpr_spill_count`, `.private_segment_fixed_size`
   (scratch). Then `llvm-objdump -d` the hot kernel and count instructions in its inner
   loop. Compare against a reference build (upstream's own HIP path, or the CUDA path's
   intent) at the same kernel.
3. Swap one difference at a time. When the hot code lives in a shared library, build only
   that target per variant and point `LD_LIBRARY_PATH` at it; `DT_RUNPATH` yields to it.

(MMseqs2)

## Measured causes

### Scalar emulation of a packed intrinsic (`__hmax2` as two `__hmax`) cost 15x or more

CUDA has `__hmax2`/`__hmin2` for `half2`; HIP (checked through HIP 7.16) has only the
scalar `__hmax`/`__hmin`, and ROCm's scalar `__hmax` is three explicit `__hisnan` tests plus
a compare. Emulating `__hmax2` as two scalar `__hmax` calls is correct, but the compiler
keeps the NaN tests as compares, selects and `s_and_saveexec` branches instead of one
`v_pk_max_f16`. In a max-heavy inner loop that is roughly 14x more instructions per
iteration, and the bloat then pushes `#pragma unroll` loops over the unroll threshold, so
per-thread register arrays land in scratch.

Use the compiler's packed builtin, which lowers to `v_pk_max_f16` and keeps CUDA's
non-NaN-operand semantics (`llvm.maxnum`):

```
typedef _Float16 h2v __attribute__((ext_vector_type(2)));
return __builtin_bit_cast(__half2, __builtin_elementwise_max(
    __builtin_bit_cast(h2v, a), __builtin_bit_cast(h2v, b)));
```

Do not use `asm volatile("v_pk_max_f16 ...")`. It emits the right instruction, but
`volatile` pins every asm block in program order, so the scheduler cannot hoist LDS loads
ahead of them and the loop waits on each load. Plain `asm` (no `volatile`) recovers most of
it; the builtin is fastest and portable across gfx9/gfx11/gfx12.

Measured (MMseqs2 `ungappedprefilter`, 1000 queries, gfx1100, identical output in every
row): scalar `__hmax` emulation 328 s; `asm volatile` 19.4-21.0 s; non-volatile `asm`
11.4 s; `__builtin_elementwise_max` 9.2-10.5 s. The same pattern applies to any SIMD-in-word
CUDA intrinsic emulated lane by lane (`__vmaxs2`, `__viaddmax_*`, `__vadd4`): check whether
a packed AMD instruction or an `__builtin_elementwise_*` form exists before writing scalar
lanes, and look at the ISA of the result either way. (MMseqs2)

### Tile and block-size tables are second order, and a project's own autotuner can pick a slower one

Pinning a conservative NVIDIA tile table (T4/sm75) instead of an AMD-tuned one cost about
10% once the arithmetic was fast, and the "AMD-tuned" table was the slower of the two on
RDNA3. With the slow scalar emulation above, the tuned table was 37% slower, because its
larger per-thread tiles spill. Fix the instruction stream first, then tune tiles with the
profiler.

A project's own autotuner can still pick the wrong table. On an RDNA4 card (gfx1201) a full
sweep of MMseqs2/libmarv's tuner, choosing the fastest config per tile size, gave a table
that was 2-9% slower end to end than the untuned generic one on two database sizes, with
identical results. The tuner ranked 512-thread blocks 10-40% above the generic 128-thread
configs, but in the application the generic table, built from 128-thread configs, was
faster, and a different RDNA table did no better. Time the real workload against the
existing configs before keeping a tuned table, and keep the raw tuner output so the
comparison can be checked. (MMseqs2)

### Compare kernel variants warmed up, interleaved, against a noise floor

On a shared multi-GPU board (other jobs at 100% on sibling GPUs), the same code's kernel times
varied 4-18x between short processes, and per-process profiler runs starting from an idle GPU
compared different clock states: a change looked 15-48% faster in 11 of 12 cells that way and
was 0-54% slower when timed after a warm-up. What held up: warm up inside the process and time
many launches; interleave the two builds in rounds (A, B, A, B) and take the median of per-run
medians; include in the same rounds a metric whose code is identical in both builds, as the
noise floor; and for functional regressions that show only as speed (a degenerate but valid
acceleration structure passes correctness tests), add a query or throughput check of what was
built. (warp-1.14)

### What did not matter: RDC layout and maskless shuffles

A SHARED library with `-fgpu-rdc` versus a static library with CMake's device link: the same
time within noise for the same source. Redirecting `__shfl_*_sync` to maskless shuffles had
no effect, because the hot kernels use cooperative-groups tile shuffles, which compile to
the same `ds_bpermute_b32` either way. (MMseqs2)
