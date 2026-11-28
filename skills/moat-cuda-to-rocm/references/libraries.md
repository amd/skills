# Libraries: substitutions and gaps

Which ROCm library replaces which CUDA library, and where the replacement is not one-to-one.

<!-- toc:begin -->
Contents:

- Substitution table
- BLAS, hipBLASLt and Composable Kernel
- MIOpen
- hipSPARSE
- rocThrust, hipCUB and rocPRIM
- CuPy on ROCm
- Random numbers, profiling and monitoring
- RAPIDS (ROCm-DS)
- RCCL
- Image and video codecs
<!-- toc:end -->

## Substitution table

**Library swaps.** "Drop-in" below means the hip* library keeps the CUDA library's call
shapes, so a compat header can alias the names; it does not mean every routine or mode
exists. A row marked "not verified" states what the library is, not what a port measured.

| CUDA | ROCm | drop-in? | where it is not 1:1 |
|---|---|---|---|
| cuBLAS | hipBLAS (rocBLAS underneath) | yes | handle types; the v2 enums change a few signatures; 16-bit GEMM accuracy (faults-numerics.md) |
| cuBLASLt | hipBLASLt | mostly | epilogues, compute types, the ReLU aux format ("BLAS, hipBLASLt and Composable Kernel" below) |
| cuFFT | hipFFT (rocFFT) | yes | plan creation during graph capture (faults-memory-streams.md); hipFFT complex types under heFFTe (below) |
| cuRAND | hipRAND (rocRAND) | yes | device generator state internals; C++14 from ROCm 10.1 |
| cuSPARSE | hipSPARSE (rocSPARSE) | yes | half-precision SpMM, zero workspace sizes, graph capture ("hipSPARSE" below) |
| cuSOLVER | hipSOLVER (rocSOLVER) | partly; not verified | hipSOLVER has a cuSOLVER-style dense API (`hipsolverDn*`). None of the ports behind this file exercised more than handle creation, so check every routine you call against hipSOLVER's function list, and its workspace-size and `info` conventions, before assuming a rename is enough. One gap is known: the low-level sparse factorization interface (`cusolverSpXcsrqr*` and `csrchol*`, with analysis, setup, factor and solve as separate calls), `csrlsvluHost` and `csrsymrcm` have no hipSOLVER counterpart (requested in ROCm/hipSOLVER issue 443), and cuDSS has none either; a sparse direct solve needs another library (RXMesh) |
| cuDNN | MIOpen | NO | a different API (descriptors, find-then-run instead of fixed algorithm enums): the call layer is rewritten, and whole operations are missing ("MIOpen" below) |
| Thrust, CUB | rocThrust, hipCUB (rocPRIM) | yes | `thrust/system/cuda/error.h`, temp-storage sizes, segmented `ArgMax` ("rocThrust, hipCUB and rocPRIM" below) |
| libcu++ | libhipcxx | yes | `cuda::` spelling kept; ROCm 10.1 and later bundle CCCL 3.0.2 as `hipccl`; 1- and 2-byte CAS, under-aligned 8-byte `atomic_ref` ("rocThrust, hipCUB and rocPRIM" below) |
| NCCL | RCCL | yes | same `nccl*` API under another soname (`librccl.so.1`); validation needs several GPUs ("RCCL" below) |
| nvCOMP | hipCOMP (ROCm-DS) | not verified | ships inside the `amd-libhipdf` wheel; hipDF 3.0's Parquet writer still reports no Zstandard compressor, so do not assume codec parity |
| NPP | RPP (`rpp.h`) | NO; not verified | AMD's image primitives have their own API, so calls are rewritten rather than renamed; coverage against NPP was not measured. `rpp.h` collides with `miopen.h` ("MIOpen" below) |
| NVTX | roctx | ranges only | no analogue of the NVTX injection API |
| NVML | AMD SMI | NO | a different API, keyed by PCI address |
| nvJPEG, nvImageCodec | rocJPEG | decode only | 8-bit baseline JPEG; no encoder, no JPEG 2000 ("Image and video codecs" below) |
| NVDEC (cuvid) | rocDecode | NO | no FFmpeg hwaccel; you drive the parser and decoder |
| NVENC | none | -- | name the software encoder path |
| CUPTI | none | -- | see the next entry |
| RMM, cuDF, cuVS | hipMM, hipDF, hipVS (ROCm-DS) | yes | "RAPIDS (ROCm-DS)" below |
| CUTLASS, CuTe, TensorRT | Composable Kernel, FlyDSL, MIGraphX | reimplementation | no-hip-equivalent.md |

**Negative entries for that table.** CUPTI has no ROCm analogue -- stub the optional
per-kernel profiling path inert and let the timer fall back to `hipEvent` wall-clock rather
than substituting something. ck_tile's fused MHA ships headers but no prebuilt instance
library in `/opt/rocm/lib`, so `fmha_fwd` is declared-only and using it means vendoring CK's
codegen ("Fused FP16 attention" below). (cuSZ, lc0)

**A swapped library may be less defensive about bad arguments than the CUDA one.** Where
cuFFT returns an error code, rocFFT can hang. `hipfftPlan1d(&plan, 0, HIPFFT_C2C, 1)` --
transform length zero -- never returns on ROCm 7.2.3, while cuFFT answers
`CUFFT_INVALID_SIZE`; `hipfftPlan1d(..., batch=0)` returns SUCCESS rather than an error.
This bites when you deliberately force an error to test a port's error-check macros or its
failure paths: pick a trigger that is verified to return, such as
`hipfftExecC2C(plan, nullptr, nullptr, dir)` (status 6, on a planned or unplanned handle).
For hipBLAS the equivalent verified trigger is a GEMM with negative extents:
`hipblasCgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, -1, -1, -1, &alpha, nullptr, 1, nullptr,
1, &beta, nullptr, 1)` returns `HIPBLAS_STATUS_INVALID_VALUE` (3) on a valid handle and
touches no device memory, so it is safe to fire from a probe. More generally, before
concluding a port hangs, reproduce the hang in a few-line program that includes none of the
project's headers: that separates a library bug from your change. (TurboFNO)

**heFFTe's rocFFT backend does not accept hipFFT complex types.** heFFTe's cuFFT backend
recognizes `cufftComplex`/`cufftDoubleComplex`, but its rocFFT backend registers no hipFFT
type. A CUDA code that keeps its FFT buffers as `cufftComplex` and maps the name to
`hipfftComplex` (which is `HIP_vector_type<float,2>`) fails to compile with `static assertion
... check_types<rocfft, float, HIP_vector_type<float, 2>>: Using either an unknown complex
type`. heFFTe documents `heffte::is_ccomplex<T>` and `is_zcomplex<T>` as the specialization
point for custom complex types: specialize them for `hipfftComplex` and
`hipfftDoubleComplex` in the HIP branch, with no patch to heFFTe. heFFTe then reinterprets
the buffers as `std::complex`, whose layout they share. (ESPResSo)

**Choosing the NVIDIA or AMD build of a library at run time.** A runtime with both a CUDA and a
HIP backend must pick per device, from the device's own backend, and never by enumerating or
starting every installed backend: creating a backend that cannot start (HIP on an NVIDIA host)
throws, and a static initializer that does so breaks the library on the device that works. Where
the two builds have different sonames (NCCL `libnccl.so.2` and RCCL `librccl.so.1`), bind both,
each loaded on first use. Where the HIP port keeps the soname (hipVS `libcuvs_c.so`, hipDF
`libcudf.so`), a process can hold only one, so detect which loaded: a `dlsym` through the
library's handle also searches its dependencies, so finding `hipGetDevice` marks the HIP build.
Then serve only devices of that backend. (TornadoVM)

## BLAS, hipBLASLt and Composable Kernel

**A scalar's WIDTH follows the compute/scale type, not the matrix type.** When a ROCm
library refuses the narrow compute type its CUDA counterpart accepted and you widen it to
32F, the `alpha`/`beta` passed through a `const void*` must widen too. The library reads
four bytes through that pointer; aiming it at a two-byte `half` scales by whatever adjacent
bytes happen to be there -- large enough to give infinities in one call, near enough to zero
to erase the product in the next, with no error status anywhere. Grep every call whose
compute or scale type you changed for the scalars that go with it. (marian-dev: hipBLASLt
rejects `HIPBLAS_COMPUTE_16F` and hipSPARSE rejects `HIP_R_16F` as an SpMM compute type;
both call sites kept passing `half` scalars, and the hipBLASLt one survived review and three
platform validations because the assertion that would have caught it was never reached.)

**`hipblasGemmEx` rejects `HIPBLAS_COMPUTE_32F_FAST_16F` for FP16 inputs.** A shim that maps
`CUBLAS_COMPUTE_32F_FAST_16F` to `HIPBLAS_COMPUTE_32F_FAST_16F` (the natural spelling since
`hipblasComputeType_t` arrived) gets `HIPBLAS_STATUS_NOT_SUPPORTED` (7) on an FP16 x FP16 -> FP32
GEMM. hipBLAS documents FAST_16F as "32-bit input can use 16-bit compute", and the rocBLAS
backend does not take it with 16-bit inputs. Map it to `HIPBLAS_COMPUTE_32F` for those GEMMs
(older shims mapped to `HIPBLAS_R_32F`, which is why they worked), and check every compute-type
enum a shim maps, not only the ones the main path uses. (koboldcpp, ROCm 10.2, gfx942)

**hipBLASLt epilogues are not cuBLASLt's, per type and per format.** On gfx90a with hipBLASLt
1.4, DEFAULT/BIAS/RELU/RELU_BIAS exist for f32/f16/bf16 but double has only DEFAULT, and the
AUX/DRELU/BGRAD epilogues exist only for f16. cuBLASLt's ReLU auxiliary output is a 1-bit mask
(leading dimension a multiple of 128) that callers size and may write themselves; hipBLASLt's is
the pre-activation matrix in the output type, so mapping the enum alone writes far past a buffer
sized for a mask. Probe availability with `hipblasLtMatmulAlgoGetHeuristic` per type and fall
back to a plain GEMM plus a small kernel. hipBLASLt cannot read a layout's TYPE back
(`hipblasLtMatrixLayoutGetAttribute` returns invalid attribute) and fixes the compute type at
descriptor creation, so a shim has to remember both. (oneflow)

**PyTorch's ROCm MX GEMM wants unswizzled scales, M, N and K in multiples of 32, and no
bias.** `torch.nn.functional.scaled_mm(..., ScalingType.BlockWise1x32, ...)` on ROCm (gfx950,
hipBLASLt rocRoller, torch 2.12) rejects cuBLAS's `SWIZZLE_32_4_4` scale layout; it needs
`NO_SWIZZLE` with plain row-major `[M, K/32]` e8m0 scales (transposed scales run but return
garbage). M, N and K must all be multiples of 32: pad the activation rows with zeros, not only
the scales. A `bias=` argument fails at the heuristic query (`HIPBLAS_STATUS_INVALID_VALUE`),
and the output must be BF16 or FP16. So a CUDA MXFP8 linear ports as: quantize to row-major
scales, pad rows to 32, GEMM without bias, slice, add the bias. Per-tensor FP8 `_scaled_mm`
takes `float8_e4m3fnuz` (max 240) on gfx942 and only `float8_e4m3fn` (max 448) on gfx950 and
gfx12, so pick the e4m3 type per device (as PyTorch's own tests do with `e4m3_type`) instead
of hard-coding either. (Matrix-Game)

**Probe `torch._scaled_mm`'s fp8 types once per device instead of listing architectures.** On
gfx942 it accepts only `float8_e4m3fnuz`/`float8_e5m2fnuz` and fails `HIPBLAS_STATUS_NOT_SUPPORTED`
for the OCP types; gfx950 and gfx12 take OCP; gfx90a and gfx11 have none. Try a 16x16
`_scaled_mm` with the OCP pair, then the FNUZ pair, once per device (cached, and excluded from
`torch.compile`), and cast to whatever works. A test of the fp8 GEMM should skip on "no pair
works", not on the compute-capability number, which on ROCm is an AMD architecture number.
Keep fp8 data exchanged between ranks in the OCP formats so it does not depend on the GPU.
(ColossalAI)

**Generic GEMM on CK device operations.** For CUTLASS-style row-major GEMMs with fused
epilogues: `DeviceGemmMultipleD_Xdl_CShuffle` on CDNA (FP16/BF16/FP32),
`DeviceGemmMultipleD_Wmma_CShuffleV3` on RDNA3/4 (FP16/BF16; its `MakeArgumentPointer` takes
an extra split-K argument), `DeviceGemmMultipleD_Dl` for FP32 on RDNA. Try each and run the
first whose `IsSupportedArgument` passes on the current device. Epilogues are elementwise
functors `op(e, c, d...)`; a bias is a D operand with row stride 0. Operand vectors must
respect the real pointer alignment (8 bytes for data after a 24-byte header: 4-element FP16
vectors). The DL operation has no K padding (needs M, N, K tile multiples) and its D-operand
epilogue gave wrong results on gfx950: run it on zero-padded copies and combine `beta * C` in
a small store kernel. CK's `ck/config.h` is generated: `configure_file` its `config.h.in` with
`CK_ENABLE_FP16/BF16/FP32`, `CK_ENABLE_DL_KERNELS`, `CK_USE_XDL`, `CK_USE_WMMA`. (TornadoVM)

One XDL instance can serve every CDNA generation. An instance taken from a gfx950 example
(K1=16, sized for the double-rate MFMA) looks gfx950-only, but CK picks the MFMA instruction
per offload target at compile time (`xdlops_gemm.hpp`, `GetMfma<half_t,32,32>`:
`mfma_f32_32x32x16f16` under `__gfx950__`, `mfma_f32_32x32x8f16` otherwise; KPack only has to
be a multiple of `k_per_blk`), and `IsSupportedArgument` checks only `is_xdl_supported()` plus
layout and vector conditions. Compile the same instance for every CDNA target and confirm it
from the code objects (disassemble per target; check `.wavefront_size`, LDS and scratch)
before adding a K1=8 tile, which is a tuning choice, not a correctness need. (oidn, CK
3e6d21ad, ROCm 10.2)

**Composable Kernel older than tag rocm-7.2.0 races on gfx11 LDS barriers.** A project that
vendors or pins such a CK and runs its WMMA (or any LDS-staged) kernels on RDNA3/RDNA3.5
(gfx11xx) can get wrong, nondeterministic results, while the same code is correct on gfx10,
gfx12 and CDNA. `ck::block_sync_lds()` there emits `__builtin_amdgcn_s_waitcnt(0xc07f)`, the
gfx9/gfx10 encoding of `lgkmcnt(0)`, which on gfx11 decodes to `vmcnt(48) lgkmcnt(7)`, so
`s_barrier` does not wait for LDS operations. Disassemble the gfx11 code object and look for
`s_waitcnt vmcnt(48) lgkmcnt(7)` right before `s_barrier`; the symptoms also go away at device
`-O0`. Fixed in CK #3103 (331273b, in rocm-7.2.0); moving the pin to a 7.2.x tag is enough.
Hand-written `s_waitcnt` immediates are architecture-specific in general: gfx9/10, gfx11 and
gfx12 each encode the counters differently. (oidn)

**Fused FP16 attention (cuDNN SDPA, flash attention): MIOpen has no path, CK's FMHA does.**
MIOpen's find-2.0 MHA problem (`miopenCreateMhaProblem`, ROCm 10.1 SDK) offers only the
`MhaForward` solver on gfx950: FP32, packed BHSD only (no strided tensors), and the causal
mask (`miopenTensorMhaMask`) is not applied -- the output equals the unmasked attention. For
FP16 generate ck_tile FMHA instances: fetch CK's `include` and `example/ck_tile/01_fmha`
(sparse checkout of a pinned commit), run `generate.py --api fwd --optdim 32,64,128,256
--filter <fnmatch>` and compile the instances with the FMHA example's flags; call
`fmha_fwd(traits, args, stream_config{stream})` from a small C ABI. Gotchas: the filter
matches the kernel name WITHOUT the `_gfx950` suffix (end the pattern at `_nsink`, or it
matches nothing); release tag rocm-7.2.4 generates no gfx11 forward kernels (develop does);
cuDNN's `set_causal_mask` is CK `mask_top_left` (window left -1, right 0); keep
`-fgpu-flush-denormals-to-zero` on the FMHA sources only; for one build spanning gfx9 and gfx11
define `USE_NEW_UNIFIED_FRAMEWORK=0` (CK develop's unified framework needs a single host-side
target, and `CK_CMAKE_GPU_TARGET_IDS` takes the first for every pass, failing the gfx11
device pass). 139 FP16 inference instances for
nine architectures build in about 2.5 minutes at 128 jobs. (TornadoVM)

**PyTorch's fused SDPA on gfx1100 fails at head dims 257 to 512 with an unrelated HIP error.**
On gfx1100 with torch 2.12 (ROCm 10.2), `F.scaled_dot_product_attention` takes the AOTriton
flash or memory-efficient kernel for those head dims, and the launch fails with `CUDA error:
profiler is not initialized` (`hipErrorProfilerNotInitialized`), after which the context is
unusable. Head dims up to 256 are fine; above 512 the dispatcher finds no kernel and falls back
to the math path. VAEs with single-head spatial attention over the full channel count (head
dim 512) hit it. Rerun the call under `with sdpa_kernel([SDPBackend.MATH])` to tell it from a
real fault, and route such calls to the math path, chunked if the [L, L] matrix is large.
(Matrix-Game)

**flash-attn 2.8.3's Triton AMD backward reads `dout` with `out`'s strides.** In flash-attn
2.8.3.post2's Triton AMD backend (the RDNA path, `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`), the
backward preprocess indexes `dout` with `out`'s strides. A non-contiguous `dout` whose last
stride is 1 (the gradient of `out.transpose(1, 2)`) passes `maybe_contiguous` and gives wrong
dq and dk while dv stays correct. The same backend returns `rng_state=None` without dropout,
which breaks `torch.compile` against the op's fake implementation. Both are fixed on flash-attn
main, where the backend comes from ROCm/aiter; no release had the fix as of 2026-11. Test an
FA2 integration on RDNA with a transposed upstream gradient and under `torch.compile`, not only
with a contiguous `dout`. (ColossalAI)

**rocBLAS `gemv` over more than 65535 batches returns wrong entries, and `cupy.einsum` reaches it.**
Root cause (https://github.com/ROCm/rocm-libraries/issues/14819): for transA=N and m < 2n, rocBLAS
selects `rocblas_gemvn_kernel<32,16>`, which loops over the batches inside the kernel in steps of
65535 (the grid z limit) and has no `__syncthreads()` after its shared-memory reduction, so a wave
starting the next batch overwrites partial sums another wave is still reading. Only non-final loop
iterations are corrupted, so batch_count <= 65535 is always clean, and large m is affected too (m =
n = 16 fails, m = 32 with n = 64 fails under load). `gemv_strided_batched` reaches it on
gfx11xx/gfx12xx; on gfx942/gfx90a small shapes take a different kernel and the rest fail only under
heavy load. `gemm_strided_batched_ex` with m or n = 1 reaches it where rocBLAS routes that call to
gemv (gfx1100); on gfx1201 and gfx942 it goes to Tensile, which splits the batch on the host, and is
clean. complex128 fails on an idle RDNA GPU; float64 mostly needs a second process on the GPU, so
test-runner parallelism decides whether CI sees it.

How a CuPy project reaches it: CuPy computes each einsum pairwise step as `(B, L, K) @ (B, K, R)`
after dropping size-1 axes, so any step with L or R = 1 is a candidate, for example
`einsum('xg,xg->g')`, and so is a structurally clean contraction whose free index has size 1 (one
state, spin set or k-point). Fix it in the project, not in CuPy: do not replace `cupy.matmul` on the
CuPy module (a process-global monkeypatch of a third-party library that misses `@`; a reviewer
rejected it). Add a project-local `einsum` that follows CuPy's default pairwise order and, on ROCm,
computes a matrix-vector step with more than 65535 batches as a broadcast multiply and sum (on CUDA
it is `cupy.einsum` itself). Splitting the batch into chunks of 65535 also avoids the kernel loop,
but measured on gfx1201 the multiply and sum is 4x to 250x faster than the rocBLAS call for these
shapes. The simulation must drop each operand's size-1 axes before it decides which labels the
operands still waiting keep: CuPy builds a step's kept labels from the squeezed subscripts, so a
label of size 1 in an earlier operand is contracted, not batched. Taking the full subscripts misses
steps such as `'da,deab,dac->'` with the first operand shaped (1, 4), a matrix-vector product over
`a`, and reroutes some steps that are not one.

Test the trigger differentially: record the shapes `cupy.matmul` receives inside `cupy.einsum` (it
looks the attribute up per call) over random patterns with size-1 broadcasting, with the batch
threshold lowered so small shapes reach it, and require the helper to reroute exactly the calls that
issue a matrix-vector step. The recorder sees the pairwise steps only when cuTENSOR is not a CuPy
routine accelerator (CUDA CI often sets `CUPY_ACCELERATORS=cub,cutensor`), so clear the list with
`cupy._core._accelerator.set_routine_accelerators([])` for the test and restore it in a `finally`;
CuPy reads it on every call. Bind the helper as the contraction wrapper's einsum and switch the
direct `cupy.einsum` sites to it; find them statically by simulating CuPy's pairwise order from the
subscripts, then judge `numpy.einsum` calls that may receive CuPy arrays. Do not hunt for sites with
an LD_PRELOAD shim on rocBLAS that takes the GIL in every call: two W7800s wedged (first H2D copy
never completes, 95% GFX activity with no process) during such a run with 8 to 12 workers.
(gpu4pyscf)

**On gfx12, rocBLAS complex GEMM with n > 2^21 leaves columns unwritten.** rocBLAS (5.7.0, TheRock
10.1) has only generic `fallback` Tensile kernels for complex types on gfx12
(`Cijk_..._ZB/CB_MT32x32x8_..._WG16_16_1_WGM8`), launched through `hipExtModuleLaunchKernel` with
the 32-column tiles of n in grid y. gfx12 work-group IDs in y and z are 16 bits and that launch call
wraps instead of failing (platform-limits.md, "gfx12 work-group IDs y and z are 16 bits"), so every
zgemm/cgemm (all four transposes) with n > 65536 * 32 = 2^21 returns columns >= 2^21 unwritten (a
poisoned output keeps its poison), independent of m and k. Real GEMMs run hipBLASLt-generated
kernels with the tiles in grid x and are clean to n = 8M; batched complex GEMM over 65536 batches is
clean (the batch is split on the host). `AMD_LOG_LEVEL=3` prints each `hipExtModuleLaunchKernel`
with its global sizes, which shows the grid. CuPy's einsum reaches it through k-point transforms
such as `einsum('Lk,pLqG->kpqG')`, where n = p*q*G is millions. In a CuPy port fix it in the
project-local einsum of the previous entry: for a complex pairwise step whose row or column count
exceeds 2^21, split the outermost free output axis of the larger side and recurse. Split the
outermost axis, not the largest: splitting the innermost axis made the strided writes into the
result cost 1.5x, the outermost-axis split costs 1.1x to 1.5x at 3.5M to 6.9M columns and is 0.26x
(faster than the unsplit call) at 26M. Gate the split on the device's `gcnArchName` starting with
`gfx12` (cached per device), so other GPUs keep the path they were tested on, and force the gate in
the regression tests so CI on any GPU checks the split. (gpu4pyscf)

## MIOpen

**Some MIOpen convolution solvers are not run-to-run reproducible; establish the noise floor
before comparing two code paths.** A video model that repeated bit-exactly on gfx950 and
gfx1100 gave a different output on every run on gfx942 (about 34 dB PSNR between identical
runs), which made every A/B comparison there ambiguous. MIOpen (PyTorch's `cudnn` backend on
ROCm) chose `ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC` for some bf16 convolutions on gfx942:
a batch-1 3x3 512->256 conv differed on every call, by up to one bf16 ulp, while the same conv
at batch 12, or as 1x1, was exact. To find the layer, register forward hooks on every leaf
module, run the same input twice in one process and report the first module whose output
differs; then repeat that layer alone at the observed shape. `MIOPEN_ENABLE_LOGGING=1
MIOPEN_LOG_LEVEL=6` prints the chosen solver. `torch.backends.cudnn.deterministic = True`
makes MIOpen choose a deterministic solver (`GemmFwdRest` there) at no measurable cost in that
model; set it for A/B validation, and run the reference twice before reading any difference as
real. (Matrix-Game)

**A "find the best algorithm" call WRITES to the tensor named as its output.** MIOpen's
`miopenFindConvolution{Forward,BackwardData,BackwardWeights}Algorithm` benchmark by running
each candidate, so the destination argument is clobbered. cuDNN's fixed-algorithm enums need
no such step, so a port that introduces one has nothing to inherit this from. Give each find
a buffer that is about to be overwritten anyway: the forward find may take the real output,
but the backward finds must NOT be handed the live input and weight tensors merely because
the signature accepts them. Getting it wrong corrupts the model in place and presents as
training diverging to NaN, not as a wrong kernel. Cache the choice per input shape -- a graph
replays the same convolution once per batch. (marian-dev)

**MIOpen honours only a unit alpha and a zero beta for 2D convolution, pooling and bias.**
cuDNN's accumulate form (`beta = 1`, used by every backward pass that adds into an existing
gradient) has to be rebuilt as an operation into a staging buffer followed by
`miopenOpTensor(miopenTensorOpAdd, ...)`. MIOpen's pooling backward also reads the indices
its forward pass recorded, so that workspace must outlive the forward call rather than being
a local. (marian-dev) MIOpen also pools an NHWC tensor by transposing it through that workspace,
in the forward pass and for average pooling too: `miopenPoolingForward` with a null workspace
faults in `batched_transpose_*` at address 0 for channels-last input. Always query
`miopenPoolingGetWorkSpaceSizeV2` and pass the workspace, even for an inference-only forward.
(oneflow)

**MIOpen runs a convolution only on a handle where Find ran for it.** Find registers the
solution's invokers on the handle it ran on; `miopenConvolutionForward` (and the backward calls)
on any other handle fails with "No invoker was registered for convolution forward. Was find
executed?". A cuDNN-shaped caller that searches once on a pooled handle and then runs on each
stream's own handle trips it. Before the first run of a problem on a handle, run a
non-exhaustive Find there into the output buffer the run is about to overwrite. Also:
`miopenGetConvolutionNdDescriptor` rejects a requested dimension count larger than the
descriptor's (cuDNN returns what it has), and MIOpen pooling reads alpha/beta as float and has
no double. Key the "Find ran here" record on the live handle and drop it in the destroy wrapper:
`miopenCreate` hands out a destroyed handle's address again (about once in 60 create/destroy
cycles on an MI250X with ROCm 10.1), and a record keyed on the bare address then skips the Find
on a handle that never ran one. (oneflow)

**MIOpen 3.6 (ROCm 10.1) lacks several cuDNN operations outright.** Measured on gfx90a:
double batch norm finds no solver (and `miopenDeriveBNTensorDescriptor` always derives a FLOAT
parameter descriptor, so a cuDNN-shaped type check on gamma fails first); double convolution has
no solver either (Find fails compiling its im2col fallback); `miopenConvolutionBackwardBias` is
deprecated and returns `miopenStatusNotImplemented` for every type and layout; and
`miopenReduceTensor` ADD fails for bfloat16 (its runtime kernel does not compile) while float,
half and double work. A shim that forwards these calls aborts the first time a user reaches
them. Run double batch norm on small native kernels, and compute a bias gradient as dy times a
ones vector with the project's own GEMM. Accumulate that sum in float: one GEMM per sample with
beta = 1 into a half/bf16 output rounds N times (7-12x the single-rounding error at N = 256);
reduce to float row sums with a mixed-type GEMM (`hipblasGemmEx`, f16/bf16 in, f32 out) and
round once. (oneflow)

**Two ROCm library headers can collide.** `rpp.h` (RpptInterpolationType) and `miopen.h` both
declare an unscoped enumerator `BILINEAR`; a translation unit cannot include both. Keep the RPP
call in its own file behind a small interface. (oneflow)

## hipSPARSE

**A zero workspace size does not mean "skip the operation".** Upstream code shaped like
`if(bufferSize > 0) { alloc; call; free; }` is correct only where the vendor always wants
scratch. cuSPARSE always does for `CUSPARSE_SPMM_CSR_ALG2`, so the shortcut is invisible on
NVIDIA; hipSPARSE wants none at these shapes, reports zero, and the multiplication is never
issued -- the output keeps whatever it held, which reads as wrong numbers rather than as an
error. Make the call unconditional and only the allocation conditional. Suspect the pattern
wherever a `*_bufferSize` query gates its own operation. (marian-dev)

**hipSPARSE has no uniform half-precision SpMM.** Half operands are accepted only in the
mixed-precision mode, which pairs half sparse and dense inputs with a FLOAT result matrix and
float scalars; asking for a half result is `HIPSPARSE_STATUS_NOT_SUPPORTED` (and an all-half
compute type is `INVALID_VALUE`). Compute into a float scratch and cast back. The supported
combinations are tabulated in `rocsparse_spmm.h`, which is faster to read than to probe.
(marian-dev)

**Library calls inside HIP graph capture.** hipSPARSE's generic API breaks capture in three
ways cuSPARSE does not: destroying a CSR `SpMat` descriptor (`hipsparseDestroySpMat`) calls
`hipDeviceSynchronize`, which invalidates the capture (`hipStreamEndCapture` returns 901, in
every capture mode; destroying a `DnVec` does not, ROCm 10.1 on gfx950), the default CSR SpMV
algorithm analyses on first use
(`HIPSPARSE_STATUS_INTERNAL_ERROR` while capturing), and a non-null external buffer whose
queried size is 0 returns `INVALID_VALUE` once descriptors are reused. Create descriptors
before capture and cache them per operand set, use `HIPSPARSE_SPMV_CSR_ALG2`, and pass a null
buffer when the size is 0. `AMD_LOG_LEVEL=3` shows the call that broke the capture. (TornadoVM)

**A device pointer passed while the handle is in host pointer mode crashes only on some
hosts.** An output argument that CUDA code passes as device memory under host pointer mode
(hipSPARSE `csrgeam2Nnz`'s `nnzTotalDevHostPtr`, which cuSPARSE tolerates) is dereferenced on
the CPU by rocSPARSE. It works where the runtime can CPU-map `hipMalloc` memory, as on a
Linux discrete card with a large (resizable) BAR, and segfaults in host code inside the
library elsewhere, for example with a 256 MiB BAR; Windows machines, an integrated APU
included, crashed in the same call (the cause there was not measured). It looks like an OS-
or arch-specific library bug. Check the BAR size (`/sys/bus/pci/devices/<bdf>/resource`) and
repeat the call with a host pointer or with `hipsparseSetPointerMode(DEVICE)`. The fix is a
host variable, or device pointer mode with device scalars. (arrayfire)

## rocThrust, hipCUB and rocPRIM

**rocThrust's `thrust/system/cuda/error.h` is CUDA-only.** It exists (so the include
resolves) but its `errc_t` uses `cudaErrorXxx` enums. Shim it to `thrust/system/hip/error.h`
plus `cuda_category() -> hip_category()` inside `THRUST_NAMESPACE_BEGIN`. Likewise never
reopen `namespace thrust { namespace hip {} }` to alias `thrust::cuda`: current rocThrust
declares its systems in a versioned inline namespace, so the empty one makes every
`thrust::hip` ambiguous (AMD's xgboost 2.1.3 ROCm branch hit this). (h2o4gpu)

**Prefer the specific hipCUB algorithm header over the `<hipcub/hipcub.hpp>` umbrella.** The
umbrella pulls in `device/device_for.hpp` -> `util_mdspan.hpp`, which needs C++23
`std::extents` (`<mdspan>`, libstdc++ 15 or newer). With libstdc++ 13 that is about 20 errors
("no template named 'extents' in namespace 'std'") in a TU that only wanted a radix sort.
`#include <hipcub/device/device_radix_sort.hpp>` compiles cleanly. CUB's umbrella header does
not have this problem, so the umbrella include is what the CUDA source will have. A
`namespace cub = hipcub;` alias next to the guarded include keeps every call site in the CUB
spelling on both back ends. (CuRast: `src/CubSort.cu`, ROCm 7.2.3 with GCC 13.)

**rocPRIM never asks for less than 4 bytes of temp storage.** Every rocPRIM-backed hipCUB
device algorithm (`DeviceReduce`, `DeviceSegmentedReduce`, ...) reports
`max(needed, 4)` from its size query, even when it needs nothing; CUB reports 1 there.
Upstream code that sizes the scratch from the INPUT (`elem_cnt * sizeof(T)`) and then checks
it against the query passes on NVIDIA and fails for a 2- or 3-element reduce of a one-byte
type on AMD. Size the scratch as `max(own estimate, library query)`. (oneflow:
`flow.tensor([True, False], device="cuda").all()` aborted.)

**hipCUB's segmented `ArgMax` returns an out-of-range key for a segment of NaNs.** rocPRIM
folds a seed `{INT_MAX, lowest}` into every segment and keeps it when no element compares
greater, so an all-NaN row yields `INT_MAX - segment_begin`; CUB returns an index inside the
row. Anything that then indexes with the result (one-hot scatter, gather) traps or faults.
Clamp keys outside `[0, segment_size)` after the call. (oneflow: `gumbel_softmax(hard=True)`
in half precision, where `rand` rounds to exactly 1.0 often enough to make a NaN column.)

**`<complex.h>` defines the macro `I`, which rocPRIM uses as a template parameter name.** A
translation unit that includes `<complex.h>` before a hipCUB shim fails deep inside
`rocprim/types/tuple.hpp` with `expected ')'` and `_Complex float` conversion errors. If the
translation unit does not use `I`, `#undef I` in the shim before including hipCUB. (gpu4pyscf)

**Three rocPRIM radix sort defects give a wrong order or a broken permutation where CUB is
correct.** They are the next three entries: float keys that mix signaling and quiet NaNs, a bit
range on 64-bit keys that ends at bit 64 and starts above bit 0, and the choice of workaround for
that bit range. All three are silent, so when a port's results go wrong after a `DeviceRadixSort`
(or `torch.sort` on ROCm, which inherits the first), check the sort's output directly before looking
further.

**rocPRIM radix sort corrupts float keys that mix signaling and quiet NaNs.** hipCUB/rocPRIM
`DeviceRadixSort::SortPairs` on float keys that contain both signaling and quiet NaNs returns a
broken permutation: values duplicated and missing, and output slots never written. CUDA's cub sorts
the bit patterns and is unaffected. Either NaN kind alone sorts correctly. Uninitialized device
memory (a `torch::empty` buffer only partly written by a kernel) is where such keys come from, and
`torch.sort` on ROCm inherits the defect. The symptom is usually far from the sort: a later gather
on the sorted indices trips torch's device-side index assert (`HSA_STATUS_ERROR_EXCEPTION 0x1016`
naming `index_elementwise_kernel`) only when the stale slot happens to hold an out-of-range value,
so the failure looks intermittent. Diagnose by checking the sort output right after the sort for
duplicates (`bincount(idx).max() > 1`), not only for range. Fix on the caller's side: never sort
uninitialized data (zero-fill the buffer or sort only the valid prefix). Observed on the ROCm 10.1
pip SDK (rocPRIM 4.6.0, hipCUB 4.7.0) on gfx90a, gfx942 and gfx1201, and unchanged on
10.3.0a20261119 (rocPRIM 5.1.0); CUDA 12.9 sorts the same keys correctly. It affects float and
double keys, pairs and keys-only, for 1025 to 1,048,576 elements: in that range rocPRIM merges
radix-sorted 1024-item tiles with a comparator that adds zero to each key to fold -0.0, which
quiets signaling NaNs, so the merge order disagrees with the tile order. Sizes outside the range
take other paths and pass. Reported as ROCm/rocm-libraries#16364. Exact duplicate and missing
counts vary run to run. (LiteGS)

**rocPRIM radix sort of 64-bit keys is wrong when `end_bit` is 64 and `begin_bit` is not 0.** With
rocPRIM 4.8 (ROCm 10.2), `hipcub::DeviceRadixSort::SortKeys` on 64-bit keys returns keys not ordered
by the selected bits when `end_bit == 64` and `begin_bit > 0`, for more than 1024 keys; `[0,64)` and
any range ending below bit 64 are correct. Code that sorts packed `{payload, key}` words by their
high half (`begin_bit = 32, end_bit = 64`, common in BVH builders) silently mis-groups. Workaround
when the key is the high half: sort all 64 bits. Check this in any port that passes a sub-range
ending at bit 64. (warp-1.14)

**Choosing the workaround for that bit-range defect: narrowing the range is only safe when the top
key bit is 0 everywhere.** For the defect of the previous entry (64-bit keys, `end_bit` 64,
`begin_bit` > 0; cuBQL records it at 1025 to 2^20 keys) two workarounds avoid the broken path: sort
the full width (`begin_bit = 0`), or stop one bit short (`end_bit = width - 1`). The second is exact
and stable but only when the top key bit is 0 for every element, and packed keys often break that: a
sentinel such as `(uint32_t)-1` in the upper half ("invalid, sort last") sets it, and under the
narrowed range the sentinel then ties with the largest valid value. A unit test at ordinary sizes
passes either way, because the colliding valid value is near 2^31. Read where every packed field is
written, sentinels included, before narrowing; otherwise sort the full width, which is correct
whenever no caller depends on the order of keys equal in the selected bits (a 5-12% cost at up to
2^20 64-bit keys on MI355X). (cuBQL)

**libcu++ -> libhipcxx, and two atomic defects in it.** `ROCm/libhipcxx` keeps the
`cuda::`/`cuda::std::` spelling, so CCCL-heavy code (`cuda::atomic_ref`, `cuda::stream_ref`,
`cuda::std::span`) ports without renames. ROCm 10.1 and later bundle CCCL 3.0.2 as
`/opt/rocm/core-<ver>/include/hipccl`; newer CCCL levels come from the `amd-develop` branch.
Two defects in `cuda/std/__atomic/functions/atomic_hip_generated.h`, present in both
(`amd-develop` at `832dc407f`, gfx942):
- The block- and device-scope `__atomic_compare_exchange_cuda` ignores its weak flag and always
  calls `__hip_atomic_compare_exchange_weak`. A 1- or 2-byte CAS is a 32-bit CAS on the
  containing word, so the weak form also fails when a NEIGHBORING byte changes. With each
  thread flipping only its own byte, `compare_exchange_strong` failed 75% (1 byte) and 50%
  (2 bytes) of the time, while the raw strong builtin failed 0 times. Code that reads a failed CAS
  on an empty slot as "taken" loses or duplicates keys (cuCollections: 11 failing
  `static_set<int8_t/int16_t>` cases).
- `atomic_ref<T>` on an 8-byte type with 4-byte `alignof` (`struct {int a, b;}`, aligned to 8
  by the caller as `atomic_ref` requires) fails to build with `unsupported cmpxchg: instruction
  alignment 4 is smaller than the required 8-byte alignment`. NVIDIA's libcu++ compiles it.
Both fixes are small: honor the weak flag, and view the object through a `sizeof(T)`-aligned
typedef. The header is generated, so an upstream fix belongs in its generator,
`codegen/generators/hip_intrinsics.h`; a port can patch the installed header. Carry the patch in
the port and tell the user. Validate 1-, 2- and 8-byte
atomic paths explicitly, because 4-byte tests pass either way. (cuCollections)

## CuPy on ROCm

libhipcxx's headers, unlike libcu++'s under g++, do not compile with a plain host compiler
(`'__host__' does not name a type`). It is found with `find_package(libhipcxx)` (target
`libhipcxx::libhipcxx`, headers under `include/hipccl`, with the `cuda/std/...` spellings). A
host `.cpp` that includes a header exposing `cuda::std` types must therefore be compiled as HIP
too, or those types' ABI differs between translation units. A `namespace cuda::std = ::std`
shim does not substitute in device code: libstdc++'s `std::optional::value()` calls a host-only
throw helper that hip-clang rejects. Include `<cuda/std/cmath>` and `<cuda/std/functional>`
explicitly where the sources relied on CCCL including them transitively (`cuda::std::abs(float)`
is otherwise ambiguous). (autoware_universe)

**torchaudio's RNN-T loss and CTC forced alignment run only on the CPU in ROCm builds.**
`NotImplementedError: Could not run 'torchaudio::rnnt_loss' with arguments from the 'CUDA'
backend` (likewise `torchaudio::forced_align`) as soon as the inputs are on an AMD GPU:
torchaudio compiles those kernels only for CUDA, and `libtorchaudio.so` in the ROCm wheels has
no HIP code (checked 2.8.0+rocm6.4, 2.8.0a0+rocm7.12.0, 2.11.0+rocm7.2). Move the inputs to the
CPU for these two calls, or use k2's RNN-T losses, which have a HIP build. Check before porting
a speech project whether its default loss is torchaudio's. (wenet)

**CuPy-ROCm gaps a CuPy-based port meets at run time.** `cupy.cuda.runtime.deviceSetLimit` is a stub
that always raises hipErrorUnknown (`cupy_backends/hip/cupy_hip_runtime.h`); call
`hipDeviceSetLimit` through ctypes from `libamdhip64.so.<major>`, the library CuPy already loaded
(`runtimeGetVersion() // 10000000`). RawKernel sources compile with hiprtc, which has no
`cuComplex.h` (use `hip/hip_complex.h` and alias `cuDoubleComplex`, `make_cuDoubleComplex`,
`cuCreal`, `cuCimag`) and rejects a 32-bit mask in `__shfl_down_sync`; prepend those definitions on
ROCm only, leaving the CUDA source untouched. CuPy-ROCm ships no cuTENSOR binding (fall back to
`cupy.einsum`), has no `sharedMemPerBlockOptin` device attribute, and CUTLASS-based extension
kernels need a replacement such as one GEMM per group. Seeded `cupy.random` streams differ from
CUDA's (diagnosis.md, "Seeded `cupy.random` inputs differ between CUDA and ROCm"), and two rocBLAS
defects reached through `cupy.einsum` are under "BLAS, hipBLASLt and Composable Kernel" above.
(gpu4pyscf)

## Random numbers, profiling and monitoring

**Code that reads cuRAND's device generator state internals needs a rocRAND translation, not
an alias.** cuRAND caches one spare Box-Muller normal and flags it with
`state.boxmuller_flag == EXTRA_FLAG_NORMAL`. rocRAND caches the same spare value but marks an
EMPTY slot with a NaN sentinel in `boxmuller_float`, so the equivalent predicate is
`rocrand_device::detail::engine_boxmuller_helper<Engine>::has_float(&engine)` rather than a
flag compare. `skipahead` and the `*_normal2` generators exist natively. A kernel that skips
ahead deterministically depends on this predicate being right, so a wrong translation is a
silent numeric divergence, not a build error. (CV-CUDA: gaussian noise.)

**NVTX -> roctx.** Profiling range calls (`nvtxRangePushA`/`nvtxRangePop`) map onto roctx,
which the ROCm profilers report the same way, so a project whose build FATALS without NVTX
headers does not need its ranges stubbed out. The header moved between ROCm versions: prefer
`<rocprofiler-sdk-roctx/roctx.h>` and fall back to `<roctracer/roctx.h>` behind
`__has_include`; both declare `roctxRangePushA`/`roctxRangePop`. Link `rocprofiler-sdk-roctx`
or `roctx64` to match. A forwarding `nvtx3/nvToolsExt.h` on the HIP include path keeps every
call site untouched. Only the ranges map: NVTX's injection API (`NvtxGetExportTableFunc_t`,
`NVTX_ETID_CALLBACKS`, `NVTX_CBID_CORE_*`), which interposer and probe test helpers use, has
no roctx analogue, so check whether the project builds such a helper before calling the shim
complete. (CV-CUDA)

**GPU power (the NVML replacement) is AMD SMI.** `amdsmi_init(AMDSMI_INIT_AMD_GPUS)`, find
the GPU by PCI address (`hipDeviceAttributePciDomainId`/`PciBusId`/`PciDeviceId` into the
64-bit `amdsmi_bdf_t`; a HIP device index counts only the `HIP_VISIBLE_DEVICES` GPUs), then
`amdsmi_get_power_info`: watts, `current_socket_power` on MI300 and later, unsupported
fields read `UINT32_MAX`. (TornadoVM)

## RAPIDS (ROCm-DS)

**RAPIDS libraries: hipDF, hipVS and hipMM as C++ wheels.** AMD's package index
(`https://pypi.amd.com/rocm-7.2.3/simple`) carries `amd-libhipdf` (libcudf headers, libcudf,
hipcomp, CMake packages), `amd-libhipmm` (librmm; it requires `amd-rocmds-logger`, which
holds rapids_logger), `amd-libhipvs` (libcuvs, libcuvs_c) and `amd-libraft`, laid out like the
RAPIDS wheels; `amd-libhipdf` and `amd-libhipvs` declare no dependencies, so add
`amd-libhipmm` yourself (pip then brings `amd-rocmds-logger`; unpacking wheels by hand, take
all three). hipDF keeps cuDF's API and ships CUDA-named
compat headers (`cudf/cuda_runtime.h`), so a cuDF client compiles with ROCm clang, C++20 and
`find_package(cudf)` (CMake >= 3.30.4; put each wheel's `lib64/cmake` on the prefix path, and
the wheels before ROCm's, whose hipccl lacks `CCCL::CCCL`). Differences from current RAPIDS:
hipMM 4.0 keeps memory resources under `rmm/mr/device/` and `set_current_device_resource`
takes a pointer; hipDF 3.0's Parquet writer has no Zstandard compressor ("nvCOMP 2.4 or
newer is required") and `LIBCUDF_HOST_COMPRESSION` does not route around it; hipVS takes
DLPack device tensors as `kDLCUDA` and refuses `kDLROCM`, builds a host dataset's
all-neighbors graph into device outputs only, and needs `libomp.so` from ROCm's
`lib/llvm/lib`. The libraries link `libamdhip64.so.7` and resolved against a ROCm 10.1 SDK.
(TornadoVM)

### RMM -> hipMM (and check ROCm-DS before stubbing any RAPIDS library)

RMM (RAPIDS Memory Manager) does not build on HIP. Do not vendor a stub of the types a project uses: hipMM (https://github.com/AMD-Ecosystem/hipMM, the ROCm-DS port of RMM) keeps RMM's `rmm` CMake target, `rmm::` namespace and header layout, so it replaces RMM without source changes. HEonGPU first shipped a 600-line stub and then deleted it. Check the ROCm-DS and AMD-Ecosystem organizations for a port of any RAPIDS library before writing a replacement. For prebuilt wheels instead of a source build, see "RAPIDS libraries: hipDF, hipVS and hipMM as C++ wheels" above.

Integration facts from fetching hipMM with FetchContent (HEonGPU, hipMM 5.0.0, ROCm 10.1):

- Match hipMM's version to the RMM API the project uses, not to RMM's version number. hipMM 4.0 has the RMM 25.x resource API (templated `pool_memory_resource<Upstream>`, headers under `rmm/mr/device/`); hipMM 5.0 is derived from RMM 26.06 (non-template resources, headers under `rmm/mr/`). If upstream is ahead of every hipMM release, wait for or ask for a newer hipMM before writing HIP-only branches around the difference.
- The HIP arm of the FetchContent block that fetches rmm on CUDA fetches hipMM. Pin it by commit, and name the right subdirectory: a `SOURCE_SUBDIR cpp` against a line that predates the `cpp/` layout silently skips `add_subdirectory` and fails only at link time as `-lrmm`.
- hipMM has its own version numbers. Where the CUDA build asks for a minimum `rmm` version (a system-package option, or `find_dependency(rmm <version>)` in an installed package config), the HIP branch needs hipMM's number, or an older hipMM with a different header layout is accepted and fails in the consumer's compile.
- hipMM's own `.cpp` sources need a HIP-capable compiler, because its hip dependency adds `-x hip --offload-arch` to their flags. Mark them `LANGUAGE HIP` after the fetch. `get_target_property(... SOURCES)` returns paths relative to the target's source directory, so prepend that directory first or the property binds to nonexistent paths and silently does nothing.
- If the project already called `find_package(rocthrust)` in a parent directory, hipMM's scripts fail with "Attempt to promote imported target ... to global scope". Use `find_package(rocthrust ... GLOBAL)` in the parent.
- A dependency that hipMM's scripts download from a repository that does not resolve can be overridden without patching hipMM: FetchContent and CPM use the first `FetchContent_Declare` of a name, so declare it earlier with a working repository and a pinned commit. Check which copy was used with `git remote get-url origin` in the dependency's fetched source directory in the build tree. (HEonGPU did this for libhipcxx.)
- hipMM 5.0 calls `hipMemGetDefaultMemPool`, so it needs a ROCm whose HIP declares it (present in HIP 7.16 with ROCm 10.1, absent in HIP 7.14).
- hipMM is Linux-only, as its README says. hipMM 5.0.0 does not compile on Windows (https://github.com/AMD-Ecosystem/hipMM/issues/11), so a project that depends on it has no Windows ROCm build until that changes.
- On the hipMM 4.0 line, re-configuring a build tree after `cmake --install` into a prefix on the search path failed while promoting `roc::rocthrust` to global scope; use a fresh tree. Not rechecked on 5.0.

## RCCL

### RCCL collectives need the whole multi-GPU host at once

NCCL->RCCL is a mechanical swap (RCCL is API-compatible and ships in ROCm), but
validating it is unlike validating a kernel: collectives are a multi-GPU feature, so
the test process must hold two or more same-architecture GPUs SIMULTANEOUSLY. Two
consequences for scheduling on a multi-GPU host:

- Parallelising work across a host's GPUs -- one build or test run pinned per
  device with `HIP_VISIBLE_DEVICES` -- is suspended for the duration of a
  collectives run. The run owns every device it uses; another job landing on a
  sibling GPU mid-run contends with it, and both results are suspect.
- The one-device-per-process pin that diagnosis.md prescribes for mixed-architecture hosts must be
  LIFTED for the collectives process: give it exactly the same-architecture devices
  it should use (read the device list at run time, never a remembered index), and
  keep mixed-architecture pairs out of one process regardless.

A single-GPU host cannot exercise the collectives path at all; record that as the
arch's limitation rather than letting a collectives-off run stand in for it
silently. Read the device list before concluding that, though. What the run has to show is the library announcing
itself and the communicators constructing, plus a training job spanning every
device converging and its model decoding correctly -- a job that starts and does
not crash proves only that the collectives were never called. (marian-dev, on
four gfx1100)

## Image and video codecs

**rocJPEG writes past the row pitch.** Its output conversion writes each row out to a
multiple of 8 pixels regardless of `RocJpegImage::pitch`, so with a tight pitch the first
pixels of every following row are overwritten -- nondeterministically, up to full-scale
errors. Decode into staging with rows padded to 16 pixels and copy the visible region out
(`hipMemcpy2DAsync`), which also orders rocJPEG's own stream before the caller's.
(LichtFeld-Studio)

**Fall back to the CPU only for JPEGs rocJPEG cannot decode, and say so.** rocJPEG 1.10 on
VCN decodes only 8-bit baseline (SOF0) JPEG, 1 or 3 components, 4:4:4/4:4:0/4:2:2/4:2:0,
at least 64x64. Extended sequential (SOF1, which nvJPEG decodes), progressive,
arithmetic-coded, CMYK and 4:1:1 streams fail `rocJpegStreamParse` with `BAD_JPEG` -- the
same status a corrupt file gets -- and small images fail `rocJpegDecode` with
`JPEG_NOT_SUPPORTED`. So classify from the JPEG frame header
(SOFn marker, precision, component count, sampling factors) yourself, decode only those cases
on the CPU with a warning naming the reason, and let every other failure (OOM, decoder
creation, corrupt data) surface as an error. A catch-all fallback hides real failures behind
a silently slower path. (LichtFeld-Studio review)

**A JPEG that ends before its scan hangs the VCN JPEG engine.** rocJPEG 1.10 parses such a
stream as SUCCESS (dimensions and all) and submits it; the engine hangs until the driver
resets the `jpeg_dec` ring, and the process's GPU context is lost ("context is guilty of a
hard recovery", then abort). One corrupt file in a dataset kills the app, where nvJPEG
returns an error. Only the missing scan was measured to hang: a JPEG cut inside its scan
(no end-of-image marker) decodes, with no reset, and one with no scan hangs. A scan header
with no data was only parse-checked (SUCCESS). Walk the markers on the host before calling
rocJPEG and require at least a scan with data; the port also requires a frame header,
defined quantization tables, a well-formed scan header and an end-of-image marker, as a
precaution against corrupt streams, and treats any failure as a decode error. Such a
stream may appear only in a test of that host-side check, which is then the one barrier
in front of the hardware: a regression in the check costs a ring reset, so run that test
on a GPU no other job is using. (LichtFeld-Studio)

**NVDEC through FFmpeg becomes FFmpeg demux plus rocDecode.** FFmpeg has no rocDecode
hwaccel, and the CUDA device context behind `*_cuvid` fails on AMD, so the code silently
falls back to software decode. TheRock's SDK ships rocDecode's library, headers and CMake
package but not the samples' `VideoDemuxer`/`RocVideoDecoder` helpers, so keep FFmpeg for
demux and drive `rocDecCreateVideoParser`/`rocDecParseVideoData` yourself behind an
`avcodec_send_packet`/`receive_frame`/`flush_buffers`-shaped class: H.264/HEVC packets
through `h264_mp4toannexb`/`hevc_mp4toannexb`, the AV1 `av1C` config OBUs (extradata after
4 bytes) prepended to the first packet after open or flush, a flush that destroys and
recreates the parser (keep the decoder when the next sequence header matches), and frames
of `AV_PKT_FLAG_DISCARD` packets (outside an MP4 edit list) dropped by pts as libavcodec
does, or the frame count is one higher than the software decoder's. Copy each displayed
picture (`rocDecGetVideoFrame`, offset by the display area) into an `AVBufferPool` of
`hipMalloc`'d NV12/P010 buffers so frames stay on the GPU. Measured on gfx1100 against
libavcodec/libdav1d: H.264 High, HEVC Main/Main 10, VP9 profiles 0/2 and AV1 8/10-bit
bit-exact, timestamps preserved. (LichtFeld-Studio)

**rocDecode's capability query is not a support check by itself.** `rocDecGetDecoderCaps`
takes codec, chroma format and bit depth but no profile, and rocDecode 1.10 decodes an
H.264 High 4:4:4 Predictive stream with 4:2:0 chroma (x264 lossless) with
`rocDecodeStatus_Success` and wrong pixels (its AVC picture parameters have no field for
the SPS lossless flag `qpprime_y_zero_transform_bypass_flag`, so the engine is never told).
Gate the hardware path on the stream's profile too (H.264 Baseline/Main/High; HEVC Main/Main
10/Main Still) and on the caps size limits, and decode the rest on the CPU with a logged
reason. The caps limits are coded sizes: before the minimum check, round the display size
up to the codec's block size (16 for H.264, 8, the smallest coding block, for HEVC; VP9 and
AV1 code the frame size itself). Comparing the display size refuses valid streams: a 60x60
H.264 or HEVC stream is coded 64x64, inside VCN 4's 64x64 minimum, and decodes bit-exact.
The maxima (4096, 8192, 4352) are multiples of 16, so the display size gives the same answer
there. Also do not trust the parser's `video_signal_description` for VP9 and AV1:
it reports limited range and matrix 0 for full-range streams; take those from FFmpeg's
stream parameters (H.264/HEVC VUI values are right). Validate a port's decode against the
software decoder frame by frame, not by eye. (LichtFeld-Studio)

**A rocDecode success status does not mean right pixels: check a reference clip once per
process before trusting a codec.** On gfx90a (MI250X, amdgpu-dkms 6.16.6), rocDecode 1.10
from TheRock SDK 10.1 loads the SDK's bundled Mesa VA-API driver
(`rocm_sysdeps/lib/radeonsi_drv_video.so`), and through it every H.264, HEVC and VP9 frame,
8- and 10-bit, is wrong in nearly every byte while `rocDecGetDecodeStatus` says success;
Ubuntu's Mesa driver on the same card (`LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri`)
and the bundled driver on gfx1100 are bit-exact. Caps, profile and size gates cannot see
this, so the port decodes an embedded two-frame 64x64 clip per codec and bit depth the first
time a process asks about it, through the same demux, bitstream filter and copy code as real
files, and compares each frame's MD5 with the digest of FFmpeg's software decode
(`ffmpeg -i clip -pix_fmt nv12|p010le -f framemd5 -`, which hashes the frame packed with no
row padding, the layout `av_image_copy_to_buffer(..., 1)` gives a host copy). A mismatch
refuses that codec for the process with a named CPU reason. Key the result by device,
codec and bit depth under a mutex, run it inside the capability query (lazily, not at
start-up), and do not cache a decode error (that is an error, not a verdict). Cost on
MI250X: 40-46 ms for the first codec (includes loading the VA driver), 7-18 ms per further
codec. Even the intra frame was wrong, and 10-bit failed with 8-bit, but check each codec
and depth: one codec's result is not evidence for another. Test it without a broken driver
by injecting a wrong digest, and add a test that the verdict agrees with a full decode of
the fixtures, so the same suite proves the check on a good driver and a bad one. Do not
switch the VA driver from inside the application. (LichtFeld-Studio)

When frames are wrong on one architecture, split the cause with three checks before blaming
the port, each on valid streams only. `strace -f -e trace=openat` shows which render node and
which `*_drv_video.so` the process opens (a wrong card is the other suspect). Rerun the same
test binary with `LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri` (the distribution's Mesa;
the bundled libva loads it). And decode the clip with the system `ffmpeg -hwaccel vaapi
-hwaccel_device /dev/dri/renderD<n> -hwaccel_output_format vaapi -vf hwdownload,format=nv12`
against a software decode, which proves the video engine itself. When comparing against
software decode, compare luma only or keep the source's pixel format: `-pix_fmt nv12` on a
full-range stream rescales the range and makes a correct decode look wrong.

**The software fallback for AV1 needs libdav1d.** FFmpeg's built-in `av1` decoder only
drives hardware accelerators (without one it fails with "Get current frame error"), and
vcpkg's `ffmpeg` port builds no software AV1 decoder unless the `dav1d` feature is on. A
CUDA project that relies on NVDEC for AV1 often never enables it, so a ROCm port's CPU
path has nothing to decode AV1 with on GPUs without AV1 decode hardware (gfx90a), or on a
build without rocDecode. Add `dav1d` for the ROCm build only, as a manifest feature (a
`rocm` entry under `features` in `vcpkg.json` that lists `ffmpeg` with `dav1d`) appended
to `VCPKG_MANIFEST_FEATURES` under `if(USE_HIP)` before `project()`, the way projects
already select a `tests` feature; a CUDA configure never selects it, so its dependency set
is unchanged. Do not generate a modified copy of the manifest in the build tree instead: it
is a script rewriting a dependency choice (overlay paths, registries, user overrides and
pkg-config caches all need handling) where the manifest has a declarative mechanism. The
vcpkg toolchain caches `VCPKG_MANIFEST_DIR` with FORCE, so a build tree that ever pointed it
elsewhere keeps doing so until reconfigured with `-UVCPKG_MANIFEST_DIR`. (LichtFeld-Studio)
