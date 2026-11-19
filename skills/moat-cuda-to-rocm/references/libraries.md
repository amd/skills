# Libraries: substitutions and gaps

Which ROCm library replaces which CUDA library, and where the replacement is not one-to-one.

<!-- toc:begin -->
Contents:

- Substitution table
- BLAS, hipBLASLt and Composable Kernel
- MIOpen
- hipSPARSE
- rocThrust, hipCUB and rocPRIM
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
| cuFFT | hipFFT (rocFFT) | yes | plan creation during graph capture (faults-memory-streams.md) |
| cuRAND | hipRAND (rocRAND) | yes | device generator state internals; C++14 from ROCm 10.1 |
| cuSPARSE | hipSPARSE (rocSPARSE) | yes | half-precision SpMM, zero workspace sizes, graph capture ("hipSPARSE" below) |
| cuSOLVER | hipSOLVER (rocSOLVER) | partly; not verified | hipSOLVER has a cuSOLVER-style dense API (`hipsolverDn*`). None of the ports behind this file exercised more than handle creation, so check every routine you call against hipSOLVER's function list, and its workspace-size and `info` conventions, before assuming a rename is enough. One gap is known: the low-level sparse factorization interface (`cusolverSpXcsrqr*` and `csrchol*`, with analysis, setup, factor and solve as separate calls), `csrlsvluHost` and `csrsymrcm` have no hipSOLVER counterpart (requested in ROCm/hipSOLVER issue 443), and cuDSS has none either; a sparse direct solve needs another library (RXMesh) |
| cuDNN | MIOpen | NO | a different API (descriptors, find-then-run instead of fixed algorithm enums): the call layer is rewritten, and whole operations are missing ("MIOpen" below) |
| Thrust, CUB | rocThrust, hipCUB (rocPRIM) | yes | `thrust/system/cuda/error.h`, temp-storage sizes, segmented `ArgMax` ("rocThrust, hipCUB and rocPRIM" below) |
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

**hipBLASLt epilogues are not cuBLASLt's, per type and per format.** On gfx90a with hipBLASLt
1.4, DEFAULT/BIAS/RELU/RELU_BIAS exist for f32/f16/bf16 but double has only DEFAULT, and the
AUX/DRELU/BGRAD epilogues exist only for f16. cuBLASLt's ReLU auxiliary output is a 1-bit mask
(leading dimension a multiple of 128) that callers size and may write themselves; hipBLASLt's is
the pre-activation matrix in the output type, so mapping the enum alone writes far past a buffer
sized for a mask. Probe availability with `hipblasLtMatmulAlgoGetHeuristic` per type and fall
back to a plain GEMM plus a small kernel. hipBLASLt cannot read a layout's TYPE back
(`hipblasLtMatrixLayoutGetAttribute` returns invalid attribute) and fixes the compute type at
descriptor creation, so a shim has to remember both. (oneflow)

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

## MIOpen

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

## rocThrust, hipCUB and rocPRIM

**rocThrust's `thrust/system/cuda/error.h` is CUDA-only.** It exists (so the include
resolves) but its `errc_t` uses `cudaErrorXxx` enums. Shim it to `thrust/system/hip/error.h`
plus `cuda_category() -> hip_category()` inside `THRUST_NAMESPACE_BEGIN`. Likewise never
reopen `namespace thrust { namespace hip {} }` to alias `thrust::cuda`: current rocThrust
declares its systems in a versioned inline namespace, so the empty one makes every
`thrust::hip` ambiguous (AMD's xgboost 2.1.3 ROCm branch hit this). (h2o4gpu)

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
