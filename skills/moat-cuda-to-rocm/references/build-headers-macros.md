# Build: headers, macros and source compatibility

Compile errors, and code that compiles but is silently skipped, in the port's own sources and compat header.

<!-- toc:begin -->
Contents:

- The compat header
- Macros CUDA defines and HIP does not
- Syntax and spellings clang rejects
- helper_math.h and vector types
- Keeping the CUDA path intact
- Language standard
<!-- toc:end -->

## The compat header

**A shared compat header must be host-includable.** Host `.cpp` TUs reach the shim through
ordinary headers, so an unconditional `#include <cub/...>` or `<hipcub/...>` leaks device
headers into g++ and fails there. Put the shim in the lowest common layer, keep host-safe
includes (`hip_runtime.h`, `hipfft.h`) unconditional, and gate device-only ones behind
`__CUDACC__` or `__HIPCC__ || __HIP_DEVICE_COMPILE__`. (SCAMP, stdgpu)

The same rule catches HELPER FUNCTIONS added to a shim, not just includes. A `__device__`
helper written at file scope in the compat header is parsed by g++ for every plain `.cpp`
in the project, where `__device__` expands to nothing and `__ballot`/`__lane_id` are
undeclared, so the header stops being host-includable the moment a macro is replaced with a
function. Wrap such a helper in `#if defined(__HIPCC__)` and leave the project macro that
calls it unconditional: the macro only expands inside device code, so the host TUs never
name the helper. (GPU_IPC)

**Do not name compat macros after the warp functions ROCm declares.** ROCm declares real
overloaded functions named `__shfl_sync`, `__ballot_sync`, `__shfl_down_sync` and the rest
(`hip/amd_detail/amd_warp_sync_functions.h`), and `hip/amd_detail/amd_hip_bf16.h` declares
its own bf16 overloads of them. A `#define __shfl_sync(mask, var, srcLane) ...` of the same
name mis-parses those declarations in every translation unit that includes them, and Thrust
pulls in `hip_bf16.h`, so the breakage lands far from the header you wrote and reads as a
mysterious parse error inside a ROCm header. Put the translation in real (template)
functions under project-owned names (`WARP_SHFL_DOWN`-style wrappers) that the call sites go
through; the same-name macros then have no reason to exist. The collision arrived only as
ROCm's headers grew those overloads, so a project whose macro layer built on an older ROCm
can start failing on a newer one with no change of its own. What the functions must compute
is in faults-wavefront.md ("A compat layer for CUDA's `*_sync` warp functions must pin every
member to 32 lanes"). (GPU_IPC)

**Missing-include errors are usually pre-existing upstream omissions**, not port breakage:
CUDA's headers supplied them transitively and the narrower HIP include graph unmasks them.
Fix additively -- it helps the CUDA build too. (LC-framework)

**cuRAND's device header is host-includable only AFTER the CUDA runtime headers**, which
makes a shim the one place it must never go. The fault is ORDER, not host versus device:
CUDA 12.8's `curand_mtgp32_kernel.h` declares `threadIdx`/`blockDim` with C++ linkage and
`device_launch_parameters.h` declares the same names with C linkage, and g++ 13 rejects the
pair only when curand was seen first. Reproducer in both directions, `g++ -fsyntax-only`:
`<curand_kernel.h>` before `<cuda_runtime.h>` + `<device_launch_parameters.h>` gives two
conflicting-declaration errors; the reverse order is clean. A shim tends to be reached very
early -- the lowest common layer is where you put it -- so a curand include there lands in
the failing order for any TU that gets to the shim before some `.cuh` has dragged in the
launch parameters, and the shim's own `#include <cuda_runtime.h>` does not rescue it because
cuda_runtime.h includes device_launch_parameters.h only under `#if defined(__CUDACC__)`. It
is not universal and you do not need it to be: one exposed TU is enough. Establish it by
preprocessing rather than by reasoning about the include graph -- `g++ -E <the target's own
flags> foo.cpp | grep -n 'device_launch_parameters.h"\|<shim>.h"'` gives the two line numbers
and settles the order per TU. (In HEonGPU exactly one of the library's four host files was
exposed.) Give a shim the runtime header only and leave the RNG include in the kernel headers
that use it: those are reached later, after the launch parameters, which is why they are
safe. If a host TU ever reaches the RNG header first, put `<device_launch_parameters.h>`
ahead of it there rather than moving the include back into the shim. Nothing on the AMD
side can catch any of this, since the HIP branch of the shim includes hipRAND instead and hipRAND is
host-safe. (HEonGPU)

**Before gating or moving a device-library include that a compat header already has, check
which HOST headers name its types.** Where a project's compat header does carry the RNG
include, it is tempting to shrink what host files see (a cuRAND/rocRAND kernel header is
about 25k preprocessed lines) by wrapping the include in
`#if defined(__HIPCC__) || defined(__CUDACC__)`. Check HEADERS, not just `.cc`/`.cpp`: a
host TU often CALLS no device-RNG function yet NAMES the state type in a declaration it
picks up through a header chain, and that is enough to break. In colmap,
`grep curandState src/**/*.cc` is empty, but `mvs/gpu_mat.h` declares
`FillWithRandomNumbers(..., const GpuMat<curandState>&)` and `mvs/gpu_mat_prng.h` declares
`class GpuMatPRNG : public GpuMat<curandState>`; the host-compiled `mvs/patch_match.cc`
reaches both through `patch_match_cuda.h -> cuda_texture.h -> gpu_mat.h`, and the gated
header fails that TU with `'curandState' was not declared in this scope` (measured: gated
include 4 errors, unmodified header 0). A forward declaration of the state type does compile
that TU (gated include plus `struct hiprandState;` gives 0 errors, because a class
template's members are instantiated lazily and neither a base-class use nor a reference
parameter needs the type complete), but it is not portable as one line: `hiprandState` is a
CLASS (`hiprand_kernel_rocm.h` expands
`DEFINE_HIPRAND_STATE(hiprandState, rocrand_state_xorwow)` to
`struct hiprandState : public rocrand_state_xorwow`), while `curandState` is a TYPEDEF on
the CUDA side (`typedef struct curandStateXORWOW curandState;`). Prefer the CMake fix,
because it repairs something the forward declaration does not: make every target that
includes the compat header declare the library whose header it therefore pulls in
(`hip::hiprand` and `roc::rocrand` alongside `hip::host`, mirroring whatever the CUDA arm
already lists, such as `CUDA::curand`). A monolithic `/opt/rocm` hides an omission there
because every ROCm header sits under one prefix; a split pip/TheRock SDK install does not,
and that is where the missing declaration surfaces as a not-found include. Leave a one-line
comment at the include saying why it is not narrowed, or the next reader tries the gate
again. (colmap)

**For plain C++ TUs the HIP headers `#define __noinline__ __attribute__((noinline))`**
(nvcc defines it only when compiling CUDA). glib's `__has_attribute(__noinline__)` then fails
("too few arguments provided to function-like macro"), so any TU that includes GTK after the
HIP runtime breaks. `#undef __noinline__` in the compat header when `!defined(__HIP__)`.
(LichtFeld-Studio)

**CUDA error codes HIP lacks still appear in switch labels.** Error classifiers name codes
such as `cudaErrorMisalignedAddress`, `cudaErrorInvalidPc` or `cudaErrorSystemNotReady`
that have no `hipError_t`. Aliasing them to the nearest HIP code duplicates a case label
when that code is also listed. Define each as `static_cast<hipError_t>(<CUDA value>)` after
checking no `hipError_t` enumerator uses the value: the labels stay distinct, and since HIP
never returns them they are unreachable. (LichtFeld-Studio)

The library status enums have the same gap. A `switch` that names every `cublasStatus_t` or
`cufftResult` value loses a label: hipBLAS has no `CUBLAS_STATUS_LICENSE_ERROR`, and hipFFT
1.0.26 (ROCm 10.1) has dropped `HIPFFT_LICENSE_ERROR` (its enum now skips 15). In a shim,
map each to a distinct value: `HIPBLAS_STATUS_UNKNOWN`, and `static_cast<hipfftResult>(15)`
(inside the enum's range, and the value cuFFT used). Do not define `CUDART_VERSION` to switch
the cases off: rocThrust and hipCUB key on it. (tsne-cuda)

**`hipError_t` is `[[nodiscard]]`, so a `-Werror` host build breaks on ignored results CUDA
allows.** HIP declares `hipError_t` as `[[nodiscard]]` whenever the code is compiled as C++17 or
later (`__HIP_NODISCARD` in `hip_runtime_api.h`, with no opt-out macro), while CUDA's `cudaError_t`
and `CUresult` carry no such attribute. Code that deliberately drops an error code (clearing a
sticky error with `cudaGetLastError();`, or calling an error-string getter for its out-parameter)
compiles silently with nvcc or g++ against CUDA headers and warns under HIP; in a host build that
uses `-Werror` (Warp's g++ flags) the warning becomes an error. Fix the few host-side sites (cast to
`(void)` or route through a void-returning wrapper) instead of adding `-Wno-unused-result`, which
would also hide real ignored errors; in device-side translation units without `-Werror` the warnings
are harmless noise. (warp-1.14)

**Never name the pointee struct of an opaque CUDA handle.** `CUstream_st`/`CUevent_st` do
not exist under HIP. Spell it `std::remove_pointer_t<cudaStream_t>` so it survives the
typedef swap. (cuSZ)

**`char` vs `signed char` vector base types differ.** HIP defines
`using char4 = HIP_vector_type<char, N>` with plain `char` members; NVIDIA's `char4` has
`signed char` members, and the two are distinct C++ types even when identically
represented. A type-machinery layer must report `char` as the base type of the canonical
`charN` on HIP, so a reference accessor binds. A test asserting
`is_same_v<signed char, BaseType<MakeType<signed char,N>>>` then fails on HIP. This is
dictated by the upstream HIP header, not the porting layer: forcing it would break the
reference accessor or introduce a non-canonical vector family into shared code. Numeric
behaviour is identical -- document it as an upstream type-identity difference, not a bug.
(CV-CUDA: TypeTraitsMakeTypeVectorTest/3.)

**A torch header the compat layer already aliases for can grow new symbols: alias the whole
helper, not just the names the compiler printed.** A plain CMake consumer of LibTorch
compiles `c10/cuda/*.h` un-hipified (torch hipifies only at `CUDAExtension` build time), so
its compat header aliases the `cuda*` names those headers use, and a torch update can add
names the alias list lacks (`cudaGraph_t`, `cudaHostFn_t`, `cudaUserObject_t`,
`cudaGraphUserObjectMove`, `cudaUserObjectRelease`, `cudaStreamGetCaptureInfo_v2` arrived in
`c10/cuda/CUDAGraphsC10Utils.h`). Do not route around it with an older pinned torch: that
hides the gap on one machine until the next refresh. Confirm each missing name has a `hip*`
counterpart in the installed HIP headers, then add the aliases.

Treat the compiler's error list as a LOWER BOUND. Two mechanisms hide a missing alias. (a)
clang drops the undeclared-callee diagnostic when one of the call's arguments is an
error-typed expression, and a variable declared with a missing type name is error-typed
exactly when that type name has no typo-correction candidate. So a type name clang does not
offer a "did you mean" for (here `cudaGraph_t`) masks every missing function alias in each
call that passes a value of that type; those names appear only in a second wave, after the
type aliases are added. (b) A call whose arguments are dependent is looked up only at
instantiation, so inside a template nobody instantiates it produces no error at all. Check
rather than reason: syntax-only compile the real torch header behind the compat header,
`#undef`-ing one alias at a time. A removal that still gives rc 0 means that name is
diagnosed only at instantiation; a removal that errors means it is required at parse time,
and if it was absent from the original error list, something in the same call masked it.
(LichtFeld-Studio, torch 2.14.0a0)

## Macros CUDA defines and HIP does not

**A macro that nvcc or a CUDA header defines and HIP does not reads as 0, and every gate on
it silently takes the other branch.** Nothing fails to compile: an arch-gated kernel body
becomes an empty stub or its `#else` printf, a dtype path disappears, and a feature looks
built yet never runs. Before calling a port complete, list every gate of every macro below
with the branch HIP takes.

| macro | under HIP | what to do at the gate |
|---|---|---|
| `__CUDA_ARCH__` | undefined in both compile passes | widen it with `\|\| defined(__HIP_DEVICE_COMPILE__)`; read the `__CUDA_ARCH__` entry below before defining it in a shim |
| `CUDART_VERSION`, `CUDA_VERSION` | undefined, unless the compat header pins one | guard the site (`\|\| defined(USE_HIP)`, or `!defined(USE_HIP) &&` in front of a legacy branch); never define `CUDART_VERSION` |
| `__CUDACC__` | undefined; HIP sources get `__HIPCC__` | treat it as `defined(__CUDACC__) \|\| defined(__HIPCC__)`; never define it globally |
| feature macros (`__CUDA_BF16_TYPES_EXIST__`) | not defined by the HIP counterpart header | define it in the shim that stands in for the defining header |
| library version macros (`CUBLAS_VERSION`, `CUBLAS_VER_*`, `NVJPEG_VER_*`, `__CUDACC_VER_*`) | undefined | resolve each gate by hand |
| `__HIP_PLATFORM_AMD__` (the reverse case) | defined by `hip/hip_runtime.h`, not by the compiler | above the includes use `__HIP__` or the `__GFX*__` predefines |

The entries that follow give the detail for each row. Two checks apply to all of them. Where
the legacy branch of a version guard names intrinsics HIP lacks (`h2div` against `__h2div`),
make HIP take the modern spelling. And confirm with the disassembly that an arch-gated kernel
was really built: it should contain real instructions (`v_pk_*_f16` for half2 code), not a
stub. (gpuRIR, oneflow, h2o4gpu, LichtFeld-Studio)

**`__HIP_PLATFORM_AMD__` is undefined until `hip/hip_runtime.h` has been included in that
TU.** It comes from the HIP headers, not from the compiler, so a gate in a header included
BEFORE the runtime header (wave-width selection above all) silently takes the CUDA branch:
it compiles and runs, with wave32 code on wave64 hardware. hipify-perl prepends the runtime
include at line 1, which masks the bug, so the gate is correct only by accident and breaks
for anyone building without hipify. Verify include order rather than trusting the prepend,
and prefer the compiler's own `__GFX8__`/`__GFX9__`/`__GFX11__`/`__GFX12__` device-pass
predefines, which need no header at all and are never defined by nvcc or by a host compiler:
`#if defined(__GFX8__) || defined(__GFX9__)` is both shorter and self-contained. Verify a
candidate gate with `hipcc -x hip --offload-arch=<arch> -dM -E` on an empty file before
trusting it, and assert the result per arch with a `static_assert` under
`#ifdef __HIP_DEVICE_COMPILE__` (an unguarded one in a `__global__` body is also evaluated in
the host pass, where the answer is legitimately different). One caveat on "undefined": the
`hip::host` and `hip::device` CMake targets put `-D__HIP_PLATFORM_AMD__=1` on the command
line, so a target that links them has the macro everywhere, host `.cpp` files included; a
bare `-x hip` compile, a Makefile build or a torch extension does not. (LC-framework)

**Defining `__CUDA_ARCH__` for the HIP device pass also feeds every numeric gate.** A compat
header that sets `__CUDA_ARCH__` (to 1, say) so `#ifdef __CUDA_ARCH__` device/host selections
work makes `#if __CUDA_ARCH__ < 530` TRUE in HIP device code. oneflow's `atomic.cuh` keeps a
`half2` atomic-add overload under exactly that gate whose body is `__trap()`, for pre-sm_53 GPUs;
on HIP it won overload resolution, so every `half2` atomic add raised an HSA hardware exception
(code 0x1016, no fault address) in an otherwise ordinary kernel. Grep every numeric
`__CUDA_ARCH__` comparison, not just `#ifdef`: `>=` gates silently drop native paths (slow but
correct), `<` gates can select traps or legacy emulation. Guard each with `!defined(USE_HIP)`
or the project's HIP macro at the site. (oneflow)

Whether to define it at all: prefer not to. Widening each gate at its site with
`|| defined(__HIP_DEVICE_COMPILE__)` touches nothing else, while a defined `__CUDA_ARCH__` is
read by HIP's own headers: `hip/hip_common.h` takes a nonzero value to mean "this is the device
pass" and sets `__HIP_DEVICE_COMPILE__` from it, so a definition that reaches the host pass, or
precedes the HIP runtime include, mislabels host code as device code. Define it only when a
codebase has more `#ifdef __CUDA_ARCH__` device/host selections than can be edited, and then
only under `#ifdef __HIP_DEVICE_COMPILE__`, after the HIP runtime include, with every numeric
comparison audited as above. (gpuRIR widened the gates; oneflow defined the macro.)

The `>=` gates are not only slow paths. A "no half support" stub behind `#if __CUDA_ARCH__ >= 530`
that does `printf(...); assert(false);` returns garbage silently, because release builds drop the
assert (oneflow's float16 softmax-cross-entropy gradient was off by up to 0.99 with exit code 0).
The same gate around a HOST launcher (`#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)`)
compiles the launch in the host pass but drops the kernel's instantiation in the device pass, so
the launch fails "Cannot find Symbol" at run time. A first audit that looked only at trap bodies
missed seven such sites; list every comparison (both operand orders, continuation lines) with its
HIP resolution rather than sampling. (oneflow)

**A code generator's device-pass guards live in strings, and fail only when a model uses
them.** A project that generates CUDA at run time (a simulator that turns model equations
into `.cu` files) can keep its `#if defined(__CUDA_ARCH__)` device/host selections in Python
string templates, one per generated function. A grep of the headers does not see them as
code, and each one reaches the compiler only when a user's model calls that function, so a
HIP port can pass a test set of ordinary models while every model with a lookup table, a
per-element random draw or a binomial sample fails to compile ("reference to __host__
variable ... in __global__ function"). Grep the generator's own source for the macro names as
well as the headers, and make the GPU tests call every function implementation the generator
can emit, not just representative models. If all the snippets pass through one assembly
function, widening the condition there only when the HIP backend is selected leaves the CUDA
output byte-identical, which is easy to prove by generating projects with both trees and
diffing them. (brian2cuda)

**Audit every macro class that gates a dtype or arch path, not just `__CUDA_ARCH__`.** CUDA
headers define feature macros that their HIP counterparts do not: `cuda_bf16.h` defines
`__CUDA_BF16_TYPES_EXIST__`, `hip_bf16.h` does not, so a shim that maps `<cuda_bf16.h>` to
`hip_bf16.h` turns every `#if defined(__CUDA_BF16_TYPES_EXIST__)` false. In oneflow that dropped
the `ncclBfloat16` case (every bf16 collective aborted "UNIMPLEMENTED" although RCCL supports
it), the bf16 gather/scatter kernels and five bf16 collective-kernel registrations, and two
audits that listed only `__CUDA_ARCH__` comparisons missed it. Define the macro in the shim that
stands in for the defining header, so it exists exactly where CUDA has it. Library version
macros that HIP leaves undefined (`CUBLAS_VERSION`, `CUBLAS_VER_*`, `NVJPEG_VER_*`,
`__CUDACC_VER_*`) read as 0 and silently skip whatever they gate (oneflow's
`cublasSetWorkspace`), while a `CUDA_VERSION` the compat header pins (to 11080, say) makes every
`< 12000` gate take the CUDA 11 branch. List each gate of each macro with its HIP resolution.
(oneflow)

**`CUDART_VERSION` is undefined under HIP, so every `#if CUDART_VERSION >= N` guard silently
takes its fallback.** Audit them: some fallbacks change behaviour, not just speed. A
`cudaMallocAsync` path that falls back to `cudaMalloc` makes the allocation uncapturable, and
`hipMalloc` inside a graph capture fails with `hipErrorStreamCaptureUnsupported`; when the
caller then bails out without ending the capture, every later allocation in the process
fails the same way and a whole test binary goes red behind one root cause. Extend the guard
(`|| defined(USE_HIP)`) where the HIP API exists, rather than defining `CUDART_VERSION`
globally, which flips untested pool paths at once. A pool whose fallback frees with
`hipFree` also synchronizes the device on every free, so a free while another stream waits on
a host callback blocks forever (a gated-producer test deadlocks; the async free returns at
once). To prove the CUDA view unchanged, note that `unifdef -UUSE_HIP` leaves
`#if X || defined(USE_HIP)` as written (it does not reduce `X || 0`); compare with
`-UUSE_HIP -DCUDART_VERSION=<N>` on both the upstream and the port file, for an N on each
side of every guard. Never define `CUDART_VERSION` in the compat header to get the modern
branches: hipSPARSE, hipSOLVER and other hip* headers test `#if defined(CUDART_VERSION)` to
select their NVIDIA backend, and with it defined on AMD they skip their own type definitions
(`hipsparseStatus_t` became an "unknown type"; h2o4gpu). Once the HIP default pool is in use, do not count on
`hipMemPoolAttrReleaseThreshold`. On ROCm 10.1 (Linux) it reaches the pool's VM heap only if
it is set after the pool's first allocation; set before it (as most code does, in setup), it
is lost and freed blocks are unmapped at the next synchronize whatever its value. Even when
set late it keeps memory only in 128 MiB chunks, so a threshold below one chunk (such as
64 MiB) keeps nothing, and a free can trim below the threshold. Correctness is unaffected;
measure before working around it. (LichtFeld-Studio)

**Project code that splits on `__CUDACC__` means "compiled by the device compiler", and HIP
does not define it.** Upstreams increasingly hide host-only C++23 (`std::expected` error
types, `std::source_location` capture) from nvcc with `#if !defined(__CUDACC__)` and show
device helpers only under `#if defined(__CUDACC__)`. A HIP source defines `__HIPCC__`
instead, so it parses the host-only branch (a C++23 `#error` or missing `std::expected`
under the HIP C++20 standard) and silently skips device fast paths. Treat `__CUDACC__` in
project code as `defined(__CUDACC__) || defined(__HIPCC__)`; `__HIPCC__` is set only for
HIP sources, so plain C++ TUs keep the host branch. Do not define `__CUDACC__` globally
instead: rocThrust takes its CUDA-system path under it. Re-grep after every upstream sync,
since new splits arrive with new code. (LichtFeld-Studio)

That rule is for a split whose host branch fails under HIP or hides a device path. A guard
that only keeps a language feature away from nvcc is different, and widening it by reflex is
a mistake. The shape:

    #if !defined(__CUDACC__) && defined(__cpp_concepts) && __cpp_concepts >= 201907L
    #define PROJECT_REQUIRES(...) requires(__VA_ARGS__)
    #else
    #define PROJECT_REQUIRES(...)
    #endif

Under HIP the guard is TRUE, so the `requires` clauses DO reach device compilation, the
opposite of the CUDA build. The guard exists because of an nvcc front-end limitation; hipcc
is clang, and clang compiles constrained templates in device code without complaint
(verified on ROCm 7.2, clang 22, `-std=c++23 -Werror`). Adding `&& !defined(__HIPCC__)`
would drop overload constraints on the AMD path only, a behavioural divergence introduced to
fix a problem that does not exist. So grep for every `!defined(__CUDACC__)` guard, TEST what
the AMD build does with the branch it takes, and extend the guard only if that compile
actually fails, saying so in the commit message. A force-included compat header that
deliberately leaves `__CUDACC__` undefined (because rocThrust keys its backend on it) is what
keeps such a guard live. (CubbyFlow)

**Upstream CCCL 3.0 compat shims are CUDA-only by construction: keep them out of the HIP
path.** CCCL 3.0 (CUDA 13) removed the CUB fancy iterators (`cub::TransformInputIterator`,
`cub::CountingInputIterator`), so upstreams are adding alias shims keyed to
`CUB_VERSION >= 300000` right after `#include <cub/cub.cuh>` -- which is exactly where a HIP
guard around the CUB include ends, so a later merge of upstream conflicts there and git can
even match the shim's closing `#endif` with the guard's. Resolve by placing the shim inside
the CUDA branch of the guard: only CUDA's CUB defines `CUB_VERSION`, and hipCUB (through
ROCm 7.14 at least) still ships `hipcub/iterator/transform_input_iterator.hpp` and
`counting_input_iterator.hpp`, so call sites reaching iterators through a
namespace-qualifier macro need no HIP-side counterpart. Expect this shim to appear in any
CUB-using upstream as it adopts CUDA 13. (aihwkit)

## Syntax and spellings clang rejects

**nvcc accepts a split launch chevron, `kernel << <grid, block >> >(args)`; clang does not.**
nvcc's front end tolerates spaces inside the triple angle brackets. clang lexes `<<<` as one
token, so the spaced form is a hard parse error ("expected expression" at the launch line).
Visual Studio's auto-formatter produces it, and Inria-derived 3DGS/2DGS rasterizers carry it
at every launch site. Respace to `kernel<<<grid, block>>>(args)`, which both compilers accept;
expect one edit per launch site, not per file. (diff-surfel-rasterizations, cuda_voxelizer)

**Three CUDA spellings that do not resolve under HIP, all cheap to guard.** They travel
together in the same family of projects, so check for all three at once:
`#include "device_launch_parameters.h"` (no such header; `threadIdx` and friends are
intrinsic under hipcc), `#include <cooperative_groups/reduce.h>` (hipify's
`CUDA_INCLUDE_MAP` maps only `cooperative_groups.h`, so this spelling is left verbatim and
then does not resolve), and `__trap()` (not declared by HIP's device runtime;
`#define __trap __builtin_trap` under `USE_ROCM`, which is what HIP's own `abort()` is).
Guard each with `#if !defined(USE_ROCM)` / `#if defined(USE_ROCM)` so the CUDA path is
untouched. (diff-surfel-rasterizations)

On a recent ROCm the include is the only thing missing in the cooperative-groups case:
`cg::reduce` exists in `<hip/cooperative_groups/hip_reduce.h>` (implemented in
`hip/amd_detail/amd_hip_cooperative_groups_reduce.h`), which `hip_cooperative_groups.h` does
NOT pull in. That header is present from ROCm 7.13 and absent in every 7.2.x release (through
7.2.4), so the `USE_ROCM` arm must not be a plain include: use
`#if __has_include(<hip/cooperative_groups/hip_reduce.h>)` and supply a fallback reduction in
the `#else`. strategy-b-torch.md ("Headers and functions hipify leaves broken") has both arms
and how to prove the fallback against real 7.2.x headers. Dropping the include outright is
right only when the code uses just `this_grid()`, `this_thread_block()` and `thread_rank()` --
grep for `cg::reduce`, `reduce_update_async` and `plus<>` first. (diff-surfel-rasterizations,
faster-gaussian-splatting)

**Tile shuffles of pointers and enums need `__shfl` overloads, declared before the
cooperative-groups header.** CUDA's `thread_block_tile::shfl` takes any trivially copyable type.
HIP's tile `shfl` calls `__shfl(var, src, width)`, which exists only for arithmetic types, so
`g.shfl(ptr, lane)` or `g.shfl(enum_value, lane)` fails with "no matching function for call to
'__shfl'" deep inside `amd_hip_cooperative_groups.h`. Add template overloads (a pointer through
`unsigned long long`, an enum through `std::underlying_type`), declared after
`hip/hip_runtime.h` and before `hip/hip_cooperative_groups.h`: for a pointer to a fundamental
type there is no associated namespace for argument-dependent lookup, so only declarations
visible at the template's definition are found. A HIP tile `ballot()` returns `unsigned long
long` (the tile's own bits, compacted), so `const auto v = g.ballot(...); __ffs(v)` becomes
ambiguous; store it in `unsigned int` as the CUDA type was. (HierarchicalKV, ROCm 10.2)

**Count cooperative-groups call sites per feature before calling them a blocker.** A grep
for `cooperative_groups` or `cg::` counts include lines and namespace aliases, and generated
code repeats those in every file: one project's "2392 references" were 2347 includes and
aliases and 45 real calls in two runtime files. Count `cg::<feature>(` per feature, then
check each against the installed HIP, not an older note. As probed on HIP 7.16 (the ROCm
10.1 SDK), HIP provides `this_thread_block`, `this_grid().thread_rank()` in an ordinary
launch, `tiled_partition<N>` with `meta_group_rank`, `cg::reduce` (include
`<hip/cooperative_groups/hip_reduce.h>`), `coalesced_threads`, `binary_partition` and
`memcpy_async` (include `<hip/cooperative_groups/memcpy_async.h>`). It lacks
`labeled_partition`, `cg::wait` (use `group.sync()`, which HIP documents as the completion
point), `atomicAdd_block` (alias it to `atomicAdd` for shared memory) and a `const`
`thread_block::group_dim()`. A `labeled_partition` followed by a `reduce` is replaceable
without assuming a wave width, by per-lane shared atomics or by `__match_any` with a 64-bit
mask and a `__shfl` loop over its bits. A hand-written CUDA replacement that loops
`lane < 32` with `__shfl_sync(mask32, v, lane)` does not port by widening the mask: on
wave64, lanes 32-63 read lanes 0-31 and the sums are silently wrong. Use
`__shfl(v, lane, 32)` with a per-half mask, and avoid `__shfl_sync` when the two halves
take different paths, because HIP's mask check traps. (colmap)

`cg::grid_group` is thinner than CUDA's as well: it offers `thread_rank()`, `size()`,
`num_threads()` and `sync()` and nothing else (`amd_hip_cooperative_groups.h`), so
`grid.block_rank()` and `grid.thread_index()` are CUDA-only. Substitute the raw builtins,
`blockIdx.x` for a one-dimensional launch grid and `blockIdx.x*blockDim.x + threadIdx.x` for
`thread_index()`, and note in a comment the grid dimensionality the substitution assumes.
(CuRast: `kernel_stage4_blend`.)

**HIP's bf16 header has no `make_bfloat162` and no `__floats2bfloat162_rn`.**
`amd_hip_bf16.h` offers only `__float22bfloat162_rn` (from a `float2`); the two-scalar
builders that FasterTransformer-derived kernels (masked MHA, `cuda_bf16_fallbacks.cuh`) call
everywhere are absent from every ROCm release checked, through clr trunk. Define them under
`__HIP_PLATFORM_AMD__` in the project's own bf16 fallback header (set `.x`/`.y`, and
`__float22bfloat162_rn(make_float2(a, b))`), not in a force-included compat header, where a
later ROCm definition would collide in every TU. (llm-awq)

**HIP has no `half2` overload of `__hmax2`.** `amd_hip_fp16.h` provides `__hmax`/`__hmin` for
`__half` and `__hmax2` only for `__hip_bfloat162`, so `__hmax2(half2, half2)` fails with "no
matching function" and a conversion error from `__hip_bfloat162`. Defining it from `__hmax` on
`__low2half`/`__high2half` keeps CUDA's NaN rule (a NaN yields the other input) and is correct, but
in a max-heavy inner loop the packed builtin in performance.md ("Scalar emulation of a packed
intrinsic") is many times faster, so prefer that form there. Putting the definition in the project's
namespace does not protect against a future HIP overload: `__half2` is a global type, so
argument-dependent lookup adds a global `__hmax2(__half2, __half2)` and the call becomes ambiguous
(checked with a simulated overload and `-fsyntax-only`). That failure is a loud compile error at the
call; the fix then is to delete the local definition. (Quest, ROCm SDK 10.1)

**A `__shared__` array of a struct that has default member initializers is a hard error on
HIP and silently accepted by nvcc.** Shared variables cannot be initialized; nvcc drops the
initializers with no diagnostic at all (measured on CUDA 12.8.93, and equally silent under
`-Xcudafe --display_error_number`), while clang reports `initialization is not supported for
__shared__ variables` and stops. The fingerprint is a struct that looks innocent,
`struct S { float sum = 0.0; float max = -1.0; };`, declared as
`__shared__ S sdata[blockSize]`. Check whether the kernel writes both members before reading
them (it usually does, because the initializers were never reachable in device code anyway):
if so, deleting the initializers is correct on every architecture and changes nothing on
CUDA. Only if a host caller genuinely relies on the defaults do you need an aligned byte
buffer plus `reinterpret_cast`. (rmcl: `SimpleLikelihoodStats` in the reduction of
`resampling.cu`.)

**MSVC-only upstreams accept code that clang and gcc reject**, so the HIP build (and the
CUDA build under nvcc) is a stricter compiler than the project has ever seen. Velvet carried
a member template whose parameter pack shadowed the class pack -- accepted by MSVC, rejected
by Clang and GCC. Expect missing `typename`/`template` disambiguators, two-phase lookup
failures and narrowing conversions. Fixes are additive and arch-independent, so they help
the CUDA build too and are not HIP-specific hacks. (Velvet)

**Eigen DECOMPOSITIONS called from `__device__` code do not survive the move to hipcc, even
though Eigen's core arithmetic does.** Eigen defines `EIGEN_DEVICE_FUNC` as
`__host__ __device__` for hipcc just as it does for nvcc, so `Matrix`, `Map`, coefficient
access, `.cross()` and small products all compile. The boundary is not a matrix SIZE, it is
which entry point you call: `SelfAdjointEigenSolver::computeDirect()` (defined only for 2x2
and 3x3) is device-safe, while the general `compute()` path -- which the CONSTRUCTOR runs,
at every size including 3x3 -- reduces through a Householder tridiagonalization that calls
Eigen's general matrix-matrix product kernels (`scaleAndAddTo`, `applyThisOnTheLeft`,
`BlasUtil`'s `extractScalarFactor`), and those are host-only. Do not read "the small sizes
are fine": a 3x3 constructor fails at `Tridiagonalization.h:434`. The diagnostic is a
handful of `reference to __host__ function ... in __host__ __device__ function` errors deep
in Eigen headers, all tracing to ONE kernel: read the `note: called by` line and fix the
call site, do not conclude Eigen is unusable on the device. Physics and graphics codebases
hit this because per-element positive-semi-definite projection of a small Hessian is a
standard move. Replace the solver with a cyclic Jacobi sweep over plain `Scalar[N][N]`
arrays: no workspace, no matrix product, backward stable on symmetric input, and one
implementation for both back ends rather than an `#ifdef` fork (normalize the input first:
faults-numerics.md, "A hand-written Jacobi/eigenvalue sweep replacing a library solver must
normalize first"). Do the reconstruction with explicit loops as well, since a fixed-size
12x12 `V * D * V.transpose()` re-enters the same host-only product path. Verify it on the
HOST against the Eigen original over a few thousand random symmetric matrices before
touching the GPU: that separates "my replacement is wrong" from every other hypothesis in
about five minutes. (GPU_IPC: `PDSNK<double,12>`, agreed to 1.5e-14 relative over 20000
trials.)

**`__ldg` is usually a non-problem on HIP; check before working around it.** ROCm ships
`__ldg` overloads for the scalar AND vector types (`hip/amd_detail/hip_ldg.h`: `char2`,
`int4`, `longlong2`, ...; `amd_hip_fp16.h` adds `__half`/`__half2`), and every one of them
is literally `return ptr[0]`, because AMD has no separate read-only-cache instruction to
select. So a `RO_CACHE(x)` macro over scalars needs no HIP arm at all, and `(x)` and
`__ldg(&x)` are the same code. Only an arbitrary USER type is unsupported. A CUDA generics
wrapper that provides `__ldg` for arbitrary types by aliasing (`generics/ldg.h`, guarded on
`__CUDA_ARCH__ >= 350`) also compiles under hipcc: the guard is false, so it falls back to
`*ptr`, and its non-template HIP competitors win overload resolution for the scalar calls.
GooFit's port first added a HIP-only `RO_CACHE(x) (x)` on the belief that neither worked;
compiling a two-line hipcc test against the wrapper disproved it and the whole arm was
deleted. Test the actual header before adding a guard. (GooFit)

**An uninitialised header-only submodule fails as a HIP error, not as a missing file.**
When a project bundles GLM (or Eigen, or any header-only library) as a submodule and the
clone did not recurse, the `-I` path still exists and is simply empty, so the compiler falls
back to the system copy. A system GLM predating HIP support does not recognise `__HIP__` as
a device compiler, so every `glm::dot`/`glm::length` call from device code is reported as
"call to __host__ function from __device__ function" and `operator[]` becomes const -- dozens
of errors that read like a broken port and are a missing checkout. Note the include paths in
the diagnostic: if they point at `/usr/include`, run `git submodule update --init
--recursive` before changing a single line of source. (diff-surfel-rasterizations)

**clang checks kernel bodies in the HIP host pass; nvcc does not.** A macro that splits
device and host forms on `#if defined(__CUDA_ARCH__)` takes the host form when clang
compiles the host side of a HIP TU, and clang still type-checks the instantiated
`__global__` bodies there. A host-only call in that branch (a string formatter, a throw)
inside a kernel is then `call to __host__ function from __global__ function ... when
compiling for host`, while nvcc builds the same code. It shows only where the macro is live,
typically a debug assert that is `((void)0)` under NDEBUG, so Release passes and the first
Debug build fails. Build Debug once after every upstream sync, and give HIP sources the
kernel-safe form in both passes (`defined(__CUDA_ARCH__) || defined(__HIPCC__)`) or make
the callee `__host__ __device__`. Only calls that go through overload resolution (templates,
overload sets) fail at once; a call to a single non-template `__host__` function in a kernel
is a deferred diagnostic that never fires, because the host pass does not emit kernel bodies.
That gives a narrower fix that keeps the host form for host code in HIP TUs: under
`#elif defined(__HIPCC__)` evaluate the message argument in `[&]() { return (message); }()`.
clang makes an unattributed lambda `__host__ __device__`, so the template call inside it is
legal in a kernel's host-pass body, while host functions still get the formatted message.
(LichtFeld-Studio)

## helper_math.h and vector types

**The CUDA Samples `helper_math.h` makes vector arithmetic ambiguous under HIP.** HIP's
`float3`/`int3`/... are `HIP_vector_type`, which already defines element-wise operators: the
vector-vector compound assignments as non-template MEMBERS and vector-vector `*`, `/` as
non-template FRIENDS, the rest as templates. helper_math's free non-template
`operator+=(float3&, float3)`, `operator*(float3, float3)`, ... tie with those, so every such
expression fails as ambiguous, on host and device, including inside helper_math itself
(`smoothstep`). Do not guard the colliding functions one by one (about 40 of them): wrap the
whole negate/add/subtract/multiply/divide block in `#if !defined(__HIP_PLATFORM_AMD__)` and
let HIP's operators, which compute the same per-component results, take every case.
(cuda_voxelizer)

The header's second block, the host fallbacks for `fminf`/`fmaxf`/`min`/`max`/`rsqrtf` under
`#ifndef __CUDACC__`, needs its guard extended as the next entry describes, because clang's
HIP math header already defines host `min`/`max(int, int)` and the copies are redefinitions in
`.hip`/`LANGUAGE HIP` TUs. Host `.cpp` files compiled by the host compiler still need the
block. Guarding it off neither causes nor cures the separate truncation fault: stock
helper_math never supplied a floating-point host `min`/`max`, CUDA's own headers did, and
under HIP a host call with float arguments lands on `(int, int)` whatever this header does
(faults-numerics.md, "Host `min`/`max` on floating-point values truncate to int under
hipcc"). Run that entry's detection pass on any project that uses helper_math.

**HIP's `blockIdx`/`threadIdx` do not convert to `uint3`, and a `uint3` does not convert to
`dim3`.** In CUDA the built-in index variables are `uint3`, and a `uint3` (from
`make_uint3()`) converts to `dim3`. In HIP the built-ins are distinct types, so three things
CUDA accepts fail to compile: passing them to a function that takes `uint3` (waLBerla's
`gpu::FieldAccessor::set(uint3, uint3)`), `uint3(blockIdx)`, and returning a `uint3` from a
function declared to return `dim3`. Construct from the members,
`make_uint3(blockIdx.x, blockIdx.y, blockIdx.z)` or `uint3{threadIdx.x, ...}`, and return
`dim3(...)`. The built-ins do convert to `dim3`: `dim3(blockIdx)` compiles. Do not write a
helper that takes a built-in by value, because their copy constructor is deleted. (ESPResSo
with waLBerla, HIP 7.16)

**HIP's host-side `__half` needs the HIP compiler, and `HIP_vector_type`'s operators hijack
mixed half arithmetic.** Host code that uses `__half` (integer constructors,
`__ushort_as_half`, half arithmetic) cannot be built with a plain C++ compiler on ROCm:
`hip/hip_fp16.h` falls back to `amd_detail/hip_fp16_gcc.h` there, whose `__half` has only
float and double constructors (`__half(int)` is ambiguous), no arithmetic and no bit-cast
functions. CUDA's `cuda_fp16.h` gives g++ the full host API, so this shows up only on ROCm.
Compile those sources as HIP (`LANGUAGE HIP`), then expect two follow-ups: file-scope names
`min`/`max` collide with the global `min`/`max` HIP declares, and `__half_raw{bits}`
brace-initializes HIP's `_Float16` union member numerically (use `__ushort_as_half`).
Separately, `HIP_vector_type`'s `operator OP(U, const vec&)` and `(const vec&, U)` accept any
`U`, so `half3 - float3` with a project's own `half3` struct resolves to HIP's operator and
fails inside `make_vector_type`; a concrete overload for the project's half type is more
specialized and wins. Its member unary `-` and `~` also keep `char`/`short` element types
where CUDA's promote to `int`; only non-template overloads beat the member templates.
`__double2half` does not exist in HIP; `__half(static_cast<_Float16>(d))` rounds once, as
CUDA does. (CV-CUDA)

**A helper whose name the standard library also declares must NOT be guarded on
`__CUDA_ARCH__`/`__HIP_DEVICE_COMPILE__`.** Those macros answer "am I in the device pass",
not "is this name already taken", and two standard libraries answer the second question
differently: libstdc++'s `<math.h>` does `using std::lerp;` at global scope (arriving
transitively via `<torch/extension.h>`), while the MSVC STL declares `lerp` in namespace
`std` only. So a pass-keyed guard fixes one OS and breaks the other: the Windows host pass
found NO viable `lerp(float,float,float)` and failed to parse the device function body, then
supplying one unconditionally made the Linux host pass find TWO ("declaration conflicts with
target of using declaration already in scope"). Any C++20 name the standard library also
declares is exposed to this: `lerp`, `midpoint`, `clamp`, `hypot`, `gcd`. The fix is not a
better `#if`, and never `_WIN32`/`_MSC_VER` as a proxy for the standard library: give the
helper a name the standard library does not use and define it once, unconditionally, as
`__device__ __host__`, so it resolves in both passes on every toolchain. Keep only the
same-name overloads whose PARAMETER TYPES differ from the standard one (`lerp(float2,...)`
never conflicted). (faster-gaussian-splatting: the `helper_math.h` scalar `lerp`)

When the maintainer keeps the name and every caller is device code, a second fix exists: on
HIP only, declare the helper `__device__` (not `__host__ __device__`), unconditionally, not
keyed on the compile pass. clang's HIP mode treats a `__device__` function and a host function
of the same signature as distinct overloads, so it coexists with libstdc++'s global `using
std::lerp;`, and because it is declared in both passes, the host pass with the MSVC STL still
finds it when parsing device function bodies. With hipcc (HIP 7.16) on a file that includes
`<math.h>` and calls `lerp(float, float, float)` from device code, `__host__ __device__` fails
at `-std=c++20` with "declaration conflicts with target of using declaration already in
scope", while `__device__` compiles at C++17 and C++20, and the device ISA was identical to
the renamed version. It is not valid when host code also calls the helper. What exposes the
collision in the first place: PyTorch's headers require C++20 from 2.14 on (`#error C++20 or
later compatible compiler is required to use PyTorch`; 2.13 and earlier accept C++17), on
CUDA and ROCm alike. (faster-gaussian-splatting)

**Extend a CUDA guard for HIP; never re-key it.** Upstream's `#ifndef __CUDACC__` around
host fallbacks for `fminf`/`min`/`max`/`rsqrtf` (CUDA-Samples `helper_math.h` and its
copies) also admits the HIP device pass, because HIP never defines `__CUDACC__`. Re-keying
it to `!defined(__CUDA_ARCH__)` exposes the fallbacks to nvcc's HOST pass, where `int
max(int, int)` redefines CUDA's and every TU including the header fails -- invisible on
ROCm. Add the HIP macro to the original test instead (`!defined(__CUDACC__) &&
!defined(__HIP__)`), which leaves the CUDA and plain C++ expansions identical. (clang
predefines `__HIP__` and `__HIPCC__` alike in both passes of every HIP compile, so either
spelling works; `__HIPCC__` is also set when HIP's headers wrap CUDA, so `__HIP__` is the
narrower test. Pick one per project.) HIP's host
pass does not need the fallbacks when no host-emitted code calls them; compile the host
objects without them to check. (LichtFeld-Studio)

## Keeping the CUDA path intact

**A ROCm-only branch inside a shared template still COMPILES on the CUDA path.** Gating the
divergence on a `static constexpr bool` predicate that is false on NVIDIA (`if(needsCast())`)
keeps the branch from RUNNING there, but a plain `if` instantiates both arms in every
instantiation, so an expression that is ill-formed for the CUDA element type breaks the
NVIDIA build -- and nothing on ROCm notices, because there it is the branch that runs. Write
such branches `if constexpr`: the discarded arm of a condition that is not value-dependent
after instantiation is never instantiated, which is exactly the guarantee a
platform-conditional block needs. The trap that found it: `(ScalarType)0` with
`ScalarType = __half`. In a HOST translation unit, CUDA headers up to 12.1 keep `__half`'s
integer converts behind `#if defined(__CUDACC__)`, so `int` converts to `float` and to
`double` at equal rank and the cast is ambiguous ("call of overloaded `__half(int)` is
ambiguous"). CUDA 12.2 relaxed that gate to `#if !defined(__CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__) || defined(__CUDACC__)`,
so every release up to 12.1 is affected, plus any 12.2-or-newer build that defines the
disable macro; a recent toolkit hides the bug unless the project opts out. Prefer a float
literal (`0.f`) in half comparisons anywhere, and compile-check the CUDA path against 12.1
headers or older, not only the newest toolkit installed -- bisected against the real
headers, 12.1 fails and 12.2.53 onward compiles. (marian-dev)

**"Portable" is a claim to verify against nvcc, not a property of how the edit reads.** A
device primitive that only one back end has, replaced by something that looks neutral, can
still be HIP-only. 3DUNDERWORLD-SLS-GPU_CPU swapped a never-taken overflow guard's
`asm("trap;")` (illegal on amdgcn) for `__builtin_trap()`, unconditionally, with a commit
message calling it portable. True for clang, where it is device-callable; false for nvcc,
which rejects `__builtin_trap()` as a host-only function called from `__device__` and fails
the CUDA build outright. Ask of every edit inside a `.cu`/`.cuh` touched for HIP: does this
identical text also compile under nvcc, or does it only build because the port never tried?
The repair that keeps it from recurring is a compat-header macro
(`#define gpuTrap() __builtin_trap()` on the HIP side, `#define gpuTrap() asm("trap;")` on
the CUDA side) called at the use site, not an open-coded `#if defined(USE_HIP)` in the
`.cuh`: the back-end condition then lives in the one file that owns the `cuda*`-to-`hip*`
aliases, and the CUDA branch reproduces upstream's text verbatim. A device intrinsic with no
cross-toolchain spelling is a compat-header entry like any alias. validation.md ("Diff the
PTX") has the proof that the CUDA path was restored exactly. (3DUNDERWORLD-SLS-GPU_CPU)

**C++17 `std::clamp` and friends work in HIP device code but not under nvcc.** clang treats
a `constexpr` function as implicitly `__host__ __device__`, so `std::clamp` compiles in a
HIP kernel; nvcc rejects it without `--expt-relaxed-constexpr`. Fine inside an AMD-only
branch, a build break in SHARED device code that nvcc also compiles. Check which side of
the guard it lands on. (popsift)

**An arch-specific fix keyed on the OS or the ROCm version can BREAK an already-validated
arch**, because those are only PROXIES for the real dependency. A Windows setup on TheRock
nightly wheels can pair a newer ROCm with an OLDER torch while a Linux setup pairs an older
ROCm with a newer torch, so OS, ROCm version and torch hipify generation are tangled and can
ANTI-correlate. A guard written `#ifdef _WIN32` when the true dependency is the
hipify generation works only by that accidental correlation. Guard on the real axis, and
there are three:

- **OS and toolchain** -> `_WIN32`: MSVC-vs-MinGW linkage, DLL import libs, path
  separators, and linker workarounds such as an MSVC `/ALTERNATENAME` directive standing in
  for a symbol a DLL fails to export.
- **HIP runtime version** -> `HIP_VERSION_MAJOR`/`MINOR`: header behaviour, added or
  removed symbols. (MMseqs2: a Windows commit defined `HIP_DISABLE_WARP_SYNC_BUILTINS`
  unconditionally because `<amd_hip_bf16.h>` defines `__shfl_*_sync` on ROCm 7.14+, but on
  7.2.x `__syncwarp` lives inside the SAME guard, so the define stripped `__syncwarp` and
  broke Linux.)
- **torch hipify generation** -> `torch.utils.hipify.__version__`. v1 renamed the `c10`/`at`
  `cuda` classes to `hip`; v2 keeps the CUDA spellings public. It matters only in a TU hipify
  never sees; strategy-b-torch.md ("hipify generation: v1 renames, v2 masquerades") has the
  build-time detection and the neutral define. (aihwkit: a Windows commit hard-coded
  `c10::hip::getCurrentHIPStream`, needed on TheRock's older torch but absent on Linux's
  newer one, breaking that build.)

## Language standard

**rocRAND's device headers need C++14 from ROCm 10.1.** `rocrand_threefry_common.h` uses
`std::enable_if_t`, so a TU that includes `hiprand_kernel.h` under `-std=c++11` fails with
`no template named 'enable_if_t' in namespace 'std'`. ROCm 7.2 still accepted C++11, so a
port validated there breaks only on a newer SDK. Look for a build script that pins
`-std=c++11` for nvcc and reuses it for hipcc; raise it on the HIP path only. (TIGRE)

### HIP + C++23 on an older toolchain: `isfinite cannot overload` (and how to stay forward-compatible)

Symptom, on the first HIP translation unit compiled at `-std=c++23`:

```
__clang_cuda_math_forward_declares.h:93:17: error: __device__ function 'isfinite'
  cannot overload __host__ __device__ function 'isfinite'
<MSVC>\include\cmath:672:17: note: previous declaration is here
  672 | _CLANG_BUILTIN1(isfinite)
```

**Cause.** C++23 (P0533R9) makes the standard math functions `constexpr`. HIP treats a
`constexpr` function as implicitly `__host__ __device__`, so clang's
`__clang_cuda_math_forward_declares.h`, which declares `isfinite`/`isinf`/`isnan`/`isnormal`
`__device__`-only, can no longer overload them. Every HIP TU fails identically -- they all
pull the same `__clang_hip_runtime_wrapper.h` chain.

**Already fixed upstream**: llvm/llvm-project#201563 (merged 2026-06-09) includes the
forward-declares BEFORE `<cmath>`. Reported at ROCm/llvm-project#285 and #2669, both closed.
Do NOT file this again.

**Which toolchains have it** (verified 2026-08-24 by reading the shipped header, not by
version arithmetic -- ROCm cherry-picks, so commit ancestry misleads):

| Toolchain | Has fix? |
|---|---|
| ROCm 7.14 (checked on the nightly build `7.14.0a20260612`) | no |
| ROCm 10.0 (`release/therock-10.0`, llvm pin 2026-07-28) | **no** -- it will ship without it |
| ROCm 10.1 nightly (`10.1.0a20260823` and later) | **yes** |
| ROCm/llvm-project `amd-compiler-2026-06` .. `-09`, and `main` | no |
| ROCm/llvm-project `amd-staging` | yes |

To check any toolchain in one line, read the order in its own header:
`grep -n -E '#include <cmath>|__clang_cuda_math_forward_declares' $(clang++ -print-resource-dir)/include/__clang_hip_runtime_wrapper.h`
-- forward-declares on the LOWER line number means fixed.

**Preferred fix: upgrade.** ROCm 10.1 nightlies carry it. Note the index moved: the old
`rocm.nightlies.amd.com` is frozen at `7.14.0a20260612`; current builds are at
`https://nightly.repo.amd.com/rocm/whl-next/`. Package layout changed too --
`rocm-sdk-devel` is now a thin metapackage whose copy of this header is a SYMLINK into
`_rocm_sdk_core`, and device packages renamed `rocm-sdk-libraries-gfx*` ->
`rocm-sdk-device-gfx*`.

**If you must stay on an older ROCm**, use `assets/HipCxx23Shim.cmake`. It generates a copy
of the INSTALLED wrapper with the two includes reordered and puts it on the include path
ahead of the resource dir -- the upstream fix, applied locally, with the language standard
untouched.

The asset's `execute_process` probe can also be written as two `try_compile(...
SOURCE_FROM_CONTENT ... NO_CACHE HIP_STANDARD 23)` calls, which reuse the project's own HIP
compiler settings. Keep `<hip/hip_runtime.h>` in the probe: a bare `__global__` fails the host
pass ("Can't find declaration for hipLaunchKernel") and makes the shim look ineffective. In a
target that mixes HIP and C++ sources, append the shim's `-I` to `CMAKE_HIP_FLAGS` so it
reaches HIP translation units only. (CubbyFlow)

Measured alternatives, all tried against ROCm 7.14 on gfx1151:

| Approach | Compiles at C++23 | Keeps C++23 language | Keeps C++23 STL |
|---|---|---|---|
| `-std=c++20` pin | yes | **no** | **no** |
| `-D_CONSTEXPR23=` | **no** (the STL redefines it) | -- | -- |
| `-D_HAS_CXX23=0` | yes | yes | **no** |
| generated wrapper shim on `-I` | yes | yes | yes |

Prefer the shim. Both `-std=c++20` and `-D_HAS_CXX23=0` leave HIP TUs and host TUs with
DIFFERENT standard-library configurations, which is an ODR hazard whenever a shared type
crosses that boundary -- and MSVC has no `detect_mismatch` for `_HAS_CXX*`, so it fails
silently at runtime rather than loudly at link. That matters most in exactly the projects
that hit this, since they are C++23 project-wide.

**Forward compatibility is the whole design of the module**, because a shim that keeps
applying after the toolchain is fixed would shadow a good header with a stale copy:
generated from the installed header at configure time and never checked in; returns empty if
a plain probe already compiles; returns empty if the installed header is already in the fixed
order; and re-probes WITH the shim, discarding it and warning if it did not actually help.

Verified in four directions on a gfx1151 Windows machine: ROCm 7.14 at C++23 -> shim generated and a real HIP
target builds; a simulated ROCm 10.1 toolchain (real 10.1 header via `-resource-dir`) ->
"no shim needed", nothing generated; a C++20 project -> "no shim needed"; and a negative
control with the shim disabled -> the same target fails on the same header, proving the pass
was not vacuous.

Nothing here is guarded on Windows. MSVC's STL is merely the first to implement P0533R9;
ROCm/llvm-project#285 notes that libstdc++ doing so will break Linux identically. A
`WIN32`-only guard, or a ROCm version check, would miss that. Probe for the capability.
