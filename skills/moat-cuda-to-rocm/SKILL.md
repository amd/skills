---
name: moat-cuda-to-rocm
description: Port a CUDA project to ROCm/HIP for AMD GPUs and keep the port working. Covers strategy selection (CMake HIP language or PyTorch extension), the faults where AMD is strict and CUDA is lenient, wave32/wave64 wavefront portability, CUDA library substitutions and their gaps (cuBLAS, cuSOLVER, cuDNN, cuSPARSE, Thrust, NCCL, RAPIDS, nvJPEG, NVDEC), ROCm build and link problems on Linux and Windows (clang-cl, lld-link, DLL loading, the pip-installed ROCm SDK), and the validation bar. Use when planning, implementing, reviewing or validating a CUDA-to-HIP port, when a HIP build fails to compile or link, when it compiles but misbehaves at runtime, when a CUDA library has no obvious ROCm equivalent, or when syncing a ROCm port with upstream changes.
---

# CUDA to ROCm/HIP

Accumulated across about a hundred real ports. This file routes; the knowledge is in
`references/`, one topic per file, loaded only when you need it.

**Most porting bugs are not in symbol names.** Hipify handles the renaming. The bugs are
semantic differences where AMD is strict and CUDA is lenient. Most surface at runtime on a
build that compiled cleanly, but a clean hipify is not a clean build either: hipify leaves
literals and types alone, so some idioms (a 32-bit `*_sync` mask is the common one) hipify
without complaint and then fail to compile under HIP.

## Naming: ROCm or HIP?

- **HIP** is the programming model: the kernel dialect plus the `hipXxx` runtime API, the
  analogue of CUDA C++ and the CUDA runtime.
- **ROCm** is the platform: compiler, runtime, driver and the roc*/hip* libraries, the
  analogue of the CUDA Toolkit.

The CODE port is "to HIP"; the TARGET, build flag and libraries are "ROCm". Never call the
platform "HIP" or the kernel dialect "ROCm". Details: `references/naming.md`.

## ROCm version numbers in this skill

Entries say where they were seen. The numbers compare the ordinary way: 7.2 < 7.13 < 7.14 <
10.0 < 10.1 < 10.2 < 10.3. Three things can mislead:

- **Dates do not order them.** The 7.2.x patch releases (7.2.1 to 7.2.4, March to May 2026)
  came out alongside 7.11, 7.12 and 7.13, so 7.13 (May 2026) is newer code than 7.2.4
  (also May 2026).
- **The major number jumps from 7 to 10.** 7.14 was released in July 2026 (with a 7.14.1
  patch in August), and the next releases are 10.0 (August 2026), 10.1 (October 2026) and
  10.2 (November 2026). Nothing was released in between.
- **A date suffix marks a nightly build, not a different series.** `7.14.0a20260612` is a
  pre-release build of 7.14 and `10.3.0a20261118` one of 10.3; every version has such builds
  before its release. An entry seen on a nightly says so, and may or may not hold on the
  release of the same number.

HIP has its own version number, which stayed on the 7.x line when ROCm's went to 10: ROCm
7.14 carries HIP 7.14, ROCm 10.1 carries HIP 7.16, and the 10.2 and 10.3 builds carry 7.17
and 7.18. So gate code on a feature test or on `HIP_VERSION_*`, never on the ROCm number. A
workaround tagged with a version or a date may have expired: re-check it on a newer ROCm
before carrying it.

## Which strategy

Classify the build first; implementing the wrong strategy correctly is still wrong.

| build | strategy | shape |
|---|---|---|
| pure CMake | **A** (preferred) | one `cuda_to_hip.h` compat header, `enable_language(HIP)`, `set_source_files_properties(... LANGUAGE HIP)`; sources keep CUDA spelling |
| pytorch extension | **B** | rely on torch's build-time hipify; fix only what it cannot |
| anything else | neither | driver-API, runtime PTX, Go/cgo, qmake and codegen builds: `references/platform-limits.md`, "Runtime-compiled kernels" |

**How to tell:** `find_package(Torch)`, `torch.utils.cpp_extension`, `CUDAExtension`, or a
torch dependency in `setup.py`/`pyproject.toml` means a pytorch extension, Strategy B.
Otherwise treat it as Strategy A, whatever the build system: a meson or Makefile project
ports Strategy-A-shaped, writing by hand what `enable_language(HIP)` would have given.

Strategy A is the minimal-footprint model: the compat header is a no-op on NVIDIA, so the
CUDA build is untouched. Where the project includes CUDA headers by name, prefer the
shim-header method (shadow headers on the HIP include path only), which leaves the sources
byte-identical to upstream. Recipes: `references/strategy-a-cmake.md`,
`references/strategy-b-torch.md`.

## Before porting

Check whether an AMD port already exists and whether it is authoritative: an AMD-official
work in progress shifts the value to validating and improving it, while a one-off community
fork is a hint, never code to inherit. `references/assess-existing-support.md`.

Some CUDA has no HIP analogue to translate INTO (CUTLASS, CuTe, TensorRT, raw PTX), so
hipify succeeds at doing nothing and the port looks small. A reimplementation against an
AMD-native target is not the same verdict as "no target exists"; say which.
`references/no-hip-equivalent.md`.

## Where to look

| you have | open |
|---|---|
| wrong results, a hang or a fault on one wavefront width; code with `warpSize`, a literal 32, `__shfl*`, `__ballot`, lane masks | `references/faults-wavefront.md` |
| a crash, a hang, stale or garbage data, an intermittent failure; streams, events, async copies, graph capture, managed memory, several GPUs | `references/faults-memory-streams.md` |
| textures (also `tex*` compile errors on gfx94x), OpenGL or Vulkan interop | `references/faults-textures-interop.md` |
| numbers that differ from CUDA or the CPU: FMA contraction, rounding, `min`/`max`, half-precision GEMM, big-integer arithmetic | `references/faults-numerics.md` |
| a compile error, or code silently skipped: macros HIP does not define, includes, `__CUDA_ARCH__`, `helper_math.h`, C++23 | `references/build-headers-macros.md` (C++23: `assets/HipCxx23Shim.cmake`) |
| CMake, link or load errors: `LANGUAGE HIP`, `-fgpu-rdc`, OpenMP, RUNPATH, LibTorch, installed packages, submodules | `references/build-cmake-link.md` |
| anything on Windows: SDK install, clang drivers, CRT and link errors, DLL loading, Windows-only wrong results, debugging | `references/windows.md` (link wrapper for a clang-cl host with `-fgpu-rdc`: `assets/hip_link_win.py`) |
| a CUDA library to replace, or a missing routine or mode: cuBLAS, cuSOLVER, cuDNN, cuSPARSE, Thrust/CUB, cuRAND, NCCL, RAPIDS, nvJPEG, NVDEC, NVML | `references/libraries.md` |
| a device query that answers differently, `clock64`, occupancy, a grid too large to launch, kernels compiled at run time (PTX, NVRTC, Triton), an integrated APU | `references/platform-limits.md` |
| a failure you cannot place yet, or a suspected ROCm bug | `references/diagnosis.md` |
| correct results but far slower than the CUDA build: profiling, kernel ISA, emulated packed intrinsics | `references/performance.md` |
| what counts as validated; compiling the CUDA path with nvcc | `references/validation.md` |
| a headless Linux GPU host that needs a GL or Vulkan context | `references/linux-host-setup.md` |
| an open PR that upstream has moved under | `references/upstream-sync.md` |

## Reviewing a port

A review is a sweep, not a diagnosis. Finding one fault that explains the reported symptom
does not end it; ported code usually carries several independent faults, and the one nobody
asked about is the one that ships.

1. Walk the index below against the code, file by file, and note a hit or a clean pass for
   each group.
2. Check what will not compile under HIP separately from what will misbehave at runtime.
   "Hipify ran clean" says nothing about the first.
3. For each fresh device allocation, find its first access. If that is a read-modify-write
   (an `atomicAdd` counter, an accumulator, a histogram bin, `+=`), a memset or an init
   kernel must precede it on the same stream: ROCm does not hand back zeroed memory.
4. Report every hit with its fix, even when it is unrelated to the question asked.

## Portability rule for shared code

Any fix to code that is not arch-guarded must be correct on **both** wavefront sizes. A
per-arch hack that fixes wave32 and regresses wave64 makes the archs ping-pong, and each
"fix" then costs a full re-test on every architecture. Prefer a runtime `warpSize` query on
the host and a compile-time constant in device code.

## Validation bar

A real-GPU test run is the only evidence that counts. Lint is not validation, and a CPU-only
build proves compilation and nothing else. Cover both wavefront widths and each OS the
project supports: at least one wave64 (CDNA) GPU, one wave32 (RDNA) GPU, and Windows when
the project builds there. Also compile the CUDA path with nvcc as a no-regression check: an
additive ROCm port must leave the CUDA build a pure passthrough. `references/validation.md`.

## Keeping an open PR mergeable

Sync with a MERGE of upstream's default branch, never a rebase on a published branch, and
treat the clean auto-merge as the risk: upstream writes new code against the surfaces the
port multiplexed, so it lands the NVIDIA half only. `references/upstream-sync.md`.

## Writing it up

Commit messages, PR bodies and code comments go to upstream maintainers who do not know your
team's vocabulary. Before pushing, check the whole branch, not just the last commit, for
in-house terms: internal stage or state names, tool names, and this skill's own labels such
as "Strategy A" or "fault class". Also grep new comments for the names of OTHER projects you
have ported: "the same fix as in project X" is a dangling reference to an upstream reader.
State the hazard itself. `references/naming.md`.

## Index

One line per entry of the reference files above, generated from the entry titles (do not
edit by hand). Scan it for your symptom, then open the file for the cause and the fix.

<!-- index:begin -->
**`references/faults-wavefront.md`** -- Faults: wavefront and warp width

- Never hardcode 32
- A static array sized from the warp width needs the bound that matches the DIRECTION the ...
- A block NARROWER than one wavefront makes `blockDim.x / warpSize` zero, and a block ...
- The shuffles themselves read lanes that hold no thread, whatever the count says
- Warp-tiled dispatch tables written for 32 lanes break when `kWarpSize` becomes 64
- Special case: a warp-width-dependent SERIALIZED FORMAT
- HIP's device `warpSize` is not a plain `int`
- Lane masks are 64-bit on HIP: `__ballot` and `__activemask` RETURN `unsigned long long` ...
- And the mask VALUE must be built in 64 bits, which the destination type does not do for ...
- A 32-bit literal COMBINED with a 64-bit mask is the same class in a second shape, and it ...
- Sweep for THREE patterns, not one, and sweep the vendored subtrees the port ships
- A compat `__ballot_sync` that casts `__ballot()` to `uint32_t` takes the LOW 32 lanes
- A full 64-bit `*_sync` mask in 32-lane group code traps in a HIP debug build on wave64
- Width-32 LOGICAL warp ops are arch-agnostic and fine
- A compat layer for CUDA's `*_sync` warp functions must pin every member to 32 lanes, and ...
- A fixed-cluster warp reduction is wavefront-safe exactly when the cluster size equals the ...
- Pass that logical width on BOTH paths, and prefer the project's own pre-CUDA-9 branch ...
- An over-wide `__shfl*` width does not error, it silently degenerates
- Do not hardcode wave64 GEOMETRY either
- Intra-wave barrier divergence
- A lock-retry loop deadlocks when the lock holder is in the same wavefront
- `cub`/`hipCUB` block collectives race on a 64-thread block without an explicit ...
- A block collective called in a loop without a barrier races only under concurrent GPU load
- `__smid()` can EXCEED `multiProcessorCount` on AMD
- A leader-only write broadcast by shuffle must be initialised on every lane
- An uninitialized value RETURNED from a device function lets clang assume the thread index
- Validate wave-size work on BOTH widths, and where there is no host reference use a ...

**`references/faults-memory-streams.md`** -- Faults: memory, lifetime and streams

- Out-of-bounds reads
- A latent OOB that stopped reproducing is still there, and index math on `size_t` operands ...
- An allocation sized by a different expression than the launch indexes it by
- `atomicInc(p, val)` is a WRAPPING increment, so the bounded-slot idiom ...
- Fresh device allocations are NOT zero on ROCm
- An uninitialised local passed by value lets clang delete a padding-lane guard, and the ...
- Type punning through a word pointer lets clang drop the stores it reads
- Host code reading a `__device__` variable through a pointer gets HIP's host shadow, not ...
- Rule-of-five on resource handles
- A functor returning a reference to a by-value or forwarded parameter
- Querying an event after its recording stream was destroyed is a use-after-free on HIP
- Using a stream after `hipDeviceReset` corrupts the heap (Windows shows it)
- A duplicate `hipFree` in one thread can crash a different thread
- `cudaMemcpy*Async` lifetime and ordering: a pageable 1D source is safe, a pageable 2D or ...
- Do not add a sync after a 1D pageable upload "for HIP"
- A pageable async upload also hides a cross-stream race with no free involved
- An init-time kernel launched on the legacy null stream is NOT ordered against streams ...
- A zero-filling allocator helper races a non-blocking upload the same way
- A plain device-to-device copy is the same race in a less suspicious costume
- A "last block reduces" kernel needs `__threadfence()`, and MI300X is where its absence ...
- A GPU-written host flag can arrive before an earlier small device-to-host copy
- `hipFree` is synchronizing
- Device-side `new[]`/`delete[]` per thread is worth replacing, but it is NOT a known ...
- A copy kind that disagrees with the pointers is `hipErrorInvalidValue`
- `hipMemcpyToSymbol` rejects `hipMemcpyDefault`
- `cudaLaunchHostFunc`/`hipLaunchHostFunc` callback threads must not call ANY runtime API
- A host-function callback must not block either
- HIP after `fork()` crashes in the child
- Creating a hipBLASLt or MIOpen handle while ANY thread captures a graph exits the process
- hipFFT plan creation fails the same way, but quietly
- Bitwise and min/max atomics on `hipMallocManaged` memory are silently dropped on a ...
- Cross-GPU flags in ordinary device memory are never seen by the peer on MI250X
- The DATA a peer reads, not only the flags, needs uncached memory -- and on gfx1100 ...
- Streams share hardware queues on HIP, so a spinning kernel can block another stream's ...

**`references/faults-textures-interop.md`** -- Faults: textures and graphics interop

- Texture pitch alignment is 256 bytes on AMD against 32 on NVIDIA
- A hardware linear-filter texture over an element-read float array is ARCH-SPECIFIC, not a ...
- Gate the hardware-vs-software texture path on a VERIFIED runtime self-test
- A layered `cudaArray` collapses across kernel launches -- use a non-layered 3D array
- gfx94x+ (CDNA3 and newer: MI300-class) disables the HIP image/texture object API at ...
- A project that reads EVERYTHING through texture objects, with address modes, needs more ...
- A global-memory texture replacement must COPY the source, never alias it
- A graphics-interop device pointer dies at the unmap
- Compute-only CDNA has no graphics pipeline
- External memory ports 1:1; external semaphores do not
- HIP VMM memory exported to Vulkan works on Linux
- Match HIP and Vulkan devices by PCI location (Linux) or LUID (Windows), never by UUID

**`references/faults-numerics.md`** -- Faults: floating point and integer arithmetic

- clang for HIP defaults to `-ffp-contract=fast`
- One contraction setting cannot satisfy both kinds of bit-exact test
- When bit-exact tests pin nvcc's results and no global setting is clean, read nvcc's fused ...
- A test that fails deterministically on some AMD targets only, with an error that scales ...
- AMDGPU can fold a float-to-half conversion into the preceding fma, rounding once where ...
- `-ffast-math` is NOT the HIP translation of nvcc's `--use_fast_math`, and reaching for it ...
- `-ffast-math` implies `-ffinite-math-only` under clang; nvcc's `--use_fast_math` does not ...
- `__fsqrt_rn` is not always correctly rounded on gfx90a
- Device `cuda::min`/`cuda::max` NaN-selection can differ from host `std::min`/`std::max`
- An exact float-equality branch fed by approximate division can HANG, not just drift
- A hand-written Jacobi/eigenvalue sweep replacing a library solver must normalize first
- Host `min`/`max` on non-`int` values truncate to int under hipcc
- A HIP workaround in shared host code must not change CUDA's numbers
- HIP's OCP fp8 conversions do not compile for gfx942; E5M2 is the top byte of a half
- Converting a NEGATIVE float to an UNSIGNED integer diverges silently between the back ends
- PTX carry-chain arithmetic (multi-word integers)
- Shift counts of 0 or >= 64 in hand-rolled 128-bit arithmetic
- rocBLAS `gemm_ex` rounds a 16-bit output once per k-chunk when k is long
- gfx11 and gfx12 WMMA drift over long k, in every library, with any output type
- A GEMM workaround covers every entry point that reaches the library, fused epilogues ...

**`references/build-headers-macros.md`** -- Build: headers, macros and source compatibility

- A shared compat header must be host-includable
- Do not name compat macros after the warp functions ROCm declares
- Missing-include errors are usually pre-existing upstream omissions
- cuRAND's device header is host-includable only AFTER the CUDA runtime headers
- Before gating or moving a device-library include that a compat header already has, check ...
- For plain C++ TUs the HIP headers `#define __noinline__ __attribute__((noinline))`
- CUDA error codes HIP lacks still appear in switch labels
- `hipError_t` is `[[nodiscard]]`, so a `-Werror` host build breaks on ignored results CUDA ...
- Never name the pointee struct of an opaque CUDA handle
- `char` vs `signed char` vector base types differ
- A torch header the compat layer already aliases for can grow new symbols: alias the whole ...
- A macro that nvcc or a CUDA header defines and HIP does not reads as 0, and every gate on ...
- `__HIP_PLATFORM_AMD__` is undefined until `hip/hip_runtime.h` has been included in that TU
- Defining `__CUDA_ARCH__` for the HIP device pass also feeds every numeric gate
- A code generator's device-pass guards live in strings, and fail only when a model uses ...
- Audit every macro class that gates a dtype or arch path, not just `__CUDA_ARCH__`
- `CUDART_VERSION` is undefined under HIP, so every `#if CUDART_VERSION >= N` guard ...
- Project code that splits on `__CUDACC__` means "compiled by the device compiler", and HIP ...
- Upstream CCCL 3.0 compat shims are CUDA-only by construction: keep them out of the HIP ...
- nvcc accepts a split launch chevron, `kernel << <grid, block >> >(args)`; clang does not
- Three CUDA spellings that do not resolve under HIP, all cheap to guard
- Tile shuffles of pointers and enums need `__shfl` overloads, declared before the ...
- Count cooperative-groups call sites per feature before calling them a blocker
- HIP's bf16 header has no `make_bfloat162` and no `__floats2bfloat162_rn`
- HIP has no `half2` overload of `__hmax2`
- A `__shared__` array of a struct that has default member initializers is a hard error on ...
- MSVC-only upstreams accept code that clang and gcc reject
- Eigen DECOMPOSITIONS called from `__device__` code do not survive the move to hipcc, even ...
- `__ldg` is usually a non-problem on HIP; check before working around it
- An uninitialised header-only submodule fails as a HIP error, not as a missing file
- clang checks kernel bodies in the HIP host pass; nvcc does not
- The CUDA Samples `helper_math.h` makes vector arithmetic ambiguous under HIP
- HIP's `blockIdx`/`threadIdx` do not convert to `uint3`, and a `uint3` does not convert to ...
- HIP's host-side `__half` needs the HIP compiler, and `HIP_vector_type`'s operators hijack ...
- A helper whose name the standard library also declares must NOT be guarded on ...
- Extend a CUDA guard for HIP; never re-key it
- A ROCm-only branch inside a shared template still COMPILES on the CUDA path
- "Portable" is a claim to verify against nvcc, not a property of how the edit reads
- C++17 `std::clamp` and friends work in HIP device code but not under nvcc
- An arch-specific fix keyed on the OS or the ROCm version can BREAK an already-validated ...
- rocRAND's device headers need C++14 from ROCm 10.1
- HIP + C++23 on an older toolchain: `isfinite cannot overload` (and how to stay ...

**`references/build-cmake-link.md`** -- Build: CMake, device linking and dependencies

- A force-included compat header creates no build dependency edge
- Use the CUDA build's host compiler for host C++ in a large app
- The vcpkg toolchain already overrides `add_library`/`add_executable`
- An unset CMake build type runs every HIP kernel unoptimized
- `CMAKE_COMPILE_WARNING_AS_ERROR` outranks `-Wno-error`
- Never mark the host `.cpp` files `LANGUAGE HIP` to dodge the flag leak
- A `__device__` body in its own `.cu` has two answers on AMD, and the choice is about ...
- `-fgpu-rdc` changes the consumer contract, so the project's downstream docs owe a line ...
- Cross-TU `__device__` calls ARE inlined on AMD under `-fgpu-rdc`
- Moving the bodies into a host-reachable header UNGUARDED silently breaks the CUDA build
- No AMD build, review round or GPU validation can detect this class
- `-fgpu-rdc` on the compile line alone trades the first link error for a more confusing ...
- `-flto` on the COMPILE line does not survive a `-fgpu-rdc` device link
- Settle any "which flags does this build actually use" question by reading the generated ...
- One OpenMP runtime per binary; which one follows from who compiles the files that use ...
- OpenMP in a mixed g++/HIP library: one runtime only
- CMake `LANGUAGE HIP` on host `.cpp` files drops OpenMP
- A HIP-linked shared library, or a host-linked executable, gets no RUNPATH to the ROCm lib ...
- TheRock's torch wheels cannot be linked from C++ as installed: their RUNPATH points into ...
- A library that interposes the HIP registration entry points must bind them locally
- With the ROCm clang as the host compiler, a found system libunwind links but does not load
- LibTorch's CMake package adds `-D__HIP_NO_HALF_OPERATORS__=1 ...
- ld.lld rejects an underlinked shared library that GNU ld accepted
- FindCUDA leaves NOTFOUND variables behind on a machine without CUDA
- Build protobuf (and abseil) with the same compiler as the code that includes its ...
- On a multi-GPU host a rapids-cmake HIP dependency fails configure with `mismatch between ...
- A source-built HIP library and a PyTorch wheel in one process must load the same ROCm ...
- A vcpkg-manifest project needs `autoconf-archive` on the build machine, and a full vcpkg ...
- LibTorch on the include path shadows the system `fmt`, and spdlog stops compiling
- Submodules pinned to commits that do not exist
- A local-package option must be a variable only the user sets, not `<Pkg>_DIR`
- Install and consume the port, not just build it
- A ROCm wheel's runtime dependency on the SDK, with setuptools and a static ...
- Every optional target the project can build, not just the one you build
- A cached build-script decision from a wrong env var survives fixing the env var
- An nvcc Makefile's link line does not carry over to hipcc verbatim

**`references/windows.md`** -- Windows

- Use TheRock ROCm, not the Windows HIP SDK
- TheRock's devel wheel can ship with hipCUB/rocThrust headers missing
- A torch wheel needs the matching `[device-<gfx>]` extra
- On Windows, Triton comes from the `triton-windows` package
- A dev package neither TheRock nor vcpkg ships
- An all-clang Windows HIP build hides three CMake/export faults that MSVC builds never hit
- CMake's HIP language on Windows needs ONE clang driver for every language, and ROCm's ...
- Keep one clang driver across every CMake language, and check the CMake version
- `find_package(Torch)` in a CMake HIP project needs the architecture and the ROCm root
- CRT flags, compiler-rt builtins and Git Bash flag mangling in a CMake HIP build
- Bundled third-party libraries that break under the clang toolchain: take them from vcpkg
- TheRock clang 23 can hang in x86 register allocation
- Linking HIP through CMake: what each compiler arrangement needs
- A ROCm library linked by name may have no `.lib`
- A fetched ROCm-DS dependency has to be built static
- A C++ LibTorch consumer must force the torch_hip.dll import
- Stage the ROCm runtime next to the .exe, kpack included
- A null top frame above a vendored DLL means a runtime library was never loaded
- `hipcc.exe` and hiprtc traps for setuptools/Python builds
- A buffer that crosses into Python and gets released with `free()` must come from `malloc`
- Linux-only host APIs and LLP64 truncation inside third-party C APIs
- A `core.autocrlf` mismatch can break `git apply` of a committed patch
- A `std::stringstream` can fail on a multi-GB payload on Windows only
- Kernels on different streams run one after another
- Late submission exposes missing cross-stream ordering
- Graph relaunch races its own memory-free nodes
- VMM address ranges that were mapped are never reused
- A crash trace without a PDB names the wrong function
- A CPU crash that Linux never shows is often a heap over-read
- A hang can be one kernel that never finishes

**`references/libraries.md`** -- Libraries: substitutions and gaps

- Library swaps
- Negative entries for that table
- A swapped library may be less defensive about bad arguments than the CUDA one
- heFFTe's rocFFT backend does not accept hipFFT complex types
- Choosing the NVIDIA or AMD build of a library at run time
- A scalar's WIDTH follows the compute/scale type, not the matrix type
- `hipblasGemmEx` rejects `HIPBLAS_COMPUTE_32F_FAST_16F` for FP16 inputs
- hipBLASLt epilogues are not cuBLASLt's, per type and per format
- PyTorch's ROCm MX GEMM wants unswizzled scales, M, N and K in multiples of 32, and no bias
- Probe `torch._scaled_mm`'s fp8 types once per device instead of listing architectures
- Generic GEMM on CK device operations
- Composable Kernel older than tag rocm-7.2.0 races on gfx11 LDS barriers
- Fused FP16 attention (cuDNN SDPA, flash attention): MIOpen has no path, CK's FMHA does
- PyTorch's fused SDPA on gfx1100 fails at head dims 257 to 512 with an unrelated HIP error
- flash-attn 2.8.3's Triton AMD backward reads `dout` with `out`'s strides
- rocBLAS `gemv` over more than 65535 batches returns wrong entries, and `cupy.einsum` ...
- On gfx12, rocBLAS complex GEMM with n > 2^21 leaves columns unwritten
- Some MIOpen convolution solvers are not run-to-run reproducible; establish the noise ...
- A "find the best algorithm" call WRITES to the tensor named as its output
- MIOpen honours only a unit alpha and a zero beta for 2D convolution, pooling and bias
- MIOpen runs a convolution only on a handle where Find ran for it
- MIOpen 3.6 (ROCm 10.1) lacks several cuDNN operations outright
- Two ROCm library headers can collide
- A zero workspace size does not mean "skip the operation"
- hipSPARSE has no uniform half-precision SpMM
- Library calls inside HIP graph capture
- A device pointer passed while the handle is in host pointer mode crashes only on some ...
- rocThrust's `thrust/system/cuda/error.h` is CUDA-only
- Prefer the specific hipCUB algorithm header over the `<hipcub/hipcub.hpp>` umbrella
- rocPRIM never asks for less than 4 bytes of temp storage
- hipCUB's segmented `ArgMax` returns an out-of-range key for a segment of NaNs
- `<complex.h>` defines the macro `I`, which rocPRIM uses as a template parameter name
- Three rocPRIM radix sort defects give a wrong order or a broken permutation where CUB is ...
- rocPRIM radix sort corrupts float keys that mix signaling and quiet NaNs
- rocPRIM radix sort of 64-bit keys is wrong when `end_bit` is 64 and `begin_bit` is not 0
- Choosing the workaround for that bit-range defect: narrowing the range is only safe when ...
- libcu++ -> libhipcxx, and two atomic defects in it
- torchaudio's RNN-T loss and CTC forced alignment run only on the CPU in ROCm builds
- CuPy-ROCm gaps a CuPy-based port meets at run time
- Code that reads cuRAND's device generator state internals needs a rocRAND translation ...
- NVTX -> roctx
- GPU power (the NVML replacement) is AMD SMI
- RAPIDS libraries: hipDF, hipVS and hipMM as C++ wheels
- RMM -> hipMM (and check ROCm-DS before stubbing any RAPIDS library)
- RCCL collectives need the whole multi-GPU host at once
- rocJPEG writes past the row pitch
- Fall back to the CPU only for JPEGs rocJPEG cannot decode, and say so
- A JPEG that ends before its scan hangs the VCN JPEG engine
- NVDEC through FFmpeg becomes FFmpeg demux plus rocDecode
- rocDecode's capability query is not a support check by itself
- A rocDecode success status does not mean right pixels: check a reference clip once per ...
- The software fallback for AV1 needs libdav1d

**`references/platform-limits.md`** -- Platform limits and other build classes

- AMD compute-capability values COLLIDE with NVIDIA arch numbers
- `rocminfo` prints pool sizes as `N (0xH) KB`, with a space
- A query HIP reports as unsupported leaves a sticky error that the NEXT check reports
- `clock64()` is a 20-bit wrapping counter on RDNA3, so a cycle-bounded spin, delay or ...
- On RDNA, a nonzero occupancy does not mean the dynamic shared memory fits
- Dynamic shared memory is not 16-byte aligned on AMDGPU unless declared so
- Launch geometry computed from device properties can overshoot a bound sized for NVIDIA
- `__launch_bounds__(1024)` changes nothing on HIP and something on CUDA
- A kernel launched with fewer than 1024 threads needs `__launch_bounds__` on HIP to get ...
- A host-side launch heuristic keyed on the register count can break a kernel whose block ...
- Kernels that opt in to more than 64 KB of shared memory per block cannot launch on most ...
- A refused shared-memory opt-in trips the last-error check placed before the project's own ...
- AMD's 64 KB LDS breaks "half the per-SM shared memory" tile heuristics, and the ...
- gfx12 work-group IDs y and z are 16 bits: a grid y or z above 65536 is rejected or ...
- HIP limits each grid dimension to fewer than 2^32 work-items, not to 2^31 - 1 blocks
- Runtime PTX plus the CUDA Driver API is a THIRD build class
- NVRTC (CUDA C++ generated at run time) ports to hipRTC
- Runtime-compiled kernels inherit nothing from the host build
- Triton on AMD accepts `input_precision='tf32'` only on gfx942
- Triton `num_warps` above 16 cannot launch on a 64-wide wavefront
- CUDA-namespaced libdevice calls and hard-coded FP8 types
- An integrated APU reports the whole shared memory pool as device memory
- An integrated APU can crash the device linker under build parallelism
- A low-CU integrated GPU can outrun a fixed upstream test timeout without any fault

**`references/diagnosis.md`** -- Diagnosing a suspected AMD fault

- One architecture gets wrong numbers while the others pass
- Rule out a stale tree and the toolchain version before accepting "arch-specific"
- A "data-dependent, later-data-corrupts-earlier, per-tile" corruption signature is the ...
- Triangulate single- against double-precision before blaming the wavefront
- Test "it reads uninitialized device memory" with a fill-pattern differential, not an audit
- To close an "it might silently fall back" risk, trace it; then bound it from the code ...
- `torch.cuda.device_count()` returns 0 while `torch.cuda.is_available()` is True: a ...
- A once-per-suite host crash that stress reruns never reproduce: widen the suspected race ...
- ROCm 10.1 `AMD_LOG_MASK` takes decimal only, and graph replays are invisible to it
- Before blaming the code under test for a wrong result, save the output, the inputs as ...
- When a wrong result survives a `hipDeviceSynchronize` on every device, probe the HIP copy ...
- A kernel that hangs only in a Debug build: rebuild Debug at `-O2` before blaming an ...
- An intermittent crash seen only in the GPU build needs a CPU-only build control, run as ...
- A probe that contradicts a passing test, or that newly fails on a second arch, is wrong ...
- A cascading wall of SegFault/OOM/"no kernel image available" partway through a long ...
- A "fixed seed" run that relies on C `rand()` is not reproducible on some ROCm builds: the ...
- Seeded `cupy.random` inputs differ between CUDA and ROCm
- A hang that depends on inlining: long branches clobber the return address
- Proving a real miscompile: catch the faulting instruction, then flip one backend pass
- Before removing a compiler workaround, rerun the whole suite on the new release
- An instruction-selection abort: change how the value is produced
- A fault inside a vendored JIT entry point: rebuild-with-a-forced-throw beats a debugger
- scikit-learn's KMeans can crash a Python test on a machine with very many cores
- Failures that appear only when the whole suite runs in one process
- A sync-bound GPU test that passes alone can time out under `ctest -jN`
- A free-memory leak check measures the whole GPU, not your process
- A free-memory-delta test can fail once per process on gfx90a
- A fixed "leak" after the first launch of a spilling kernel is the scratch allocation ...
- O_DIRECT I/O fails with EINVAL on a container's overlay filesystem
- Two GPUs visible to one process can crash the runtime

**`references/performance.md`** -- Performance: what a correct port can still lose

- Profile the hot kernels and read their ISA before blaming tile tables or wavefront size
- Scalar emulation of a packed intrinsic (`__hmax2` as two `__hmax`) cost 15x or more
- Tile and block-size tables are second order, and a project's own autotuner can pick a ...
- Compare kernel variants warmed up, interleaved, against a noise floor
- What did not matter: RDC layout and maskless shuffles

**`references/validation.md`** -- Validation

- `-mwavefrontsize64` on an RDNA card checks lane logic, and cannot judge an application
- Compile and link the CUDA path with nvcc before claiming the CUDA build is unchanged
- A CUDA toolkit for the check on a machine with no NVIDIA GPU
- Diff the PTX to prove the CUDA path is byte-restored, not merely compiling
- With a conda toolkit, legacy `find_package(CUDA)` does not find the headers, and many ...
- The newest toolkit is not the whole check when the port touches `__half` in host code
- A legacy `FindCUDA` project fails the CUDA configure under CMake 3.27 or newer; that is ...
- A dependency that FetchContent-clones with full history can eat the whole time budget of ...
- A torch extension needs a CUDA-built torch for this check, not just the toolkit
- `nvcc fatal: Unsupported gpu architecture 'compute_35'` at the first ...
- CUDA gate for torch-extension ports: the host's installed PyTorch itself can be the wall
- A Windows machine can syntax-check the Linux HIP host pass of a `.cu` without a Linux box
- Proving two builds carry the same device code: build both in the SAME directory
- A passing assertion COUNT is not a list of what was tested
- A green result does not prove a GPU test ran: count kernel dispatches
- An onnxruntime GPU provider that fails at session creation falls back to the CPU silently
- Read what the project's own cross-backend checker actually compares
- A project with no test suite: author the harness, then ship it WITH the port
- A GPU test the port adds must skip on a machine with no device, and an in-`main` check ...
- Golden-differential validation for a behavior-preserving rewrite

**`references/linux-host-setup.md`** -- Linux host setup for graphics and interop tests

- Getting a real hardware GL context on a headless RDNA host
- Headless Vulkan on a Linux GPU host: Xvfb plus Mesa's software WSI
- Driving a compute demo whose only entry point is an interactive GL window
- A GPU test that hangs only under `ctest -jN` is usually the software GL stack
- Headless EGL on an Instinct host lists every GPU before the software device
- HIP and Vulkan in one process can oops the kernel on an old amdgpu-dkms
<!-- index:end -->
