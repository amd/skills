---
name: moat-cuda-to-rocm
description: Port a CUDA project to ROCm/HIP for AMD GPUs. Strategy selection, the AMD-strict-where-CUDA-is-lenient fault classes, wavefront-size portability, library substitutions, and the validation bar. Use when planning, implementing, reviewing or validating a CUDA-to-HIP port, or when a HIP build compiles but misbehaves at runtime.
---

# CUDA to ROCm/HIP

Accumulated across many real ports. This file is the index and the decision-critical parts;
everything else is in `references/`, loaded only when you need it.

**Most porting bugs are not in symbol names.** Hipify handles the renaming. The bugs are
semantic differences where AMD is strict and CUDA is lenient, and most surface at runtime
on a build that compiled cleanly -- though some, like a 32-bit `*_sync` mask, fail to
compile even after hipify runs clean. Read the fault-class index below before concluding a
port works.

## Naming: ROCm or HIP?

They name different things and reviewers notice.

- **HIP** is the programming model: the kernel dialect plus the `hipXxx` runtime API. The
  analogue of CUDA C++ and the CUDA runtime.
- **ROCm** is the platform: compiler, runtime, driver, and the roc*/hip* libraries. The
  analogue of the CUDA Toolkit.

So the CODE port is "to HIP" (hipify, the compat header, runtime symbols), while the
TARGET, build flag and libraries are "ROCm" (`USE_HIP`/`USE_ROCM`, "ROCm 7.2.1",
cuFFT -> hipFFT). A language+runtime port is precisely "a HIP port targeting ROCm". Never
call the platform "HIP" or the kernel dialect "ROCm". Details: `references/naming.md`.

## Which strategy

Classify the build first -- implementing the wrong strategy correctly is still wrong.

| build | strategy | shape |
|---|---|---|
| pure CMake | **A** (preferred) | one `cuda_to_hip.h` compat header, `enable_language(HIP)`, `set_source_files_properties(... LANGUAGE HIP)`; sources keep CUDA spelling |
| pytorch extension | **B** | rely on torch's build-time hipify; fix only what it cannot |
| anything else | neither | driver-API, runtime PTX, Go/cgo, qmake and codegen builds exist -- see the runtime-PTX fault class |

**How to tell:** look for `find_package(Torch)`, `torch.utils.cpp_extension`,
`CUDAExtension`, or a torch dependency in `setup.py`/`pyproject.toml`. If any is present it
is a pytorch extension -- Strategy B. Otherwise treat it as a pure CMake (or Makefile, or
meson) project -- Strategy A. The build system does not pick the strategy; what the sources
look like does. lc0 is meson and ports Strategy-A-shaped -- it just means writing by hand
what `enable_language(HIP)` would have given you, and `references/strategy-a-cmake.md` says
what.

Strategy A is the minimal-footprint model: the compat header is a no-op on NVIDIA, so the
CUDA build is untouched. Where the project includes CUDA headers by name, prefer the
**shim-header method** -- shadow headers on the HIP include path only -- which leaves the
source tree byte-identical to upstream and produces a far more mergeable diff. Full recipes: `references/strategy-a-cmake.md`,
`references/strategy-b-torch.md`.

## Before porting

Check whether an AMD port already exists, and whether it is authoritative. The deciding
axis is authoritativeness, not existence -- an AMD-official work-in-progress shifts the
value to validating and improving it, while a one-off community fork is a hint, never code
to inherit. Cheapest high-signal check first: grep the upstream's own docs for
`amd|rocm|hip|gfx[0-9]`, because reference repos routinely link platform forks.
`references/assess-existing-support.md`.

Some CUDA has no HIP analogue to translate INTO, so hipify succeeds at doing nothing and the port looks small. CUTLASS reimplements against Composable Kernel; CuTe has a real structural correspondence in AMD's FlyDSL, whose layout algebra is modelled on it. That is a reimplementation with a defined target, not the same verdict as "no target exists" -- and both get written off as "not portable" unless you say which. `references/no-hip-equivalent.md`.

## Fault classes -- the index

Scan this list against what you are doing. If any line could apply, open
`references/fault-classes.md` for the full entry with evidence and fix.

**Wavefront and warp semantics**
- Warp size: NVIDIA is always 32; AMD is 64 on CDNA (gfx9xx) and 32 on RDNA. Never hardcode.
- Warp-derived array sizing: quantities scaling WITH `warpSize` need the 64 upper bound, quantities scaling as `blockDim/warpSize` need the 32 LOWER bound. Getting it backwards writes out of bounds.
- HIP's `*_sync` warp functions reject a 32-bit mask at compile time; CUDA's `0xffffffff` needs a 64-bit spelling on HIP.
- A compat `__ballot_sync` that casts `__ballot()` to `uint32_t` returns the wrong 32 lanes on wave64.
- An over-wide `__shfl*` width silently clamps to the physical wavefront instead of erroring; the symptom is a metric shift, not a crash.
- A full 64-bit `*_sync` mask in 32-lane group code traps in a HIP debug build on wave64 when the halves diverge; pass the group mask.
- Intra-wave barrier divergence: a per-row early return before `__syncthreads()` is benign on CUDA, faults on wave64.
- `cub`/`hipCUB` block-collective `TempStorage` reuse races on a 64-thread block without an explicit `__syncthreads`.
- `__smid()` can EXCEED `multiProcessorCount` on AMD, unlike NVIDIA.
- HIP's device `warpSize` is not an `int`; cast it at any `printf` vararg.

**Memory and lifetime**
- Out-of-bounds reads: CUDA often tolerates one element past an allocation; AMD faults.
- Fresh device allocations are NOT zeroed on ROCm, where CUDA's allocator often hands back zeroed pages. Whenever you port or review code, trace every allocation whose first use is a read-modify-write -- an `atomicAdd` counter, an accumulator, histogram bins, a `+=` output -- and confirm a memset or init kernel runs first on the same stream. Report a missing one even when the rest of the review is about something else.
- `cudaMemcpy*Async` from a soon-freed pageable host buffer: CUDA stages synchronously, HIP does not. The same hides an unsynchronized upload on one stream feeding a kernel on another.
- Bitwise/min/max atomics on `hipMallocManaged` memory are silently dropped on discrete xnack- cards (PCIe AtomicOps); whether a build is hit depends on optimization level, so check the app's ISA.
- Rule-of-five on resource handles: CUDA tolerates double-destroy of a texture/stream/event; AMD does not.
- `hipEventQuery`/`hipEventSynchronize` dereference the event's recording stream: querying an event after that stream is destroyed (legal on CUDA) hangs or crashes intermittently. Keep destroyed streams alive: route stream create/destroy through a sync-park-reuse pool in the compat header, not per-holder sweeps.
- A functor returning a reference to a by-value parameter (lifetime UB) is tolerated by nvcc, not by clang/HIP.
- `hipLaunchHostFunc` callback threads must not call ANY runtime API.
- A `hipLaunchHostFunc` callback must not block either: ROCr runs it on the async-events thread, so a callback waiting on a host gate stalls every later event on the device (a multi-stream test hung on gfx1100).
- HIP after `fork()` crashes in the child (segfault in hipEventQuery); a test that forks and then touches the GPU fails on ROCm.
- A query HIP reports as unsupported (`cudaLimitPrintfFifoSize` -> hipErrorUnsupportedLimit) leaves a sticky last error that the NEXT launch check reports; clear it under the HIP branch.

**Textures**
- Texture pitch alignment: AMD requires 256-byte row pitch for pitched 2D binds (32 on NVIDIA).
- Layered `cudaArray` collapses across kernel launches -- use a non-layered 3D array (confirmed gfx90a bug).
- Hardware linear-filter texture over an element-read float array is arch-specific, not a blanket AMD limitation.
- Gate hardware-vs-software texture paths on a VERIFIED runtime self-test, never on creation success or `_WIN32`.

**Floating point**
- `__fsqrt_rn` is not always correctly rounded on gfx90a or gfx1100 (1 ULP high); CUDA's is IEEE-correct. Map it to `sqrtf` when a test compares against host sqrt.
- Without `__HIP_DEFINE_EXTENDED_HOST_MIN_MAX__` HIP's HOST `min`/`max` take int, so `min(float, float)` in `__host__ __device__` code silently truncates on the host (whole-pixel camera errors); define it for the HIP build.
- clang(HIP) defaults to `-ffp-contract=fast` and forms FMAs ACROSS expressions; nvcc contracts expression-only.
- Exact float-equality branches fed by `__fdividef` produce out-of-range indices.
- HIP device `cuda::min`/`max` NaN-selection can differ from host `std::min`/`max`.
- HIP host `min`/`max` in `.cu` files exist only for `(int, int)`: float arguments silently truncate. Fix the call sites with `fmin`/`fmax`; the macro is ROCm >= 7.2 only and defaults off.

**Headers, includes and build**
- A shared compat header must be host-includable: gate device-only includes behind `__CUDACC__`/`__HIPCC__`, or CUB leaks into host TUs. Hit independently by two projects.
- `__HIP_PLATFORM_AMD__` is undefined until `hip_runtime.h` is included; a gate placed before it silently picks the CUDA branch.
- Missing includes in a HIP port are usually pre-existing upstream omissions unmasked by the narrower include graph.
- A `-include`d compat header creates no dependency edge: wipe objects after editing it or you validate stale code.
- clang type-checks kernel bodies in the HIP host pass, where `__CUDA_ARCH__` is unset: a `__CUDA_ARCH__`-split debug macro calling host code fails only in Debug.
- The CUDA Samples `helper_math.h` operators tie with `HIP_vector_type`'s own: guard its arithmetic block off on HIP, wholesale.

**Types, dispatch and platform limits**
- Never name the pointee struct of an opaque handle (`CUstream_st`); use `std::remove_pointer_t<cudaStream_t>`.
- CUPTI has no ROCm analogue; ck_tile fmha ships headers but no instance library (generate the instances; MIOpen's MHA has no FP16 path).
- CUTLASS-style GEMMs map to CK device ops (XDL on CDNA, WMMA on RDNA, DL for RDNA FP32 with padded copies).
- cuDF/cuVS/RMM: hipDF, hipVS, hipMM ship as C++ wheels on AMD's package index, with a few API gaps.
- hipSPARSE breaks HIP graph capture where cuSPARSE does not (destroying a CSR SpMat syncs, default-algorithm analysis, size-0 buffers).
- Pick a library's NVIDIA or AMD build per device, never by starting every backend; a HIP port that keeps the soname (hipVS, hipDF) allows one per process.
- NVML -> AMD SMI, keyed by PCI address, not device index.
- `clock64()` is a 20-bit wrapping counter on RDNA3; cycle-bounded spins and delays end early. Use `wall_clock64()`.
- `char` vs `signed char` vector base types differ between HIP and CUDA.
- AMD compute-capability values COLLIDE with NVIDIA arch numbers, so CC-keyed dispatch picks the wrong path.
- Library swaps: cuBLAS -> hipBLAS, cuFFT -> hipFFT, cuRAND -> hipRAND, cuSPARSE -> hipSPARSE, cuDNN -> MIOpen, Thrust/CUB -> rocThrust/hipCUB. Handle types and v2-enum signatures differ.
- nvJPEG/nvImageCodec -> rocJPEG for JPEG decode only (VCN; baseline SOF0 only: no SOF1 or progressive, nothing under 64x64, no encoder, no JPEG 2000); the rest needs a named CPU path, chosen from the frame header, not a catch-all. A JPEG with no scan hangs the VCN engine: validate markers on the host first. NVDEC -> FFmpeg demux + rocDecode's parser and decoder (no FFmpeg hwaccel); gate on profile as well as `rocDecGetDecoderCaps`, which has no profile and lets unsupported H.264 profiles decode wrong, and whose size limits are coded sizes (round the display size up to the block size); a success status can still carry wrong pixels on some GPU and VA driver pairs, so decode a reference clip per codec once per process before trusting it; the CPU path needs libdav1d for AV1. NVENC has no ROCm equivalent: name the software encoder path. See fault-classes "Vulkan interop and image codecs".
- Runtime PTX + CUDA Driver API is a third build class beyond A and B.
- MSVC-only upstreams accept code gcc/clang reject; the HIP build is a stricter compiler.
- No graphics pipeline on compute-only CDNA: OpenGL/Vulkan-interop apps build but cannot run on gfx90a.
- Vulkan timeline semaphores do not import into HIP on either OS, and the Windows OpaqueWin32 import can silently miss a wait: order on the host.
- An arch-specific fix keyed on OS or ROCm version can BREAK an already-validated arch.

Two diagnostic methods in `references/validation.md` are worth knowing before you escalate a
suspected AMD fault: a per-tile "later data corrupts earlier" signature usually means the
reproducer's input dtype is wrong, and SP-vs-DP triangulation distinguishes FP reassociation
from a genuine wavefront bug. Both were learned the expensive way.

## Portability rule for shared code

Any fix to code that is not arch-guarded must be correct on **both** wavefront sizes. A
per-arch hack that fixes wave32 and regresses wave64 makes the archs ping-pong, and each
"fix" then costs a full re-test on every architecture. Prefer a runtime `warpSize` query on the
host and a compile-time constant in device code.

## Validation bar

A real-GPU test run is the only evidence that counts. Lint is not validation, and a
CPU-only build proves compilation and nothing else. Cover both wavefront widths and each
OS the project supports: at least one wave64 (CDNA) GPU, one wave32 (RDNA) GPU, and Windows
when the project builds there. Also compile the
CUDA path with nvcc as a no-regression gate: an additive ROCm port must leave the CUDA
build a pure passthrough. `references/validation.md`.

## Keeping an open PR mergeable

Upstream moves while a PR is in review. Sync with a MERGE of upstream's default branch,
never a rebase on a published branch, and treat the clean auto-merge as the risk: upstream
writes new code against the surfaces the port multiplexed (per-vendor backends, platform
predicates), so it lands the NVIDIA half only. `references/upstream-sync.md`.

## Writing it up

Commit messages, PR bodies and code comments go to upstream maintainers who do not know
your team's vocabulary. Before pushing, check the whole branch -- not just the commit you
wrote last -- for in-house terms: internal stage or state names, tool names, and this
skill's own labels such as "Strategy A" or "fault class". A term list with a replacement
for each is worth keeping and running mechanically.

A term list does not catch a comment that explains a hazard by naming ANOTHER ported
project as shorthand for it. arrayfire carried eleven such comments; the six saying "fault
class" were caught by a term check and five saying "the amgcl void*-aliasing fix" or "the
gsplat target-attribute rule" were not. To an upstream reader both are dangling
references to a codebase they have never seen. State the hazard itself -- "hipSPARSE typedefs
several descriptor types to the same void*, so a wrapper keyed on the handle type cannot
separate them" -- and before pushing, grep new comments for the names of other projects you
have ported.
