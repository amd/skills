# Strategy A: pure CMake (preferred, minimal footprint)

<!-- toc:begin -->
Contents:

- Large libraries: `--offload-compress` to fit the host image under the 2 GiB x86-64 relocation reach
- The shim-header method: a port with zero source edits
- Vendored CUDA-samples helpers: keep them off the AMD build entirely
- Build hygiene
<!-- toc:end -->

Goal: only `.cu`/`.hip` translation units see the HIP toolchain; host C++ is untouched; the diff stays small.

1. Add one CUDA-to-HIP compat header (e.g. `src/.../cuda_to_hip.h`). On ROCm it aliases the CUDA spellings the project uses to their HIP equivalents and includes the HIP runtime; on NVIDIA it is a no-op include of the CUDA runtime. Everywhere else keep the plain CUDA spelling (`cudaXxx`, `curand*`, `cublas*`). This header is the only file that knows about HIP.

       #pragma once
       #if defined(USE_HIP) || defined(__HIP__)
       #include <hip/hip_runtime.h>
       #define cudaMalloc        hipMalloc
       #define cudaFree          hipFree
       #define cudaMemcpy        hipMemcpy
       #define cudaStream_t      hipStream_t
       #define cudaError_t       hipError_t
       #define cudaSuccess       hipSuccess
       // ... only the symbols the project actually uses
       #else
       #include <cuda_runtime.h>
       #endif

   Use hipify's mapping tables as the authoritative cuda->hip name source when adding aliases: `torch/utils/hipify/cuda_to_hip_mappings.py` in a pytorch checkout lists 3000+ symbol mappings.

2. In CMake, gate the language on a HIP option instead of renaming files. The `HIP` language needs CMake 3.21 or newer (3.29 on Windows, see windows.md):

       option(USE_HIP "Build with HIP for AMD GPUs" OFF)
       if(USE_HIP)
         # Do NOT pin CMAKE_HIP_ARCHITECTURES to a literal arch (e.g. gfx90a) here.
         # enable_language(HIP) already does the right thing: it honors an explicit
         # -DCMAKE_HIP_ARCHITECTURES, else auto-detects the host GPU(s) via
         # rocm_agent_enumerator, else errors (FATAL_ERROR "Failed to find a default
         # HIP architecture"). A pin BEFORE this call preempts that auto-detect, so a
         # non-gfx90a user who omits -D silently builds gfx90a objects that fail to
         # load at runtime ("no kernel image"); a pin AFTER it is dead code (the call
         # already resolved the arch or errored). Either way another host still
         # overrides with -DCMAKE_HIP_ARCHITECTURES=<arch> and never edits this file.
         enable_language(HIP)
         set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE HIP)
         set_target_properties(<tgt> PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
       else()
         enable_language(CUDA)
       endif()

   Marking the existing `.cu` files `LANGUAGE HIP` keeps the diff minimal and the NVIDIA build intact. Configure with `-DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a` (add `-DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++` if CMake does not find it; ROCm releases older than the `lib/llvm` layout have it at `/opt/rocm/llvm/bin/clang++`). Because the target reads `${CMAKE_HIP_ARCHITECTURES}`, one commit builds for any AMD target with only `-DCMAKE_HIP_ARCHITECTURES=<arch>` and no source change, so validating on another architecture needs no commit of its own. Pass every architecture you can test at planning time so the first bring-up is right.

   Test the auto-detect path once by configuring with no `-DCMAKE_HIP_ARCHITECTURES`. A target created before `enable_language(HIP)` runs gets its `HIP_ARCHITECTURES` property initialized empty. Any executable that links a library with HIP objects links as HIP, so configure stops with `HIP_ARCHITECTURES is empty for target <exe>`, while explicit-arch builds still work and hide the problem. Detection also returns one entry per GPU (`gfx942;gfx942;...`). The fix: after `enable_language(HIP)`, run `list(REMOVE_DUPLICATES CMAKE_HIP_ARCHITECTURES)`, read `get_property(... DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)`, and set `HIP_ARCHITECTURES` on every target in that list (qrack: tests and examples were created before `cmake/CUDA.cmake` was included).

   **Outside CMake you write that rule yourself, and the README has to say what you wrote.**
   A meson/Makefile project has no `enable_language(HIP)` to get this right for you, so the
   tempting shapes -- `option('amd_gfx', value: 'gfx90a')`, or `if gfx == ''` then
   `gfx = 'gfx90a'` -- reintroduce exactly the silent default the CMake note above warns
   about: a user on another card builds objects that load nowhere. Give the option an EMPTY
   default, autodetect by taking the first non-`gfx000` line of `rocm_agent_enumerator`, and
   hard-`error()` naming the flag to set when detection yields nothing. Two traps when you
   do: `run_command()` on a `find_program()` result needs meson >= 1.2.0, so guard the
   autodetect branch on the meson version and remember the error path is what pre-1.2.0
   users hit; and you cannot rehearse that error path by hiding the GPU, because
   `rocm_agent_enumerator` reads the sysfs topology and ignores `HIP_VISIBLE_DEVICES` /
   `ROCR_VISIBLE_DEVICES` (and a `find_program` fallback to the absolute
   `/opt/rocm/bin/rocm_agent_enumerator` defeats hiding it from `PATH`). Then re-read the
   README block you wrote about the flag: lc0 dropped its gfx90a fallback in a review round
   and left the README claiming "defaulting to gfx90a" through two further rounds until a
   documentation check during validation caught it. Any review round that changes what a
   build option DOES carries a documentation edit with it.

   **Document a clean-environment-safe build: pass `-DCMAKE_PREFIX_PATH=/opt/rocm`.** When the project's CMake calls `find_package(hip)` / `find_package(hipcub)` / `find_package(rocThrust)` (most Strategy A ports do, to link `hip::device`/`hip::host`), CMake locates ROCm's config packages by deriving the `/opt/rocm` prefix from `/opt/rocm/bin` being on `PATH`. A developer machine usually has ROCm on `PATH`, so the build "just works" there; a clean ROCm container (e.g. `rocm/dev-ubuntu`) does not, and the exact documented command fails with `hip_DIR-NOTFOUND` (or hipcub/rocThrust NOTFOUND). Setting `-DCMAKE_HIP_COMPILER` by absolute path is NOT enough (it points the HIP language at the compiler but does not seed the package search), and `ROCM_PATH=/opt/rocm` alone is NOT enough either (both proven experimentally). Always pass `-DCMAKE_PREFIX_PATH=/opt/rocm` (append `;/opt/rocm` when the command already sets a prefix for another dependency, e.g. a vendored gtsam/pangolin install) in BOTH the recipe you run and every documented build block (README, install guide, porting notes), plus a one-line "or put `/opt/rocm/bin` on `PATH`" note. Watch the silent variant: `find_package(hip QUIET)` does not error on a clean container, it disables the GPU path and builds a CPU-only library. Builds that use only `enable_language(HIP)` with no roc/hip `find_package` and no `hip::` link are not affected. (visionaray #53 surfaced this while standing up a clean-container CI job.)

3. Guard genuinely divergent code with `#if defined(USE_HIP)`; keep such guards rare. Dispatch sites that accept either backend use `#if defined(USE_CUDA) || defined(USE_HIP)`.

This is how colmap was ported (PR 4420 plus follow-ups): one compat header, `.cu` marked `LANGUAGE HIP`, a few guarded fixes. PyTorch validated that this isolates HIP: on an MI250 build only the HIP translation units receive `-x hip`; host files are untouched.

## Large libraries: `--offload-compress` to fit the host image under the 2 GiB x86-64 relocation reach

A heavily-templated library (cudf: rocThrust/rocPRIM `DeviceReduce`/`DeviceScan`/`DeviceRadixSort` instantiated over a full type x op matrix) embeds an enormous device code object PER translation unit. The HIP offload bundle is stored UNCOMPRESSED by default, so a single reduce `.o` can reach ~200 MB (~159 MB of which is the `.hip_fatbin`). Linking many such TUs into one shared object can push the HOST image past the +/-2 GiB reach of x86-64 32-bit PC-relative relocations -- the link fails with `R_X86_64_PC32` overflow first, then (if you reach for `-mcmodel=large`, which only fixes data refs) `R_X86_64_TLSGD`/`TLSLD`/`__tls_get_addr` and `.eh_frame` "PC offset too large", neither of which the large code model covers. nvcc fatbins are compact, so the CUDA build never hits this; it is a ROCm-specific build-scale wall, not a HIP-correctness one.

Fix: add `--offload-compress` to the HIP compile options (`target_compile_options(<tgt> PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-compress>)`). A documented ROCm clang HIP flag (7.2.1+: "Compress offload device binaries (HIP only)") that stores the embedded device bundle in the compressed CCOB format; the HIP runtime decompresses it on load. Measured on a cudf rocPRIM reduce TU @ gfx90a: object 207.8 MB -> 52.1 MB (3.99x), `.hip_fatbin` 158.6 MB -> 2.89 MB (54.9x) -- the device ISA matrix is hugely redundant so it compresses ~55x. This let cudf's full reductions + aggregation + groupby + join surface link into ONE ~540 MB libcudf.so with no multi-.so split and no `-mcmodel=large`. For a non-RDC build (no `-fgpu-rdc` device-link step) the per-TU compile is the only place the bundle is formed, so there is no separate device-link to also flag. Prefer this BEFORE splitting a library or capping the template matrix. Also build a single target arch (no multi-arch fatbin) and add `-ffunction-sections -fdata-sections` + `--gc-sections` to prune unreachable host instantiations. (cudf.)

## The shim-header method: a port with zero source edits

The plain Strategy A shape adds `#include "cuda_to_hip.h"` to every `.cu`/`.cuh` that needs
it. That works, but it touches every source file, and the diff a maintainer reviews is
dominated by include lines rather than by the port.

Better, where the project includes CUDA headers by name: put SHADOW HEADERS named after the
CUDA ones in a `hip_compat/` directory, and prepend that directory to the include path on
the HIP build only:

    target_include_directories(<tgt> BEFORE PRIVATE ${CMAKE_SOURCE_DIR}/hip_compat)

`hip_compat/cuda_runtime.h` includes `hip/hip_runtime.h` and defines the aliases;
`cublas_v2.h` maps to hipBLAS; `cufft.h`/`cufftXt.h` to hipFFT. CUDA-only headers the port
does not need (`mma.h`, the CUDA-samples `helper_cuda.h`/`helper_functions.h`) become empty
stubs. The CUDA build never sees the directory, so its include resolution is untouched.

TurboFNO went from 38 edited source files (53 files changed) to **zero source edits** (21
files changed), leaving the `.cu`/`.cuh` tree byte-identical to upstream. That was verified
behaviour-preserving: bit-identical GPU results, and all 10 device code objects byte-identical
across two architectures.

The shim is mandatory, not just tidier, when the header it shadows is vendored NVIDIA
sample code under an EULA notice: see "Vendored CUDA-samples helpers" below.

Prefer this when the project includes CUDA headers by name. It does not apply where sources
use CUDA symbols without including a CUDA header, or where the build already injects a
compat header with `-include`.

## Vendored CUDA-samples helpers: keep them off the AMD build entirely

Many CUDA projects vendor `helper_cuda.h` / `helper_math.h` / `helper_string.h` from a
pre-2017 CUDA samples drop. Those copies carry "Please refer to the NVIDIA end user license
agreement (EULA)", so they are proprietary-marked files. Keep them out of the HIP build
completely -- not compiled, not executed, not a build input -- and do not modify them in the
port's diff either (editing one makes the diff a derivative work of it). Never delete a
licence or notice file to get there.

Two moves, both needed:

1. **Write your own substitute.** Grep first: in practice the only symbol a project uses from
   `helper_cuda.h` is `checkCudaErrors` (plvs: 36 sites; Velvet: 9). Write it INDEPENDENTLY --
   own name and namespace, a non-template function taking `hipError_t` exactly, your own
   message format, no `DEVICE_RESET` dance. Do not paraphrase the sample's `check<T>()`
   template or its `"CUDA error at %s:%d code=%d(%s)"` string; a HIP block that mirrors them
   is the same derivative problem one indirection away. Velvet's first port did exactly that
   and it had to be rewritten.
2. **Keep the directory off the HIP include path**, so an accidental include fails to resolve
   instead of compiling. Either wrap the entry (`if(NOT USE_HIP) target_include_directories(...
   Velvet/External/cuda) endif()`, Velvet) or put a shadow header in a `hip_compat/` directory
   prepended only on the HIP path (`include_directories(BEFORE ...)` inside `if(USE_HIP)`,
   plvs/alien). The CUDA build's include resolution is untouched either way.

**Prove it; do not infer it from the source.** `#if !defined(USE_HIP)` guards around the
include lines are not evidence the file is not a build input, and CMake's own
`HIP.includecache` lists textual `#include` lines it never resolved (it does not evaluate
`#if`), so it will show `helper_cuda.h` on a build that never opened it. What settles it is a
per-TU include trace with the build's real flags, taken from
`build/CMakeFiles/<tgt>.dir/flags.make`:

    hipcc -x hip -fsyntax-only -H <the build's exact flags> <tu> 2>&1 | grep -c helper_cuda

Expect 0 for EVERY translation unit, plus an empty `grep -rl "<vendor dir>" build/`.

**The CUDA path can usually be cleaned too.** NVIDIA republished all three headers under
BSD 3-Clause in `NVIDIA/cuda-samples` under `Common/`, and they are API supersets of the old
drops, so replacing the vendored copies with those releases verbatim is normally a no-op for
the CUDA build. Verify rather than assume: compile the TUs that include them with `nvcc` (and
the host TUs with the host compiler) at both the base and the new sha and diff the results, and
`-H`-trace one TU to confirm the swapped files are actually opened so the check is not vacuous.
Record the upstream commit and the sha256 of each file you copied in. (plvs, Velvet.)

**Never add a HIP branch inside the vendored file.** A `#if defined(USE_HIP)` block inside
`helper_cuda.h` makes the port's own diff a derivative modification of
NVIDIA-proprietary-marked code. plvs carried 41 lines of `check()`/`checkCudaErrors` inside
the vendored `include/cuda/helper_cuda.h`, mirroring NVIDIA's own signature and message
strings, and it took a separate fix to undo. Leave the vendored file byte-identical (check
with `git diff <base> -- <file>` printing nothing, not by eye) and search the tree early for
the markers: `grep -rliE "end user license agreement|NVIDIA CORPORATION and its licensors" .`
names exactly the files the diff must not touch. Reproducing the API NAMES the project's own
sources call (`checkCudaErrors`) is unavoidable and fine.

Do not justify the shim with "the vendored header does not parse under hipcc" without
checking: plvs's parses clean once the compat header is force-included, because its
cuBLAS/cuFFT enum tables sit behind `#ifdef CUBLAS_API_H_` and compile out. The real reason
there was narrower: `checkCudaErrors` is defined inside `#ifdef __DRIVER_TYPES_H__`, a macro
only the CUDA toolkit's `driver_types.h` sets, so on ROCm the header supplies nothing. A
one-line probe TU that calls the macro settles it, and the commit message should give the
accurate reason. (plvs)

### A per-backend directory is not a reason to add a HIP directory

Projects with a GPU abstraction often keep one directory per backend (`gpu/cl/`, `gpu/cuda/`)
and a target per backend. Do not add a third, copied `hip/` directory and `.hip` copies of the
`.cu` files: that is thousands of lines of hipify output that must track every CUDA change.
Masquerade the CUDA backend instead. A HIP option configures the existing CUDA targets, marks
their `.cu` files `LANGUAGE HIP`, links `hip::host` where the CUDA build links cudart, and puts
a shim directory on the HIP include path. Keep the CUDA target and type names; renaming them
for HIP adds a diff and no behaviour. Three things nvcc did silently need stating:

- nvcc implicitly includes `cuda_runtime.h` in every `.cu` file, so device code may use
  `threadIdx` before any include. Add `$<$<COMPILE_LANGUAGE:HIP>:SHELL:-include cuda_runtime.h>`
  so the HIP build does the same, through the shim.
- The backend's `.cpp` files are host C++ under CUDA. Leave them `CXX`; marking them HIP adds a
  device pass where `__HIP_DEVICE_COMPILE__` guards hide host-only declarations they use.
- Every `__CUDACC__` check gains `|| defined(__HIPCC__)`, and every `__CUDA_ARCH__` device-pass
  check gains `__HIP_DEVICE_COMPILE__`. Preprocess the `.cu` files before and after with
  `-D__CUDACC__` (and `-D__CUDA_ARCH__=700`) to show the CUDA build sees identical code.

ohm's first port added such a directory (~3,800 lines). The masquerade was +157/-16 over
24 files and passed the same GPU tests (8/8 and 58/58 on gfx90a). (ohm)

## Build hygiene

- **Do not pin `--offload-arch` or `CMAKE_HIP_ARCHITECTURES` in the committed build.** Pass
  the arch at build time so another architecture reuses the recipe with only
  `--offload-arch=<arch>`; `enable_language(HIP)` auto-detects. A pinned arch is the single
  most common reason a port builds on the machine it was written on and nowhere else.
  (TurboFNO, LC-framework)
- **A build-time wavefront width must refuse a mixed arch list.** When the warp size reaches
  a kernel template argument it has to be one macro visible to both compile passes (the
  device-only `__GFX9__` cannot feed the host launch), so it is derived from
  `CMAKE_HIP_ARCHITECTURES` -- and a list holding a gfx9 and a gfx10/11/12 arch silently
  gives one family the other's width: wrong reductions, no error. Auto-detect produces such
  a list on any host with both kinds of GPU. `message(FATAL_ERROR)` on a mixed list rather
  than picking one. (oneflow)
- **Take the ROCm include path from `find_package(hip)`, never `/opt/rocm/include`.** For
  host `.cpp` that need HIP or hipCUB/RCCL headers, use `${hip_INCLUDE_DIRS}` (set by
  hip-config.cmake) or `hip::host`'s include directories. The ROCm Python SDK wheels put the
  tree under site-packages with no `/opt/rocm` at all. (oneflow)
- **When the project drives CMake from its own shell scripts, the toggle belongs there too --
  and the toolchain probe must not hardcode `/opt/rocm`.** A bare `option(USE_HIP ...)` in
  `CMakeLists.txt` is unreachable from the documented build if users are told to run
  `./build.sh`: they get a CPU build and nothing says why. Put the toggle beside the existing
  CUDA one in the config script and forward it from wherever that script forwards
  `-DWITH_CUDA=ON`, which also makes the config script -- the file that already documents the
  CUDA build -- document the ROCm build. In the probe, mirror the CUDA probe's shape but accept
  a `hipcc` on `PATH` and honor an existing `ROCM_PATH`, instead of testing only
  `/opt/rocm/bin/hipcc`: the TheRock/pip ROCm SDK installs under a Python `site-packages`
  prefix, so a `/opt/rocm`-only test silently disables the GPU path on exactly the hosts that
  validate the port. Keep the "not found" message inside the enabled branch so CUDA and
  CPU-only builds print nothing new. (plvs)
- **Never export a fallback `ROCM_PATH` such as `export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"`.**
  `hipconfig` honors `ROCM_PATH`, and CMake runs `hipconfig` to find the toolchain for
  `enable_language(HIP)`, so a `ROCM_PATH` naming a directory that does not exist turns a
  working configure into `CMakeDetermineHIPCompiler: Failed to find ROCm root directory` -- the
  config script disables the very build its own `PATH`-based probe just declared available.
  Export only a prefix that resolves: keep an inherited value when `$ROCM_PATH/bin/hipcc`
  exists, else derive it from the accepted `hipcc` (`hipconfig --rocmpath` run with `ROCM_PATH`
  cleared, falling back to `dirname $(dirname $(readlink -f $(command -v hipcc)))`), else use
  `/opt/rocm` only if it exists, else export nothing. Print that same prefix in the "HIP found"
  message, not the fallback. Test it the way the documented build runs: a configure that passes
  `-DCMAKE_HIP_COMPILER=...` by hand bypasses discovery entirely and hides this. (plvs)
- **Grep the build scripts for every `CUDA_FOUND`-style gate, not just the flag forwarding.**
  Adding `-DUSE_HIP=ON` where the script forwards `-DWITH_CUDA=ON` is only half the wiring: a
  bundled GPU component built under `if [ $CUDA_FOUND -eq 1 ]` is still skipped on a machine
  with no CUDA toolkit, while the main `CMakeLists.txt` links its archive unconditionally, so
  the build configures cleanly and dies at link. Widen each such gate to
  `[ $CUDA_FOUND -eq 1 ] || [ $USE_HIP -eq 1 ]` and run the script on a CUDA-free machine.
  (plvs, its `Thirdparty/libsgm`)
- **Then prove each GPU component actually has device code in the HIP build.** Two more
  CUDA-only gates make a HIP build quietly CPU-only, and nothing fails: a source list that
  names the `.cu` files only inside `if(USE_CUDA)` (the HIP branch's
  `set_source_files_properties(${sources_gpu} ... LANGUAGE HIP)` then applies to an empty
  list), and a runtime selector such as `#ifdef USE_CUDA new ElasGPU #else new Elas` that the
  HIP build, defining only `USE_HIP`, never takes. The library builds, links, and passes every
  test on the CPU implementation, and the review and validation rounds accept it. Check for
  `--offload-arch` in `build.ninja`, a `.hip_fatbin` section in the archive members
  (`readelf -SW`), and the kernel names in an `AMD_LOG_LEVEL=3` trace. A file that includes
  the GPU header for the selector also needs the HIP headers, so build it as HIP (the CUDA
  branch gets them from `include_directories(${CUDA_INCLUDE_DIRS})`). (plvs, its
  `Thirdparty/libelas-gpu`, which ran CPU ELAS on every AMD platform for five months)
- **hipify-perl, when that is the mechanism rather than CMake HIP language:** let the
  conversion loop run to completion -- a whole-tree loop takes a while, and one cut short
  leaves the remaining headers in CUDA form, so re-running it finishes the job. The signal
  for that is the compiler, not a grep: un-hipified files surface as "undeclared identifier
  cudaMalloc". Grepping the tree for `cudaMalloc` instead reports the `.prehip` backups
  hipify itself writes -- 8 of 12 files on a fully converted tree -- and answers differently
  depending on whether the grep honors `.gitignore`, which usually excludes them. Note that
  hipify prepends `#include "hip/hip_runtime.h"`, which breaks a g++ CPU reference build, so
  build that from a separate non-hipified copy. (LC-framework)
- **`-inplace` re-hipifies from the `.prehip` backup, not from the file on disk.** The first
  run saves `foo.cu` as `foo.cu.prehip`; every later run reads the backup and overwrites
  `foo.cu` from it. So regenerating a file and hipifying it again silently resurrects the
  previous generation's content, with no diagnostic and a plausible-looking build. Delete
  the stale `*.prehip` (or work in a fresh copy of the tree) whenever a generator has
  rewritten a file that was already hipified. (LC-framework)
