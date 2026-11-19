# Build: CMake, device linking and dependencies

CMake's HIP language, device linking, OpenMP, runtime library paths and dependencies. Windows-specific build problems are in windows.md.

<!-- toc:begin -->
Contents:

- Which files compile as HIP
- Relocatable device code (-fgpu-rdc)
- OpenMP and runtime library paths
- Dependencies
- Installed packages and optional targets
- Other build systems
<!-- toc:end -->

## Which files compile as HIP

**A force-included compat header creates no build dependency edge.** After editing a header
injected with `-include`, object files are NOT rebuilt: wipe them manually or you validate
stale code and get a silent false pass. (lc0)

**Use the CUDA build's host compiler for host C++ in a large app.** On Linux, HIP needs clang
only for the HIP language; `CMAKE_C/CXX_COMPILER` can stay gcc. (Windows is the exception:
every CMake language there must use one clang driver, see windows.md.) Upstream code tested only with gcc can
trip clang-only C++ rejections unrelated to HIP (`__attribute__((visibility)) [[nodiscard]]`
ordering; `std::expected<std::any, E>` constraint recursion with libstdc++ 14).
(LichtFeld-Studio)

**The vcpkg toolchain already overrides `add_library`/`add_executable`.** A second
top-scope override (the MPPI retag trick) makes `_add_library` point at vcpkg's override,
which recurses. In a vcpkg project retag `.cu` sources with a pass over all targets after
the subdirectories are added: walk `BUILDSYSTEM_TARGETS` and `SUBDIRECTORIES`, and use
`set_source_files_properties(... TARGET_DIRECTORY <tgt> PROPERTIES LANGUAGE HIP)` with paths
made absolute against the target's `SOURCE_DIR`. (LichtFeld-Studio)

### Never mark the host `.cpp` files `LANGUAGE HIP` to dodge the flag leak

The tempting shortcut when a mixed CXX/HIP target fails to compile is to mark *every* source
`LANGUAGE HIP` so one set of flags applies uniformly. Do not. That defines `__HIPCC__` for
the whole program, and CUDA projects routinely key host-only behaviour off `__CUDACC__` --
which the port then extends to `__HIPCC__`, correctly, for the device path. The classic
shape is a macro that suppresses default member initializers, because a
`__device__ __constant__` copy of the struct cannot be dynamically initialized:

    #if defined(__CUDACC__) || defined(__HIPCC__)
        #define HOST_INIT(val)
    #else
        #define HOST_INIT(val) = val
    #endif

    struct SimParams { int numSubsteps HOST_INIT(2); glm::vec3 gravity HOST_INIT(...); };

On NVIDIA the `.cpp` files are plain C++ and never see `__CUDACC__`, so the host copy keeps
its defaults. Compile them as HIP and the host copy silently loses every initializer. Velvet
shipped that way from its first commit: the application ran, rendered and reported a healthy
GPU, but the solver read zero substeps, zero iterations and zero gravity, so nothing ever
moved. It survived four platform validations because each used a standalone synthetic kernel
test rather than the real binary. Grep for `__CUDACC__` and `__CUDA_ARCH__` in host headers
before choosing an all-HIP target, and treat a hit as a hard blocker on that shortcut.

An all-HIP target is not wrong in itself. Projects whose host headers key nothing on those
macros build every source as HIP and work (LEAP, HEonGPU). The default is still a mixed
target with only the `.cu` files compiled as HIP, and an all-HIP target owes three checks:
the grep above, the all-clang OpenMP wiring ("OpenMP and runtime library paths" below), and
the host `min`/`max` detection pass over the former host files (faults-numerics.md).

The failure the shortcut was dodging has its own fix. Linking `hip::hipcub`,
`roc::rocthrust` or `roc::rocprim` to a target that also has g++ host sources puts
`--offload-arch=...` on the g++ command line ("unrecognized command-line option"; h2o4gpu,
Velvet). The flags leak because those targets reach `hip::device` transitively
(`roc::rocprim_hip` -> `hip::device` in `rocprim-targets.cmake`), and `hip::device`'s
interface is built by

    function(hip_add_interface_compile_flags TARGET)
      set_property(TARGET ${TARGET} APPEND PROPERTY
        INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CXX>:${_HIP_SHELL}${ARGN}>")
    endfunction()

called with `-x hip` and `--offload-arch=<arch>`. Note the genex is `COMPILE_LANGUAGE:CXX`,
so it hits exactly the plain C++ sources and never the HIP ones -- the opposite of what the
name suggests. Those three packages are header only, so take their include directories and
link `hip::host` (which carries only `__HIP_PLATFORM_AMD__=1` and the `amdhip64` import
library, no compile flags) for the runtime:

    set_source_files_properties(${CU_SOURCES} PROPERTIES LANGUAGE HIP)
    foreach(rocm_target hip::hipcub roc::rocthrust roc::rocprim_hip)
      if(TARGET ${rocm_target})
        get_target_property(inc ${rocm_target} INTERFACE_INCLUDE_DIRECTORIES)
        if(inc)
          target_include_directories(<tgt> SYSTEM PRIVATE ${inc})
        endif()
      endif()
    endforeach()
    target_link_libraries(<tgt> PRIVATE hip::host)

Before splitting, check two things or you trade a silent-zeros bug for a silent-no-op bug:
no header that a `.cpp` includes may contain a `<<<>>>` launch (a launch macro gated on
`__CUDACC__ || __HIPCC__` expands to nothing in the host units, so the launch disappears),
and no variable whose initializer the macro suppresses may be an `inline`/`extern` variable
that both language groups emit (the linker folds one COMDAT and may pick the uninitialized
one). Velvet passed both: launches live only in the two `.cu` files, and the `inline`
`Global::simParams` is included by no `.cu`.

Verify the split landed rather than assuming: the build log should show "Building CXX
object" for the host files (Velvet: 11 CXX against 2 HIP, where all 13 previously said HIP),
the recorded compile line for a host object should carry neither `-x hip` nor
`--offload-arch`, and a temporary `#if defined(__HIPCC__) #error` probe in one host file
should compile clean.

Then expect the split to unmask missing standard-library includes in host headers, and build
on every supported OS before calling the split done. The HIP and CUDA runtime headers drag
in a lot of the standard library, so a header that has always been compiled as a GPU source
can use `std::vector` or `std::string` without including it and never be caught. Narrowing
the language property makes it a plain C++ TU and the omission becomes a hard error that
reads like a regression the split introduced; it is not, and the fix is the one-line
`#include`, never a revert of the split. The masking is also platform-dependent, so the
identical delta can pass on one OS and fail on another: MSVC's `<unordered_map>` supplies
`<vector>`, libstdc++'s does not. Velvet's split passed a full Windows application
validation and then failed to compile at all on Linux under both GCC 13 and the ROCm clang,
on `unordered_map<string, vector<hipEvent_t>>` in a header including only `<iostream>`,
`<unordered_map>` and `<string>`. Such a gap usually predates the port and breaks the
project's own NVIDIA build on the same OS, so fixing it is upstream-friendly rather than
AMD-specific -- say so in the commit message.

The same holds for any change in which files compile as the GPU language. A
`LANGUAGE HIP`/`LANGUAGE CUDA` blanket on host sources is a smell to look for in review, and
narrowing it to the real device files exposes whatever the blanket was compensating for, so
the delta needs a rebuild on every supported OS and with every host compiler. A CUDA-path
compile of the same file may already have recorded the identical error as pre-existing; the
two findings do not contradict. One path was always broken and stayed broken; the other was
accidentally working and stopped. (Velvet: `Timer.hpp` used `std::vector` with no
`#include <vector>`, on gcc's and clang's libstdc++ alike.)

## Relocatable device code (-fgpu-rdc)

**A `__device__` body in its own `.cu` has two answers on AMD, and the choice is about
consumers, not about speed.** When a project keeps `__device__` function definitions in their
own `.cu` with explicit instantiations, it is relying on relocatable device code (nvcc
`-rdc`, which CMake's `CUDA_SEPARABLE_COMPILATION` turns on). A HIP compile resolves device
symbols one TU at a time, so without a flag it stops with
`lld: error: undefined hidden symbol`. Either of these makes it link:

1. **`-fgpu-rdc`**, HIP's own relocatable device code, the direct counterpart of the setting
   the CUDA side already has. It leaves the project's sources untouched -- often zero diff in
   the files a maintainer cares about -- and it still inlines the cross-TU calls (see below).
   CMake has no `HIP_SEPARABLE_COMPILATION` property (checked on CMake 4.0.3), so wire it by
   hand, and remember the LINK side and the consumers:

   ```cmake
   target_compile_options(mylib PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
   # whoever links the archive performs the device link, including a consumer of the
   # installed export, so put it on the interface rather than in everyone's build
   target_compile_options(mylib INTERFACE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
   target_link_options(mylib INTERFACE -fgpu-rdc)
   ```

   On Windows, `-fgpu-rdc` under a clang-cl driver also meets the `/Fo` bundler bug
   (windows.md, "CMake's HIP link rule mangles MSVC-style flags and paths"); apply the
   compile-rule fix given there before judging the flag.

2. **A `__CUDACC__`/`__HIPCC__` guard** around the definitions moved into the header, with the
   declarations left unconditional (the shape and the two-phase-lookup reason are two entries
   below). It costs two edited files and it drops the `.cu` from the source list, but it puts
   NOTHING on the link interface, so consumers are unaffected.

Both were measured on the same project (HEonGPU, gfx90a, ROCm 7.2.1): 20/20 suites pass
either way, benchmarks equal within run-to-run noise, so performance does not decide it. What
decides it is whether the project ships an installed library to third parties. `-fgpu-rdc`
imposes a contract on every consumer that the CUDA path does not impose (next entry). A
library whose consumers are strangers should prefer the guard for that reason; an application,
or a library built in-tree by its only consumer, can take `-fgpu-rdc` and keep the smaller
diff. HEonGPU, an installed library, chose the guard. Whichever you choose, say in the code
which one it is and why -- the reviewer's first question is why the other one was not used.

**`-fgpu-rdc` changes the consumer contract, so the project's downstream docs owe a line
about it.** This is the cost of option 1 above. An archive of relocatable device code carries
no complete device image, so the device link happens at the consumer's final link, and only
the HIP driver in HIP link mode performs it. The interface options in the recipe make the
common consumer work; they do not make the requirement go away.

Symptoms at the consumer (ROCm 7.2.1, gfx90a): a `g++` link of the installed archive gives
`undefined reference to __hip_gpubin_handle_<hash>` and `__hip_fatbin_<hash>`; a gcc-driven
link that does receive the interface flag stops earlier with
`g++: error: unrecognized command-line option '-fgpu-rdc'`.

Three conditions have to hold, and which one a consumer lacks decides its remedy:

1. HIP is an ENABLED LANGUAGE in the consuming project (`project(app LANGUAGES CXX HIP)` or
   `enable_language(HIP)`). `find_package(hip)` and linking `hip::host` do not enable it, and
   nothing set on a target substitutes for it.
2. The link DRIVER is the HIP compiler. With the language enabled, CMake arranges that when
   the target has HIP sources of its own, links a target that has them, links an IMPORTED
   target whose `IMPORTED_LINK_INTERFACE_LANGUAGES` is HIP (`install(EXPORT)` records that for
   a library built from HIP sources, and it propagates through a plain C++ intermediate
   library), or sets `LINKER_LANGUAGE HIP`.
3. The link LINE carries `--hip-link`, which CMake adds ONLY for a target that has HIP
   sources of ITS OWN.

Measured against an installed export and a hand-imported archive (ROCm 7.2.1, gfx90a, CMake
4.0.3):

| consumer target | link driver | needs |
| --- | --- | --- |
| one of its own sources compiled as HIP | HIP, `--hip-link` added | nothing |
| plain C++ in a HIP-enabled project, links the exported target directly or through a plain C++ library | HIP, no `--hip-link` | `target_link_options(app PRIVATE --hip-link)` |
| plain C++ in a HIP-enabled project, links a local HIP-source library that links the archive | HIP, no `--hip-link` | same |
| the same target in a project that never enabled HIP | gcc | `enable_language(HIP)`; nothing set on the target alone works |
| plain C++ in a HIP-enabled project, links an IMPORTED archive with no link interface language | gcc | `LINKER_LANGUAGE HIP` AND `--hip-link`; `--hip-link` alone only adds a second unrecognized option |
| the archive named by path in a HIP-enabled project, no imported target | gcc | `LINKER_LANGUAGE HIP`, `--hip-link` AND `-fgpu-rdc`, since nothing supplies the flag either |

`LINKER_LANGUAGE HIP` never adds `--hip-link`, so it is useless alone and needed only where
the driver is still gcc; it exists only where HIP is enabled (otherwise generation fails with
`Missing variable is: CMAKE_HIP_LINK_EXECUTABLE`), which is why the fourth row has no
target-level remedy. Wrapping the interface flag in `$<LINK_LANGUAGE:HIP>` does not help
either: it trades the unrecognized-option message for the undefined-symbol one. Note the
asymmetry in the recipe: the interface compile option is conditioned on the HIP language and
the interface link option is not, so `-fgpu-rdc` reaches, and breaks, a consumer that pins
`LINKER_LANGUAGE CXX` in a HIP-enabled project. Decide deliberately whether that is what you
want.

The PRODUCER can enable the language for its consumers. A package config file is included at
the caller's file scope, so `enable_language(HIP)` as the first line of
`cmake/Config.cmake.in` takes a `project(app LANGUAGES CXX)` consumer from the
unrecognized-option error to configuring and linking with only `--hip-link` on its target,
also when `find_package` is called from an `add_subdirectory`. It has costs: `enable_language`
must run at file scope in the highest directory common to all targets using the language, so
a consumer that wraps `find_package` in a function gets no language at all (configure
succeeds, the link fails as before), and every consumer pays HIP compiler detection at
configure time (1 to 3 seconds).

Either way, fix it in prose. Wherever the project tells a CUDA consumer to set
`CUDA_SEPARABLE_COMPILATION ON`, the HIP consumer needs the parallel sentence, and if the
config file does not enable the language the docs must say enabling HIP is a requirement, not
a suggestion. Scope the sentence to the consumer shape it is true of: "add `--hip-link`" is
right for a HIP-enabled project consuming an installed export and wrong both for a
hand-rolled import and for a C++-only project. (HEonGPU)

**Cross-TU `__device__` calls ARE inlined on AMD under `-fgpu-rdc`.** Do not repeat the nvcc
intuition here: nvcc `-rdc=true` really does leave a call unless you add `-dlto`, but HIP's
device link is a bitcode link followed by optimization of the whole image. Measured on gfx90a
with ROCm 7.2.1, on a `__device__` function template with its explicit instantiation in one
TU called from a `__global__` in another: the linked device image contains only the kernel,
`s_swappc_b64` count 0, and the callee's `s_barrier`s appear inline in it. What `-fgpu-rdc`
does cost is link time and image size, and both are worth measuring before deciding
(HEonGPU, gfx90a: default library build 53.7 s vs 52.1 s, i.e. unchanged, but 193.6 s vs
326.6 s once 42 test/example/benchmark executables are enabled, i.e. about 3.2 s of device
link each, since every one of them device-links the whole library; device code per test
binary 1,575,096 -> 1,684,264 bytes; TFHE gate timings and 20/20 suites identical either
way, and BFV/CKKS benchmarks unchanged to within run-to-run noise).

**Moving the bodies into a host-reachable header UNGUARDED silently breaks the CUDA build.**
This is the tempting move: mark them
`__device__ inline` in the header. It works on AMD, and g++ compiling any host TU that
reaches the header must then parse device bodies and rejects `__syncthreads` ("there are no
arguments to `__syncthreads` that depend on a template parameter") -- even though the
templates are never instantiated in that TU, because `__syncthreads` is a non-dependent name
and two-phase lookup resolves it at definition time. If you have a reason to keep the bodies
in the header anyway, keep the declarations unconditional and guard the DEFINITIONS:

```cpp
template <typename T> __device__ void SmallForwardNTT(T*, ...);   // host TUs see this

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T> __device__ inline void SmallForwardNTT(T*, ...) { ... __syncthreads(); }
#endif
```

Test the COMPILER (`__CUDACC__`/`__HIPCC__`), not the target: the requirement is that only a
device compiler parses a device body, and both back ends compile the calling `.cu` files with
one. `__CUDA_ARCH__`/`__HIP_DEVICE_COMPILE__` would be wrong here -- they are false during
nvcc's host pass, which still has to parse the same TU. Adding `inline` at the definition
after a non-inline declaration is legal. The third arrangement, keeping the separate `.cu`
for CUDA and the header definitions for HIP, is the worst of them: it duplicates the bodies
in two files that must not drift.

**No AMD build, review round or GPU validation can detect this class.** In HEonGPU two review
rounds and a full 20/20 GPU validation passed over the inlined header; only compiling the
CUDA path with nvcc found it. Any port that touches how device bodies are compiled owes an
nvcc compile check (validation.md, "The CUDA no-regression compile"). Adding the guard is codegen-neutral on AMD --
verified with a comparison of device code objects, identical exported symbols and device ISA
on all 42 gfx90a binaries -- because every calling TU is a device TU where the guard is true.
Switching to `-fgpu-rdc` is NOT: it changes the device ISA of every binary (the device link
re-optimizes the whole image), so it is a real code delta and every arch owes a fresh GPU
run. That cuts both ways, and it is a free correctness check if you ever switch back:
reverting HEonGPU's `-fgpu-rdc` to the guard reproduced the pre-RDC build byte for byte in
device ISA and exported symbols on all 42 binaries, which is what a clean revert should look
like. Keep a snapshot of the earlier `bin/` around for exactly that comparison. (HEonGPU)

## OpenMP and runtime library paths

**One OpenMP runtime per binary; which one follows from who compiles the files that use
OpenMP.** g++ objects call libgomp (`GOMP_*`), clang objects call libomp (`__kmpc_*`), and a
target that contains HIP objects is LINKED by clang. The two entries after this one are the
two rows of this table with their evidence; they prescribe opposite link lines because they
are different build shapes.

| target | compile | link |
|---|---|---|
| mixed: host `.cpp` compiled by g++, `.cu` compiled as HIP | OpenMP on the g++ files only; the HIP TUs without it (their pragmas run serially) | `OpenMP::OpenMP_CXX` (libgomp); no `-fopenmp` on the link line |
| every source compiled by clang (all `LANGUAGE HIP`, or clang as the host compiler too) | `-fopenmp` on the HIP sources as well | `-fopenmp` on the link line (clang adds its own libomp and a runpath to it); no `OpenMP::OpenMP_CXX` |

Check either with `readelf -d <lib> | grep NEEDED`: exactly one of `libgomp` and `libomp`.

**OpenMP in a mixed g++/HIP library: one runtime only.** g++ objects call libgomp
(`GOMP_*`); clang `-fopenmp` emits libomp (`__kmpc_*`). A target containing HIP objects is
LINKED by clang, so an `-fopenmp` link flag there adds `libomp.so` + `libomptarget.so`
next to libgomp -- two OpenMP runtimes in one process. Link `OpenMP::OpenMP_CXX` (libgomp)
and drop `-fopenmp` from the HIP-linked target's link flags; compile the HIP TUs without
OpenMP (their pragmas run serially) unless the whole build uses clang. Check with
`readelf -d <lib> | grep NEEDED`. (h2o4gpu)

**CMake `LANGUAGE HIP` on host `.cpp` files drops OpenMP.** `OpenMP::OpenMP_CXX` adds
`-fopenmp` through `$<$<COMPILE_LANGUAGE:CXX>:...>` and links the host compiler's runtime
(libgomp for g++). A build that sets every source to `LANGUAGE HIP` therefore compiles its
CPU code without OpenMP: `#pragma omp` loops run serially with no warning, and code that
mixes OpenMP runtime calls with the pragmas can misbehave (LEAP's `downSample_cpu` aborted
with `free(): invalid pointer`; the same code built by g++ ran). Passing `-fopenmp` for HIP
alone while still linking `OpenMP::OpenMP_CXX` fails at load time (`undefined symbol:
__kmpc_fork_call`: clang emits libomp calls, the link brought libgomp).
`find_package(OpenMP COMPONENTS HIP)` failed to find flags with the TheRock SDK and CMake
3.31. What worked: `target_compile_options(t PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fopenmp>)`,
`target_link_options(t PRIVATE -fopenmp)`, and no `OpenMP::OpenMP_CXX` on the HIP target;
clang links its own libomp and adds a runpath to it. Check with `grep -c fopenmp
compile_commands.json` and `ldd` on the library. (LEAP version two.) Keeping
`OpenMP::OpenMP_CXX` as well gives the binary both runtimes (`readelf -d` shows
`libgomp.so.1` and `libomp.so`), and it can look fine: ROCm ships `lib/llvm/lib/libgomp.so.1`
as a symlink to its own `libomp.so`, so both names resolve to one runtime until a distro
library path puts GNU's libgomp first. Then clang's `__kmpc_*` calls still fork a real team
from libomp, but `omp_get_thread_num` answers from libgomp and reports 0 on every thread
(measured: four distinct OS thread ids, `0 0 0 0`), so code that picks a per-thread resource
by thread id has every thread pick element 0 while still printing correct results. Also drop
any `LINKER_LANGUAGE CXX` left on the target: `$<LINK_LANGUAGE:HIP>` is false under a CXX
link, so a HIP-conditioned OpenMP link flag silently never applies. (HEonGPU)

**A HIP-linked shared library gets no RUNPATH to the ROCm lib dir.** The ROCm lib dir is
one of the HIP compiler's implicit link directories, so CMake leaves it out of the build
RUNPATH; clang adds it for executables (`h2o4gpu_test` had it) but not for a `SHARED` or
`MODULE` target. Anything else that happened to inject it (h2o4gpu's was clang's
`-fopenmp` link) hides the gap until it is removed. Then, with the pip/TheRock SDK and no
`LD_LIBRARY_PATH`, the module fails to load -- and a Python loader that swallows the
`ImportError` (h2o4gpu's `GPUlib.get()`) silently runs every solver on the CPU while the
tests pass. Set it explicitly on the target:
`BUILD_RPATH $<TARGET_FILE_DIR:hip::amdhip64>` (and `INSTALL_RPATH`), then check
`readelf -d <lib> | grep RUNPATH` and run the GPU path with `LD_LIBRARY_PATH` unset.
Whether a `/opt/rocm` package registers its lib dir with ld.so.conf was not verified; do
not rely on it. (h2o4gpu)

**With the ROCm clang as the host compiler, a found system libunwind links but does not load.**
The ROCm clang searches the `include/` directory next to its LLVM install ahead of the system
include directories, so `<libunwind.h>` resolves to LLVM's header, while a `FindUnwind` module
links the system GNU `libunwind.so`. GNU's header maps `unw_getcontext`, `unw_init_local`,
`unw_step` and `unw_get_reg` to `_Ux86_64_*` through macros; LLVM's declares the plain names. The
build succeeds and the library then fails to load with `undefined symbol: unw_getcontext`, but
only on machines with `libunwind-dev` installed, so it hides until a host gains the package.
Check `nm -D <lib> | grep ' U unw_'`. A bare `clang++ -E` may resolve the system header; run
`-E -H` on the real compile command from `compile_commands.json`. glog also links libunwind when
it finds it. Under the HIP option, set `CMAKE_DISABLE_FIND_PACKAGE_Unwind ON` before the
dependencies are configured; backtrace code falls back to `_Unwind_Backtrace`. (oneflow)

## Dependencies

**LibTorch's CMake package adds `-D__HIP_NO_HALF_OPERATORS__=1 -D__HIP_NO_HALF_CONVERSIONS__=1`
to every target that links it** (`torch_hip` INTERFACE_COMPILE_OPTIONS). Project `.cu` test
files that use `__half` arithmetic then fail with "invalid operands ... '__half'". If no HIP TU
includes LibTorch, `#undef` both in the force-included compat header under `__HIP__`.
(LichtFeld-Studio)

**On a multi-GPU host a rapids-cmake HIP dependency fails configure with `mismatch between
CMAKE_HIP_ARCHITECTURES and GPU_TARGETS`.** ROCm's `hip-config.cmake` fills the `GPU_TARGETS`
cache variable with one entry per visible device, not per distinct architecture, so four
cards of one kind report `gfx1100;gfx1100;gfx1100;gfx1100` against a single-valued
`CMAKE_HIP_ARCHITECTURES`, and the string comparison in `rapids_hip_init_architectures` (used
by hipMM/RMM) trips although only one arch was requested. It is an artifact of the host, not
a port defect: the identical configure passes with `HIP_VISIBLE_DEVICES=0`. Work around it on
the command line, with no source edit: pass `-DGPU_TARGETS=<arch> -DAMDGPU_TARGETS=<arch>`
alongside `-DCMAKE_HIP_ARCHITECTURES=<arch>`. (HEonGPU, four gfx1100)

### LibTorch on the include path shadows the system `fmt`, and spdlog stops compiling

Symptom: after a torch update, a project that links LibTorch only as a test oracle
(`find_package(Torch)` in the test target) fails inside `spdlog/common.h` with a
template-not-found error (`fmt::basic_format_string`), in a TU that has nothing to do with
ROCm. Cause: recent PyTorch ships its own `fmt` under `torch/include/fmt/`. Its `core.h` is
a reduced stub, but its `format.h` is the full current fmt (12.x), and once
`-isystem .../torch/include` is on a TU's search path every `#include <fmt/...>`, including
spdlog's, resolves to torch's copy first, because the compiler's implicit system directories
are searched last. A distro `libspdlog` built against an older fmt API then does not compile.
Confirm by checking that `torch/include/fmt/format.h` exists and reports a newer
`FMT_VERSION` than the system's `fmt/core.h`; a second, unrelated compiler hitting the same
error rules out a compiler quirk.

Fix without touching the port: build spdlog from source with `-DSPDLOG_FMT_EXTERNAL=OFF`
(it then uses its own bundled fmt) and point `spdlog_DIR` at that install. Pick the spdlog
version by the fmt it bundles: bundled fmt 10.x and older fail under a recent clang with
`call to consteval function 'fmt::basic_format_string<...>' is not a constant expression`,
a second failure that looks like the first and reproduces with no torch include at all
(spdlog 1.14.1 bundles fmt 10.2.1 and fails; 1.15.3 bundles fmt 11.x and is clean). Triage
either symptom in a second with `clang++ -std=c++20 -fsyntax-only -I<spdlog>/include` on a
two-line TU calling `spdlog::info("{:.6e}", f)`, instead of rebuilding the project. A torch
used this way is a moving dependency even when the port's source does not change; the other
fault the same torch bump produced is in build-headers-macros.md ("A torch header the compat
layer already aliases for can grow new symbols"). (LichtFeld-Studio, torch 2.14.0a0)

### Submodules pinned to commits that do not exist

**A port that edits a git submodule in place is lost the moment the clone is deleted, and
the branch it leaves behind cannot even be cloned.** If the submodule's working tree is
edited and the resulting gitlink is committed in the parent without the submodule commit
being pushed anywhere, the gitlink names a SHA that exists in no repository. (HEonGPU pins
GPU-NTT, GPU-FFT and RNGonGPU as submodules and lost its submodule port this way.)

Check for it early: `git ls-tree HEAD <submodule-path>` in the port branch, then
`git submodule update --init`. If the update cannot find the commit, the branch is
unbuildable and no amount of local success proves otherwise.

Forking every submodule is often not an option (a chain of forks has to be re-pointed in
`.gitmodules` and kept in sync). The durable shape that needs no new repositories is to
**reset the gitlinks to the real upstream commits and carry the submodule changes as patch
files in the parent repo**, applied by whatever script the project already runs at configure
time:

    git submodule update --init --recursive
    if [ "${1:-OFF}" = "ON" ]; then                       # only for AMD builds
        for name in ...; do
            patch="$here/patches/$name.patch"
            git -C "$here/$name" apply --reverse --check "$patch" 2>/dev/null && continue
            git -C "$here/$name" apply "$patch"
        done
    fi

The reverse-check is what makes repeated configures idempotent. Pass the AMD flag in from
CMake so an NVIDIA build sees the submodules untouched. Add `ignore = dirty` to each entry
in `.gitmodules`: applying the patches permanently dirties those working trees, and without
this every later `git status` reads as an unclean tree and fails any check for a clean tree,
while a genuine gitlink bump still shows. This also states the honest upstream position --
these changes belong in the submodules' own repositories, and the patch files are exactly
the diffs to send there. (HEonGPU)

## Installed packages and optional targets

### Install and consume the port, not just build it

A project that ships `install()` rules and a `<Name>Config.cmake` has a second interface the
in-tree tests never touch, and it is routinely left CUDA-only by a port that otherwise works:
the exported config `find_dependency(CUDAToolkit)` unconditionally, the exported link
interface names a HIP imported target with no `find_package` that defines it, a replacement
for a CUDA-only dependency installs its export set without a config file to find it by, and
the shim headers that resolve `cuda_runtime.h` / `curand_kernel.h` in the INSTALLED headers
are neither installed nor on the interface include path. Every one of these is invisible
until someone consumes the installation, so **install to a scratch prefix and build a
five-line consumer project against it as part of the port**. `configure_file(... @ONLY)`
makes the config-template fix a one-liner with no CMakeLists change:

    if("@USE_HIP@")
        find_dependency(hip REQUIRED)
    else()
        find_dependency(CUDAToolkit REQUIRED)
    endif()

Quote the substitution: when the option is not defined (the subproject built standalone) it
expands to nothing, and a bare `if()` is a CMake error while `if("")` is false. Note also
that `cmake --install --prefix` does NOT override a prefix a project baked into its install
rules at configure time; reconfigure with `-DCMAKE_INSTALL_PREFIX` instead of concluding the
install is broken. Write the downstream snippet into the docs only after the consumer
actually configures, builds and runs. (HEonGPU)

### Every optional target the project can build, not just the one you build

`BUILD_TESTS` is on while you work, so the test targets get the full treatment -- source
compiled as HIP, the `USE_HIP` definition, the shim include directory -- and `BUILD_EXAMPLES`
and `BUILD_BENCHMARKS`, which default OFF, quietly do not. A half-converted branch of that
kind is worse than no branch at all, because it reads as tested: on HEonGPU the example
targets linked the HIP runtime but were still compiled as plain C++, so the first project
header pulled in `cuda_runtime.h` and then rocThrust, and the benchmark targets still linked
`CUDA::cudart` unconditionally and failed at configure time. **Turn every option the project
documents ON once and build it**, then run one binary from each group; that is minutes, and
it is the only way the claim in your commit message is true.

## Other build systems

### A cached build-script decision from a wrong env var survives fixing the env var

If a Rust `build.rs` reads an env var such as `ROCM_PATH` to emit `cargo:rustc-link-search`
but does not declare `cargo:rerun-if-env-changed=<that var>`, cargo does not re-run the script
when only the variable changes. A build attempted once with a wrong value caches the wrong
`-L` path in `target/debug/build/<crate>-<hash>/output` and keeps using it after the variable
is corrected. The compile can appear to pass (an `.rlib` or static archive does not need the
search path), and the FIRST step that links a final binary, typically a `cargo test` binary,
fails with `unable to find library -l<name>`, which reads like a missing runtime library.
Fix: `cargo clean -p <crate>` before rebuilding. More generally, check
`env | grep -i <toolkit>_path` against what the build script expects before trusting a value
copied from another machine: `/opt/rocm`, a pip-installed `_rocm_sdk_devel` tree and a conda
env are different layouts. (mahout)
