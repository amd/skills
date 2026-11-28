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

**An unset CMake build type runs every HIP kernel unoptimized.** Symptom: a HIP port passes its
tests but its kernels are tens to hundreds of times slower than expected;
`-Rpass-analysis=kernel-resource-usage` shows kilobytes of scratch per lane. Cause: nvcc optimizes
device code whatever `CMAKE_BUILD_TYPE` says, but with no build type CMake passes no `-O` flag and
clang compiles HIP device code at `-O0`. A README configure command without `-DCMAKE_BUILD_TYPE` is
the usual trigger, and a Windows generator defaulting to Debug is the same thing. Fix: in the
project's CMake, when the HIP backend is selected and `NOT CMAKE_BUILD_TYPE AND NOT
CMAKE_CONFIGURATION_TYPES`, set the cache entry to `Release`; check `ninja -t commands` for `-O3`.
For Windows that default has to run before `project()`, because `Windows-Clang.cmake` sets
`CMAKE_BUILD_TYPE_INIT Debug` and `project()` caches it. Placed there it must also test the
`CMAKE_BUILD_TYPE` environment variable, which CMake (3.22+) applies only inside `project()`, or it
forces Release over a build type the user gave that way (on Linux too): `get_property(_mc GLOBAL
PROPERTY GENERATOR_IS_MULTI_CONFIG)`, then `if(NOT CMAKE_BUILD_TYPE AND "$ENV{CMAKE_BUILD_TYPE}"
STREQUAL "" AND NOT _mc)`, then `set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)`.
Device `-O0` also changes which faults appear (faults-memory-streams.md, "Bitwise and min/max
atomics on `hipMallocManaged` memory"), so build the README recipe as written before and after the
fix. Seen on MI300X with the ROCm SDK 10.1: paged KV append 11 ms -> 26 us, and FA2 prefill at qo=1
went from 957 ms for kv=1024 (unoptimized) to 13.8 ms for kv=4096 once combined with launch bounds
(platform-limits.md, "A kernel launched with fewer than 1024 threads needs `__launch_bounds__`").
(Quest)

**`CMAKE_COMPILE_WARNING_AS_ERROR` outranks `-Wno-error`.** A project that sets it (ROS 2's
`autoware_package()` does on CMake 3.24 and later) gets `-Werror` appended after every target
compile option, so `target_compile_options(t PRIVATE -Wno-error)` has no effect. When `.cu`
sources move from nvcc (which never saw the flag through `CUDA_NVCC_FLAGS`) to the HIP language,
upstream kernel warnings become errors; set the target property `COMPILE_WARNING_AS_ERROR OFF`
on the HIP kernel target instead. (autoware_universe)

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

The same holds for linking `hip::device` directly, which most hand-written HIP CMake does:
on a mixed target the plain `.cpp` files fail with `c++: error: unrecognized command-line
option '--offload-arch=gfx942'`. PRIVATE does not help (it only stops the flags reaching
consumers, not the target's own C++ sources), and `$<COMPILE_LANGUAGE:HIP>` cannot rescue it
either, because that genex is rejected in `target_link_libraries`. The link half is a
separate helper, `hip_add_interface_link_flags`, which puts `--hip-link` and `--offload-arch`
into `INTERFACE_LINK_LIBRARIES`; whether it is language-gated depends on the version pair
(ROCm 7.14 wraps them in `$<$<LINK_LANGUAGE:CXX>:...>` on CMake 3.20+ and leaves them ungated
below it), so assume `hip::device` also reaches a consumer's link line. With
`enable_language(HIP)` and `LANGUAGE HIP` on the sources you do not need `hip::device` at
all: CMake emits the offload flags for the HIP sources from `CMAKE_HIP_ARCHITECTURES` and
links the HIP runtime itself. A target that is 100% HIP sources never shows any of this,
which is why it surfaces the first time a port compiles kernels into an existing mixed
library. (rmcl: `rmcl_ros_cuda`, five C++ sensor sources next to two `.cu`; CuRast, which
took `${hipcub_INCLUDE_DIRS}` from `find_package(hipcub REQUIRED)` instead of linking
`hip::hipcub`.)

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

The link half of that interface is separate and outlives a compile-side fix.
`hip_add_interface_link_flags` puts `--hip-link` and `--offload-arch=` into `hip::device`'s
`INTERFACE_LINK_LIBRARIES` under `$<$<LINK_LANGUAGE:CXX>:...>` (CMake 3.20+; below that,
unconditionally), so a consumer whose sources no longer see the compile flags but whose
*link* language is still CXX gets the identical unrecognized-option error from the link
line; the residual case is a target with no HIP sources of its own that merely links a HIP
library. Never strip the flags off `hip::device` to cure either half: the imported target
is regenerated from the ROCm package on every configure, and the flags are how a project
that compiles device code through the CXX language with hipcc gets compiled at all.

Look for the targets a port's own helper function does not cover. A project that routes its
libraries through an `add_library` wrapper usually has a few built with a plain
`add_library()` somewhere else -- Python bindings, tools, examples -- and those inherit the
flags without the treatment. Two cheap checks: grep for `add_library(`/`add_executable(`
outside the wrapper, and read the generated `build.ninja` for a `CXX_COMPILER_<target>` rule
whose `FLAGS` contain `-x hip` (Makefiles generator: `CMakeFiles/<target>.dir/flags.make`).
Source-file properties are DIRECTORY-scoped, so when the treatment is a source property set
it in the directory that owns the target; for sources appended from a subdirectory with
`target_sources()`, `set_source_files_properties(... TARGET_DIRECTORY <tgt> ...)` (CMake
3.18+) follows the target where `DIRECTORY <owning-dir>` hardcodes a path. (GooFit: its
`_Core`/`_Basic`/`_Combine` pybind11 modules were outside the wrapper.)

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

Compiling host sources as HIP also changes which compiler mangles their symbols. Against
libraries built with GCC, a few template signatures mangle differently: a return type of
`std::enable_if_t<std::is_same_v<T, X>, ...>` is `Xsr3stdE9is_same_v...` from clang and
`X9is_same_v...` from GCC 13. The shared library still links (undefined symbols are allowed in
a `.so`), and under lazy binding the process dies with "symbol lookup error" only when the call
is first bound, so tests that never reach it pass; with `-Wl,-z,now` or `LD_BIND_NOW=1` it fails
at load. Run `ldd -r` on every built library and executable (it lists unresolved symbols; a
plugin library may legitimately leave some to the process that loads it, so read each hit). Keep
sources that only call host libraries on the host compiler, as the CUDA build does.
(autoware_universe, GCC 13.3)

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
   CMake has no `HIP_SEPARABLE_COMPILATION` property (checked on CMake 3.31.6 and 4.0.3:
   `cmake --help-property-list | grep SEPARABLE` lists only `CUDA_SEPARABLE_COMPILATION`),
   and setting one is silently accepted and leaves `build.ninja` byte-identical, so a port
   can ship dead `set_target_properties` calls believing they did the compile half (GooFit
   did). Wire it by hand, and remember the LINK side and the consumers:

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

**`-fgpu-rdc` on the compile line alone trades the first link error for a more confusing
one, and the fix is the link flag, not a restructured build.** The same need arises when a
project declares `__device__`/`__constant__` globals or device function-pointer tables in
headers and defines them in a separate `.cu` (first symptom, e.g.
`lld: error: undefined hidden symbol: MetricTaker::operator()(thrust::tuple<...>) const`).
With the flag only in `target_compile_options`, the in-tree link then fails with

    ld.lld: error: undefined hidden symbol: __hip_gpubin_handle_b33c4363e66c5890
    ld.lld: error: undefined symbol: __hip_fatbin_b33c4363e66c5890

Dump the generated link line before theorising; it says so directly:
`clang++ --offload-arch=gfx1100 --hip-link ... main.cu.o -o myexe libmylib.a` has
`--hip-link` (CMake adds it for a target with HIP sources of its own) and no `-fgpu-rdc`,
so the driver runs a non-relocatable link that never emits the fatbinary registration
symbols the RDC-compiled objects reference. One line fixes it:
`target_link_options(<tgt> PRIVATE $<$<LINK_LANGUAGE:HIP>:-fgpu-rdc>)`, or `INTERFACE` on
the usage-requirement target every executable and shared library of the project consumes
(`GooFit_Common`, `<proj>_options`); inside one project the `LINK_LANGUAGE:HIP` condition
keeps the flag off targets that link with the plain host compiler, which would reject it
(for consumers of an INSTALLED archive see "`-fgpu-rdc` changes the consumer contract"
above, where that condition does not help).

Archives are NOT invisible to the HIP device link, so do not restructure a project's
`STATIC` libraries into `OBJECT` libraries gathered behind one shared library for this:
that is a large, mergeability-hostile change to somebody else's build system for a missing
flag. Verified twice on ROCm 7.2.3: raw hipcc, with a `__device__` global and a
`__device__` function defined in `a.hip`, archived into `liba.a` and called from a kernel in
`b.hip`, links and runs via `hipcc -fgpu-rdc --hip-link b.o liba.a`; and at GooFit's real
shape (several `STATIC` libraries behind an `INTERFACE` aggregate, a cross-TU `__device__`
global plus a device function-pointer table) adding only the link option builds and passes
the full suite with every library left an ordinary `.a`. The archive caveat that IS real is
the ordinary host-linker one: the linker extracts only an archive member that resolves an
undefined HOST symbol, so a member whose device code is needed but whose host symbols are
never referenced is never pulled in and its device code never reaches the device link.
`-Wl,--whole-archive` around that archive is the fix; reach for it only when you can name
the member being dropped. A companion from the same port: delete host globals that take the
address of a `__device__` function (`device_function_ptr p = device_Step;` at namespace
scope), which is invalid on HIP and dead code on CUDA. (GooFit; RXMesh needed only the
compile-side flag, because its device code was already in one target.)

**`-flto` on the COMPILE line does not survive a `-fgpu-rdc` device link.** Measured with
RAW `hipcc` on ROCm 7.2.3 / gfx1100, using a two-TU `__device__`-global reproducer. Two of
the three configurations are fine:

| how `-flto` is applied alongside `-fgpu-rdc`, driving hipcc by hand | result |
| --- | --- |
| compile and link in ONE `hipcc` invocation | links |
| `-flto` only on the LINK line, objects compiled without it | links |
| `-flto` on the COMPILE line, then a separate link | fails |

The failure is `ld.lld: error: undefined symbol: __hip_fatbin_<hash>`, `referenced by
__hip_fatbin_wrapper`: the object is LLVM bitcode, and the fatbinary symbol its wrapper
references is only produced by the separate device-link step. A reproducer that compiles
and links in one command therefore reports LTO as fine when it is not.

Those rows are about hipcc, not CMake, and the two do not currently meet: as of CMake
3.31.6, `INTERPROCEDURAL_OPTIMIZATION` does nothing to HIP sources. Verified at the shape a
real port has (a HIP `STATIC` library plus a HIP executable, `-fgpu-rdc` on compile and
link, the property `ON` for both): the generated flags carry no `-flto` and the link
succeeds, while a plain CXX target in the same project does receive `-flto=thin`. The reason
is a language filter in the generator, not a missing flag: `Compiler/Clang-HIP.cmake` calls
`__compiler_clang(HIP)`, so `CMAKE_HIP_COMPILE_OPTIONS_IPO` really is `-flto=thin`, but
CMake applies IPO only to C, CXX, CUDA and Fortran. `check_ipo_supported()` cannot tell you
any of this: it probes only those four languages, returns NO with
`language(s) 'HIP' not supported` when asked for `LANGUAGES HIP`, and with no languages
filters `ENABLED_LANGUAGES` down to the four, so a YES from it is a statement about the host
C++ compiler and never about hipcc. Its answer can also change under you (GooFit's flipped
NO -> YES across an upstream sync). So excluding HIP from a project's IPO switch is
precautionary, not a fix: worth doing if the project already has a per-backend exclusion to
sit next to (GooFit disables IPO for CUDA in the same place, because CUDA device LTO hangs
the device linker), with a comment saying it is precautionary. A CMake that later drives HIP
IPO would make the third row reachable, so pin any claim here to a CMake version. (GooFit)

**Settle any "which flags does this build actually use" question by reading the generated
`build.ninja`, never by grepping the build system's sources.** On GooFit that grep produced
two confident and opposite wrong answers about the IPO question above in successive rounds.
It fails in two ways at once: CMake assigns most per-language variables through
`CMAKE_${lang}_...`, so the expanded name you search for is not in the source at all; and a
`grep -r` over a directory that does not exist prints nothing and looks exactly like a clean
no-match (the module path is wherever *your* `cmake` lives, which for a conda or pip cmake
is not `/usr/share/cmake-<ver>`). `cmake --system-information | grep CMAKE_ROOT` gets you
the right directory; `message(STATUS "[${CMAKE_HIP_COMPILE_OPTIONS_IPO}]")` in a throwaway
`CMakeLists.txt` gets you the value; `build.ninja` gets you the truth. (GooFit)

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

**A HIP-linked shared library, or a host-linked executable, gets no RUNPATH to the ROCm lib
dir.** The ROCm lib dir is one of the HIP compiler's implicit link directories
(`CMAKE_HIP_IMPLICIT_LINK_DIRECTORIES`), so CMake leaves it out of the build RUNPATH; clang
adds it for executables (`h2o4gpu_test` had it) but not for a `SHARED` or `MODULE` target.
A plain C++ executable is not covered either when it links the HIP runtime or rocFFT only
indirectly, through a static library built with HIP sources (ESPResSo's, through heFFTe
with `Heffte_ENABLE_ROCM`); targets that link `hip::host` directly do get the directory.
Such an executable gets `NEEDED librocfft.so.0`/`libamdhip64.so.7` and no RUNPATH entry,
so it runs where the ROCm lib dir is on the loader path and fails with "cannot open shared
object file" where it is not (the pip SDK). Anything else that happened to inject it (h2o4gpu's was clang's
`-fopenmp` link) hides the gap until it is removed. Then, with the pip/TheRock SDK and no
`LD_LIBRARY_PATH`, the module fails to load -- and a Python loader that swallows the
`ImportError` (h2o4gpu's `GPUlib.get()`) silently runs every solver on the CPU while the
tests pass. Set it explicitly on the target:
`BUILD_RPATH $<TARGET_FILE_DIR:hip::amdhip64>` (and `INSTALL_RPATH`), then check
`readelf -d <lib> | grep RUNPATH` and run the GPU path with `LD_LIBRARY_PATH` unset.
CUDA projects usually already do the same for the CUDA library directories, so add the HIP
line beside it.
Whether a `/opt/rocm` package registers its lib dir with ld.so.conf was not verified; do
not rely on it. (h2o4gpu, ESPResSo)

**TheRock's torch wheels cannot be linked from C++ as installed: their RUNPATH points into the
wheel builder's tree.** Linking an executable against `torch/share/cmake/Torch` from such a wheel
warns `librocm-openblas.so.0, needed by libtorch_cpu.so, not found` and fails with undefined
`sgemm_`/`zgetrf_` (and `librocm_sysdeps_liblzma.so.5` for aotriton), while the same wheel works
in Python. `libtorch_cpu.so`/`libtorch_hip.so` carry `RUNPATH
$ORIGIN:/opt/python/cp311-cp311/lib/python3.11/site-packages/_rocm_sdk_devel/lib/host-math/lib:...`,
valid only on the build machine; Python works because the SDK package preloads its libraries. Put
the SDK's `_rocm_sdk_core/lib`, `lib/host-math/lib`, `lib/rocm_sysdeps/lib` and
`_rocm_sdk_libraries_<family>/lib` on `LD_LIBRARY_PATH` (GNU ld also searches it for indirect
dependencies), or into the executable's RPATH with `--disable-new-dtags`. The download.pytorch.org
ROCm wheels instead bundle these libraries next to libtorch with `DT_RPATH $ORIGIN` (an RPATH,
which also reaches indirect dependencies). (wenet, torch 2.8.0+rocm7.12.0)

**A library that interposes the HIP registration entry points must bind them locally.** A shared
library that defines `__hipRegisterFatBinary`/`__hipRegisterFunction`/`__hipRegisterVar`/
`__hipUnregisterFatBinary` to capture its own code objects (for a driver-API module table) and
forwards to the runtime with `dlsym(RTLD_NEXT)` works only while nothing made `libamdhip64` global
first: with default visibility its module constructors' calls resolve through the global scope,
so in a process that already uses HIP (or under `LD_PRELOAD=libamdhip64.so`) the runtime's
definitions win and the library's table stays empty. Give the wrappers
`__attribute__((visibility("protected")))` so the static linker binds the calls inside the
library; `readelf -rW lib.so | grep __hipRegister` should then show no relocations. Test once
with the runtime preloaded. (PhysX)

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

**ld.lld rejects an underlinked shared library that GNU ld accepted.** A HIP executable links
with the ROCm SDK's clang, and TheRock's clang uses ld.lld, which rejects any undefined symbol in
a linked shared library that no input on the link line defines (`undefined reference ...
(disallowed by --no-allow-shlib-undefined)`). The CUDA build links the same executable with GNU
ld, which accepts it. A host library that uses a dependency without linking it (no `NEEDED`
entry, a g++-built library calling fmt) therefore breaks only the ROCm link, and only for
executables that link it directly (often unit tests). Check `readelf -d <lib> | grep NEEDED`, and
confirm with `-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=bfd`: if GNU ld links it, the library is
underlinked. Link the dependency to that library, or add it to the affected HIP targets in the
build files, guarded to the HIP build; a linker flag in a recipe is not a fix. (autoware_universe)

**FindCUDA leaves NOTFOUND variables behind on a machine without CUDA.** When a project calls
the legacy `find_package(CUDA)` and falls back to a HIP branch on `NOT CUDA_FOUND`, FindCUDA still
sets `CUDA_LIBRARIES` (`CUDA_CUDART_LIBRARY-NOTFOUND`) and `CUDA_INCLUDE_DIRS`
(`CUDA_TOOLKIT_INCLUDE-NOTFOUND`). A target line shared by both backends that uses them
(`target_link_libraries(t ${CUDA_LIBRARIES} ...)`, often in tests) fails the generate step: "The
following variables are used in this project, but they are set to NOTFOUND". It does not appear
when configure skips `find_package(CUDA)` (a "use HIP even if CUDA is found" option), so a port
tested only that way can ship a documented command that fails. Empty the FindCUDA result
variables that shared lines use on the HIP branch, or move those lines into the CUDA branch, and
test the documented command on a machine without CUDA. (autoware_universe)

**Build protobuf (and abseil) with the same compiler as the code that includes its generated
headers.** The first parse of a message segfaults inside
`google::protobuf::internal::TcParser::FastMtS1` with a jump to a garbage address (MIGraphX's
`migraphx-driver read model.onnx` crashed on any ONNX file). protobuf v30's table-driven parser
calls its fast-path functions with a clang-only calling convention (`PROTOBUF_CC`), so generated
code and inline headers compiled by the ROCm clang call into a `libprotobuf.a` that g++ compiled
with the default one. Build the dependency prefix with the ROCm clang (for cget, `cget init --cxx
<rocm>/lib/llvm/bin/clang++`) and rebuild. When such a library is loaded into a process that
already has another protobuf (onnxruntime), link it with `-Wl,--exclude-libs,ALL -Wl,-Bsymbolic`
so it binds to its own copy. (wenet)

**On a multi-GPU host a rapids-cmake HIP dependency fails configure with `mismatch between
CMAKE_HIP_ARCHITECTURES and GPU_TARGETS`.** ROCm's `hip-config.cmake` fills the `GPU_TARGETS`
cache variable with one entry per visible device, not per distinct architecture, so four
cards of one kind report `gfx1100;gfx1100;gfx1100;gfx1100` against a single-valued
`CMAKE_HIP_ARCHITECTURES`, and the string comparison in `rapids_hip_init_architectures` (used
by hipMM/RMM) trips although only one arch was requested. It is an artifact of the host, not
a port defect: the identical configure passes with `HIP_VISIBLE_DEVICES=0`. Work around it on
the command line, with no source edit: pass `-DGPU_TARGETS=<arch> -DAMDGPU_TARGETS=<arch>`
alongside `-DCMAKE_HIP_ARCHITECTURES=<arch>`. (HEonGPU, four gfx1100)

**A source-built HIP library and a PyTorch wheel in one process must load the same ROCm
runtime.** A process gets one `libamdhip64.so.7`. When a port builds a HIP library from
source (Warp, a native extension, a JIT runtime) and the project also imports PyTorch, build
the library against the ROCm that the PyTorch wheel brings, not against whatever ROCm the
machine happens to have. IsaacLab on MI300X: AMD Warp built against a TheRock HIP 7.16 plus
`torch==2.11.0+rocm7.2` from download.pytorch.org fails in an order-dependent way. With
torch imported first, Warp prints `CUDA Toolkit 71626.38, Driver 70226.1 (insufficient CUDA
driver version!)` and exposes no GPU; with Warp imported first, torch raises `RuntimeError:
No CUDA GPUs are available`. Neither message names the runtime clash, and both look like a
broken port or a driver problem. Building Warp against apt ROCm 7.2 to match the wheel
stalled (hipCUB `DeviceReduce` compiling for over 25 minutes). What worked: TheRock's Python
packages carry the SDK and a matching PyTorch together. Install `torch==X+rocmV`,
`torchvision==Y+rocmV` and `"rocm[libraries,devel]==V"` from
`https://repo.amd.com/rocm/whl/<gfx-family>/` (with uv, give that index as
`--extra-index-url` next to PyPI as `--index-url`; uv favours the extra index, and the
reverse resolves PyPI's CUDA torch). Then build the library with
`ROCM_PATH=$(rocm-sdk path --root)`, and keep `ROCM_PATH` and
`LD_LIBRARY_PATH=$ROCM_PATH/lib` (prepended) set at run time: a runtime compiler (hipRTC)
finds its headers through `ROCM_PATH`, and torch preloads `libamdhip64`/`libhiprtc` but not
every library a separately built `.so` links (`libhiprtc-builtins.so.7` for Warp). Triage
first: print which `libamdhip64` each import resolved (`/proc/<pid>/maps`, or the version
each library reports) before rebuilding anything. (IsaacLab)

The download.pytorch.org ROCm wheels show the reverse case: they bundle `libamdhip64.so`,
`libhsa-runtime64.so` and friends in `torch/lib`, and the bundled HSA runtime keeps the SONAME
`libhsa-runtime64.so.1`. A library built for a newer system ROCm (onnxruntime's MIGraphX
provider) then fails to load after `import torch` with `libamdhip64.so.7: undefined symbol:
hsa_amd_memory_get_preferred_copy_engine, version ROCR_1`, because the loader reuses torch's older
HSA runtime, and onnxruntime quietly retries on the CPU provider (validation.md). Use a torch built
for the same system ROCm (AMD publishes such wheels at
`repo.radeon.com/rocm/manylinux/rocm-rel-<version>/`; they link the system ROCm instead of
bundling it), and check `/proc/self/maps` for a single `libamdhip64`/`libhsa-runtime64` pair.
(wenet)

**A vcpkg-manifest project needs `autoconf-archive` on the build machine, and a full vcpkg
clone.** A project that moved its dependencies to a vcpkg manifest configures with
`VCPKG_ROOT` set and builds the whole dependency set during configure. If the manifest pulls
`pybind11`, vcpkg builds its own `python3`, whose `libb2` dependency fails `autoreconf` with
a bare `error: building libb2:x64-linux failed with: BUILD_FAILED` unless `autoconf-archive`
is installed; it is routinely missing from the project's documented apt list. Install
`autoconf autoconf-archive automake libtool pkg-config python3-dev zip unzip curl` before
the first configure. And do not clone vcpkg shallow: a `--depth 1` clone fails the first
manifest configure with `vcpkg install failed` and "vcpkg was cloned as a shallow
repository... failed to git show versions/baseline.json" for every port, because
versioned-baseline resolution walks vcpkg's own history. Both look like a project build
failure and neither involves the port. (CubbyFlow)

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

### A local-package option must be a variable only the user sets, not `<Pkg>_DIR`

To let a CMake build that downloads a dependency use a local copy instead (a ROCm LibTorch in
place of a downloaded CUDA one), add one explicit cache variable that only the user sets, such as
`set(LIBTORCH_PATH "" CACHE PATH "...")`; do not infer the request from `<Pkg>_DIR`.
`find_package` writes `<Pkg>_DIR` into the cache itself, so after the first configure
`if(Torch_DIR)` fires on every reconfigure and the download never runs again, and every attempt
to tell the user's value from the cached one misses a case (symlinked checkouts, relative values,
a download directory that did not exist yet). When the variable is set,
`find_package(Torch PATHS ${LIBTORCH_PATH} NO_DEFAULT_PATH)` and skip the download; when it is
empty, leave the original code untouched. A PATH cache entry turns a relative `-D` value into an
absolute path. Details that matter: a set path with no package should stop configure with an
error naming the variable, never fall back to the download; do not load it `QUIET`, which
suppresses the messages that say why a config rejected itself; and a package config caches more
than `<Pkg>_DIR` (LibTorch's caches `Caffe2_DIR` and library paths), so unset the entries that
point into the package before loading the local copy and after, and guard the case where the
local config stops configure with a fatal error (an INTERNAL marker set before loading and
cleared after; when the variable is empty and the marker is set, unset the package's entries).
Test with several configures in one build directory (download then local, local then download),
from the checkout and through a symlink. (wenet, CMake 3.14 and 3.31)

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

### A ROCm wheel's runtime dependency on the SDK, with setuptools and a static `pyproject.toml`

To add `Requires-Dist: rocm>=X.Y,<X.(Y+1)` to a ROCm build's wheel without making `dependencies`
dynamic (which changes how lock files see the project), append the requirement after setuptools
has applied `pyproject.toml`, in a `Distribution.parse_config_files` override: set both
`self.install_requires` and `self.metadata.install_requires` (setuptools copies the former into
the latter inside `parse_config_files`, and Requires-Dist is written from the metadata), and only
when there is something to add, because any reassignment marks the field `Dynamic: requires-dist`
in the CUDA wheel too. Pin the version from what the build used (`<rocm>/.info/version`), not a
hard-coded one. (warp-1.14)

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

Configure once with the project's DEFAULTS as well, because a recipe that passes usually
turns such targets OFF. GooFit's documented HIP build worked only because every recorded
session passed `-DGOOFIT_PYTHON=OFF`; the option defaults ON whenever Python development
files are found, so the build command in the project's own README failed for anyone who ran
it verbatim on an ordinary machine ("Never mark the host `.cpp` files `LANGUAGE HIP`" above
has the flag leak those binding targets hit). (HEonGPU, GooFit)

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

### An nvcc Makefile's link line does not carry over to hipcc verbatim

Porting a hand-written Makefile from nvcc to hipcc, two link-line idioms change meaning, one loudly
and one silently. (1) nvcc splits the argument of `-Xlinker` at commas, clang does not: `-Xlinker
-soname,libfoo.so` reaches ld.lld as one argument and fails with "unknown argument
'-soname,libfoo.so'"; use `-Wl,-soname,libfoo.so` for the HIP branch (plain `-Xlinker -z -Xlinker
nodelete` and `-Xlinker -rpath=dir` are fine). (2) `-fopenmp` on a hipcc link line links LLVM's
`libomp.so`; if the host application or library was built by GCC (`libgomp.so.1`), the process ends
up with two OpenMP runtimes ("One OpenMP runtime per binary" above). When the HIP objects make no
OpenMP call (`nm -u | grep -i omp` is empty), keep `-fopenmp` on the compile line and drop it from
the link line only.

Also: a project that compiles device code with nvcc `-ccbin mpicxx` gets the MPI include path from
the wrapper for free; hipcc does not use a host compiler, so ask the wrapper where `mpi.h` is (`echo
'#include <mpi.h>' | mpicxx -x c++ -E -H -o /dev/null -`, which relies only on the compiler's `-H`
header trace, not on a wrapper option; tested with MPICH) instead of `--showme:compile` (Open MPI
only). Seen on plumed2's cudaCoord plugin, ROCm 10.1 (clang 24), 2026-11. (plumed2)
