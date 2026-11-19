# Windows

Everything specific to building and running a HIP port on Windows.

<!-- toc:begin -->
Contents:

- Toolchain install
- CMake and compiler drivers
- Linking
- DLL loading
- Source portability
- Runtime behaviour
- Debugging
<!-- toc:end -->

## Toolchain install

### Use TheRock ROCm, not the Windows HIP SDK

**Build and run against a full ROCm distribution from TheRock (its PyTorch wheels and the
venv around them), never the Windows HIP SDK.** The HIP SDK is a compiler-and-headers
package: its runtime and library set are narrower than a full ROCm, and a build that picks
up its `amdhip64` runtime -- or mixes its libraries with TheRock's -- fails at RUNTIME in
ways that look exactly like a broken port. Symptoms are misleading and widespread rather
than localized: a process exiting 127, "DLL"/"cannot load"/"image not found", or
`hipErrorLaunchFailure` (719) on the FIRST kernel launch.

Triage that class as an environment fault, not a GPU or port fault. Fix the environment
once -- confirm which ROCm the process actually loaded -- and do NOT rebuild in response to
a DLL-load error; rebuilding a correct binary against a broken runtime is the single
biggest time sink on Windows. A port is not "broken on Windows" until it has been run
against a full ROCm distribution.

### TheRock's devel wheel can ship with hipCUB/rocThrust headers missing

`_rocm_sdk_devel`'s `include/` can be present but genuinely empty (0 entries, no
error) while `rocm-sdk init` reports success, and a sibling directory the extractor
staged into can carry an ACL that even the file owner cannot read (`Access is
denied` from a non-elevated PowerShell `Get-ChildItem`, even though `icacls` shows
the parent directory itself is readable). `_devel.tar`, the archive `rocm-sdk init`
would expand, is often already consumed (its `RECORD` entry exists but the file
is gone), and a full re-download is slow. Confirm
before assuming corruption: `tar -tf` the still-present tar (if any) and grep for
the missing library by name -- an absent hit means the wheel build never packaged
it, a present hit narrows it to this machine's extraction. Either way it is a
toolchain gap, not a code fault: it fires identically for any project whose HIP
path needs hipCUB or rocThrust, regardless of what the port touched.

Workaround that unblocked a real validation (LC-framework, gfx1151, hipCUB's
`DeviceScan` and rocThrust's `minmax_element`/`device_ptr`): the three libraries
now live together in the `ROCm/rocm-libraries` monorepo, so a version-matched set
is one sparse checkout away rather than three independently-versioned clones that
can skew against each other (an `ROCm/hipCUB` `main` against an unrelated
`ROCm/rocPRIM` branch produced real compile errors from an API mismatch, not from
the port). `git clone --filter=blob:none --sparse --depth 1
https://github.com/ROCm/rocm-libraries.git && cd rocm-libraries && git
sparse-checkout set projects/hipcub projects/rocprim projects/rocthrust` pulls a
consistent set in under a minute. The `cd` is load-bearing: `git clone` does not
change directory, so without it `git sparse-checkout set` applies to whatever
repository you are standing in and empties its working tree of everything outside
the named paths. Each ships a `*_version.hpp.in` that only a CMake configure step
normally expands (`hipcub_version.hpp`, `rocprim_version.hpp`,
`rocthrust_version.hpp`) -- hand-write the trivial expansion (three `#define`s,
any plausible version number; nothing in this class of build reads the value) and
point `-I` at `projects/hipcub/hipcub/include`, `projects/rocprim/rocprim/include`,
and `projects/rocthrust` (its `thrust/` subdirectory is the include root). This is
local scaffolding to prove the port compiles and runs correctly on real hardware -- it
is not part of the port, and it does not change what the project's own documented build
needs once the ROCm installation is intact.

**Prefer repairing the SDK to carrying the scaffolding.** Reinstalling the four
packages together (`rocm`, `rocm-sdk-core`, `rocm-sdk-devel`,
`rocm-sdk-libraries-<gfx>` on ROCm 7.14, all pinned to one date-stamp from the
per-arch nightly index) restored a populated `include/` -- 30 entries, `hipcub/`,
`rocprim/` and `thrust/` among them -- and LC-framework then rebuilt and passed
every gate on gfx1151 with no `-I` scaffolding at all. Two things follow. The
half-expanded tree is a local extraction failure, not a wheel that omits the
libraries, so `tar -tf` finding nothing does not settle it. And a failed run
recorded against this class says nothing about the port: re-run it after the SDK is
repaired before treating it as evidence about the code. Cross-check the layout after any repair, since it moves between SDK
versions -- `hipcc`, the device bitcode and `libexec/hipify` sat under
`_rocm_sdk_core` on 7.13 and under `_rocm_sdk_devel` on 7.14.

**Why the tree is half-expanded, and why it does not heal.** `rocm-sdk-devel` is
distributed as a `_devel.tar` that expands on first use, and most of its entries are
relative **symlinks** into the sibling runtime packages. Creating those needs
`SeCreateSymbolicLinkPrivilege` -- Developer Mode on, or an elevated shell. Without it the
expansion still "succeeds": the verbatim/hardlinkable files land under `bin/`, and `cmake/`,
`include/`, `lib/`, `libexec/`, `share/` are left **empty**. That is what an empty
`lib/cmake/hip/` actually means, so "TheRock ships no `hip`/`hiprand`/`rocthrust` CMake
packages on Windows" is the wrong conclusion. TheRock's own
`docs/development/windows_support.md` states the requirement. The expander deletes the
tarball as its **last** step, and `get_devel_root()` then short-circuits on "`__init__.py`
exists and no tarball", so **re-running `rocm-sdk init` does nothing** -- it believes it
finished.

Diagnose with the SDK's own check, not by eyeballing directories:

```
python -m rocm_sdk test          # healthy: "OK"; a half-expanded tree gave 1 failure + 12 errors
```

Put the venv's `Scripts/` on PATH first, or `testCLIUsesDevelRootPath` errors spuriously
(it shells out to a bare `hipconfig`).

Repair, from an **elevated** shell -- all four packages pinned to one version, because the
devel tree symlinks into its siblings and cannot be mixed:

```
python -m pip uninstall -y rocm rocm-sdk-core rocm-sdk-devel rocm-sdk-libraries-<gfx>
# then DELETE the leftover _rocm_sdk_* trees by hand: pip removes only files listed in
# each RECORD, and an expanded devel tree is not in any RECORD
python -m pip install --index-url https://rocm.nightlies.amd.com/v2/<gfx>/ \
    "rocm==$V" "rocm-sdk-core==$V" "rocm-sdk-devel==$V" "rocm-sdk-libraries-<gfx>==$V"
python -m rocm_sdk init && python -m rocm_sdk test
```

The package names and the index in that recipe are those of the ROCm 7.13 and 7.14
nightlies; that index is frozen at `7.14.0a20260612`. ROCm 10.x builds are at
`https://nightly.repo.amd.com/rocm/whl-next/`, where `rocm-sdk-devel` is a thin metapackage
and the per-architecture package is `rocm-sdk-device-<gfx>` instead of
`rocm-sdk-libraries-<gfx>`. Substitute those names there, and list what is installed
(`pip list | grep -i rocm`) rather than trusting either set from memory.

A healthy tree then has `lib/cmake/{hip,hip-lang,hiprand,rocprim,rocthrust}/*-config.cmake`,
rocThrust's headers under `include/thrust/`, and `bin/hipcc.exe`. Point the build at
`python -m rocm_sdk path --root`. The per-arch index (`/v2/<gfx>/`) carries release families
the general multi-arch index has already moved past, and the reverse also holds: check both
before concluding which ROCm version is the newest available for Windows.

**Never hand-author CMake configs into the SDK tree.** A `hip-lang-config.cmake` stub
written directly into `site-packages/_rocm_sdk_core/lib/cmake/` was found two months later,
looked like a plausible dated file, and was taken for a shipped one. Fabricated files in a
package directory are indistinguishable from shipped ones -- `pip uninstall` leaves them
behind, since they are in no RECORD, so they even survive a reinstall. If a shim is
genuinely unavoidable, put it in scratch space and pass `-D<name>_DIR=`; check any suspect
file against the owning package's RECORD (`cut -d, -f1 <dist-info>/RECORD | grep lib/cmake`)
before believing it. (HEonGPU)

### A torch wheel needs the matching `[device-<gfx>]` extra

TheRock publishes torch on several indexes and they do not carry the same thing. The
PER-ARCH index (`.../v2/<gfx>/`) can lag a ROCm release behind, so matching a 7.14 SDK
means `whl-staging-multi-arch` -- and the plain multi-arch wheel there carries device
code for only some architectures. The failure is quiet in exactly the wrong way:
`torch.cuda.is_available()` returns True, `get_device_name` and `gcnArchName` read back
correctly, and then the first real kernel raises

    torch.AcceleratorError: CUDA error: device kernel image is invalid   (hipErrorInvalidImage)

Install `torch[device-<gfx>]==<same version>` (it pulls `amd-torch-device-<gfx>` and
`rocm-sdk-device-<gfx>`) and it works. **`hipErrorInvalidImage` on the first kernel of a
correctly-detected device is a missing architecture in the wheel, never a port fault.**
Pin one date-stamp across torch, `rocm-sdk-core/libraries/devel` and the device packages,
because they resolve against each other. (gfx1151, torch 2.12.0+rocm7.14.0a20260519.)

### A dev package neither TheRock nor vcpkg ships

conda-forge ships MSVC-toolchain win-64 builds of common C/C++ libraries (GMP, OpenSSL,
zlib), compatible with clang-cl since both are MS-ABI. Fetch one directly with `curl`
against `api.anaconda.org`'s redirect, with no `conda` or `vcpkg` installation: `.conda`
files are zip archives containing `pkg-*.tar.zst`, and Windows's built-in `tar.exe` extracts
`.zst` natively. When a download fails, check which specific domains are blocked before
concluding the network is down. A library conda-forge does not ship for `win-64` either
(check `api.anaconda.org/package/conda-forge/<name>`'s platform list) may still have its own
historical Windows/MSVC fallback worth finding before assuming a from-scratch port is
needed: NTL's `dosify`/`ResetFeatures`/`mach_desc.win`/`NTL_WINPACK` turned what looked like
"write a Windows build system for a 76-file C++ library" into copying four pre-existing
pieces and compiling directly with clang-cl. (HEonGPU)

## CMake and compiler drivers

**An all-clang Windows HIP build hides three CMake/export faults that MSVC builds never
hit.** (1) `WIN32`-guarded export macros (`#ifdef WIN32` + `#ifdef PROJECTOR_EXPORTS` ->
`__declspec(dllexport)`) export NOTHING: clang predefines `_WIN32`, not `WIN32` (`WIN32` is
CMake's `/DWIN32` in `Platform/Windows-MSVC.cmake` `CMAKE_<LANG>_FLAGS_INIT`, the MSVC-syntax
flag `clang++ -x hip` rejects, so a build that works around that gap loses it), and where a
header does pick `dllimport`, clang drops it on the definition instead of exporting as MSVC
does. The DLL links fine and `ctypes` fails at the first symbol; check `dumpbin /exports` (and
a ~1 KB import lib). Fix: `if (WIN32 AND <hip build>) target_compile_definitions(t PRIVATE
WIN32 <X>_EXPORTS)`; defining `WIN32` also switches every other `#ifdef WIN32` path in the
sources, so run the tests on that build. (2) The vendored `hipify_torch` `cmake/Hipify.cmake`
(still so upstream in ROCm/hipify_torch at 1ea3231) runs `hipify_cli.py` and
`replace_cuda_with_hip_files.py` as the `execute_process` program, relying on the shebang:
on Windows that fails ("Failed to hipify files!"); prefix `${Python3_EXECUTABLE}` from
`find_package(Python3 COMPONENTS Interpreter)`. (3) `get_hipified_list` returns native
backslash paths; fed to `file(GLOB_RECURSE)` they glob from the drive root and configure
runs for many minutes -- normalise each element with `file(TO_CMAKE_PATH)`. (LEAP, windows)

**CMake's HIP language on Windows needs ONE clang driver for every language, and ROCm's
GNU-style `clang++` is the one that works on every CMake version; make the project choose it,
do not document a workaround.** Decision rule: default the project to the GNU-style driver
for every language, which works from CMake 3.29 on; all-`clang-cl` is a valid second choice
from CMake 4.0 (natural when the rest of the build passes MSVC-style flags) and fails on
3.31; any other mix fails, except LibTorch's own setup. "Keep one clang driver across every
CMake language" below has the version table. `Platform/Windows-Clang.cmake`
is include-guarded and picks MSVC-style or GNU-style rules for ALL languages from whichever
language includes it first (MSVC-style if any language's frontend variant is MSVC). So with
cl.exe as CXX, `enable_language(HIP)` dies with "mixes Clang and MSVC" (`check_language(HIP)`
cannot report this: it probes in a separate HIP-only project, so a "no HIP compiler" message
after it means the probe found no `clang++`); with clang-cl in `project(... CXX HIP)` it dies
with "mixed frontend variants"; and with clang-cl in `project(... CXX)` followed by a later
`enable_language(HIP)` the check is skipped and HIP gets cl-style flags that `clang++` rejects
in CMakeTestHIPCompiler (4.4.4: `/DWIN32 /D_WINDOWS /Ob0 /Od /RTC1` on the compile; 3.31.0:
`/machine:x64` on the link). Fix inside the project, before `project()`, guarded by
`CMAKE_HOST_WIN32` and the HIP option: find the HIP `clang++` (`HIPCXX`, else `hipconfig
--hipclangpath` with `HIP_PATH`/`ROCM_PATH`/the Python interpreter's directory as hints -- a
pip-installed ROCm SDK puts `hipconfig.exe` in the venv `Scripts`) and set `CMAKE_CXX_COMPILER`
to it unless the user gave one. Three more Windows gaps travel with it: CMake picks the
default `CMAKE_HIP_ARCHITECTURES` with `rocm_agent_enumerator`, which ROCm does not ship on
Windows, and silently falls back to gfx906 (run `amdgpu-arch` from the clang directory
instead); `find_package(hip)` needs `CMAKE_HIP_COMPILER_ROCM_ROOT` on `CMAKE_PREFIX_PATH`
(there is no `/opt/rocm/bin` on PATH to derive it from); and CMake before 3.29 links a Windows
HIP DLL without `-shared` (`lld-link: error: subsystem must be defined`; 3.28.4 fails, 3.29.2
works). Outside a Visual Studio developer prompt three more fail, and the documented command
should still work there: CMake enables RC for Windows `clang++` and finds `rc.exe` only on
PATH (take the newest from the Windows SDK, `KitsRoot10` in the registry, `bin/10.*/x64`);
FindOpenMP finds `libomp.lib` for Windows `clang++` only through the prompt's `LIB` (preset
`OpenMP_CXX_FLAGS=-fopenmp`, `OpenMP_CXX_LIB_NAMES=""` when every source is HIP and linked with
`-fopenmp`, which adds libomp itself); and pip's isolated build sets `PYTHONPATH` to a
sitecustomize that hides the environment's own site-packages, so a pip-installed ROCm SDK's
`hipconfig.exe` in that same environment prints nothing (`unset(ENV{PYTHONPATH})` in the
Windows HIP configure; CMake and hipify call hipconfig too, and hipify then misreads the ROCm
version). scikit-build-core picks a Visual Studio generator on Windows, which has no HIP
support. Do not use a `[[tool.scikit-build.overrides]]` with `if.failed = true` to retry with
Ninja: any `if.failed` override makes scikit-build-core withdraw its
`prepare_metadata_for_build_*` hooks on every platform (pip then compiles during "Preparing
metadata" and metadata-only queries build the library), and a failing Windows CUDA or CPU build
is retried with Ninja. Instead use a small in-tree PEP 517 backend (`build-backend` +
`backend-path`) that re-exports `scikit_build_core.build` and, only on Windows with the HIP
option set and no generator chosen, appends `-GNinja` to `CMAKE_ARGS` in the `get_requires_*`
and `build_*` hooks; scikit-build-core (1.0 through 1.1.2) then also lists `ninja` as a build
requirement when Ninja is missing, and every other build is untouched. (LEAP, windows-gfx1151)

### Keep one clang driver across every CMake language, and check the CMake version

With CMake's `HIP` language on Windows, all three of `CMAKE_C_COMPILER`,
`CMAKE_CXX_COMPILER` and `CMAKE_HIP_COMPILER` must use the SAME clang driver. Mixing
drivers fails before one TU compiles, with an error that does not name the cause:
`CXX=clang-cl` with `HIP=clang++` makes CMake apply MSVC link rules to a GNU-driver link
line, giving `clang++: error: no such file or directory: '/machine:x64'`.

Which driver to pick depends on the CMake version, so check it first:

- **CMake 4.x: either works.** All-`clang-cl` is fine, and is what a project whose other
  dependencies are MSVC-built will want. (UCM's recorded gfx1151 recipe; re-confirmed on
  dietgpu with CMake 4.3.2.)
- **CMake 3.31: `clang-cl` cannot be the HIP compiler.** `Windows-Clang-HIP.cmake` never
  populates `CMAKE_HIP_SIMULATE_VERSION`, so configure dies inside `Windows-MSVC.cmake`
  with "MSVC compiler version not detected properly". Use the GNU-driver
  `clang`/`clang++` for all three instead -- it still targets the MSVC ABI, so MSVC-built
  dependencies and vendored submodules link normally. (Proven on dietgpu.)

Read that failure as "this CMake is too old", not "clang-cl cannot host HIP". A newer
CMake is one `pip install cmake` away and TheRock's own venv already carries 4.3.2, so
reach for it before restructuring a build around the older constraint.

The one working mixed setup is LibTorch's own: `clang-cl` for C and C++, `clang++` for HIP,
with HIP enabled AFTER C and C++ (`project(... LANGUAGES CXX C)` then
`set(CMAKE_HIP_COMPILER_FORCED TRUE)` + `enable_language(HIP)`) and the HIP rules
overridden to GNU style: compile object, depfile, `-isystem`, the MSVC runtime-library
options, and C++ link rules for non-RDC targets. On CMake 4.3+ also strip the MSVC-style
defaults (`/DWIN32 /O2 ...`) from `CMAKE_HIP_FLAGS` and its per-config variants and map the
MSVC debug-information options to nothing, or clang++ rejects `/DWIN32` and `-Zi`. A
project whose tests link ROCm LibTorch inherits this setup, because `find_package(Torch)`
applies the same rules. See LichtFeld-Studio `cmake/HipCompute.cmake` and pytorch
`cmake/public/LoadHIP.cmake`. (LichtFeld-Studio)

Note the split with PyTorch extension ports, where `clang-cl` IS the required host
compiler (strategy-b-torch.md, "Windows"): torch drives the compile itself and never goes
through CMake's `HIP` language.

### CRT flags, compiler-rt builtins and Git Bash flag mangling in a CMake HIP build

**`MSVC_RUNTIME_LIBRARY` is not known for the HIP language, and disabling it removes the
CRT flag from every language.** Seen with CMake 3.31 and clang-cl; the HEonGPU Windows
builds on CMake 4.4.4 still passed the same flags, and whether 4.x still needs them was not
checked. The `MSVC_RUNTIME_LIBRARY` target property has no entry for `HIP`
(`Help/prop_tgt/MSVC_RUNTIME_LIBRARY.rst` documents C, CXX, CUDA, OBJC, OBJCXX, Fortran
only): `CMake Error ... MSVC_RUNTIME_LIBRARY value 'MultiThreadedDLL' not known for this
HIP compiler` at generate time, on the first HIP target CMake evaluates. It is not a CMP0091
policy issue (OLD and NEW fail identically). `-DCMAKE_MSVC_RUNTIME_LIBRARY=""` (empty, not
merely unset) disables the property mechanism -- but for **every** language, and nothing
then supplies a CRT flag at all: no `/MD`, no `-D_DLL`, in `CMAKE_C_FLAGS`,
`CMAKE_CXX_FLAGS` or `CMAKE_HIP_FLAGS` (read `CMAKE_CXX_FLAGS_RELEASE` out of
`CMakeCache.txt` and confirm). clang-cl's no-flag default is the STATIC CRT, so every object
you build gets `MT_StaticRelease` while conda-forge/vcpkg dependencies are
`MD_DynamicRelease`, and lld-link then refuses the link with `/failifmismatch: mismatch
detected for 'RuntimeLibrary'` naming one of your objects and one of theirs. Pass the CRT
flag yourself to all three languages: `-DCMAKE_HIP_FLAGS="/MD ..."`,
`-DCMAKE_CXX_FLAGS="... -EHsc -MD"`, `-DCMAKE_C_FLAGS="... -MD"`. Fixing only one language
just moves the mismatch to the next pair, so read the two object names in each error and
keep going until they agree. `-Dgtest_force_shared_crt=ON` does NOT solve this for a
FetchContent googletest: googletest only rewrites an existing `/MD` into `/MT`, so where no
CRT flag exists there is nothing for the option to rewrite. (HEonGPU)

**clang-cl does not link the compiler-rt builtins.** Host code using 128-bit integer
arithmetic -- ordinary in crypto and modular-arithmetic libraries -- links fine on Linux
(libgcc/compiler-rt is implicit) and fails on Windows with `lld-link: error: undefined
symbol: __udivti3` (or `__umodti3`, `__divti3`). The library ships with the toolchain: add
`<rocm>/lib/llvm/lib/clang/<ver>/lib/windows` to `LIB` and `clang_rt.builtins-x86_64.lib` to
`CMAKE_EXE_LINKER_FLAGS` and `CMAKE_SHARED_LINKER_FLAGS`. This is toolchain plumbing, not a
port defect. (HEonGPU)

**A leading-slash MSVC flag passed from Git Bash is mangled into a path.**
`-DCMAKE_CXX_FLAGS="/DWIN32 ..."` reaches the compiler as `C:/Program Files/Git/DWIN32`
(MSYS argument conversion), and the build fails with `clang-cl: error: no such file or
directory: 'C:/Program'`. Use the dash spellings, which clang-cl accepts identically: `-MD`,
`-EHsc`, `-DWIN32`. Do not reach for `MSYS_NO_PATHCONV=1`/`MSYS2_ARG_CONV_EXCL='*'` as a
blanket escape -- it is inherited by every child process, so a project script that runs
`git -C /d/path/...` inside the configure step stops resolving its own paths and fails with
a much more confusing error. (HEonGPU)

**CMake gives clang-cl `/EHsc` for C++ sources but not for HIP sources.** With CMake and an
MSVC-style clang (`clang-cl`) as `CMAKE_HIP_COMPILER`, `CMAKE_HIP_FLAGS_INIT` gets no
`/EHsc` (CMake's `Windows-MSVC.cmake` sets `/GR /EHsc` only in `_FLAGS_CXX`). HIP sources
then compile with C++ exceptions disabled, and clang does not reject `try`/`throw` in that
mode: the throw still unwinds, but catch handlers are silently dropped, so every exception
passes the code meant to catch it (tests see `EXPECT_THROW` fail with "C++ exception ...
thrown in the test body"; retry-on-`bad_alloc` loops become dead). It looks like an ABI
problem with multiply inherited exception classes but is not. Check `build.ninja` for
`/EHsc` on HIP objects; fix with
`if(CMAKE_HIP_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC") add_compile_options($<$<COMPILE_LANGUAGE:HIP>:/EHsc>) endif()`
before fetched dependencies are added. That directory-scoped option reaches only the
project's own build: a library whose public API throws, and whose consumers compile its
headers as HIP, must also hand the flag to those consumers, either as an INTERFACE compile
option on the exported target evaluated in the consumer's context
(`target_compile_options(<lib> INTERFACE $<$<AND:$<COMPILE_LANGUAGE:HIP>,$<STREQUAL:$<HIP_COMPILER_FRONTEND_VARIANT>,MSVC>>:/EHsc>)`,
CMake 3.30+, which is written verbatim into the installed targets file) or as a documented
requirement; otherwise a consumer's catch around a library call is dropped the same way.
(HEonGPU: a two-file reproducer isolated it, and the fix took the Windows suite from 190 of
295 passing to 289 of 293.)

### TheRock clang 23 can hang in x86 register allocation

A host C++ TU that compiles in about a second at `-O1` but never finishes at `-O2`/`-O3`
is a compiler fault, not a slow build. Find the function with
`-mllvm -debug-pass=Executions` under a `timeout`: the last lines name the pass and the
function (plvs libelas: `Greedy Register Allocator` on `Elas::gapInterpolation`; at `-O1`
with `-funroll-loops` the same pass crashes instead). `-mllvm -regalloc=basic` on that
one source file keeps the optimization level and compiles it in a second.

## Linking

### CMake's HIP link rule mangles MSVC-style flags and paths

On Windows, CMake's HIP link path routes through `hipcc.exe`, which invokes `clang.exe
--driver-mode=g++` for `--hip-link`. That GCC driver then receives CMake's MSVC-style
response file and flags, and misreads them two ways: it treats bare MSVC linker flags
(`/machine:x64`, `/subsystem:console`) as file paths, and it applies GNU backslash-escaping
to the response file, eating every backslash so `D:\path\torch.lib` arrives as
`D:pathtorch.lib`. CMake also injects `-fuse-ld=lld-link`, which conflicts with `--hip-link`.

**This has nothing to do with `-fgpu-rdc`.** A project with no relocatable device code at all
still hits it if it links HIP through CMake on Windows. Do not conclude you are safe because
you are not using RDC -- that mistake was made once already and cost a round.

Fix: interpose a wrapper on the link step, set as `CMAKE_HIP_LINK_EXECUTABLE`.
`assets/hip_link_win.py` is one. It strips `-fuse-ld=*`, converts `-Xlinker` pairs and bare
MSVC `/flags` to `-Wl,` form, and rewrites `-lXXX.lib` and bare `XXX.lib` to
`-Xlinker XXX.lib` so lld-link resolves them via `LIB`. It is under 100 lines of standard
library Python and carries no project-specific state. **Copy it into the project**
(conventionally `cmake/hip_link_win.py`): the project's CMake calls it, so a pull request
cannot depend on a file outside the project.

Unresolved: the wrapper rewrites the argument list only. It does not read the response file
and does nothing about the backslash loss described above, so either the projects that use
it avoid that symptom another way or the symptom is no longer current. Verify on the next
Windows CMake HIP build before relying on either statement.

Guard the override to Windows. Related and separate: the clang-cl `/Fo` bundler bug, where the
CMake HIP compile rule must emit `-o <OBJECT>` instead of `/Fo<OBJECT>` because the driver
hands the same path to `clang-offload-bundler` as both input and output and Windows refuses to
rewrite a memory-mapped file (ROCm/TheRock#5615). The fix is a compile-rule override after
`enable_language(HIP)`, under the MSVC frontend variant only, of the shape
`set(CMAKE_HIP_COMPILE_OBJECT "<CMAKE_HIP_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")`;
start from the rule your CMake version defines (`CMakeHIPInformation.cmake`) and change only
the output spelling, since the rule's other placeholders vary by version. A project can need the compile-rule fix, the
link wrapper, or both -- they are independent. (alien, origin of the wrapper; LichtFeld-Studio,
which needs the link half.)

### A fetched ROCm-DS dependency has to be built static

Measured on the hipMM 4.0 line. hipMM 5.0.0 does not yet compile on Windows at all
(https://github.com/AMD-Ecosystem/hipMM/issues/11), so the hipMM-specific parts below apply
again once it does; the `LINKER:` and `BUILD_SHARED_LIBS` points hold for any fetched
dependency.

ROCm-DS libraries -- hipMM (the RMM port) and the rapids_logger it fetches transitively --
are built shared by default and neither one works that way under an MSVC-style toolchain.
Two independent defects, both found porting HEonGPU, both in the dependency rather than in
the port:

- **They export nothing.** `rmm/detail/export.hpp` defines `RMM_EXPORT` as
  `__attribute__((visibility("default")))` under `__GNUC__` and as nothing otherwise (its
  comment says only glibc is supported); rapids_logger's macro is unconditionally the ELF
  attribute with no `#else`. The DLLs themselves build and link, so the failure surfaces
  far downstream at the **first executable link**, as `undefined symbol:
  rmm::cuda_stream_view::...` / `rapids_logger::logger::log(...)`. Identify the fault class
  with one command instead of chasing mangling or link order: `llvm-readobj --coff-exports
  rmm.dll` (or `dumpbin /exports`) lists nothing at all for a library whose sources clearly
  define those symbols, and returns 409 entries for the same DLL once
  `WINDOWS_EXPORT_ALL_SYMBOLS` is on. If a shared build is genuinely required, the fix
  is `set_target_properties(<tgt> PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)` on each of
  them (measured to link cleanly), but see below for why static is better.
- **A GNU-only linker option reaches an MSVC-style driver.** rapids_logger applies
  `target_link_options(rapids_logger PRIVATE "LINKER:--exclude-libs,libspdlog")` whenever
  `RAPIDS_LOGGER_HIDE_ALL_SPDLOG_SYMBOLS` is ON, which is its default under
  `BUILD_SHARED_LIBS`. CMake's `LINKER:` prefix translates only the *spelling* of the
  driver's flag-passing convention, never the flag itself, so `--exclude-libs libspdlog`
  arrives verbatim; lld-link warns that it does not know `--exclude-libs` and then treats
  the bare word `libspdlog` as an input file name and fails. Generalize this: any GNU/ELF
  linker idiom behind `LINKER:` (`--exclude-libs`, `--version-script`, `-Bsymbolic`,
  `--as-needed`) passes through untranslated and is a Windows build break waiting to
  happen.

Setting `BUILD_SHARED_LIBS` OFF for the directory scope that fetches them disposes of
both at once -- the export macros stop mattering, and rapids_logger's
`cmake_dependent_option` forces its symbol hiding off for a static build -- and it also
avoids a third problem, that Windows has no RPATH: with shared dependencies every
executable needs `rmm.dll`, `rapids_logger.dll`, `spdlog.dll` and `fmt.dll` copied next to
it before it will even start, which breaks `gtest_discover_tests` during the build itself
(exit `0xc0000135` right after a successful link).

Set it as a **normal variable in that directory scope**, not a cache entry. RMM declares
`option(BUILD_SHARED_LIBS "Build RMM shared libraries" ON)` in its own CMakeLists, and an
`option()` writes the cache, so fetching RMM silently flips every *later* fetch in the
project to shared as well -- that is how a project's googletest turns into `gtest.dll`
halfway through a port. Under CMP0077 NEW (any dependency requiring CMake 3.13+) an
`option()` defers to an existing normal variable and creates no cache entry at all, so the
scope-local `set(BUILD_SHARED_LIBS OFF)` keeps both the dependency and everything fetched
after it on the default static path. Do not "harden" it into
`set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)`: an `OFF` cache entry builds static too,
so it is not what reintroduces the bug, but it overrides a `-DBUILD_SHARED_LIBS` the user
set for their own build and persists in the build tree across reconfigures, including a
later one that no longer takes this branch. Guard the whole thing on `MSVC` (true for clang-cl,
and the correct predicate for "MSVC-style link driver") so Linux and the CUDA path are
untouched. (HEonGPU)

### A C++ LibTorch consumer must force the torch_hip.dll import

**A C++ test that gates on `torch::cuda::is_available()` can fail on Windows against a
port that works perfectly**, because the executable never loads `torch_hip.dll` at all.
The GPU hooks register from static initializers in that DLL, and the MSVC-ABI linker keeps
a DLL import only when something references one of its symbols. A test that calls only
`torch::` and `at::` APIs living in `torch_cpu.dll` references none, so the import is
dropped, the registry stays empty and `is_available()` answers false on a machine whose
GPU is fine. Check with `dumpbin /dependents <exe>` (prefix `MSYS2_ARG_CONV_EXCL="*"` from
Git Bash): `torch_hip.dll` is missing from the list.

PyTorch handles this for CUDA: `caffe2/CMakeLists.txt` puts
`-INCLUDE:?warp_size@cuda@at@@YAHXZ` on `torch_cuda`'s interface link options for MSVC, so
every consumer keeps the DLL. ROCm LibTorch has no such option on `torch_hip`, so a ROCm
consumer has to add it to its own test target:
`target_link_options(<tgt> PRIVATE "/INCLUDE:?warp_size@cuda@at@@YAHXZ")` under `WIN32`.
With it, every torch-gated LichtFeld-Studio test that had failed now runs: 2044/2048 pass on gfx1151 and 2042/2048 on gfx1200.

The tell is the split: Python `import torch` reports devices correctly, because importing
the package loads every DLL, while the standalone C++ executable says there is no GPU, and
kernel tests that do not call `is_available()` first pass on the same run.

Do not read it as "the project does not work on Windows" -- that mistake nearly dropped
Windows support for LichtFeld-Studio. Check the executable's import table before any linker
theory: a plausible one (`lld-link` dropping the `.CRT$XCU` initializers of `torch_hip.dll`)
was chased first and was wrong.

## DLL loading

### Stage the ROCm runtime next to the .exe, kpack included

On a host whose System32 `amdhip64_7.dll` is broken, `PATH` does not save you -- Windows
resolves System32 first -- so copy the SDK runtime into the directory holding each test
executable. Copy `rocm_kpack.dll` along with it: `amdhip64_7.dll` imports it, and when it
is absent the process dies with a bare `error while loading shared libraries: ?` and exit
127, naming no module at all. That empty `?` is the fingerprint of a missing transitive
import, not of a bad port.

`llvm-objdump -p <binary> | grep "DLL Name"` walks the import table and finds the gap in
seconds; run it on the DLLs too, not just the .exe, since the missing module is usually
one level down.

Why `PATH` loses: on Windows the *loader*, not the linker, picks the runtime. It searches
the executable's own directory, then System32, and only then `PATH`, and the Adrenalin
display driver ships its own `amdhip64_7.dll` and `amd_comgr*.dll` in System32. The `_7`
suffix is a major-version name, so every ROCm 7.x maps onto the same filename, and a binary
linked entirely against TheRock still runs on the driver's copy. The System32 copy need not
be broken for this to fail; it only has to be a different vintage than the device bitcode
the build used (measured: a 7.2 driver runtime against a 7.14 TheRock build). The runtime's
internal blit/transfer kernels then fail to JIT-link (`undefined hidden symbol:
__ockl_dm_init_v1`, `__amd_streamOpsWrite`) and the device transfer manager is never
created. The visible symptom is narrow and easy to misread as a hardware limitation:
`hipMemGetInfo` returns `hipErrorInvalidValue` with `free=0 total=0` while
`hipDeviceTotalMem` still works, and kernels may hang or the process may crash at exit.
Known upstream as ROCm/TheRock issues 2019 and 4755. Python/PyTorch processes escape it
because `rocm_sdk.preload_libraries()` loads the runtime by absolute path first, which is
why a torch extension can work on a host where a native CMake project fails.

Two traps when triaging this:

- A minimal reproducer that contains **no device code** does not reproduce it. The failure
  needs the runtime to load a code object, so a ten-line program that only calls
  `hipMemGetInfo` succeeds against the very driver that fails the real binary. Reproduce
  against a binary that actually launches a kernel.
- The API that reports the error is often not the API that caused it. Code in the shape
  `someCall(...); CHECK(hipGetLastError());` reports the *sticky* error, so the blamed
  line may be innocent. Check the call's own return value before naming a culprit.

Do not work around this in the port. Sizing a memory pool from
`hipDeviceProp_t::totalGlobalMem` when `hipMemGetInfo` fails is a source change to shared
code that buys nothing once the right DLLs are in place. (HEonGPU)

### A null top frame above a vendored DLL means a runtime library was never loaded

Read the top frame before forming any hypothesis. Python's `-X faulthandler` prints the
faulting frame first; `0x0000000000000000 <unknown module>` above a vendored DLL means a call
through a NULL function pointer, which in an Orochi/hipew-style loader means the library that
pointer comes from was never loaded. The usual cause is a hard-coded versioned DLL name.
TheRock names the runtime compiler after the HIP version with a digit layout that has changed
between releases (`hiprtc0702.dll`, `hiprtc0714.dll`, and `hiprtc07013.dll` for HIP 7.13:
major in two digits, minor in three), so a fixed list of four-digit names
(`hiprtc0714.dll` ... `hiprtc0503.dll`) finds nothing, hiprtc silently stays unloaded, and
the first `orortcCreateProgram` jumps to address 0. Any project that `LoadLibrary`s hiprtc
(or amdhip64, amd_comgr) by a hard-coded versioned name has this fault: list
`${ROCM_PATH}/bin` for the versioned file instead (`hiprtc0*.dll`; the `0` excludes
`hiprtc-builtins*.dll`), and compare the names the loader tries against `ls $ROCM_PATH/bin`
before anything else. (diff-surfel-tracing: HIP RT, Orochi and the port all searched the
fixed list, and two rounds chased the JIT before anyone read the top frame.)

### `hipcc.exe` and hiprtc traps for setuptools/Python builds

Found porting CuPy's Windows build. Each one looks like a broken port and is not:

- **`hipcc.exe -c` links the static CRT by default.** The object carries `RuntimeLibrary=MT_StaticRelease`, so linking it with setuptools' `/MD` objects is LNK2038. Pass `-fms-runtime-lib=dll`.
- **No target means gfx906, silently.** Without `--offload-arch` or `HCC_AMDGPU_TARGET`, hipcc on Windows builds for gfx906, and `--offload-arch=native` fails. The symptom is "invalid device function" at launch. Require an explicit target at build time; at run time pass `--offload-arch=<gcnArchName>` of the current device.
- **`hipcc.exe` does not quote what it forwards to clang.** An argument with a space or a quote is split: `-DX=extern "C"` becomes `-DX=extern` plus a file named `C`, and `-IC:\Program Files (x86)\...` becomes three arguments (`clang: error: no such file or directory: 'Files'`). Keep such arguments out of hipcc commands; hiprtc and clang-cl are not affected.
- **hiprtc rejects an absolute program name.** `hiprtcCreateProgram(..., name="C:\...\k.cu")` then `hiprtcCompileProgram` returns `HIPRTC_ERROR_COMPILATION` with an EMPTY log; `k.cu` or `dir\k.cu` compiles the same source. Pass a base name.
- **hiprtc kernels that include C++ standard headers compile against the MSVC STL, and `<complex>` breaks.** Two failures, in order: `MSVC ...\include\complex:575: call to 'signbit' is ambiguous` (MSVC calls `std::signbit(long double)`; hiprtc declares only float and double), then, once that compiles, `ld.lld: error: undefined hidden symbol: _Exp` (MSVC's complex `exp`/`cosh`/`sinh` call the UCRT helpers `_Exp`, `_Cosh`, `_Sinh`, `_FExp`, `_FCosh`, `_FSinh`). TheRock ships no libc++ headers, so dropping the MSVC include dirs is no fix. A pre-included header guarded by `__HIPCC_RTC__ && _MSC_VER` that adds `signbit(long double)` (plus `namespace std { using ::signbit; }`) and defines the six helpers as `__device__ __attribute__((weak))` functions bound to the C names with `__asm__("_Exp")` etc. makes it compile and link; a plain `extern "C" __device__ _Exp` definition is refused ("`__host__ __device__` function cannot overload `__device__` function") because clang's `cuda_wrappers/complex` forces the STL declarations host-device. It must come before the first `<complex>` include. User `RawKernel`/`RawModule` code that does not include that header is still affected.
- **A `friend` naming a `__device__` function template needs `__device__` itself** with the clang in ROCm 7.14 (Linux and Windows): without it the friend no longer matches, and the template loses access (`'x_' is a private member of ...`). NVRTC also rejects the unannotated friend unless `-default-device` is given. Annotate the friend; it is correct on both.
- **The hiprtc DLL's name carries the HIP version, and the digit layout has changed between releases** (`hiprtc0702.dll`, `hiprtc0714.dll` and `hiprtc07013.dll` have all shipped), so it is not a soname-like name and must not be hard-coded: see "A null top frame above a vendored DLL" above.
- **A symbol looked up in your own extension by name must be `__declspec(dllexport)`.** `ctypes.CDLL(__file__)` / `GetProcAddress` sees only exported symbols on Windows; on Linux every non-static function is visible, so this is Windows-only and fails at first use, not at link.
- **`os.add_dll_directory(<rocm>/bin)` is enough** for a Python extension that links the ROCm DLLs, even on a host whose System32 `amdhip64_7.dll` is broken: the extension-load search order puts the added directory before System32, and hiprtc's later by-name loads of `amd_comgr.dll` resolved from the same directory (checked with `GetModuleFileNameW`). No preloading by full path was needed. Keep the returned handle alive.
- **A missing ROCm root does not fail the import; the extension binds the display driver's System32 `amdhip64_7.dll` without any error.** If the extension skips `add_dll_directory` because it cannot find the root, the import still succeeds against the driver's HIP runtime (7.12 on the machine where this was seen). The failure comes later and points elsewhere (cupy: `failed to load hiprtc0712.dll`, a hiprtc name derived from the wrong runtime version). TheRock's `hipconfig.exe` / `rocm-sdk.exe` are console scripts in the venv's `Scripts` directory, which is on PATH only when the venv is activated, so an unactivated `venv\Scripts\python.exe` (from an IDE, a service, or a build with `--no-build-isolation`) finds neither. Look up `shutil.which('hipconfig', path=sysconfig.get_path('scripts'))` when PATH has no hipconfig, and warn by name when no root is found. Verify which DLL loaded with `GetModuleFileNameW`. (cupy)

## Source portability

**A buffer that crosses into Python and gets released with `free()` must come from `malloc`.** Old SWIG/NumPy bindings often hand a C++ result to NumPy by overwriting the array's `data` pointer (a custom `T** INPLACE_ARRAY1` typemap passing `&PyArrayObject->data`), with the callee setting `*out = (new std::vector<T>(n))->data()` or a `thrust::host_vector` equivalent. NumPy later `free()`s it. On Linux, libstdc++ hands back the raw `malloc` pointer, so the mismatch passes silently. MSVC's `std::allocator` over-aligns every allocation of 4 KiB or more and returns an offset pointer, so `free()` corrupts the heap. The crash, `0xc0000374` (STATUS_HEAP_CORRUPTION), comes from a LATER, unrelated free: the next solver call or interpreter exit. Small outputs (a few centroids) survive while large ones (per-row labels) do not. Run each solver alone in a fresh process and read the real exit code (MSYS bash reports 127), and do not blame whichever solver happened to run last. Fix: fill a `malloc`/`calloc` buffer (with `thrust::copy` from device iterators straight into the raw pointer). That keeps the handoff contract on every OS. (h2o4gpu: KMeans fit/predict/transform; the failure had first been blamed on the ElasticNet call that ran after it.)

### Linux-only host APIs and LLP64 truncation inside third-party C APIs

**A Linux-only host API can be the entire Windows blocker in an otherwise portable port**,
and it will not surface until the first Windows build, however many Linux architectures
have already validated. The recurring set is `<sys/sysinfo.h>` + `sysinfo()` (replace with
`GlobalMemoryStatusEx()`/`MEMORYSTATUSEX::ullAvailPhys` under `#ifdef _WIN32`, leaving the
POSIX branch untouched), the BSD typedefs `u_int32_t`/`u_int64_t` (the Microsoft headers
have only the standard `uint32_t` spellings, which are the same types on Linux, so just
change them), and GNU-style `-g`/`-O3` appended to `CMAKE_<LANG>_FLAGS_*` (unused
arguments for clang-cl, which already receives `/Zi` and `/O2 /Ob2`, and *fatal* in any
subproject that compiles with warnings as errors -- googletest does; guard them with
`if(NOT MSVC)`). Grep for the whole set at once before the first Windows build rather than
discovering them one failed compile at a time. (HEonGPU)

**Windows is LLP64, so `unsigned long` is 32 bits -- and the dangerous instances are
inside third-party C APIs, not in the port's own declarations.** Grepping the project for
`long` finds nothing when the truncation lives in GMP's `mpz_*_ui` family, NTL's
`conv(ZZ, long)`/`to_long`, or any other library whose "unsigned integer" entry points are
typed `unsigned long`. On Linux those carry 64 bits and the code is correct; on Windows
every value above 2^32 is silently cut to its low half, with no warning, because the
implicit conversion is legal. Crypto, big-integer and modular-arithmetic libraries are
saturated with this shape.

What makes it expensive is that the symptom looks like a GPU or codegen fault. In HEonGPU
the moduli reaching `mpz_mul_ui` were 30-55 bits, so every CRT constant was built for the
wrong modulus chain: fifteen test binaries returned wrong numbers on Windows (gfx1151)
while the identical source passed on three Linux architectures, one of them the same
wavefront width. Nothing threw, nothing crashed, and the results were bit-for-bit
reproducible run to run.

Two things make it cheap to find instead. First, when Windows alone computes wrong values,
grep for the `_ui`/`_si`/`long` entry points of every C library on the host path *before*
suspecting the device: `mpz_.*_ui`, `to_long`, `static_cast<long>`, `%lu`. Second,
bisect host versus device with a ten-line probe rather than by reading kernels -- replicate
the library's host arithmetic in a standalone `clang-cl` program and check it against
Python or `__uint128_t` truth, and run the same operations in a trivial kernel. Both coming
back clean is what proves the fault is in the surrounding host plumbing.

The fix is data-model-independent, not `#ifdef`-ed, so the LP64 result stays bit identical:
`mpz_import(rop, 1, -1, sizeof(uint64_t), 0, 0, &value)` then `mpz_mul`/`mpz_mod`/
`mpz_fdiv_q` instead of the `_ui` call, and `NTL::ZZFromBytes`/`NTL::BytesFromZZ` over the
8-byte little-endian representation instead of `conv(ZZ, long)`/`to_long`. Note that
`mpz_set_ui(x, 1)` and other small literals are fine -- only values that can exceed 2^32
matter. The same width problem appears in dependencies' own constants (`1UL << 40`).
(HEonGPU)

### A `core.autocrlf` mismatch can break `git apply` of a committed patch

An in-tree `git apply` of a committed patch can fail with a real (non-whitespace) hunk
mismatch, on Windows only, if the patch file and the tree it applies to were checked out
under different `core.autocrlf` values at different times in the same working copy (e.g.
because `git submodule update --init`, invoked from inside a build script, ran before a
config change and something else ran after). Both sides being CRLF (or both LF) is what
matters, not which value. Symptom: `patch does not apply` at a real hunk, not just a
"trailing whitespace" warning. Fix locally without touching the tracked patch:
`git config core.autocrlf false` (both in the repo and in the affected submodules) then
force every affected file to be re-materialized from the index (`rm <file>; git checkout --
<file>` -- a plain `git checkout -- .` does not always rewrite a file whose working-tree
bytes already happen to match under the stale convention). (HEonGPU)

## Runtime behaviour

**A `std::stringstream` can fail on a multi-GB payload on Windows only.** Code or tests that
serialize multi-GB objects (FHE bootstrapping keys, large models) into a `std::stringstream`
pass on Linux and fail on Windows with a generic "write failed". Observed with a 4.8 GB
payload under the MSVC STL, whose string buffer did not grow past about 2 GiB. Measure the
payload with a counting `streambuf` before suspecting the GPU port. (HEonGPU)

### Kernels on different streams run one after another

**A test that asserts a speedup from multi-stream concurrency fails on every Windows host
and passes on Linux.** HIP on Windows does not overlap kernels launched on separate
non-blocking streams: four 1-block 20 ms kernels take 80.5 ms on one stream and 81.3 ms on
four (gfx1151; the same project's concurrency tests failed identically on gfx1101 and
gfx1201). `GPU_MAX_HW_QUEUES` does not change it. Results stay correct; only the overlap
is lost. If a failing test's assertion is a timing ratio between a serial and a
concurrent run, and every functional test passes, this is the cause and the port is not
at fault. (FLAMEGPU2, where a 20-line repro was written.)

### VMM address ranges that were mapped are never reused

On TheRock ROCm 7.14 for Windows (amdhip64_7.dll, gfx1151), `hipMemAddressFree` on a range
that once had a chunk mapped (`hipMemCreate`, `hipMemMap`, `hipMemSetAccess`, then
`hipMemUnmap`, `hipMemRelease`) returns `hipSuccess`, but the range is never handed out again:
each later `hipMemAddressReserve` lands higher. After about 240 GiB of cumulative
reservations, reserve logs `failed to create a va range mem object` and then faults with
0xC0000005 inside amdhip64 instead of returning an error; gtest reports an SEH exception and
the process often hangs. A reserve/free loop with no map step reuses one address, so a
short reproducer that skips the map will not show it. Code that sizes VA reservations
generously (tens of GiB per cache) and recreates them dies after a few cycles: the test
passes alone and fails on its Nth repeat or deep into a suite. `AMD_LOG_LEVEL=3` shows the
climbing bases. Run such suites one gtest suite per process, with a per-test fallback, and
treat it as a runtime defect, not a port bug. (LichtFeld-Studio)

## Debugging

### A crash trace without a PDB names the wrong function

**The function names in a Windows crash trace from a binary linked without `/DEBUG` are
the nearest EXPORTED symbol, not the function that was running.** The tell is the
offset: `save_object_data() + 0x214FE7 byte(s)` is two megabytes past that symbol, so it
is not that function. plvs lost three debugging sessions on two machines to such a trace: a
`0xC00000FD` stack overflow whose frames all read
`boost::archive::...oserializer<...KeyFrame>::save_object_data` was diagnosed as
"clang-cl miscompiles boost.serialization into infinite recursion", tried against
static/dynamic boost and `-O0`, and recorded as a toolchain wall. Relinking with
`-Xlinker /DEBUG` (a PDB, no recompile) and feeding the trace's image offsets
(`0x140000000 + offset`) to `llvm-symbolizer --obj=<exe>` showed the truth in one run:
seven frames, ending in `__chkstk` under `Tracking::MonocularInitialization`.

Rerun the binary after relinking -- `/DEBUG` changes the `/OPT` defaults, so the old
trace's offsets do not map onto the new image. A short trace that ends in `__chkstk`
(`_alloca_probe`) is a big stack frame, not recursion: get the frame sizes from the
`movl $imm, %eax` before each `__chkstk` call in `llvm-objdump -dr <file>.obj`. Windows
gives an executable 1 MB of stack and Linux threads get 8 MB, so code that keeps large
objects on the stack (plvs: three nested ~420 KB `Frame` locals) runs on Linux and
overflows on Windows. The fix is a stack reserve on the Windows link
(`add_link_options(LINKER:/STACK:8388608)` under `if(WIN32)`); `std::thread` workers take
their size from the same PE header. A `/STACK` change that "did not help" is worth
checking with `llvm-readobj --file-headers <exe> | grep StackReserve` -- the flag may
never have reached the linker.

### A CPU crash that Linux never shows is often a heap over-read

**An access violation that hits only the Windows build, in CPU code the port did not
touch, and moves between functions from run to run, is worth reading as an
out-of-bounds read before suspecting threads or the toolchain.** On Linux a read a few
KB past the end of a vector usually lands in already-mapped heap memory, returns
garbage and goes unnoticed; the Windows heap commits pages on demand, so the same read
can land on an unmapped page. plvs (TUM-VI stereo, fisheye cameras) crashed 13 of 13
Windows runs while Linux finished most runs, and three upstream CPU bugs were behind
it. The every-run crash (`LocalMapping::CreateNewMapFeatures`, `KeyFrameCulling`) was
`mvuRight`/`mvDepth` sized to the left-image keypoint count but indexed over both
images (code inherited from ORB-SLAM3, so any ORB-SLAM3 derivative has it). Fixing that
allocation still left 3 of 10 runs crashing: a loop-closure keyframe vector shorter
than the point vector it is indexed with (the `LoopClosing` crash, plvs-only) and a
line observation counted twice. Only with all three fixed were 10 of 10 runs clean, so
keep running the sequence after the first fix. AddressSanitizer on a Linux CPU-only
build shows the over-reads in a run that exits 0, which makes it the cheaper search.

Getting file:line costs one rebuild and no admin rights:
- Build a CPU-only tree with `-DCMAKE_CXX_FLAGS="... -g -gcodeview"` and
  `-DCMAKE_EXE_LINKER_FLAGS="-Xlinker /DEBUG"` (the GNU-driver clang++ emits CodeView);
  `/DEBUG` alone, as in the entry above, gives function names but no lines or locals.
- `cdb.exe` without installing anything: download the WinDbg `.msixbundle` (the URL is
  in `https://windbg.download.prss.microsoft.com/dbazure/prod/1-0-0/windbg.appinstaller`),
  `unzip` the `windbg_win-x64.msix` out of it, and `unzip` that; `amd64/cdb.exe` runs
  from the extracted tree.
- Run `cdb -g -G -y <pdb dir> -c "sxd av; sxe -c2 \".lines -e; .ecxr; r; kn 40; dv /t /v; q\" av; g" <exe> <args>`
  (`MSYS2_ARG_CONV_EXCL="*"` under Git Bash). `sxd av` skips first-chance faults the
  program handles; the second-chance command prints the stack with source lines and
  the faulting registers, which show the index and the base pointer of the bad read.
- If several build trees write their executables into the same source-tree directory
  (plvs does), copy each exe and its PDB under a distinct name before building the next
  tree, or a run silently uses the other build.

### A hang can be one kernel that never finishes

**A Windows "hang" that keeps the host blocked in `hipMemcpy` can be one kernel that never finishes.** Confirm it before you hunt a host-side loop: `AMD_LOG_LEVEL=3` shows the last `hipLaunchKernel` followed only by repeated `PAL fence isn't ready! result:3`. A symbol-carrying build that keeps Release codegen locates the call site (clang-cl: `-O2 -Ob2 -DNDEBUG -Z7` plus `-DEBUG` at link). If the hang depends on inlining (`/Ob2` hangs, `/Ob1` passes), check for the long-branch return-address bug (diagnosis.md, "A hang that depends on inlining").
