## Validation policy

- A port is validated only when the project's real test suite builds and passes on a real AMD GPU for the target arch, with no regression in non-GPU tests.
- Cover both wavefront widths and each OS the project supports: at least one wave64 (CDNA) GPU, one wave32 (RDNA) GPU, and Windows when the project builds there. Any GPU of the right family covers its width; each validates the same branch independently.
- Validating on another architecture should not introduce a NEW commit for anything non-essential; if that arch needs no code change, leave the branch untouched. When a fix IS genuinely necessary (e.g. configurable arch), put it in a NEW commit on top -- NEVER amend a commit another arch has already validated: the amended commit is no longer the one that was tested, so every passed arch has to test again. A new commit is cheap by comparison, and one that changes only documentation, comments or formatting, or leaves the HIP code objects byte-identical, needs no new GPU run.
- A CPU-only docker build (image `rocm/dev-ubuntu-24.04:7.2.4-complete`) proves the code compiles and links under ROCm. It cannot observe any fault-class bug, since no GPU runs, so it is never a validation gate. Do NOT wire it into CI that appears to gate the port: a workflow that proves nothing and emails failure noise at whoever watches the repository is pure cost. Run a CPU-only docker build locally if you want a manual compile check.
- gfx90a and gfx942 are CDNA (wave64); the gfx11xx/gfx12xx parts are RDNA (wave32). A change that passes on one width can still fail on the other via the warp-size class, which is why each arch validates on its own hardware rather than inheriting a result.
- PR-prep gate -- nvcc CUDA-build check: before opening any PR whose claim is "the CUDA/NVIDIA build is unchanged", compile the CUDA path (`USE_HIP=OFF`) with nvcc. No NVIDIA GPU is needed: a dedicated conda env works on a ROCm-only machine (`conda create -y -n cuda-12.8 -c nvidia cuda-toolkit=12.8` -- the full `cuda-toolkit`, not compiler-only `cuda-nvcc`, because the check must LINK; host gcc 13 works. ALWAYS pin the arch, e.g. `-DCMAKE_CUDA_ARCHITECTURES=80`: `native` autodetection on a host with no NVIDIA GPU silently degrades to an ancient arch, and `atomicAdd(double*)` "no instance of overloaded function" is that fingerprint, not a real failure). GPU validation on AMD cannot see a broken CUDA path -- writing "the CUDA path is preserved" in your notes is NOT the same as having compiled it. State it honestly in the PR ("compile-checked with nvcc, not run"). REQUIRED when the port adds a CUDA/HIP-split backend (e.g. `Foo_cuda.cpp` vs `Foo_hip.cpp`): that CUDA file is never compiled during HIP-only porting and can be badly broken. The check must reach the LINK stage (build one real target/demo, not compile-only of a single TU) -- the undefined-reference class (e.g. explicit instantiations that do not match real call-site signatures) only surfaces at link. If a fix leaves the HIP code objects byte-identical, it needs no new GPU run.
  - No conda on the host: the PyPI `nvidia-cuda-nvcc-cu12` 12.8 wheels ship only `ptxas`, `nvvm` and `crt` headers, no `nvcc` driver, so they cannot be the toolkit. Assemble a `CUDA_HOME` from NVIDIA's redistributable archives instead: `https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json` lists each component (cuda_nvcc, cuda_cudart, cuda_cccl, libcublas, libcurand, libcufft, libcusolver, libcusparse, libnvjitlink, cuda_nvtx, cuda_cupti, plus libnvjpeg/libnpp if the build links them) with a sha256; extract them all with `--strip-components=1` into one directory and pass `-DCUDAToolkit_ROOT`. cuDNN and NCCL can come from their wheels after adding the unversioned `.so` symlinks CMake's `find_library` looks for. Old third-party pins may need a scratch-only fix under a new toolkit (oneflow's flash-attention needed `#include <cstdio>` with 12.8); patch the extracted copy, never the fork, and record it. (Source: oneflow, 2026-10-30.)
  - Setup: point `-I` at the project's deps (vcpkg include dir, plus any CUDA-Samples headers it uses -- `helper_cuda.h`, `helper_math.h`, `helper_string.h`) and select the project's non-MATLAB/Python build macro. For a Thrust/CUB project also install `cuda-cccl`: on CUDA 13.x they ship there under `include/cccl/{thrust,cub}` (nvcc finds them automatically, but a host-compiler OpenMP-backend check needs that path on `-I` explicitly, and stdgpu's CMake wants `THRUST_INCLUDE_DIR` pointed at it).
  - For a header-only or template library the changed headers only compile when instantiated: CMake-configure the CUDA backend to generate its config headers, then `nvcc -c` a small TU that `template class`-instantiates the affected containers, rather than compiling headers alone.
  - The newest toolkit is not the whole check: a host TU that touches `__half` can be ill-formed on every CUDA release the project supports and fine on the one installed here, because 12.2 exposed `__half`'s integer converts to host compilers (11.0 through 12.1 hide them, and a 12.2-or-newer build hides them again if it defines `__CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__`). When the port touches half-typed host code, also compile that translation unit (or a repro of its template shape) against an older toolkit's headers -- `conda create -n cuda-12.1-hdr -c nvidia cuda-cudart-dev=12.1 cuda-cccl=12.1` is enough for a `g++ -std=c++17 -I<env>/include` check, no nvcc needed. See "A ROCm-only branch inside a shared template still COMPILES on the CUDA path" in fault-classes.md for what it catches. (marian-dev)
  - The class this catches that a HIP-only build cannot: an unconditional device-header include reaching host translation units. Used on 8 projects in one six-week window; it caught a template-shadow regression in Velvet and an stdgpu regression of exactly that include class. (stdgpu, SCAMP, cuSZ, mahout, lc0, cuPDLPx, TIGRE, Velvet)
  - A torch `CUDAExtension` cannot get this gate from the toolkit alone, so install a CUDA-built torch wheel in a throwaway venv first (the torch-extension bullet below). Where no such wheel can be installed, the check falls back to the host's torch: `torch/extension.h` pulls in the host's installed PyTorch, and on a machine whose only torch is a ROCm build with no CUDA-flavored counterpart, the check dies in the ambient install's headers before it can judge the port. See "CUDA gate for torch-extension ports" below for the fingerprints, the diagnosis, and how to record it. (FaithC, accelerated-scan)
  - A CUDA-path configure failure at `find_package(CUDA <ver> REQUIRED)` ("did not find CUDAConfig.cmake / cuda-config.cmake") on a project whose `cmake_minimum_required` floor is >= 3.27 is a CMake/toolchain incompatibility, not a port regression, if the project still calls the legacy `FindCUDA` module (as opposed to pure `enable_language(CUDA)` + `FindCUDAToolkit`, which is unaffected). CMake 3.27 added policy CMP0146 ("the FindCUDA module is removed"); once the project's minimum-required version reaches or exceeds a policy's introduction version, that policy defaults to NEW and the module self-disables (its file is found on disk but returns immediately) -- confirm with `--debug-find-pkg=CUDA`, which shows "The file was found at .../FindCUDA.cmake" right above the "did not find" error. `-DCMAKE_POLICY_DEFAULT_CMP0146=OLD` does NOT recover it once minimum-required is at/above the policy version (verified with a two-line reproducer CMakeLists.txt); only an in-project `cmake_policy(SET CMP0146 OLD)` before the `find_package(CUDA...)` call works, which would be an edit to code the port does not own. Confirm pre-existing by building the identical upstream base sha with the identical cmake/nvcc/gcc versions: if it fails identically, record the CUDA check as not validated, with the CMP0146 explanation, rather than as a port regression. (arbor)
  - For a torch-extension port (Strategy B), plain nvcc is not enough: the project's `setup.py` picks CUDA vs HIP by reading the installed torch (`hasattr(torch.version, 'hip') and torch.version.hip is not None`, or similar), so the CUDA leg needs a CUDA-*built* torch, not just the toolkit. This is still doable on a host with no NVIDIA GPU, and still compile/link-only: `pip install --index-url https://download.pytorch.org/whl/cu121 torch==<pinned>` into a throwaway venv (pick an older CUDA minor, e.g. cu121, so a prebuilt cp312 wheel exists; the wheel bundles its own `nvidia-*-cu12` shared libs, so no system CUDA runtime is needed for compile/link), point `CUDA_HOME` at the dedicated nvcc conda env, and pin `TORCH_CUDA_ARCH_LIST=8.0` (the `torch.utils.cpp_extension` analogue of `-DCMAKE_CUDA_ARCHITECTURES=80` -- unset, it calls `torch.cuda.get_device_capability()` via autodetection and fails or silently degrades on a host with no NVIDIA GPU). Build a scratch copy of the extension source (`git archive HEAD | tar -x`) rather than the project's own working tree, so the CUDA build's `build/` and `.o` files never land as untracked-and-possibly-unignored noise next to the ROCm ones. Confirm the pin took with `cuobjdump --list-elf` on the built `.so` (expect `sm_80` cubins, one per translation unit) the same way `llvm-objdump --offloading` confirms the HIP gfx target. (faster-gaussian-splatting)
- Cross-OS HIP host-pass check from Windows: a Windows host can syntax-check the Linux HIP host pass of a `.cu` without a Linux box. No root is needed: in WSL, `apt-get download libstdc++-14-dev libgcc-14-dev libc6-dev linux-libc-dev libc6 libcrypt-dev` then `dpkg-deb -x` each into a directory and copy `usr/include` and `usr/lib/gcc` out (`tar -ch` to resolve symlinks). Then run the HIP clang with the TU's own `-D/-I/-include` flags plus `--target=x86_64-unknown-linux-gnu --sysroot=<dir> --gcc-install-dir=<dir>/usr/lib/gcc/x86_64-linux-gnu/14 -nogpulib -x hip --cuda-host-only -fsyntax-only`; `-E -dM` should show `__linux__`, `__GLIBCXX__` and no `_MSC_VER`. Use the same ROCm toolchain as the Linux validators, both the clang major and the HIP headers: in the LichtFeld-Studio `std::pow(__half, __half)` case, clang 23 with the ROCm 7.14 headers accepted the call that clang 24 with the 10.x headers (the Linux build's toolchain) rejected as ambiguous. `__half`'s conversion operators live in ROCm's `amd_hip_fp16.h`, and that run did not separate the compiler from the headers, so a check on an older toolchain proves nothing. Reproduce the Linux error first (negative control) before trusting a clean result. It is syntax evidence only; the Linux validator's real build stays the gate. (LichtFeld-Studio)

## Platforms

A platform is an OS plus a GPU architecture. What matters for coverage follows from the
architecture family -- its wavefront width -- and from the OS. Know a family's wavefront
width before trusting a result on it, because guessing it wrong mis-sizes shared memory
silently.

Ones seen so far, for orientation rather than as a roster:

- gfx90a: MI200-class CDNA2, wavefront 64.
- gfx942: MI300-class CDNA3, wavefront 64. Exercises fp8 paths gfx90a cannot.
- gfx1100: RDNA3 (Radeon), wavefront 32. Watch warp-size assumptions and RDNA occupancy.
- gfx1201: RDNA4 (RX 9070 XT), wavefront 32. On Windows one machine covers wave32 and Windows together.
- gfx1101, gfx1151: RDNA3 / RDNA3.5, wavefront 32.

## Windows: use TheRock ROCm, not the Windows HIP SDK

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

## Windows: TheRock's devel wheel can ship with hipCUB/rocThrust headers missing

`_rocm_sdk_devel`'s `include/` can be present but genuinely empty (0 entries, no
error) while `rocm-sdk init` reports success, and a sibling directory the extractor
staged into can carry an ACL that even the file owner cannot read (`Access is
denied` from a non-elevated PowerShell `Get-ChildItem`, even though `icacls` shows
the parent directory itself is readable). `_devel.tar`, the archive `rocm-sdk init`
would expand, is often already consumed (its `RECORD` entry exists but the file
is gone) and there is no full re-download budget in a validation session. Confirm
before assuming corruption: `tar -tf` the still-present tar (if any) and grep for
the missing library by name -- an absent hit means the wheel build never packaged
it, a present hit narrows it to this host's extraction. Either way it is a
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
validator-side scaffolding to prove the port compiles and runs correctly on real
hardware -- it is not part of the fork commit, and it does not change what the
project's own documented build needs once a host's ROCm installation is intact.

**Prefer repairing the SDK to carrying the scaffolding.** Reinstalling the four
packages together (`rocm`, `rocm-sdk-core`, `rocm-sdk-devel`,
`rocm-sdk-libraries-<gfx>`, all pinned to one date-stamp from the per-arch
nightly index) restored a populated `include/` -- 30 entries, `hipcub/`,
`rocprim/` and `thrust/` among them -- and LC-framework then rebuilt and passed
every gate on gfx1151 with no `-I` scaffolding at all. Two things follow. The
half-expanded tree is a local extraction failure, not a wheel that omits the
libraries, so `tar -tf` finding nothing does not settle it. And a
`validation-failed` recorded against this class says nothing about the port:
re-run it after the host is repaired before treating it as evidence about the
code. Cross-check the layout after any repair, since it moves between SDK
versions -- `hipcc`, the device bitcode and `libexec/hipify` sat under
`_rocm_sdk_core` on 7.13 and under `_rocm_sdk_devel` on 7.14.

## Windows: keep one clang driver across every CMake language, and check the CMake version

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
compiler (see strategy-b-torch.md): torch drives the compile itself and never goes
through CMake's `HIP` language.

## Windows: stage the ROCm runtime next to the .exe, kpack included

On a host whose System32 `amdhip64_7.dll` is broken, `PATH` does not save you -- Windows
resolves System32 first -- so copy the SDK runtime into the directory holding each test
executable. Copy `rocm_kpack.dll` along with it: `amdhip64_7.dll` imports it, and when it
is absent the process dies with a bare `error while loading shared libraries: ?` and exit
127, naming no module at all. That empty `?` is the fingerprint of a missing transitive
import, not of a bad port.

`llvm-objdump -p <binary> | grep "DLL Name"` walks the import table and finds the gap in
seconds; run it on the DLLs too, not just the .exe, since the missing module is usually
one level down.

## Windows: a crash trace without a PDB names the wrong function

**The function names in a Windows crash trace from a binary linked without `/DEBUG` are
the nearest EXPORTED symbol, not the function that was running.** The tell is the
offset: `save_object_data() + 0x214FE7 byte(s)` is two megabytes past that symbol, so it
is not that function. plvs lost three porter sessions on two hosts to such a trace: a
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

## Windows: a CPU crash that Linux never shows is often a heap over-read

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
  `/DEBUG` alone, as in the section above, gives function names but no lines or locals.
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

## Windows: TheRock clang 23 can hang in x86 register allocation

A host C++ TU that compiles in about a second at `-O1` but never finishes at `-O2`/`-O3`
is a compiler fault, not a slow build. Find the function with
`-mllvm -debug-pass=Executions` under a `timeout`: the last lines name the pass and the
function (plvs libelas: `Greedy Register Allocator` on `Elas::gapInterpolation`; at `-O1`
with `-funroll-loops` the same pass crashes instead). `-mllvm -regalloc=basic` on that
one source file keeps the optimization level and compiles it in a second.

## Windows: a torch wheel needs the matching `[device-<gfx>]` extra

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

## An integrated APU can crash the device linker under build parallelism

    clang: error: amdgcn-link command failed due to signal (use -v to see invocation)
    clang: note: diagnostic msg: Error generating preprocessed source(s).

This is resource exhaustion, not a broken toolchain and not bad code. It showed up on a
20-CU APU building 14 template-heavy rasterizer variants at `MAX_JOBS=8`, always in the
device-link step; the same variant alone at `MAX_JOBS=2` built immediately, and the whole
14-variant sweep at `MAX_JOBS=4` passed with no failures. An APU shares one memory pool
between CPU and GPU, so concurrent device links compete with everything else on the box.

Halve the job count before investigating anything else, and do not read a signal-killed
`amdgcn-link` as evidence about the port. Related but distinct from the reboot class: the
same host has hard-reset under heavy parallelism plus sustained GPU load, so a build that
takes the machine down is the same lesson one notch further along.

## Windows: a C++ LibTorch consumer must force the torch_hip.dll import

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
The missing interface option is tracked as `pytorch-windows-torch-hip-include-symbol`.

The tell is the split: Python `import torch` reports devices correctly, because importing
the package loads every DLL, while the standalone C++ executable says there is no GPU, and
kernel tests that do not call `is_available()` first pass on the same run.

Do not read it as "the project does not work on Windows" -- that mistake nearly dropped
Windows support for LichtFeld-Studio. An earlier version of this entry blamed
`lld-link` omitting `.CRT$XCU` from `torch_hip.dll`; that was wrong, and checking the
executable's import table first would have found the real cause.

## Windows: VMM address ranges that were mapped are never reused

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

## Windows: `hipcc.exe` and hiprtc traps for setuptools/Python builds

Found porting CuPy's Windows build. Each one looks like a broken port and is not:

- **`hipcc.exe -c` links the static CRT by default.** The object carries `RuntimeLibrary=MT_StaticRelease`, so linking it with setuptools' `/MD` objects is LNK2038. Pass `-fms-runtime-lib=dll`.
- **No target means gfx906, silently.** Without `--offload-arch` or `HCC_AMDGPU_TARGET`, hipcc on Windows builds for gfx906, and `--offload-arch=native` fails. The symptom is "invalid device function" at launch. Require an explicit target at build time; at run time pass `--offload-arch=<gcnArchName>` of the current device.
- **`hipcc.exe` does not quote what it forwards to clang.** An argument with a space or a quote is split: `-DX=extern "C"` becomes `-DX=extern` plus a file named `C`, and `-IC:\Program Files (x86)\...` becomes three arguments (`clang: error: no such file or directory: 'Files'`). Keep such arguments out of hipcc commands; hiprtc and clang-cl are not affected.
- **hiprtc rejects an absolute program name.** `hiprtcCreateProgram(..., name="C:\...\k.cu")` then `hiprtcCompileProgram` returns `HIPRTC_ERROR_COMPILATION` with an EMPTY log; `k.cu` or `dir\k.cu` compiles the same source. Pass a base name.
- **hiprtc kernels that include C++ standard headers compile against the MSVC STL, and `<complex>` breaks.** Two failures, in order: `MSVC ...\include\complex:575: call to 'signbit' is ambiguous` (MSVC calls `std::signbit(long double)`; hiprtc declares only float and double), then, once that compiles, `ld.lld: error: undefined hidden symbol: _Exp` (MSVC's complex `exp`/`cosh`/`sinh` call the UCRT helpers `_Exp`, `_Cosh`, `_Sinh`, `_FExp`, `_FCosh`, `_FSinh`). TheRock ships no libc++ headers, so dropping the MSVC include dirs is no fix. A pre-included header guarded by `__HIPCC_RTC__ && _MSC_VER` that adds `signbit(long double)` (plus `namespace std { using ::signbit; }`) and defines the six helpers as `__device__ __attribute__((weak))` functions bound to the C names with `__asm__("_Exp")` etc. makes it compile and link; a plain `extern "C" __device__ _Exp` definition is refused ("`__host__ __device__` function cannot overload `__device__` function") because clang's `cuda_wrappers/complex` forces the STL declarations host-device. It must come before the first `<complex>` include. User `RawKernel`/`RawModule` code that does not include that header is still affected.
- **A `friend` naming a `__device__` function template needs `__device__` itself** with the clang in ROCm 7.14 (Linux and Windows): without it the friend no longer matches, and the template loses access (`'x_' is a private member of ...`). NVRTC also rejects the unannotated friend unless `-default-device` is given. Annotate the friend; it is correct on both.
- **The hiprtc DLL's name carries the HIP version, and the digit layout has changed between releases** (`hiprtc0702.dll`, `hiprtc0714.dll` and `hiprtc07013.dll` have all shipped), so it is not a soname-like name and must not be hard-coded: list `<rocm>/bin` for `hiprtc0*.dll` (the `0` excludes `hiprtc-builtins*.dll`).
- **A symbol looked up in your own extension by name must be `__declspec(dllexport)`.** `ctypes.CDLL(__file__)` / `GetProcAddress` sees only exported symbols on Windows; on Linux every non-static function is visible, so this is Windows-only and fails at first use, not at link.
- **`os.add_dll_directory(<rocm>/bin)` is enough** for a Python extension that links the ROCm DLLs, even on a host whose System32 `amdhip64_7.dll` is broken: the extension-load search order puts the added directory before System32, and hiprtc's later by-name loads of `amd_comgr.dll` resolved from the same directory (checked with `GetModuleFileNameW`). No preloading by full path was needed. Keep the returned handle alive.
- **A missing ROCm root does not fail the import; the extension binds the display driver's System32 `amdhip64_7.dll` without any error.** If the extension skips `add_dll_directory` because it cannot find the root, the import still succeeds against the driver's HIP runtime (7.12 on the gfx1151 host). The failure comes later and points elsewhere (cupy: `failed to load hiprtc0712.dll`, a hiprtc name derived from the wrong runtime version). TheRock's `hipconfig.exe` / `rocm-sdk.exe` are console scripts in the venv's `Scripts` directory, which is on PATH only when the venv is activated, so an unactivated `venv\Scripts\python.exe` (from an IDE, a service, or a build with `--no-build-isolation`) finds neither. Look up `shutil.which('hipconfig', path=sysconfig.get_path('scripts'))` when PATH has no hipconfig, and warn by name when no root is found. Verify which DLL loaded with `GetModuleFileNameW`. (Source: cupy.)

## Windows: kernels on different streams run one after another

**A test that asserts a speedup from multi-stream concurrency fails on every Windows host
and passes on Linux.** HIP on Windows does not overlap kernels launched on separate
non-blocking streams: four 1-block 20 ms kernels take 80.5 ms on one stream and 81.3 ms on
four (gfx1151; the same project's concurrency tests failed identically on gfx1101 and
gfx1201). `GPU_MAX_HW_QUEUES` does not change it. Results stay correct; only the overlap
is lost. If a failing test's assertion is a timing ratio between a serial and a
concurrent run, and every functional test passes, this is the cause and the port is not
at fault. (FLAMEGPU2, where a 20-line repro was written.)

## Using a stream after `hipDeviceReset` corrupts the heap (Windows shows it)

**A process that finishes its work and then dies at exit with `0xC0000374` (heap
corruption), `0xC0000409` or `0xC0000005` -- or `exit 127` under MSYS bash -- may be
destroying a stream that `hipDeviceReset()` already tore down.** Projects that call a
reset in a cleanup function while a C++ object owning streams is still in scope (its
destructor runs after the reset) hit it. Both `cudaDeviceReset` and `hipDeviceReset`
document passing a destroyed resource to a later call as undefined behavior, so this is
the application's misuse; CUDA happens to tolerate it, HIP frees the stream object in the
reset and `hipStreamDestroy` then uses it again (valgrind: use-after-free on Linux too,
where glibc aborts only some runs). The port-side fix is to destroy the owning object
before the reset; whether to change the project is the upstream maintainer's call when
a test exercises the order on purpose. Stale `hipFree` and `hipHostFree` fail cleanly
with `hipErrorInvalidValue` in a single-threaded process (in a multithreaded one, see the
next section); stale `hipEventDestroy` returns `hipSuccess` and was not
checked under valgrind. Tell it apart from a missing DLL (which dies before `main`
prints anything) by the work having completed first. A 12-line repro and a parity
request to HIP are on FLAMEGPU2.

## A duplicate `hipFree` in one thread can crash a different thread

When a library frees the same device pointer twice and another thread allocates between
the two frees, the second free can release that thread's buffer. The failure then shows
up somewhere unrelated, and only in some runs: for example an OpenCV `cv::cuda` call
aborting with `(-217) invalid argument` from `cudaGetLastError()`, or a jump to address 0.
plvs hit this once its libelas GPU path was enabled. The upstream code freed two buffers
twice, and on gfx1100 4-5 of 13 dense-stereo SLAM runs died that way, against 0 of 26
runs of three controls (the double free removed, the CPU fallback, no GPU code). On
gfx90a the same head never showed it, so a clean run on one architecture does not clear
the code. Read the code for duplicate frees when a crash kind appears only where a newly
enabled GPU path runs. Separate it from upstream races with the counting method under
"Diagnosing a suspected AMD fault before escalating" (CPU-only control, many runs, one
stack per crash kind). Without core dumps (`ulimit -c 0`), and when gdb slows the run too
much to finish, a 15-line `LD_PRELOAD` handler that calls `backtrace_symbols_fd` on
SIGSEGV/SIGABRT still gives the stack. (plvs)

## A free-memory leak check measures the whole GPU, not your process

`hipMemGetInfo` (like `cudaMemGetInfo`) reports free memory for the device, so a leak
harness that compares free memory across repeated calls also counts every other process
allocating on that GPU. plvs: a 21-call libelas leak check reported 8.00 MB lost on one
image while a SLAM run shared the card, and 0.00 MB once it ran alone, which is the
recorded figure. Run memory-delta checks with nothing else on the device (check
`rocm-smi --showpids`), and repeat any non-zero result alone before calling it a leak.
(plvs)

## Two GPUs visible to one process can crash the runtime

On ROCm 7.14 a process that could see a mixed RDNA3 + RDNA4 pair crashed in the HIP
runtime before any kernel ran. It presents as a launch-time failure of a port that is
correct, so it is easy to spend an afternoon on the wrong suspect. If a machine has
more than one GPU and they are not the same architecture, pin `HIP_VISIBLE_DEVICES`
to exactly one per process and see whether the fault survives.

Do not carry an index in a script. Which index a card holds depends on what is
installed in that machine, so a pinned `HIP_VISIBLE_DEVICES=1` copied from older
notes silently selects a different card, or none. Read the device list at the time
you use it (`rocminfo`, `hipInfo`).

`amd-smi` numbers GPUs in its own order, which need not be HIP's. On a four-W7800 host
`HIP_VISIBLE_DEVICES=2` was amd-smi's GPU 0, the card another job was using, while amd-smi
GPU 2 was HIP device 0. To pick an idle card, match by PCI address: print
`hipDeviceGetPCIBusId` for each `HIP_VISIBLE_DEVICES=<i>`, compare with `amd-smi list`
(BDF) and `amd-smi process`, and read that card's `/sys/class/drm/card*/device/`
`gpu_busy_percent` / `mem_info_vram_used` (the sysfs card number is a third ordering).
(LichtFeld-Studio)

## RCCL collectives need the whole multi-GPU host at once

NCCL->RCCL is a mechanical swap (RCCL is API-compatible and ships in ROCm), but
validating it is unlike validating a kernel: collectives are a multi-GPU feature, so
the test process must hold two or more same-architecture GPUs SIMULTANEOUSLY. Two
consequences for scheduling on a multi-GPU host:

- Parallelising work across a host's GPUs -- one build or test run pinned per
  device with `HIP_VISIBLE_DEVICES` -- is suspended for the duration of a
  collectives run. The run owns every device it uses; another job landing on a
  sibling GPU mid-run contends with it, and both results are suspect.
- The one-device-per-process pin the mixed-arch section above prescribes must be
  LIFTED for the collectives process: give it exactly the same-architecture devices
  it should use (read the device list at run time, never a remembered index), and
  keep mixed-architecture pairs out of one process regardless.

A single-GPU host cannot exercise the collectives path at all; record that as the
arch's limitation rather than letting a collectives-off run stand in for it
silently. Read the device list before concluding that, though: a host remembered
as single-GPU may not be one. What the run has to show is the library announcing
itself and the communicators constructing, plus a training job spanning every
device converging and its model decoding correctly -- a job that starts and does
not crash proves only that the collectives were never called. (marian-dev, on
four gfx1100)

## A passing assertion COUNT is not a list of what was tested

Catch2 stops iterating a test case's remaining SECTIONs once one of them throws,
so a single unported path that aborts hides every section after it. marian-dev's
operator suite reported "284 of 287 assertions; the 3 failures are the sparse
path" across three platforms and two reviews. Fixing the sparse path turned the
same suite into 603 assertions and immediately failed three of the newly reached
ones, in an unrelated operation that had been broken on ROCm from the start.

When a suite has a known-failing section, its pass count is a LOWER BOUND on what
went untested, and "the failures are isolated to X" describes what ran, not the
port. Compare the assertion count against the CUDA build's, or filter the failing
section out and watch whether the total moves. A deferral that makes a test case
abort quietly defers the rest of the file with it. (marian-dev)

## Failures that appear only when the whole suite runs in one process

Two kinds show up on Linux and not on a host that runs one suite per process (as the
Windows runners do), and neither is a port defect:

- gtest death tests (`EXPECT_EXIT`, `EXPECT_DEATH`) under the default `fast` style
  `fork()` without exec. Once an earlier test has initialized the HIP runtime, the child
  can die with SIGSEGV before running the test body, even when the code under test is
  CPU-only; gtest's "Death tests use fork() ... detected N threads" warning is the
  sign. Rerun the test with `--gtest_death_test_style=threadsafe` (re-exec). If it
  passes, the failure was the harness.
- Tests that assume the state of a process-global allocator or cache: they pass alone
  and fail after some unrelated earlier test. Run the test alone, then bisect the
  suites that precede it with `--gtest_filter` to find the one that leaves the state
  behind, and check that the port does not touch either file before you record it.

## One architecture gets wrong numbers while the others pass

A clean build that produces wrong results on exactly one architecture -- an iterative
solver, an LM/Newton fit, an FP regression head -- has two very different causes: a real
bug that only that architecture's timing or width exposes, or floating-point
accumulation divergence. The checks for a real bug are cheap, so they come first:

- An unsynchronized upload: a `cudaMemcpy*Async` on one stream consumed by a kernel on
  another, or an init-time kernel on the null stream racing other streams (both in
  fault-classes.md under streams and events). Run the same call 20 times and compare
  bitwise with the first; add one stream synchronize after the last upload and see
  whether the error disappears.
- A device-to-device copy racing its producer, and an allocation the code reads before
  writing (CUDA often hands back zeros; HIP does not).
- A wavefront-width assumption: compare against a second architecture of the same width.
- Single against double precision: if every GPU variant diverges from the
  double-precision result identically at the same positions, and the double path matches
  the CPU, it is reassociation at a comparison boundary (see "Diagnosing a suspected AMD
  fault before escalating" below).

Only when those are clean is accumulation-order divergence the likely answer; RDNA3.5
(gfx1151) is where it has shown up in iterative solvers. Then record the error magnitude
beside the other architectures' and stop: the comparison that matters is against them,
not against a fix. Stopping at the first step is how one value-head error stayed
recorded as a BLAS-library problem for two months before it turned out to be a
null-stream upload race that was latent on CUDA too (lc0).

## Read what the project's own cross-backend checker actually compares

A project that ships a "compare backend A against backend B" harness makes an attractive gate, but check which outputs it compares before trusting a pass, and check whether the test INPUT can reach the code you changed at all.

Two independent ways such a gate silently proves less than it looks:

- **It may compare a subset of the network's outputs.** lc0's `--backend=check` compares only the value and policy heads; `GetMVal` (moves left) is on the same interface and is never compared. A corrupted moves-left weight therefore passes the gate at 34/34 while the head itself is wrong. Grep the checker for every output accessor the interface exposes and confirm each one appears in a comparison.
- **The chosen test input may not instantiate the code path.** The default lc0 test net (maia-1100) has no moves-left head at all, so no batch count on it can exercise moves-left uploads. Decode the input's own metadata to confirm the feature is present rather than inferring it from the file name -- for lc0, `format.network_format.moves_left` in the gzipped protobuf, which a ~30-line varint walk reads without the generated bindings.

When the gate cannot see the path, find a second observable rather than reporting the gate's pass as coverage. lc0 exposes the missing one behind `--show-movesleft`, which prints the moves-left output in the UCI info line, so the same two backends can be compared on it directly.

The decisive form of this evidence is a BASE CONTROL: rebuild the pre-fix source and show the gate passing while the second observable is wrong. On lc0 the pre-fix build passed 34/34 on a moves-left net and simultaneously reported moves left 72 against the CPU reference's 82 in fp16. That single pair is what turns a reviewer's argued finding into a measured one, and it costs one file's recompile when the fix is one file.

Watch the precision variant too: a gate pinned to the fp32 backend can miss a defect that only exists in the fp16 one, and fp16 is often what the auto-selecting backend actually gives users. lc0 registers `hip` as fp32 and `hip-fp16` as half, and `hip-auto` picks half on every gfx11 part, so the recorded fp32 gate command was not testing the configuration users run. Expect an fp16-vs-CPU comparison to fail tight tolerances on ordinary rounding (policy relative 3.4e-02 here) and read the error MAGNITUDE rather than the pass/fail bit: rounding sits orders of magnitude below a corrupted-constant signature.

## Diagnosing a suspected AMD fault before escalating

Two patterns that each cost a deep investigation before the real cause was found.

- **A "data-dependent, later-data-corrupts-earlier, per-tile" corruption signature is the fingerprint of a REPRODUCER bug, not a codegen fault.** cuSZ chased a suspected miscompile to a BLOCKED state and an IR bisect; the actual cause was the test input -- `np.arange(..., dtype=float32) * (python float)` promotes to float64, so `.tofile()` wrote 8 bytes per element and the tool read the stream as f32. Validate the byte width and dtype of any binary test input before escalating to an ISA bisect or a ROCm bug report. (cuSZ)
- **Triangulate single- against double-precision before blaming the wavefront.** When a warp-collective rewrite shows SP divergence, run a second GPU variant and compare both to the DP oracle. If both GPU variants diverge from DP identically at the same positions, and the DP path is bit-identical to the CPU oracle, it is floating-point reassociation at a comparison boundary -- not a wave-size fault. SCAMP used this to clear a ~0.5 divergence at 10/8093 positions as a threshold-boundary artifact. (SCAMP)
- **A once-per-suite host crash that stress reruns never reproduce: widen the suspected race window instead of rerunning.** If the stack sits in runtime code the port does not touch (`git diff` empty there), read it for a publish-before-register ordering, insert a temporary sleep between the two steps, rebuild, and run a tiny loop. A crash that becomes deterministic, and also happens with CPU-only work, is an upstream race that a fast GPU merely exposes; revert, rebuild and record it. oneflow's `LaunchLazyJob` pushed a graph's SourceTick before `EnqueueNNGraph`, so a tiny graph's finish callback could pop an empty queue: 1 crash in ~550 suite files on two arches, 0 in 70 stress reruns, 5 of 5 with a 20 ms sleep (CPU graphs included). (oneflow)

- **ROCm 10.1 `AMD_LOG_MASK` takes decimal only, and graph replays are invisible to it.** A hex value such as `0x60000` is silently read as nothing (level 4 then prints one banner line), so check a mask on a tiny HIP program before trusting a quiet log. Bits on that runtime: 1 = API calls, 131072 = memory, 2097152 = `ShaderName` per dispatch. A graph's kernel nodes log `ShaderName` once, at `hipGraphInstantiate`, never on replay, and an illegal access logs `Memory Fault Error` with no address at any level (and `dmesg` is usually not readable in a container), so neither logging nor `AMD_SERIALIZE_KERNEL=3` names the kernel that faults inside a graph replay. Varying the replay order changes how often such a fault shows (oneflow: 0 of 12 to 7 of 8 at one commit), so an apparent regression between two commits can be the order hiding it.
- **Before blaming the code under test for a wrong result, save the output, the inputs as read back and the reference, and recompute the reference on the CPU.** A test that checks a GPU result against an in-process GPU reference (torch, or the project's own float64 path) fails the same way when the REFERENCE is the wrong side. oneflow's "corrupt first-graph replay" on two gfx1100 GPUs survived two rounds of graph-capture hypotheses; dumping the arrays showed the graph output matching a CPU product of the read-back inputs to 9e-7 while the float64 reference GEMM on the other GPU came back as garbage, just before the illegal access. Also turn the suspected feature off first: the same fault with HIP graph capture disabled (`ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH=0`, 2 of 2) cleared graphs in two runs. (oneflow)
- **When a wrong result survives a `hipDeviceSynchronize` on every device, probe the HIP copy path itself before suspecting kernels.** Between the steps of the failing script, copy a known pattern host-to-device and back with plain `hipMemcpy` (ctypes is enough) into a host buffer pre-filled with -1, and report the wrong byte ranges. In oneflow's two-GPU nn.Graph run on gfx1100 (ROCm 10.1 SDK), such a probe on device 0 came back with whole 48 KiB pieces never delivered (still -1), with rc 0, after the larger graphs ran on device 1. That probe and torch's reference uploads run no oneflow code at all, but the state that makes them fail can still come from the application's two-device activity, so a clean probe on one device does not clear the application. **Do not attribute a timing-dependent fault from one single-variable condition on one build.** `HSA_ENABLE_SDMA=0` (copies by blit kernels instead of the SDMA engines) first looked decisive there (0 of 10 against 23 of 23 at the default), yet interleaved runs on two builds of the same source, differing only by inert env-gated debug hooks, showed it moving the rate in either direction: 8 of 8 to 0 of 8 on one build, 0 of 8 to 3 of 8 on the other, with the illegal access itself in a blit-kernel copy. The two builds also gave 8 of 8 against 0 of 8 at the default, and the clr queue toggles (`DEBUG_CLR_AQL_BARRIER_OPT`, `DEBUG_CLR_AQL_DEV_QUEUE`, `DEBUG_CLR_ENABLE_KDQ`, `DEBUG_CLR_DOORBELL_SKIP`, `DEBUG_CLR_USE_MOVDIR64B`, 3 of 3 each) and `DEBUG_HIP_DYNAMIC_QUEUES=0` (6 of 8) did not remove it. Before a condition attributes anything, interleave it with its control, at least 8 runs per cell, on more than one build. A newer runtime under the same build is the second cheap test: preloading a later nightly's `libamdhip64.so.7` and `libhsa-runtime64.so.1` (`LD_PRELOAD`; ABI-compatible within the same soname) ran 0 of 13 there. That is consistent with a runtime defect but does not prove one, because a runtime swap also changes timing and an application race (a missing cross-device dependency, a buffer freed while another device still reads it) would react the same way; it takes a reproducer without the application, or the one runtime commit that makes the difference, to place the bug in the runtime. `AMD_LOG_LEVEL=3 AMD_LOG_MASK=256` logs the per-stream SDMA engine assignment (one INFO line per assignment, little overhead). A probe on the legacy null stream returns 906 while another thread's stream is capturing, so place probes outside a capture or run with capture off. (oneflow)
- **A kernel that hangs only in a Debug build: rebuild Debug at `-O2` before blaming an assert or the port.** A Debug HIP build turns two things on at once, `-O0` device codegen and every `assert` (including HIP's own `__hip_check_mask` in the `*_sync` intrinsics). Configure a second Debug tree with `-DCMAKE_HIP_FLAGS_DEBUG="-O2 -g0"` (NDEBUG still unset). If the test passes there, the asserts are not what fires and the hang is `-O0` codegen; find the kernel with `AMD_SERIALIZE_KERNEL=3 AMD_LOG_LEVEL=3` (last launch that never completes) and the PC with the SDK's `rocgdb` after a SIGINT. Run a Debug suite one test per process with a timeout so one hang does not hide the rest. The asserts-on `-O2` tree is also the better Debug evidence for mask and wave64 checks. (LichtFeld-Studio)

## A fault inside a vendored JIT entry point: rebuild-with-a-forced-throw beats a debugger

When a crash's stack bottoms out inside a vendored dependency's own runtime-JIT
call (e.g. `hiprtBuildTraceKernels`, which drives HIP RT's own `hipRTC`/`comgr`
compile of its BVH-builder and trace kernels) rather than in the port's own
device code, do not reach for a debugger or a minidump first -- that is
porter/investigation scope, not a validator's, and it is slow. A cheaper test
that still yields real evidence: read the JIT call's own error-handling source
for a place where a compile failure could be silently swallowed (a `result !=
SUCCESS` check that only acts when an accompanying log string is non-empty is
the classic shape -- an empty log on a genuine failure is common on Windows,
where comgr can fail silently for reasons that have nothing to do with the
actual compile), patch that ONE spot locally to unconditionally `throw` with
the raw result code, rebuild just that translated unit (seconds, not the whole
dependency), and rerun. Two outcomes, both informative and neither requiring a
debugger:

- Your forced exception fires with a real code and message -- you have found
  the actual failure, cheaply.
- The crash is byte-for-byte identical (same fault address relative to the
  containing function, e.g. `hiprtBuildTraceKernels() + 0x894` unchanged
  across rebuilds even though ASLR moves the DLL's base) and your `throw`
  never fires -- you have FALSIFIED that code path as the cause, which is
  just as real a finding: the crash is earlier, or in a different function
  entirely, and no further time should go into that call.

Revert the diagnostic edit before finishing (it is a throwaway, never
committed) and record the falsification, not just "still crashes" -- a bare
re-run tells the next person nothing they did not already know; a rebuild
that proves a specific mechanism is NOT the cause narrows the search space
for whoever picks it up next. Two independent hypotheses can be dispatched
this way inside a single 30-minute retry budget: an environment-variable
hypothesis (does the crash change when a plausible cause is toggled?) and a
source-level hypothesis (does the crash change when the suspected swallowed
error is forced to surface?) are both answerable without a debugger, and a
"both falsified, same relative offset both times" result is a complete,
honest retry outcome -- it is not a reason to escalate to a debugger inside
the validator role. Also check host-global scratch (e.g. a prior session's
`upstream-issues` drafts) for an existing diagnosis of the same dependency's
JIT/loader class before re-deriving one from scratch -- but verify it against
the CURRENT patched source rather than assuming it still applies: a
previously-reported defect in the same call chain may already be fixed in the
project's own vendored-dependency patch, which rules it out rather than
explaining the new crash. (diff-surfel-tracing)

**Read the top frame before forming any hypothesis.** Python's `-X faulthandler`
prints the faulting frame first; `0x0000000000000000 <unknown module>` above a
vendored DLL means a call through a NULL function pointer, which in an
Orochi/hipew-style loader means the library that pointer comes from was never
loaded. That was the whole diff-surfel-tracing crash, after two retries chased
the JIT: TheRock names the runtime compiler `hiprtc07013.dll` (HIP 7.13 -- major
in two digits, minor in three), while HIP RT, Orochi and the port all searched a
fixed list of four-digit names (`hiprtc0714.dll` ... `hiprtc0503.dll`), so hiprtc
silently stayed unloaded and the first `orortcCreateProgram` jumped to address 0.
Any project that `LoadLibrary`s hiprtc (or amdhip64, amd_comgr) by a hard-coded
versioned name has this fault on TheRock: list `${ROCM_PATH}/bin` for the
versioned file instead (`hiprtc0*.dll` -- the `0` excludes
`hiprtc-builtins*.dll`), and compare the names the loader tries against
`ls $ROCM_PATH/bin` before anything else. (diff-surfel-tracing)
- **An intermittent crash seen only in the GPU build needs a CPU-only build control, run as often and under the same load.** A multi-threaded application (SLAM, pipelines with worker threads) can carry an upstream race that the GPU path only changes the timing of. plvs stereo segfaulted in 4 of 9 HIP runs and in 0 of 6 CPU-only runs, which looks like a port fault, but the crash was in CPU line-mapping code (a null `KeyFrame*` in `MapLine::UpdateNormalAndDepth`). Six more of each, run side by side, gave HIP 0/6 and CPU 1/6. Build the control from the same tree with the GPU option off, into a copy of the source tree when the project writes its outputs into the source tree, get a backtrace (`gdb -batch -ex run -ex bt`), and count runs before you decide. A handful of runs per side cannot tell a rate of 25% from a rate of 8%. (plvs)

## CUDA gate for torch-extension ports: the host's installed PyTorch itself can be the wall

On a host whose only installed PyTorch is a ROCm dev build (built from source with `USE_ROCM=1`), nvcc-compiling a torch C++/CUDA extension fails for reasons that have nothing to do with the port. The first choice is still a genuine CUDA-flavored torch wheel in a throwaway venv (see the validation policy above), which gives a full `CUDAExtension` compile and link. Without one, the closest available check is a raw `nvcc -c` of the `.cu` against `torch.utils.cpp_extension.include_paths()` -- and that dies in the ambient install's headers. Two fingerprints of the same root cause:

- The shipped `torch/include` tree was generated for the ROCm build and is missing CUDA-only generated headers, e.g. `#include <c10/cuda/impl/cuda_cmake_macros.h>: No such file or directory` (that file is emitted by cmake only when `USE_CUDA=1`).
- `torch/headeronly/util/complex.h` guards `#include <thrust/complex.h>` with `#if defined(__HIPCC__) || defined(__HIPCC__)` -- a duplicated-token typo, evidently meant `__CUDACC__ || __HIPCC__` -- so under nvcc the include is skipped while `c10/util/complex.h`/`complex_math.h` still reference `thrust::complex` unconditionally under `__CUDACC__`, cascading into ~100 "identifier thrust is undefined" errors on ANY project that includes `torch/extension.h`, regardless of that project's own code (observed on dev build `2.14.0a0+gitb6b444c`).

Neither is a defect in the port: they are defects in the ambient PyTorch install, present identically whether you build the pristine upstream or the ROCm branch. Diagnose it as environmental by checking that the errors bottom out in `torch/headeronly`/`c10` (not the port's own files) and that `grep -n '__builtin_trap\|__trap\|__HIP\|hip[A-Z]\|amdgcn\|USE_ROCM'` over the port's own changed sources is empty. Do not chase this by patching the installed torch headers (that is fixing the environment, not the port). Record the CUDA check as not validated, quoting the missing-header or duplicated-guard error, and, if there is time budget left, substitute a source-level check: diff the port's CUDA-facing code (e.g. the non-ROCm branch of a dual-path file) against upstream and confirm it is byte-for-byte the same logic, only guarded differently -- that is as much passthrough evidence as a real nvcc pass would give without one. (FaithC, accelerated-scan)

Before settling for the diff-only fallback, check whether the project has a SECOND build path that reaches the CUDA-facing device code without going through `torch.utils.cpp_extension`/`CUDAExtension` at all -- a `CUDA_PTX_COMPILATION` target invoked from its own `CMakeLists.txt`, an `enable_language(CUDA)` project that only needs `find_package(CUDAToolkit)` for the device kernels themselves. That path only needs `find_package(Torch)` for include directories (no ROCm/CUDA mismatch, no linking), so it configures and builds against the ambient ROCm torch without incident, and it gives a REAL pinned-arch nvcc compile of the port's touched CUDA headers -- strictly better evidence than a diff, and worth the extra ten minutes of reading the project's CMake before writing off the gate as environmental. Confirm first that the headers the port touches are actually `#include`d by whatever `.cu` files that path compiles, or the pass proves nothing about the port's own edits. (diff-surfel-tracing)

## A project with no test suite: author the harness, then ship it WITH the port

Some projects have no tests at all -- no `tests/`, no CI, no assertions -- so the GPU gate has to be authored. Where the script lives is not bookkeeping, because the commit-message Test Plans name it: a Test Plan is a promise that a reader with the hardware can re-run the command, and a fenced `python3 validate_foo.py` that resolves to nothing in a clone of the fork reads as "there is a validation script you have not found" rather than "this lives elsewhere". Every fenced command in every commit body must be runnable from a clone of the fork.

Default to shipping the harness in the port when it is ONE standalone file importing only the standard library, the framework the project already depends on, and the project itself. That is a script, not an imposed test framework, so "we cannot invent a test framework for someone else's repository" is not a reason to keep it out; and it is the direct answer to "how do I know this works on my machine" for a PR that asks a maintainer to take a whole new back end on trust. Put it beside whatever standalone script the project already ships (its `example/` or `demo/` directory), name it for what it actually checks (a script that asserts `torch.version.hip` is the ROCm check, not a general test), and point at it from the same doc section that documents the ROCm build. Keep ONE copy -- move it, do not copy it, because two copies drift and the drifting one is the one the gate runs. Keep it in your own porting records only when it genuinely cannot ship: it needs fixtures, data, or a framework the project does not have. Never leave it in an untracked scratch directory: a harness written there is lost to every other machine. (diff-surfel-tracing, EnvGS)

## A patch you carry against a pinned dependency: check "unfixed upstream" per entry

A port that pins a third-party dependency and carries fixes against it will want to say so in the README and the patch header, and the natural sentence -- "these are bugs in <dep> and none of them is fixed upstream" -- is a claim about a moving target that nobody re-checks after the first round. Some of the entries may be fixes the dependency has ALREADY made, on commits later than the tag you pinned; you backported them, which is why they are in your patch, and calling them unfixed upstream is wrong in a way a maintainer can disprove in one click. Two other counts drift the same way: how many entries the patch has (count `^--- a/` and the header table, they can disagree) and which project each entry belongs to (a dependency's vendored `contrib/` copy of a third project is that project's bug, not the dependency's).

Check each entry against the dependency's default branch, not the patch as a whole, and record the commit you checked against with the date. Read-only GitHub API fetches of the patched paths plus a grep for the fixed expression is a few minutes for a dozen files, and it partitions the patch into backports (keep the hunk, it is needed at the pinned tag, but say why it is there) and genuine defects (keep the hunk AND report it). Then scope every upstream-visible sentence to the partition it is true of, and prefer sending the reader to the patch header over restating part of it in a README, so there is one place to keep correct. `diff-surfel-tracing` shipped "none of them is fixed at HEAD" over eleven entries; three of them were already on HIP RT's main branch, and it took three review rounds to stop restating the count wrongly. (diff-surfel-tracing)

- **A cascading wall of SegFault/OOM/"no kernel image available" partway through a long serial suite can be one poisoned physical card, not a port defect -- check `rocm-smi --showpids` against `--showmeminfo vram` before concluding anything.** arrayfire on an 8x MI300X host: a full `ctest` run started clean, crashed (SegFault) on an early binary, and every binary after it degraded -- more SegFaults, a "no kernel image is available for execution on the device" (hipErrorNoBinaryForGpu) on a JIT op, then explicit "Device out of memory". `rocm-smi --showmeminfo vram` showed that GPU pinned near its full capacity; `rocm-smi --showpids` showed *no process attached to it at all*, including after `kill -9` of every test binary -- the VRAM was orphaned at the KFD/driver level, not held by a live allocator, and a process crash should not leak GPU memory past its own exit. Re-running the identical suite (fresh JIT disk cache) on a different, verified-idle card on the same host, same build, same commit, came back 131/132 clean with no crash, no OOM, and no "no kernel image" anywhere. The disagreement between the two `rocm-smi` views is the tell: if `--showmeminfo` says a card is nearly full but `--showpids` shows nothing on it, that card's state cannot be trusted for this session -- switch cards to disambiguate before spending time on the port. (arrayfire)

## A torch build used only as a test oracle can break a build two different ways as it drifts

When a project's own tensor/rasterizer/etc. library is libtorch-free and pulls in a ROCm
PyTorch build purely as a parity-test oracle (`find_package(Torch)` in the test target
only), that torch dependency is a moving target across sessions/hosts even though the
port's own source did not change. Two independent fault shapes turned up on the same host,
in the same build, from the same torch version bump (LichtFeld-Studio, torch `2.14.0a0`):

- **Torch's vendored `fmt` shadows the system `fmt` any spdlog build expects.** Recent
  PyTorch ships its own copy of `fmt` under `torch/include/fmt/` for its "stable ABI"
  surface. `torch/include/fmt/core.h` is a small stub that, unless a project explicitly
  opts in, includes only a reduced `base.h` -- but `torch/include/fmt/format.h` is the
  REAL, full, current fmt (e.g. 12.x). Once `-isystem .../torch/include` is on a
  translation unit's search path (added for the parity-oracle link), every `#include
  <fmt/...>` in that TU -- including a completely unrelated `spdlog` header -- resolves to
  torch's copy first, because a compiler's own implicit system directories (where the
  distro's compatible `fmt`/`spdlog` actually live) are always searched last, after every
  explicit `-I`/`-isystem`. If the distro's `libspdlog` was built against an older fmt API
  (e.g. `fmt::basic_format_string`, renamed to `fmt::fstring` in fmt 12), the mismatch
  surfaces as a template-not-found error inside `spdlog/common.h`, nothing about ROCm/HIP
  at all. Confirm by swapping compilers (a different, unrelated compiler hitting the exact
  same error rules out a compiler-version quirk) and by checking whether `torch/include/
  fmt/format.h` exists and reports a newer `FMT_VERSION` than the system's `fmt/core.h`.
  Fix without touching the fork: build `spdlog` from source with `-DSPDLOG_FMT_EXTERNAL=OFF`
  (bundles spdlog's own matched fmt copy) and point `spdlog_DIR` at that install -- the same
  "vendor a known-good copy locally" pattern already used for pinned `glm`/header-only deps.
  Pick the spdlog version by the fmt it bundles, not by whatever is newest in the distro:
  bundled fmt 10.x and older hit `call to consteval function 'fmt::basic_format_string<...>'
  is not a constant expression` from `FMT_STRING` under a recent Clang (seen with a ROCm SDK
  carrying an LLVM 23 dev snapshot), which is a second, unrelated failure that looks like the
  first one and reproduces with no torch include at all. spdlog 1.14.1 bundles fmt 10.2.1 and
  fails; spdlog 1.15.3 bundles fmt 11.x and is clean. Triage both symptoms in a second with
  `clang++ -std=c++20 -fsyntax-only -I<spdlog>/include` on a two-line TU calling
  `spdlog::info("{:.6e}", f)`, instead of rebuilding the project to find out.
- **A libtorch `c10/cuda/*.h` header a project's compat layer already aliases for can grow
  new symbols the alias list has not caught up to.** A CUDA-graph-capture/user-object-retain
  helper added to `c10/cuda/CUDAGraphsC10Utils.h` in a newer torch build referenced
  `cudaGraph_t`/`cudaHostFn_t`/`cudaUserObject_t`/`cudaUserObjectNoDestructorSync`/
  `cudaGraphUserObjectMove`/`cudaUserObjectRelease`/`cudaStreamGetCaptureInfo_v2` literally
  (this header is never hipified for a plain CMake consumer -- torch only hipifies at
  `CUDAExtension` build time), while the project's own compat header already aliased
  *other* symbols from the very same file (`cudaStreamCaptureMode`, `cudaStreamBeginCapture`,
  ...) -- proof the aliasing strategy is right and the coverage merely fell behind a torch
  version bump. This is a real, small, escalatable gap (confirm every missing symbol has a
  `hip*` equivalent in the installed HIP runtime headers before reporting it, so the fix is
  known to be "add N aliases", not an open-ended investigation), and it is not something to
  route around with an older pinned torch -- an older torch only hides the gap for one host's
  one session; the same coverage will fall behind again on the next torch refresh. Send it
  back to the porter with the exact symbol list and confirmed HIP counterparts.
  **Treat the compiler's error list as a lower bound and alias the whole helper, not just the
  names it printed.** Two independent mechanisms hide a missing alias, and only one of them is
  the familiar template one. (a) Clang drops the undeclared-callee diagnostic when one of the
  call's arguments is an *error-typed expression*. A variable declared with a missing type name
  is error-typed only when clang cannot recover its declaration, i.e. when that type name has no
  typo-correction candidate; most missing `cuda*` type names DO have one and clang then reports
  the callee normally -- so **a type name clang does not correct** (here `cudaGraph_t`, printed
  with no "did you mean") **masks every missing function alias in each call passing a value of
  that type**, and those function names appear only in a second wave, after the type aliases are
  added. What pins that to correctability rather than to the argument being a type: `#undef`-ing
  a type whose name typo-corrects leaves the callee reported, redefining that same type to an
  uncorrectable spelling suppresses both callees, and redefining `cudaGraph_t` to a correctable
  spelling brings its callee back -- flip only the correctability and the suppression flips with
  it, in both directions. In that header the unaliased `cudaGraph_t` hid
  both `cudaGraphRetainUserObject(graph, ...)` -- whose arguments are all non-dependent, so
  ordinary lookup would have rejected it at template-definition time -- and
  `cudaStreamGetCaptureInfo_v2(..., &graph, ...)`, whose argument list contains no undeclared
  name at all. An argument that is merely an undeclared *identifier* does not suppress the
  callee when clang can typo-correct it (`cudaGraphUserObjectMove` -> "did you mean
  'hipGraphUserObjectMove'?", and the call around it is still reported), so do not reason about
  this from a synthetic reproducer whose undeclared name has no correction candidate; measure
  on the real header with the probe below. (b) A call whose arguments *are* dependent is
  looked up only at instantiation, so inside a template nobody instantiates it produces no
  error at all while the helper is merely parsed (`cudaUserObjectCreate(&obj, data.get(), ...)`
  on a `std::unique_ptr<T>`). Either way, covering the parameter types alone leaves the body
  half translated. Check rather than reason about it: syntax-only compile the real torch header
  behind the compat header, `#undef`-ing one alias at a time. Removal that still gives rc 0
  means that name is diagnosed only at instantiation; removal that errors means it is required
  at parse time, and if it was still absent from the original error list, something else in the
  same call masked it. (LichtFeld-Studio)

## HIP + C++23 on an older toolchain: `isfinite cannot overload` (and how to stay forward-compatible)

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
| ROCm 7.14 (`7.14.0a20260612`, the last of the old nightly stream) | no |
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

## A cached build-script decision from a wrong env var survives fixing the env var

If a `build.rs` reads an env var (e.g. `ROCM_PATH`) to emit `cargo:rustc-link-search`, but
does not also declare `cargo:rerun-if-env-changed=<that var>`, cargo has no reason to
re-run the script when only the env var changes -- so a build attempted once with a wrong
value (e.g. an old `ROCM_PATH` copied from another host's notes, pointing at a path that
doesn't exist on this one) caches the wrong `-L` search path in `target/debug/build/<crate>-<hash>/output`
and keeps reusing it even after the env var is corrected. The compile step can appear to
pass (a `.rlib`/static-archive target does not need the search path), while the FIRST
step that actually links a final binary -- a `cargo test` binary, not `cargo build` of a
lib -- fails with `unable to find library -l<name>`, which looks like a missing runtime
library rather than a stale cache. Fix: `cargo clean -p <crate>` (not just re-running with
the right env) before rebuilding. Generalizes beyond ROCm: always check `env | grep -i
<toolkit>_path` against what the build script actually expects before trusting old notes
from a different machine -- ROCm/CUDA install layouts (`/opt/rocm` vs a TheRock/pip
`_rocm_sdk_devel` tree vs a conda env) are not portable between hosts, even ones set up by
the same team. (mahout)

## Headless Vulkan on a Linux GPU host: Xvfb plus Mesa's software WSI

A Vulkan app or test that needs a window can run on a headless RDNA host under `xvfb-run`,
but Xvfb has no DRI3, so RADV reports no presentation support and the app silently picks
llvmpipe (which lacks external memory/semaphore interop and skips the interop tests).
`export MESA_VK_WSI_DEBUG=sw VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json` lets
RADV present through Mesa's software path and hides llvmpipe; the GPU still does all the
rendering and interop. Screenshots come from `xwd -root -display :N`. (LichtFeld-Studio)


## HIP and Vulkan in one process can oops the kernel on an old amdgpu-dkms

A process that runs both HIP and RADV shares one amdgpu VM (libdrm hands both the same device
file), and once KFD owns that VM it is a compute VM. amdgpu-dkms 6.16.13 dereferences a NULL
`bo_va` in `amdgpu_gem_va_ioctl` when the Vulkan driver sends `AMDGPU_VA_OP_CLEAR` in a compute
VM: `BUG: kernel NULL pointer dereference, address: 0000000000000038`, `RIP:
amdgpu_gem_va_ioctl+0x731`. From user space it looks like a GPU hang: an iteration counter
stops, the process sits in D state on kill, and every later `open("/dev/kfd")` on every GPU in
the host blocks in `kfd_create_process -> __flush_workqueue`. `amd-smi reset` does not clear
it; only a reboot does. amdgpu-dkms 6.19.14 has the guard (`if (bo_va &&
bo_va->last_pt_update)`). A HIP-only or Vulkan-only process never reaches it, so a headless
run passing proves nothing about the interop path. Check `cat /sys/module/amdgpu/version`
before running any HIP+Vulkan interop app, and do not debug the port for this symptom on an
older driver. (LichtFeld-Studio: GUI training crashed twice at 6.16.13; on 6.19.14 the same
binary trained 5000 iterations cleanly.)

## rocDecode frames wrong on one architecture: swap the VA driver before blaming the port

rocDecode decodes through VA-API, and a TheRock SDK loads its own bundled Mesa VA driver
(`<SDK>/lib/rocm_sysdeps/lib/radeonsi_drv_video.so`), not the distribution's. On gfx90a
with SDK 10.1.0 (bundled "Mesa Gallium 26.3.0-devel", amdgpu-dkms 6.16.6) every H.264,
HEVC and VP9 frame came back wrong in nearly every byte while `rocDecGetDecodeStatus`
said success; gfx1100 with the same SDK was bit-exact. Three checks split it, each on
valid streams only: `strace -f -e trace=openat` shows which render node and which
`*_drv_video.so` the process opens (a wrong card is the other suspect); rerun the same
test binary with `LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri` (the distribution's
Mesa; the bundled libva loads it), which was bit-exact; and decode the clip with the
system `ffmpeg -hwaccel vaapi -hwaccel_device /dev/dri/renderD<n>
-hwaccel_output_format vaapi -vf hwdownload,format=nv12` against a software decode,
which proves the video engine itself. When comparing against software decode, compare
luma only or keep the source's pixel format: `-pix_fmt nv12` on a full-range stream
rescales the range and makes a correct decode look wrong. (LichtFeld-Studio)

## A free-memory-delta test can fail once per process on gfx90a

HIP keeps device memory from some first uses for the life of the process. On gfx90a a
plain program's first pageable 100 MiB `hipMemcpy` round trip took 150 MiB of
`hipMemGetInfo` free memory for good, and later round trips took none. A test that reads
free memory before an allocation and asserts it comes back afterwards can then fail by a
fixed amount on its first run in a process and pass on repeats. Check with
`--gtest_repeat=3` in one process before calling it a leak: a leak grows each
repetition, a first-use cost appears once. (LichtFeld-Studio)
