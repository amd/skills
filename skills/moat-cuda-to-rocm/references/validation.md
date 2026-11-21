# Validation

What counts as validated, what a passing run does not prove, and how to check that the CUDA path still builds.

<!-- toc:begin -->
Contents:

- Validation policy
- Platforms
- The CUDA no-regression compile
- Other compile-only checks
- What a pass does not prove
- A project with no tests
- Validating a rewrite
<!-- toc:end -->

## Validation policy

- A port is validated only when the project's real test suite builds and passes on a real AMD GPU for the target arch, with no regression in non-GPU tests.
- Cover both wavefront widths and each OS the project supports: at least one wave64 (CDNA) GPU, one wave32 (RDNA) GPU, and Windows when the project builds there. Any GPU of the right family covers its width; each validates the same branch independently.
- Validating on another architecture should not introduce a NEW commit for anything non-essential; if that arch needs no code change, leave the branch untouched. When a fix IS genuinely necessary (e.g. configurable arch), put it in a NEW commit on top -- NEVER amend a commit another arch has already validated: the amended commit is no longer the one that was tested, so every passed arch has to test again. A new commit is cheap by comparison, and one that changes only documentation, comments or formatting, or leaves the HIP code objects byte-identical, needs no new GPU run.
- A CPU-only docker build (a `rocm/dev-ubuntu-24.04:<version>-complete` image) proves the code compiles and links under ROCm. It cannot observe any fault-class bug, since no GPU runs, so it is never a validation gate. Do NOT wire it into CI that appears to gate the port: a workflow that proves nothing and emails failure noise at whoever watches the repository is pure cost. Run a CPU-only docker build locally if you want a manual compile check.
- For a GUI, viewer or otherwise interactive port, a standalone synthetic kernel test is a SMOKE CHECK, never a validation. Launch the real binary, drive its default scene, and assert on the state the application itself computes (visible geometry, panel values, a per-frame NaN/extent count read back from the live buffers) -- a headless host cannot do this, so route such a project to a platform with a display rather than substituting a kernel test. Velvet was signed off on four platforms against a synthetic kernel program that never constructed the solver's parameter struct, never entered the solve loop and never opened the window; two defects that made the application useless (all simulation parameters zeroed, and a graphics-interop copy that silently transferred nothing) survived every one of those validations and were found the first time anyone ran the actual executable. If the project ships screenshots or a demo scene, reproducing one IS the test.
- gfx90a and gfx942 are CDNA (wave64); the gfx11xx/gfx12xx parts are RDNA (wave32). A change that passes on one width can still fail on the other via the warp-size class, which is why each arch validates on its own hardware rather than inheriting a result.

- A Windows-LABELED change is NOT automatically inert elsewhere: it can touch a shared header that compiles differently against another platform's ROCm or torch. When one lands after other platforms have passed, REBUILD on those platforms rather than skipping them on "it is Windows-gated" reasoning. Rebuilding is how two such regressions were caught (aihwkit, MMseqs2; build-headers-macros.md, "An arch-specific fix keyed on the OS or the ROCm version").

## Platforms

A platform is an OS plus a GPU architecture. What matters for coverage follows from the
architecture family -- its wavefront width -- and from the OS. Know a family's wavefront
width before trusting a result on it, because guessing it wrong mis-sizes shared memory
silently.

| target | family | wavefront | worth knowing |
|---|---|---|---|
| gfx90a | CDNA2 (MI200 class) | 64 | no graphics pipeline |
| gfx942, gfx950 | CDNA3 and later (MI300 class and later) | 64 | no texture API; exercises fp8 paths gfx90a cannot |
| gfx1100, gfx1101 | RDNA3 (Radeon) | 32 | warp-size assumptions, RDNA occupancy |
| gfx1151 | RDNA3.5 (integrated APU) | 32 | shared system memory, few compute units |
| gfx1200, gfx1201 | RDNA4 (Radeon) | 32 | |

A Windows machine with an RDNA card covers the wave32 and the Windows requirement in one
run.

### `-mwavefrontsize64` on an RDNA card checks lane logic, and cannot judge an application

It is tempting to treat gfx1100 as a stand-in for a wave64 GPU, and half of that works. The
mode is real: on gfx1100 the flag produces genuine wave64 kernels (measured `warpSize` 64,
`__ballot` popcount 64, a width-64 `__shfl_down` reduction summing 64 lanes), so a shuffle
mapping, a ballot shift or a width-32 group reduction can be compared against a naive
version and the comparison MEANS something. Use it for that; it is much faster than waiting
for CDNA hardware and it settled real lane-indexing questions on GPU_IPC.

What it cannot do is judge a whole application, and the reason is a compile-time constant,
not a prebuilt binary. rocPRIM and rocThrust are HEADER-ONLY (`/opt/rocm/lib` holds no
library for either, and both compile with the application's own flags), so there is no
rebuilt rocPRIM to go looking for. What breaks is that rocPRIM takes its wavefront constant
from the target ARCHITECTURE macro and never from the effective wavefront mode.
`rocprim/intrinsics/arch.hpp` (ROCm 7.2, lines 68-77) is literally

```cpp
constexpr unsigned int min_size()
{
#if __HIP_DEVICE_COMPILE__ && !__SPIRV__
    #if ROCPRIM_NAVI
    return 32u;
    #endif
    return 64u;
```

so on an RDNA target under `-mwavefrontsize64` the hardware and `warpSize` say 64 while
`rocprim::arch::wavefront::min_size()` still says 32, and rocPRIM's algorithms size their
shared memory and lay out their per-warp partials for the wrong wave. Measured on GPU_IPC
(gfx1100, ROCm 7.2): `thrust::sequence` + `thrust::sort_by_key` over 38386 uint64 keys
returns a valid permutation at the default width and a CORRUPT one with only
`-mwavefrontsize64` added -- thousands of out-of-range slots and tens of thousands of
duplicates, reproduced twice with different counts (10156/27913 and 4197/33760). Downstream
that is a `hipErrorIllegalAddress` in whichever kernel next indexes with the result, which
reads exactly like a port defect and is not one. The full application died in
`thrust::exclusive_scan` before frame 1.

So a crash in a forced-wave64 RDNA build is UNATTRIBUTED until you separate the two, and the
separation is cheap. Give the rocThrust/rocPRIM translation units `-mno-wavefrontsize64`
per source and leave the project's own kernels at wave64:

```cmake
set_source_files_properties(a.cu b.cu PROPERTIES COMPILE_OPTIONS "-mno-wavefrontsize64")
```

Mixing compiles and links, but it is only SOUND when nothing crossing a TU boundary depends
on the width. The wavefront size is a per-kernel field in the code object rather than a
per-binary one, so a wave32 TU can call a `__device__` function from a `-mwavefrontsize64`
TU under `-fgpu-rdc` and it will run, with the callee's `warpSize` folded to 64 while the
dispatch is 32. Before trusting such a build, confirm that no function reachable across the
boundary uses `warpSize`, `__AMDGCN_WAVEFRONT_SIZE__` or a cross-lane op.

Be equally precise about what the mixed build COVERS: exactly the wave64 TUs. On GPU_IPC
that was five files, of which `PCG_SOLVER.cu` is the only one outside the three rocThrust
TUs using both the compat shuffle and the compat ballot, so the run does prove both
compat-header lane fixes execute on 64-lane hardware; the four remaining wave64 TUs contain
no cross-lane ops at all, which is also why the boundary condition above holds there. It
proves nothing about the files given `-mno-wavefrontsize64`: GPU_IPC's MASPreconditioner is
one of them, so its bank reductions were NOT covered. Record the result as what it is,
evidence that specific wave64 paths execute, not a wave64 validation; that still needs a
gfx90a or gfx942 run. (GPU_IPC)

**Proving a binary really carries device code for the target architecture: use
`llvm-objdump --offloading`, since `roc-obj-ls` does not work from the pip-installed SDK.**
`roc-obj-ls <binary>` lists the embedded code objects, but it is a packaged-ROCm script. On
an SDK-wheel layout (no `/opt/rocm` at all) the wheel installs the console script on PATH
and its entry point does not exist, so the whole `roc-obj*` family dies with `ImportError:
cannot import name 'roc_obj_ls' from 'rocm_sdk_core._cli'` instead of command-not-found. The
portable equivalent is `$(hipconfig --hipclangpath)/llvm-objdump --offloading <binary>`,
which prints one `hipv4-amdgcn-amd-amdhsa--gfxNNNN` bundle per device translation unit. Do
not conclude that a build is untargeted, or that the ROCm install is broken, because
`roc-obj-ls` tracebacks or is missing. `hipconfig --hipclangpath` also gives the
`CMAKE_HIP_COMPILER` value to pass when CMake cannot find the HIP compiler on such a
machine, which keeps a documented configure command runnable outside `/opt/rocm`.
(3DUNDERWORLD-SLS-GPU_CPU on gfx942.)

## The CUDA no-regression compile

**Compile and link the CUDA path with nvcc before claiming the CUDA build is unchanged.**
GPU validation on AMD cannot see a broken CUDA path, and writing "the CUDA path is preserved"
is not the same as having compiled it. No NVIDIA GPU is needed. The check is REQUIRED when
the port adds a CUDA/HIP-split backend (`Foo_cuda.cpp` beside `Foo_hip.cpp`: the CUDA file is
never compiled during HIP-only work and can be badly broken), and when it changes how
`__device__` function bodies are compiled or moves them into a header, because a host TU
that reaches that header then has to parse device code (build-cmake-link.md, "Relocatable
device code"). It must reach the LINK stage: build one real target or demo, not a single TU,
because the undefined-reference class (explicit instantiations that do not match real
call-site signatures) only surfaces there. Say what was done in the PR ("compile-checked with
nvcc, not run"). A later fix that leaves the HIP code objects byte-identical needs no new GPU
run. The class this catches that no HIP build can is an unconditional device-header include
reaching host translation units. (stdgpu, SCAMP, cuSZ, mahout, lc0, cuPDLPx, TIGRE, Velvet;
in HEonGPU a device-body header passed two reviews and a full GPU validation and the nvcc
check found it in one run, together with a cuRAND include a compat header had put in front of
the CUDA runtime headers.)

**A CUDA toolkit for the check on a machine with no NVIDIA GPU.** A dedicated conda env
works: `conda create -y -n cuda-12.8 -c nvidia cuda-toolkit=12.8` (the full `cuda-toolkit`,
not compiler-only `cuda-nvcc`, because the check must link; host gcc 13 works). ALWAYS pin the
arch, e.g. `-DCMAKE_CUDA_ARCHITECTURES=80`: `native` autodetection on a host with no NVIDIA
GPU silently degrades to an ancient arch, and `atomicAdd(double*)` "no instance of overloaded
function" is that fingerprint, not a real failure.

- No conda: the PyPI `nvidia-cuda-nvcc-cu12` 12.8 wheels ship only `ptxas`, `nvvm` and `crt` headers, no `nvcc` driver, so they cannot be the toolkit. Assemble a `CUDA_HOME` from NVIDIA's redistributable archives instead: `https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json` lists each component (cuda_nvcc, cuda_cudart, cuda_cccl, libcublas, libcurand, libcufft, libcusolver, libcusparse, libnvjitlink, cuda_nvtx, cuda_cupti, plus libnvjpeg/libnpp if the build links them) with a sha256; extract them all with `--strip-components=1` into one directory and pass `-DCUDAToolkit_ROOT`. cuDNN and NCCL can come from their wheels after adding the unversioned `.so` symlinks CMake's `find_library` looks for. Old third-party pins may need a scratch-only fix under a new toolkit (oneflow's flash-attention needed `#include <cstdio>` with 12.8); patch the extracted copy, never the port, and record it. (oneflow)
- Setup: point `-I` at the project's deps (vcpkg include dir, plus any CUDA-Samples headers it uses -- `helper_cuda.h`, `helper_math.h`, `helper_string.h`) and select the project's non-MATLAB/Python build macro. For a Thrust/CUB project also install `cuda-cccl`: on CUDA 13.x they ship there under `include/cccl/{thrust,cub}` (nvcc finds them automatically, but a host-compiler OpenMP-backend check needs that path on `-I` explicitly, and stdgpu's CMake wants `THRUST_INCLUDE_DIR` pointed at it).
- For a header-only or template library the changed headers only compile when instantiated: CMake-configure the CUDA backend to generate its config headers, then `nvcc -c` a small TU that `template class`-instantiates the affected containers, rather than compiling headers alone.

**Diff the PTX to prove the CUDA path is byte-restored, not merely compiling.** When a fix
is meant to put the CUDA side back exactly as upstream had it, compare the PTX of the port
against the PTX of the pre-port sources; it is much stronger than "the build went green"
and costs one command per translation unit:

```bash
nvcc -arch=sm_80 -ptx -I src/lib -o after.ptx src/lib/<X>.cu
git archive <merge-base> src/lib | tar -x -C base
nvcc -arch=sm_80 -ptx -I base/src/lib -I src/lib -o before.ptx base/src/lib/<X>.cu
diff before.ptx after.ptx
```

Give the extracted base tree the PORT's include path as a fallback (`-I src/lib` second)
when the port added headers the base does not have; the base's own `-I` comes first so its
versions still win and the diff stays honest. (3DUNDERWORLD-SLS-GPU_CPU)

**With a conda toolkit, legacy `find_package(CUDA)` does not find the headers, and many
projects then build with CUDA silently OFF.** conda's `cuda-toolkit=12.x` puts
`cuda_runtime.h` under `<env>/targets/x86_64-linux/include/`, not `<env>/include/`. The
deprecated `FindCUDA` module (projects that predate `enable_language(CUDA)`) looks in
`CUDA_TOOLKIT_ROOT_DIR/include` and reports `Could NOT find CUDA (missing:
CUDA_INCLUDE_DIRS) (found version "12.8")` although `nvcc --version` works. The trap is what
happens next: projects commonly write `find_package(CUDA)` / `if (CUDA_FOUND) ... else()
set(USE_CUDA OFF) endif()`, so the configure SUCCEEDS, the build succeeds, and NOT ONE CUDA
file was compiled. A `cmake --build` that returns 0 is worthless evidence here. Confirm the
project's `Using CUDA: <version>` line (or its equivalent) in the configure output, or grep
`compile_commands.json` for `nvcc`, before recording the check as passed. Passing
`-DCUDA_INCLUDE_DIRS=` does not help (FindCUDA recomputes it), and `-DCMAKE_CUDA_COMPILER=`
is ignored by legacy `FindCUDA` ("Manually-specified variables were not used"); it matters
only once the project calls `enable_language(CUDA)`. Two fixes that worked:

- Point the legacy variables at the `targets` subtree:
  `-DCUDA_TOOLKIT_ROOT_DIR=<env>/targets/x86_64-linux`,
  `-DCUDA_TOOLKIT_INCLUDE=<env>/targets/x86_64-linux/include`,
  `-DCUDA_CUDART_LIBRARY=<env>/lib/libcudart.so`, with `<env>/nvvm/bin` on `PATH` so nvcc
  finds `cicc`. (CubbyFlow)
- Build a throwaway directory of symlinks (`bin -> <env>/bin`,
  `include -> <env>/targets/x86_64-linux/include`, `lib` and `lib64 -> <env>/lib`) and point
  `-DCUDA_TOOLKIT_ROOT_DIR` at it. (3DUNDERWORLD-SLS-GPU_CPU)

Such a project has no `CMAKE_CUDA_ARCHITECTURES` to pin, so pin the arch with
`-DCUDA_NVCC_FLAGS="-arch=sm_80"` instead.

**The newest toolkit is not the whole check when the port touches `__half` in host code.** A
host TU that touches `__half` can be ill-formed on every CUDA release the project supports
and fine on the one installed, because 12.2 exposed `__half`'s integer converts to host
compilers (11.0 through 12.1 hide them, and a 12.2-or-newer build hides them again if it
defines `__CUDA_FP16_DISABLE_IMPLICIT_INTEGER_CONVERTS_FOR_HOST_COMPILERS__`). Also compile
that translation unit (or a repro of its template shape) against an older toolkit's headers:
`conda create -n cuda-12.1-hdr -c nvidia cuda-cudart-dev=12.1 cuda-cccl=12.1` is enough for a
`g++ -std=c++17 -I<env>/include` check, no nvcc needed. See "A ROCm-only branch inside a
shared template still COMPILES on the CUDA path" in build-headers-macros.md for what it
catches. (marian-dev)

**A legacy `FindCUDA` project fails the CUDA configure under CMake 3.27 or newer; that is
CMP0146, not a port regression.** Symptom: `find_package(CUDA <ver> REQUIRED)` reports "did
not find CUDAConfig.cmake / cuda-config.cmake" on a project whose `cmake_minimum_required`
floor is 3.27 or higher and which still calls the legacy `FindCUDA` module (pure
`enable_language(CUDA)` + `FindCUDAToolkit` is unaffected). Cause: policy CMP0146 removes the
module, and once the project's minimum version reaches the policy's introduction version the
policy defaults to NEW and the module returns immediately. Confirm with
`--debug-find-pkg=CUDA`, which shows "The file was found at .../FindCUDA.cmake" right above
the error. `-DCMAKE_POLICY_DEFAULT_CMP0146=OLD` does NOT recover it at that floor; only an
in-project `cmake_policy(SET CMP0146 OLD)` before the call does, which is an edit to code the
port does not own. Build the identical upstream base commit with the same cmake, nvcc and
gcc: if it fails identically, record the CUDA check as not done, with this explanation.
(arbor)

**A dependency that FetchContent-clones with full history can eat the whole time budget of
the check; fetch the pinned commit shallowly instead.** Read the pinned sha out of the
dependency's own version file (rapids-cmake keeps it in `rapids-cmake/cpm/versions.json`),
`git fetch --depth 1` that one commit, and point the build at the checkout. For a
rapids-cmake project set BOTH `-DCPM_<Name>_SOURCE=` and `-DFETCHCONTENT_SOURCE_DIR_<NAME>=`,
since CPM reads the first and plain FetchContent the second, and they spell the name
DIFFERENTLY: CPM looks up `CPM_${CPM_ARGS_NAME}_SOURCE`, the package name exactly as
`CPMAddPackage` received it and case-sensitive, while CMake upper-cases the name for
`FETCHCONTENT_SOURCE_DIR_<NAME>`. On a lower-case package (`spdlog`, `fmt`)
`-DCPM_SPDLOG_SOURCE=` is silently ignored and the full clone happens anyway. Read the real
spelling from the key in `versions.json`, or from the first argument of the project's
`CPMAddPackage` call. Same commit as the pin, so no version drift, and the check becomes a
repeatable script. (HEonGPU: RMM pulls NVIDIA/cccl.)

**A torch extension needs a CUDA-built torch for this check, not just the toolkit.** The
project's `setup.py` picks CUDA or HIP by reading the installed torch
(`hasattr(torch.version, 'hip') and torch.version.hip is not None`, or similar). The check is
still possible with no NVIDIA GPU, and still compile and link only:
`pip install --index-url https://download.pytorch.org/whl/cu121 torch==<pinned>` into a
throwaway venv (an older CUDA minor such as cu121 so a prebuilt wheel exists for your Python;
the wheel bundles its own `nvidia-*-cu12` shared libraries), point `CUDA_HOME` at the nvcc
env, and pin `TORCH_CUDA_ARCH_LIST=8.0` (unset, torch autodetects through
`torch.cuda.get_device_capability()` and fails or silently degrades). Build a scratch copy of
the sources (`git archive HEAD | tar -x`) so the CUDA build's `build/` and `.o` files never
land beside the ROCm ones. Confirm the pin took with `cuobjdump --list-elf` on the built `.so`
(expect `sm_80` cubins, one per translation unit), the way `llvm-objdump --offloading`
confirms the HIP target. Where no CUDA-built wheel can be installed, the check falls back to
the machine's own torch, and the next entry describes how that fails.
(faster-gaussian-splatting, FaithC, accelerated-scan)

Details that cost time on CuMesh. Name an index explicitly, the one matching the toolkit's
major.minor where it has a wheel for your Python (toolkit 12.8:
`--index-url https://download.pytorch.org/whl/cu128`); PyPI's default `pip install torch`
pulls whatever CUDA minor is current and can mismatch the toolkit. The CUDA-built wheel is the reliable lever because its `torch.version.hip` is
`None`; merely unsetting `ROCM_HOME`/`ROCM_PATH` in a ROCm environment is not, since torch
can rediscover ROCm from its SDK package, from `hipcc` on `PATH` or from `/opt/rocm`. If the
project has an explicit back-end override (`BUILD_TARGET=cuda`), set it as well. The conda
toolkit keeps `cuda.h`/`cuda_runtime.h` under `targets/x86_64-linux/include/`, so a TU the
HOST compiler builds (not nvcc) that includes them needs that directory on
`CPLUS_INCLUDE_PATH`/`C_INCLUDE_PATH`. Run `setup.py build_ext` from the extension's own
directory when its `sources` list is relative to the working directory. (CuMesh)

**`nvcc fatal: Unsupported gpu architecture 'compute_35'` at the first
`enable_language(CUDA)` test compile is a torch-wheel and toolkit mismatch, not a port
fault.** It fires inside `Caffe2Config.cmake`/`public/cuda.cmake`, before the project's own
CMakeLists runs, for a project that does a plain `find_package(Torch REQUIRED)`: a
`pip install torch --index-url .../cu128` wheel's exported `CMAKE_CUDA_FLAGS` can still
contain `-gencode arch=compute_35,code=sm_35` (Kepler), which nvcc from CUDA 12.0 on no
longer accepts. Confirm by building the merge-base upstream commit with the same toolchain
and wheel: the same message at the same call stack means it is pre-existing. Pick an older
toolkit (12.1) or an older matching torch wheel to exercise the check, or record the CUDA
check as not done with this explanation. Do not patch the project's CMake for it; the
failure is inside PyTorch's shipped CMake config. (k2)

### CUDA gate for torch-extension ports: the host's installed PyTorch itself can be the wall

On a host whose only installed PyTorch is a ROCm dev build (built from source with `USE_ROCM=1`), nvcc-compiling a torch C++/CUDA extension fails for reasons that have nothing to do with the port. The first choice is still a genuine CUDA-flavored torch wheel in a throwaway venv (the entry above), which gives a full `CUDAExtension` compile and link. Without one, the closest available check is a raw `nvcc -c` of the `.cu` against `torch.utils.cpp_extension.include_paths()` -- and that dies in the ambient install's headers. Two fingerprints of the same root cause:

- The shipped `torch/include` tree was generated for the ROCm build and is missing CUDA-only generated headers, e.g. `#include <c10/cuda/impl/cuda_cmake_macros.h>: No such file or directory` (that file is emitted by cmake only when `USE_CUDA=1`).
- `torch/headeronly/util/complex.h` guards `#include <thrust/complex.h>` with `#if defined(__HIPCC__) || defined(__HIPCC__)` -- a duplicated-token typo, evidently meant `__CUDACC__ || __HIPCC__` -- so under nvcc the include is skipped while `c10/util/complex.h`/`complex_math.h` still reference `thrust::complex` unconditionally under `__CUDACC__`, cascading into ~100 "identifier thrust is undefined" errors on ANY project that includes `torch/extension.h`, regardless of that project's own code (observed on dev build `2.14.0a0+gitb6b444c`).

Neither is a defect in the port: they are defects in the ambient PyTorch install, present identically whether you build the pristine upstream or the ROCm branch. Diagnose it as environmental by checking that the errors bottom out in `torch/headeronly`/`c10` (not the port's own files) and that `grep -n '__builtin_trap\|__trap\|__HIP\|hip[A-Z]\|amdgcn\|USE_ROCM'` over the port's own changed sources is empty. Do not chase this by patching the installed torch headers (that is fixing the environment, not the port). Record the CUDA check as not validated, quoting the missing-header or duplicated-guard error, and, if there is time budget left, substitute a source-level check: diff the port's CUDA-facing code (e.g. the non-ROCm branch of a dual-path file) against upstream and confirm it is byte-for-byte the same logic, only guarded differently -- that is as much passthrough evidence as a real nvcc pass would give without one. (FaithC, accelerated-scan)

Before settling for the diff-only fallback, check whether the project has a SECOND build path that reaches the CUDA-facing device code without going through `torch.utils.cpp_extension`/`CUDAExtension` at all -- a `CUDA_PTX_COMPILATION` target invoked from its own `CMakeLists.txt`, an `enable_language(CUDA)` project that only needs `find_package(CUDAToolkit)` for the device kernels themselves. That path only needs `find_package(Torch)` for include directories (no ROCm/CUDA mismatch, no linking), so it configures and builds against the ambient ROCm torch without incident, and it gives a REAL pinned-arch nvcc compile of the port's touched CUDA headers -- strictly better evidence than a diff, and worth the extra ten minutes of reading the project's CMake before writing off the gate as environmental. Confirm first that the headers the port touches are actually `#include`d by whatever `.cu` files that path compiles, or the pass proves nothing about the port's own edits. (diff-surfel-tracing)

## Other compile-only checks

**A Windows machine can syntax-check the Linux HIP host pass of a `.cu` without a Linux
box.** No root is needed: in WSL, `apt-get download libstdc++-14-dev libgcc-14-dev libc6-dev linux-libc-dev libc6 libcrypt-dev` then `dpkg-deb -x` each into a directory and copy `usr/include` and `usr/lib/gcc` out (`tar -ch` to resolve symlinks). Then run the HIP clang with the TU's own `-D/-I/-include` flags plus `--target=x86_64-unknown-linux-gnu --sysroot=<dir> --gcc-install-dir=<dir>/usr/lib/gcc/x86_64-linux-gnu/14 -nogpulib -x hip --cuda-host-only -fsyntax-only`; `-E -dM` should show `__linux__`, `__GLIBCXX__` and no `_MSC_VER`. Use the same ROCm toolchain as the Linux build you are standing in for, both the clang major and the HIP headers: in the LichtFeld-Studio `std::pow(__half, __half)` case, clang 23 with the ROCm 7.14 headers accepted the call that clang 24 with the 10.x headers (the Linux build's toolchain) rejected as ambiguous, so a check on an older toolchain proves nothing. Reproduce the Linux error first (negative control) before trusting a clean result. It is syntax evidence only; the real Linux build stays the gate. (LichtFeld-Studio)

**Proving two builds carry the same device code: build both in the SAME directory.** The
claim "this host-side change leaves the device code unchanged" (and with it "no new GPU run
is needed") is checked by dumping `.hip_fatbin` from each object with
`llvm-objcopy --dump-section=.hip_fatbin=<out> <obj>` and comparing sha256, plus a sorted
diff of `llvm-nm -D --defined-only <binary>` for the exported symbols. The raw comparison is
sound ONLY when both builds happen in one directory. `__hip_cuid_<hash>` is a hash of the
whole compile command line: the source path as spelled, plus `-o`, `-I` and `-D`, which in a
CMake build all carry the build tree's absolute path. Two checkouts of the SAME commit built
side by side in different directories therefore yield different fatbins, identical in size
and differing in a few dozen bytes, and spelling the source relatively does not defeat it,
because the other arguments still differ. Followed across two paths, the recipe reports
every object changed with no source change whatever (TurboFNO: a 10/10 false alarm before
the cause was found). The remedy is one directory: build commit A, capture the objects,
`git checkout` commit B in place, and rebuild there. Excluding the cuid is sound only at
SYMBOL level, and it must exclude BOTH cuid-derived symbols, since the object also carries
`__hip_gpubin_handle_<same hash>`: `llvm-nm <obj> | grep -vE '__hip_(cuid|gpubin_handle)_'`
(or a `sed` normalizing the hash in both names); filtering only `__hip_cuid_` still reports
the handle line as a difference on an artifact-only pair. Never exclude the cuid as a
byte-level normalization of the dumped fatbin: a length-preserving substitution of the name
still left 8 of 40 differing bytes, name-derived table bytes it cannot reach. And a
differing cuid says NOTHING about whether real device code also changed: the difference has
the same shape for an artifact-only rebuild and for an edited kernel, so it shows the
artifact is present, never that it is the whole difference. Only the same-directory rebuild
(or a comparison of the disassembled device ISA) answers that. The pip/TheRock SDK has the
LLVM tools under `<sdk>/lib/llvm/bin` and no `/opt/rocm`. (TurboFNO, gfx90a, a ROCm 7.14
TheRock nightly, 2026-08)

## What a pass does not prove

### A passing assertion COUNT is not a list of what was tested

An exception or an abort part-way through a test case hides every assertion that would
have run after it, so a single unported path can hide most of a file. (How much is skipped
depends on the framework and on whether the failure is a thrown exception or a process
abort; the case below is a Catch2 suite, and its exact skipping rule was not established.) marian-dev's
operator suite reported "284 of 287 assertions; the 3 failures are the sparse
path" across three platforms and two reviews. Fixing the sparse path turned the
same suite into 603 assertions and immediately failed three of the newly reached
ones, in an unrelated operation that had been broken on ROCm from the start.

When a suite has a known-failing section, its pass count is a LOWER BOUND on what
went untested, and "the failures are isolated to X" describes what ran, not the
port. Compare the assertion count against the CUDA build's, or filter the failing
section out and watch whether the total moves. A known failure that makes a test case
abort quietly hides the rest of the file with it. (marian-dev)

### A green result does not prove a GPU test ran: count kernel dispatches

A GPU test that never touched the GPU reports PASS. This is common enough to plan for: a
test-fixture helper compiled to an empty stub without a windowing toolkit, a
`skipIf(not is_available())` that evaluates true because the runtime is not visible in that
shell, an entire suite gated behind a `GPU_ENABLED` that a headless configure turned off.
The suite says "100% tests passed" and the port is unvalidated. Wall time is the obvious
heuristic and it is NOT reliable. Use kernel dispatches:

    AMD_LOG_LEVEL=3 ./the_test --gtest_filter='TheGpuTest.*' 2>&1 \
      | grep -oE 'YourKernelName|OtherKernelName' | sort | uniq -c

The HIP runtime names every kernel it launches. Zero dispatches of your kernels means the
body did not run, whatever the result line says. It is direct evidence, cheap, and works on
any project without instrumenting the code (for kernels replayed from a captured graph,
which log only once, see diagnosis.md on `AMD_LOG_MASK`).

colmap is the source. Its GPU SIFT tests run through a `RunGpuTest` helper that calls
`RunThreadWithOpenGLContext`, which is an empty inline when the GUI is disabled, so a
headless build silently skipped every GPU test body and reported 145 tests passing. Worse
for the timing heuristic: after the fix, the GPU matcher tests take 0-3 ms in the headless
build and about 200 ms in the GUI build, and the difference is entirely Qt constructing an
offscreen context. Timing would have labelled the real run a skipped one and the skipped run
a real one; the dispatch count was right both times. Three follow-ons:

- When a project's test infrastructure has such a stub, FIXING it is porting work, not scope
  creep, and it is not platform-specific: the same trap was hiding the same thing from the
  CUDA build. Add a test that asserts the body ran, so the no-op cannot return.
- The dispatch log also tells you WHICH backend ran. A project with both a compute and a
  GLSL/CPU fallback will pass its whole suite on the fallback; seeing your kernel names is
  what distinguishes "the port works" from "the fallback works" (plvs ran a CPU fallback on
  every AMD GPU for months: strategy-a-cmake.md, "Build hygiene").
- When a second architecture runs the same fixed test inputs, DIFF the dispatch counts
  against the first architecture's recorded numbers rather than just checking they are
  nonzero. If the test data and algorithm are deterministic (colmap's `sift_test` runs fixed
  images through fixed SIFT parameters), the per-kernel counts are architecture-independent
  even though the wavefront width differs: gfx1100 (wave32) and gfx90a (wave64) produced the
  identical kernel-for-kernel, count-for-count table. An exact match is stronger
  corroboration than a second "it is nonzero"; a mismatch with no code difference between
  the two runs is a signal worth chasing. (colmap)

### Read what the project's own cross-backend checker actually compares

A project that ships a "compare backend A against backend B" harness makes an attractive gate, but check which outputs it compares before trusting a pass, and check whether the test INPUT can reach the code you changed at all.

Two independent ways such a gate silently proves less than it looks:

- **It may compare a subset of the network's outputs.** lc0's `--backend=check` compares only the value and policy heads; `GetMVal` (moves left) is on the same interface and is never compared. A corrupted moves-left weight therefore passes the gate at 34/34 while the head itself is wrong. Grep the checker for every output accessor the interface exposes and confirm each one appears in a comparison.
- **The chosen test input may not instantiate the code path.** The default lc0 test net (maia-1100) has no moves-left head at all, so no batch count on it can exercise moves-left uploads. Decode the input's own metadata to confirm the feature is present rather than inferring it from the file name -- for lc0, `format.network_format.moves_left` in the gzipped protobuf, which a ~30-line varint walk reads without the generated bindings.

When the gate cannot see the path, find a second observable rather than reporting the gate's pass as coverage. lc0 exposes the missing one behind `--show-movesleft`, which prints the moves-left output in the UCI info line, so the same two backends can be compared on it directly.

The decisive form of this evidence is a BASE CONTROL: rebuild the pre-fix source and show the gate passing while the second observable is wrong. On lc0 the pre-fix build passed 34/34 on a moves-left net and simultaneously reported moves left 72 against the CPU reference's 82 in fp16. That single pair is what turns a reviewer's argued finding into a measured one, and it costs one file's recompile when the fix is one file.

Watch the precision variant too: a gate pinned to the fp32 backend can miss a defect that only exists in the fp16 one, and fp16 is often what the auto-selecting backend actually gives users. lc0 registers `hip` as fp32 and `hip-fp16` as half, and `hip-auto` picks half on every gfx11 part, so the recorded fp32 gate command was not testing the configuration users run. Expect an fp16-vs-CPU comparison to fail tight tolerances on ordinary rounding (policy relative 3.4e-02 here) and read the error MAGNITUDE rather than the pass/fail bit: rounding sits orders of magnitude below a corrupted-constant signature.

## A project with no tests

### A project with no test suite: author the harness, then ship it WITH the port

Some projects have no tests at all -- no `tests/`, no CI, no assertions -- so the GPU gate has to be authored. Where the script lives is not bookkeeping, because the commit-message Test Plans name it: a Test Plan is a promise that a reader with the hardware can re-run the command, and a fenced `python3 validate_foo.py` that resolves to nothing in a clone of the fork reads as "there is a validation script you have not found" rather than "this lives elsewhere". Every fenced command in every commit body must be runnable from a clone of the fork.

Default to shipping the harness in the port when it is ONE standalone file importing only the standard library, the framework the project already depends on, and the project itself. That is a script, not an imposed test framework, so "we cannot invent a test framework for someone else's repository" is not a reason to keep it out; and it is the direct answer to "how do I know this works on my machine" for a PR that asks a maintainer to take a whole new back end on trust. Put it beside whatever standalone script the project already ships (its `example/` or `demo/` directory), name it for what it actually checks (a script that asserts `torch.version.hip` is the ROCm check, not a general test), and point at it from the same doc section that documents the ROCm build. Keep ONE copy -- move it, do not copy it, because two copies drift and the drifting one is the one the gate runs. Keep it in your own porting records only when it genuinely cannot ship: it needs fixtures, data, or a framework the project does not have. Never leave it in an untracked scratch directory: a harness written there is lost to every other machine. (diff-surfel-tracing, EnvGS)

### A GPU test the port adds must skip on a machine with no device, and an in-`main` check cannot always do it

A test you add is usually gated on the GPU back end being *installed*, which says nothing
about a device being present, so a build machine without one turns "runs nothing" into a
red test. Which mechanism you need is decided by whether the test process reaches `main` on
a device-less machine, not by preference.

**Nothing aborts before `main`.** Ask the runtime for a device count in `main` and return
77, with `set_tests_properties(<test> PROPERTIES SKIP_RETURN_CODE 77)`. Add the device-count
spellings to the compat header rather than reaching for the HIP names directly, so the CUDA
build keeps compiling.

**A dependency constructs a GPU context at load time.** A namespace-scope global that builds
one is constructed while its shared library is loaded, so on a machine with no device the
process throws out of `cudaGetDeviceProperties`/`hipGetDeviceProperties` and aborts before
`main`. The in-`main` 77 check can then never execute, and two CTest mechanisms that look
like they rescue it do not, both measured: `SKIP_REGULAR_EXPRESSION` is not consulted for a
process that dies by a signal, and a `FIXTURES_SETUP` probe either lets the dependent test
run anyway (probe skipped) or reports it `***Not Run` and counts both as failures (probe
failed). Only a normal exit can produce a skip, so the decision has to be made by a process
that never loads that library. Register a launcher as the test command: a probe executable
links the GPU runtime *only*, not the library that pulls the dependency in, and exits 77
when it counts no device, and a `cmake -P` script runs the probe first. On 77, and on 77
alone, the script stops with `message(FATAL_ERROR "no GPU device available, skipping")`,
which the matching `SKIP_REGULAR_EXPRESSION` turns into a skip because the script itself
exits normally. Any other nonzero probe result must fail with the status it returned:
routing every nonzero result to the skip path reports a probe that segfaulted or could not
load the runtime as a machine without a GPU, hiding a test that never ran. With a device,
the script runs the real test and forwards its status, so a failing test stays a failure.
Measured on one build: `Passed` with a device, `***Skipped` under `HIP_VISIBLE_DEVICES=-1`,
and `***Failed` with `device probe exited with 1` for a broken probe. The real fix belongs
in the dependency (a lazy context), so raise it there as well instead of leaving the
launcher as the whole answer. (rmcl ships this; its `rmagine` dependency defines
`CudaContextPtr cuda_def_ctx(new CudaContext(0));` at namespace scope.)

## Validating a rewrite

### Golden-differential validation for a behavior-preserving rewrite

When a round REWRITES working code instead of adding guarded edits (a licence cleanup, a
de-vendoring, a structural refactor), the project's own test suite is too coarse: it samples
a few hundred points where the rewrite changed thousands of lines. Validate differentially
against the code being replaced. (Method built for cubvh's clean-room BVH rewrite.)

- Capture goldens from the OLD build FIRST, before any rewrite lands: deterministic
  procedurally generated inputs (include the nasty cases: open meshes, degenerate zero-area
  faces), fixed query sets, every public operation, AND the serialized state. Each
  architecture captures its own goldens from the old commit. A machine that arrives late can
  still do so because the old commit stays in history, but it must capture BEFORE validating
  the new head, not after.
- Compare with an ADJUDICATED policy, not blind equality:
  - Floats: a documented absolute tolerance per operation (observed FMA-reordering noise
    about 1e-7 on unit-scale scenes; 2e-5 is generous and still catches algorithm errors).
  - Discrete winners (nearest-element ids, argmin results): exact match across builds is
    UNACHIEVABLE on tie-dense inputs. Tie winners are decided by 1-ulp differences, and
    hipcc at `-O3` contracts FMAs differently per inlining context, so even IDENTICAL source
    produces 1-ulp differences between translation units. Gate the NON-TIE mismatch fraction
    (a mismatch whose value gap exceeds the tolerance) and verify that every remaining
    mismatch is a genuine tie.
  - Intentional behavior changes: MASK the affected queries out of old-against-new, and
    gate them with a SELF-consistency check instead (cubvh: zero-area faces became invisible
    to distance queries; the new build on the full mesh must equal the new build on the mesh
    with those faces removed, exact 0.0 there).
  - Serialization: load the OLD build's serialized state into the NEW build and require
    identical query behavior. That is the layout and encoding contract, and it catches what
    struct-level `static_assert`s cannot (index-encoding semantics).
- For a clean-room rewrite specifically: separate the spec author from the implementer (the
  author reads the derived code and writes a behavior spec; the implementer never sees the
  originals, so blank those files in the working tree and a stray read returns nothing).
  Then verify independence mechanically: normalized `difflib` similarity plus the fraction
  of verbatim non-trivial lines of each new file against the ANCESTOR snapshot (era-matched,
  not upstream HEAD) and against the old files, and inspect every residual match. Run the
  scan even on code classified "original, keep verbatim": on cubvh it caught three functions
  that a careful manual provenance analysis had misclassified as original and that were in
  fact derived.
- Performance is a deliverable of a rewrite, not a sanity check: same machine, old commit
  against new commit back to back, at least 5 repetitions, medians, several input scales,
  every hot operation. First rewrites commonly regress 10 to 25 percent on GPU hot loops
  through innocent structure (early-out branches that diverge where the old code was
  single-exit; runtime-bound child loops where the old code unrolled a compile-time fan-out;
  a data-dependent insertion sort where a fixed compare-exchange network ran; an extra cross
  product or sqrt per primitive test). All of those reclaim to parity without behavior
  changes; re-run the goldens after each performance fix. (cubvh)
