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

### CUDA gate for torch-extension ports: the host's installed PyTorch itself can be the wall

On a host whose only installed PyTorch is a ROCm dev build (built from source with `USE_ROCM=1`), nvcc-compiling a torch C++/CUDA extension fails for reasons that have nothing to do with the port. The first choice is still a genuine CUDA-flavored torch wheel in a throwaway venv (the entry above), which gives a full `CUDAExtension` compile and link. Without one, the closest available check is a raw `nvcc -c` of the `.cu` against `torch.utils.cpp_extension.include_paths()` -- and that dies in the ambient install's headers. Two fingerprints of the same root cause:

- The shipped `torch/include` tree was generated for the ROCm build and is missing CUDA-only generated headers, e.g. `#include <c10/cuda/impl/cuda_cmake_macros.h>: No such file or directory` (that file is emitted by cmake only when `USE_CUDA=1`).
- `torch/headeronly/util/complex.h` guards `#include <thrust/complex.h>` with `#if defined(__HIPCC__) || defined(__HIPCC__)` -- a duplicated-token typo, evidently meant `__CUDACC__ || __HIPCC__` -- so under nvcc the include is skipped while `c10/util/complex.h`/`complex_math.h` still reference `thrust::complex` unconditionally under `__CUDACC__`, cascading into ~100 "identifier thrust is undefined" errors on ANY project that includes `torch/extension.h`, regardless of that project's own code (observed on dev build `2.14.0a0+gitb6b444c`).

Neither is a defect in the port: they are defects in the ambient PyTorch install, present identically whether you build the pristine upstream or the ROCm branch. Diagnose it as environmental by checking that the errors bottom out in `torch/headeronly`/`c10` (not the port's own files) and that `grep -n '__builtin_trap\|__trap\|__HIP\|hip[A-Z]\|amdgcn\|USE_ROCM'` over the port's own changed sources is empty. Do not chase this by patching the installed torch headers (that is fixing the environment, not the port). Record the CUDA check as not validated, quoting the missing-header or duplicated-guard error, and, if there is time budget left, substitute a source-level check: diff the port's CUDA-facing code (e.g. the non-ROCm branch of a dual-path file) against upstream and confirm it is byte-for-byte the same logic, only guarded differently -- that is as much passthrough evidence as a real nvcc pass would give without one. (FaithC, accelerated-scan)

Before settling for the diff-only fallback, check whether the project has a SECOND build path that reaches the CUDA-facing device code without going through `torch.utils.cpp_extension`/`CUDAExtension` at all -- a `CUDA_PTX_COMPILATION` target invoked from its own `CMakeLists.txt`, an `enable_language(CUDA)` project that only needs `find_package(CUDAToolkit)` for the device kernels themselves. That path only needs `find_package(Torch)` for include directories (no ROCm/CUDA mismatch, no linking), so it configures and builds against the ambient ROCm torch without incident, and it gives a REAL pinned-arch nvcc compile of the port's touched CUDA headers -- strictly better evidence than a diff, and worth the extra ten minutes of reading the project's CMake before writing off the gate as environmental. Confirm first that the headers the port touches are actually `#include`d by whatever `.cu` files that path compiles, or the pass proves nothing about the port's own edits. (diff-surfel-tracing)

## Other compile-only checks

**A Windows machine can syntax-check the Linux HIP host pass of a `.cu` without a Linux
box.** No root is needed: in WSL, `apt-get download libstdc++-14-dev libgcc-14-dev libc6-dev linux-libc-dev libc6 libcrypt-dev` then `dpkg-deb -x` each into a directory and copy `usr/include` and `usr/lib/gcc` out (`tar -ch` to resolve symlinks). Then run the HIP clang with the TU's own `-D/-I/-include` flags plus `--target=x86_64-unknown-linux-gnu --sysroot=<dir> --gcc-install-dir=<dir>/usr/lib/gcc/x86_64-linux-gnu/14 -nogpulib -x hip --cuda-host-only -fsyntax-only`; `-E -dM` should show `__linux__`, `__GLIBCXX__` and no `_MSC_VER`. Use the same ROCm toolchain as the Linux build you are standing in for, both the clang major and the HIP headers: in the LichtFeld-Studio `std::pow(__half, __half)` case, clang 23 with the ROCm 7.14 headers accepted the call that clang 24 with the 10.x headers (the Linux build's toolchain) rejected as ambiguous, so a check on an older toolchain proves nothing. Reproduce the Linux error first (negative control) before trusting a clean result. It is syntax evidence only; the real Linux build stays the gate. (LichtFeld-Studio)

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
