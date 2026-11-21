# Strategy B: pytorch extension

<!-- toc:begin -->
Contents:

- Never commit the hipified mirror; add it to .gitignore instead
- Headers and functions hipify leaves broken
- A bundled header-only library (GLM): hipify sees only `include_dirs`
- A vendored library with its own sources: the `includes` pattern decides which headers are hipified
- Keep compiler flags in the right command-line namespace
- hipify generation: v1 renames, v2 masquerades
- A torch API used by a ROCm-only branch is a version floor nobody sees
- Clean the hipified mirror by the path rule, not with another port's clean step
- Windows
- Dependency environment for PyTorch projects (the install path is part of the port)
<!-- toc:end -->

Torch hipifies extension sources at build time. Do not add a compat header and do not hand-rename symbols.

- Building a `CUDAExtension` on a ROCm torch automatically runs `torch.utils.hipify` on the extension's `.cu`/`.cuh` sources and links the HIP runtime (`amdhip64`, `c10_hip`, `torch_hip`). See `torch/utils/cpp_extension.py`.
- Keep sources in CUDA spelling. In a hipified TU that is correct under both hipify generations (see below), so it needs no guard. Fix only what hipify cannot (warp size above all: faults-wavefront.md) in source, guarded by `USE_ROCM`.
- Build against a ROCm torch. If the tree was hipified once and is stale after edits, re-run the project's hipify step before rebuilding (a known incremental-build gotcha: edits to `.cu` can recompile the stale hipified mirror unless you re-hipify first).
- For projects shipping their own `.cu` plus a setup.py, the change is often just: build against a ROCm torch and fix the faults indexed in SKILL.md.

### Never commit the hipified mirror; add it to .gitignore instead

Torch hipify writes its output into the working tree on every build, usually as SIBLING FILES next to the CUDA sources. They are build output that happens to land in the source tree, and `git add -A` after a build sweeps all of them in.

`torch/utils/hipify/hipify_python.get_hip_file_path` (reached from `cpp_extension.py` via `hipify(..., is_pytorch_extension=True)`) decides each output path, and it does NOT always append `_hip`. It rewrites `cuda`->`hip`, `CUDA`->`HIP` and `THC`->`THH` and always rewrites a `.cu` extension to `.hip`. It appends `_hip` to the stem ONLY when that rewrite left both the directory and the full filename unchanged; otherwise the mirror is simply the renamed path. So where the output lands depends on whether the source path already contains the word:

| source | hipified output |
| --- | --- |
| `src/kernel.cu` | `src/kernel.hip` |
| `src/kernel.cuh` | `src/kernel_hip.cuh` |
| `src/util.h` | `src/util_hip.h` |
| `src/cuda/kernel.cuh` | `src/hip/kernel.cuh` -- new directory, filename unchanged |
| `src/cuda_kernels/kernel.cuh` | `src/hip_kernels/kernel.cuh` -- substring, not a whole component |
| `src/cuda_utils.cuh` | `src/hip_utils.cuh` -- same directory, no `_hip` suffix |
| `src/THCFoo.h` | `src/THHFoo.h` |

The directory rewrite is a plain substring `replace` over the whole dirpath string, not a match on path components, so `cuda_kernels/` and `my_cuda/` are rewritten exactly like `cuda/` is. A `cuda`-containing directory or filename is common in CUDA projects, so the renaming rows are not exotic. Copying a suffix-only ignore list into such a project produces a list that misses the entire mirror.

A mirror appears only where hipify actually CHANGES the content. `hipify_python.preprocessor` returns `[skipped, no changes]` and writes no file when the translated output is byte-identical to the input and the dirpath did not move; `preprocess_file_and_save_result` is only the wrapper that calls it and prints the status, so look in `preprocessor` for the skip. So the table predicts WHERE a mirror lands, not THAT one exists: a source hipify leaves alone produces nothing at all. That is why faster-gaussian-splatting's 17 `.h` files yielded only 10 `_hip.h`, and why its `bindings.cpp` yielded no mirror despite being an eligible input. Deriving the ignore list from the table therefore over-predicts, which is harmless; what it must not do is under-predict, so verify against a real build rather than against a count.

Committing them is worse than noise. A maintainer opening the PR sees a machine translation of their own kernels sitting beside the originals, dwarfing the real diff -- faster-gaussian-splatting pushed 26 such files, 5,883 of 5,971 added lines, for an 88-line port. It is also a staleness trap that defeats any clean-tree check: hipify rewrites the files in place, so a from-clean build leaves `git status` clean and everything looks fine, but the next source edit dirties TRACKED files, which then either get committed out of sync with the sources or read as a dirty tree.

So in any Strategy B port, before the first commit, put in the project's `.gitignore`, in its existing style:

    *.hip
    *_hip.h
    *_hip.cuh

Those three cover the suffix case only, which is the whole mirror only when no source path contains `cuda`/`CUDA`/`THC`. Derive the rest from the table: a directory path CONTAINING `cuda`/`CUDA`/`THC` anywhere in it puts its mirror in a sibling directory with that substring rewritten, so ignore that directory (`src/hip/` for `src/cuda/`, `src/hip_kernels/` for `src/cuda_kernels/`); a `cuda`/`CUDA`/`THC` in a filename gives a mirror with the rewritten name in the same directory, so ignore that name (`hip_utils.cuh`, or `hip_*` if there are many and the project has no hand-written ones).

Do not trust the derivation on its own -- verify. `git clean -fdx`, full build, and `git status --porcelain` must print nothing: no tracked file modified AND no generated file newly untracked. The second half is the part that catches this; a tree where the artifacts are tracked passes the first half. It also catches a rename you did not predict, which is why it is the check that matters rather than the pattern list. If the project genuinely hand-writes a `.hip` file or already has a `hip/` directory (Strategy B ports normally do not), narrow the pattern rather than dropping it. (faster-gaussian-splatting)

### Headers and functions hipify leaves broken

A clean hipify run does not mean every name resolved. Two gaps surface only as compile errors in the hipified mirror, and both arrived in an upstream sync rather than the original port (faster-gaussian-splatting):

- `<cooperative_groups/reduce.h>` has no mapping, so the mirror keeps the CUDA header name and fails with "file not found". HIP gained `cg::reduce` in `<hip/cooperative_groups/hip_reduce.h>` (not included by `<hip/hip_cooperative_groups.h>`) together with `cg::plus`/`cg::greater`/`cg::less` in the main header, all in one change (ROCm/rocm-systems `1fe7989f90`, 2026-03-23): present from `therock-7.13`, absent from every 7.2.x release (through `rocm-7.2.4`) and from `therock-7.12`. A port that must build on ROCm 7.2.x therefore needs both arms under `USE_ROCM`: `#if __has_include(<hip/cooperative_groups/hip_reduce.h>)` includes it, and the `#else` defines the functor and a `reduce(const thread_block_tile<size, ParentCGTy>&, T, Op)` in `namespace cooperative_groups` as a `size/2 .. 1` `tile.shfl_xor` loop, so the CUDA call site stays untouched. One `__has_include` gates both names because they arrived together; forcing the fallback on a newer HIP to test it collides with the built-in `greater`, so rename the fallback symbols in a scratch copy for that check. Prove the fallback arm against real 7.2.x headers: the `hip-dev`, `rocm-llvm`, `rocm-device-libs`, `hip-runtime-amd`, `hipcc`, `rocm-core`, `hipcub-dev` and `rocprim-dev` debs from `repo.radeon.com/rocm/apt/7.2.4` extract with `dpkg-deb -x` (no install) and compile the hipified TUs with `--rocm-path`; any extra include directory taken from a newer SDK must leave out its `hip/` subtree, or `__has_include` finds the newer `hip_reduce.h` and the test proves nothing. `<cooperative_groups/scan.h>` maps to `<hip/cooperative_groups/hip_scan.h>` only in builds newer than `therock-7.14` (absent at that tag), so bound it the same way.
- `curand_Philox4x32_10` (one raw Philox4x32-10 evaluation from `<curand_philox4x32_x.h>`) is renamed to `hiprand_Philox4x32_10`, which exists in neither hipRAND nor rocRAND. rocRAND keeps the rounds `protected` inside `rocrand_device::philox4x32_10_engine`. Define the function under `USE_ROCM` instead: ten rounds with multipliers `0xD2511F53`/`0xCD9E8D57` and key increments `0x9E3779B9`/`0xBB67AE85` between rounds. Keep the CUDA name, because hipify renames the definition and the call in the same way. Check it against the Random123 known-answer vectors (counter 0, key 0 gives `6627e8d5 e169c58d bc57ac4c 9b00dbd8`) so the noise stays bit-compatible with CUDA for a given seed and offset. hipRAND is gaining the function itself (ROCm/rocm-libraries PR #14808), so prefer it when present, but select it with a feature test, not a `HIPRAND_VERSION` threshold: a version guessed before the release ships breaks the build if the number arrives without the function. In `setup.py`, under the ROCm branch only, run `hipcc -x hip -fsyntax-only` with the extension's `nvcc` flags and `-I` for `include_dirs + include_paths('cuda')` on a two-line source passed on stdin (include `<hiprand/hiprand_kernel.h>`, call the function in a `__device__` function), discard its output, and append a `-D` macro to the HIP flags only on success; the header keeps the local copy in the `#else`. Return False before building the hipcc path when `cpp_extension.ROCM_HOME` is None (a ROCm torch wheel on a machine without the SDK): `os.path.join(None, 'bin')` raises a TypeError that hides torch's own "ROCM_HOME environment variable is not set" error from `CUDAExtension`. Syntax-only from stdin writes no files and costs about 2 s. (faster-gaussian-splatting)

### A bundled header-only library (GLM): hipify sees only `include_dirs`

Torch's AOT hipify does NOT walk every header the compiler can reach. `CUDAExtension` calls it with `header_include_dirs=kwargs.get('include_dirs', [])` and `hipify_extra_files_only=True` (`torch/utils/cpp_extension.py`; same call in 2.9 through 2.14), so it preprocesses the listed sources plus the headers under the extension's `include_dirs` -- and nothing else. A vendored tree supplied only as `-I<dir>` inside `extra_compile_args` is invisible to it and stays pristine on disk.

A tree that IS listed in `include_dirs` gets mangled: hipify copies only `.hpp`/`.h` into the mirror and content-rewrites them, so a header-only library with `.inl` bodies (GLM: 139 of them) loses files, and its own `__CUDACC__`/`__HIP__` compiler detection is rewritten, so it stops marking its functions `__device__`. The build then fails inside the third-party headers, which sends you looking in the wrong place.

The fix is to leave the library alone rather than to fix up what hipify did. A modern GLM (0.9.9.9) already detects `__HIP__` (`glm/simd/platform.h` -> `GLM_COMPILER_HIP`) and compiles verbatim under `-x hip`, so it needs no shim at all. In order of preference:

1. Take the vendored directory out of `include_dirs` and pass it as `-I` in `extra_compile_args`, which hipify never reads.
2. Only when the tree has to stay in `include_dirs`, and as a last resort because it pins a private torch API, monkeypatch hipify in `setup.py` before `setup()` to skip the directory:

        def _patch_hipify_ignore_glm():
            import torch
            if not torch.version.hip:
                return
            from torch.utils.hipify import hipify_python
            orig = hipify_python.hipify

            def hipify_no_glm(*args, **kwargs):
                kwargs["ignores"] = list(kwargs.get("ignores", ())) + [os.path.join(GLM_DIR, "*"), GLM_DIR + "*"]
                kwargs["header_include_dirs"] = [
                    d for d in kwargs.get("header_include_dirs", [])
                    if os.path.abspath(d) != os.path.abspath(GLM_DIR)
                ]
                return orig(*args, **kwargs)

            hipify_python.hipify = hipify_no_glm

   Gate on `torch.version.hip` so the CUDA path returns immediately. `cpp_extension` looks `hipify_python.hipify` up as a module attribute at call time, so patching the attribute works even though `CUDAExtension` was imported first. (diff-surfel-rasterizations)

Do not add the override to "protect" a bundled library until you have SEEN it get mangled. It is the largest and least palatable hunk in an otherwise mechanical diff, and it is a no-op whenever the tree is `-I`-only. Cheap proof, one build each: build with and without the override, deleting `build/`, the hipified mirror (`hip_*/`, `*.hip`) and `*.egg-info` between runs, then `diff -rq` the mirrors and hash the vendored tree. Identical output plus an unmodified tree settles it. (op43dgs -- a monkeypatch carried over from another port whose include wiring differed; verified no-op on torch 2.9.1, 2.13 and 2.14, then deleted)

### A vendored library with its own sources: the `includes` pattern decides which headers are hipified

Known first half: `matched_files_iter` in `torch/utils/hipify/hipify_python.py` PRUNES `third_party/` from its file walk (it re-adds only `third_party/nvfuser`), so a vendored library's `.cu` sources are reported "[ignored, not to be hipified]" and the raw CUDA is handed to the compiler. The usual workaround is to run `hipify_python.hipify()` yourself, rooted at the vendored library, so the prune does not apply.

The second half is the one that bites: **which HEADERS get rewritten is decided by the `includes` pattern, not by the walk.** `hipify()` extends `all_files` with every file under `header_include_dirs` that matches `includes`, and the include-rewrite hook then recursively hipifies any `#include`d header whose basename appears in `all_files`. Root your own pass at the library's `src/` with `includes=[<lib>/src/*]` and its `include/` tree stays raw, even though you passed the include directory in `header_include_dirs`.

That asymmetry is why this hides on one platform and breaks on another. torch's own pass uses `project_directory=<extension root>` with `includes=[<root>/*]`, and that pattern DOES match `third_party/<lib>/include/...` although the walk pruned it, so torch quietly hipifies the vendored headers for you. A hand-rolled pass rooted narrower does not, and the platform that needs the hand-rolled pass is the only one that breaks.

The symptom is not a missing symbol, it is two header trees in one TU. A vendored header that includes `<ATen/cuda/CUDAContext.h>` stays raw while the source's `#ifdef USE_ROCM` include pulls `<ATen/hip/...>`, and those trees are alternatives that were never meant to coexist: torch's hipified `c10/hip/HIPStream.h` opens `namespace c10::cuda {` and declares the same `class CUDAStream` that `c10/cuda/CUDAStream.h` declares. You get a redefinition cascade across `c10/hip/HIPStream.h`, `HIPException.h`, `HIPFunctions.h` and `HIPDeviceAssertionHost.h`, plus `unknown type name 'cudaStreamCaptureStatus'; did you mean 'hipStreamCaptureStatus'` from `is_capturing()` in `c10/cuda/CUDAStream.h`. That method arrived with a torch update (the port had last built against 2.9.1 and failed against 2.12.0+rocm7.14.0a20260519), so it reads like a hand-written `cuda.h`/`cuda_runtime.h` shim that fell behind, and on Windows, where TheRock's SDK ships no CUDA compat headers, such a shim is usually present to be blamed.

**Fix the scope, not the symptom.** Add the library's include directory to `includes` (`includes=[<lib>/src/*, <lib>/include/*]`, rooted at `<lib>`) so hipify emits `api_hip.h` carrying `<ATen/hip/HIPContext.h>` and rewrites the including source to match. Adding the missing capture symbols to the shim looks like the fix and is not: it silences the first error and leaves the redefinition cascade untouched. Keep the patterns tight; a bare `<lib>/*` would drag a nested Eigen or other vendored headers into hipify's scope.

Two diagnostics worth carrying:

- If you see "redefinition" of `c10/hip/*` classes, look UP the include chain for a raw `ATen/cuda` include, not for a missing symbol. `ATen/cuda/*` and `ATen/hip/*` are mutually exclusive in a single TU.
- Prefer making hipify's WALK find your sources over listing them in `extra_files`. The walk unix-normalizes each path; the `extra_files` append does not, and `preprocess_file_and_save_result` tests membership against the normalized form, so on Windows an `extra_files` entry outside the walk root can be silently skipped.

(CuMesh, whose bundled `cubvh` failed exactly this way on Windows while Linux had built it for months.)

### Keep compiler flags in the right command-line namespace

Torch's `CUDAExtension` uses the `"nvcc"` entry in `extra_compile_args` for both nvcc on CUDA builds and hipcc on ROCm builds. That does not make their flag syntaxes interchangeable: hipcc accepts GCC/clang host flags such as `-fvisibility=hidden` directly, while nvcc needs the same flag wrapped as `-Xcompiler=-fvisibility=hidden` and rejects it bare (`nvcc fatal : Unknown option '-fvisibility=hidden'`). The `"cxx"` entry is a separate host-compiler command line and keeps the bare spelling. A flag added for the HIP build and appended to the shared list unguarded therefore looks clean on every AMD machine and breaks the CUDA build.

Map a shared host flag according to all three cases: pass it bare to hipcc, wrap it for nvcc with a GCC/clang host, and drop it for nvcc with MSVC when there is no MSVC equivalent. Derive the wrapped list from the bare list so the two cannot drift. Do not use only an `IS_HIP` guard: that misses the Windows CUDA case and can hand cl.exe an unsupported `-f...` option.

Verify the wrap POSITIVELY, not just that the build stopped failing: a mis-wrapped flag can be accepted and dropped, so the build goes green while the intent (here, symbol hiding) is gone. Confirm the effect in the artifact: `nm -D --defined-only <ext>.so | grep <ns>` should show only the module entry point while `nm --defined-only` shows the namespace's symbols as local. In CuMesh's `_cubvh` that was 1 exported symbol (`PyInit__cubvh`) against 309 local `cubvh::` symbols. And when a build-flag fix is meant to touch only the CUDA path, prove the ROCm path is untouched by DIFFING the generated flag lists instead of arguing from the source diff: monkeypatch `setuptools.setup` to print each extension's `extra_compile_args` and exit, run it on either side of the change with the ROCm back end selected, and diff. An identical dump is a much stronger claim than "the guard looks right", and it shows the architectures that already passed that nothing changed for them.

A cheap diagnostic for a failing CUDA compile check: if the flag or construct does not appear anywhere in the pre-port upstream source (`git show <upstream_sha>:<file> | grep <token>` is empty), it is new port code, not pre-existing breakage, and there is no need to build upstream to compare. (CuMesh: `setup.py` appended `-fvisibility=hidden -fvisibility-inlines-hidden` to the `"nvcc"` key for both back ends.)

### hipify generation: v1 renames, v2 masquerades

Which generation runs decides what the `c10`/`at` CUDA classes are called, so it is worth
knowing before writing a stream or allocator call.

- **v1** renamed them: only `c10::hip::getCurrentHIPStream` existed after hipification.
- **v2** (pytorch#174087, `torch/utils/hipify/version.py` bumped 1.0.0 -> 2.0.0) STOPS
  renaming them. The CUDA spellings stay public and are the masquerading API, while the
  hip-spelled symbols become `#ifdef USE_ROCM`.

**In an ordinary Strategy B extension this costs you nothing: write the CUDA spelling.**
v1 renames it, v2 keeps it, and no guard is needed either way. That is the whole reason
Strategy B keeps sources in CUDA spelling.

**The trap is a TU that hipify never sees.** A host `.cpp` deliberately routed around
hipify -- through hipcc for a Windows link fix, say -- or a CMake/`USE_HIP` port where torch
source-hipify never runs at all, gets no translation. Hard-coding either spelling there
breaks the other generation. Detect it at build time and branch:

    Version(torch.utils.hipify.__version__) >= "2.0.0"   ->  pass -DTORCH_HIPIFY_V2

    #if defined(TORCH_HIPIFY_V2)
    ... c10::cuda::getCurrentCUDAStream ...
    #else
    ... c10::hip::getCurrentHIPStream ...
    #endif

Use a neutral define, never one named after your team or tooling, since it ends up in the upstream diff.

**Do not reach for `_WIN32` or a ROCm version as a proxy for the generation.** They
correlate only by accident on any given set of machines and can anti-correlate -- a newer ROCm shipping an
older torch -- so a guard keyed on the wrong axis silently breaks the other platform's
build. That fault, with the two regressions that produced it, is in
build-headers-macros.md ("An arch-specific fix keyed on the OS or the ROCm version"). (aihwkit)

### A torch API used by a ROCm-only branch is a version floor nobody sees

A HIP branch compiles only against the ROCm PyTorch on the porting machine, which is usually far newer than the project's pinned torch, and one failing call takes the whole extension down with it, kernels that already worked included.

`ROCM_VERSION` is never defined in a `cpp_extension` build. Only PyTorch's own CMake sets it (`add_definitions(-DROCM_VERSION=...)` in `cmake/Dependencies.cmake`); `cpp_extension` passes only `-D__HIP_PLATFORM_AMD__=1 -DUSE_ROCM=1` (plus `-DHIPBLAS_V2`), no installed torch header defines it, and `rocm_version.h` has only `ROCM_VERSION_MAJOR/MINOR/PATCH`. Checked at v2.2.0, v2.3.0, v2.4.0, v2.8.0, v2.9.0 and main. So every `ROCM_VERSION` guard in an installed header reads false in an extension, current releases included: v2.8.0 `ATen/cuda/CUDADataType.h` maps the fp8 types only under `!defined(USE_ROCM) || ROCM_VERSION >= 60300`, so on ROCm an extension never sees those cases.

`getCurrentCUDABlasLtHandle()` in `ATen/cuda/CUDAContextLight.h` is the case that broke llm-awq: up to 2.3 the declaration and the `<cublasLt.h>` include sit under `... (defined(USE_ROCM) && ROCM_VERSION >= 50700)`, so it is undeclared in an extension; 2.4 (ae9a4fa, "enforce ROCM_VERSION >= 6.0") dropped that guard, which is the only reason the handle is reachable from 2.4 on. `getCUDABlasLtWorkspace()` and `getCUDABlasLtWorkspaceSize()` exist only from 2.9. Before calling a torch API from a HIP branch, `git show <oldest supported tag>:<header>` and read the preprocessor guard around the declaration, not just whether the name is present. A guard inside an inline function body (`ROCM_VERSION < 50201` in `deviceCount()` of `HIPGuardImplMasqueradingAsCUDA.h`) changes behaviour, not what is declared.

When calling hipBLASLt directly, keep the handle and workspace in the extension: one `hipblasLtHandle_t` per device from `hipblasLtCreate` on first use (mutex-guarded, never destroyed, as torch does), the current stream, and a byte-tensor workspace (`at::empty({n}, input.options().dtype(at::kByte))`, sized from the chosen heuristic's `workspaceSize`, none when zero). The hipBLASLt API is itself a floor: 0.3 (ROCm 5.7) takes `hipblasDatatype_t` and `hipblasLtComputeType_t` in `hipblasLtMatrixLayoutCreate` / `hipblasLtMatmulDescCreate`; 0.6 (ROCm 6.0) takes `hipDataType` and `hipblasComputeType_t` (`HIP_R_8I`, `HIPBLAS_COMPUTE_32I`). Gate the hipBLASLt code on `HIPBLASLT_VERSION_MAJOR/MINOR` from `<hipblaslt/hipblaslt-version.h>` (installed by both) and keep a portable kernel for older releases, instead of letting it raise the ROCm floor of the whole extension. (llm-awq)

### Clean the hipified mirror by the path rule, not with another port's clean step

A stale mirror gets recompiled in place of the source you edited, so the clean step has to
remove all of it, and where it lands follows from the source path (the table under "Never
commit the hipified mirror"). A `.cu` in a directory whose path contains `cuda` goes to the
renamed directory, headers copied along with it (`cuda_rasterizer/forward.cu` ->
`hip_rasterizer/forward.hip`), not beside the source. A clean step inherited from a port
whose paths had no `cuda` in them -- `rm -rf build *.egg-info cuda_rasterizer/*.hip` --
therefore leaves a whole stale hipified tree behind and you rebuild code you did not edit.
Remove the renamed directory too, and add it to `.gitignore` so a later run does not commit a
tree of generated sources by accident. The path rule does not depend on the hipify
generation: `get_hip_file_path` is the same before and after the v2 change (pytorch#174087).
(diff-surfel-rasterizations)

### Windows

On Windows a ROCm torch builds a `CUDAExtension` with `clang-cl` as the host compiler. Torch
drives the compile itself and never goes through CMake's `HIP` language, so the
one-clang-driver rule for CMake HIP builds does not apply here. The rest of the Windows
knowledge is in windows.md: the torch wheel needs its per-architecture device package
("A torch wheel needs the matching `[device-<gfx>]` extra"), Triton is a separate package
there ("On Windows, Triton comes from the `triton-windows` package"), a CMake extension's
`find_package(Torch)` needs the architecture and the ROCm root handed to it
("`find_package(Torch)` in a CMake HIP project needs the architecture and the ROCm root"), a
C++ program that links
LibTorch must force the `torch_hip.dll` import, and an extension that links a ROCm library
by name may find no `.lib` for it ("A ROCm library linked by name may have no `.lib`").

### Dependency environment for PyTorch projects (the install path is part of the port)

For a PyTorch-ecosystem project the in-repo `.cu` is only half the bring-up: the project also `pip install`s a dependency graph that defaults to CUDA wheels, and getting a ROCm environment that actually EXERCISES your ported code is its own recurring task. The failure modes repeat across the whole 3D-vision / point-cloud / VLA / world-model space, so they generalize. (AMD's rocm3d -- Andy Luo and David Li, 2026, andyluo7.github.io -- catalogs these as reusable recipes for that domain; the dependency-family patterns below are the transferable core. Note the model: these are environment substitutions to get a project RUNNING on ROCm, not upstreamed source ports -- the deliverable is still the in-repo HIP port + multi-arch PR, so confirm your ported code is the code under test, not a swapped-in prebuilt wheel.)

- A pip install silently overwrites the ROCm torch. An unfiltered `pip install -r requirements.txt` reinstalls the CUDA `torch`/`torchvision`/`xformers`/`flash-attn`/`triton` wheels on top of your ROCm ones, after which the project "has no GPU". Strip the CUDA-default pins first -- `grep -vEi '^(torch|torchvision|torchaudio|xformers|triton|flash.attn|cupy|gsplat)' requirements.txt | pip install -r /dev/stdin` -- then install the ROCm builds explicitly.
- The same trap fires on ANY pip install that has torch in its dependency graph, not just a requirements file, and a source-built ROCm torch is the worst case. `pip install timm torch_geometric` on a machine whose torch is a locally built dev wheel (`2.14.0a0+git<sha>`) replaced torch with the PyPI CUDA build, pulled the whole `nvidia-*` stack, and swapped `triton` 3.8.0 for the CUDA `triton` 3.7.1. The dev version is on no index, so pip cannot put it back and nothing warns you. Default to `pip install --no-deps <pkg>` for every dependency on such a machine, add the handful of pure-Python transitive deps by hand, and before you start, find the wheel that produced the current torch so a restore is `pip install --no-deps --force-reinstall <wheel>`. Verify with `python -c "import torch; print(torch.__version__, torch.version.hip)"` after any install that touched a torch-dependent package. (Pointcept)
- Wheel name != import name. AMD ships some ROCm wheels on its own index (`pip install <pkg> --extra-index-url=https://pypi.amd.com/<rocm-ver>/simple/`); the distribution name can differ from the import (`amd_gsplat` imports as `gsplat`). Pin the ROCm index so pip does not silently prefer a CUDA wheel from default PyPI.
- Build source extensions AGAINST the installed ROCm torch: `PYTORCH_ROCM_ARCH=<arch> pip install <submodule> --no-build-isolation`. `--no-build-isolation` is load-bearing, but get the reason right when you document it for users: the isolated build env contains only the build backend plus whatever `[build-system] requires` names, so a project that pins torch there gets a CUDA torch built against, while the far more common project that pins nothing gets no torch at all and dies at metadata generation on `setup.py`'s top-level `import torch` (`ModuleNotFoundError` inside `/tmp/pip-build-env-*/overlay`). Check the project's `pyproject.toml`/`install_requires` before writing "keeps pip from pulling a CUDA torch" into a README -- and note that `--no-deps` protects nothing when no dependency is declared. (op43dgs: neither exists in any of its four submodules, so the isolation failure is the import, not a CUDA wheel.)
- A common dep usually already has a ROCm port -- consume it, do not re-port it. Apply the existing-AMD-support test (assess-existing-support.md) to each DEPENDENCY, not just the top repo: `spconv` -> `spconv-triton` (Apache-2.0 on PyPI, a drop-in for the spconv 2.x API written in Triton; the GitHub repository named `spconv-rocm`, jiaqiwang969/spconv-rocm, is empty, and the one live derivative found, HYGON-AI/spconv-das, targets Hygon DCU (hcu), not AMD GPUs; checked 2026-11), `torch_scatter`/`torch_cluster` -> the pyg-rocm-build wheels, or build them from sdist (on ROCm 7.14: `PYTORCH_ROCM_ARCH=<arch> FORCE_ONLY_CUDA=1 pip install . --no-build-isolation --no-deps`, with torch_scatter 2.1.2 needing its `at::Half` shuffle wrappers widened to a 64-bit mask first; Pointcept), `gsplat` -> AMD's `amd_gsplat`. (So a dependency being un-portable at the source level does not by itself block a downstream consumer -- a dep-level ROCm substitute may exist.) Consume these as environment dependencies; they are not deliverables.
- Import success != correct backend, and that gap is a VALIDATION trap. A project with several attention paths (PyTorch SDPA, FlashAttention-2 Triton, AITER Triton/CK) will often `import` fine and "run" by silently falling back to SDPA -- exercising NONE of the kernel you care about. Enable the AMD path explicitly (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` for flash-attn; install `aiter` for the AITER paths) and then CONFIRM the runtime backend (logs/profiler, steady-state iteration time, a produced artifact), not just that the import worked. This binds straight to the validation policy (`validation.md`): when swapped ROCm wheels carry the kernels, a run can "pass" on a CPU/SDPA fallback while testing nothing of your port -- prove the code under test is YOUR ported code and that it ran on the GPU.
- A Triton-based ROCm dependency can spend tens of minutes cold-compiling its first step, and that is not a hang. The cost is large on every AMD architecture and larger on smaller GPUs, so set a persistent `TRITON_CACHE_DIR`. A pip-installed, pure-Python-plus-Triton ROCm drop-in library (e.g. a spconv reimplementation) spends a model's first forward+backward pass inside `triton/backends/amd/compiler.py:make_amdgcn`, autotuning: one sparse U-Net's cold first step took 1123 s on MI300X (CDNA3) against 10 s warm, and on the same set of models gfx1100 (RDNA3) was 11x to 49x slower cold than MI300X with the same Triton (one first step took 97 minutes). Before concluding a hang, distinguish real (if slow) compilation from a stuck process: `py-spy dump --pid <pid>` repeatedly and confirm every sample lands inside `make_amdgcn` or `torch.cuda.synchronize` (benchmarking an autotune candidate), and watch the Triton cache directory grow. The cache also says which kernel the time goes to, without a profiler: each compiled candidate writes `<kernel>.hsaco` into its own hash directory, so sort all `.hsaco` files by modification time and charge each gap to the kernel written at its end, which gives compile seconds per kernel name; the `*.autotune.json` files (one per finished autotune key, with the timing of every candidate) show how many keys are done. Use a fresh cache and a single process for that, or the gap between two runs is charged to whichever kernel compiled next. In the 97-minute step a weight-gradient kernel with 21 candidates (deep pipelining, 128-wide tiles) took 25 s per candidate against 3 s for the forward gather-GEMM and accounted for 72% of the time, which points the fix at the library's candidate list, not at the port. If a bounded test window elapses without a completed step, record the wall-clock magnitude, confirm the backend selection independently, and stop. The persistent cache makes the cost one-time per kernel shape and arch (a cached first step took 1-2 s), and a run killed part way keeps what it compiled. On Windows Triton itself is a separate install (windows.md, "On Windows, Triton comes from the `triton-windows` package"). (Pointcept)
- Repo-local native op with no ROCm wheel = an upstreamable port, not a swap. When the project ships its own CUDA op that no ROCm package provides (e.g. a deformable-attention `.cu`), the clean pattern is: build the forward path as a HIP extension, register it as a PyTorch custom op with a pure-PyTorch fallback for any unported path, and validate by numerical diff against the reference (tol ~1e-6) plus a latency check. This IS the source contribution and the upstream PR; the dependency recipes above are only the environment around it.
- The host's default `torch` can be an editable dev checkout whose include/cmake tree is incomplete, and a `CUDAExtension`/`BuildExtension` build hits this even without CMake. `torch.utils.cpp_extension.include_paths()`/`library_paths()` derive from module-level `_TORCH_PATH`/`TORCH_LIB_PATH`, computed once from `torch.__file__` -- if that resolves to a source checkout (e.g. a scikit-build-core editable install redirecting pure-Python imports back to a git tree), its own `torch/include` can be nearly empty even though `import torch` and `torch.cuda.is_available()` work fine (the compiled `torch._C` extension itself can still load from a separate, complete sibling install). Check for that sibling first: the same environment's `site-packages/torch/` can hold a full wheel-style payload (`include/`, `lib/*.so`, `share/cmake/Torch/`) installed before the editable checkout overwrote only the pure-Python top level. If it exists, monkeypatch `cpp_extension._TORCH_PATH`/`TORCH_LIB_PATH` to that sibling directory before `setup.py`'s module body constructs its `CUDAExtension(...)` (`CUDAExtension` reads the globals at call time, so a `runpy.run_path("setup.py", ...)` wrapper that patches the module first, then executes `setup.py` in the same process, works with no fork edit needed) -- `pip install -e . --no-build-isolation` itself cannot be wrapped this way since it forks a fresh subprocess; run the equivalent `python wrapper.py develop --no-deps` instead. A raw-CMake project hits the same class through `find_package(Torch)` against `torch.utils.cmake_prefix_path`; the fix there is the same sibling, applied as `-DCMAKE_PREFIX_PATH=<sibling>/share/cmake` (not `TORCH_INSTALL_PREFIX`, which only fixes `TorchConfig.cmake` and not the independently-derived prefix in `Caffe2Targets.cmake`). (diff-surfel-tracing, linux-gfx1100 validation)
- Files a ROCm port stages into the package at build time (runtime-compiled kernel sources, a vendored runtime library) must be staged BEFORE `build_py`, not inside `build_ext`. A wheel build (what every current `pip install .` does) runs `build_py`, which resolves `package_data`, before `build_ext`, so staging from `build_extensions()` gives a first install with none of the files and every later install the PREVIOUS build's copies. `build_ext --inplace` never runs `build_py`, so a validation that only builds in place cannot see it: validate the documented install route from a fresh clone, then delete the clone and run the GPU test from elsewhere. Fix: a `build_py` subclass that stages, drops the lazily cached `data_files` (`self.__dict__.pop("data_files", None)`), then calls `super().run()`, registered only on the HIP branch so the CUDA `cmdclass` is unchanged. A shared library the extension links ships the same way, named explicitly in `package_data` (not `*.so`, which also matches a stale in-place `_C*.so`), with `-Wl,-rpath,$ORIGIN` in place of an rpath into the build tree; setuptools links without a shell, so `$ORIGIN` arrives literally (`readelf -d` shows `RUNPATH [$ORIGIN]`). (diff-surfel-tracing, linux-gfx942)
