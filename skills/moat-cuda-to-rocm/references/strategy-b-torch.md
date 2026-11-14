## Strategy B: pytorch extension

Torch hipifies extension sources at build time. Do not add a compat header and do not hand-rename symbols.

- Building a `CUDAExtension` on a ROCm torch automatically runs `torch.utils.hipify` on the extension's `.cu`/`.cuh` sources and links the HIP runtime (`amdhip64`, `c10_hip`, `torch_hip`). See `torch/utils/cpp_extension.py`.
- Keep sources in CUDA spelling. In a hipified TU that is correct under both hipify generations (see below), so it needs no guard. Fix only what hipify cannot (warp size, see the fault classes) in source, guarded by `USE_ROCM`.
- Build against a ROCm torch. If the tree was hipified once and is stale after edits, re-run the project's hipify step before rebuilding (a known incremental-build gotcha: edits to `.cu` can recompile the stale hipified mirror unless you re-hipify first).
- For projects shipping their own `.cu` plus a setup.py, the change is often just: build against a ROCm torch and fix the fault classes below.

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

Committing them is worse than noise. A maintainer opening the PR sees a machine translation of their own kernels sitting beside the originals, dwarfing the real diff -- faster-gaussian-splatting pushed 26 such files, 5,883 of 5,971 added lines, for an 88-line port. It is also a staleness trap that defeats the integrity gate: hipify rewrites the files in place, so a from-clean build leaves `git status` clean and everything looks fine, but the next source edit dirties TRACKED files, which then either get committed out of sync with the sources or read as a dirty fork.

So in any Strategy B port, before the first commit, put in the project's `.gitignore`, in its existing style:

    *.hip
    *_hip.h
    *_hip.cuh

Those three cover the suffix case only, which is the whole mirror only when no source path contains `cuda`/`CUDA`/`THC`. Derive the rest from the table: a directory path CONTAINING `cuda`/`CUDA`/`THC` anywhere in it puts its mirror in a sibling directory with that substring rewritten, so ignore that directory (`src/hip/` for `src/cuda/`, `src/hip_kernels/` for `src/cuda_kernels/`); a `cuda`/`CUDA`/`THC` in a filename gives a mirror with the rewritten name in the same directory, so ignore that name (`hip_utils.cuh`, or `hip_*` if there are many and the project has no hand-written ones).

Do not trust the derivation on its own -- verify. `git clean -fdx`, full build, and `git status --porcelain` must print nothing: no tracked file modified AND no generated file newly untracked. The second half is the part that catches this; a tree where the artifacts are tracked passes the first half. It also catches a rename you did not predict, which is why it is the check that matters rather than the pattern list. If the project genuinely hand-writes a `.hip` file or already has a `hip/` directory (Strategy B ports normally do not), narrow the pattern rather than dropping it. (faster-gaussian-splatting)

### Headers and functions hipify leaves broken

A clean hipify run does not mean every name resolved. Two gaps surface only as compile errors in the hipified mirror, and both arrived in an upstream sync rather than the original port (faster-gaussian-splatting):

- `<cooperative_groups/reduce.h>` has no mapping, so the mirror keeps the CUDA header name and fails with "file not found". HIP gained `cg::reduce` in `<hip/cooperative_groups/hip_reduce.h>` (not included by `<hip/hip_cooperative_groups.h>`) together with `cg::plus`/`cg::greater`/`cg::less` in the main header, all in one change (ROCm/rocm-systems `1fe7989f90`, 2026-03-23): present from `therock-7.13`, absent from every stable release through `rocm-7.2.4` and from `therock-7.12`. A port that must build on ROCm 7.2.x therefore needs both arms under `USE_ROCM`: `#if __has_include(<hip/cooperative_groups/hip_reduce.h>)` includes it, and the `#else` defines the functor and a `reduce(const thread_block_tile<size, ParentCGTy>&, T, Op)` in `namespace cooperative_groups` as a `size/2 .. 1` `tile.shfl_xor` loop, so the CUDA call site stays untouched. One `__has_include` gates both names because they arrived together; forcing the fallback on a newer HIP to test it collides with the built-in `greater`, so rename the fallback symbols in a scratch copy for that check. Prove the fallback arm against real 7.2.x headers: the `hip-dev`, `rocm-llvm`, `rocm-device-libs`, `hip-runtime-amd`, `hipcc`, `rocm-core`, `hipcub-dev` and `rocprim-dev` debs from `repo.radeon.com/rocm/apt/7.2.4` extract with `dpkg-deb -x` (no install) and compile the hipified TUs with `--rocm-path`; any extra include directory taken from a newer SDK must leave out its `hip/` subtree, or `__has_include` finds the newer `hip_reduce.h` and the test proves nothing. `<cooperative_groups/scan.h>` maps to `<hip/cooperative_groups/hip_scan.h>` only in builds newer than `therock-7.14` (absent at that tag), so bound it the same way.
- `curand_Philox4x32_10` (one raw Philox4x32-10 evaluation from `<curand_philox4x32_x.h>`) is renamed to `hiprand_Philox4x32_10`, which exists in neither hipRAND nor rocRAND. rocRAND keeps the rounds `protected` inside `rocrand_device::philox4x32_10_engine`. Define the function under `USE_ROCM` instead: ten rounds with multipliers `0xD2511F53`/`0xCD9E8D57` and key increments `0x9E3779B9`/`0xBB67AE85` between rounds. Keep the CUDA name, because hipify renames the definition and the call in the same way. Check it against the Random123 known-answer vectors (counter 0, key 0 gives `6627e8d5 e169c58d bc57ac4c 9b00dbd8`) so the noise stays bit-compatible with CUDA for a given seed and offset. hipRAND is gaining the function itself (ROCm/rocm-libraries PR #14808), so prefer it when present, but select it with a feature test, not a `HIPRAND_VERSION` threshold: a version guessed before the release ships breaks the build if the number arrives without the function. In `setup.py`, under the ROCm branch only, run `hipcc -x hip -fsyntax-only` with the extension's `nvcc` flags and `-I` for `include_dirs + include_paths('cuda')` on a two-line source passed on stdin (include `<hiprand/hiprand_kernel.h>`, call the function in a `__device__` function), discard its output, and append a `-D` macro to the HIP flags only on success; the header keeps the local copy in the `#else`. Return False before building the hipcc path when `cpp_extension.ROCM_HOME` is None (a ROCm torch wheel on a machine without the SDK): `os.path.join(None, 'bin')` raises a TypeError that hides torch's own "ROCM_HOME environment variable is not set" error from `CUDAExtension`. Syntax-only from stdin writes no files and costs about 2 s. (faster-gaussian-splatting)

### A vendored header tree passed only as `-I` is never hipified

Torch's AOT hipify does NOT walk every header the compiler can reach. `CUDAExtension` calls it with `header_include_dirs=kwargs.get('include_dirs', [])` and `hipify_extra_files_only=True` (`torch/utils/cpp_extension.py`; same call in 2.9 through 2.14), so it preprocesses the listed sources plus the headers under the extension's `include_dirs` -- and nothing else. A vendored tree supplied only as `-I<dir>` inside `extra_compile_args` is invisible to it and stays pristine on disk.

So do not add a hipify override (monkeypatching `torch.utils.hipify.hipify_python.hipify` to extend `ignores` or strip `header_include_dirs`) to "protect" a bundled header library until you have SEEN it get mangled. Check how the tree reaches the compiler first. Such a patch is the largest and least palatable hunk in an otherwise mechanical diff, it pins a private torch API, and it is a no-op whenever the tree is `-I`-only.

Cheap proof, one build each: build with and without the override, deleting `build/`, the hipified mirror (`hip_*/`, `*.hip`) and `*.egg-info` between runs, then `diff -rq` the mirrors and hash the vendored tree. Identical output plus an unmodified tree settles it.

If the project DOES pass the tree in `include_dirs`, the exposure is real: hipify copies only `.hpp`/`.h` into the mirror and content-rewrites them, so a header-only library with `.inl` bodies (GLM: 139 of them) loses files and can mis-detect its compiler. Fix it by not putting the vendored dir in `include_dirs` rather than by patching torch. GLM 0.9.9.9 in particular already has a `GLM_COMPILER_HIP` path and compiles verbatim under `-x hip`, so it needs no shim at all. (op43dgs -- a monkeypatch carried over from a gsplat precedent whose include wiring differed; verified no-op on torch 2.9.1, 2.13 and 2.14, then deleted)

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
build. That fault class, with the two regressions that produced it, is in
`fault-classes.md` under "Types, dispatch and platform limits". (aihwkit)

### hipify rewrites a bundled header-only library it can see

torch's hipify walks every `.hpp` under the extension's include dirs into its file set and
content-rewrites the ones a source pulls in. List a bundled GLM (or any header-only
library vendored in the repo) in `include_dirs` and hipify mangles it: it copies only `.hpp`
and `.h`, so the library's `.inl` implementation files are dropped, and it rewrites the
library's own `__CUDACC__`/`__HIP__` compiler detection, so the library stops marking its
functions `__device__`. The build then fails inside the third-party headers, which sends you
looking in the wrong place.

The fix is to leave the library alone rather than to fix up what hipify did. A modern GLM
already detects `__HIP__` (`glm/simd/platform.h` -> `GLM_COMPILER_HIP`) and compiles verbatim
under `-x hip`. First choice: take the vendored directory out of `include_dirs` and pass it
as `-I` in `extra_compile_args`, which hipify never reads (see "A vendored header tree passed
only as `-I` is never hipified" above). Only when the tree has to stay in `include_dirs`, and
as a last resort because it pins a private torch API, monkeypatch hipify in `setup.py` before
`setup()` to skip the directory:

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

The source keeps including `<glm/...>` through `-I`, resolved against the pristine tree.
Gate on `torch.version.hip` so the CUDA path returns immediately. `cpp_extension` looks
`hipify_python.hipify` up as a module attribute at call time, so patching the attribute works
even though `CUDAExtension` was imported first. (diff-surfel-rasterizations)

### Where the hipified mirrors land depends on the hipify generation

Stale mirrors are a known trap (above), but the clean step has to match the generation, and
the two do not put the output in the same place. Older hipify wrote `.hip` files beside the
sources (`cuda_rasterizer/forward.hip`); hipify 2.x renames the DIRECTORY as well
(`cuda_rasterizer/forward.cu` -> `hip_rasterizer/forward.hip`, headers copied along with it).
A clean step inherited from an earlier port -- `rm -rf build *.egg-info cuda_rasterizer/*.hip`
-- therefore leaves a whole stale hipified tree behind and you rebuild code you did not edit.
Remove the renamed directory too, and add it to `.gitignore` so a later run does not commit
a tree of generated sources by accident. (diff-surfel-rasterizations)

### Dependency environment for PyTorch projects (the install path is part of the port)

For a PyTorch-ecosystem project the in-repo `.cu` (above) is only half the bring-up: the project also `pip install`s a dependency graph that defaults to CUDA wheels, and getting a ROCm environment that actually EXERCISES your ported code is its own recurring task. The failure modes repeat across the whole 3D-vision / point-cloud / VLA / world-model space, so they generalize. (AMD's rocm3d -- Andy Luo and David Li, 2026, andyluo7.github.io -- catalogs these as reusable recipes for that domain; the dependency-family patterns below are the transferable core. Note the model: these are environment substitutions to get a project RUNNING on ROCm, not upstreamed source ports -- the deliverable is still the in-repo HIP port + multi-arch PR, so confirm your ported code is the code under test, not a swapped-in prebuilt wheel.)

- A pip install silently overwrites the ROCm torch. An unfiltered `pip install -r requirements.txt` reinstalls the CUDA `torch`/`torchvision`/`xformers`/`flash-attn`/`triton` wheels on top of your ROCm ones, after which the project "has no GPU". Strip the CUDA-default pins first -- `grep -vEi '^(torch|torchvision|torchaudio|xformers|triton|flash.attn|cupy|gsplat)' requirements.txt | pip install -r /dev/stdin` -- then install the ROCm builds explicitly.
- Wheel name != import name. AMD ships some ROCm wheels on its own index (`pip install <pkg> --extra-index-url=https://pypi.amd.com/<rocm-ver>/simple/`); the distribution name can differ from the import (`amd_gsplat` imports as `gsplat`). Pin the ROCm index so pip does not silently prefer a CUDA wheel from default PyPI.
- Build source extensions AGAINST the installed ROCm torch: `PYTORCH_ROCM_ARCH=<arch> pip install <submodule> --no-build-isolation`. `--no-build-isolation` is load-bearing, but get the reason right when you document it for users: the isolated build env contains only the build backend plus whatever `[build-system] requires` names, so a project that pins torch there gets a CUDA torch built against, while the far more common project that pins nothing gets no torch at all and dies at metadata generation on `setup.py`'s top-level `import torch` (`ModuleNotFoundError` inside `/tmp/pip-build-env-*/overlay`). Check the project's `pyproject.toml`/`install_requires` before writing "keeps pip from pulling a CUDA torch" into a README -- and note that `--no-deps` protects nothing when no dependency is declared. (op43dgs: neither exists in any of its four submodules, so the isolation failure is the import, not a CUDA wheel.)
- A common dep usually already has a ROCm port -- consume it, do not re-port it. Apply the existing-AMD-support test (above) to each DEPENDENCY, not just the top repo: `spconv` -> `spconv_rocm`, `torch_scatter`/`torch_cluster` -> the pyg-rocm-build wheels, `gsplat` -> AMD's `amd_gsplat`. (So a dependency being un-portable at the source level does not by itself block a downstream consumer -- a dep-level ROCm substitute may exist.) Consume these as environment dependencies; they are not deliverables.
- Import success != correct backend, and that gap is a VALIDATION trap. A project with several attention paths (PyTorch SDPA, FlashAttention-2 Triton, AITER Triton/CK) will often `import` fine and "run" by silently falling back to SDPA -- exercising NONE of the kernel you care about. Enable the AMD path explicitly (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` for flash-attn; install `aiter` for the AITER paths) and then CONFIRM the runtime backend (logs/profiler, steady-state iteration time, a produced artifact), not just that the import worked. This binds straight to the validation policy (`validation.md`): when swapped ROCm wheels carry the kernels, a run can "pass" on a CPU/SDPA fallback while testing nothing of your port -- prove the code under test is YOUR ported code and that it ran on the GPU.
- Repo-local native op with no ROCm wheel = an upstreamable port, not a swap. When the project ships its own CUDA op that no ROCm package provides (e.g. a deformable-attention `.cu`), the clean pattern is: build the forward path as a HIP extension, register it as a PyTorch custom op with a pure-PyTorch fallback for any unported path, and validate by numerical diff against the reference (tol ~1e-6) plus a latency check. This IS the source contribution and the upstream PR; the dependency recipes above are only the environment around it.
- The host's default `torch` can be an editable dev checkout whose include/cmake tree is incomplete, and a `CUDAExtension`/`BuildExtension` build hits this even without CMake. `torch.utils.cpp_extension.include_paths()`/`library_paths()` derive from module-level `_TORCH_PATH`/`TORCH_LIB_PATH`, computed once from `torch.__file__` -- if that resolves to a source checkout (e.g. a scikit-build-core editable install redirecting pure-Python imports back to a git tree), its own `torch/include` can be nearly empty even though `import torch` and `torch.cuda.is_available()` work fine (the compiled `torch._C` extension itself can still load from a separate, complete sibling install). Check for that sibling first: the same environment's `site-packages/torch/` can hold a full wheel-style payload (`include/`, `lib/*.so`, `share/cmake/Torch/`) installed before the editable checkout overwrote only the pure-Python top level. If it exists, monkeypatch `cpp_extension._TORCH_PATH`/`TORCH_LIB_PATH` to that sibling directory before `setup.py`'s module body constructs its `CUDAExtension(...)` (`CUDAExtension` reads the globals at call time, so a `runpy.run_path("setup.py", ...)` wrapper that patches the module first, then executes `setup.py` in the same process, works with no fork edit needed) -- `pip install -e . --no-build-isolation` itself cannot be wrapped this way since it forks a fresh subprocess; run the equivalent `python wrapper.py develop --no-deps` instead. A raw-CMake project hits the same class through `find_package(Torch)` against `torch.utils.cmake_prefix_path`; the fix there is the same sibling, applied as `-DCMAKE_PREFIX_PATH=<sibling>/share/cmake` (not `TORCH_INSTALL_PREFIX`, which only fixes `TorchConfig.cmake` and not the independently-derived prefix in `Caffe2Targets.cmake`). (diff-surfel-tracing, linux-gfx1100 validation)
- Files a ROCm port stages into the package at build time (runtime-compiled kernel sources, a vendored runtime library) must be staged BEFORE `build_py`, not inside `build_ext`. A wheel build (what every current `pip install .` does) runs `build_py`, which resolves `package_data`, before `build_ext`, so staging from `build_extensions()` gives a first install with none of the files and every later install the PREVIOUS build's copies. `build_ext --inplace` never runs `build_py`, so a validation that only builds in place cannot see it: validate the documented install route from a fresh clone, then delete the clone and run the GPU test from elsewhere. Fix: a `build_py` subclass that stages, drops the lazily cached `data_files` (`self.__dict__.pop("data_files", None)`), then calls `super().run()`, registered only on the HIP branch so the CUDA `cmdclass` is unchanged. A shared library the extension links ships the same way, named explicitly in `package_data` (not `*.so`, which also matches a stale in-place `_C*.so`), with `-Wl,-rpath,$ORIGIN` in place of an rpath into the build tree; setuptools links without a shell, so `$ORIGIN` arrives literally (`readelf -d` shows `RUNPATH [$ORIGIN]`). (diff-surfel-tracing, linux-gfx942)
