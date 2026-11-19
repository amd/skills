## Strategy B: pytorch extension

Torch hipifies extension sources at build time. Do not add a compat header and do not hand-rename symbols.

- Building a `CUDAExtension` on a ROCm torch automatically runs `torch.utils.hipify` on the extension's `.cu`/`.cuh` sources and links the HIP runtime (`amdhip64`, `c10_hip`, `torch_hip`). See `torch/utils/cpp_extension.py`.
- Keep sources in CUDA spelling. In a hipified TU that is correct under both hipify generations (see below), so it needs no guard. Fix only what hipify cannot (warp size, see the fault classes) in source, guarded by `USE_ROCM`.
- Build against a ROCm torch. If the tree was hipified once and is stale after edits, re-run the project's hipify step before rebuilding (a known incremental-build gotcha: edits to `.cu` can recompile the stale hipified mirror unless you re-hipify first).
- For projects shipping their own `.cu` plus a setup.py, the change is often just: build against a ROCm torch and fix the fault classes below.

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
