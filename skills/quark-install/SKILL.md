---
name: quark-install
description: >
  Installs or verifies AMD Quark and ensures the selected Python environment has an accelerator-matched PyTorch. Applies to "install Quark", "set up Quark", "pip install amd-quark", Quark dependency or import failures, ModuleNotFoundError for quark, and Quark kernel compiler errors. Defaults to GPU and requires explicit confirmation before CPU mode. Covers PyPI universal, exactly matched native, local-wheel, and local-source installs. Does not handle standalone PyTorch requests, ONNX Runtime-only setup, or quantization.
---

# Quark Install

## Purpose

Install or verify `amd-quark` without relying on another skill, a Quark checkout, or pre-generated workflow artifacts. Detect the actual hardware, ensure PyTorch matches it, choose the smallest safe Quark install, obtain approval, execute it, and verify the requested capabilities.

## Prerequisites

- GPU support depends on the selected PyTorch build and Quark wheel. Record the actual runtime, host kernel and driver, and GPU architecture (`gfx...` from `gcnArchName` on AMD); verify compatibility with that environment instead of assuming a fixed ROCm version or GPU target.
- No container image is required. If running in a container, record its image name or digest when available and verify GPU device access.
- Inspect and preserve `QUARK_ACCELERATOR`, `PYTORCH_ROCM_ARCH`, `QUARK_BUILD_DISABLE_JIT_FALLBACK`, `HSA_OVERRIDE_GFX_VERSION`, `HIP_VISIBLE_DEVICES`, and `CUDA_VISIBLE_DEVICES`. Include any required changes in the confirmed install plan. Report `PIP_EXTRA_INDEX_URL` only as set or unset because its value may contain credentials.

## Inputs

- Python interpreter or environment to modify. Default to the active interpreter, but make its resolved path explicit.
- Requested Quark source: existing environment, PyPI, an AMD native-wheel index, a local wheel, or a local source directory. Default to the universal PyPI wheel.
- Optional version constraint.
- Compute intent: GPU by default; CPU only after explicit user confirmation.
- Required capabilities: basic Quark import, Quark PyTorch kernels, ONNX custom operators, or `quark-cli`.

Do not require pre-existing environment or install-result artifacts. A user-provided wheel or source directory is an input, not a skill dependency.

## Outputs

Always write `quark_install_result.json` in the user's working directory at a terminal outcome and validate it against [`references/contracts/quark_install_result.schema.json`](references/contracts/quark_install_result.schema.json).

Preserve the compatibility fields `status`, `quark_version`, `install_source`, `extras_installed`, `verification`, and `failure_reason`. Also record the resolved interpreter, imported Quark module path, PyTorch version/backend, GPU availability, whether CPU mode was explicitly confirmed, exact executed commands, and `pip check` result.

- Use `status: "ok"` only when installation and every requested verification pass.
- Use `status: "skipped"` when a suitable existing install is verified or the user declines changes.
- Use `status: "failed"` with the exact failing command and error for install or verification failure.
- Set unrequested optional checks to `null`; do not report `llm_ptq_deps: true` merely because `quark-cli --help` works.

## Installation Rules

- Quark requires Python 3.11, 3.12, or 3.13 and PyTorch 2.2 or newer.
- The `amd-quark` runtime wheel does not install PyTorch. Resolve and verify PyTorch before installing Quark.
- GPU is the default. A CUDA/ROCm build tag alone is insufficient: GPU readiness requires `torch.cuda.is_available()`, a positive device count, and a successful tensor operation on `cuda` (PyTorch uses this device spelling for both CUDA and ROCm).
- If PyTorch is missing, CPU-only, backend-mismatched, or unable to access a GPU, stop and ask whether to install/fix the matching GPU build or explicitly continue in CPU mode. Never install or accept CPU mode silently.
- Install a GPU PyTorch build only from an official index matching detected CUDA/ROCm support. Query available versions first, choose a Python-compatible stable release satisfying `torch>=2.2`, show the exact command, and obtain approval:

  ```bash
  "<PYTHON>" -m pip index versions torch --index-url "https://download.pytorch.org/whl/<GPU_INDEX>"
  "<PYTHON>" -m pip install "torch==<VERSION>" --index-url "https://download.pytorch.org/whl/<GPU_INDEX>"
  ```

- After explicit CPU-mode confirmation, use the official CPU index instead of an unqualified PyPI install:

  ```bash
  "<PYTHON>" -m pip install "torch==<VERSION>" --index-url "https://download.pytorch.org/whl/cpu"
  ```

- The universal PyPI wheel is the safe default. It supports different PyTorch backends but compiles Quark kernels/custom operators on first use and therefore may require a C++ compiler plus `nvcc` or `hipcc`:

  ```bash
  "<PYTHON>" -m pip install "amd-quark"
  ```

- Quark Torch import currently requires Transformers even though the base `amd-quark` wheel may not declare it. When kernel verification is requested and Transformers is absent, preview and confirm a release/model-compatible install; use `"transformers<5.16"` as the current upper bound unless a narrower requirement is known. Treat a missing Transformers module as a dependency failure, not a compiler failure.
- Native-wheel availability changes by release. Never select one from a hard-coded matrix or a bare `--extra-index-url`. Use only an observed candidate whose filename/local version exactly matches the active PyTorch major.minor, CUDA/ROCm major.minor (or CPU), CPython tag, OS, and architecture; pin its complete version with `===`:

  ```bash
  "<PYTHON>" -m pip install "amd-quark===<FULL_VERSION_WITH_VARIANT>" --index-url "<AMD_INDEX_URL>" --extra-index-url "https://pypi.org/simple"
  ```

- Apply the same ABI checks to a user-provided native wheel. A `py3-none-any` wheel is universal. For trusted local input:

  ```bash
  "<PYTHON>" -m pip install "<ABSOLUTE_WHEEL_PATH>"
  "<PYTHON>" -m pip install --no-build-isolation "<ABSOLUTE_SOURCE_DIR>"
  ```

- Use the source form only when the directory contains `pyproject.toml` or `setup.py`, PyTorch is already verified, and the user explicitly trusts and requests executing that source.
- Add `[cli]` only when requested. Before doing so, inspect installed `onnxruntime*` distributions and the dry-run plan because the current CLI extra may add CPU `onnxruntime`; do not silently combine it with `onnxruntime-gpu` or bypass dependencies with `--no-deps`.
- ONNX Runtime variant selection is outside this skill. Verify and report an existing runtime when ONNX custom operators are requested, but do not silently choose or replace its CPU/GPU distribution.
- Always use the same absolute `"<PYTHON>" -m pip`; never use unqualified `pip` or `sudo pip`.

## Interaction Flow

Complete these steps in order.

### Step 1 — Inspect the environment

Resolve `<PYTHON>` to the exact interpreter that would be modified. Quote paths containing spaces. Run read-only checks:

```bash
"<PYTHON>" -c "import platform, sys; print('executable:', sys.executable); print('prefix:', sys.prefix); print('python:', platform.python_version()); print('os:', platform.platform()); print('machine:', platform.machine())"
"<PYTHON>" -m pip --version
"<PYTHON>" -m pip show amd-quark torch
"<PYTHON>" -c "import torch; ok=torch.cuda.is_available(); n=torch.cuda.device_count(); print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('hip:', torch.version.hip); print('gpu_available:', ok); print('device_count:', n); print('device:', torch.cuda.get_device_name(0) if ok and n else None)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
rocm-smi --showproductname
hipcc --version
```

Run only hardware commands available on the host; command-not-found is evidence, not permission to install utilities. Do not import optional Quark kernels/custom operators during intake because a universal wheel may compile them.

#### Checkpoint 1 — Confirm compute mode

Stop before Quark planning if Python is unsupported, PyTorch is below 2.2, hardware is ambiguous, PyTorch is CPU-only, the build family conflicts with detected hardware, or a GPU build cannot access a device.

Ask: “I found `<EVIDENCE>`. Shall I install/fix the matching GPU PyTorch build, or do you explicitly want CPU mode?”

Do not continue in CPU mode without an explicit answer. If GPU mode is selected, show and approve the exact official-index PyTorch command, then verify a real GPU tensor operation before continuing.

### Step 2 — Select the install

1. If a suitable Quark version is already installed and no upgrade or reinstall was requested, skip installation and proceed to verification.
2. Prefer an explicitly requested and validated local wheel/source.
3. Use a native wheel only after exact candidate and ABI validation; otherwise use the universal PyPI wheel.
4. Add a requested version constraint to the package spec, for example `"amd-quark==<VERSION>"` or `"amd-quark[cli]==<VERSION>"`.
5. Preview every required package change, including Transformers for kernel use, with `"<PYTHON>" -m pip install --dry-run "<PACKAGE_SPEC>"`. If dry-run is unsupported, report that limitation rather than mutating the environment.
6. Treat the installed distribution metadata as authoritative. Do not claim the source-tree version or dependency limits apply to a different published release.

### Step 3 — Confirm changes

Show:

- the resolved interpreter and environment;
- detected hardware, PyTorch build, GPU runtime result, and confirmed GPU/CPU mode;
- selected Quark source and, for a native wheel, every matched ABI dimension;
- dry-run dependency changes and exact commands;
- whether verification may trigger C++/CUDA/HIP compilation.

Ask: “Shall I run these exact commands?”

CPU-mode acceptance and command execution both require explicit approval. Do not install, upgrade, uninstall, invoke a system package manager, or compile optional components without it.

### Step 4 — Install and verify

After approval, run only the confirmed commands. Install/fix PyTorch first when required and verify the selected compute mode before installing Quark. Stop on the first failure and preserve full output. For a wheel install, run import verification from a neutral directory outside any Quark source checkout and require the imported module path to reside under the selected environment; for a source/editable install, require it to resolve to the confirmed source directory.

Run basic checks:

```bash
"<PYTHON>" -m pip check
"<PYTHON>" -c "from importlib.metadata import version; import quark, torch; print('amd-quark:', version('amd-quark')); print('quark:', quark.__version__); print('quark_file:', quark.__file__); print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('hip:', torch.version.hip)"
```

For GPU mode, require an actual operation:

```bash
"<PYTHON>" -c "import torch; assert torch.version.cuda or torch.version.hip, 'CPU-only PyTorch build'; assert torch.cuda.is_available(), 'GPU runtime unavailable'; assert torch.cuda.device_count() > 0, 'No visible GPU'; x=torch.ones(1024, device='cuda'); y=(x*x).sum(); torch.cuda.synchronize(); print(torch.cuda.get_device_name(0), y.item())"
```

For requested Quark kernels, assert that the extension loaded. Use the first command for a universal wheel; for a native wheel, disable JIT fallback to prove the packaged extension is valid:

```bash
"<PYTHON>" -c "import quark.torch.kernel; from quark.torch.kernel.hw_emulation import extensions; assert extensions.kernel_ext is not None; print('Quark kernel OK')"
"<PYTHON>" -c "import os; os.environ['QUARK_BUILD_DISABLE_JIT_FALLBACK']='1'; import quark.torch.kernel; from quark.torch.kernel.hw_emulation import extensions; assert extensions.kernel_ext is not None; print('Native Quark kernel OK')"
```

Run only other requested checks:

```bash
"<PYTHON>" -m quark.experimental.cli.main -h
"<PYTHON>" -c "import onnxruntime as ort; from quark.onnx.operators.custom_ops import get_library_path; p=get_library_path('CPU'); so=ort.SessionOptions(); so.register_custom_ops_library(p); print('Quark ONNX custom ops OK:', p)"
```

Do not run both kernel commands. Report base installation, GPU runtime, and optional capabilities separately. Write and validate `quark_install_result.json`; do not claim success from an install exit code or build tag alone.

## Recovery

- **CPU-only or unavailable GPU:** keep GPU as the default, show hardware/build/runtime evidence, and ask whether to repair GPU PyTorch or explicitly accept CPU. Do not silently fall back.
- **No exact native wheel:** show each mismatched ABI dimension, propose the universal wheel, and request approval again.
- **Compiler failure:** universal-wheel kernel/custom-op checks may require a C++ compiler plus `nvcc` or `hipcc`. Report the missing tool and get approval before system changes.
- **Missing Transformers:** Quark Torch import can fail after the extension loads if Transformers is absent. Show the exact `ModuleNotFoundError`, preview a compatible Transformers install, and request approval before retrying.
- **CLI/ONNX Runtime conflict:** show all installed/planned `onnxruntime*` distributions. Prefer a separate environment or an explicitly revised plan; never hide the conflict with `--no-deps`.
- **Dependency or import failure:** preserve `pip check`, the exact failing command, versions, and full error. Prefer a fresh environment over forced downgrades.
- **Wrong environment or source shadowing:** compare `sys.executable`, `pip --version`, distribution version, and `quark.__file__`; rerun wheel verification from a neutral directory with one absolute interpreter path.
- **Network/index failure:** retain the exact URL and error. Do not switch indexes or retry repeatedly without confirmation.
- **Unsupported Python:** propose a fresh Python 3.13 environment and restart at Step 1 after approval.
