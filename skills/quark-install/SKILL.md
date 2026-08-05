---
name: quark-install
description: >-
  Install or verify the AMD Quark package and its dependencies. Trigger for "install Quark", "set
  up Quark", "pip install amd-quark", dependency errors, import failures for quark modules,
  ModuleNotFoundError for quark, or missing C++ compiler errors. For PyTorch installation or torch
  version issues, use quark-torch-install instead.
---

# quark-install

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

## Purpose

Install the AMD Quark package and its dependencies after PyTorch is already set up. This skill handles Quark-specific setup: the `amd-quark` package, core dependencies, optional ONNX Runtime, LLM PTQ extras, and compiler requirements. It exists separately from `quark-torch-install` (which handles PyTorch) and from PTQ planning because getting the environment right is a prerequisite — a missing dependency or wrong compiler will cause cryptic failures later.

## Inputs

- `env_context.json` for OS/Python/accelerator facts
- `pytorch_install_result.json` confirming PyTorch is installed and verified

## Outputs: quark_install_result.json

Records the installed Quark version, optional extras (ONNX runtime, LLM PTQ deps), and verification status.

Schema: [`quark_install_result.schema.json`](https://github.com/amd/Quark/blob/1b229f781a1974cc742884e42d8eefc1eebb4f0a/.claude/skills-impl/shared/contracts/quark_install_result.schema.json)

```json
{
  "status": "ok",
  "quark_version": "0.12",
  "install_source": "pypi",
  "extras_installed": {
    "onnxruntime": false,
    "llm_ptq_deps": true
  },
  "verification": {
    "import_ok": true,
    "kernel_ok": true,
    "onnx_ops_ok": null
  }
}
```

On failure, set `status: "failed"` and include a `failure_reason` with the exact failing verification command.

## Quark Package Info

- **PyPI package**: `amd-quark` (current version: 0.12)
- **Install from PyPI (universal wheel, recommended default)**: `pip install amd-quark`. Works on any OS/Python/accelerator regardless of PyTorch version, but compiles the fast quantization kernels and ONNX custom-op library on first import (requires a C++ compiler, plus `nvcc`/`hipcc` for GPU).
- **Install a pre-built wheel (optional, PyTorch 2.10+)**: ships pre-compiled C++ extensions, so no C++ compiler and no first-run compilation are needed. Hosted on the AMD package index (Python 3.11–3.13); point `pip` at the matching index:

  ```bash
  pip install amd-quark --extra-index-url https://pypi.amd.com/quark/cpu/simple     # CPU
  pip install amd-quark --extra-index-url https://pypi.amd.com/quark/cu128/simple   # CUDA 12.8
  pip install amd-quark --extra-index-url https://pypi.amd.com/quark/rocm71/simple  # ROCm 7.1, Linux only
  pip install amd-quark --extra-index-url https://pypi.amd.com/quark/rocm72/simple  # ROCm 7.2, Linux only
  ```

- **Install from source**:

  ```bash
  git clone --recursive https://github.com/AMD/Quark
  cd Quark
  git submodule sync && git submodule update --init --recursive
  pip install .
  ```

- **Install from wheel**: `pip install amd_quark*.whl`

## Python Version Requirements

- **Supported**: Python 3.11, 3.12, 3.13
- **Not supported**: Python 3.14+
- **Recommended for new setups**: Python 3.13 via Miniforge/Miniconda

## ONNX Runtime (Optional)

- Version constraint: `>=1.22.2, <=1.24.2`
- GPU variant: `pip install onnxruntime-gpu` (for CUDA)
- CPU variant: `pip install onnxruntime`
- ROCm note: use the CPU variant of ONNX Runtime for ROCm 7.0+ due to build compatibility issues

## LLM PTQ Additional Dependencies

For running `quantize_quark.py`, install these extras:

```bash
pip install accelerate datasets evaluate>=0.4.0 gguf>=0.10.0 lm-eval transformers<5.3
```

## Core Dependencies (from requirements.txt)

```text
evaluate, joblib, ninja, numpy>=2.0, onnx>=1.21.0,<=1.22.0, onnxscript,
onnxslim>=0.1.84, pandas, plotly, protobuf, psutil, pydantic, rich, scipy,
sentencepiece, tqdm, zstandard
```

## Compiler Requirements

- **Linux**: `sudo apt install build-essential` (includes g++, needed for kernel compilation)
- **Windows**: Visual Studio 2022+ with "Desktop development with C++" workload

## Rules

- **Ensure PyTorch is already installed and verified.** If PyTorch is missing or mismatched with the accelerator, hand off to `quark-torch-install` first. Do not attempt to install Quark without a working PyTorch.
- **Never skip verification.** After installation, always run verification commands.
- **Show exact commands before execution.** The user should see every `pip install` command and every version before anything runs.

## Verification Commands

```bash
# Basic import
python -c "import quark; print('Quark version:', quark.__version__)"

# Optional: kernel compilation test
python -c "import quark.torch.kernel; print('Kernel compilation OK')"

# Optional: ONNX custom ops
python -c "import quark.onnx.operators.custom_ops; print('ONNX custom ops OK')"
```

## Interaction Flow

1. **Intake**: Determine what the user already has installed and what they need. Check if `quark-torch-install` has already run and PyTorch is verified.
2. **Plan**: Present the installation plan as a numbered sequence of commands, with version justifications.
3. **Confirm**: Required before any package installation. Show: what will be installed and what environment will be modified.
4. **Execute**: Run the installation commands.
5. **Verify**: Run all verification commands. Report pass/fail for each.

## Recovery

- **If verification fails**: Show the exact failing check and the most likely cause. Common issues:
  - `ModuleNotFoundError: No module named 'quark'` — Quark not installed or wrong Python environment
  - `ImportError: quark.torch.kernel` — Missing `build-essential` / C++ compiler
- **If PyTorch is missing or mismatched**: Hand off to `quark-torch-install` with the specific issue noted. Do not attempt to fix PyTorch issues from this skill.
- **If Python version is wrong**: Recommend creating a new conda environment with a supported version.

## Windows-Specific Notes

- If pip fails with long path errors: Enable Win32 long paths via Group Policy Editor (Computer Configuration > Administrative Templates > System > Filesystem > Enable Win32 long paths)
- WSL2 with Ubuntu is recommended as an alternative for Windows users
- ROCm is not supported on Windows — only CUDA and CPU

## Docker Option

Quark provides official Dockerfiles for reproducible environments:

- `Dockerfile.cuda` — NVIDIA CUDA (base image: `nvidia/cuda:11.8.0-base-ubuntu22.04`)
- `Dockerfile.rocm` — AMD ROCm (base image: `rocm/dev-ubuntu-24.04:6.4`)
- `Dockerfile.cpu` — CPU only (base image: `ubuntu:22.04`)

Build with:

```bash
docker build -f tools/ci/docker/images/Dockerfile.cuda \
  --build-arg PYTHON_VERSION=3.13 \
  --build-arg PYTORCH_VERSION=2.10.0 \
  --build-arg ACCELERATOR_VERSION=cuda-12.6 \
  -t quark:cuda .
```
