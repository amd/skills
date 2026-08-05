---
name: quark-torch-install
description: >-
  Install or verify the correct PyTorch build for a user's accelerator backend before Quark
  installation. Trigger for "install PyTorch", "pip install torch", "set up torch for ROCm", "set
  up torch for CUDA", "torch version mismatch", "CPU-only torch installed",
  torch.cuda.is_available() returns False, or any request to get the correct PyTorch build
  running.
---

# quark-torch-install

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

## Purpose

Install the correct PyTorch build for the user's accelerator backend. PyTorch must be installed before Quark because Quark depends on PyTorch at both install time and runtime. Getting this wrong — installing the CPU build on a GPU machine, or mixing a CUDA-built PyTorch with a ROCm environment — causes cryptic failures that are hard to diagnose later. This skill exists separately from `quark-install` so that PyTorch setup has a clear, single-responsibility boundary.

## Inputs

- `env_context.json` with detected accelerator info

## Outputs: pytorch_install_result.json

Records the installed PyTorch build, accelerator backend tag, and verification status.

Schema: [`pytorch_install_result.schema.json`](https://github.com/amd/Quark/blob/1b229f781a1974cc742884e42d8eefc1eebb4f0a/.claude/skills-impl/shared/contracts/pytorch_install_result.schema.json)

```json
{
  "status": "ok",
  "pytorch_version": "2.5.1+cu126",
  "accelerator_tag": "cu126",
  "torchvision_version": "0.20.1+cu126",
  "torchaudio_version": "2.5.1+cu126",
  "verification": {
    "import_ok": true,
    "cuda_available": true,
    "gpu_count": 1
  }
}
```

On failure, set `status: "failed"` and include a `failure_reason` with the exact failing verification command.

## Python Version Requirements

- **Supported**: Python 3.11, 3.12, 3.13
- **Not supported**: Python 3.14+
- **Recommended for new setups**: Python 3.13 via Miniforge/Miniconda

## PyTorch Version Matrix

**Authoritative source**: `tools/ci/install_torch.sh`

Before generating install commands, always read this script to get the current list of verified accelerator/PyTorch combinations. The script defines which PyTorch versions are tested with each accelerator backend (ROCm, CUDA, CPU) and the corresponding `--index-url` values.

### How to read the source

1. Open `tools/ci/install_torch.sh` and locate the version arrays or case/if blocks that map accelerator tags to PyTorch versions.
2. Extract the accelerator identifier (e.g., `rocm7.1`, `cu126`, `cpu`).
3. Extract the supported PyTorch version list for that accelerator.
4. Construct the install command using the pattern below.

### Install command pattern

```bash
# ROCm — torchvision only, no torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/<rocm_tag>

# CUDA — includes torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/<cuda_tag>

# CPU — includes torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Replace `<rocm_tag>` with the ROCm version tag (e.g., `rocm6.4`, `rocm7.0`, `rocm7.1`) and `<cuda_tag>` with the CUDA version tag (e.g., `cu118`, `cu126`, `cu128`, `cu130`). Always use the exact tags from `install_torch.sh`.

**Critical**: Never use bare `pip install torch` for GPU setups — it installs the CPU version by default.

## Rules

- **Always read `tools/ci/install_torch.sh` before generating install commands.** The version matrix changes with each Quark release. Never rely on memorized version numbers — always verify against the upstream script.
- **Always detect the accelerator before choosing the PyTorch package.** Run or reference `quark-env-preflight` if hardware facts are missing. The entire install plan depends on getting this right.
- **Bind torch and accelerator to the same backend.** Never mix a CUDA-built PyTorch with ROCm environment or vice versa. If `torch.version.cuda` shows a CUDA version but the user says they want ROCm, flag the conflict.
- **Never skip verification.** After installation, always run verification commands.
- **If accelerator is unclear, stop after the plan.** Present the install plan but do not execute. Hand the gap back to `the matching public quark-torch skill` so it lands in `session_context.json`'s `open_questions`, and ask the user to confirm their hardware.
- **Show exact commands before execution.** The user should see every `pip install` command, every version, and every `--index-url` before anything runs.

## Verification Commands

```bash
# PyTorch backend check
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('HIP:', torch.version.hip)"

# GPU availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

## Interaction Flow

1. **Intake**: Determine what the user already has installed and what accelerator they need. Check if `quark-env-preflight` has already run.
2. **Plan**: Present the accelerator-specific PyTorch installation command with version justifications.
3. **Confirm**: Required before any package installation. Show: what will be installed, which `--index-url` will be used, and what environment will be modified.
4. **Execute**: Run the installation commands.
5. **Verify**: Run all verification commands. Report pass/fail for each.

## Recovery

- **If `torch.cuda.is_available() == False`**: PyTorch CPU build was installed instead of GPU build. Show the exact uninstall + reinstall commands with the correct `--index-url`.
- **If torch and accelerator mismatch**: Explain that PyTorch must be reinstalled with the correct `--index-url`. Show the exact uninstall + reinstall commands.
- **If Python version is wrong**: Recommend creating a new conda environment with a supported version (3.11, 3.12, or 3.13).

## Windows-Specific Notes

- If pip fails with long path errors: Enable Win32 long paths via Group Policy Editor (Computer Configuration > Administrative Templates > System > Filesystem > Enable Win32 long paths)
- WSL2 with Ubuntu is recommended as an alternative for Windows users
- ROCm is not supported on Windows — only CUDA and CPU
