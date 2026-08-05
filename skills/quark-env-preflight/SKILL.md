---
name: quark-env-preflight
description: >-
  Collect and normalize environment facts (OS, Python, GPU, CUDA/ROCm, container state) before
  Quark installation or PTQ planning. Trigger for "check my environment", "what GPU do I have",
  "is my setup ready for Quark", or when any accelerator-related assumption is unconfirmed. Also
  trigger before any install/quantization step where hardware facts are missing.
---

# quark-env-preflight

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

When the workflow names one of these internal procedures, read its bundled reference before carrying out that step:

- [`quark-torch-quant-plan`](references/quark-torch-quant-plan.md)

## Purpose

Collect raw environment facts and normalize them into `env_context.json` so that downstream skills (install, model intake, PTQ planning) can make correct decisions without guessing. This skill is the single source of truth for hardware and toolchain state — getting it wrong here cascades into wrong install commands, incompatible packages, or failed quantization runs.

## Inputs

- None — runs standalone, reads from environment

## Outputs: env_context.json

Carries OS, Python, and hardware facts collected at preflight.

Schema: [`env_context.schema.json`](https://github.com/amd/Quark/blob/1b229f781a1974cc742884e42d8eefc1eebb4f0a/.claude/skills-impl/shared/contracts/env_context.schema.json)

```json
{
  "environment": {
    "os": "linux",
    "python": "3.13",
    "containerized": false
  },
  "hardware": {
    "accelerator": "nvidia-cuda",
    "cuda_version": "12.6",
    "gpu_count": 1,
    "gpu_model": "RTX 4090",
    "memory_gb": 24
  }
}
```

`env_context.json` is for raw machine facts only — no installation results, no user goal, no open questions. Installed PyTorch and Quark versions live in `pytorch_install_result.json` and `quark_install_result.json` respectively. Unresolved questions belong in `session_context.json` (owned by `the matching public quark-torch skill`).

## What to Detect

### OS and Python

- OS family and version (Linux distro, Windows version, WSL)
- Python version — Quark requires `>=3.11, <3.14` (pyproject.toml says `>=3.11`, setup.py says `>=3.9.0,<3.14`)
- Whether running inside conda/venv/virtualenv and the environment name
- Container state: Docker, Podman, or bare metal

### Accelerator

- **AMD ROCm**: check `ROCM_PATH`, `HIP_VISIBLE_DEVICES`, `rocm-smi` output, ROCm version (supported: 6.4, 7.0, 7.1)
- **NVIDIA CUDA**: check `CUDA_HOME`, `CUDA_VISIBLE_DEVICES`, `nvidia-smi` output, CUDA version (supported: 11.8, 12.6, 12.8, 13.0)
- **CPU-only**: only set `cpu` when the user explicitly requests CPU-only OR no GPU evidence exists after thorough checking
- Normalize to one of: `amd-rocm`, `nvidia-cuda`, `cpu`, `unknown`

### Existing Quark Installation

- Check `python -c "import quark; print(quark.__version__)"` — current version is 0.12
- Check PyTorch version and its CUDA/ROCm build tag (`torch.version.cuda`, `torch.version.hip`)
- Check if `torch` and accelerator backend are from the same family (never mix CUDA torch with ROCm environment)

## Detection Commands

```bash
# OS and Python
python --version
uname -a  # or systeminfo on Windows

# GPU detection (try both, one will fail gracefully)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null
rocm-smi --showproductname 2>/dev/null

# Environment variables
echo $CUDA_HOME $CUDA_VISIBLE_DEVICES $ROCM_PATH $HIP_VISIBLE_DEVICES

# Existing packages
pip show amd-quark torch 2>/dev/null
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip)"
```

## Rules

- **Only collect and normalize facts.** Do not choose install commands, PTQ schemes, or workflow branches — that is the job of `quark-install` or `quark-torch-quant-plan`.
- **Prefer explicit user input** over heuristic inference when they conflict. If the user says "I'm on ROCm" but `CUDA_HOME` is also set, trust the user.
- **Missing evidence ≠ CPU.** Treat missing `CUDA_VISIBLE_DEVICES`, `ROCM_PATH`, or `HIP_VISIBLE_DEVICES` as insufficient evidence for `cpu`. Keep the accelerator as `unknown` until something definitive is found.
- **Never upgrade `unknown` to `cpu`** unless the user explicitly says CPU-only or detection confirms zero GPU hardware.
- **Capture version dependencies** even if the exact version is not yet known. For example, if the user mentions "ROCm" but not the version, record `accelerator: amd-rocm, rocm_version: unknown`.

## Interaction Flow

1. **Intake**: Ask what downstream task the user is headed toward (install? PTQ? just checking?). This determines which facts are critical vs. nice-to-have.
2. **Detect**: Run the detection commands above. Present what was found in a clear summary table.
3. **Clarify**: If the accelerator is ambiguous or versions are uncertain, ask the user — do not guess. Show them the conflicting evidence.
4. **Emit**: Write the confirmed facts to `env_context.json`. Hand any unresolved items back to the caller (typically `the matching public quark-torch skill`) so they land in `session_context.json` under `open_questions`.

## Recovery

- If environment evidence is contradictory (e.g., both CUDA and ROCm libraries present), keep both raw facts in the summary, set `accelerator=unknown`, and explain the conflict.
- If a detection command cannot run (e.g., no permissions for `nvidia-smi`), report exactly which signal is blocked and suggest the smallest manual check: "Run `nvidia-smi` in a terminal with GPU access and paste the output."
- If Python version is outside `3.11–3.13`, flag it immediately — Quark will not work.
