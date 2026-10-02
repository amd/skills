# Environment

Use this reference to confirm the environment before proposing a command, and to
report exact versions when something fails. This skill never installs or
upgrades anything on its own; it reports what is missing and asks.

## Provenance

Maintainer source paths in the Quark repository:

- `pyproject.toml`
- `requirements-cli.txt`
- `tools/ci/install_torch.sh`
- `examples/torch/language_modeling/llm_ptq/requirements.txt`

These paths are provenance only. At runtime, inspect the installed environment
rather than trusting any version written here.

## What the CLI needs

The workflow invokes the installed `quark-cli torch-llm-ptq` command.

- Python 3.11 to 3.13.
- `torch`, installed separately and matched to the accelerator. It is not a
  dependency of `amd-quark`, and the default PyPI build is usually wrong for a
  ROCm or CUDA machine.
- `amd-quark[cli]`, which installs the `quark-cli` entry point and its optional
  command-line dependencies, including Transformers and Accelerate.
- `datasets`, required by calibration and evaluation data loading.

Install the accelerator-matched PyTorch build first, then install
`amd-quark[cli]`. Confirm before changing either package.

## Matching PyTorch to the accelerator

The accelerator decides the wheel index, not the torch version alone. Upstream
publishes each combination under `https://download.pytorch.org/whl/<slug>`,
where `<slug>` is `cpu`, a CUDA tag such as `cu128`, or a ROCm tag such as
`rocm7.2`. A torch release exists on some slugs and not others, so pick the slug
first and then a version that slug publishes.

Detect what the machine actually has before choosing:

```bash
# AMD
rocm-smi --showproductname 2>/dev/null | head -5
cat /opt/rocm/.info/version 2>/dev/null

# NVIDIA
nvidia-smi --query-gpu=name,driver_version --format=csv 2>/dev/null
```

On ROCm, PyTorch keeps the CUDA spelling: `torch.cuda.is_available()`,
`--device cuda`, and `torch.version.hip` holds the HIP runtime. Use
`HIP_VISIBLE_DEVICES` to pin devices.

## Confirming the environment

```bash
python3 - <<'PY'
import importlib.metadata as md
import shutil
import sys

print("Python:", sys.version.split()[0])
for name in ("torch", "amd-quark", "transformers", "datasets", "accelerate"):
    try:
        print(f"{name}: {md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}: MISSING")

print("quark-cli:", shutil.which("quark-cli") or "MISSING")

try:
    import torch
    print("accelerator available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    print("CUDA runtime:", torch.version.cuda, "| HIP runtime:", torch.version.hip)
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("device 0:", torch.cuda.get_device_name(0),
              f"| {p.total_memory / 1024**3:.0f} GB")
except Exception as exc:
    print("torch check failed:", repr(exc))
PY
```

Run `quark-cli torch-llm-ptq --help` after the package checks. A missing command
usually means the active shell and the Python environment used for installation
do not match. Pass absolute paths for local models and output directories so
they remain stable throughout CLI execution.

A torch build whose accelerator does not match the machine reports
`available: False` while still importing cleanly. Treat that as an environment
mismatch and show the evidence rather than proposing a CPU run the user did not
ask for.

## When something is missing

Report the interpreter path, the missing package, and the smallest install that
unblocks the confirmed plan. Get approval before changing the environment, then
return to Checkpoint 3 and show the command again. Never install packages merely
to gather evidence, and never silently switch the device to work around a
mismatched build.
