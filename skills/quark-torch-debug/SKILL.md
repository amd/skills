---
name: quark-torch-debug
description: >-
  Diagnose failed Quark Torch PTQ installation, execution, script generation, or export attempts.
  Use when the user reports a torch-side error, stack trace, invalid artifact, missing dependency,
  CUDA OOM, version mismatch, or unexpected PTQ results. Trigger for "Quark error", "PTQ failed",
  "quantization crashed", "CUDA out of memory", "import error", "model loading failed", "wrong
  results", any Python traceback mentioning quark.torch / torch._dynamo / transformers /
  accelerate. Not for onnxruntime tracebacks — use quark-onnx-debug.
---

# quark-torch-debug

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

## Purpose

Convert failures into a structured diagnostic report with the smallest safe recovery path. Debugging Quark issues is tricky because errors can originate from many layers — Python environment, PyTorch, transformers, CUDA/ROCm drivers, model architecture, or Quark itself. This skill systematically narrows down the root cause.

## Inputs

- Error message and stack trace from a failing run
- `env_context.json`, `pytorch_install_result.json`, `quark_install_result.json` (optional, for environment and install state)

## Outputs: validation_report.md

Diagnostic report with root cause, evidence, and the smallest safe fix.

Schema: [`validation_report.schema.json`](https://github.com/amd/Quark/blob/1b229f781a1974cc742884e42d8eefc1eebb4f0a/.claude/skills-impl/shared/contracts/validation_report.schema.json)

```markdown
# Debug Report

## Common Error Patterns

### Installation Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `ModuleNotFoundError: No module named 'quark'` | Quark not installed or wrong Python env | `pip install amd-quark` or activate correct conda env |
| `ImportError: quark.torch.kernel` | Missing C++ compiler | `sudo apt install build-essential` |
| `torch.cuda.is_available() == False` | CPU-only PyTorch installed | Reinstall PyTorch with correct `--index-url` |
| `RuntimeError: CUDA error: no kernel image` | PyTorch CUDA version ≠ system CUDA | Match PyTorch build to system CUDA version |

### Model Loading Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `ValueError: Unrecognized model in config` | `trust_remote_code` needed | Add `--trust_remote_code` flag |
| `OSError: Can't load tokenizer` | Missing tokenizer files or sentencepiece | `pip install sentencepiece` and check model path |
| `ImportError: ... requires transformers>=X.Y.Z` | Transformers version too old | `pip install transformers==X.Y.Z` |
| `OutOfMemoryError` during loading | Model too large for single GPU | Use `--multi_gpu auto` or `--multi_device` |

### Quantization Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `CUDA out of memory` during quantization | Insufficient GPU memory | Reduce `--num_calib_data` or `--batch_size`, or use `--multi_gpu` |
| `RuntimeError: expected scalar type Half` | Data type mismatch | Set `--data_type float16` or `bfloat16` explicitly |
| `KeyError: 'model.layers.0...'` | Model architecture not matching template | Check `model_type` in config.json matches Quark template |
| `ValueError: ... is not a valid quantization scheme` | Typo in scheme name | Check against the 21 supported schemes |
| `AssertionError` in AWQ/GPTQ | Algorithm config mismatch | Verify `--quant_algo_config_file` matches model architecture |

### Export Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `ModuleNotFoundError: gguf` | GGUF package missing | `pip install gguf>=0.10.0` |
| `ONNX export failed` | Model has unsupported ops | Try `--model_export hf_format` instead |
| `PermissionError` on output dir | No write permission | Check output directory permissions |

### Transformers Compatibility

Quark checks compatibility before quantization. Known issues:
- `seen_tokens` removed in transformers > 4.53.3
- `get_max_length` removed in transformers > 4.48.3
- `get_usable_length` removed in transformers > 4.53.3
- General constraint: `transformers < 5.3`

## Diagnostic Process

1. **Read the error** — get the exact error message, full stack trace, and the command that was run.
2. **Identify the layer** — is this an install problem, model loading, quantization runtime, or export issue?
3. **Check the environment** — Python version, PyTorch version, CUDA/ROCm version, Quark version, transformers version.
4. **Match against known patterns** — use the tables above.
5. **Propose the fix** — smallest change that resolves the issue without side effects.

## Diagnostic Commands

```bash
# Environment snapshot
python -c "
import sys; print('Python:', sys.version)
try:
    import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'HIP:', torch.version.hip)
except: print('PyTorch: not installed')
try:
    import quark; print('Quark:', quark.__version__)
except: print('Quark: not installed')
try:
    import transformers; print('Transformers:', transformers.__version__)
except: print('Transformers: not installed')
"

# GPU memory status
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null
rocm-smi --showmemuse 2>/dev/null

# Check Quark compatibility
python -c "
from quark.torch.utils.llm.compatibility import check_compatibility_before_quantization
print('Compatibility check available')
"
```

## Rules

- **Always ask for the full error message and the command that was run.** Partial errors lead to wrong diagnoses.
- **Do not guess the fix.** Narrow down the root cause first, then propose a specific solution.
- **Confirm before mutating.** If the fix involves reinstalling packages, changing environment, or modifying user files, present the plan and get confirmation.
- **Consider cascading effects.** Upgrading transformers might fix one issue but break compatibility with Quark. Always check version constraints.

## Error Summary

CUDA out of memory during quantization of Qwen/Qwen3-8B with FP8

## Root Cause

Single GPU (24GB) insufficient for FP8 quantization with 512 calibration samples

## Evidence

- GPU memory: 22GB / 24GB used at crash point
- Model size: ~16GB in FP16
- Calibration data: 512 samples at seq_len=512

## Fix

Reduce calibration data or use multi-GPU:

```bash
# Option A: Reduce calibration samples
python quantize_quark.py ... --num_calib_data 64 --batch_size 1

# Option B: Use multi-GPU (if available)
python quantize_quark.py ... --multi_gpu auto
```

## Prevention

For models > 7B parameters on GPUs with < 48GB VRAM, start with
--num_calib_data 64 and increase if memory allows.

```text

## Interaction Flow

1. **Gather evidence**: Get the error message, stack trace, command, and environment info.
2. **Classify**: Determine which layer failed (install / load / quantize / export).
3. **Diagnose**: Match against known patterns and run diagnostic commands if needed.
4. **Propose fix**: Present the smallest change that resolves the issue.
5. **Confirm**: Get user approval before any environment changes.
6. **Verify**: After the fix, re-run the failing step to confirm resolution.

## Recovery

- If the root cause is upstream (transformers API change, PyTorch bug), hand off to `the Quark repository's internal Torch documentation drift checker` or `the Quark repository's internal Torch skill synchronizer`.
- If the fix requires a PyTorch reinstall (wrong build, version mismatch, CPU instead of GPU), hand off to `quark-torch-install` with the specific requirement noted.
- If the fix requires a Quark package or dependency change, hand off to `quark-install` with the specific requirement noted.
- If the error is in a custom model or unsupported architecture, suggest registering a custom template via `LLMTemplate.register_template()`.
