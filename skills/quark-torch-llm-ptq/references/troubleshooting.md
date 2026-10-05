# PTQ troubleshooting

Use this reference after a failed command. Do not retry before collecting the
exact command, exit status, full traceback, package versions, accelerator state,
and available memory.

## Provenance

Maintainer source paths in the Quark repository:

- `examples/torch/language_modeling/llm_ptq/quantize_quark.py`
- `examples/torch/language_modeling/llm_ptq/requirements.txt`
- `quark/torch/utils/llm/model_preparation.py`
- `quark/torch/utils/llm/compatibility.py`

The paths are provenance only. Diagnose with the installed environment and this
skill's artifacts.

## Evidence snapshot

This captures import failures specifically. For a fuller picture of versions and
accelerator state, use the check in `environment.md`.

```bash
python3 - <<'PY'
import shutil
import sys
print("Python:", sys.version)
print("quark-cli:", shutil.which("quark-cli") or "MISSING")
for name in ("torch", "transformers", "quark"):
    try:
        module = __import__(name)
        print(name, getattr(module, "__version__", "unknown"))
    except Exception as exc:
        print(name, "IMPORT FAILED:", repr(exc))
try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA runtime:", torch.version.cuda, "HIP runtime:", torch.version.hip)
    print("visible devices:", torch.cuda.device_count())
except Exception:
    pass
PY
```

Use `nvidia-smi` or `rocm-smi` when available. Do not install packages or change
the environment merely to gather evidence.

## Failure patterns

### Import or environment

- `No module named 'quark'`: wrong interpreter/environment or `amd-quark` is
  absent. Show the interpreter path before proposing installation.
- `torch.cuda.is_available() == False`: CPU-only/mismatched PyTorch build,
  unavailable device, or driver/runtime issue. Distinguish these before
  reinstalling.
- Missing `quark-cli` or command dependencies: verify that the active
  environment contains `amd-quark[cli]`. Install or repair it only after
  approval.

### Model/config loading

- Hub access failure: verify the repository ID, authentication, and network.
- Unrecognized custom model code: ask whether the model repository is trusted
  before omitting `--no_trust_remote_code` from the command.
- Tokenizer failure: verify tokenizer assets and model-specific dependencies.
- Unsupported `model_type`: compare with
  `LLMTemplate.list_available()`. Do not patch the installed CLI during this
  workflow.

### Transformers compatibility

The CLI package requirements constrain Transformers to `>5,<5.16`. Quark's
compatibility checker also detects model code that uses APIs removed after
specific releases, including `seen_tokens`, `get_max_length`, and
`get_usable_length`. The model's own `config.json` may provide a version hint.

Do not blindly upgrade or downgrade. Report:

1. installed Transformers version;
2. model-declared version;
3. exact missing/deprecated API;
4. Quark requirement;
5. smallest compatible change.

### Memory and device failures

- OOM during load with one visible accelerator: the model does not fit as
  loaded. Consider freeing memory or using `--multi_device`.
- OOM during calibration/quantization: reduce `num_calib_data`, sequence length,
  or batch size only after documenting the accuracy tradeoff.
- Huge multi-device MoE runs can exhaust Linux memory mappings and surface
  misleading OOM/resource errors. Capture `vm.max_map_count`; changing it is a
  privileged system mutation and requires explicit approval.
- On ROCm, keep `--device cuda` but use `HIP_VISIBLE_DEVICES` for pinning.

### Quantization configuration

- Invalid scheme: query `LLMTemplate.get_supported_schemes()` in the installed
  package; do not rely on a static count.
- Algorithm failure or missing built-in config: verify that the algorithm is
  registered and has a built-in configuration for the current model.
- Layer override has no effect: verify the wildcard against actual module naming
  for that architecture.
- Unexpected quantization of excluded layers: remember that an explicit
  `--exclude_layers` replaces template defaults, and an empty occurrence
  disables exclusions.

### Export/output

- Permission or disk error: verify parent permissions and free space without
  deleting user data.
- Output in an unexpected directory: regenerate the command with an absolute
  `--output_dir`; use an absolute `--model_dir` for local models as well.
- Missing output after exit code zero: inspect logs and the Hugging Face
  safetensors output; do not report success.

## Recovery protocol

1. Classify the failure as environment, load, compatibility, memory,
   configuration, quantization, or export.
2. Tie the diagnosis to concrete evidence.
3. Propose the smallest bounded change and state side effects.
4. Get confirmation before package, environment, system, or artifact mutation.
5. Return to the affected checkpoint, update the plan and manifest, show the
   revised exact command, and obtain execution approval again.
6. Preserve failed artifacts and logs. Never patch or bypass the installed
   `quark-cli`, and never retry blindly.
