---
name: quark-torch-llm-ptq
description: >
  Runs an end-to-end AMD Quark post-training quantization workflow for PyTorch /
  Hugging Face LLMs: inspect a Hub or local model, choose a quantization plan,
  create reproducible artifacts, request execution approval, and produce a
  quantized model. Applies to Llama, Qwen, Mistral, and similar transformer LLM
  requests involving FP8, INT4, or another Quark scheme. Does not handle .onnx
  model inputs or ONNX PTQ.
---

# Quark Torch PTQ

## Purpose

Take a PyTorch / Hugging Face LLM from model identification through confirmed
AMD Quark PTQ. Perform intake, planning, manifest generation, execution, and
output verification as one self-contained workflow.

This skill stops at the quantized model. It does not accept `.onnx` model input,
train or fine-tune a model, or modify Quark package/source files.

## Prerequisites

- Python 3.11 to 3.13, accelerator-matched PyTorch 2.2 or newer, `amd-quark[cli]`, and `datasets`.
- The required ROCm version and GPU architecture depend on the PyTorch build and selected quantization scheme. Record the actual runtime, GPU architecture (`gfx...` from `gcnArchName` on AMD), host kernel, and driver, and verify support for the confirmed plan.
- No container image is required. If running in a container, record its image name or digest when available and verify GPU device access.
- Inspect and preserve `HIP_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, `HSA_OVERRIDE_GFX_VERSION`, `PYTORCH_ROCM_ARCH`, and `PYTORCH_HIP_ALLOC_CONF`. Include any required changes to device visibility, architecture, or memory allocation in the confirmed plan.

## Inputs

- Model source: a Hugging Face repository ID or local model directory.
- Output directory.
- Quantization intent: requested precision/scheme, target hardware, accuracy
  priority, and optional calibration settings.
- Optional environment facts: Python, PyTorch, accelerator, available memory,
  installed `amd-quark`, and Transformers versions.

Do not require pre-existing workflow artifacts. Create all artifacts in the
user's working directory by following this skill's local references:

- [`references/model-intake.md`](references/model-intake.md)
- [`references/quant-plan.md`](references/quant-plan.md)
- [`references/environment.md`](references/environment.md)
- [`references/troubleshooting.md`](references/troubleshooting.md)

## Outputs

Produce these three artifacts before or during execution:

1. `model_analysis.json`, validated against
   [`references/contracts/model_analysis.schema.json`](references/contracts/model_analysis.schema.json).
2. `quant_plan.json`, validated against
   [`references/contracts/quant_plan.schema.json`](references/contracts/quant_plan.schema.json).
3. `run_manifest.yaml`, validated against
   [`references/contracts/run_manifest.schema.json`](references/contracts/run_manifest.schema.json).

The quantized model and its configuration/tokenizer files are written under the
confirmed output directory. Record actual files and the final status in the
manifest.

## Interaction Flow

Always complete the following four steps in order. Show concrete facts,
artifacts, and commands. Stop at every checkpoint and wait for the user.

### Step 1 — Model intake

1. Confirm whether the model source is local or remote. For a local source,
   resolve it to an absolute path and verify that the directory and
   `config.json` exist. For a remote source, preserve the repository ID.
2. Follow `references/model-intake.md`. Read configuration only; do not load
   model weights during intake.
3. Determine `model_type`, architecture/loading hints, hidden-layer facts,
   multimodal or MoE signals, default exclusions, compatibility risks, and a
   defensible estimate of quantizable linear layers.
4. Write schema-valid `model_analysis.json` and show its summary.

#### Checkpoint 1 — Confirm the model analysis

Ask the user to confirm or correct the model analysis.

Do not plan quantization until the user confirms.

### Step 2 — Quantization plan

1. Follow `references/quant-plan.md` and use the confirmed analysis plus the
   user's priorities.
2. Select and explain the global scheme, optional KV-cache scheme, per-pattern
   overrides, exclusions, algorithms, and calibration data.
3. Treat scheme, algorithm, and model-template support as version-dependent.
   Use the current installed Quark API or `quark-cli torch-llm-ptq --help` when
   a choice needs verification; do not rely on historical list sizes.
4. Show a decision table, write schema-valid `quant_plan.json`, and set
   `requires_confirmation: true` until approved.

#### Checkpoint 2 — Confirm the quantization plan

Ask the user to confirm or adjust the complete plan.

After confirmation, update `requires_confirmation` to `false`. If the user
changes a decision, rewrite and revalidate the plan before continuing.

### Step 3 — Manifest and execution confirmation

Use the public `quark-cli torch-llm-ptq` command installed by
`amd-quark[cli]`. Do not import its implementation module directly or locate,
copy, generate, or patch another PTQ runner.

Build an argument-array-safe command equivalent to:

```bash
quark-cli torch-llm-ptq \
  --model_dir "<MODEL_OR_ABSOLUTE_LOCAL_PATH>" \
  --output_dir "<ABSOLUTE_OUTPUT_PATH>" \
  --quant_scheme "<SCHEME>" \
  --num_calib_data "<N>" \
  --seq_len "<LENGTH>" \
  --device cuda \
  --no_trust_remote_code
```

Add only confirmed options:

- `--dataset <NAME>` and `--batch_size <N>` when the plan changed them from the
  CLI defaults.
- `--kv_cache_dtype <SCHEME>` for confirmed KV-cache quantization.
- One `--layer_quant_scheme <PATTERN> <SCHEME>` per override.
- `--quant_algo <comma-separated-list>` when required.
- `--exclude_layers <patterns...>` only when overriding template defaults.
- `--multi_device` when its constraints are understood.
- `--no_trust_remote_code` unless the user explicitly accepts executing remote
  model code. The CLI trusts remote code when this flag is omitted.
- `--skip_evaluation` when the requested scope ends strictly at model output.
- `--evaluation_dataset <NAME>` when evaluation is requested with a
  non-default CLI-supported dataset.

Resolve local model and output directories to absolute paths, then quote all
user-controlled paths and values. On ROCm, `--device cuda` is still the PyTorch
device spelling; use `HIP_VISIBLE_DEVICES` to pin a GPU when needed. Resolve
`quark-cli` from the same Python environment that provides `amd-quark`.

Write `run_manifest.yaml` with:

- `workflow: quark-torch-ptq`
- input and output paths;
- all four checkpoint reasons;
- an analyze step producing `model_analysis.json`;
- a plan step producing `quant_plan.json`;
- a generate step producing `run_manifest.yaml`;
- a run step containing the exact command and expected output directory.

Confirm the environment before showing the command, using
`references/environment.md`. A missing package or an accelerator-mismatched
PyTorch build should surface here, not after the execution gate.

Show the full command, destination, estimated resource needs, remote-code
choice, and expected outputs.

#### Checkpoint 3 — Approve execution

Ask: “Shall I run this exact command?”

This is the execution gate. Do not create the output directory, download model
weights, change the environment, or run PTQ without explicit approval such as
“yes”, “run it”, or “execute”. A prior plan confirmation is not execution
approval.

### Step 4 — Execute and verify

Only after Checkpoint 3 approval:

1. Create the output directory if needed.
2. Run the exact confirmed `quark-cli` command.
3. Monitor output. On failure, stop; collect the command, exit status, full
   error, versions, and resource state, then use
   `references/troubleshooting.md`. Never retry blindly.
4. On success, inspect the output directory and report actual model shards,
   configuration/tokenizer files, size, format, and any requested metrics.
5. Update `run_manifest.yaml` with the observed status and outputs without
   changing the recorded command.

#### Checkpoint 4 — Accept the verified result

Present the verified result and ask the user to accept it or request a bounded
follow-up.

Do not claim success from exit status alone. If expected files are absent,
report a partial/failed result and preserve diagnostics.

## Recovery

- Missing or invalid artifact: regenerate it from the corresponding local
  reference and validate it against the local schema; do not invent fields
  around a validation error.
- Intake uncertainty: mark `analysis_status` as `partial`, record a risk, and
  ask for the missing fact. Do not load weights merely to fill metadata.
- Unsupported model type or scheme: report the installed Quark evidence and
  ask whether to change the plan. Do not patch the installed CLI.
- OOM or device failure: retain the failed manifest and propose the smallest
  plan change, such as fewer calibration samples, a shorter sequence length, or
  multi-device execution. Return to Checkpoint 2.
- Dependency or compatibility failure: show exact installed and required
  versions. Get confirmation before package changes, then return to
  Checkpoint 3 with a newly recorded command.
- User changes intent: return to the earliest affected checkpoint and preserve
  still-valid artifacts. Never bypass the execution confirmation.
