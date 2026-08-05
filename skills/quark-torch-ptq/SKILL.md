---
name: quark-torch-ptq
description: >-
  Torch LLM PTQ workflow for AMD Quark — for PyTorch / HuggingFace transformers models
  (safetensors input). Use when the user wants a complete PTQ pipeline: model inspection,
  quantization planning, script generation, and optional execution. Stops at the quantized output.
  Trigger for "quantize my model", "run PTQ", "run model quantization", "full quantization
  pipeline", "quantize Llama/Qwen/Mistral with FP8/INT4", or any request that spans more than one
  PTQ step. For a run that also validates and evaluates accuracy use quark-torch-llm-ptq-eval. Not
  for .onnx input models — use quark-onnx-ptq instead.
---

# quark-torch-ptq

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

When the workflow names one of these internal procedures, read its bundled reference before carrying out that step:

- [`quark-torch-quant-plan`](references/quark-torch-quant-plan.md)

## Purpose

Chain the full PTQ path — model intake → quantization planning → manifest generation → confirmed execution — while keeping the user informed at each checkpoint. This workflow orchestrates the atomic skills so the user does not have to manually chain them. It stops at the quantized output; for a run that also validates and evaluates the result, use the `quark-torch-llm-ptq-eval` recipe.

## Inputs

- Model path (HuggingFace ID or local directory)
- User goal (target precision, hardware, accuracy)
- Output directory for the quantized model
- `session_context.json` for user goal and constraints
- `env_context.json` for hardware facts
- `workspace_context.json` for validated paths
- `pytorch_install_result.json` and `quark_install_result.json` to confirm runtime is ready

## Outputs: run_manifest.yaml

Records the executed command and config. Side artifacts: quantized model files written to the user's output directory. The manifest is built in Step 3 above.

Schema: [`run_manifest.schema.json`](https://github.com/amd/Quark/blob/1b229f781a1974cc742884e42d8eefc1eebb4f0a/.claude/skills-impl/shared/contracts/run_manifest.schema.json)

## Interaction Flow

1. **Intake** → `quark-torch-model-intake` → `model_analysis.json`
2. **Plan** → `quark-torch-quant-plan` → `quant_plan.json`
3. **Confirm** → `run_manifest.yaml` (stop for user approval)
4. **Execute** → run the confirmed command, write quantized model

## CRITICAL RULES

1. **NEVER run `quantize_quark.py` directly.** Always go through the 4 steps below.
2. **NEVER skip a step.** Even if the user provides all details upfront, execute each step in order.
3. **STOP at every checkpoint** and wait for user confirmation before continuing.
4. **Show concrete output** at each step — tables, JSON, commands — not just prose descriptions.
5. **NEVER modify Quark's own source code or examples.** The Quark repo (`quark/`, `examples/`, `tools/`, `docs/`, `tests/`) is read-only from this workflow's perspective. See *Upstream Quark Code is Read-Only* below.

## Upstream Quark Code is Read-Only

The Quark repository (`quark/`, `examples/`, `tools/`, `docs/`, `tests/`, `pyproject.toml`, `requirements.txt`) may be read freely but never modified. This includes "just to make the script accept my flag" patches to `examples/torch/language_modeling/llm_ptq/quantize_quark.py` — they break reproducibility against a clean Quark install.

**When the shipped example does not cover the user's needs**, write a fresh standalone script in the user's working directory (or `/tmp/`) that imports from `quark`:

```python
# user_workspace/my_custom_ptq.py — NOT inside the Quark repo
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import Config
# ... user-specific logic ...
```

Reference *that* script in `run_manifest.yaml`. The shipped `quantize_quark.py` stays untouched; the run remains reproducible against any Quark version. If the user actually needs an upstream Quark change, surface it as a contribution — do not silently patch their local checkout.

## Required Artifact Flow

```text
Step 1 (Intake)   ──► model_analysis.json
Step 2 (Plan)     ──► quant_plan.json
Step 3 (Manifest) ──► run_manifest.yaml      (contains the exact command)
Step 4 (Execute)  ──► quantized model output (only after user says yes)
```

---

## Step 1: Model Intake

**Goal**: Produce `model_analysis.json`.

### Actions

1. Locate the Quark PTQ script (`quantize_quark.py`) under `<Quark repo>/examples/torch/language_modeling/llm_ptq/` — `find / -name quantize_quark.py -path "*/llm_ptq/*" 2>/dev/null | head -1` if the path is unknown. Record it for Step 3.

2. Call `quark-torch-model-intake` with the model path. It handles config parsing (no weight load), supported-template matching, and risk identification (MoE, >70B, transformers version constraints) and emits `model_analysis.json`.

### Output to Show User

Render the summary from `model_analysis.json`:

```text
Model Analysis:
  Model path:       Qwen/Qwen3-8B
  Model type:       qwen3
  Hidden layers:    36
  Linear layers:    ~224 (quantization targets)
  MoE:              No
  Exclude defaults: [lm_head]
  Risks:            <list>
  Compatibility:    OK
```

### >>> CHECKPOINT 1: Confirm model analysis is correct before continuing

---

## Step 2: Quantization Plan

**Goal**: Build `quant_plan.json` from the model analysis and user's stated preferences.

### Actions

1. **Determine the scheme.** If the user stated a scheme (e.g., "FP8"), use it. Otherwise, recommend based on their priority:

   | Priority | Recommended Scheme | Algorithm |
   |----------|--------------------|-----------|
   | Best accuracy | `fp8` or `ptpc_fp8` | `smoothquant` (optional) |
   | Smallest model | `int4_wo_32` | `awq` or `gptq` |
   | CPU deployment | `int8` | none |
   | AMD MI300X/MI355X | `fp8` or `amdfp4` | none |
   | NVIDIA H100 | `fp8` | none |
   | GGUF export | `uint4_wo_32` | `awq` |

2. **Fill the decision table.** Show ALL decisions with defaults:

   | Decision | Value | Reason |
   |----------|-------|--------|
   | `global_scheme` | `fp8` | User requested FP8 |
   | `kv_cache_scheme` | `fp8` | Recommended for FP8 inference |
   | `exclude_layers` | `["lm_head"]` | Standard — lm_head stays full precision |
   | `layer_quant_config` | `{}` | No per-pattern overrides (or e.g. `{"*self_attn*": "fp8"}` if the user asked to quantize attention with a non-global scheme) |
   | `algorithm` | `null` | RTN baseline (fastest) |
   | `calibration_dataset` | `pileval` | Fast default |
   | `num_calib_data` | `128` | Standard default |
   | `seq_len` | `512` | Standard default |
   | `evaluation_intent` | `smoke` | Quick PPL check after quantization |

3. **Ask the user if they want to change anything.**

### >>> CHECKPOINT 2: User MUST confirm or adjust the plan before continuing

Wait for the user to say "ok", "confirm", "looks good", "continue", or similar. If they request changes (e.g., "use smoothquant", "increase calibration to 256"), update the table and re-present.

---

## Step 3: Manifest Generation

**Goal**: Translate the confirmed plan into the exact `quantize_quark.py` command and produce `run_manifest.yaml`.

### Actions

1. **Build the command.** Map plan fields to CLI arguments:

   | Plan Field | CLI Argument |
   |------------|-------------|
   | model path | `--model_dir` |
   | output dir | `--output_dir` |
   | `global_scheme` | `--quant_scheme` |
   | `kv_cache_scheme` | `--kv_cache_dtype` (only if non-null) |
   | `layer_quant_config` | one `--layer_quant_scheme PATTERN SCHEME` per dict entry (only if non-empty) |
   | `exclude_layers` | `--exclude_layers` |
   | `algorithm` | `--quant_algo` (only if non-null) |
   | `num_calib_data` | `--num_calib_data` |
   | `seq_len` | `--seq_len` |
   | export format | `--model_export` (default: `hf_format`) |
   | data type | `--data_type auto` |
   | device | `--device cuda` |

2. **Present the exact command:**

   ```bash
   python3 <path_to_quantize_quark.py> \
     --model_dir <MODEL> \
     --output_dir <OUTPUT> \
     --quant_scheme <SCHEME> \
     --kv_cache_dtype <KV_SCHEME> \
     --num_calib_data <N> \
     --seq_len <LEN> \
     --model_export hf_format \
     --data_type auto \
     --device cuda
   ```

   **Note on `layer_quant_config` patterns.** Patterns are wildcard module-name matches against the model's `named_modules()`. Common LLaMA-style picks: `'*self_attn*'` (attention block), `'*experts*'` (MoE experts — covers all expert FFN submodules in one entry), `'lm_head'` (output head). For models with different naming (e.g. `attention`, `attn`, `self_attention`, DeepSeek MLA), the pattern matches nothing silently and no override is applied — inspect `named_modules()` and adjust the pattern before running.

3. **Show expected output** — `config.json` + `model.safetensors` (possibly sharded) + tokenizer files + `quark_profile.yaml` under `<OUTPUT>/`.

### >>> CHECKPOINT 3: Show the command and ask "shall I run this?"

Do NOT proceed to execution unless the user explicitly confirms. Acceptable confirmations: "yes", "run it", "go", "execute", or similar.

If the user says "no" or wants changes, go back to the relevant step.

---

## Step 4: Execute PTQ

**Goal**: Run the quantization command and report results.

### Precondition

**This step runs ONLY after the user explicitly confirms in Step 3.**

### Actions

1. **Create the output directory:**

   ```bash
   mkdir -p <OUTPUT_DIR>
   ```

2. **Pick the accelerator and GPU.** Read `env_context.json` for the backend. On ROCm, pin with `HIP_VISIBLE_DEVICES` (not `CUDA_VISIBLE_DEVICES`) — `--device cuda` still works on ROCm torch. On a shared host, check for a free GPU first and pin to it.

3. **Run the quantization command** from Step 3. Monitor for:
   - CUDA/ROCm OOM → suggest reducing `--num_calib_data` or using `--multi_gpu`
   - Transformers version errors → report the version mismatch
   - Model loading failures → check `trust_remote_code` or model path

4. **After completion, verify outputs exist:**

   ```bash
   ls -lh <OUTPUT_DIR>/
   ```

5. **Report results:**

   ```text
   Quantization complete:
     Output:      <OUTPUT_DIR>/
     Model size:  X.X GB
     Format:      HuggingFace SafeTensors
     Perplexity:  X.XX (wikitext)
   ```

### Error Recovery

- If quantization fails, do NOT retry blindly. Report the error and suggest fixes based on `quark-torch-debug` patterns.
- If OOM occurs, suggest: reduce `--num_calib_data`, reduce `--batch_size 1`, or use `--multi_gpu auto`.
- If transformers version is wrong, show the required version range.

---

## Complete Example

For an end-to-end walkthrough (FP8 quantization of Qwen3-8B), see
[`example-fp8-qwen3-8b.md`](example-fp8-qwen3-8b.md) alongside this file.

## Recovery

- If any upstream artifact is missing, stop and name the missing producer skill. Do not improvise a partial artifact.
- If the workflow hits a blocker (model cannot be loaded, scheme incompatible, execution fails), report it with diagnostic context rather than attempting ad-hoc fixes.
- If the user wants to change a decision mid-workflow (e.g., switch from FP8 to INT4 after seeing model analysis), go back to the relevant step — do not restart from scratch.
