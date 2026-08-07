---
name: quark-torch-model-intake
description: >-
  Inspect a HuggingFace transformers / safetensors model and prepare metadata for Quark Torch PTQ
  planning. Use for model path validation, architecture detection, quantization target discovery,
  layer counting, risk assessment, or transformers compatibility checks. Trigger for "analyze my
  model", "check this model", "what architecture is this", "can Quark quantize X", "is this model
  supported". Not for .onnx model graphs — use quark-onnx-model-intake.
---

# quark-torch-model-intake

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

When the workflow names one of these internal procedures, read its bundled reference before carrying out that step:

- [`quark-torch-quant-plan`](references/quark-torch-quant-plan.md)
- [`quark-workspace-validate`](references/quark-workspace-validate.md)

## Purpose

Validate the target model and extract the structural facts that `quark-torch-quant-plan` needs to make correct quantization decisions. This step exists because different model architectures have different quantization requirements — MoE models need expert module replacement, some models require `trust_remote_code`, and certain architectures have known compatibility issues with specific transformers versions.

## Inputs

- Model path (HuggingFace ID or local directory)
- `env_context.json` for Python and accelerator constraints
- `workspace_context.json` for the validated model path

## Outputs: model_analysis.json

Captures architecture facts, quantizable layer count, transformer compatibility, and risks.

Schema: [`model_analysis.schema.json`](https://github.com/amd/Quark/blob/1b229f781a1974cc742884e42d8eefc1eebb4f0a/.claude/skills-impl/shared/contracts/model_analysis.schema.json)

```json
{
  "analysis_status": "complete",
  "model": {
    "model_path": "Qwen/Qwen3-8B",
    "model_type": "qwen3",
    "trust_remote_code": false,
    "transformers_version_required": ">=4.48.0,<5.3",
    "loading_class": "AutoModelForCausalLM",
    "is_moe": false,
    "num_hidden_layers": 36,
    "hidden_size": 4096,
    "estimated_size_gb": 16.0
  },
  "quantization_targets": {
    "linear_layer_count": 224,
    "has_non_linear_experts": false,
    "exclude_defaults": ["lm_head"],
    "needs_moe_preparation": false
  },
  "risks": [
    {
      "severity": "low",
      "message": "Model size fits in single GPU with 24GB+ VRAM for FP8/INT8 schemes.",
      "recovery_hint": "Use --multi_gpu for INT4 with large calibration datasets if OOM occurs."
    }
  ]
}
```

## Supported Model Architectures

Quark has built-in templates for 36 model families:

| Category | Models |
|----------|--------|
| **Standard LLMs** | llama, mistral, opt, phi, phi3, qwen, qwen2, gptj, cohere, olmo |
| **Advanced LLMs** | qwen3, qwen3_next, deepseek, deepseek_v2, deepseek_v3, deepseek_v32 |
| **MoE Models** | mixtral, dbrx, llama4, qwen2_moe, qwen3_moe, qwen3_5_moe, gpt_oss, granitemoehybrid, glm4_moe |
| **Vision-Language** | mllama, deepseek_vl_v2, qwen3_vl_moe |
| **Other** | chatglm, gemma2, gemma3, gemma3_text, grok-1, instella, kimi_k25, minimax_m2 |

Models not in this list may still work if they follow standard HuggingFace `AutoModelForCausalLM` patterns, but need extra attention.

## What to Extract

### From config.json

- `model_type` — must match a Quark template name (e.g., `"llama"`, `"qwen3"`, `"mistral"`)
- `num_hidden_layers` — determines the number of quantizable linear layers
- `hidden_size`, `intermediate_size` — affects memory estimates
- `num_attention_heads`, `num_key_value_heads` — relevant for KV cache quantization
- Architecture-specific fields (e.g., `num_experts` for MoE models)

### Quantization Targets

- Count total linear layers (these are what gets quantized)
- Identify default exclusions — `lm_head` is almost always excluded
- For MoE models, note that expert modules need special preparation via `prepare_for_moe_quant()`

### Transformer Compatibility

Some models require specific minimum transformers versions:

- `llama4` → transformers >= 4.51.0
- `gpt_oss`, `granitemoehybrid` → transformers >= 4.55.1
- `qwen3_vl_moe` → transformers >= 4.57.0
- `qwen3_5_moe` → transformers >= 5.2.0
- General requirement for LLM PTQ: `transformers < 5.3`

### Special Loading Requirements

- `deepseek_vl_v2` → uses `AutoModel` instead of `AutoModelForCausalLM`
- `mllama` → uses `MllamaForConditionalGeneration`
- `gpt_oss` → needs `Mxfp4Config(dequantize=True)` for loading
- Models with custom code → need `trust_remote_code=True`

### Risks

Flag anything that could cause failures downstream:

- Model type not in Quark's template list
- Transformer version incompatibility
- Very large models that may need `--multi_gpu` or `--file2file_quantization`
- MoE models that need module replacement

## Concrete Actions

### Action 1: Read model config (NEVER load full weights)

```bash
python3 -c "
from transformers import AutoConfig
import json
config = AutoConfig.from_pretrained('<MODEL_PATH>', trust_remote_code=True)
info = {
    'model_type': config.model_type,
    'num_hidden_layers': config.num_hidden_layers,
    'hidden_size': config.hidden_size,
    'intermediate_size': getattr(config, 'intermediate_size', None),
    'num_attention_heads': config.num_attention_heads,
    'num_key_value_heads': getattr(config, 'num_key_value_heads', None),
    'vocab_size': config.vocab_size,
    'num_experts': getattr(config, 'num_local_experts', getattr(config, 'num_experts', None)),
    'torch_dtype': str(getattr(config, 'torch_dtype', 'unknown')),
}
print(json.dumps(info, indent=2))
"
```

### Action 2: Check model size (for local models)

```bash
du -sh /path/to/model/
ls -lh /path/to/model/*.safetensors
```

### Action 3: Present summary table to user

After running the above, format the results as:

```text
Model Analysis:
  Model path:       <path or HuggingFace ID>
  Model type:       <model_type from config>
  Hidden layers:    <num_hidden_layers>
  Linear layers:    ~<estimated count>
  MoE:              Yes/No
  Exclude defaults: [lm_head]
  Risks:            <list or "None">
  Compatibility:    OK / <version constraints>
```

This table is what the user sees at Checkpoint 1 of `quark-torch-ptq`.

## Rules

- **Run or reference `quark-workspace-validate` first** to confirm that model paths are valid before attempting to read `config.json`.
- **Do not load the full model** during intake. Reading `config.json` and listing files is enough — loading weights is expensive and belongs to the quantization step.
- **Preserve ambiguity** when a model reference could be local or remote. Note both possibilities and let the user resolve.
- **Surface new environment constraints** (e.g., model requires transformers >= 4.57.0) by recording them in `model_analysis.json` under `risks` and asking `the matching public quark-torch skill` to add them to `session_context.json`'s `open_questions`. Do not write directly to `env_context.json`.

## Interaction Flow

1. **Confirm model source**: Is it a local directory or a HuggingFace ID? Check if `quark-workspace-validate` already confirmed the path.
2. **Read config**: Extract architecture facts from `config.json`. Present a summary table to the user.
3. **Assess compatibility**: Check model type against Quark's template list. Flag any transformer version requirements.
4. **Identify risks**: Note anything that could affect downstream PTQ planning.
5. **Emit**: Write `model_analysis.json`. Surface any new constraints back to `the matching public quark-torch skill` so they land in `session_context.json`.

## Recovery

- If `analysis_status: "partial"` — some facts were extracted but the model could not be fully inspected. Common cause: model needs `trust_remote_code=True` but the user has not approved it.
- If model type is not in Quark's template list — report this clearly. The user may need to register a custom template (see `LLMTemplate.register_template()` in `quantize_quark.py`).
- If transformer version is incompatible — show the exact version mismatch and the upgrade/downgrade command.
