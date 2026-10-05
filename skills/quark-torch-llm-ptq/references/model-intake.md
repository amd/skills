# Model intake

Use this reference to produce `model_analysis.json` without loading model
weights.

## Provenance

Maintainer source paths in the Quark repository:

- `skills/_legacy_impl/l1-atomic/torch/quark-torch-model-intake/SKILL.md`
- `examples/torch/language_modeling/llm_ptq/quantize_quark.py`
- `quark/torch/quantization/config/template.py`
- `quark/torch/utils/llm/model_preparation.py`
- `quark/torch/utils/llm/compatibility.py`

These paths document provenance only. At runtime use the installed package,
model metadata, and files in this skill directory.

## Safe inspection

First classify the model source:

- Local: verify the directory and `config.json`; inspect safetensors filenames
  and sizes without opening full tensors.
- Hugging Face Hub: use `AutoConfig.from_pretrained` with the repository ID.
  Network access and custom remote code require user awareness.

Read configuration with a small script:

```bash
python3 - <<'PY'
import json
from transformers import AutoConfig

model = "<MODEL>"
config = AutoConfig.from_pretrained(
    model,
    trust_remote_code=False,
)
keys = (
    "model_type", "architectures", "num_hidden_layers", "hidden_size",
    "intermediate_size", "num_attention_heads", "num_key_value_heads",
    "num_local_experts", "num_experts", "n_routed_experts",
    "transformers_version", "dtype",
)
print(json.dumps({key: getattr(config, key, None) for key in keys}, indent=2, default=str))
PY
```

If configuration loading requires custom code, explain that
`trust_remote_code=True` executes code from the model repository and ask before
enabling it.

## Template and loading facts

Do not copy a historical template count into the analysis. The installed Quark
version is authoritative:

```bash
python3 -c "from quark.torch import LLMTemplate; print(LLMTemplate.list_available())"
```

Compare `model_type` with that result. A missing template is a high-risk or
partial analysis. Record the missing template and ask the user how to proceed
rather than modifying the installed CLI.

Current Quark loading code has architecture-specific paths. Examples include:

- `mllama` and `llama4` conditional-generation loaders.
- `gpt_oss` loading with `Mxfp4Config(dequantize=True)`.
- `deepseek_vl_v2` through `AutoModel`.
- current multimodal and MoE families through their matching Transformers
  classes and preprocessing replacements.

Record architecture and version evidence from the model configuration. Quark's
example requirements constrain Transformers to `>5,<5.16`; some architecture
classes additionally have minimum versions in model preparation (for example
Llama 4, GPT-OSS/Granite MoE, Qwen3 VL MoE, and Qwen3.5 families). Do not
generalize those examples into one minimum for every model.

## Quantization targets

The schema fixes the shape. Required are `analysis_status`, `model.model_path`,
`model.model_type`, `quantization_targets.linear_layer_count`,
`quantization_targets.exclude_defaults`, and a `risks` array whose entries each
carry `severity` and `message`. Put the facts below inside that shape; extra
properties are allowed, but renaming a required one is not.

Record:

- `model_type`, loading architecture, hidden-layer count, hidden size, and
  optional expert count;
- whether modality sub-configs such as `vision_config`, `audio_config`,
  `image_config`, or `video_config` are present;
- whether the model is MoE or has fused/pre-quantized expert tensors;
- template-provided default exclusions (often, but not always, `lm_head`);
- estimated quantizable linear count and how it was estimated;
- local/remote source, model size estimate, and trust-remote-code decision.

Do not use one hard-coded formula for all architectures. Dense decoder blocks,
fused QKV projections, multimodal wrappers, and MoE experts differ. If a
reliable count cannot be derived from config alone, use a conservative estimate
and explain it. Do not load full weights during intake.

## Compatibility and risks

Add structured risks for:

- unregistered or ambiguous `model_type`;
- model-declared Transformers version mismatch;
- use of APIs known to vary by Transformers version;
- custom remote code;
- memory pressure, huge MoE models, or likely need for multi-device execution;
- pre-quantized inputs;
- missing tokenizer/config/safetensors files for a local model.

Use `analysis_status: partial` whenever a material fact is unavailable.
Otherwise use `complete`. Validate the result with
`contracts/model_analysis.schema.json`, then show a summary and wait at
Checkpoint 1.
