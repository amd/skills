# Quantization planning

Use this reference to turn confirmed model analysis and user intent into
`quant_plan.json`.

## Provenance

Maintainer source paths in the Quark repository:

- `skills/_legacy_impl/l1-atomic/torch/quark-torch-quant-plan/SKILL.md`
- `quark/torch/quantization/config/template.py`
- `quark/torch/quantization/config/algo_configs.py`
- `examples/torch/language_modeling/llm_ptq/quantize_quark.py`
- `examples/torch/language_modeling/llm_ptq/README.md`

These are provenance, not runtime paths. The installed Quark package and the
public `quark-cli torch-llm-ptq` command define executable support.

## Verify support instead of counting it

Scheme and algorithm catalogs evolve. Never treat an old total as a permanent
fact. Query the installed package:

```bash
python3 - <<'PY'
from quark.torch import LLMTemplate
from quark.torch.quantization.config.algo_configs import get_supported_algorithm_types

print("schemes:", LLMTemplate.get_supported_schemes())
print("algorithms:", get_supported_algorithm_types())
PY
```

Also use `quark-cli torch-llm-ptq --help` for the exact command surface.
Algorithm registration alone does not guarantee a built-in configuration for
every model type; verify the selected model/algorithm pairing.

## Scheme guide

Current source includes these representative families:

- Weight-only INT4/UINT4: grouped and per-channel variants for size-oriented
  deployment; AWQ or GPTQ may improve accuracy when the model has a compatible
  configuration.
- `int8`: static per-tensor weight and activation quantization.
- `fp8` and `ptpc_fp8`: common GPU-oriented choices; PTPC uses per-channel
  weights and dynamic per-token activations.
- OCP microscaling variants such as MXFP4/MXFP6 and mixed MXFP4+FP8.
- AMD FP4 variants, NVFP4/block-scale variants, INT4+FP8, MX6, and BFP16.

Use exact names returned by the installed API. Hardware names are
recommendation context, not proof of kernel availability. Confirm the intended
inference stack separately.

Use only KV-cache schemes listed by `quark-cli torch-llm-ptq --help` and
supported by the selected model template.

## Required decisions

Show every decision and its reason:

| Decision | Requirement |
|---|---|
| `model.model_type` | Copied from the confirmed analysis; required |
| `model.analysis_ref` | Path to the `model_analysis.json` this plan is built on; required |
| `global_scheme` | Exact installed scheme name; required |
| `kv_cache_scheme` | `fp8` or `null` after compatibility review |
| `exclude_layers` | Start from the selected model template; do not assume every model excludes only `lm_head` |
| `layer_quant_config` | Pattern-to-scheme map; `{}` when unused |
| `algorithm` | `null`, one name, or a confirmed combination |
| calibration dataset | CLI-supported dataset |
| calibration sample count | Explicit integer justified by accuracy/memory tradeoff |
| sequence length | Explicit integer appropriate for the model and dataset |
| output format | Hugging Face safetensors |
| evaluation intent | `none`, `smoke`, or `full` |
| remote code | Explicitly approved or disabled |
| device strategy | Single accelerator, multi-device, or CPU |

The current CLI defaults to `pileval`, batch size `1`, sequence length `512`,
and `512` calibration samples. Defaults can change; record the actual chosen
values in the plan instead of omitting them. Reduce samples only with an
explicit accuracy/resource rationale.

## Per-layer overrides

`layer_quant_config` is the single map for module-pattern overrides. Each entry
becomes one repeated CLI pair:

```json
{
  "*self_attn*": "fp8",
  "*experts*": "int4_wo_32"
}
```

```bash
--layer_quant_scheme '*self_attn*' fp8 \
--layer_quant_scheme '*experts*' int4_wo_32
```

Patterns match model module names. Naming differs across LLaMA-style,
fused-QKV, MLA, MoE, and multimodal models. An unmatched pattern may silently
apply no override, so derive patterns from known architecture naming and flag
uncertainty before execution.

Passing `--exclude_layers` replaces template defaults. Omit the option to keep
the model-specific defaults; an empty `--exclude_layers` intentionally disables
all exclusions and must never be emitted accidentally.

## Resource and execution choices

- Prefer normal PTQ for calibration-based schemes.
- `--multi_device` can reduce accelerator-memory pressure but is slower and is
  intended for common quantization flows without algorithms.
- Emit `--no_trust_remote_code` unless the user explicitly approves executing
  model-repository code. The CLI trusts remote code when the flag is omitted.

Validate `quant_plan.json` with `contracts/quant_plan.schema.json`. Keep
`requires_confirmation: true` until the user confirms the displayed decision
table at Checkpoint 2, then write `false`.
