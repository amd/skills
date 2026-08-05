---
name: quark-torch-quant-plan
description: >
  Build a Quark Torch LLM PTQ quantization plan from model analysis and user intent. Use when the
  user needs quantization scheme recommendations, exclusion lists, algorithm selection, KV cache
  decisions, per-layer overrides, or a draft quant_plan. Trigger for "quantize with FP8",
  "what scheme should I use", "plan PTQ", "INT4 quantization", "choose quantization config",
  "quantization plan", or when the user has a model analysis and needs to decide how to quantize it.
layer: l1-atomic
primary_artifact: quant_plan.json
source_knowledge:
  - quark/torch/quantization/config/template.py
  - examples/torch/language_modeling/llm_ptq/README.md
  - examples/torch/language_modeling/llm_ptq/quantize_quark.py
---

# quark-torch-quant-plan

## Purpose

Convert a model analysis plus the user's intent into a confirmed `quant_plan.json`. This skill makes the quantization decisions — which scheme, which algorithm, what to exclude — without generating scripts or executing PTQ. The plan is the contract between the user's intent and the execution step.

## Inputs

- `model_analysis.json` from quark-torch-model-intake
- `env_context.json` for accelerator-aware scheme recommendations
- User preferences (target precision, accuracy goal)

## Outputs: quant_plan.json

Records the chosen scheme, algorithm, layer overrides, calibration settings, and evaluation intent.

Schema: [`quant_plan.schema.json`](https://github.com/amd/Quark/blob/1b229f781a1974cc742884e42d8eefc1eebb4f0a/.claude/skills-impl/shared/contracts/quant_plan.schema.json)

```json
{
  "model": {
    "model_type": "qwen3",
    "analysis_ref": "./model_analysis.json"
  },
  "global_scheme": "fp8",
  "kv_cache_scheme": "fp8",
  "exclude_layers": ["lm_head"],
  "layer_quant_config": {},
  "algorithm": null,
  "calibration": {
    "dataset": "pileval",
    "num_calib_data": 128,
    "seq_len": 512
  },
  "evaluation_intent": "smoke",
  "requires_confirmation": false
}
```

## Available Quantization Schemes (21 total)

### Weight-Only INT4 (best for deployment size reduction)

| Scheme | Description | Use Case |
|--------|-------------|----------|
| `int4_wo_32` | INT4, group size 32 | Highest accuracy among INT4 |
| `int4_wo_64` | INT4, group size 64 | Good balance |
| `int4_wo_128` | INT4, group size 128 | Smaller overhead |
| `int4_wo_per_channel` | INT4, per-channel | Least overhead |
| `uint4_wo_32/64/128/per_channel` | Unsigned INT4 variants | GGUF export compatibility |

### Weight+Activation INT8

| Scheme | Description | Use Case |
|--------|-------------|----------|
| `int8` | INT8 per-tensor for both W and A | CPU deployment, good accuracy |

### FP8 (best accuracy-size tradeoff for GPU inference)

| Scheme | Description | Use Case |
|--------|-------------|----------|
| `fp8` | FP8 E4M3 per-tensor | Standard GPU quantization |
| `ptpc_fp8` | Per-Token-Per-Channel FP8 | Higher accuracy, dynamic activation quantization |

### OCP Microscaling Formats

| Scheme | Description | Use Case |
|--------|-------------|----------|
| `mxfp4` | OCP MXFP4 | Aggressive compression |
| `mxfp6_e3m2` | OCP MXFP6 (E3M2) | Better range |
| `mxfp6_e2m3` | OCP MXFP6 (E2M3) | Better precision |
| `mxfp4_mxfp6_e2m3` | MXFP4 weights + MXFP6 activations | Mixed precision |
| `mxfp4_fp8` | MXFP4 weights + FP8 activations | Mixed precision |

### AMD-Specific

| Scheme | Description | Use Case |
|--------|-------------|----------|
| `amdfp4` | amdfp4, group size 16 | AMD MI300X optimized |
| `amdfp4_g32` | amdfp4, group size 32 | AMD MI300X, less overhead |

### Other

| Scheme | Description | Use Case |
|--------|-------------|----------|
| `nvfp4` | NVFP4: FP4 group_size=16 with FP8 E4M3 scale | NVIDIA Blackwell/Hopper |
| `mx6` | MX6 format | Experimental |
| `bfp16` | Block Floating Point 16-bit | Experimental |
| `int4_wa_64` | INT4 weights + activations, group 64 | Research |

## Available Algorithms (7 primary)

| Algorithm | Compatible Schemes | Description |
|-----------|-------------------|-------------|
| `awq` | INT4/UINT4 weight-only | Activation-aware weight quantization — finds optimal per-channel scaling |
| `gptq` | INT4/UINT4 weight-only | Second-order weight optimization — often better than AWQ for small models |
| `smoothquant` | INT8, FP8 | Migrates quantization difficulty from activations to weights |
| `autosmoothquant` | INT8, FP8 | Automatic SmoothQuant with optimal alpha search |
| `rotation` | Various | Rotation-based optimization to equalize weight distribution |
| `gptaq` | INT4/UINT4 | GPTAQ variant combining GPTQ with activation quantization |
| `qronos` | Various | Custom algorithm for time-series-aware quantization |

Algorithms can be combined: `--quant_algo awq,smoothquant`

## KV Cache Quantization

- Only `fp8` is supported for KV cache (`--kv_cache_dtype fp8`)
- Adds `--min_kv_scale` option (default 0.0) to prevent extreme scale values
- `--kv_cache_post_rope` quantizes KV cache after RoPE (inside cache) instead of at k_proj/v_proj outputs — can improve accuracy for some models

## Decision Guide

Help the user choose based on their priorities:

**"I want the best accuracy"** → `fp8` or `ptpc_fp8`, optionally with `smoothquant`
**"I want the smallest model"** → `int4_wo_32` with `awq` or `gptq`
**"I need CPU deployment"** → `int8` (the only scheme that works well on CPU)
**"I need GGUF format"** → `uint4_wo_32` with `awq`, export as GGUF
**"I'm on AMD MI300X"** → `amdfp4` for best hardware utilization
**"I'm on NVIDIA H100/Blackwell"** → `fp8` or `nvfp4`
**"I want to experiment"** → `mxfp4` for aggressive compression research

## Decision Table (MUST show to user)

**ALWAYS present this table to the user and WAIT for confirmation before finalizing.** Do not skip this step.

Fill in the "Value" column based on the user's request and model analysis, then show:

| Decision | Value | Reason |
|----------|-------|--------|
| `global_scheme` | _(fill)_ | _(why this scheme)_ |
| `kv_cache_scheme` | _(fill: `fp8` or `null`)_ | _(explain)_ |
| `exclude_layers` | `["lm_head"]` | Standard — lm_head stays full precision |
| `layer_quant_config` | _(fill: dict of `pattern -> scheme`, or `{}` if none)_ | _(explain which patterns and why)_ |
| `algorithm` | _(fill: algorithm or `null`)_ | _(explain)_ |
| `calibration_dataset` | `pileval` | Fast default |
| `num_calib_data` | `128` | Standard default |
| `seq_len` | `512` | Standard default |
| `evaluation_intent` | `smoke` | Quick PPL check post-quantization |

After showing the table, ask: "Confirm this plan? Any changes?"

**Do NOT proceed until the user confirms.**

## Layer-Specific Overrides via `layer_quant_config`

The `layer_quant_config` plan field is a dict of `pattern -> scheme` pairs. It is the
single mechanism for any "quantize layer/module X with scheme Y" intent — including
attention modules, MoE experts, lm_head, etc. Each entry emits one
`--layer_quant_scheme PATTERN SCHEME` CLI argument.

```json
"layer_quant_config": {
  "*self_attn*": "fp8",
  "lm_head": "int8",
  "*experts*": "fp8"
}
```

translates to:

```bash
--quant_scheme <global_scheme> \
--layer_quant_scheme '*self_attn*' fp8 \
--layer_quant_scheme lm_head int8 \
--layer_quant_scheme '*experts*' fp8
```

When a user asks for attention-module quantization (e.g. "self_attn in fp8"), populate
this field with the appropriate pattern (commonly `*self_attn*` for LLaMA-style models;
adjust for models whose attention submodule has a different name). Do NOT introduce a
dedicated attention field — keep all per-pattern overrides in `layer_quant_config`.

## Rules

- **Keep scope to plan creation only.** Do not generate scripts, do not run quantization, do not export. Those are separate skills.
- **Require a model analysis first.** Without knowing the model architecture and layer count, you cannot make informed scheme recommendations. If `model_analysis.json` is missing, route back to `quark-torch-model-intake`.
- **Always present the decision table** before finalizing. The user should explicitly confirm the scheme, algorithm, and exclusions.
- **If a risky scheme is chosen** (e.g., `mxfp4` on a model where accuracy loss may be significant), keep the user's choice but record the risk in the plan.

## Interaction Flow

1. **Check prerequisites**: Is `model_analysis.json` available? If not, route to `quark-torch-model-intake` first.
2. **Gather intent**: What does the user care about most — accuracy, size, speed? What hardware will run inference?
3. **Present the decision table**: Show defaults, explain the tradeoffs, and let the user adjust.
4. **Confirm**: Always required. Show the final plan summary before writing it.
5. **Emit**: Write `quant_plan.json`.

## Recovery

- If the model analysis is incomplete, produce a draft plan with `requires_confirmation: true` and note what facts are missing.
- If the user picks an unusual combination (e.g., `awq` with `fp8` — AWQ is designed for INT4), explain why it might not work well and suggest alternatives, but respect the user's choice if they insist.
- If calibration dataset preferences are unclear, default to `pileval` with 128 samples — it is the fastest option and works for most models.
