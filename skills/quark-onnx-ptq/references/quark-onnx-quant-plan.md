---
name: quark-onnx-quant-plan
description: >
  Build a Quark ONNX PTQ quantization plan from `model_analysis.json` and user intent. Use when the
  user needs preset selection (XINT8 / A8W8 / A16W8 / BF16 / BFP16 / MX* / MXFP* …),
  calibration method choice (MinMax / Entropy / Percentile / Distribution / NonOverflow / MinMSE /
  LayerWisePercentile), algorithm selection (CLE / AdaRound / AdaQuant / BiasCorrection /
  AutoMixprecision), deployment-target gating (CPU / CUDA / ROCm / AMD
  NPU CNN / AMD NPU Transformer), op-type include/exclude lists, weights-only INT4 (MatMulNBits)
  decisions, dynamic vs static quantization, or a draft `quant_plan.json`. Trigger for "what
  preset should I use for my ONNX model", "choose XINT8 vs A8W8", "plan ONNX PTQ", "INT4
  "BFP16 / MXFP4 for my model", "calibration method for Ryzen AI",
  "should I use AdaRound or AdaQuant", "SmoothQuant alpha", or when the user has a model analysis
  and needs to decide how to quantize an ONNX model.
layer: l1-atomic
primary_artifact: quant_plan.json
source_knowledge:
  - quark/onnx/quantization/config/custom_config.py
  - quark/onnx/quantization/config/algorithm.py
  - quark/onnx/quantization/config/config.py
  - quark/onnx/calibration/methods.py
  - examples/onnx/yolo_quantization/quantize_yolo.py
  - examples/onnx/accuracy_improvement/quarot/quantize_model.py
  - tutorials/onnx/ryzen_ai/resnet50/onnx_ryzen_ai_resnet50_tutorial.ipynb
  - tutorials/onnx/ryzen_ai/yolov8/onnx_ryzen_ai_yolov8_tutorial.ipynb
  - tutorials/onnx/accuracy_improvement/
  - docs/source/onnx/user_guide_config_description.rst
  - docs/source/onnx/appendix_full_quant_config_features.rst
---

# quark-onnx-quant-plan

## Purpose

Convert an ONNX `model_analysis.json` plus the user's intent into a confirmed `quant_plan.json`.
This skill makes the quantization decisions for the ONNX-to-ONNX flow — which preset, which
calibration method, which algorithm, which op types to include/exclude, whether to enable an NPU
target, whether to use external-data — **without** generating scripts or executing the
quantization. The plan is the contract between the user's intent and the execution step.

## Inputs

- `model_analysis.json` from `quark-onnx-model-intake` (architecture, op histogram, opsets,
  external-data state, `preset_candidates`, risks)
- `env_context.json` for accelerator-aware preset gating (CUDA major / ROCm major / NPU presence)
- `onnx_install_result.json` (optional) — gates BFP/MX/Extended presets (require the custom-ops
  library to be compiled for the target EP)
- User preferences: deployment target, accuracy goal, model-size goal, calibration data
  availability

## Outputs: quant_plan.json

Records the chosen preset (or custom config), calibration method, algorithm list, layer/op
overrides, NPU flag, external-data setting, and evaluation intent. Shares the schema with the
Torch plan; ONNX-specific fields live under `onnx_specific`.

Schema: [`quant_plan.schema.json`](https://github.com/amd/Quark/blob/1b229f781a1974cc742884e42d8eefc1eebb4f0a/.claude/skills-impl/shared/contracts/quant_plan.schema.json)

```json
{
  "model": {
    "model_type": "onnx",
    "architecture_guess": "cnn",
    "analysis_ref": "./model_analysis.json"
  },
  "backend": "onnx",
  "deployment_target": "npu_cnn",
  "preset": "XINT8",
  "calibration": {
    "method": "PowerOfTwo_MinMSE",
    "data_size": 200,
    "batch_size": 1,
    "use_external_data_format": false,
    "optimize_mem": false,
    "worker_num": 1
  },
  "algorithms": ["CLE"],
  "onnx_specific": {
    "enable_npu_cnn": true,
    "enable_npu_transformer": false,
    "include_cle": true,
    "include_fast_ft": false,
    "op_types_to_quantize": null,
    "nodes_to_quantize": null,
    "nodes_to_exclude": null,
    "subgraphs_to_exclude": [],
    "extra_options": {
      "OpTypesToExcludeOutputQuantization": []
    },
    "use_external_data_format": false,
    "execution_providers": ["CPUExecutionProvider"]
  },
  "evaluation_intent": "smoke",
  "requires_confirmation": false
}
```

## Available Presets (from `DefaultConfigMapping`)

Quark ONNX ships 50+ named presets in `quark/onnx/quantization/config/custom_config.py`. Pick the
**smallest viable set** for the user's architecture + deployment target; never list them all.

### AMD NPU CNN — Ryzen AI / VAI (`enable_npu_cnn=True`, PoF2 scales, NHWC)

| Preset | Description | Picks |
|--------|-------------|-------|
| `XINT8` | INT8 input + INT8 weight, optimized for NPU | Default for any CNN targeting NPU |
| `XINT8_ADAROUND` | XINT8 + AdaRound fast-finetune | When base XINT8 loses accuracy |
| `XINT8_ADAQUANT` | XINT8 + AdaQuant fast-finetune | When AdaRound is not enough |
| `VINT8` | INT8 optimized for VAIML | VAIML deployment |

### AMD NPU Transformer (`enable_npu_transformer=True`)

| Preset | Description | Picks |
|--------|-------------|-------|
| `INT16_TRANSFORMER_DEFAULT` | INT16 activations + INT8 weights, fast | Outlier-heavy activations |
| `INT16_TRANSFORMER_ACCURATE` | INT16 + accuracy algorithms | Largest accuracy headroom |

### General CPU / CUDA / ROCm INT8 (deployment-agnostic)

| Preset | Description | Picks |
|--------|-------------|-------|
| `A8W8` | INT8 sym activations + INT8 sym weights | Standard CPU/GPU INT8 |
| `A8W8_ADAROUND` / `A8W8_ADAQUANT` | + fast-finetune | Accuracy-tight A8W8 |
| `A16W8` | INT16 sym activations + INT8 sym weights | Wide-activation needs |
| `A16W8_ADAROUND` / `A16W8_ADAQUANT` | + fast-finetune | Accuracy-tight A16W8 |
| `S8S8_AAWS` / `U8S8_AAWS` / `U8U8_AAWA` / `S16S8_ASWS` / `U16S8_AAWS` (+ ADAROUND / ADAQUANT variants) | Various sym/asym INT8/INT16 combos | Fine-tune sym/asym choice per ORT target |
| `INT8_CNN_DEFAULT` / `INT8_CNN_ACCURATE` / `INT16_CNN_DEFAULT` / `INT16_CNN_ACCURATE` | CNN-tuned INT8/INT16 | CPU/GPU CNN deployment |

### Float Fallbacks

| Preset | Description | Picks |
|--------|-------------|-------|
| `FP16` / `FP16_ADAQUANT` | FP16 W+A | Accuracy-first when INT is too lossy |
| `BF16` / `BF16_ADAQUANT` | BFloat16 W+A | Same, with BF16 range |

### Block Formats (require Quark custom-ops library compiled for the target EP)

| Preset | Description | Picks |
|--------|-------------|-------|
| `BFP16` / `BFP16_ADAQUANT` | Block Floating Point 16-bit | AMD accelerator deployments |
| `MX4` / `MX6` / `MX9` (+ ADAQUANT) | Micro-Exponents BFP variants | Bit-budget exploration |
| `MXFP4E2M1` / `MXFP6E2M3` / `MXFP6E3M2` / `MXFP8E4M3` / `MXFP8E5M2` / `MXINT8` (+ ADAQUANT) | OCP MX formats | Modern AMD/NVIDIA accelerators |

### Mixed-Precision

| Preset | Description | Picks |
|--------|-------------|-------|
| `BF16_BFP16` / `BF16_MIXED_BFP16` / `BF16_MIXED_BFP16_ADAQUANT` | BF16 + BFP16 hybrid | High-accuracy + AMD HW |
| `BF16_MXINT8` / `BF16_MIXED_MXINT8` / `BF16_MIXED_MXINT8_ADAQUANT` | BF16 + MXInt8 hybrid | Same, OCP MX flavor |
| `MX9_INT8` | MX9 + INT8 hybrid | Bit-budget exploration |
| `S16S16_MIXED_S8S8` | INT16 + INT8 mixed | Outlier-aware INT mix |

## Available Calibration Methods

From `quark/onnx/calibration/methods.py` (+ ORT built-ins):

| Method | When to pick |
|--------|--------------|
| `MinMax` | Default for most CNN / weights-only; cheap, deterministic |
| `Entropy` | KL-divergence-based; helps when activations have long tails |
| `Percentile` | Clip outliers at a chosen percentile (default 99.999) |
| `Distribution` | Distribution-matching; useful for FP8 `p3`/`same` |
| `LayerWisePercentile` | Auto-picks per-tensor optimal percentile (MAE/MSE) — AMD-specific |
| `PowerOfTwo_NonOverflow` (a.k.a. `NonOverflow`) | **Required for AMD NPU XINT8** — picks the smallest PoF2 scale that doesn't overflow |
| `PowerOfTwo_MinMSE` (a.k.a. `MinMSE`) | **Required for AMD NPU XINT8** — picks the PoF2 scale that minimizes MSE; usually better than NonOverflow |
| `Int16Method.MinMax` | For INT16 configs |

NPU CNN / XINT8 targets **must** use a `PowerOfTwo*` method. Non-PoF2 scales are rejected at NPU
runtime — flag this as a hard constraint in the plan.

## Available Algorithms

From `quark/onnx/quantization/config/algorithm.py` and `examples/onnx/accuracy_improvement/`:

| Algorithm | Kind | Compatible presets | Description |
|-----------|------|-------------------|-------------|
| `CLE` (Cross-Layer Equalization) | Pre | INT8 CNN configs | Folds BN, equalizes per-channel scales across consecutive Conv/Linear layers (Nagel et al., 2019) |
| `BiasCorrection` | Post | INT8 CNN | Post-hoc bias adjustment (Nagel et al., 2019) |
| `AdaRound` | Post (fast-finetune) | XINT8 / A8W8 / A16W8 / block formats | Adaptive rounding optimization; needs cal data + LR + iterations; GPU-accelerated |
| `AdaQuant` | Post (fast-finetune) | Same as AdaRound | Layer-wise calibration tuning; usually after AdaRound is not enough |
| `AutoMixprecision` | Post | Block formats / mixed-precision | Auto-selects sensitivity-based per-layer dtype; can do dual BFP16+MX hybrid |

Algorithms compose: e.g. `CLE` (pre) + `AdaRound` (post) is a common XINT8 recipe. Combinations
beyond two algorithms are usually a red flag — flag them in `risks`.

## Decision Guide

Help the user choose based on their priorities and the architecture from
`model_analysis.json.model.onnx_specific.architecture_guess`:

| User intent | Architecture | Recommended starting plan |
|-------------|--------------|---------------------------|
| "Best accuracy, AMD GPU" | any | `BF16` or `BF16_MIXED_BFP16` |
| "INT8 CNN, CPU/GPU deployment" | cnn | `INT8_CNN_DEFAULT` or `A8W8`; add `CLE` if accuracy drops |
| "INT8 CNN → Ryzen AI NPU" | cnn | `XINT8` + `CLE`, calibration = `PowerOfTwo_MinMSE`; usually NHWC pre-conversion via `quark.onnx.tools.convert_nchw_to_nhwc` |
| "Ryzen AI NPU, accuracy-tight CNN" | cnn | `XINT8_ADAROUND` (then `XINT8_ADAQUANT` if still short) |
| "Block format MXFP4 / BFP16 experimentation" | any | `BFP16` or `MXFP4E2M1`; require `quark.onnx.operators.custom_ops` to be compiled |
| "Hybrid mixed-precision for best size/accuracy" | any | `BF16_MIXED_BFP16` or `S16S16_MIXED_S8S8` + `AutoMixprecision` |

## Deployment-Target Gating (HARD constraints)

| Target | Preset must satisfy | Calibration must be | Notes |
|--------|--------------------|--------------------|-------|
| `npu_cnn` (Ryzen AI CNN) | `enable_npu_cnn=True`, PoF2 symmetric INT8 per-tensor, NCHW→NHWC done | `PowerOfTwo_MinMSE` or `PowerOfTwo_NonOverflow` | Reject `A8W8` / `BFP16` / `MX*` / `FP16` if user requests `npu_cnn` |
| `npu_transformer` (Ryzen AI Transformer) | `enable_npu_transformer=True`, INT8/INT16 QDQ on MatMul/Gemm | `MinMax` / `Percentile` typically | Reject CNN-only presets |
| `cuda` | `onnxruntime-gpu` present, `CUDAExecutionProvider` available | Any | Block-format presets require custom-ops |
| `rocm` | `onnxruntime_rocm` (ROCm 6.x) **or** CPU ORT on ROCm 7.x (`tools/ci/install_onnxruntime.sh`) | Any | Custom-ops library must compile for ROCm |
| `cpu` | Any | Any | Block-format presets work via CPU custom-ops; expect speed cost |

If the deployment target conflicts with a requested preset, the plan must either (a) downgrade
to a viable preset and explain, or (b) leave it unset with a high-severity risk in `quant_plan.json`.

## Op-Type / Node Include/Exclude Levers

Three commonly-used knobs the plan should expose:

- **`op_types_to_quantize`** — restrict QDQ insertion to a subset, e.g. `["Conv"]` for
  CNN-only quantization.
- **`nodes_to_quantize` / `nodes_to_exclude`** — surgical per-node control by graph node name.
  Node names change after pre-processing (NCHW→NHWC, BN folding, etc.), so resolve them after
  any pre-processing pass.
- **`extra_options["OpTypesToExcludeOutputQuantization"]`** — keep certain op outputs in float
  while still quantizing their inputs/weights.

## Extra Options Commonly Set in `quant_plan.onnx_specific.extra_options`

From real examples in `examples/onnx/`:

| Option | Typical value | Source example |
|--------|---------------|----------------|
| `SimplifyModel` | `True` / `False` | toggle OnnxSlim pre-pass |
| `QuantizeFP16` | `True` | FP16-input models |
| `OpTypesToExcludeOutputQuantization` | `["Add", "Mul"]` etc. | keep selected op outputs in float |
| `FastFinetune` | `{"DataSize": 200, "BatchSize": 2, "NumIterations": 1000, "LearningRate": …, "OptimAlgorithm": "adaround"/"adaquant", "OptimDevice": "cuda:0"/"cpu", "InferDevice": "cuda:0"/"cpu", "EarlyStop": True}` | AdaRound / AdaQuant tutorials and Auto-Search tutorials |

## Decision Table (MUST show to user)

**ALWAYS present this table and WAIT for confirmation before finalizing.** Fill the "Value"
column from the user's request, the model analysis, and the deployment-target gates above.

| Decision | Value | Reason |
|----------|-------|--------|
| `backend` | `onnx` | Fixed for this skill |
| `deployment_target` | _(fill: `cpu` / `cuda` / `rocm` / `npu_cnn` / `npu_transformer`)_ | _(from env or user)_ |
| `preset` | _(fill: name from `DefaultConfigMapping` or `"custom"`)_ | _(why)_ |
| `calibration.method` | _(fill: `MinMax` / `Entropy` / `Percentile` / `Distribution` / `LayerWisePercentile` / `PowerOfTwo_MinMSE` / `PowerOfTwo_NonOverflow`)_ | _(why)_ |
| `calibration.data_size` | `200` (default) | _(why)_ |
| `calibration.batch_size` | `1`–`4` | _(why)_ |
| `algorithms` | _(fill: list e.g. `["CLE"]`, `["AdaRound"]`, `["AdaQuant"]`, `["BiasCorrection"]`)_ | _(why)_ |
| `onnx_specific.enable_npu_cnn` | _(fill: bool)_ | Hard-tied to `deployment_target == "npu_cnn"` |
| `onnx_specific.enable_npu_transformer` | _(fill: bool)_ | Hard-tied to `deployment_target == "npu_transformer"` |
| `onnx_specific.include_cle` | _(fill: bool)_ | CNN INT8 default `true` |
| `onnx_specific.op_types_to_quantize` | `null` or `["MatMul"]` etc. | _(why)_ |
| `onnx_specific.use_external_data_format` | `true` if model > 2 GB (from `model_analysis.json`) | Required for large models |
| `onnx_specific.extra_options` | _(fill: dict)_ | _(why — list each key)_ |
| `evaluation_intent` | `smoke` (default), `mlperf`, `mAP` | _(why)_ |

After showing the table, ask: **"Confirm this plan? Any changes?"**

**Do NOT proceed until the user confirms.**

## Per-Layer Overrides

For fine-grained control, individual layers can override the global config via `QLayerConfig`
(see `examples/onnx/yolo_quantization/quantize_yolo.py`):

```python
from quark.onnx import QConfig, QLayerConfig, XInt8Spec, CLEConfig

config = QConfig(
    global_config=QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()),
    algo_config=[CLEConfig()],
    EnableNPUCnn=True,
    exclude=[
        # YOLOX-style: keep a specific subgraph in float
        (["/_head/_modules_list.14/Transpose"], ["/_head/_modules_list.14/Concat_9"]),
    ],
)
```

Record any per-layer overrides under `onnx_specific.subgraphs_to_exclude` (list of
`(start_nodes, end_nodes)` tuples) or `onnx_specific.nodes_to_exclude` (flat name list).

## Rules

- **Keep scope to plan creation only.** Do not generate scripts, do not run quantization, do not
  export. Those are separate skills.
- **Require a model analysis first.** Without architecture, op histogram, opsets, and the
  already-QDQ flags, you cannot make informed preset / op-type recommendations. If
  `model_analysis.json` is missing, route back to `quark-onnx-model-intake`.
- **Honor hard deployment-target constraints.** NPU CNN ⇒ `XINT8` family + `PowerOfTwo*`
  calibration + NHWC. NPU Transformer ⇒ `INT8_TRANSFORMER_*` / `INT16_TRANSFORMER_*`. Reject
  conflicting preset requests with a clear explanation.
- **Custom-op preconditions for block formats.** Any preset in `{BFP16*, MX*, MXFP*, MXINT8*,
  BF16_*BFP*, BF16_*MXINT8*}` requires the Quark custom-ops library to have compiled. If
  `onnx_install_result.json` shows the compile failed, suppress these presets from
  recommendations and emit a risk pointing back to `quark-onnx-install`.
- **External-data flag tracks model size.** If `model_analysis.json.model.estimated_size_gb` > 2,
  set `onnx_specific.use_external_data_format = true` automatically. Otherwise default `false`.
- **Always present the decision table** before finalizing. The user must explicitly confirm.
- **Record risks.** If the user picks a risky combination (e.g. `XINT8` without PoF2 calibration,
  `MXFP4` on a model that hasn't been validated with `AutoMixprecision`), keep the user's choice
  but record the risk in the plan.
- **Do not duplicate fields between `model_analysis.json` and `quant_plan.json`.** The plan
  references the analysis via `analysis_ref`.

## Interaction Flow

1. **Check prerequisites**: Is `model_analysis.json` available? If not, route to
   `quark-onnx-model-intake` first. Is `onnx_install_result.json` available? If a block-format
   preset is on the table, require it.
2. **Confirm deployment target**: From `session_context.json.constraints.deployment_target` or
   ask. Apply the hard gates from the table above.
3. **Narrow presets**: Start from `model_analysis.json.quantization_targets.onnx_specific.preset_candidates`,
   filter by deployment target and custom-op availability, present the top 1–3 with rationale.
4. **Pick calibration method**: Default per target (MinMax for general; `PowerOfTwo_MinMSE` for
   NPU CNN; `Percentile` for outlier-heavy transformers; consider `LayerWisePercentile` when
   accuracy is critical). Decide `data_size` and `batch_size`.
5. **Pick algorithms**: Default per architecture and preset (CLE for INT8 CNN;
   AdaRound→AdaQuant for accuracy-tight CNN; BiasCorrection as a cheap post-hoc fixup).
   At most two algorithms unless justified.
6. **Set extra options**: From the table of common knobs, only set what's needed; document each.
7. **Present the decision table**: Show defaults, explain the tradeoffs, let the user adjust.
8. **Confirm**: Always required. Show the final plan summary before writing.
9. **Emit**: Write `quant_plan.json`. Surface any new constraints back to `the matching public quark-onnx skill` so
   they land in `session_context.json`.

## Recovery

- If `model_analysis.json` shows `has_qdq_already=true` or `has_quark_custom_ops=true` — do
  **not** produce a plan. Route back to `quark-onnx-model-intake` with a recommendation to
  remove existing QDQ first (`quark.onnx.tools.remove_qdq`).
- If the user picks a preset that conflicts with their deployment target — explain why it won't
  work (e.g. "`A8W8` uses non-PoF2 scales which the AMD NPU CNN runtime rejects"), suggest the
  closest viable alternative (e.g. `XINT8`), and respect the user's choice if they insist —
  recording the risk.
- If `model_analysis.json` shows architecture `unknown` — produce a draft plan with
  `requires_confirmation: true`, list the op histogram, and ask the user which of CNN /
  transformer / hybrid path to follow.
- If calibration data is unavailable — explain that PTQ requires representative inputs for
  accurate scale estimation, and ask the user to supply a small calibration set (≥ a few dozen
  samples) before the plan can be finalized.
- If a block-format preset is requested but custom-ops compile is unverified — hand off to
  `quark-onnx-install` to verify, then resume.
- If the user picks an unusual combination (e.g. `CLE` + transformer, three or more algorithms)
  — explain why it's atypical and suggest the standard recipe, but respect the user's choice
  with a recorded risk.
