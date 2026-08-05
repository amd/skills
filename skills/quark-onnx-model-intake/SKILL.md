---
name: quark-onnx-model-intake
description: >-
  Inspect a target ONNX model and prepare metadata for Quark ONNX PTQ planning. Use for `.onnx`
  path validation, opset / IR version detection, input-output shape and dtype discovery, op-type
  histogram, quantizable-op counting, deployment-target compatibility checks (CPU / CUDA / ROCm /
  AMD NPU CNN / AMD NPU Transformer), and risk assessment. Trigger for "analyze my ONNX model",
  "check this onnx model", "what opset is this", "can Quark quantize this .onnx", "is my model
  NPU-compatible", "does my model already have QDQ", "is my model larger than 2 GB", or before any
  ONNX quantization step that needs model facts that are missing.
---

# quark-onnx-model-intake

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

When the workflow names one of these internal procedures, read its bundled reference before carrying out that step:

- [`quark-workspace-validate`](references/quark-workspace-validate.md)

## Purpose

Validate the target `.onnx` model and extract the structural facts that an ONNX quant plan needs
to make correct quantization decisions. This step exists because different ONNX models have very
different quantization requirements — a CNN destined for AMD NPU CNN needs NHWC layout and PoF2
scales, an LLM exported to ONNX needs MatMulNBits or SmoothQuant + GPTQ, a >2 GB model needs
external-data handling, and a model that already contains QDQ nodes (or Quark's custom-op domains)
must not be silently re-quantized.

## Inputs

- Model path — a single `.onnx` file, or a directory containing `model.onnx` plus its
  `model.onnx_data` / external-weight blobs
- `env_context.json` for Python and accelerator constraints (CUDA major / ROCm major / NPU
  presence)
- `workspace_context.json` for the validated model path
- `onnx_install_result.json` (optional) — the installed `onnxruntime*` variant determines which
  execution providers are realistic deployment targets

## Outputs: model_analysis.json

Captures the model's graph metadata, op-type coverage, quantizable-op count, deployment-target
compatibility, and risks. Schema is shared with the Torch intake; ONNX-specific fields live under
`model.onnx_specific` and `quantization_targets.onnx_specific`.

Schema: [`model_analysis.schema.json`](../../shared/contracts/model_analysis.schema.json)

```json
{
  "analysis_status": "complete",
  "model": {
    "model_path": "/data/models/resnet50/model.onnx",
    "model_type": "onnx",
    "loading_class": "onnx.load",
    "estimated_size_gb": 0.10,
    "onnx_specific": {
      "ir_version": 8,
      "producer_name": "pytorch",
      "producer_version": "2.5.1",
      "opsets": [{"domain": "ai.onnx", "version": 17}],
      "uses_external_data": false,
      "external_data_files": [],
      "input_layout_guess": "NCHW",
      "inputs": [
        {"name": "input", "dtype": "float32", "shape": [1, 3, 224, 224]}
      ],
      "outputs": [
        {"name": "output", "dtype": "float32", "shape": [1, 1000]}
      ],
      "has_dynamic_dims": false,
      "has_qdq_already": false,
      "has_quark_custom_ops": false,
      "architecture_guess": "cnn"
    }
  },
  "quantization_targets": {
    "linear_layer_count": 53,
    "onnx_specific": {
      "op_counts": {"Conv": 53, "BatchNormalization": 53, "Relu": 49, "MaxPool": 1, "GlobalAveragePool": 1, "Gemm": 1},
      "quantizable_op_counts": {"Conv": 53, "Gemm": 1, "MatMul": 0, "Add": 16, "Mul": 0},
      "preset_candidates": ["XINT8", "A8W8", "INT8_CNN_DEFAULT"],
      "exclude_defaults": []
    }
  },
  "risks": [
    {
      "severity": "low",
      "message": "Inputs are NCHW. AMD NPU CNN target expects NHWC.",
      "recovery_hint": "Run quark.onnx.tools.convert_nchw_to_nhwc before quantization."
    }
  ]
}
```

## Supported Domains and Preset Families

Quark ONNX ships a large `DefaultConfigMapping` of named presets in
`quark/onnx/quantization/config/custom_config.py`. The intake should narrow the candidate set
based on what the model actually looks like:

| Category | Representative presets | When to suggest |
|----------|------------------------|-----------------|
| **AMD NPU CNN (XINT8 family)** | `XINT8`, `XINT8_ADAROUND`, `XINT8_ADAQUANT`, `XINT8_WEIGHTSONLY_ADAROUND` | Model is Conv-heavy (CNN), inputs are 4D, deployment target is `npu_cnn`; requires PoF2 scales and (usually) NHWC layout |
| **AMD NPU Transformer (INT16 transformer)** | `INT16_TRANSFORMER_DEFAULT`, `INT16_TRANSFORMER_ACCURATE` | Model is MatMul/Gemm-heavy, deployment target is `npu_transformer` |
| **General CPU/GPU INT8** | `A8W8`, `A8W8_ADAROUND`, `A8W8_ADAQUANT`, `INT8_CNN_DEFAULT`, `INT8_CNN_ACCURATE` | Generic INT8 deployment on CPU/CUDA/ROCm |
| **A16W8 / wide activations** | `A16W8`, `A16W8_ADAROUND`, `A16W8_ADAQUANT` | Activations need 16-bit headroom (outlier-heavy models) |
| **Block formats (custom-ops required)** | `BFP16`, `BFP16_ADAQUANT`, `MX4`, `MX6`, `MX9`, `MXFP4`, `MXFP6`, `MXFP8`, `BF16_MIXED_BFP16`, `BF16_MIXED_MXINT8` | Modern AMD accelerators with `quark.onnx.operators.custom_ops` available |
| **Float fallbacks** | `FP16`, `BF16`, `FP16_ADAQUANT`, `BF16_ADAQUANT` | Accuracy-first; no INT quant |

`preset_candidates` in `model_analysis.json` should list the 1–3 presets that are realistic given
architecture + deployment target; never list all of them.

## What to Extract

### From the ONNX model proto (without loading weights)

- **IR version**: `model.ir_version`
- **Producer**: `model.producer_name`, `model.producer_version`
- **Opsets**: `[(opset.domain or 'ai.onnx', opset.version) for opset in model.opset_import]` —
  for QDQ insertion, `ai.onnx` opset must be ≥ 13 (≥ 19 recommended for full INT4/FP8 support)
- **Custom domains**: presence of `com.amd.quark` or `com.microsoft` indicates the model has
  already been processed by Quark or ORT and should not be naively re-quantized
- **External data**: `os.path.exists(model_path + '_data')` or any tensor with
  `tensor.data_location == TensorProto.EXTERNAL`; total file size > 2 GB without external data is
  invalid
- **Inputs / outputs**: name, dtype (from `type.tensor_type.elem_type`), shape (dim_value or
  symbolic `dim_param`)
- **Dynamic dims**: any `dim_param` or `dim_value == 0` in inputs → flag as risk for NPU targets
- **Input layout guess**: 4D float input with channel dim == 3 or small power-of-2 → likely NCHW
  (channel-first) or NHWC (channel-last); use the convention of the producer
- **Architecture guess**: histogram of op types — Conv-dominated → CNN; MatMul/Gemm + softmax →
  Transformer; LSTM / GRU / RNN → recurrent; mix of Conv + MatMul + Attention → hybrid

### Quantization Targets

- **Quantizable op counts**: count `Conv`, `Gemm`, `MatMul`, `ConvTranspose`, `Add`, `Mul`,
  `BatchNormalization` (will be folded into Conv during pre-processing)
- **`linear_layer_count`**: sum of `MatMul` + `Gemm` + `Conv` (the canonical "linear ops" in
  ONNX)
- **`exclude_defaults`**: usually empty for ONNX (Quark doesn't have a `lm_head`-equivalent
  default), but flag the final classifier `Gemm`/`MatMul` if accuracy is at risk
- **Already-quantized check**: presence of `QuantizeLinear` / `DequantizeLinear` /
  `BFPQuantizeDequantize` / `MXQuantizeDequantize` / `ExtendedQuantizeLinear` → set
  `model.onnx_specific.has_qdq_already = true` or `has_quark_custom_ops = true`

### Deployment-Target Compatibility

Cross-reference op types against the chosen target:

- **NPU CNN** (`enable_npu_cnn=True`, XINT8): expects NHWC, PoF2 scales, per-tensor symmetric
  INT8, no MatMul-based attention. Flag NCHW inputs, dynamic dims, MatMul-heavy graphs.
- **NPU Transformer** (`enable_npu_transformer=True`): QDQ on Gemm/MatMul only; flag heavy use
  of unsupported ops (e.g. custom attention kernels).
- **CPU/CUDA/ROCm general**: largely permissive; flag custom-domain ops that require
  matching custom-ops library at inference.
- **Block formats (BFP16 / MX / MXFP*)**: require Quark's custom-ops library compiled for the
  target EP. If `onnx_install_result.json` shows the custom-ops compile failed, suppress these
  presets from `preset_candidates` and emit a risk.

### Calibration Data Considerations

Surface what the user must prepare separately:

- Input names + dtypes + shapes — the `CalibrationDataReader` must yield batches keyed by these
  names with these dtypes
- Suggested sample count: 64–512 for CV; 128–1024 for LLM; reduce if `optimize_mem=True` is
  needed
- Flag dynamic batch / seq dims that the data reader must concretize

### Risks

Flag anything that could cause failures downstream:

- Model file > 2 GB **without** external data → ProtoBuf serialization will fail
- Opset < 13 → QDQ insertion may fail; suggest `onnx.version_converter.convert_version(model, 13+)`
- Dynamic dims in input shape → NPU targets and many post-processing steps don't tolerate
  unconcretized dims; recommend `quark.onnx.tools.fix_shapes`
- NCHW inputs for an NPU CNN target → recommend `quark.onnx.tools.convert_nchw_to_nhwc`
- FP16-only model → some calibrators expect FP32; recommend
  `quark.onnx.tools.convert_fp16_to_fp32` first
- Model already contains QDQ or Quark custom ops → do not re-quantize; suggest
  `quark.onnx.tools.remove_qdq` or treat as a no-op
- Custom-domain ops present (`com.microsoft`, third-party) → flag, may require
  `op_types_to_quantize` curation
- Unusual loading requirements (encrypted / crypto mode) → require the user to pass an
  `onnx.ModelProto` rather than a path (see `input_check.py:137`)

## Concrete Actions

### Action 1: Read model metadata (NEVER load external weights)

```bash
python3 - <<'PY'
import json, os, sys, collections
import onnx
from onnx import TensorProto

MODEL_PATH = "<MODEL_PATH>"

# load_external_data=False so we never pull the >GB tensors into memory
model = onnx.load(MODEL_PATH, load_external_data=False)

# basic metadata
opsets = [{"domain": o.domain or "ai.onnx", "version": o.version} for o in model.opset_import]
producer = {"name": model.producer_name, "version": model.producer_version, "ir_version": model.ir_version}

def fmt_shape(t):
    return [(d.dim_value if d.dim_value else (d.dim_param or "?")) for d in t.type.tensor_type.shape.dim]

dtype_map = {v: k for k, v in TensorProto.DataType.items()}
def fmt_dtype(t):
    return dtype_map.get(t.type.tensor_type.elem_type, "UNKNOWN")

inputs = [{"name": i.name, "dtype": fmt_dtype(i), "shape": fmt_shape(i)} for i in model.graph.input]
outputs = [{"name": o.name, "dtype": fmt_dtype(o), "shape": fmt_shape(o)} for o in model.graph.output]

# op-type histogram
op_counts = collections.Counter(n.op_type for n in model.graph.node)

# external data check
file_size_bytes = os.path.getsize(MODEL_PATH)
uses_external = any(init.data_location == TensorProto.EXTERNAL for init in model.graph.initializer)

# already-quantized check
qdq_ops = {"QuantizeLinear", "DequantizeLinear"}
quark_ops = {"BFPQuantizeDequantize", "MXQuantizeDequantize",
             "ExtendedQuantizeLinear", "ExtendedDequantizeLinear"}
has_qdq = any(n.op_type in qdq_ops for n in model.graph.node)
has_quark_custom = any(n.op_type in quark_ops for n in model.graph.node)

# dynamic dim check
has_dynamic = any(isinstance(d, str) or d == "?" for inp in inputs for d in inp["shape"])

# architecture guess (very rough)
arch = "unknown"
if op_counts.get("Conv", 0) > 5 * op_counts.get("MatMul", 0):
    arch = "cnn"
elif op_counts.get("MatMul", 0) + op_counts.get("Gemm", 0) > 5 * op_counts.get("Conv", 0):
    arch = "transformer"
elif {"LSTM", "GRU", "RNN"} & set(op_counts):
    arch = "recurrent"
elif op_counts.get("Conv", 0) and op_counts.get("MatMul", 0):
    arch = "hybrid"

info = {
    "ir_version": producer["ir_version"],
    "producer_name": producer["name"],
    "producer_version": producer["version"],
    "opsets": opsets,
    "file_size_bytes": file_size_bytes,
    "uses_external_data": uses_external,
    "inputs": inputs,
    "outputs": outputs,
    "has_dynamic_dims": has_dynamic,
    "has_qdq_already": has_qdq,
    "has_quark_custom_ops": has_quark_custom,
    "architecture_guess": arch,
    "op_counts": dict(op_counts.most_common(30)),
    "quantizable_op_counts": {k: op_counts.get(k, 0) for k in ("Conv", "Gemm", "MatMul", "ConvTranspose", "Add", "Mul", "BatchNormalization")},
}
print(json.dumps(info, indent=2))
PY
```

### Action 2: Optional — run schema and shape-inference checks

```bash
python3 - <<'PY'
import onnx
MODEL_PATH = "<MODEL_PATH>"
model = onnx.load(MODEL_PATH, load_external_data=False)
onnx.checker.check_model(model, full_check=False)   # raises on schema errors
inferred = onnx.shape_inference.infer_shapes(model)  # may reveal shape issues
print("checker: OK, shapes inferred")
PY
```

### Action 3: Check model size on disk (catches the 2 GB ProtoBuf limit early)

```bash
du -sh /path/to/model.onnx
ls -lh /path/to/model.onnx*       # picks up model.onnx_data
```

### Action 4: Present summary table to user

```text
ONNX Model Analysis:
  Model path:        <path>
  File size:         <human-readable>  external-data: Yes/No
  IR / Opset:        ir=<n>, ai.onnx=<n>  (custom domains: <list or "None">)
  Architecture:      <cnn | transformer | recurrent | hybrid | unknown>
  Inputs:            <name>: <dtype> <shape>
  Outputs:           <name>: <dtype> <shape>
  Quantizable ops:   Conv=<n>, MatMul=<n>, Gemm=<n>  (linear_layer_count=<sum>)
  Already QDQ:       Yes/No   Already Quark custom ops: Yes/No
  Preset candidates: [<top 1–3 from DefaultConfigMapping>]
  Risks:             <list or "None">
```

## Rules

- **Run or reference `quark-workspace-validate` first** to confirm `.onnx` and adjacent
  `.onnx_data` paths are valid before attempting to parse the model.
- **Do not load external data.** Pass `load_external_data=False` to `onnx.load`. The intake reads
  graph metadata only; loading multi-GB tensor blobs is the quantizer's job.
- **Refuse to silently re-quantize.** If `has_qdq_already` or `has_quark_custom_ops` is true,
  emit a high-severity risk and require the user to confirm intent (likely they want
  `quark.onnx.tools.remove_qdq` first, or they meant to quantize a different file).
- **Preserve ambiguity** when a model reference could be a single `.onnx` or a directory of
  external-data shards. Note both possibilities and let the user resolve.
- **Filter `preset_candidates` by deployment target and custom-op availability.** Never suggest
  BFP/MX presets when `onnx_install_result.json` shows the custom-ops library failed to compile.
  Never suggest XINT8 outside an NPU CNN target.
- **Surface new environment constraints** (e.g. model requires opset upgrade, or NHWC conversion
  before NPU CNN) by recording them in `model_analysis.json` under `risks` and asking
  `the matching public quark-onnx skill` to add them to `session_context.json`'s `open_questions`. Do not write
  directly to `env_context.json`.
- **Never call `quantize_static` / `ModelQuantizer.quantize_model` here.** Intake is read-only.

## Interaction Flow

1. **Confirm model source**: Is the path a single `.onnx` or a directory? Are there
   `.onnx_data` shards present? Check if `quark-workspace-validate` already confirmed the path.
2. **Read metadata**: Run Action 1 (and optionally Action 2). Capture opsets, inputs, outputs,
   op counts, external-data state, already-QDQ flags.
3. **Classify architecture**: CNN / transformer / recurrent / hybrid based on op histogram.
4. **Cross-reference deployment target**: From `session_context.json.constraints.deployment_target`
   (or ask if missing), narrow `preset_candidates` to viable options.
5. **Identify risks**: NCHW vs NHWC, dynamic dims, > 2 GB without external data, opset too low,
   already-quantized, FP16-only, custom-domain ops, missing custom-ops compile for BFP/MX.
6. **Present summary table** to the user.
7. **Emit**: Write `model_analysis.json`. Surface any new constraints back to `the matching public quark-onnx skill`
   so they land in `session_context.json`.

## Recovery

- If `analysis_status: "partial"` — some facts were extracted but the model could not be fully
  inspected. Common causes: external-data files missing, ProtoBuf > 2 GB without external data,
  or `onnx.checker` raised a schema error.
- If `has_qdq_already == true` or `has_quark_custom_ops == true` — do **not** proceed to PTQ
  planning. Recommend `python -m quark.onnx.tools.remove_qdq` (or equivalent) and re-run intake
  on the cleaned model.
- If `opsets[ai.onnx] < 13` — recommend
  `onnx.version_converter.convert_version(model, 13)` (or 19+ for INT4/FP8) and re-run intake.
- If `has_dynamic_dims == true` for an NPU target — recommend
  `python -m quark.onnx.tools.fix_shapes --input model.onnx --output model_fixed.onnx
   --input_shape "input:1,3,224,224"` and re-run intake.
- If `input_layout_guess == "NCHW"` and target is `npu_cnn` — recommend
  `python -m quark.onnx.tools.convert_nchw_to_nhwc` and re-run intake.
- If the model is FP16-only and the chosen calibrator needs FP32 — recommend
  `python -m quark.onnx.tools.convert_fp16_to_fp32` and re-run intake.
- If the architecture is "unknown" — present the op histogram to the user and ask which target
  they intend; do not guess a preset.
