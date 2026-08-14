---
name: quark-onnx-result-validator
description: >-
  Validate Quark ONNX quantization output using four lightweight checks: auxiliary file copy
  alignment, expected non-quantized initializer MD5 byte-identity (inline `raw_data` +
  external-data byte ranges), model metadata equality after stripping quantization-only opset
  entries / Quark domains, and fuzzy node-pattern + op-type + dtype summaries with QDQ /
  `com.amd.quark` custom-op presence. Intended for post-quantization inspection of `model.onnx`
  (with or without `model.onnx_data`). Trigger for "validate ONNX quantization result", "check
  quantized .onnx output", "verify ONNX initializers", "did QDQ insertion happen", "are the
  non-quantized weights byte-identical".
---

# quark-onnx-result-validator

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

## Purpose

Run four lightweight checks on a completed Quark ONNX quantization output. Reads only ONNX graph
headers (`onnx.load(..., load_external_data=False)`), initializer metadata, and small auxiliary
files. Raw payload bytes are only touched for the bounded MD5 spot-check (and only for tensors
matched by the user's exclude rules). Results feed a structured `validation_report.md`.

## Runtime Assumptions

All scripts (`quant_validation_onnx.py`, `run_validation.py`) live in the same directory as this
`SKILL.md`, under the installed skill directory.

Resolve `SKILL_DIR` from the loaded skill location before running any command:

```bash
# Set SKILL_DIR to the directory containing this installed SKILL.md.
```

`run_validation.py` writes JSON to **stdout**; `quant_validation_onnx.py` diagnostics go to
**stderr** with the prefix `[quant-validation-onnx][tag]`. Never treat stderr as structured output.

Requires the `onnx` Python package in the supported range (`onnx>=1.21.0,<=1.22.0`, per
`requirements.txt`).

## Contracts

- Input: `session_context.json`, `quant_plan.json` (for exclude / op-type filters and model paths)
- Output: `validation_report.md`
- Schemas: `shared/contracts/validation_report.schema.json`

## Inputs

| Field | Source | Required |
|-------|--------|----------|
| `source_model_path` | user or `session_context` | Step 2 + Step 3 |
| `quantized_model_path` | user or `run_manifest` | All steps |
| `source_model_dir` | parent dir of source model (or `session_context`) | Step 1 |
| `quantized_model_dir` | parent dir of quantized model (or `run_manifest`) | Step 1 |
| `quant_config` | `quant_plan.json` or user-supplied JSON | Step 2 only |

`quant_config` for step 2 supports the following keys (all optional unless noted):

| Key | Purpose | Default |
|-----|---------|---------|
| `exclude` | Glob list of initializer names expected to remain unchanged | — |
| `exclude_initializers` | Alias of `exclude` | — |
| `nodes_to_exclude` | Node names whose initializer inputs should remain unchanged | — |
| `op_types_to_quantize` | When set, any initializer **not** wired into one of these op types becomes an implicit exclude | — |
| `max_samples` | Random spot-check cap for large models | `200` |
| `random_seed` / `seed` | Deterministic sampling seed | `None` |

At least one of `exclude` / `exclude_initializers` / `nodes_to_exclude` / `op_types_to_quantize`
**must** be provided; otherwise step 2 is marked skipped.

## Outputs

`validation_report.md` with one section per executed step. Unexecuted steps are marked **skipped**.

## Interaction Flow

1. Confirm `SKILL_DIR`, `source_model_path` (if available), and `quantized_model_path` are
   resolvable.
2. Run the self-test to verify scripts are intact:
   `python3 "$SKILL_DIR/run_validation.py" self-test`
3. Execute steps in cheap-to-expensive order (**4 → 1 → 3 → 2**).
4. Collect JSON from stdout for each step; write `validation_report.md`.
5. Surface any `ok: false` steps with their `errors` / `mismatches`.

## Four Validation Steps

| Order | Function | CLI subcommand | Purpose |
|-------|----------|----------------|---------|
| 1 | `check_auxiliary_files_copied` | `auxiliary` | Compare non-`.onnx`/non-`.onnx_data` auxiliary files between source and quantized directories |
| 2 | `check_non_quantized_initializers_md5_unchanged` | `md5` | MD5 spot-check initializer payload bytes (inline `raw_data` or external-data byte ranges) for tensors expected to remain non-quantized |
| 3 | `check_model_metadata_equal_except_quantization` | `metadata` | Compare IR version, producer, default-domain opset, and graph input/output signatures after stripping Quark-injected custom-op domains |
| 4 | `get_fuzzy_node_op_summary` | `fuzzy` | Header-only summary: op-type histogram, canonical node-name patterns, initializer dtype counts per pattern, QDQ / `com.amd.quark` custom-op presence |

Run in cost order: **4 → 1 → 3 → 2**.

## Agent Execution Contract

### Self-Test

```bash
# Set SKILL_DIR to the directory containing this installed SKILL.md.
python3 "$SKILL_DIR/run_validation.py" self-test
```

Exits `0` and prints `exported symbols self-test (__all__): ok` on success.

### CLI Commands

```bash
# Set SKILL_DIR to the directory containing this installed SKILL.md.

# 4. get_fuzzy_node_op_summary (cheapest — header only)
python3 "$SKILL_DIR/run_validation.py" fuzzy \
  --model-path ./quantized/model.onnx

# 1. check_auxiliary_files_copied
python3 "$SKILL_DIR/run_validation.py" auxiliary \
  --source-model-dir ./source-dir --quantized-model-dir ./quantized-dir \
  --ignore 'README*'

# 3. check_model_metadata_equal_except_quantization
python3 "$SKILL_DIR/run_validation.py" metadata \
  --source-model-path ./source/model.onnx \
  --quantized-model-path ./quantized/model.onnx

# 2. check_non_quantized_initializers_md5_unchanged (most expensive)
python3 "$SKILL_DIR/run_validation.py" md5 \
  --source-model-path ./source/model.onnx \
  --output-model-path ./quantized/model.onnx \
  --quant-config '{"exclude":["*.bias","embeddings.*.weight"],"max_samples":50}'
```

For `md5`, `--quant-config` accepts a JSON string or a path to a JSON file.
If `SKILL_DIR` or model paths cannot be resolved, mark the affected step **skipped**.

## Public API (for direct Python import)

```python
from quant_validation_onnx import (
    check_auxiliary_files_copied,
    check_non_quantized_initializers_md5_unchanged,
    check_model_metadata_equal_except_quantization,
    get_fuzzy_node_op_summary,
)
```

All four functions are in `__all__`. Other public-named helpers are internal utility surface.

## Recovery

| Failure | Recovery |
|---------|----------|
| Self-test exits non-zero | Report script integrity failure; do not run further steps |
| `onnx` import fails | Hand off to `quark-onnx-install`; do not run any step |
| `source_model_path` missing | Mark steps 2, 3 as skipped; run steps 1, 4 if quantized path is available |
| `quant_config` missing exclude rules | Mark step 2 as skipped |
| External-data file missing alongside `.onnx` | Recorded under `external_data_missing`; affected tensors marked `read_error` in step 2; step 4 still runs against the graph proto |
| Quantized model has zero QDQ / Quark-custom nodes | Step 4 emits a high-severity warning (`quantization_did_not_run`) |

## Report Template

```text
## Validation Report — quark-onnx-result-validator

**Step 4 — fuzzy node / op summary**: ok / FAIL / skipped
  - op types: <count>, QDQ nodes: <count>, com.amd.quark nodes: <count>
  - Notable: <pattern> → <op_type_counts>
  - Quantization signal: present / **MISSING** / partial

**Step 1 — auxiliary files**: ok / FAIL / skipped
  - missing: <count>, mismatched: <count>, extra: <count>

**Step 3 — model metadata**: ok / FAIL / skipped
  - ir_version: <source> / <quantized>
  - opset_import (default domain): <source> / <quantized>
  - input/output signature diffs: <count>
  - Quark-injected opset domains: <list>

**Step 2 — MD5 spot-check (initializers)**: ok / FAIL / skipped
  - candidates: <count>, checked: <count>, sampled: true/false
  - mismatches: <count>
  - external_data_missing: <count>
```

## Canonical Name Rules

- Replace only **pure numeric path segments** with `*`: `Conv_12` → `Conv_*`, `model.layer.3.Conv` →
  `model.layer.*.Conv`
- Do not alter digits embedded in non-numeric names: `Conv1`, `MatMul_w2` stay unchanged
- `op_type_counts` and `dtype_counts` are aggregated per pattern; multiple op types / dtypes in one
  pattern signals partial or mixed-precision quantization

## Optional Dtype Hints

Heuristics only — not mandatory pass/fail rules:

- **INT8 QDQ**: initializers with `INT8` / `UINT8`; nodes `QuantizeLinear` / `DequantizeLinear`
- **INT4 MatMulNBits**: initializers with `UINT8` packed-low-nibble shape; `MatMulNBits` op
- **BFP16 / MX / MXFP**: `com.amd.quark` opset domain present; nodes
  `BFPQuantizeDequantize` / `MXQuantizeDequantize` / `ExtendedQuantizeLinear`
- **FP16 / BF16 keep**: initializers with `FLOAT16` / `BFLOAT16` and no QDQ neighbors

## ONNX-vs-Torch Behavioural Notes

- Step 2 walks **initializers** (the ONNX analog of safetensors tensors), not safetensors entries.
  Inline tensors are read via `tensor.raw_data`; external-data tensors are read by `(offset,
  length)` from the external-data file declared in `tensor.external_data`.
- Step 3 compares model-level metadata (IR / producer / opset) and graph I/O signature. ONNX has
  no `config.json` equivalent — opset comparison strips Quark's custom domains before equality so
  the only allowed diff is the addition of `com.amd.quark` (or similar) on the quantized side.
- Step 4 is structurally similar to the Torch fuzzy summary: it groups nodes / initializers by
  canonical name pattern and reports `op_type_counts` / `dtype_counts`. Additionally surfaces
  whether quantization actually ran (`QuantizeLinear` / `com.amd.quark` presence).
