---
name: quark-torch-result-validator
description: >-
  Validate Quark Torch quantization output (HuggingFace safetensors + config.json) using four
  lightweight checks: auxiliary file copy alignment, excluded tensor MD5 byte-identity,
  config.json deep comparison after stripping quantization keys, and safetensors header
  pattern/dtype summaries. Intended for post-export or file2file validation. Trigger for "validate
  quantization result", "check quantized model output", "verify exported weights". Not for .onnx
  output validation — use quark-onnx-result-validator.
---

# quark-torch-result-validator

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

## Purpose

Run four lightweight checks on a completed Quark export or file2file output. All checks read only
safetensors headers or small metadata files — no full tensor payload is loaded except for the bounded
MD5 spot-check. Results feed into a structured `validation_report.md`.

## Runtime Assumptions

All scripts (`quant_validation.py`, `run_validation.py`) live in the same directory as this `SKILL.md`,
under the installed skill directory.

Resolve `SKILL_DIR` from the loaded skill location before running any command:

```bash
# Set SKILL_DIR to the directory containing this installed SKILL.md.
```

`run_validation.py` writes JSON to **stdout**; `quant_validation.py` diagnostics go to **stderr**.
Never treat stderr as structured output.

## Contracts

- Input: `session_context.json`, `quant_plan.json` (for exclude rules and model paths)
- Output: `validation_report.md` (structured per the report template below)
- Schemas: `shared/contracts/validation_report.schema.json`

## Inputs

| Field | Source | Required |
|-------|--------|----------|
| `source_model_dir` | user or `session_context` | Yes |
| `quantized_model_dir` | user or `run_manifest` | Yes |
| `quant_config` | `quant_plan.json` or user-supplied JSON | For step 2 only |

## Outputs

`validation_report.md` with one section per executed step. Unexecuted steps are marked **skipped**.

## Interaction Flow

1. Confirm `SKILL_DIR`, `source_model_dir`, and `quantized_model_dir` are available.
2. Run self-test to verify scripts are intact: `python3 "$SKILL_DIR/run_validation.py" self-test`
3. Execute steps in cheap-to-expensive order (4 → 1 → 3 → 2).
4. Collect JSON from stdout for each step; write `validation_report.md`.
5. Surface any `ok: false` steps with their `errors` and `mismatches`.

## Four Validation Steps

| Order | Function | CLI subcommand | Purpose |
|-------|----------|----------------|---------|
| 1 | `check_auxiliary_files_copied` | `auxiliary` | Compare non-weight auxiliary files between source and quantized dirs |
| 2 | `check_non_quantized_tensors_md5_unchanged` | `md5` | MD5 spot-check payload bytes of exclude-listed tensors |
| 3 | `check_config_json_equal_except_quantization` | `config` | Deep compare `config.json` after stripping quantization keys |
| 4 | `get_fuzzy_tensor_names` | `fuzzy` | Summarize canonical header patterns and `dtype_counts` |

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

# 4. get_fuzzy_tensor_names (cheapest — header only)
python3 "$SKILL_DIR/run_validation.py" fuzzy \
  --model-path ./quantized-model

# 1. check_auxiliary_files_copied
python3 "$SKILL_DIR/run_validation.py" auxiliary \
  --source-model-dir ./original-model --quantized-model-dir ./quantized-model \
  --ignore 'README*'

# 3. check_config_json_equal_except_quantization
python3 "$SKILL_DIR/run_validation.py" config \
  --original-config ./original-model/config.json \
  --quantized-config ./quantized-model/config.json

# 2. check_non_quantized_tensors_md5_unchanged (most expensive)
python3 "$SKILL_DIR/run_validation.py" md5 \
  --source-model-dir ./original-model --output-model-dir ./quantized-model \
  --quant-config '{"exclude":["*.lm_head.weight"],"max_samples":50}'
```

For `md5`, `--quant-config` accepts a JSON string or a path to a JSON file.
If `SKILL_DIR` or model paths cannot be resolved, mark the affected step **skipped**.

## Public API (for direct Python import)

```python
from quant_validation import (
    check_auxiliary_files_copied,
    check_non_quantized_tensors_md5_unchanged,
    check_config_json_equal_except_quantization,
    get_fuzzy_tensor_names,
)
```

All four functions are in `__all__`. Other public-named helpers are internal utility surface.

## Recovery

| Failure | Recovery |
|---------|----------|
| Self-test exits non-zero | Report script integrity failure; do not run further steps |
| `source_model_dir` missing | Mark steps 1, 2, 3 as skipped; run step 4 on quantized dir only |
| No safetensors files found | Mark steps 2 and 4 as skipped |
| `quant_config` missing exclude rules | Mark step 2 as skipped |
| File too large for SHA256 | Recorded in `hash_skipped_files`; size comparison still runs |

## Report Template

```text
## Validation Report — quark-torch-result-validator

**Step 4 — fuzzy tensor names**: ok / FAIL / skipped
  - patterns: <count>, mixed-dtype patterns: <count>
  - Notable: <pattern> → <dtype_counts>

**Step 1 — auxiliary files**: ok / FAIL / skipped
  - missing: <count>, mismatched: <count>, extra: <count>

**Step 3 — config.json**: ok / FAIL / skipped
  - mismatches: <count>
  - <key>: <source_value> / <quantized_value>

**Step 2 — MD5 spot-check**: ok / FAIL / skipped
  - candidates: <count>, checked: <count>, sampled: true/false
  - mismatches: <count>
```

## Canonical Name Rules

- Replace only **pure numeric path segments** with `*`: `layers.12` → `layers.*`
- Do not alter digits embedded in non-numeric names: `norm1`, `w2` stay unchanged
- `dtype_counts` is aggregated per pattern; multiple dtypes in one pattern signals mixed precision risk

## Optional Dtype Hints

Heuristics only — not mandatory pass/fail rules:

- FP8: dtypes such as `F8_E4M3`
- MXFP4: often `uint8`-packed
- INT schemes: depend on export naming convention
