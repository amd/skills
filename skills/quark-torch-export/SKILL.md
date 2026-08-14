---
name: quark-torch-export
description: >-
  Prepare export and downstream evaluation handoff for a planned or completed Quark Torch PTQ run.
  Input is a PyTorch / HuggingFace transformers model; output formats include HF safetensors,
  GGUF, and ONNX. Trigger for "export model", "save quantized model", "convert to GGUF", "export
  to HuggingFace format", "export to ONNX", or when the user has a completed or planned Torch PTQ
  run and needs deployment outputs. Not for exporting from an .onnx input model — use
  quark-onnx-ptq.
---

# quark-torch-export

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

When the workflow names one of these internal procedures, read its bundled reference before carrying out that step:

- [`quark-torch-quant-plan`](references/quark-torch-quant-plan.md)

## Purpose

Translate a confirmed quantization plan into export expectations and downstream evaluation requirements. Export is a post-quantization step — the model must be quantized first (or have a plan to be quantized) before export decisions make sense. This skill ensures the right export format is chosen and evaluation is properly configured.

## Inputs

- `quant_plan.json` from quark-torch-quant-plan
- `workspace_context.json` for the output directory
- `run_manifest.yaml` (optional, existing manifest to extend)

## Outputs: run_manifest.yaml

Export does not own this artifact — it updates the workflow's manifest with export and evaluation config.

Schema: [`run_manifest.schema.json`](https://github.com/amd/Quark/blob/1b229f781a1974cc742884e42d8eefc1eebb4f0a/.claude/skills-impl/shared/contracts/run_manifest.schema.json)

(Export updates the workflow's `run_manifest.yaml` with export and evaluation fields rather than producing a new artifact.)

```yaml
export:
  formats:
    - hf_format
  output_dir: ./output/qwen3-8b-fp8
  weight_format: real_quantized
  custom_mode: quark

evaluation:
  skip: false
  metrics:
    - ppl
  dataset: wikitext
  tasks: null
  batch_size: auto
```

## Supported Export Formats

### 1. HuggingFace SafeTensors (`hf_format`) — Default

- Produces: `config.json` + `*.safetensors` files with `quantization_config` metadata
- Compatible with: HuggingFace transformers loading, vLLM, TGI
- CLI flag: `--model_export hf_format`
- Weight format options:
  - `real_quantized` (default) — compressed, actual quantized weights
  - `fake_quantized` — full-precision weights with quantization metadata only

### 2. ONNX (`onnx`)

- Produces: `quark_model.onnx` with optimization passes applied
- Compatible with: ONNX Runtime, TensorRT (with conversion)
- CLI flag: `--model_export onnx`
- Supports INT4/UINT4 conversion pass automatically

### 3. GGUF (`gguf`)

- Produces: GGUF format file for llama.cpp and compatible inference engines
- Requires: `gguf>=0.10.0` package and tokenizer path
- CLI flag: `--model_export gguf`
- Best with: `uint4_wo_32` scheme + AWQ algorithm

Multiple formats can be exported simultaneously: `--model_export hf_format --model_export gguf`

## Export CLI Arguments

```bash
python quantize_quark.py \
  --model_dir /path/to/model \
  --output_dir /path/to/output \
  --quant_scheme fp8 \
  --model_export hf_format \                    # Export format(s)
  --export_weight_format real_quantized \        # Compression mode
  --custom_mode quark \                          # Export mode: quark|awq|fp8
  --pack_method reorder                          # Weight packing: order|reorder
```

## Evaluation Options

Post-quantization evaluation can be configured as part of the export step:

### Perplexity (PPL)

- Default dataset: `wikitext`
- Flag: included by default unless `--skip_evaluation` is set

### Task-Based Evaluation (via lm-eval harness)

```bash
--tasks hellaswag,winogrande,arc_easy
--eval_batch_size auto
--num_fewshot 0
```

### ROUGE/METEOR (for generation models)

```bash
# Evaluated on cnn_dailymail by default
--use_mlperf_rouge  # For MLPerf-compatible ROUGE scoring
```

### KV Cache Evaluation

```bash
--use_ppl_eval_for_kv_cache
--ppl_eval_for_kv_cache_context_size 1024
--ppl_eval_for_kv_cache_sample_size 512
```

### Skip Evaluation

```bash
--skip_evaluation  # Skip all post-quantization evaluation
```

## Model Reload for Separate Evaluation

If quantization and evaluation are done in separate steps:

```bash
# Step 1: Quantize and export
python quantize_quark.py --model_dir MODEL --quant_scheme fp8 \
  --model_export hf_format --output_dir output/ --skip_evaluation

# Step 2: Reload and evaluate
python quantize_quark.py --model_dir MODEL --model_reload \
  --output_dir output/ --skip_quantization
```

## Rules

- **Never invent export paths** that conflict with the existing `run_manifest.yaml` or `quant_plan.json`.
- **Export depends on quantization.** If the model has not been quantized yet, this skill produces an export plan attached to the manifest — it does not run quantization.
- **Match export format to deployment target.** Ask the user where the model will run: HuggingFace ecosystem → `hf_format`, llama.cpp → `gguf`, ONNX Runtime → `onnx`.
- **GGUF works best with UINT4.** If the user wants GGUF but the plan uses FP8, flag the mismatch — GGUF is primarily designed for integer quantization.

## Interaction Flow

1. **Check prerequisites**: Is there a `quant_plan.json`? Has quantization been run or is this plan-only?
2. **Choose format**: Ask where the model will be deployed and recommend the right export format.
3. **Configure evaluation**: Ask if the user wants post-quantization evaluation and which metrics.
4. **Emit**: Update `run_manifest.yaml` with export and evaluation configuration.

## Recovery

- If export is requested before quantization, return the missing prerequisites and attach the export request to the manifest under `pending_exports`.
- If GGUF export fails, check that `gguf>=0.10.0` is installed and that the tokenizer is accessible.
- If ONNX export fails on a complex model, suggest trying `hf_format` first as a fallback.
