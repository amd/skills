---
name: quark-torch-file2file-quantization
description: >-
  Low-memory file2file quantization for very large safetensors LLMs that cannot be loaded whole.
  Use when the user wants to run file2file quantization, adapt a new safetensors checkpoint
  without loading the full model, register an external LLMTemplate, inspect sharded checkpoint
  naming, generate wrapper or conversion scripts, or validate low-memory sharded quantization
  outputs. Trigger for "run file2file quantization", "quantize without loading the model", "large
  safetensors low-memory quantization", "file2file for DeepSeek/Qwen/MoE", "safetensors naming
  incompatible", "register LLMTemplate externally".
---

# quark-file2file-quantization-runner

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

## Purpose

Quantize very large safetensors checkpoints without loading the full model into memory, using
`ModelQuantizer.direct_quantize_checkpoint()`. Covers the full adaptation path: checkpoint
inspection, external `LLMTemplate` registration, optional naming normalization via a one-time
conversion script, minimum-scale experiment gating, full file2file execution, and output
validation via `quark-torch-result-validator`.

Default policy: solve naming or layer-selection mismatches with external adapters
(`LLMTemplate.register_template()`, `weight_converters`) or a temporary conversion script.
Do **not** modify Quark source unless the required capability is absent, the external path has
been ruled out, and the user explicitly agrees.

## Inputs

- `pretrained_model_path` — local directory of the safetensors checkpoint
- `save_path` — output directory
- `quant_scheme` — quantization scheme (e.g., `w_fp8_a_fp8`, `w_int4_a_bf16`)
- `device` — e.g., `cuda:0`, `cpu`
- User intent: direct quantization, checkpoint conversion first, HF-format export, or script generation only
- `model_analysis.json` from `quark-torch-model-intake` (optional but recommended)

## Outputs: run_manifest.yaml

Records the adaptation path taken, scripts generated, experiment results, and validation status.

```yaml
pretrained_model_path: /models/DeepSeek-V3
save_path: /output/DeepSeek-V3-fp8
quant_scheme: w_fp8_a_fp8
device: cuda:0
adaptation_path: external_template          # direct | external_template | weight_converters | conversion_script
conversion_script: null                     # path if generated
wrapper_script: /tmp/run_ds_v3_f2f.py
min_experiment:
  status: passed                            # passed | failed | skipped
  layers_covered: [layer_0_expert_0, ...]
  moe_covered: true
full_run_status: completed                  # pending | completed | failed
validation_status: passed                  # passed | failed | not_run
```

## Interaction Flow

### Step 1 — Inspect checkpoint

Read keys and config without loading weights:

```bash
python - <<'PY'
import json, os
from glob import glob
from safetensors.torch import safe_open

model_dir = "<pretrained_model_path>"
index_path = os.path.join(model_dir, "model.safetensors.index.json")
print("config:", os.path.exists(os.path.join(model_dir, "config.json")))
print("index:", os.path.exists(index_path))
files = sorted(glob(os.path.join(model_dir, "*.safetensors")))
print("safetensors:", len(files))
if files:
    with safe_open(files[0], framework="pt", device="cpu") as f:
        keys = list(f.keys())
    print("sample_keys (first 80):")
    for k in keys[:80]: print(" ", k)
if os.path.exists(index_path):
    with open(index_path) as f:
        wm = json.load(f).get("weight_map", {})
    print("index_keys:", len(wm))
PY
```

Verify: `model_type`, weight-name suffixes (`*.weight`, `*_scale_inv`, `*.scale`), shard count,
MoE expert / shared-expert / gate naming, and whether scale tensors are co-located with weights.

### Step 2 — Choose adaptation path

| Situation | Action |
|-----------|--------|
| Names already match Quark template | Direct file2file; tune `exclude_layers` only |
| Layer naming differs from built-in template | External `LLMTemplate.register_template()` |
| Only weight suffixes differ | `weight_converters` / `_apply_weight_converters` |
| Scale naming or dtype incompatible pre-recovery | Generate normalization conversion script first |

`_apply_weight_converters` limits: suited for post-recovery single-suffix rename or one-source split.
Not suited for multi-source merge, cross-shard scale pairing, or FP4→FP8 dtype conversion.

### Step 3 — Generate wrapper or conversion script

For external template registration, generate a wrapper script (do NOT modify `quantize_quark.py`):

```python
from quark.torch import ModelQuantizer
from quark.torch.utils.llm import LLMTemplate

template = LLMTemplate(
    model_type="<model_type>",
    kv_layers_name=["<pattern>"],
    q_layer_name=["<pattern>"],
    exclude_layers_name=["embed", "head", "<other>"],
)
LLMTemplate.register_template(template)

quantizer = ModelQuantizer(config)
quantizer.direct_quantize_checkpoint(
    pretrained_model_path=pretrained_model_path,
    save_path=save_path,
    keep_excluded_layers_as_original_model_state=False,
    weight_converters=weight_converters,
    device=device,
)
```

Wrapper must print: registered `model_type`, input/output dirs, quant scheme, and exclude rules.
Conversion scripts must stream safetensors (no full-model load), include explicit `remap_name()`,
scale/weight pairing validation, shard output in HF style, index rebuild, and atomic output.

### Step 4 — Minimum-scale experiment (mandatory gate)

This step is not optional. Full file2file must not run until the minimum experiment passes.

Construct the minimum input:

1. If `num_hidden_layers` is safely reducible, copy `config.json` with the smallest value that
   still covers at least one MoE layer (for MoE models, use `first_moe_layer_id + 1`).
2. If not, generate a subset checkpoint filtered by `--key-regex` covering at least one complete
   MoE expert + its scale tensor + shared expert/router/gate + adjacent non-quantized tensors.
3. Never set `num_hidden_layers=1` for a MoE model if layer 0 is dense — verify from config or
   key patterns which layer is the first actual MoE layer.

Run the minimum experiment, then call `quark-torch-result-validator` with:
`inspect_safetensors`, `summarize_dtypes`, `check_index_consistency`, `check_scale_pairs`,
`get_fuzzy_tensor_names`, and auxiliary-file copy check (if source dir available).

If validation fails, fix template / naming / subset and re-run. Do not proceed to full file2file.

### Step 5 — Full file2file

Set cache paths to avoid polluting home quota:

```bash
export TMPDIR=/path/to/run/tmp
export TORCH_EXTENSIONS_DIR=/path/to/run/torch_extensions
export TRITON_CACHE_DIR=/path/to/run/triton_cache
```

Run the wrapper script. After completion, re-run `quark-torch-result-validator` on the
final output with the same checks as Step 4.

### Step 6 — Deliver

Emit `run_manifest.yaml` and present to the user:

- Script paths and execution commands
- Minimum experiment summary: layers selected, MoE coverage, validation outcome
- Full run validation summary
- Any outstanding risk items

## Recovery

| Failure | Action |
|---------|--------|
| Missing Triton / compressed-tensors | Install dependency, retry |
| Incomplete source shards | Repair shards and index before proceeding |
| Template not recognized | Register via external `LLMTemplate`; do not patch Quark source |
| Scale/weight cannot be paired | Normalize checkpoint first; add unit test if Quark recovery is extended |
| Output index inconsistent | Rebuild index or fix shard write logic; re-validate |
| Minimum experiment fails | Fix adapter/naming/subset; never skip to full run |

## Rules

- **Never load full weights** for inspection — read safetensors headers only.
- **External adapters first**: use `LLMTemplate.register_template()` or `weight_converters` before
  any Quark source change.
- **Minimum experiment is a gate, not a hint** — full file2file is blocked until it passes.
- **MoE minimum experiment** must include actual MoE expert weights + their scale tensors.
  Setting `num_hidden_layers=1` when layer 0 is dense is invalid.
- **No cluster paths in shared scripts** — keep one-time paths in temporary wrapper scripts only.
- **Source changes require tests** — if Quark source must change, add a `test/test_for_torch/`
  unit test and run the relevant pytest before committing.

## Notes

- DeepSeek-V4-family: prefer external `LLMTemplate`; check whether inference-format naming needs
  a conversion script (`embed_tokens→embed`, `self_attn→attn`, `q_proj→wq`, etc.) before file2file.
- FP4/e2m1fn expert weights must be converted to FP8/e4m3fn in the conversion script, not via
  `_apply_weight_converters`.
- Sibling scale naming (`{base}.scale`) must be resolvable before recovery; post-recovery suffix
  conversion cannot substitute for pre-recovery scale identification.
- MTP embedding/head, attention, gate, `hc_*` auxiliary tensors must be explicitly excluded in
  the template or conversion script.
