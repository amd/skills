# AutoSearchPro Preset Reference

Source: `quark/onnx/quantization/auto_search/qconfig_mapping.py`. Fetch via
`get_auto_search_config(name)`.

## Built-in presets

| Preset | Activation | Weight | Calibration | Algorithms | Best for |
|--------|-----------|--------|-------------|-----------|----------|
| `ADVANCED_SEARCH` | Int8 sym+asym | Int8 sym | MinMax / Percentile / LayerwisePercentile | adaround, adaquant (LR 1e-6→1e-3, iters 3k→30k) | Accuracy-first, slowest |
| `XINT8_SEARCH` | XInt8 sym | XInt8 sym tensor | MinMSE | cle + adaround/adaquant (iters 3k→20k) | AMD Ryzen AI NPU CNN |
| `A8W8_SEARCH` | Int8 sym+asym | Int8 sym tensor | MinMax / Percentile / LayerwisePercentile + `PercentileCandidates` | adaround, adaquant, cle, DPU alignments | Generic CPU/GPU INT8 |
| `A16W8_SEARCH` | Int16 sym+asym | Int8 sym tensor | MinMax / Percentile | cle + adaround/adaquant, `AlignEltwiseQuantType` | Accuracy-sensitive (W8 lost too much) |

All defaults: `search_algo = "TPE"`, `direction = "minimize"`, `n_trials = 20`,
`n_jobs = 1`, `two_stage_search = True`, `optim_device = "cuda:0"`,
`study_storage_db = "auto_search.db"`, `load_study_if_exists = True`.

## Recommended preset by deployment

| Deployment | Priority | Recommended preset |
|------------|----------|--------------------|
| AMD NPU CNN (Ryzen AI) | Best accuracy | `XINT8_SEARCH` |
| AMD NPU Transformer | Best accuracy | custom: `BFP16Spec` weights + Int8 act |
| CPU / CUDA / ROCm general | Best accuracy | `ADVANCED_SEARCH` |
| CPU / CUDA / ROCm general | Smallest model | `A8W8_SEARCH` |
| Any | Recover lost accuracy from W8A8 | `A16W8_SEARCH` |
| Any | Custom search space provided | use as-is, validate first |

## Custom search-space rules

`quark.onnx.quantization.auto_search.utils.validate_search_space` enforces:

- **Base fields** → `list[T]`.
- **Conditional fields** → `dict` containing `"only_if"` (string ⇒ that field
  must be set; `{field: value}` ⇒ exact match) plus list-valued parameters.
- **Continuous parameters** → `{"type": "int" | "float", "low": …, "high": …,
  "step": …, "log": …}`.

Returns `{"discrete_space_size": int, "contains_continuous": bool}`. If
`n_trials > discrete_space_size` and no continuous fields, AutoSearchPro
auto-clamps — surface this instead of letting it happen silently.

## Device gating

For every list-valued device field in the selected `search_space`
(`adaround_params.{optim,infer}_device`, `adaquant_params.{optim,infer}_device`),
replace `"cuda:0"` with `"cpu"` when `env_context.json` reports no CUDA/ROCm:

```python
for key in ("adaround_params", "adaquant_params"):
    if key in cfg["search_space"]:
        cfg["search_space"][key]["optim_device"] = ["cpu"]
        cfg["search_space"][key]["infer_device"] = ["cpu"]
```

Never silently keep the CUDA value.

## Evaluator modes

| Mode | What you set | Returns | When to use |
|------|--------------|---------|-------------|
| Built-in metric | `search_metric = "L2" \| "L1" \| "cos" \| "psnr" \| "ssim"`; `search_evaluator = None` | Per-sample distance between float and quantized output `.npy` dumps | Default — quick, no extra code |
| Custom evaluator | `search_evaluator = fn` where `fn(onnx_path) -> float` | Whatever your task metric returns; AutoSearchPro uses `base_metric - quantized_metric` | Detection mAP, top-1, perplexity, BLEU |

Keep `direction = "minimize"` and return `−metric` from `fn` when your metric
is higher-is-better.

## Sampler choices

`search_algo` accepts: `TPE` (default), `Random`, `CmaEs`, `GPS`, `NSGAII`,
`QMC`, `Grid`. See `qconfig_mapping.get_sampler_dict`.
