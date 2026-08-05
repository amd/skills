---
name: quark-onnx-autosearch-pro
description: >-
  End-to-end Quark ONNX AutoSearchPro recipe — drives `quark.onnx.AutoSearchPro` (Optuna-based
  hyperparameter search) on a `.onnx` model to find the best quantization config
  (activation/weight spec, calibration method, CLE, AdaRound / AdaQuant, FastFinetune params). Use
  when the user wants to "auto search", "tune quantization", "find the best quant config", "sweep
  AdaRound/AdaQuant", "run AutoSearchPro / AutoSearch", "two-stage search", or pick one of the
  built-in presets (`ADVANCED_SEARCH`, `XINT8_SEARCH`, `A8W8_SEARCH`, `A16W8_SEARCH`) for their
  `.onnx` model. Not for HuggingFace / safetensors / PyTorch input models — use quark-torch-ptq
  instead. Not for single-shot ONNX PTQ without search — use quark-onnx-ptq.
---

# quark-onnx-autosearch-pro

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

## Purpose

Drive Quark ONNX `AutoSearchPro` (Optuna-driven) on a `.onnx` model to discover
the best quantization configuration — activation / weight specs, calibration
method, CLE, AdaRound / AdaQuant hyperparameters — without hand-tuning. The
recipe orchestrates intake, preset (or custom) search-space selection, reader
construction, generation of a standalone autosearch script in the user's
workspace, confirmed execution, and reporting of `best_params.json` plus
Optuna study artifacts. Unlike single-shot PTQ this produces many candidate
models, so preset and trial budget are explicit and the search is bounded.

## Inputs

- Input `.onnx` model (with optional sibling `.onnx_data`).
- Calibration data: folder of representative samples or a Python
  `CalibrationDataReader` class. Optional separate evaluation reader.
- User goal: preset choice (`ADVANCED_SEARCH` / `XINT8_SEARCH` /
  `A8W8_SEARCH` / `A16W8_SEARCH` / custom dict), deployment target, search
  budget (`n_trials`, `n_jobs`, `two_stage_search`), and metric (built-in or
  custom callable).
- `session_context.json` (`constraints.backend = "onnx"`), `env_context.json`,
  `workspace_context.json`, plus `onnx_install_result.json` and
  `quark_install_result.json`. **Optuna** must be importable
  (`pip install optuna`).

## Outputs: run_manifest.yaml

Records the generated autosearch script path, the exact `python3` invocation,
the preset name (or custom search-space hash), trial budget, Optuna study DB
path, and the final `best_params.json` location.

Side artifacts written by `AutoSearchPro` into `output_dir`:
`auto_search.log`, `best_params.json`, `auto_search.db` (resumable Optuna
study), `quantized_model_<trial>.onnx` per trial, and `opt_history.html` /
`param_importance.html` when `plot_results=True`.

Built in Step 4. Schema:
[`run_manifest.schema.json`](https://github.com/amd/Quark/blob/1b229f781a1974cc742884e42d8eefc1eebb4f0a/.claude/skills-impl/shared/contracts/run_manifest.schema.json)

## Interaction Flow

1. **Intake** — `quark-onnx-model-intake` → `model_analysis.json`
2. **Preset / search space** — pick preset or build custom → `quant_plan.json`
   (with `search_space` block + `n_trials`)
3. **Reader + evaluator** — confirm `CalibrationDataReader` and evaluator mode
4. **Manifest** — generate standalone autosearch script in user workspace +
   `run_manifest.yaml`; stop for user approval
5. **Execute** — run the confirmed script, report `best_params.json`

## CRITICAL RULES

1. **NEVER call `AutoSearchPro(...).run()` directly from this skill.** Generate
   a standalone script the user reviews first. Searches can take hours and
   write many `.onnx` files — user must opt in.
2. **NEVER skip preset confirmation.** Presets default to
   `optim_device = "cuda:0"`; surface device gaps before execution.
3. **NEVER silently fall back to CPU** when CUDA/ROCm was requested. Hand off
   to `quark-onnx-install` / `quark-onnx-debug`.
4. **NEVER write into the Quark repo.** Generated scripts go to the user's
   workspace, *not* under `examples/onnx/auto_search/` or
   `tutorials/onnx/auto_search/`. The shipped
   `examples/onnx/auto_search/auto_search_pro_model.py` is a template to read,
   not patch.
5. **Pre-check optuna.** `AutoSearchPro` imports it lazily; verify it imports
   in the user's env before writing the script.
6. **Bound the budget.** Refuse `n_trials > 200` without explicit user opt-in
   — one `.onnx` per trial.
7. **Validate custom search spaces** with
   `quark.onnx.quantization.auto_search.utils.validate_search_space`. If
   `n_trials > discrete_space_size` without continuous fields, AutoSearchPro
   auto-clamps — surface it instead of letting it happen quietly.

See [`presets-reference.md`](presets-reference.md) for the preset table,
deployment recommendations, custom-search-space rules, device gating snippet,
evaluator modes, and sampler list.

## Required Artifact Flow

```text
Step 1 (Intake)        ──► model_analysis.json
Step 2 (Preset/space)  ──► quant_plan.json  (search_space + n_trials)
Step 3 (Reader)        ──► (informs script body)
Step 4 (Manifest)      ──► run_manifest.yaml + user_workspace/<name>_autosearch.py
Step 5 (Execute)       ──► best_params.json + auto_search.db + quantized_model_*.onnx
```

---

## Step 1: Model Intake

Hand off to `quark-onnx-model-intake`. Stop if QDQ nodes already exist —
AutoSearchPro requires an unquantized float `.onnx`. Add one recipe-specific
line to the summary: *"Search will produce ~N quantized .onnx files in
`output_dir` — ensure ≥ N × <model_size> free."*

### >>> CHECKPOINT 1: Confirm model analysis

---

## Step 2: Preset / Custom Search-Space Selection

Pick a built-in preset or accept a user-supplied `search_space` dict. See
[`presets-reference.md`](presets-reference.md) for the full preset table and
deployment-target recommendations.

For custom search spaces, run `validate_search_space(...)` in your head
against the spec rules in the reference. Compute `discrete_space_size` and
warn if `n_trials` would be clamped.

### Decision table to show the user

| Decision | Default | Notes |
|----------|---------|-------|
| `search_space` source | preset name | `XINT8_SEARCH` for NPU CNN, `ADVANCED_SEARCH` for accuracy, etc. |
| `search_algo` | `TPE` | also: `Random`, `CmaEs`, `GPS`, `NSGAII`, `QMC`, `Grid` |
| `n_trials` | `20` | preset default; ~4 calib configs + 20 FastFT with `two_stage_search` |
| `n_jobs` | `1` | parallel processes only when GPU memory allows |
| `two_stage_search` | `True` | grid over calib first, then TPE over FastFinetune |
| `search_metric` | `L2` | or `L1` / `cos` / `psnr` / `ssim`, or set `search_evaluator` |
| `direction` | `minimize` | matches L2 distance |
| `output_dir` | `./autosearch_output/` | holds trial `.onnx`, study DB, log |
| `temp_dir` | `./autosearch_temp/` | float-model dumps for built-in evaluator |
| `study_storage_db` | `auto_search.db` | resumable Optuna study |
| `load_study_if_exists` | `True` | resume from interrupted runs |
| `plot_results` | `False` | `True` → `opt_history.html` + `param_importance.html` |
| device override | `cuda:0 → cpu` if no GPU | see device-gating snippet in reference |

### >>> CHECKPOINT 2: User confirms preset, `n_trials`, devices, metric

---

## Step 3: Calibration + Evaluation Reader

Reader patterns mirror `quark-onnx-ptq` Step 3 (vision:
`ImageDataReader`; LLM: tokenizer-driven). AutoSearchPro wraps the reader
with `CachedDataReader` — one pass over data is enough. For evaluator mode
(built-in metric vs. custom `search_evaluator(onnx_path) -> float`) and
`direction` semantics, see [`presets-reference.md`](presets-reference.md).
Pass a distinct `eval_data_reader` to score on a different slice; omit it
to reuse calibration (AutoSearchPro warns).

### >>> CHECKPOINT 3: Confirm reader + evaluator mode

---

## Step 4: Manifest Generation

Translate the confirmed plan into a runnable script in the user's workspace
and a `run_manifest.yaml`. Do not write into the Quark repo.

### Actions

1. **Script path.** Choose `./<model_name>_autosearch.py` in the user's
   working directory. Never under `examples/` or `tutorials/`.
2. **Pre-check optuna.** If `import optuna` fails in the user env, surface
   `pip install optuna` before writing the script.
3. **Build the script body.** Pieces:

   ```python
   # user_workspace/<model>_autosearch.py
   from quark.onnx import AutoSearchPro
   from quark.onnx.quantization.auto_search import get_auto_search_config
   # from onnxruntime.quantization.calibrate import CalibrationDataReader
   # ... ImageDataReader / tokenizer-driven reader from Step 3 ...

   cfg = get_auto_search_config("XINT8_SEARCH")
   cfg["model_input"]          = "./models/yolov8n.onnx"
   cfg["calib_data_reader"]    = ImageDataReader("./calib_data", cfg["model_input"])
   cfg["eval_data_reader"]     = None
   cfg["output_dir"]           = "./autosearch_output"
   cfg["temp_dir"]             = "./autosearch_temp"
   cfg["n_trials"]             = 20
   cfg["n_jobs"]               = 1
   cfg["search_algo"]          = "TPE"
   cfg["two_stage_search"]     = True
   cfg["search_metric"]        = "L2"
   cfg["direction"]            = "minimize"
   cfg["study_storage_db"]     = "auto_search.db"
   cfg["load_study_if_exists"] = True
   cfg["plot_results"]         = True

   # Apply device override here if env has no CUDA/ROCm — see presets-reference.md

   if __name__ == "__main__":
       best = AutoSearchPro(cfg).run()
       print("Best params:", best)
   ```

   If the user picked a custom search space, replace `cfg["search_space"]`
   instead of (or in addition to) using a preset.

4. **Write the script to disk** at the chosen path. Print the full body back
   to the user — never just say "generated a script".
5. **Build the command:**

   ```bash
   python3 ./yolov8n_autosearch.py 2>&1 | tee ./autosearch_output/run.stdout.log
   ```

6. **Expected output layout:**

   ```text
   ./<model>_autosearch.py            (generated here)
   ./run_manifest.yaml                (this recipe's manifest)
   ./autosearch_temp/                 (float-model dumps; cleaned at end)
   ./autosearch_output/
     ├── auto_search.log
     ├── auto_search.db               (resumable Optuna study)
     ├── best_params.json             (final answer)
     ├── quantized_model_*.onnx       (one per trial)
     └── opt_history.html / param_importance.html  (if plot_results=True)
   ```

7. **Estimate cost.** From `model_analysis.json`: disk ≈ model_size ×
   `n_trials`; flag if > 100 GB. Estimate per-trial wall-clock from
   CPU-vs-GPU + FastFinetune iter range so the user knows whether to walk
   away.

### >>> CHECKPOINT 4: Show script body, command, cost estimate; ask "shall I run this?"

If the user declines or wants changes, return to Step 2 (preset/budget) or
Step 3 (reader/evaluator) — never patch the script in place.

---

## Step 5: Execute AutoSearch

Runs ONLY after explicit Step 4 confirmation.

### Actions

1. `mkdir -p ./autosearch_output ./autosearch_temp`
2. Run the script and stream the log. AutoSearchPro logs per-trial:

   ```text
   [Trial 7] Params: {'activation': 'Int8Spec', 'algorithms': 'adaround', ...}
   L2 distance is: 0.00012
   ```

3. After completion, verify:

   ```bash
   ls -lh ./autosearch_output/best_params.json ./autosearch_output/auto_search.db
   ```

4. Read `best_params.json` and report a summary: trials run / budget, best
   trial index, best metric value, the resolved params dict, the winning
   `quantized_model_*.onnx` path, the study DB path (resumable), and the
   plot HTMLs if generated.
5. **Follow-ups:** hand off the winning `.onnx` to
   `quark-onnx-result-validator`; prune unwanted trial `.onnx` files if disk
   is tight; offer a continuation run with a tighter `search_space` around
   the winning region.

### Common failures and routing

- `ModuleNotFoundError: optuna` → `pip install optuna`, restart.
- `CUDAExecutionProvider not available` or custom-op library load failure
  (`BFPQuantizeDequantize`, `MXQuantizeDequantize`, `Extended*`) →
  `quark-onnx-install` / `quark-onnx-debug`; never silently swap to CPU.
- OOM during AdaRound/AdaQuant → narrow `num_iterations` range, lower
  `data_size`, drop `batch_size`, or set `optim_device = ["cpu"]`.
- Resumed run loaded an incompatible study → delete `auto_search.db` or pick
  a new `study_name`.
- "n_trials clamped to discrete_space_size" → expected when search space has
  no continuous fields and is exhausted; informational.

## Recovery

- If any upstream artifact is missing, stop and name the producer skill.
- If the model is already QDQ-quantized, stop — AutoSearchPro requires
  unquantized input.
- If the user wants to change preset / budget mid-run, return to Step 2 —
  do not patch the generated script.
- If `optuna` cannot be installed, surface a clear blocker; there is no
  fallback search backend.
- If the user pastes a Torch traceback or HuggingFace path, hand off to
  `the matching public quark-torch skill` and stop.

## See also

- [`presets-reference.md`](presets-reference.md) — presets, custom-space
  rules, device gating, evaluator modes, samplers.
- `quark-onnx-ptq` — single-shot PTQ alternative.
- `quark-onnx-result-validator`, `quark-onnx-debug`, `quark-onnx-install`.
