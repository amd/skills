---
name: quark-onnx-debug
description: >-
  Diagnose failed Quark ONNX installation, calibration, quantization, custom-op compilation, or
  export attempts. Use when the user reports an error, stack trace, invalid artifact, missing
  dependency, ORT execution-provider mismatch, silent CPU fallback, OOM during calibration,
  custom-op load failure (BFPQuantizeDequantize / MXQuantizeDequantize / Extended*), or unexpected
  quantization results from the ONNX flow. Trigger for "Quark ONNX error", "onnxruntime error",
  "quantize_static failed", "calibration crashed", "CUDAExecutionProvider not available",
  "ROCMExecutionProvider not available", "custom op library load failed", "model.onnx larger than
  2GB", "external data not found", "AdaRound diverged", "GPTQ ONNX failed", "QuaRot failed", "NPU
  power-of-2 scale", any Python traceback mentioning quark.onnx / onnxruntime / onnx.
---

# quark-onnx-debug

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

## Purpose

Convert ONNX-flow failures into a structured diagnostic report with the smallest safe recovery path.
Debugging Quark ONNX issues is tricky because errors can originate from many layers — Python
environment, the `onnx` package version, the installed `onnxruntime*` variant, custom-op
compilation, CUDA / ROCm / NPU EPs, calibration data plumbing, graph optimization, or the chosen
quantization config itself. This skill systematically narrows down the root cause.

## Inputs

- Error message and stack trace from a failing run
- The exact command or `ModelQuantizer` / `quantize_static` invocation that triggered it
- `env_context.json`, `onnx_install_result.json`, `quark_install_result.json` (optional, for
  environment and install state)
- The model path / size and (if relevant) the `QuantizationConfig` used

## Outputs: validation_report.md

Diagnostic report with root cause, evidence, and the smallest safe fix.

Schema: [`validation_report.schema.json`](https://github.com/amd/Quark/blob/1b229f781a1974cc742884e42d8eefc1eebb4f0a/.claude/skills-impl/shared/contracts/validation_report.schema.json)

````markdown
# Debug Report

## Common Error Patterns

### Installation Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `ModuleNotFoundError: No module named 'quark'` | Quark not installed or wrong Python env | `pip install amd-quark` or activate correct conda env |
| `ModuleNotFoundError: No module named 'onnxruntime'` | ONNX Runtime not installed | Hand off to `quark-onnx-install` |
| `ModuleNotFoundError: No module named 'onnx'` | `onnx` package not installed | `pip install "onnx>=1.21.0,<=1.22.0"` |
| `ImportError: quark.onnx.operators.custom_ops` / compile failure | Missing C++ compiler (Linux `g++`, Windows VS 2022) or `ROCM_PATH`/`CUDA_HOME` unset for GPU build | `apt install g++` (Linux), install VS 2022 (Windows), `export ROCM_PATH=/opt/rocm` or `export CUDA_HOME=/usr/local/cuda` |
| Both `onnxruntime` and `onnxruntime-gpu` (or `_rocm`) installed | Variant collision — wrong EP loads | `pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime_rocm`, then reinstall the single correct variant via `quark-onnx-install` |
| `onnx` schema / `opset` errors at import | `onnx` outside `>=1.16.0,<=1.19.0` | Pin within the supported range |
| `onnxruntime` ABI / symbol errors when loading custom ops | ORT version outside `>=1.22.2,<=1.24.2` (custom-ops built against a different ABI) | Reinstall ORT within the supported range, then re-trigger custom-ops compile |

### ONNX Runtime / Execution Provider Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `get_available_providers()` lacks `CUDAExecutionProvider` | CPU `onnxruntime` installed on a CUDA box | Reinstall `onnxruntime-gpu` via `quark-onnx-install` |
| `get_available_providers()` lacks `ROCMExecutionProvider` on ROCm 6.x | Wrong variant (need `onnxruntime_rocm` from AMD Artifactory) | Reinstall `onnxruntime_rocm` via `quark-onnx-install` |
| `ROCMExecutionProvider` missing on ROCm 7.x | **By design** — ROCm 7.x falls back to CPU `onnxruntime` per `tools/ci/install_onnxruntime.sh` (build incompatibility) | Document the CPU-only fallback; do not attempt a ROCm wheel install |
| Silent CPU fallback (no GPU utilization during calibration) | EP not requested or unavailable | Pass `execution_providers=['CUDAExecutionProvider']` / `['ROCMExecutionProvider']` explicitly; verify with `get_available_providers()` |
| `Failed to load library libonnxruntime_providers_cuda.so` | CUDA toolkit version mismatch with `onnxruntime-gpu` build | Match `onnxruntime-gpu` to the system CUDA major (CUDA 11 → Azure DevOps index; CUDA 12/13 → pypi default) |
| `RuntimeError: ... onnxruntime::rocm::...` | ROCm driver / library mismatch | Verify `rocm-smi` matches the wheel's ROCm version |

### Custom Op Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `RuntimeError: Failed to load custom op library` | Custom-ops library was not compiled, or compiled against a different ORT version | Re-run `python -c "import quark.onnx.operators.custom_ops"` and inspect the compile output |
| `Op (BFPQuantizeDequantize) ... is not a registered function/op` | The session was created without registering Quark's custom-op library | Use `quark.onnx.ModelQuantizer` / `quantize_static`, or pass the custom-op `.so`/`.dll` via `SessionOptions.register_custom_ops_library()` |
| Custom-op symbol-not-found on GPU but works on CPU | GPU kernel of the custom op was not built (missing `ROCM_PATH` / `CUDA_HOME` at compile time) | Set the env var, delete the cached `.so`/`.dll`, re-import to recompile |

### Model Loading / Shape Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `FileNotFoundError: Input model file ... does not exist.` (`api.py:103`) | Wrong `model_input` path | Check absolute path to `model.onnx` |
| `onnx.onnx_cpp2py_export.checker.ValidationError` | Model fails ONNX schema check | Run `onnx.checker.check_model()` to localize; rebuild the model with a compatible opset |
| `Message ... exceeds 2GB` / ProtoBuf size error | Model > 2 GB without external data | Save with `save_as_external_data=True`; pass `use_external_data_format=True` to the quantizer |
| `external data file not found` | `.onnx` moved but `.onnx_data` / weight blobs left behind | Move the model directory as a whole, or re-export with external data adjacent |
| Shape-inference failure | Incomplete shapes in the model | Run `quark.onnx.tools.fix_shapes` / `onnx.shape_inference.infer_shapes` before quantization |

### Calibration Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `ValueError: The data reader should implement the '__len__' method to provide the data size.` (`calibrators.py:239`) | Custom `CalibrationDataReader` missing `__len__` | Implement `__len__` returning sample count |
| `ValueError: No data is collected.` (`calibrators.py:253`) | Data reader yielded zero batches before exhaustion | Check reader's `get_next()` returns at least one batch |
| `TypeError: compute_data must return a TensorsData not <...>` (`calibrators.py:259`) | Custom calibrator subclass returned wrong type | Return `onnxruntime.quantization.calibrate.TensorsData` |
| `ValueError: No collector created and can't generate calibration data.` | `collect_data()` never called before `compute_data()` | Call `collect_data()` first, or use `ModelQuantizer.quantize_model()` which handles the order |
| `ValueError: Invalid averaging constant, which should not be < 0 or > 1.` (`calibrators.py:563`) | `moving_average=True` with out-of-range constant | Use a value in `[0, 1]`, typically `0.01` |
| `ValueError: Unsupported calibration method` (`calibrators.py:1237`) | Typo or unsupported `CalibrationMethod` | Check `CalibrationMethod` enum in `quark.onnx.quantization.config.config` |
| OOM during calibration | Activation cache too large | Set `optimize_mem=True` (disk cache) and/or `optimize_disk=True`; reduce `worker_num` if memory bound; reduce calibration sample count |

### Quantization Config Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `ValueError: Only ExtendedQuantFormat.QDQ supports wide bits quantization types.` (`input_check.py:67`) | INT16/UINT16/INT32 with `QuantFormat.QOperator` | Switch to `ExtendedQuantFormat.QDQ` |
| `ValueError: Fast finetune does not support int4 or uint4.` (`input_check.py:118`) | AdaRound/AdaQuant + INT4 weights | Disable fast-finetune for INT4, or use GPTQ instead |
| `ValueError: Invalid quant overrides ... for tensor ...` (`input_check.py:86`) | Tensor name in `extra_options['QuantOverrides']` not in the graph | Verify tensor name with `onnx.load(...).graph` |
| `ValueError: The per-channel quant override ... can not be applied on <quant_type>` (`input_check.py:94`) | Per-channel override on a quant type that doesn't support it (e.g., FP types, block formats) | Remove per-channel from that override |
| `ValueError: For the crypto mode, the input model should be in onnx.ModelProto format.` (`input_check.py:137`) | `crypto_mode=True` with a file path | Load model first: `model = onnx.load(path)` |
| `ValueError: quantization config must be one of Config and QConfig.` (`api.py:73`) | Passed a dict or wrong type | Build a `quark.onnx.quantization.config.Config` or `QConfig` |
| `ValueError: Unexpected config name: <X>` (`custom_config.py:993`) | Typo in `get_default_config(<X>)` preset name | Pick from `DefaultConfigMapping` (XINT8, S8S8_AAWS, A8W8, A16W8, BF16, BFP16, MX4/6/9, MXFP8/6/4, …) |
| `nodes_to_quantize` ignored | Node names don't match graph after pre-processing (NCHW→NHWC, BN folding) | Run pre-processing first, dump the post-preprocess graph, then re-select node names |

### Algorithm Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| AdaRound / AdaQuant device mismatch | `optim_device='cuda:0'` but no `CUDAExecutionProvider` available | Either set `optim_device='cpu'` or fix the ORT install via `quark-onnx-install` |
| AdaRound / AdaQuant divergence (loss → NaN/Inf) | LR too high for the model, or activation outliers | Lower `learning_rate`, enable `CLE` pre-processing first |
| GPTQ shape mismatch | Group-size doesn't divide the weight dimension | Pick a `group_size` that divides the weight's reduction dim (commonly 32, 64, 128) |
| SmoothQuant alpha range | `SmoothAlpha` outside `[0, 1]` | Use a value in `[0.5, 0.85]` for LLMs |
| QuaRot rotation config invalid | Missing rotation pair definitions | Provide the rotation pairs in the algorithm config; see `quark/onnx/algorithm/quarot/` |
| `AutoSearchPro: 'model_input' can not be None.` (`auto_search_pro.py:139`) | Search invoked without a model | Pass the model path or `ModelProto` |
| `AutoSearchPro: 'calib_data_reader' can not be None.` (`auto_search_pro.py:142`) | Search invoked without calibration data | Provide a `CalibrationDataReader` |
| `AutoSearchPro: Unsupported search_algo` (`auto_search_pro.py:192`) | Sampler typo | Use `'TPE'` or `'Grid'` |

### NPU-Specific Errors (Ryzen AI / VAI)

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| Power-of-2 scale validation failure | Non-PoF2 scale produced for an NPU target | Use a PoF2 calibrator (`PowOfTwoCalibrater` MinMSE / NonOverflow); confirm `enable_npu_cnn=True` |
| `enable_npu_cnn=True` but model has unsupported ops | NPU CNN backend doesn't support the op | Fold / replace via `quark.onnx.tools.*`; see `optimizations/optimize.py` |
| NCHW vs NHWC mismatch | NPU expects NHWC but model is NCHW | Run `quark.onnx.tools.convert_nchw_to_nhwc` before quantization |
| `enable_dpu` deprecated warning | Old API | Switch to `enable_npu_cnn=True` |

### Export / Post-quantization Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `PermissionError` on output dir | No write permission | Check output directory permissions |
| Quantized model > 2 GB and external data not written | `use_external_data_format=False` on a large model | Set `use_external_data_format=True` |
| Inference using the quantized model errors on a custom op | Deployment env missing Quark's custom-ops library | Ship the compiled `.so`/`.dll` and register via `SessionOptions.register_custom_ops_library()` |

## Diagnostic Process

1. **Read the error** — capture the exact error message, full stack trace, and the command or
   `quantize_static` / `ModelQuantizer` call that triggered it.
2. **Identify the layer** — installation / ORT runtime / custom-op / model loading / calibration /
   config validation / algorithm / NPU-specific / export.
3. **Check the environment** — Python version, `onnx` version, installed `onnxruntime*` variants
   and versions, available EPs, Quark version, accelerator (CUDA major / ROCm major), whether the
   custom-ops library compiled.
4. **Match against known patterns** — use the tables above.
5. **Propose the fix** — smallest change that resolves the issue without side effects.

## Diagnostic Commands

```bash
# Environment snapshot
python -c "
import sys; print('Python:', sys.version.split()[0])
try:
    import onnx; print('onnx:', onnx.__version__)
except Exception as e: print('onnx: not installed', e)
try:
    import onnxruntime as ort
    print('onnxruntime:', ort.__version__)
    print('EPs:', ort.get_available_providers())
except Exception as e: print('onnxruntime: not installed', e)
try:
    import quark; print('Quark:', quark.__version__)
except Exception as e: print('Quark: not installed', e)
try:
    import quark.onnx; print('quark.onnx loaded from', quark.onnx.__file__)
except Exception as e: print('quark.onnx: load failed', e)
"

# Confirm only ONE onnxruntime variant is installed
pip list 2>/dev/null | grep -iE '^(onnx|onnxruntime|onnxslim|onnxscript|onnxruntime-genai|onnxruntime_rocm|onnxruntime-extensions)\b'

# Force custom-ops compile (first import) and capture failures
python -c "import quark.onnx.operators.custom_ops" 2>&1 | tail -30

# Model sanity check
python -c "
import onnx, sys
m = onnx.load(sys.argv[1])
print('opset:', [(o.domain or 'ai.onnx', o.version) for o in m.opset_import])
print('inputs:', [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
onnx.checker.check_model(m, full_check=False)
print('checker: OK')
" path/to/model.onnx

# GPU memory status
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null
rocm-smi --showmemuse 2>/dev/null
```

## Rules

- **Always ask for the full error message and the call that was run.** Partial errors lead to wrong
  diagnoses. For ONNX, also ask for: the `QuantizationConfig` / preset name, the model path and
  size, the chosen `execution_providers`, and the calibration data reader class.
- **Do not guess the fix.** Narrow down the root cause first, then propose a specific solution.
- **Distinguish "EP not available" from "EP available but not requested".** Both cause silent CPU
  fallback. Check `ort.get_available_providers()` *and* the providers actually passed to the
  session.
- **Confirm before mutating.** If the fix involves reinstalling `onnxruntime*`, rebuilding the
  custom-ops library, or re-running an hour-long calibration, present the plan and get
  confirmation.
- **Consider cascading effects.** Bumping `onnxruntime` may force a custom-ops rebuild and may break
  ABI with an older `onnx` pin. Bumping `onnx` outside `<=1.19.0` may break Quark's QDQ insertion.
  Always check version constraints (`requirements.txt`, `docs/source/install.rst`).
- **For ROCm 7.x with no ROCM EP, this is by design.** Do not propose a "fix" that downgrades to
  ROCm 6.x unless the user asks; instead document the CPU-fallback trade-off (see
  `tools/ci/install_onnxruntime.sh`).

## Example: CUDA EP missing during calibration

### Error Summary

`onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: ... CUDAExecutionProvider is not in
the list of available providers` raised from `ModelQuantizer.quantize_model()` on a CUDA 12 box.

### Root Cause

CPU `onnxruntime` was installed instead of `onnxruntime-gpu`; `get_available_providers()` returns
`['CPUExecutionProvider']` only.

### Evidence

- `pip list | grep onnxruntime` shows `onnxruntime 1.23.2` only (no `-gpu` variant).
- `python -c "import onnxruntime as ort; print(ort.get_available_providers())"` →
  `['CPUExecutionProvider']`.
- `nvidia-smi` reports a healthy CUDA 12.4 driver and an idle GPU.

### Fix

Reinstall the correct variant via `quark-onnx-install` (CUDA 12 path):

```bash
pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime_rocm
pip install --no-cache-dir onnxruntime-gpu
python -c "import onnxruntime as ort; print(ort.get_available_providers())"   # expect CUDAExecutionProvider present
python -c "import quark.onnx.operators.custom_ops"                            # recompile against new ORT
```

### Prevention

Run `quark-onnx-install` (which routes through `quark-env-preflight`) before the first
quantization on a fresh environment, so the ORT variant is picked from the accelerator instead of
defaulting to CPU.
````

## Interaction Flow

1. **Gather evidence**: error message, stack trace, exact call, model size, config, EPs.
2. **Classify**: install / custom-op / ORT runtime / model / calibration / config / algorithm /
   NPU / export.
3. **Diagnose**: match against known patterns and run diagnostic commands if needed.
4. **Propose fix**: present the smallest change that resolves the issue.
5. **Confirm**: get user approval before reinstalls, custom-ops rebuilds, or long recalibrations.
6. **Verify**: after the fix, re-run the failing step (or a smaller smoke variant) to confirm
   resolution.

## Recovery

- If the fix requires an `onnxruntime*` or `onnx` package change (wrong variant, version
  mismatch, missing variant), hand off to `quark-onnx-install` with the specific requirement noted.
- If the fix requires an `amd-quark` package or shared-dependency change, hand off to
  `quark-install` with the specific requirement noted.
- If the fix requires a C++ compiler install or `ROCM_PATH`/`CUDA_HOME` setup, surface the
  environment gap and let the user resolve it before re-running custom-ops compile.
- If the root cause is upstream (ORT API change, `onnx` schema change), hand off to
  `the Quark repository's internal ONNX documentation drift checker` or `the Quark repository's internal ONNX skill synchronizer`.
- If the error is in a custom config or unsupported op for an NPU target, suggest the relevant
  pre-processing tool from `quark.onnx.tools` (e.g. `convert_nchw_to_nhwc`,
  `convert_qdq_to_qop`, `convert_a8w8_npu_to_a8w8_cpu`).
- If the issue cannot be reproduced from the supplied evidence, ask for the minimum repro
  (model, calibration data sample, config) before guessing.
