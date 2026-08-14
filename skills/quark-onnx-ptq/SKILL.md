---
name: quark-onnx-ptq
description: >-
  End-to-end ONNX PTQ workflow for AMD Quark — for `.onnx` input models (with optional sibling
  `.onnx_data` external-weights file). Use when the user wants a complete ONNX-to-ONNX pipeline:
  model intake, quantization planning, calibration-script generation, manifest, and confirmed
  execution. Trigger for "quantize my .onnx", "run ONNX PTQ end to end", "full ONNX quantization
  pipeline", "quantize yolov8/resnet50/yolo_nas with XINT8/A8W8/BFP16/MXFP*", "weights-only INT4
  for my .onnx LLM", or any request that spans more than one ONNX PTQ step. Not for HuggingFace /
  safetensors / PyTorch input models — use quark-torch-ptq instead.
---

# quark-onnx-ptq

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

When the workflow names one of these internal procedures, read its bundled reference before carrying out that step:

- [`quark-onnx-quant-plan`](references/quark-onnx-quant-plan.md)

> 📘 **Quick-start example — read this first.**
> A fully worked end-to-end walkthrough is available at **[`example-xint8-yolov8n.md`](example-xint8-yolov8n.md)** (XINT8 quantization of YOLOv8n for AMD NPU CNN deployment, calibrated on COCO val2017). It is the fastest way to see exactly what this workflow produces — open it alongside this SKILL.md before running anything.

## Purpose

Chain the ONNX PTQ path — model intake → quantization planning → script + manifest generation → confirmed execution — while keeping the user informed at each checkpoint. This workflow orchestrates the atomic ONNX skills so the user does not have to manually chain them. Unlike the Torch flow there is no single shipped `quantize_quark.py` for ONNX; the workflow generates a small standalone Python script in the user's working directory that imports from `quark.onnx`, then runs that script after the user confirms.

📂 **Worked example:** see [`example-xint8-yolov8n.md`](example-xint8-yolov8n.md) for the full YOLOv8n + XINT8 + NPU-CNN walkthrough referenced throughout the steps below.

## Inputs

- Input `.onnx` model path (with optional sibling `.onnx_data` external-weights file)
- Calibration data: folder of representative samples (or `CalibrationDataReader` Python class)
- Output `.onnx` path (and optional `.onnx_data` if `use_external_data_format=True`)
- User goal: scheme (XINT8 / A8W8 / A16W8 / BF16 / BFP16 / `MX*` / `MXFP*`), deployment target (CPU / CUDA / ROCm / AMD NPU CNN / AMD NPU Transformer), accuracy target
- `session_context.json` with `constraints.backend = "onnx"` for user goal and constraints
- `env_context.json` for hardware / execution-provider facts
- `workspace_context.json` for validated paths
- `onnx_install_result.json` and `quark_install_result.json` to confirm runtime is ready

## Outputs: run_manifest.yaml

Records the generated calibration/quantization script path, the exact `python3` invocation, and the resolved `QConfig`. Side artifacts: the generated script in the user's working directory, the quantized `.onnx` (and `.onnx_data` if external) in the user's output directory. The manifest is built in Step 3.

Schema: [`run_manifest.schema.json`](https://github.com/amd/Quark/blob/1b229f781a1974cc742884e42d8eefc1eebb4f0a/.claude/skills-impl/shared/contracts/run_manifest.schema.json)

## Interaction Flow

1. **Intake** — call `quark-onnx-model-intake` to produce `model_analysis.json`
2. **Plan** — call `quark-onnx-quant-plan` to produce `quant_plan.json`
3. **Manifest** — generate a standalone quantization script in the user's working directory + `run_manifest.yaml`, then stop for user approval
4. **Execute** — run the confirmed script and report the artifact paths

## CRITICAL RULES

1. **NEVER call `ModelQuantizer.quantize_model(...)` directly from this workflow.** Always generate a standalone script the user reviews first.
2. **NEVER skip a step.** Even if the user provides all details upfront, execute each step in order.
3. **STOP at every checkpoint** and wait for user confirmation before continuing.
4. **Show concrete output** at each step — tables, JSON, the full script body, the exact command — not just prose descriptions.
5. **NEVER modify Quark's own source code, examples, or tutorials.** The Quark repo (`quark/`, `examples/`, `tutorials/`, `tools/`, `docs/`, `tests/`) is read-only from this workflow's perspective. See *Upstream Quark Code is Read-Only* below.
6. **NEVER silently fall back to CPU execution.** If the user requested CUDA/ROCm and the provider is unavailable, stop and surface the gap to `quark-onnx-install` / `quark-onnx-debug` — do not quietly degrade.

## Upstream Quark Code is Read-Only

The Quark repository is the upstream source of truth. This workflow may read it freely (config sources, example scripts, tutorial notebooks, custom-op headers) but must not write into it. That includes:

- No edits to `quark/` package source.
- No edits to anything under `examples/onnx/` — including `examples/onnx/yolo_quantization/quantize_yolo.py`. Even small "just to make the script accept my preprocessing" patches are forbidden, because they make the run irreproducible against a clean Quark install.
- No edits to anything under `tutorials/onnx/` — the Ryzen AI tutorials for YOLOv8 / ResNet-50 / image classification are reference material only.
- No edits to `tools/`, `docs/`, `tests/`, or `pyproject.toml` / `requirements.txt`.

**If a shipped example or tutorial as-is does not cover what the user needs** (different model, different preprocessing, different exclude list, different calibration source), write a fresh standalone script in the user's working directory (or `/tmp/`) that imports from `quark.onnx`. Pattern:

```python
# user_workspace/my_onnx_ptq.py — NOT inside the Quark repo
from quark.onnx import ModelQuantizer, QConfig, QLayerConfig, XInt8Spec, CLEConfig
from onnxruntime.quantization.calibrate import CalibrationDataReader
# ... user-specific data reader + QConfig the shipped example doesn't cover ...
```

Then reference *that* script in the `run_manifest.yaml` instead of a modified example. The shipped scripts stay untouched, the user's customization is local to their workspace, and the run remains reproducible against any Quark version.

If the user actually needs an upstream Quark change (a new preset, a new algorithm config), surface it as a Quark contribution — do not silently patch their local checkout.

## Required Artifact Flow

```text
Step 1 (Intake)   ──► model_analysis.json
Step 2 (Plan)     ──► quant_plan.json
Step 3 (Manifest) ──► run_manifest.yaml  +  user_workspace/<name>_ptq.py
Step 4 (Execute)  ──► quantized .onnx (+ .onnx_data if external)  (only after user says yes)
```

---

## Step 1: Model Intake

**Goal**: Analyze the `.onnx` graph and produce `model_analysis.json`. Hand off to `quark-onnx-model-intake`.

### Actions

1. **Confirm the input path.** It must end in `.onnx`. If a sibling `.onnx_data` exists, both must move together.
2. **Run `quark-onnx-model-intake`.** That skill produces:
   - opset / IR version
   - input / output names, shapes, dtypes
   - op-type histogram and quantizable-op count
   - whether QDQ nodes are already present (if yes — model is already quantized; warn and stop)
   - external-data status (model.onnx_data present? total bytes? >2 GB?)
   - deployment-target compatibility (CPU / CUDA / ROCm / NPU CNN / NPU Transformer)
3. **Surface risks:**
   - Already quantized → stop, do not re-quantize.
   - opset < 13 → recommend upgrading to opset 17+ for best support.
   - Dynamic input shapes → calibration data reader must produce matching shapes; warn if shapes vary.
   - >2 GB model → external data must be handled; set `use_external_data_format=True` later.
   - NPU CNN target → only XINT8 / A8W8 are gated as supported; warn early.

### Output to Show User

Present a summary table:

```text
Model Analysis:
  Model path:       ./models/yolov8n.onnx
  Opset / IR:       17 / 8
  Input:            images, [1, 3, 640, 640], float32
  Output:           output0, [1, 84, 8400], float32
  Total ops:        ~226 nodes (Conv, MatMul, Add, Mul, Sigmoid, Concat, …)
  Quantizable ops:  ~118 (Conv + MatMul + …)
  QDQ already?      No
  External data:    No (6.2 MB inline)
  Risks:            None
  NPU CNN target:   Compatible (Conv-heavy, no unsupported ops)
```

### >>> CHECKPOINT 1: Confirm model analysis is correct before continuing

---

## Step 2: Quantization Plan

**Goal**: Build `quant_plan.json` from the model analysis and user's stated preferences. Hand off to `quark-onnx-quant-plan`.

### Actions

1. **Determine the preset.** If the user stated one (e.g., "XINT8"), use it. Otherwise, recommend based on deployment target + priority:

   | Deployment | Priority | Recommended Preset | Algorithm |
   |------------|----------|--------------------|-----------|
   | AMD NPU CNN (Ryzen AI) | Best accuracy | `XINT8` + `EnableNPUCnn=True` | `CLE` (optional `AdaRound`) |
   | AMD NPU Transformer | Best accuracy | `BFP16` | none |
   | CPU general | Smallest model | `A8W8` | `CLE` (optional) |
   | CPU general | Best accuracy | `A16W8` | `MinMax` |
   | CUDA / ROCm | Best accuracy | `BF16` | none |
   | CUDA / ROCm | High throughput | `BFP16` / `MXFP4` | none |
   | Any | Recover lost accuracy | + `AdaRound` / `AdaQuant` | as additional algorithm |

2. **Fill the decision table.** Show ALL decisions with defaults:

   | Decision | Value | Reason |
   |----------|-------|--------|
   | `preset` | `XINT8` | User requested XINT8 (NPU CNN) |
   | `activation_spec` | `XInt8Spec()` | Matches preset |
   | `weight_spec` | `XInt8Spec()` | Matches preset |
   | `calibration_method` | `MinMax` (default) | Standard for XINT8 |
   | `algo_config` | `[CLEConfig()]` | Improves XINT8 accuracy on Conv networks |
   | `EnableNPUCnn` | `True` | XINT8 + NPU CNN deployment |
   | `use_external_data_format` | `False` | Model < 2 GB |
   | `exclude_nodes` / `exclude_subgraphs` | `[]` | None identified |
   | `calibration_data_path` | `./calib_data/` | User-provided folder of representative samples |
   | `num_calib_data` | `100` | Standard default for vision |
   | `batch_size` | `1` | Safe default |
   | `evaluation_intent` | `smoke` | Quick mAP / Prec@1 check after quantization |

3. **Map deployment-target gates explicitly.** Note any preset+target combinations that are unsupported (e.g., BFP16 on NPU CNN). Surface them, do not silently change the user's choice.

4. **Ask the user if they want to change anything.**

### >>> CHECKPOINT 2: User MUST confirm or adjust the plan before continuing

Wait for the user to say "ok", "confirm", "looks good", "continue", or similar. If they request changes (e.g., "add AdaRound", "raise calib data to 500"), update the table and re-present.

---

## Step 3: Manifest Generation

**Goal**: Translate the confirmed plan into a runnable standalone script in the user's working directory and a `run_manifest.yaml`. Do not write into the Quark repo.

### Actions

1. **Decide the script path.** Place it in the user's working directory, e.g. `./<model_name>_ptq.py`. Never write under `examples/onnx/` or `tutorials/onnx/`.

2. **Build the script body.** Three pieces:

   a. **`CalibrationDataReader`** — pick the right reader for the input modality. Vision models follow the pattern from `tutorials/onnx/ryzen_ai/yolov8/` and `tutorials/onnx/image_classification/`:

      ```python
      from onnxruntime.quantization.calibrate import CalibrationDataReader
      import onnxruntime as ort
      import numpy as np, os, cv2

      class ImageDataReader(CalibrationDataReader):
          def __init__(self, calib_folder, model_path, hw=(640, 640)):
              sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
              self.input_name = sess.get_inputs()[0].name
              self.data = self._load(calib_folder, *hw)
              self.iter = None
          def _load(self, folder, h, w):
              out = []
              for f in sorted(os.listdir(folder)):
                  if not f.lower().endswith((".jpg", ".jpeg", ".png")): continue
                  img = cv2.imread(os.path.join(folder, f))
                  img = cv2.resize(img, (w, h))
                  arr = img.transpose(2, 0, 1).astype(np.float32) / 255.0
                  out.append(np.expand_dims(arr, 0))
              return out
          def get_next(self):
              if self.iter is None:
                  self.iter = iter([{self.input_name: d} for d in self.data])
              return next(self.iter, None)
          def rewind(self): self.iter = None
      ```

   b. **`QConfig`** built from the plan:

      ```python
      from quark.onnx import (
          ModelQuantizer, QConfig, QLayerConfig,
          XInt8Spec, Int8Spec, Int16Spec, BFloat16Spec, BFP16Spec,
          CLEConfig, AdaRoundConfig, AdaQuantConfig, CalibMethod,
      )
      activation_spec = XInt8Spec()
      weight_spec     = XInt8Spec()
      algo_config     = [CLEConfig()]
      config = QConfig(
          global_config=QLayerConfig(activation=activation_spec, weight=weight_spec),
          algo_config=algo_config,
          EnableNPUCnn=True,
          use_external_data_format=False,
          exclude=[],
      )
      ```

   c. **Driver** that wires the data reader + config + I/O paths:

      ```python
      dr = ImageDataReader("./calib_data", "./models/yolov8n.onnx")
      ModelQuantizer(config).quantize_model(
          "./models/yolov8n.onnx",
          "./models/yolov8n_xint8.onnx",
          dr,
      )
      ```

3. **Write the script to disk** at the chosen path. Print the *full* script body back to the user so they can see exactly what will run — never just say "generated a script".

4. **Build the exact command:**

   ```bash
   python3 ./yolov8n_ptq.py
   ```

5. **Show expected output layout:**

   ```text
   ./models/
     ├── yolov8n.onnx                  (input, unchanged)
     └── yolov8n_xint8.onnx            (quantized output)
   ./yolov8n_ptq.py                    (generated by this workflow)
   ./run_manifest.yaml                 (this workflow's manifest)
   ```

   If `use_external_data_format=True`, also expect `yolov8n_xint8.onnx_data` next to the `.onnx`.

### >>> CHECKPOINT 3: Show the script body and the command, then ask "shall I run this?"

Do NOT proceed to execution unless the user explicitly confirms. Acceptable confirmations: "yes", "run it", "go", "execute", or similar.

If the user says "no" or wants changes, go back to the relevant step (most commonly back to Step 2 to change preset / algorithm / exclude lists).

---

## Step 4: Execute PTQ

**Goal**: Run the generated script and report results.

### Precondition

**This step runs ONLY after the user explicitly confirms in Step 3.**

### Actions

1. **Create the output directory if needed:**

   ```bash
   mkdir -p ./models
   ```

2. **Run the generated script.** Monitor for the common ONNX-side failures:
   - ONNX Runtime EP not available → hand off to `quark-onnx-install` / `quark-onnx-debug`.
   - Custom-op library load failure (`BFPQuantizeDequantize`, `MXQuantizeDequantize`, `Extended*`) → hand off to `quark-onnx-debug`.
   - OOM during calibration → suggest reducing `num_calib_data`, reducing batch size, or moving calibration to CPU (`OptimDevice="cpu"`).
   - External-data not found → confirm `.onnx_data` sits next to `.onnx`.
   - AdaRound / AdaQuant diverged → suggest `EarlyStop=True`, lower learning rate, more iterations.
   - Calibration shapes mismatch → confirm the data reader produces the model's declared input shape.

3. **After completion, verify outputs exist:**

   ```bash
   ls -lh ./models/yolov8n_xint8.onnx*
   ```

4. **Report results:**

   ```text
   Quantization complete:
     Output:          ./models/yolov8n_xint8.onnx
     Model size:      ~3.5 MB (input was 12.3 MB → ~3.5× smaller)
     Format:          ONNX (QDQ inserted, com.amd.quark custom ops where applicable)
     External data:   No
   ```

### Error Recovery

- If quantization fails, do NOT retry blindly. Report the error verbatim and hand off to `quark-onnx-debug`.
- For OOM, suggest: reduce `num_calib_data` first, then drop `batch_size` to 1, then move calibration to CPU.
- For EP issues, do not "fix" by silently switching providers — surface the gap and let `quark-onnx-install` resolve it.

---

## Complete Example

⭐ **Recommended starting point for new users.**

The full end-to-end walkthrough — XINT8 quantization of **YOLOv8n** for AMD NPU CNN deployment, calibrated on COCO val2017 — is in **[`example-xint8-yolov8n.md`](example-xint8-yolov8n.md)** sitting next to this SKILL.md. It shows every checkpoint (intake → plan → manifest → execute) with real numbers, the generated script, and the resulting quantized model layout. Mirror it for your own CNN.

## Recovery

- If any upstream artifact is missing, stop and name the missing producer skill. Do not improvise a partial artifact.
- If the workflow reaches a blocked state (e.g., model already quantized, preset incompatible with deployment target, calibration data not in the model's input shape), report the blocker and suggest the specific fix.
- If execution fails, report the error with diagnostic context and hand off to `quark-onnx-debug` rather than attempting ad-hoc patches.
- If the user wants to change a decision mid-workflow (e.g., switch from XINT8 to BFP16 after seeing model analysis), go back to the relevant step — do not restart from scratch.
- If the user pasted a Torch traceback or a HuggingFace path, do not try to repair-route — hand off to `the matching public quark-torch skill` and stop.
