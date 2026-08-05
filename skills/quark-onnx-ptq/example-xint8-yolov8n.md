# Complete Example: XINT8 Quantization of YOLOv8n for AMD NPU CNN

End-to-end walkthrough of `quark-onnx-ptq` for an XINT8 PTQ run on the YOLOv8n object-detection model, targeting AMD NPU CNN (Ryzen AI) deployment. Calibration source: COCO `val2017`.

The workflow generates a standalone script in the user's working directory (`./yolov8n_ptq.py`) and never modifies the shipped `examples/onnx/yolo_quantization/quantize_yolo.py` or the `tutorials/onnx/ryzen_ai/yolov8/` notebook — both stay read-only references.

```text
User: "Quantize ./models/yolov8n.onnx with XINT8 for Ryzen AI deployment,
      calibrate on ./calib_data (COCO val2017 subset), output to ./models/yolov8n_xint8.onnx"

─── Step 1: Model Intake ───
[Run quark-onnx-model-intake against ./models/yolov8n.onnx]

Model Analysis:
  Model path:       ./models/yolov8n.onnx
  Opset / IR:       17 / 8
  Input:            images, [1, 3, 640, 640], float32
  Output:           output0, [1, 84, 8400], float32
  Total ops:        ~226 nodes (Conv, MatMul, Add, Mul, Sigmoid, SiLU, Concat, …)
  Quantizable ops:  ~118 (Conv + MatMul + …)
  QDQ already?      No
  External data:    No (12.3 MB inline)
  Risks:            None
  NPU CNN target:   Compatible (Conv-heavy, no unsupported ops)

>>> Does this look correct? (confirm to continue)

─── Step 2: Quantization Plan ───
[Run quark-onnx-quant-plan with the analysis + user's stated XINT8 + NPU CNN target]

Decision Table:
  | Decision                   | Value             | Reason                            |
  |----------------------------|-------------------|-----------------------------------|
  | preset                     | XINT8             | User requested XINT8 (NPU CNN)    |
  | activation_spec            | XInt8Spec()       | Matches preset                    |
  | weight_spec                | XInt8Spec()       | Matches preset                    |
  | calibration_method         | MinMax (default)  | Standard for XINT8                |
  | algo_config                | [CLEConfig()]     | Improves XINT8 accuracy on Conv   |
  | EnableNPUCnn               | True              | XINT8 + NPU CNN deployment        |
  | use_external_data_format   | False             | Model < 2 GB                      |
  | exclude_nodes              | []                | None identified                   |
  | calibration_data_path      | ./calib_data      | User-provided COCO val2017 subset |
  | num_calib_data             | 100               | Standard default for vision       |
  | batch_size                 | 1                 | Safe default                      |
  | evaluation_intent          | smoke             | Quick mAP check after quant       |

>>> Confirm this plan? Any changes? (e.g. add AdaRound, raise num_calib_data, add exclude_nodes)

─── Step 3: Manifest ───
[Generate ./yolov8n_ptq.py and ./run_manifest.yaml in the user's working directory]

Generated script (./yolov8n_ptq.py) — full body shown:

  import os
  import cv2
  import numpy as np
  import onnxruntime as ort
  from onnxruntime.quantization.calibrate import CalibrationDataReader

  from quark.onnx import (
      ModelQuantizer, QConfig, QLayerConfig,
      XInt8Spec, CLEConfig,
  )

  INPUT_MODEL  = "./models/yolov8n.onnx"
  OUTPUT_MODEL = "./models/yolov8n_xint8.onnx"
  CALIB_FOLDER = "./calib_data"
  NUM_CALIB    = 100
  HW           = (640, 640)

  class YoloImageDataReader(CalibrationDataReader):
      def __init__(self, calib_folder, model_path):
          sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
          self.input_name = sess.get_inputs()[0].name
          self.data = self._load(calib_folder)
          self.iter = None
      def _load(self, folder):
          h, w = HW
          out = []
          files = sorted(f for f in os.listdir(folder)
                         if f.lower().endswith((".jpg", ".jpeg", ".png")))
          for f in files[:NUM_CALIB]:
              img = cv2.imread(os.path.join(folder, f))
              img = cv2.resize(img, (w, h))
              arr = img.transpose(2, 0, 1).astype(np.float32) / 255.0
              out.append(np.expand_dims(arr, 0))
          return out
      def get_next(self):
          if self.iter is None:
              self.iter = iter([{self.input_name: d} for d in self.data])
          return next(self.iter, None)
      def rewind(self):
          self.iter = None

  def main():
      activation_spec = XInt8Spec()
      weight_spec     = XInt8Spec()
      config = QConfig(
          global_config=QLayerConfig(activation=activation_spec, weight=weight_spec),
          algo_config=[CLEConfig()],
          EnableNPUCnn=True,
          use_external_data_format=False,
          exclude=[],
      )
      dr = YoloImageDataReader(CALIB_FOLDER, INPUT_MODEL)
      ModelQuantizer(config).quantize_model(INPUT_MODEL, OUTPUT_MODEL, dr)
      print(f"Wrote {OUTPUT_MODEL}")

  if __name__ == "__main__":
      main()

Command:
  python3 ./yolov8n_ptq.py

Expected output layout:
  ./models/
    ├── yolov8n.onnx                  (input, unchanged)
    └── yolov8n_xint8.onnx            (quantized output, ~3.5 MB)
  ./yolov8n_ptq.py                    (generated by this workflow)
  ./run_manifest.yaml                 (this workflow's manifest)

>>> Shall I run this?

─── Step 4: Execute ───
[Running quantization...]
[CLE pass complete]
[Calibration: 100/100 samples]
[QDQ insertion complete]
[Wrote ./models/yolov8n_xint8.onnx]

Quantization complete:
  Output:          ./models/yolov8n_xint8.onnx
  Model size:      ~3.5 MB (input was 12.3 MB → ~3.5× smaller)
  Format:          ONNX (QDQ inserted, com.amd.quark custom ops where applicable)
  External data:   No

For accuracy numbers (mAP@0.5, mAP@0.5:0.95), run the YOLOv8 evaluator separately —
this workflow does not run end-task accuracy benchmarks. Expected ballpark per
tutorials/onnx/ryzen_ai/yolov8/onnx_ryzen_ai_yolov8_tutorial.ipynb:

  | Metric        | Float    | Quantized |
  |---------------|----------|-----------|
  | Model Size    | 12.26 MB | ~3.5 MB   |
  | mAP@0.5:0.95  | 0.265    | ~0.206    |
  | mAP@0.5       | 0.341    | ~0.269    |
```

## Variants you can request mid-workflow

- **Different preset** (e.g. `A8W8`, `A16W8`, `BF16`, `BFP16`) — restate at Step 2; the planner will swap `*Spec()` and any `EnableNPUCnn` / extras (`AlignSlice`, `FoldRelu`, `AlignConcat`, `AlignEltwiseQuantType`).
- **Recover lost accuracy** — add `AdaRoundConfig(...)` or `AdaQuantConfig(...)` to `algo_config` at Step 2; the planner will pick reasonable `data_size` / `num_iterations` / `learning_rate` defaults (mirrors `DEFAULT_ADAROUND_PARAMS` in `tutorials/onnx/ryzen_ai/yolov8/`).
- **Exclude specific nodes** (e.g. detection head Concats) — pass `exclude_nodes="/model.22/Concat_5"` at Step 2; the planner adds them to the script's `exclude=[...]` list. Subgraph exclusions follow the `([start_nodes], [end_nodes])` form from `examples/onnx/yolo_quantization/quantize_yolo.py`.
- **YOLO-NAS / YOLOX instead of YOLOv8** — same workflow, different input model and matching data reader. The shipped `examples/onnx/yolo_quantization/quantize_yolo.py` covers the YOLO-NAS / YOLOX export and dataset-loader patterns; the generated script imports `quark.onnx` directly rather than patching that file.
- **Bigger model (>2 GB)** — at Step 2 set `use_external_data_format=True`; the generated script will write `*.onnx_data` alongside the `.onnx`.
