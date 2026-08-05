# Complete Example: FP8 Quantization of Qwen3-8B

Walkthrough of `quark-torch-ptq` for an FP8 PTQ run — model intake
through confirmed execution. For an end-to-end run that also validates and
evaluates the quantized model, see the `quark-torch-llm-ptq-eval` recipe.

```text
User: "Quantize Qwen/Qwen3-8B with FP8, output to ./output/qwen3-8b-fp8"

─── Step 1: Model Intake ───
[Run python3 -c "from transformers import AutoConfig; ..." to read config]

Model Analysis:
  Model path:       Qwen/Qwen3-8B
  Model type:       qwen3
  Hidden layers:    36
  Linear layers:    ~252
  MoE:              No
  Exclude defaults: [lm_head]
  Risks:            None
  Compatibility:    OK

>>> Does this look correct? (confirm to continue)

─── Step 2: Quantization Plan ───

Decision Table:
  | Decision          | Value      | Reason                    |
  |-------------------|------------|---------------------------|
  | global_scheme     | fp8        | User requested FP8        |
  | kv_cache_scheme   | fp8        | Recommended for FP8       |
  | exclude_layers    | [lm_head]  | Standard                  |
  | algorithm         | null       | RTN baseline              |
  | num_calib_data    | 128        | Standard default          |
  | seq_len           | 512        | Standard default          |

>>> Confirm this plan? Any changes?

─── Step 3: Manifest ───

Command:
  python3 /path/to/quantize_quark.py \
    --model_dir Qwen/Qwen3-8B \
    --output_dir ./output/qwen3-8b-fp8 \
    --quant_scheme fp8 \
    --kv_cache_dtype fp8 \
    --num_calib_data 128 \
    --seq_len 512 \
    --model_export hf_format \
    --data_type auto \
    --device cuda

>>> Shall I run this?

─── Step 4: Execute ───
[Running quantization...]
[Done]

Quantization complete:
  Output:      ./output/qwen3-8b-fp8/
  Model size:  8.8 GB
  Format:      HuggingFace SafeTensors
  Perplexity:  9.73 (wikitext)
```
