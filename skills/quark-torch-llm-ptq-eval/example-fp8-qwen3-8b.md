# Complete Example: End-to-End FP8 PTQ of Qwen3-8B

End-to-end walkthrough of `quark-torch-llm-ptq-eval` — quantize → validate → evaluate.
Phase 1 is the `quark-torch-ptq` workflow run verbatim; Phases 2 and 3 chain the
validation and evaluation skills.

```text
User: "Quantize Qwen/Qwen3-8B with FP8 and check accuracy, output to ./output/qwen3-8b-fp8"

═══ Phase 1: Quantize (quark-torch-ptq) ═══

─── Step 1: Model Intake ───
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
  Perplexity:  9.72

═══ Phase 2: Validate (auto, mandatory) ═══
[Run quark-torch-result-validator]

Validation:
  fuzzy layout : ok  (252 linear weights = F8_E4M3; scales BF16; lm_head/norms BF16)
  MD5 spot-chk : ok  (lm_head.weight byte-identical to source)
  config.json  : ok
  aux files    : ok
All checks passed.

═══ Phase 3: Accuracy Eval (optional, ROCm only) ═══
>>> Run accuracy evaluation? (gsm8k / mmlu / ... / no)
User: "gsm8k"

[Serve with vLLM, run lm_eval gsm8k]

gsm8k (5-shot, flexible-extract): 84.6%
  vs Qwen3-8B BF16 ~85% -> <1% delta, FP8 no meaningful accuracy loss.
```
