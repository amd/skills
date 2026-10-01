# Complete example: FP8 PTQ for Qwen3-8B

This walkthrough shows `quark-torch-ptq` using its local workflow references
and the installed `quark-cli`. Values discovered from model configuration are
shown as run-specific evidence, not universal constants.

```text
User: "Quantize Qwen/Qwen3-8B with FP8 and write it to
       ./output/qwen3-8b-fp8."

Step 1 — Model intake

Read the model configuration without loading weights and write
model_analysis.json:

  Model source:      Qwen/Qwen3-8B
  Model type:        qwen3
  Hidden layers:     <value read from the current config>
  Quantized linears: <estimated value and method>
  MoE:               <value read from config>
  Default excludes:  <values from installed Quark's qwen3 template>
  Compatibility:     <installed/model version evidence>

CHECKPOINT 1: "Is this model analysis correct?"

Step 2 — Quantization plan

  | Decision            | Value       | Reason                         |
  |---------------------|-------------|--------------------------------|
  | global_scheme       | fp8         | User requested FP8             |
  | kv_cache_scheme     | null        | Not requested                  |
  | exclude_layers      | <template>  | Preserve model-specific default|
  | layer_quant_config  | {}          | No per-pattern override        |
  | algorithm           | null        | RTN baseline                   |
  | dataset             | pileval     | CLI-supported default          |
  | num_calib_data      | 512         | Current CLI default            |
  | seq_len             | 512         | Current CLI default            |
  | export              | hf_format   | Hugging Face model output      |
  | evaluation_intent   | none        | Stop at quantized output       |
  | trust_remote_code   | false       | Safer unless explicitly needed |

Write quant_plan.json with requires_confirmation=true.

CHECKPOINT 2: "Confirm this plan or tell me what to change."

After confirmation, set requires_confirmation=false.

Step 3 — Manifest and exact command

Resolve the requested relative destination to an absolute path. Write
run_manifest.yaml and show:

  quark-cli torch-llm-ptq \
    --model_dir "Qwen/Qwen3-8B" \
    --output_dir "/absolute/path/to/workspace/output/qwen3-8b-fp8" \
    --quant_scheme "fp8" \
    --num_calib_data "512" \
    --seq_len "512" \
    --device cuda \
    --no_trust_remote_code \
    --skip_evaluation

CHECKPOINT 3: "Shall I run this exact command?"

Step 4 — Execute and verify

Only after an explicit "yes", create the output directory and run the command.
Inspect the actual output instead of assuming filenames, size, or metrics:

  Status:       <exit status>
  Output:       /absolute/path/to/workspace/output/qwen3-8b-fp8/
  Model files:  <observed safetensors/config/tokenizer files>
  Total size:   <measured value>
  Format:       Hugging Face SafeTensors

Update run_manifest.yaml with observed results.

CHECKPOINT 4: "Does this verified output meet your request?"
```

If this model requires custom remote code, stop during intake, explain the
risk, and omit `--no_trust_remote_code` only after approval. If the user
changes calibration or requests KV-cache quantization, return to Checkpoint 2
and regenerate the plan and command.
