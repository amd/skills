# Troubleshooting: Error Fixes + ROCm Environment Variables

Reference for diagnosing launch failures, smoke test failures, and accuracy anomalies on AMD ROCm.

> **Backend coverage — vLLM only (today).** The error-signature table and detailed Issues below cover vLLM symptoms exclusively. SGLang and ATOM run through the same launch / readiness / smoke flow, but their failure surfaces are not yet enumerated here. SGLang/ATOM-specific failures fall through to "search GitHub Issues"; SGLang and ATOM coverage will be added in a follow-up.

**Important: this file is a troubleshooting reference, not a prevention checklist.**

- Do not preemptively apply fixes during launch
- Consult only when an error is actually encountered
- For fixes that modify source code or model config.json, **show the full diff to the user and wait for approval** before applying
- **Before modifying any file** (vLLM source or model `config.json`): `cp <file> <file>.bak.$(date +%Y%m%d_%H%M%S)`. Record `{path, backup, original_value}` under `source_edits` in `$EVAL_STATE_DIR/current-eval.yaml`. Phase 7 cleanup offers rollback.

## How to Use

1. Get the full server log or Python traceback
2. Locate the **core error line** (`AssertionError` / `AttributeError` / `Memory access fault` / `ValueError` / `KeyError` etc.)
3. Match against the error signature table below
4. Not in the table → search vLLM/SGLang GitHub Issues; same approval rule applies

## Error Signature → Fix Lookup Table

| Error signature (core line) | Root cause | Fix |
|---|---|---|
| `'CustomOp' has no attribute 'op_registry'` | aiter in emulation mode | `export VLLM_ROCM_USE_AITER=1` before launch |
| `Memory access fault by GPU node-<N>` with no traceback | CUDA Graph compilation failure | Add `--enforce-eager` to launch command |
| `AssertionError` + `param_data.shape != loaded_weight.shape` + involves `q_proj/k_proj/v_proj` | vLLM cannot unpack QKV weights | **Fix B (modify vLLM source)**: add `packed_modules_mapping = {"qkv_proj": ["q_proj","k_proj","v_proj"]}` to the model class |
| `AssertionError` + `param_data.shape != loaded_weight.shape` + involves `gate_proj/up_proj` | vLLM cannot unpack gate_up weights | **Fix A (modify vLLM source)**: add `packed_modules_mapping = {"gate_up_proj": ["gate_proj","up_proj"]}` to the model class |
| `weight key not found` / `unexpected weight name` | Checkpoint has weight names vLLM does not recognize | **Fix C (modify vLLM source)**: add `if name not in params_dict: continue` in weight loader |
| `AttributeError: '<X>Config' object has no attribute 'text_config'` | model_type not registered or mismatched in vLLM | **Fix D (modify model config.json)**: change `model_type` to a supported close relative |
| `ValueError: ... KV cache memory ... larger than the available KV cache memory` | KV cache cannot fit max_model_len | Add `--max-model-len <N>` (≤ the estimated value in the error; commonly 8192/16384) |
| `AssertionError: Aiter MLA only supports 16 or 128 number of heads. Provided <N>` | Per-card head count after TP split not in {16, 128} | Adjust `--tensor-parallel-size` so that `num_attention_heads / TP ∈ {16, 128}` |
| Model launches but lm-eval accuracy = 0 | Likely silent CUDA Graph failure producing garbage output | Add `--enforce-eager` (see Issue 3) |
| `Address already in use` | Port occupied by another process | `lsof -i :<port>` to identify; stop or use different port (see Issue 10) |
| lm-eval `404 Not Found` / `Model not found` | Model name mismatch between server and eval | Verify via `/v1/models` endpoint (see Issue 12) |
| `NotImplementedError: Loglikelihood is not supported for chat completions` | loglikelihood task (mmlu, mmlu_pro, etc.) used with `local-chat-completions` | Switch to `local-completions` + completions API (see Issue 13) |
| vLLM `error: unrecognized arguments` then health poll times out at 300s | vLLM exited immediately but readiness loop only checks health, not process liveness | Add `pgrep` liveness check to the polling loop (see Issue 14) |
| `Context length (N) + continuation length (M) > max_length (2047). Left truncating context` | lm-eval default `max_length=2048` truncating few-shot prompts on long-prompt loglikelihood tasks (mmlu, mmlu_pro) | Add `max_length=<max_position_embeddings>` to `--model_args` (see Issue 15) |

## ROCm Environment Variables

These variables are for fixing errors, not a default-enable checklist. Do not preemptively add them during launch; apply only when an error signature matches.

| Variable | Effect | When to add |
|---|---|---|
| `VLLM_ROCM_USE_AITER=1` | Enable native aiter kernel | `'CustomOp' has no attribute 'op_registry'`; MoE model accuracy abnormally low |
| `VLLM_ROCM_USE_AITER=0` | Fall back to emulation kernel | Native aiter kernel causes GPU hang |
| `VLLM_ATTENTION_BACKEND="TRITON_MLA"` | Switch to Triton MLA attention backend | MLA models (DeepseekV2/V3, Kimi-K2, etc.) with accuracy anomalies |
| `VLLM_ROCM_USE_AITER_MOE=1` | Enable fused MoE kernel | MoE model performance is poor. Note: auto-enabled when `USE_AITER=1` |
| `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0` | Disable shared expert fusion | Shared expert fusion causes errors or wrong output |
| `AMD_SERIALIZE_KERNEL=3` | Serialize kernel execution | Debugging only. When `Memory access fault` has no traceback, add this + `--enforce-eager` — next run will likely show the real traceback |
| `HIP_VISIBLE_DEVICES=<list>` | Specify which GPUs to use | Required on every launch, determined by GPU auto-discovery |

**How to add:**

```bash
# Method 1: export then start server
export VLLM_ROCM_USE_AITER=1
vllm serve <path> ...

# Method 2: inline
HIP_VISIBLE_DEVICES=6 VLLM_ROCM_USE_AITER=1 vllm serve <path> ...
```

Record environment variables in `$EVAL_STATE_DIR/server-runtime.md` for reproducibility.

### About AITER

AITER (AMD Iterative Engine for ROCm) is AMD's optimized kernel library for vLLM. Two modes:

- **Native** (`VLLM_ROCM_USE_AITER=1`): true AMD optimized kernels, better performance, but may cause hangs on some models
- **Emulation** (`VLLM_ROCM_USE_AITER=0` or unset): CPU/generic implementation for compatibility, but MoE models are prone to accuracy issues

For new models where the model card does not specify: launch per model card instructions. If issues arise, consult the variable table above.

### Low Accuracy Troubleshooting Order

This diagnostic ordering moved to `eval-frameworks.md` §Low-accuracy diagnostic order — that file is loaded in Phase 6 where accuracy issues actually surface.

## Detailed Cases

---

### Issue 1: Model loads but hangs during compilation

**Symptom:**

```text
(Worker_TP1 pid=1907) [QUARK-INFO]: C++ kernel compilation is already complete.
Memory access fault by GPU node-6 (Agent handle: 0x1fc63bc0) on address 0x7f37b4e01000. Reason: Unknown.
```

**Fix:** Add `--enforce-eager` to the launch command.

**Deep debugging (if enforce-eager is not enough):**

1. Add `--enforce-eager` to rule out CUDA Graph interference
2. `export AMD_SERIALIZE_KERNEL=3` to serialize kernel execution
3. Re-run — you will likely see a Python traceback pointing to `dequant_mxfp4()` / `mx.dq_mxfp4()`
4. Add `print + torch.cuda.synchronize()` around the crash line to bisect
5. Print weight shapes to check if dequant output exceeds int32 (>2^31-1) — if so, reduce TP or switch kernel

Historical finding: at TP=4, single worker dequant output `[96, 4096, 7168]` = 2.8B elements > int32_max; at TP=8, `[48, 4096, 7168]` = 1.4B — safe. This class of issue requires TP adjustment.

---

### Issue 2: TP=8 causes tensor_parallel_size error

**Symptom (core line):**

```text
AssertionError: Aiter MLA only supports 16 or 128 number of heads. Provided 8 number of heads.
Try adjusting tensor_parallel_size value.
```

**Fix:** Aiter MLA backend requires per-card head count ∈ {16, 128}. `num_attention_heads / TP` must land on one of these values.

- Model with 64 attn heads: TP=4 → 16 per card (OK); TP=8 → 8 per card (error)
- Solution: use TP=4

Note: this constraint only applies to **MLA models** (DeepseekV2/V3, Kimi-K2, etc.); GQA models (MiniMax-M2, Llama) are not affected.

---

### Issue 3: vLLM starts successfully but lm-eval accuracy = 0

**Symptom:**

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |    0|±  |     0|
```

**Fix:** Add `--enforce-eager`. This is a silent compilation error (CUDA Graph corrupts model output without reporting an error), manifesting as the server appearing normal but producing gibberish or empty output.

**Prevention:** The smoke test quality gate should catch this before benchmarking. If this appears at evaluation time, the gate was not strict enough.

---

### Issue 4: Weight loading shape mismatch

**Symptom (core line):**

```text
File ".../vllm/model_executor/parameter.py", line 200, in load_qkv_weight
    assert param_data.shape == loaded_weight.shape
AssertionError
```

**Debug method:** Add context to the assert for diagnosis:

```python
assert param_data.shape == loaded_weight.shape, \
    f"{param_data.shape},{loaded_weight.shape},{self.prefix}"
```

The prefix (e.g., `model.layers.71.mlp.shared_experts.gate_up_proj`) identifies which packed module is problematic.

**Locate the model file first** (works for any vLLM install path; export `VLLM_ROOT` or use `python -c "import vllm, os; print(os.path.dirname(vllm.__file__))"`):

```bash
# ARCH from config.json's "architectures": e.g., MiniMaxM2ForCausalLM
grep -rn "class ${ARCH}\b" "${VLLM_ROOT}/model_executor/models/" || \
  grep -rln "${ARCH}" "${VLLM_ROOT}/model_executor/models/"
```

Edit the matched file. Always `cp <file> <file>.bak.$(date +%Y%m%d_%H%M%S)` first and append the entry to `source_edits` in `current-eval.yaml`.

**Fix A (gate/up mismatch, e.g., `[512,3072] vs [512,6144]`) — modify vLLM source:**

```python
class ${ARCH}(nn.Module, ...):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
```

**Fix B (QKV mismatch) — modify vLLM source:**

```python
class ${ARCH}(nn.Module, ...):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
```

**Fix C (weight name not in dict) — modify vLLM source:**

```python
for name, loaded_weight in weights:
    ...
    if name not in params_dict:
        continue
    ...
```

**Reminder:** Show the full diff to the user before modifying source code.

---

### Issue 5: model_type not registered

**Symptom (core line):**

```text
AttributeError: 'DeepseekOCRConfig' object has no attribute 'text_config'.
Did you mean: 'get_text_config'?
```

**Fix D (modify model config.json):** Change `model_type` to a supported close relative:

```diff
 // <model_path>/config.json
-"model_type": "DeepseekOCR"
+"model_type": "deepseek_vl_v2"
```

**Reminder:** Show the diff to the user before modifying config.json — this modifies the user's model files.

**When to use this fix vs modifying vLLM source:** If vLLM already supports a very similar architecture, editing config.json is a low-risk quick path. If a completely new architecture needs support, adding a new model file in vLLM is a large effort beyond this skill's scope.

---

### Issue 6: max_len OOM

**Symptom (core line):**

```text
ValueError: To serve at least one request with the model's max seq len (262144),
17.16 GiB KV cache is needed, which is larger than the available KV cache memory (3.46 GiB).
Based on the available memory, the estimated maximum model length is 52816.
```

**Fix:** Add `--max-model-len <N>` to the launch command, where N ≤ the `estimated maximum model length` from the error. Common values: 8192, 16384.

**Detection:** Error text contains `max seq len` + `KV cache memory` → always this issue.

---

### Issue 7: Eval framework dependency missing

`ModuleNotFoundError: No module named '<X>'` → `pip install <X>`. Common: `tenacity`, `datasets`, `sacrebleu`.

---

### Issue 8: lm-eval accuracy differs from reported results

Cross-check these three against the model card's eval setup: (1) `--num_fewshot` value, (2) exact task name (`mmlu` vs `mmlu_direct` vs `mmlu_fewshot`), (3) model type match (chat vs base template).

---

### Issue 9: HumanEval / code generation benchmark fails

Add `--allow_code_execution` to lm_eval. Install dependency: `pip install human-eval`.

---

### Issue 10: Port already in use

`lsof -i :<port>` to identify occupant. Stop it or use `--port <other_port>`.

---

### Issue 11: Model download fails inside container

Check: (1) `HF_TOKEN` passed via `-e HF_TOKEN=<token>`, (2) proxy vars `-e HTTP_PROXY=... -e HTTPS_PROXY=...`, (3) DNS via `--network=host`. Best practice: pre-download models locally and mount via `-v`.

---

### Issue 12: Model name mismatch between server and eval

Verify served name: `curl -s http://localhost:<port>/v1/models`. Use `--served-model-name` alias if set; otherwise the model path is the name.

---

### Issue 13: Loglikelihood task fails with local-chat-completions

**Symptom:**

```text
NotImplementedError: Loglikelihood is not supported for chat completions. Consider using the completions API instead.
```

**Root cause:** Many benchmarks (mmlu, mmlu_pro, leaderboard_mmlu_pro, hellaswag, arc, winogrande, truthfulqa_mc2) use loglikelihood scoring — they need the model to return log probabilities for each candidate answer, not generate text. The `local-chat-completions` wrapper in lm-eval does not implement loglikelihood.

**Fix:** Switch to `local-completions` + completions API endpoint (export `MODEL_NAME`, `PORT`, `RESULTS_DIR`):

```bash
lm_eval \
  --model local-completions \
  --tasks leaderboard_mmlu_pro \
  --model_args "model=${MODEL_NAME},base_url=http://0.0.0.0:${PORT}/v1/completions,num_concurrent=64,max_retries=5,timeout=1800,tokenized_requests=False" \
  --gen_kwargs "temperature=0,top_p=1" \
  --log_samples \
  --output_path "${RESULTS_DIR}"
```

**Note:** This applies even for chat models. vLLM's completions endpoint works for both chat and base models — it just bypasses the chat template and uses raw text completion, which is exactly what loglikelihood scoring needs.

**Prevention:** In Phase 6, always check the task's request type (`loglikelihood` vs `generate_until`) before constructing the eval command. See eval-frameworks.md "Task Request Type" section.

---

### Issue 14: vLLM exits immediately but health poll waits 300s

**Symptom:** vLLM fails at startup (bad argument, missing module, etc.) and the process exits within seconds, but the readiness poll loop only checks the health endpoint and waits the full 300s timeout before detecting the failure.

**Root cause:** The polling loop does not check whether the server process is still alive — it only polls `curl /health`. If the process dies immediately (e.g., `error: unrecognized arguments: --enable-reasoning`), each curl silently fails and the loop sleeps 5s × 60 = 300s before timing out.

**Fix:** Use the canonical readiness loop in `backends.md` §Readiness Poll Template — it adds a `pgrep` liveness check at the top of each iteration and breaks immediately on process death (~5s vs 300s). Do not maintain a duplicate copy here.

---

### Issue 15: Loglikelihood task accuracy abnormally low due to max_length truncation

**Symptom:** lm-eval logs show repeated warnings:

```text
Context length (2220) + continuation length (1) > max_length (2047). Left truncating context.
```

Accuracy is far below expected (e.g., mmlu_pro at 47% instead of 60%+).

**Root cause:** lm-eval's API model (`local-completions`) defaults to `max_length=2048`. For benchmarks with long prompts (mmlu_pro 5-shot with 10 options, mmlu 5-shot, etc.), the context exceeds 2048 tokens. lm-eval left-truncates the prompt, removing few-shot examples from the beginning, so the model answers without sufficient context.

**Fix:** Add `max_length=<model_max_context>` to `model_args`:

```bash
--model_args "model=<name>,base_url=...,max_length=32768,..."
```

Use the model's `max_position_embeddings` from config.json (e.g., 32768 for Qwen3-8B, 40960 for Qwen3-8B's config, 131072 for Llama-3.1). Larger values are safe — lm-eval only uses what it needs.

**Prevention:** In Phase 6, always add `max_length=<max_position_embeddings>` to `model_args` for `local-completions` loglikelihood tasks. This should be the default, not an afterthought.
