# Evaluation Frameworks

Phase 6 of the main workflow calls this file.

## Model Type Detection

See `model-inspection.md` §3 for the canonical detection logic. The result is recorded as `model_type_for_eval: chat | base` in the state file.

## Benchmark → Framework Mapping

| Benchmark | Framework | Notes |
|---|---|---|
| `gsm8k` / `mmlu` / `hellaswag` / `arc_*` / `truthfulqa` / `winogrande` | **lm-eval** | Classic academic benchmarks |
| `mmlu_pro` / `ifeval` / `bbh` | **lm-eval** | Supported in lm-eval 0.4.3+ |
| `aime24` / `math_500` / `gpqa:diamond` | **lighteval** or **evalscope** | Math/reasoning benchmarks |

**Decision rule for benchmarks not in the table:**

1. Check model card / Recipes for a recommended framework → use it if found
2. No recommendation → run `lm_eval --tasks list 2>/dev/null | grep -i <benchmark>` in the container to check lm-eval support
3. lm-eval does not support it → check `lighteval` task list (`python -c "from lighteval.tasks.registry import Registry; ..."` or docs)
4. Not in lighteval → check `evalscope`
5. None of the three support it → inform the user and ask them to provide an evaluation script

## Eval Command Discovery Chain

See SKILL.md Phase 1 for the full priority chain (quantized model card → sibling quant variant → base model card → Recipes → framework repos → InferenceX → generic templates). Each item (`launch_cmd`, `eval_cmd`) walks the chain independently.

**Model card eval command takes precedence over generic templates.** If a model card (from any level in the chain) provides a specific eval command with verified accuracy, use that command regardless of the chat/base template mapping below. For example, some chat models achieve better benchmark accuracy with `local-completions` + few-shot prompting (bypassing the chat template) — the model card author tested this and confirmed the accuracy. Do not override their choice with the generic chat→`local-chat-completions` mapping.

Record the final command and its source in the state file.

## Task Request Type: loglikelihood vs generation

lm-eval tasks use one of two request types. The request type determines which API endpoint and model wrapper to use:

| Request type | Description | API endpoint | lm-eval model type |
|---|---|---|---|
| `generate_until` | Model generates free-form text (e.g., gsm8k, humaneval) | chat/completions or completions | `local-chat-completions` (chat) or `local-completions` (base) |
| `loglikelihood` | Model scores each answer option by probability (e.g., mmlu, mmlu_pro, hellaswag, arc, winogrande) | completions only | `local-completions` only |

**`local-chat-completions` does NOT support loglikelihood.** If a loglikelihood task is run with `local-chat-completions`, lm-eval raises `NotImplementedError: Loglikelihood is not supported for chat completions`.

**How to determine request type before running** (export `CONTAINER` and `TASK_NAME` first; written to a temp file per the §2 execution rule in `model-inspection.md`):

```bash
cat > /tmp/check_task_type.py <<'PYEOF'
import os
from lm_eval.tasks import TaskManager
tm = TaskManager()
cfg = tm.get_task_config(os.environ['TASK_NAME'])
print(f"output_type: {cfg.get('output_type', 'unknown')}")
PYEOF
docker cp /tmp/check_task_type.py "${CONTAINER}":/tmp/check_task_type.py
docker exec -e TASK_NAME="${TASK_NAME}" "${CONTAINER}" python3 /tmp/check_task_type.py
```

If the above fails or returns `unknown`, check common task types:

- **loglikelihood tasks**: `mmlu`, `mmlu_pro`, `leaderboard_mmlu_pro`, `hellaswag`, `arc_easy`, `arc_challenge`, `winogrande`, `truthfulqa_mc2`, `piqa`, `boolq`, `openbookqa`
- **generate_until tasks**: `gsm8k`, `math`, `humaneval`, `mbpp`, `aime24`, `math_500`, `gpqa`, `ifeval`, `bbh`

**Scope**: this decision rule only applies when constructing eval commands from generic templates (`eval_cmd_source: template`). When `eval_settings` have already been determined by a higher-priority source (model card, recipes, or paper), use those settings directly — do not re-derive eval_mode or num_fewshot from the rules below.

**Decision rule:**

1. If task is `loglikelihood` → always use `local-completions` + completions API, regardless of whether the model is chat or base. **Must add `max_length=<max_position_embeddings>` to model_args** — lm-eval defaults to 2048 which truncates few-shot prompts for long benchmarks (mmlu_pro, mmlu, etc.), causing severe accuracy loss.
2. If task is `generate_until` → use `local-chat-completions` for chat models, `local-completions` for base models

This check MUST happen in Phase 6 before constructing the eval command. Record `task_request_type: loglikelihood | generate_until` in the state file.

## Thinking Model Handling

Thinking/reasoning models (Qwen3, QwQ, DeepSeek-R1, etc.) output `<think>...</think>` blocks before the actual answer. This causes two problems for lm-eval:

1. **generate_until tasks** (gsm8k, math, etc.): lm-eval's regex extractors (e.g., `flexible-extract` looking for `#### <number>`) fail because the `<think>` block contains intermediate calculations that confuse the pattern matcher, resulting in near-zero accuracy.
2. **loglikelihood tasks** (mmlu, mmlu_pro, etc.): not directly affected by thinking output (loglikelihood doesn't generate), but may still score poorly if the model's probability estimates are distorted by its tendency to "think first."

**Detection:** In Phase 0, check if the model is a thinking model. The check must gate on the model **type**, not just the family — Qwen3 base checkpoints have no chat template and the soft `/no_think` switch is a no-op on them, so applying it silently does nothing useful and can confuse downstream prompt accounting. Required vars: `MODEL_NAME`, `MODEL_TYPE_FOR_EVAL` (from `model-inspection.md` §3).

```python
is_thinking_model = (
    MODEL_TYPE_FOR_EVAL == 'chat'
    and (
        re.search(r'(QwQ|-R1\b)', MODEL_NAME, re.I)                       # always thinking
        or re.search(r'Qwen3.*-(Instruct|Chat|Thinking)', MODEL_NAME, re.I)  # Qwen3 instruct/chat/thinking variants only
        or model_card_mentions_any(['thinking mode', 'enable_thinking', 'reasoning mode'])
    )
)
```

Record as `is_thinking_model: true|false` in the state file. Do NOT set true for Qwen3 base checkpoints (e.g., `Qwen3-8B-Base`) or for any model whose `model_type_for_eval` is `base`.

**Fix for generate_until tasks — disable thinking mode:**

The preferred approach depends on what the vLLM version and model support:

1. **Best: `--system_instruction "/no_think"`** — add to the lm-eval command. This inserts `/no_think` as the system message, which Qwen3 models recognize as a soft switch to disable thinking. The model still outputs empty `<think>\n\n</think>` tags but no reasoning content, so extractors work correctly.

   ```bash
   lm_eval --model local-chat-completions --apply_chat_template \
     --system_instruction "/no_think" \
     --tasks gsm8k ...
   ```

2. **Alternative: `chat_template_kwargs`** — pass `{"enable_thinking": false}` via the vLLM API's `chat_template_kwargs` parameter. This completely suppresses `<think>` tags. However, lm-eval may not support passing this through `gen_kwargs` or `extra_body` in all versions.

3. **If vLLM supports `--enable-reasoning`** — launch with `--enable-reasoning --reasoning-parser deepseek_r1` so vLLM separates thinking content into `reasoning_content` field, keeping `content` clean. But this flag is not available in all vLLM builds.

**Note:** Disabling thinking may reduce accuracy on reasoning-heavy benchmarks (math, aime). If the model card reports accuracy with thinking enabled, the results with `/no_think` may be lower. Document which mode was used in the eval report.

## Eval Command Type Classification

Before executing, classify the `eval_cmd` collected in Phase 1:

| Pattern | Type | Action |
|---------|------|--------|
| `python vllm/tests/evals/...` or `python tests/evals/...` | **vllm-script** | Download the single script from vLLM GitHub, run directly |
| `lm_eval --model ...` | **lm-eval** | Install framework if missing, run |
| `lighteval ...` | **lighteval** | Install framework if missing, run |
| `evalscope ...` | **evalscope** | Install framework if missing, run |
| Other `python <script>` | **custom-script** | Download the script if not present, run |

**Source priority**: if `eval_cmd_source: model_card`, use the command verbatim — only adapt `model` name and `base_url` port. If from template/recipes/inferencex, follow the generic templates below.

## vllm-script Handling

When eval_cmd references a vLLM repo script (e.g., `python vllm/tests/evals/gsm8k/gsm8k_eval.py`):

1. Construct the GitHub raw URL: `https://raw.githubusercontent.com/vllm-project/vllm/main/<path>`
   - `vllm/tests/evals/gsm8k/gsm8k_eval.py` → download from `main/tests/evals/gsm8k/gsm8k_eval.py`
2. Download **only this single file** into the container (export `CONTAINER` and `URL` first):

   ```bash
   docker exec "${CONTAINER}" bash -c "mkdir -p /tmp/vllm/tests/evals/gsm8k && curl -fsSL '${URL}' -o /tmp/vllm/tests/evals/gsm8k/gsm8k_eval.py"
   ```

3. Adapt the command: replace `python vllm/tests/evals/...` with `python /tmp/vllm/tests/evals/...`
4. Add `--port <port>` if the server is not on the script's default port
5. Run directly — do NOT install lm-eval or any other framework

## Installation

Run inside the container (the serving image usually has vLLM pre-installed, but eval frameworks often are not):

```bash
pip install "lm-eval[api]>=0.4.3,<0.5"   # OpenAI-compat client; pinning the 0.4.x line keeps `--model_args max_length=...` and TaskManager API stable
pip install lighteval==0.7.*              # CLI flag set is stable within minor; bump intentionally if a model card requires newer
pip install evalscope==0.10.*             # `--api-url` / `--datasets` / `--eval-type service` are stable in 0.10.x; older majors used different flag names
```

Check `pip show ${FRAMEWORK}` first to avoid unnecessary reinstallation. **Before constructing any eval command**, run `<framework> --help` (or `evalscope eval --help`) inside the container to confirm the flag names against the installed version — pin drift is the most common cause of "command works in docs, fails on this image".

## lm-eval Template

```bash
lm_eval \
  --model <model_type> \
  --tasks <benchmark> \
  --model_args "model=<model_name>,base_url=http://0.0.0.0:<port>/v1/<endpoint>,num_concurrent=<N>,max_retries=5,timeout=1800,tokenized_requests=False" \
  --gen_kwargs "temperature=0,top_p=1" \
  --num_fewshot <N> \
  --log_samples \
  --output_path <results_dir> \
  --limit <N>     # use 10 for smoke test; remove for full eval
```

**Differences by model type:**

| Parameter | Chat / Instruct | Base |
|-----------|----------------|------|
| `--model` | `local-chat-completions` | `local-completions` |
| `base_url` endpoint | `/v1/chat/completions` | `/v1/completions` |
| `num_concurrent` | 64 | 32 |
| `--apply_chat_template` | add this flag | omit |

**Key notes:**

- `model` in model_args must match `--served-model-name` (defaults to model path)
- `temperature=0,top_p=1`: greedy decoding for reproducibility
- `--num_fewshot`: look up from the **Standard num_fewshot Table** below — lm-eval defaults to 0-shot if omitted
- `--log_samples`: preserves per-sample input/output for debugging

### Output

lm-eval prints a markdown table to stdout:

```text
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.821|±  | 0.011|
|     |       |strict-match    |     5|exact_match|↑  |0.815|±  | 0.012|
```

Copy this table **verbatim** to the Phase 6 final output — do not reformat.

## lighteval Template

```bash
lighteval endpoint openai \
  --override-batch-size <B> \
  --tasks "extended|aime24|0|0,extended|math_500|0|0" \
  --output-dir /tmp/lighteval-out \
  --base-url http://0.0.0.0:<port>/v1 \
  --model-name <served_model_name>
```

Output is in `/tmp/lighteval-out/results/` as JSON. Read the JSON and format as a markdown table.

## evalscope Template

```bash
evalscope eval \
  --model <served_model_name> \
  --api-url http://0.0.0.0:<port>/v1 \
  --datasets gpqa \
  --eval-type service \
  --limit <N>
```

evalscope output is also JSON — format as a markdown table.

## Standard num_fewshot Table

lm-eval individual task YAMLs do not set `num_fewshot` — they default to `None` (= 0-shot). The standard evaluation settings are defined in **group YAML files** (e.g., `openllm.yaml`, leaderboard sub-task YAMLs), but these only take effect when you invoke the group name, not individual task names. When constructing eval commands with individual task names, you must explicitly set `--num_fewshot`.

### Lookup priority

1. **Model card eval command** specifies `--num_fewshot` → use that value
2. **Paper / technical report** specifies per-benchmark settings (e.g., "GSM8K (4-shot, CoT)") → use that value. See §Paper eval settings extraction.
3. **lm-eval group config** defines it → use that value (see table below)
4. **Table below** has the task → use the listed value
5. **Not in any source** → default to 0-shot, but note this in the eval report

### Common tasks

| Task | num_fewshot | Source |
|------|------------|--------|
| `arc_challenge` | 25 | `openllm.yaml` (Open LLM Leaderboard v1) |
| `arc_easy` | 25 | convention (same as arc_challenge) |
| `hellaswag` | 10 | `openllm.yaml` |
| `mmlu` | 5 | `openllm.yaml` / original paper |
| `winogrande` | 5 | `openllm.yaml` |
| `gsm8k` | 5 | `openllm.yaml` / original paper |
| `truthfulqa` | 0 | `openllm.yaml` |
| `mmlu_pro` | 5 | `leaderboard/mmlu_pro` |
| `leaderboard_bbh` | 3 | `leaderboard/bbh_mc/_fewshot_template_yaml` |
| `leaderboard_math_hard` | 4 | `leaderboard/math/_template_yaml` |
| `gpqa` | 0 | `leaderboard/gpqa/_template_yaml` |
| `ifeval` | 0 | `leaderboard/ifeval/ifeval.yaml` |
| `piqa` | 0 | loglikelihood task, standard is 0-shot |
| `wikitext` | 0 | perplexity task, no few-shot |
| `openbookqa` | 0 | standard |
| `boolq` | 0 | standard |
| `lambada_openai` | 0 | standard |

### Runtime query (for tasks not in the table)

If a task is not listed above, query its effective `num_fewshot` from inside the container. Use `TaskManager.task_index` (the documented public surface in lm-eval 0.4.x) — earlier drafts used `tm.load([...])['tasks'][...]`, which is not part of the public API. Required vars: `CONTAINER`, `TASK_NAME`.

```bash
cat > /tmp/lookup_num_fewshot.py <<'PYEOF'
import os
from lm_eval.tasks import TaskManager
tm = TaskManager()
name = os.environ['TASK_NAME']
entry = tm.task_index.get(name)
if entry is None:
    print(f'{name}: not registered (TaskManager.task_index miss)')
else:
    cfg = entry.get('yaml_path') and __import__('yaml').safe_load(open(entry['yaml_path'])) or {}
    print(f"{name}: num_fewshot={cfg.get('num_fewshot')}")
PYEOF
docker cp /tmp/lookup_num_fewshot.py "${CONTAINER}":/tmp/lookup_num_fewshot.py
docker exec -e TASK_NAME="${TASK_NAME}" "${CONTAINER}" python3 /tmp/lookup_num_fewshot.py
```

If the result is `None` (or the task is not registered), the task has no built-in standard — default to 0-shot and document this in the eval report.

### Per-task num_fewshot with multiple tasks

When running multiple tasks that need different `num_fewshot` values, lm-eval's `--num_fewshot` flag applies globally to all tasks. Two approaches:

1. **Separate runs**: run each task (or group of same-fewshot tasks) in its own `lm_eval` invocation with the correct `--num_fewshot`
2. **Use group name**: if all tasks happen to be in the same group (e.g., `--tasks openllm`), the group config sets per-task `num_fewshot` automatically

Prefer option 1 when tasks require different num_fewshot values.

## Background Eval Monitor

Eval frameworks (lm-eval, lighteval, evalscope, etc.) use tqdm progress bars that write to stderr with `\r` (carriage return). A plain `tail -f | grep` will never see these.

**Log capture** — redirect both stdout and stderr, convert `\r` to `\n`. Required vars: `CONTAINER`, `EVAL_CMD`. Launch via `Bash` with `run_in_background: true`; capture the returned `task_id` and append `eval_log_writer=${task_id}` to `${EVAL_STATE_DIR}/monitor.pid`:

```bash
docker exec "${CONTAINER}" bash -c "${EVAL_CMD} 2>&1 | stdbuf -oL tr '\r' '\n' > /tmp/eval-output.log"
```

**Monitor** — grep for progress and results. Also launch via `Bash` with `run_in_background: true`; persist as `eval_monitor=${task_id}` in `monitor.pid`:

```bash
docker exec "${CONTAINER}" tail -F /tmp/eval-output.log \
  | grep -E --line-buffered "it/s|%\||Requesting|Running|Task|tasks|Metric|exact_match|acc|elapsed|score|ERROR|Error|Traceback|FAILED|OOM|Killed"
```

`TaskStop` both IDs at every exit path (Phase 5 retry, Phase 6 early-exit, Phase 7 cleanup).

- `it/s` and `%\|` — tqdm progress bars from any framework
- `Metric|exact_match|acc|score` — result lines
- `ERROR|Traceback|FAILED|OOM|Killed` — failure signals

## Accuracy Comparison

**Timing**: reference scores are collected in **Phase 1** (info collection), not after the eval. This allows the unified confirmation to show expected accuracy and eval settings so the user can adjust before running. Phase 6 Step 2 uses the already-collected reference scores to build the comparison table.

Every benchmark in the report must have a reference value or an explicit "N/A" with reason.

### Lookup chain

For each benchmark task, walk in order. Stop at the first level that yields a score. **Also record the eval setting** (num_fewshot, CoT/direct, base/chat) from the source — setting mismatches explain most accuracy deltas.

1. **Evaluated model's own model card** — scores from Phase 1 (`quant_model_card` or `similar_quant_card`).
2. **Base model card** — scores from `base_hf_id` README.
3. **Paper / technical report** — many models (especially new releases) only publish benchmark scores in their arxiv paper, not in the model card. See §Paper benchmark extraction below.
4. **Open LLM Leaderboard dataset** — programmatic API query (see below).
5. **Similar-scale reference model** — if levels 1-4 all fail, pick a well-known model of similar size as a reference point (e.g., Llama-3-8B-Instruct for an 8B model, Qwen2.5-7B-Instruct for a 7-8B model). Record `reference_source: similar_scale_model` and note this is a cross-model comparison, not a self-comparison.

### Paper extraction (scores and eval settings)

Model cards often link to an arxiv paper. Extract both reference scores and eval settings in one pass — scores feed the unified confirmation; settings override generic templates when no eval command was found.

**Step 1: get arxiv paper ID** (apply HF fetch rules from `model-inspection.md` §2):

```bash
HF_BASE="${HF_ENDPOINT:-https://huggingface.co}"
curl -fsSL "${HF_BASE}/<hf_id>/raw/main/README.md" \
  ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} \
  | grep -oP 'arxiv\.org/abs/\K[0-9.]+' | head -1
```

**Step 2: extract reference scores** — required env vars: `PAPER_ID`, `BENCHMARK` (e.g., `GSM8K`), `MODEL_NAME` (e.g., `Qwen3-8B`):

```bash
cat > /tmp/extract_paper_table.py <<'PYEOF'
import os, re, urllib.request
paper_id = os.environ['PAPER_ID']
benchmark = os.environ['BENCHMARK']
model_name = os.environ['MODEL_NAME']
with urllib.request.urlopen(f'https://arxiv.org/html/{paper_id}') as r:
    html = r.read().decode('utf-8', errors='replace')
tables = re.findall(r'<table[^>]*>.*?</table>', html, re.DOTALL)
for t in tables:
    if benchmark in t and model_name in t:
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', t, re.DOTALL)
        for i, row in enumerate(rows):
            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL)
            cleaned = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            if cleaned:
                print(f'Row {i}: {" | ".join(cleaned)}')
        break
PYEOF
PAPER_ID=${PAPER_ID} BENCHMARK=${BENCHMARK} MODEL_NAME=${MODEL_NAME} python3 /tmp/extract_paper_table.py
```

**Step 3: extract eval settings** — search for methodology sections (e.g., "Benchmark Settings", "Experimental Setup"):

```bash
curl -fsSL "https://arxiv.org/html/${PAPER_ID}" | python3 -c "
import sys, re
html = sys.stdin.read()
for sec in re.split(r'<(?:h[1-4]|section)', html):
    sl = sec.lower()
    if any(k in sl for k in ['evaluation','benchmark','setting','setup']) \
       and any(k in sl for k in ['shot','gsm','mmlu','math','gpqa']):
        text = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', sec)).strip()
        if len(text) > 50:
            print(text[:2000]); print('---')
"
```

Key fields to extract per benchmark: `num_fewshot` ("N-shot" after benchmark name), `prompting_strategy` ("CoT" / "direct"), `eval_mode` (base vs instruct, determines `local-completions` vs `local-chat-completions`).

**Record in state file:**

```yaml
reference_scores:
  gsm8k: {value: 89.84, source: "arxiv:2505.09388 Table 4", setting: "base model, 4-shot CoT", model_variant: "Qwen3-8B Base"}
eval_settings:
  gsm8k: {num_fewshot: 4, strategy: "CoT", mode: "base", source: "arxiv:2505.09388 §3.3"}
eval_settings_source: paper
```

Paper settings take precedence over the generic `num_fewshot` table and base/chat defaults. Flag any setting mismatch between the paper and your planned eval in the unified confirmation.

### Open LLM Leaderboard API query

Two datasets exist. Pick the one that covers your benchmarks:

| Dataset | Benchmarks covered |
|---------|-------------------|
| `open-llm-leaderboard-old/results` (v1) | arc_challenge, hellaswag, gsm8k, mmlu, winogrande, truthfulqa |
| `open-llm-leaderboard/results` (v2) | mmlu_pro, gpqa, ifeval, bbh, math, musr |

Neither covers `piqa`, `boolq`, `openbookqa`, `lambada_openai` — mark these as "N/A (not in Open LLM Leaderboard)".

```bash
# 1. List available result files for the reference model
curl -sL "https://huggingface.co/api/datasets/<dataset>/tree/main/<org>/<model>" \
  | python3 -c "import sys,json; [print(i['path']) for i in json.load(sys.stdin)]"

# 2. Fetch the latest result file and extract target benchmarks
curl -sL "https://huggingface.co/datasets/<dataset>/resolve/main/<result_file>" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
for k, v in sorted(d.get('results', {}).items()):
    if isinstance(v, dict):
        for mk, mv in v.items():
            if isinstance(mv, (int,float)) and mv > 0 and 'stderr' not in mk:
                print(f'{k} / {mk}: {mv:.4f}')
"
```

### Reference model selection

Pick **one or both** for comparison:

- **Same family, previous generation**: e.g., `Qwen2-7B` for `Qwen3-8B`, `Llama-3-8B` for `Llama-3.1-8B`. Measures generational improvement.
- **Same scale, different family**: e.g., `Llama-3-8B-Instruct` for `Qwen3-8B`. Cross-family positioning.

### Anomaly detection and diagnostic order

Flag any delta exceeding **5% absolute** with `[ANOMALY]` (text prefix; no emoji). When accuracy is abnormally low (less than half expected), walk the table in order — stop at the first match. If the cause is a configuration issue, suggest a fix and offer to rerun.

| # | Symptom | Likely cause | Fix | Flag `[ANOMALY]`? |
|---|---------|-------------|-----|-------------------|
| 1 | Near-zero on generate_until tasks | Thinking model without `/no_think` | Add `--system_instruction "/no_think"`; see §Thinking Model Handling | Yes |
| 2 | Large drop on mmlu / mmlu_pro / loglikelihood tasks | `max_length=2048` truncation | Add `max_length=<max_position_embeddings>` to `--model_args`; see `troubleshooting.md` Issue 15 | Yes |
| 3 | MoE accuracy regression on AMD | `VLLM_ROCM_USE_AITER=1` missing | Add env var per `troubleshooting.md` AITER table | Yes |
| 4 | Long-prompt benchmarks silently truncated | `--max-model-len` set too low | Check lm-eval log for truncation warnings; increase `--max-model-len` | Yes |
| 5 | Reasoning model produces garbled output | `--reasoning-parser` not configured | Add matching `--reasoning-parser` (e.g., `deepseek_r1`) | Yes |
| 6 | Model-specific eval failure | Model card requirements not applied | Re-read model card for required system-prompt, extra flags, or API parameters | Yes |
| 7 | Wrong num_fewshot | Mismatched eval setting | Re-run with correct `--num_fewshot`; see §Standard num_fewshot Table | Yes |
| 8 | Loglikelihood accuracy 2–4% below base model | Chat model via `local-completions` — expected | Not a bug; note in report | No |
| 9 | 2–3% drop on quantized vs full-precision | Normal INT4/MXFP4 quantization loss | Not a bug; note in report | No |
| 10 | Delta matches reference setting difference | Setting mismatch (base vs chat, different num_fewshot, CoT vs direct) | Note difference in report; do not flag as anomaly | No |

## General Notes

- **Limit benchmark size for smoke check**: first run with `--limit 10` to verify the pipeline works; run full after confirmation
- **Preserve raw results**: `--log_samples` + `--output_path` keeps JSON/logs for reproduction or comparison
- **Unified report format**: regardless of framework, final user-facing output follows the SKILL.md Phase 6 template (accuracy table + metadata)
- **Accuracy reference thresholds** (from InferenceX): gsm8k ≥ 0.85, gpqa_diamond ≥ 0.30 (for full-precision models, not quantized)
