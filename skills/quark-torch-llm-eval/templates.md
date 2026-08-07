# User-Facing Templates

Two templates presented to the user at different phases of the eval pipeline.

## 1. Pre-run Confirmation (end of Phase 1, before Phase 2)

Present this summary after Phase 0 + 0.5 + 1 finish collecting all information, before any container or server operations begin. Wait for explicit user approval before proceeding.

````markdown
## Evaluation Plan

**Model**: <model_path or hf_id> (<hf_id>)
**Base model**: <base_hf_id if different from hf_id, e.g. the unquantized original>
**Architecture**: <architectures from config.json>
**Model type**: chat | base (detection reason: chat_template / name pattern / VLM ForConditionalGeneration)
**MoE**: Yes (<N> routed experts, <K> active per token, <S> shared) | No
**Heads**: <num_attention_heads> attention heads, <num_key_value_heads> KV heads
**Max position**: <max_position_embeddings>
**Quant**: <quant_method> (<dtype>, group_size <N>) | None
**Requires trust-remote-code**: Yes (auto_map) | No
**Is VLM**: Yes | No

**Backend**: vLLM | SGLang | ATOM
**Image**: <image> (source: model card | recipes | default) — already pulled | needs pull
**TP candidates**: <list> (must divide num_kv_heads=<N>; final TP selected at launch based on idle GPUs)
**GPUs**: selected at launch time from idle GPUs (<gpu_model>)
**Benchmark**: <benchmark>

### Model path (only if no local path yet)
> Do you have a local copy of this model?
> - **A. Yes** — please provide the local path
> - **B. No, download it** — run: `huggingface-cli download <hf_id> --local-dir <path>`

### Docker command
```bash
docker run -it -d \
  --entrypoint /bin/bash --ipc=host --shm-size=<64g|128g> --network=host \
  --name=<container-name> \
  --privileged --cap-add=CAP_SYS_ADMIN \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /shareddata:/shareddata \
  <image>
```

### Launch command (source: model card | recipes | template)
```bash
HIP_VISIBLE_DEVICES=<gpus> vllm serve <model_path> --tensor-parallel-size <tp> ...
```

### Eval command (source: model card | recipes | inferencex | template)
```bash
lm_eval --model <local-completions|local-chat-completions> --tasks <benchmark> ...
```

### Reference scores (source: paper | model card | leaderboard | similar-scale model)
| Benchmark | Reference | Setting | Source |
|-----------|-----------|---------|--------|
| gsm8k     | 89.84     | base model, 4-shot CoT | arxiv:2505.09388 Table 4 |

> [WARNING] Setting mismatch: reference uses ${REF_MODE}, ${REF_NSHOT}-shot; this eval uses ${PLAN_MODE}, ${PLAN_NSHOT}-shot. Results may differ due to eval setup, not model quality.

(Only show the [WARNING] note when the reference eval setting differs from the planned eval setting. Omit if settings match.)

Proceed? (yes/no)
````

**Display rules:** show every field (even "No"/"None"); annotate each command's **source**; preserve model card commands verbatim; note any deviation from generic templates (e.g., chat model using `local-completions`). Download section only appears when no local path is provided. Reference scores section only appears when at least one reference score was found in Phase 1.

## 2. Post-run Report (Phase 6 → eval_report.md)

The user-facing artifact produced at the end of Phase 6. Copy this skeleton and fill the placeholders. Do not reformat the lm-eval table — paste it verbatim from stdout.

````markdown
## Evaluation Result

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.821|±  | 0.011|
|     |       |strict-match    |     5|exact_match|↑  |0.815|±  | 0.012|

## Accuracy Comparison

| Benchmark | This Model | Reference | Source | Delta |
|-----------|-----------|-----------|--------|-------|
| gsm8k (flexible-extract, 5-shot) | 0.821 | 0.845 | model card (amd/Model-Name) | -0.024 |

- Reference from model card / sibling quant variant / similar-scale model
- [ANOMALY] [Analysis, if delta is significant — e.g., >5% absolute drop]

## Metadata
- model_path: /shareddata/amd/Model-Name
- hf_id: amd/Model-Name
- backend: vLLM
- image: vllm/vllm-openai-rocm:latest
- container: eval-amd-Model-Name-${USER}-0428
- HIP_VISIBLE_DEVICES: 6
- TP: 1
- max-model-len: <if set>
- env vars: <if any, e.g. VLLM_ROCM_USE_AITER=1>

## Launch command
```bash
HIP_VISIBLE_DEVICES=6 vllm serve /shareddata/... --tensor-parallel-size 1 ...
```

## Eval command
```bash
lm_eval --model local-chat-completions --tasks gsm8k ...
```
````

**Rules:**

- Copy the lm-eval / lighteval / evalscope output table verbatim — do not reformat or summarize.
- Every benchmark row in Accuracy Comparison must have a Reference value or an explicit `N/A` with reason.
- Use `[ANOMALY]` (text prefix, not emoji) for any delta exceeding 5% absolute. See `eval-frameworks.md` §Anomaly detection for the diagnostic table.
- Metadata must include every field above; substitute actual values, drop the `<if …>` placeholders.
