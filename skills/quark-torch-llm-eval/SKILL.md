---
name: quark-torch-llm-eval
description: >-
  End-to-end LLM accuracy evaluation on AMD ROCm (ROCm-only) — docker container setup OR host
  (no-docker) runtime, vLLM/SGLang/ATOM serving, lm-eval / lighteval / evalscope benchmarks. Use
  when the user wants to evaluate, benchmark, or compare an LLM's accuracy. Trigger for "evaluate
  this model", "run gsm8k/mmlu/mmlu_pro/aime/gpqa/hellaswag/arc", "test accuracy", "measure
  perplexity", "compare quantized model accuracy", "does this mxfp4 model lose accuracy". For
  evaluating Quark Agent Skills themselves, use the Quark repository's internal skill evaluator
  instead.
---

# LLM Evaluation on AMD ROCm

## Catalog portability

This standalone skill is federated from `amd/Quark` at commit `1b229f781a1974cc742884e42d8eefc1eebb4f0a`. Resolve bundled files relative to this `SKILL.md`. Repository-relative paths such as `docs/`, `examples/`, and `quark/` refer to the [pinned Quark source](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a); use a local Quark checkout when available, otherwise consult that pinned source.

End-to-end LLM accuracy evaluation on AMD ROCm — from Docker container to final accuracy table.

## When to use

Trigger this skill when the user asks to:

- Evaluate, benchmark, or test a model's accuracy
- Run gsm8k / mmlu / mmlu_pro / aime / math_500 / gpqa / hellaswag / arc
- Deploy a model with vLLM, SGLang, or ATOM and measure accuracy
- Compare quantized model quality against a reference
- Verify a newly quantized model ("does this mxfp4 model lose accuracy?")

## Inputs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `model_path` or `hf_id` | **Yes** | Local path to model weights, or HuggingFace model ID |
| `benchmark` | No | Benchmark to run (default: `gsm8k`) |
| `backend` | No | Serving backend: `vLLM` (default), `SGLang`, or `ATOM` |
| `image` | No | Docker image (default: `vllm/vllm-openai-rocm:latest`; overridden by model card / Recipes / user). Ignored when `runtime=host`. |
| `runtime` | No | `docker` (default) or `host`. Auto-detected in Phase 0.5: `docker` when the daemon is reachable, else `host` when a native `vllm` / `sglang` is importable. ATOM is docker-only. Stop and report if neither path is available. |

## Outputs

`$EVAL_STATE_DIR/eval_report.md` — see `templates.md` §2 for the full skeleton, formatting rules, and the example used as the canonical shape.

## Design Principles

1. **Trust the model card** — launch commands from model card / Recipes / Cookbook are author-verified. Use them as-is, only adjusting TP and `HIP_VISIBLE_DEVICES`. Do not preemptively stack flags.
2. **Reactive error handling** — `troubleshooting.md` is a diagnostic guide, not a prevention checklist. Launch first, consult on failure.

## Interaction Flow

### Unified confirmation gate

After Phase 0 + 0.5 + 1 finish collecting all information, present a **single confirmation summary** to the user before any container or server operations begin. Use the template in `templates.md` §1 — it defines all required fields, command format, and display rules.

Wait for explicit user approval before executing any of these commands. This avoids piecemeal confirmations and lets the user review the full pipeline at once.

**After approval**, the following are pre-approved by the unified gate — execute without re-asking:

- `docker run` / `docker exec` / `docker pull` of the planned container/image
- The exact `vllm serve` / `python -m sglang.launch_server` / `python -m atom.entrypoints.openai_server` command shown in the plan (with the planned GPU/TP/port)
- The exact `lm_eval` / `lighteval` / `evalscope` command shown in the plan
- `pip install` of frameworks listed in `eval-frameworks.md` §Installation — **docker runtime only** (inside the planned container). In host runtime, pip install is never pre-approved; see Phase 0.5 host-mode constraints.

Each of the following **still requires per-command confirmation**, even after the unified gate:

- Any edit to model `config.json` or backend source code (show full diff first)
- `kill -9` / `sudo` / `rm -rf` / `mv` outside `$EVAL_STATE_DIR` and outputs
- Any `vllm serve` / eval command that **deviates** from the plan (different flags, different port, retry with new args)
- `docker stop` / `docker rm` (Phase 7 cleanup — always ask)

Read-only probes (`ls`, `cat`, `du`, `rocm-smi`, `docker ps`, `docker images`, `curl -sI`, `curl -fsSL`, Read) do **not** need confirmation.

### Never kill other users' processes

- Never use `pkill -f <pattern>` or `ps | grep | awk | kill` broad matching
- Verify any PID is alive and belongs to the expected backend via `kill -0 <pid>` + `ps -fp <pid>` (check `USER` and `CMD` columns) before sending a signal — PIDs are recycled
- Send `SIGTERM` first, wait up to 30 s, then `SIGKILL` only if still alive

### HuggingFace fetch rules

All HuggingFace HTTP requests follow `model-inspection.md` §2 — `curl -fsSL`, `${HF_ENDPOINT:-https://huggingface.co}`, token from `~/.hf_token` then `$HF_TOKEN`. Read that section before the first HF request in any phase.

### Phase 0 — Auto-discover

**Prerequisite: user must provide `model_path` (local path) or `hf_id` (HuggingFace ID).**

- **local path** → verify it exists (`ls <model_path>/config.json`) → proceed
- **hf_id only** → fetch `config.json` and `tokenizer_config.json` from HF API (use the HuggingFace fetch rules above), then continue to Phase 0.5 → 1. Before Phase 2, ask: **A** "I have a local copy" or **B** "Download from HuggingFace"
- **model name only** (e.g., "llama 7b") → search HuggingFace, list candidates, let user choose → handle as hf_id
- **nothing** → ask for model_path or hf_id

Once a local path is confirmed, read `model-inspection.md` and follow its rules (infer hf_id, parse config.json, detect model type, estimate size, compute TP candidates). Write findings to `$EVAL_STATE_DIR/current-eval.yaml`.

### Phase 0.5 — Environment preflight (fail-fast)

Before Phase 1, run a 60-second environment probe to fail fast:

```bash
which curl python3
# Probe both runtimes; do NOT fail if docker is missing — host mode covers that.
docker --version >/dev/null 2>&1 && docker info >/dev/null 2>&1 \
  && echo "docker_available=true" || echo "docker_available=false"
python3 -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null \
  && echo "host_vllm_available=true" || echo "host_vllm_available=false"
python3 -c "import sglang; print('sglang', sglang.__version__)" 2>/dev/null \
  && echo "host_sglang_available=true" || echo "host_sglang_available=false"
```

**Runtime selection (record `runtime:` in `current-eval.yaml`):**

1. If the caller passed `runtime` explicitly (`docker` or `host`), honour it; validate the matching tool exists.
2. Else if `docker_available=true`, use `docker` (default — production path).
3. Else if `backend=vLLM` and `host_vllm_available=true`, or `backend=SGLang` and `host_sglang_available=true`, fall back to `runtime=host` and **report the fallback in the unified confirmation**.
4. Else stop and report: docker is not available, and the requested backend is not importable on the host either. List which of (docker, vllm, sglang) is missing and instruct the user to install one. **ATOM is docker-only** — if `backend=ATOM` and docker is missing, stop here regardless of host imports.

Host-mode constraints (announce in the unified confirmation):

- `image` is ignored; the host's currently-installed framework version is what runs.
- All later phases that show `docker exec ${CONTAINER} <cmd>` become `<cmd>` directly on the host (the "Host runtime variant" call-out under each phase below).
- `pip install` of frameworks is NOT performed — the host environment is used as-is. Missing a benchmark framework (e.g., `lm_eval`) → stop and ask the user to install it explicitly; do not pip install on the host without consent.

**Session path setup (first thing in this phase):** generate a unique, per-session state directory to avoid collisions between concurrent evals.

`bash` shell state does NOT persist across separate `Bash` tool calls — `EVAL_USER` / `EVAL_TS` exported in one call are gone in the next. The path must therefore be **resolved once and persisted to disk**, not re-exported each phase.

```bash
EVAL_USER=$(whoami)
EVAL_TS=$(date +%Y%m%d_%H%M%S)
EVAL_STATE_DIR="/tmp/eval-state/${EVAL_USER}/${EVAL_TS}"
mkdir -p "${EVAL_STATE_DIR}"
echo "eval_state_dir: ${EVAL_STATE_DIR}" > "${EVAL_STATE_DIR}/current-eval.yaml"
echo "${EVAL_STATE_DIR}" > /tmp/eval-state/.last-session-path-${EVAL_USER}
```

All subsequent state files live under `$EVAL_STATE_DIR`:

- `$EVAL_STATE_DIR/current-eval.yaml` — phase progress (machine-readable YAML)
- `$EVAL_STATE_DIR/server-runtime.md` — server state
- `$EVAL_STATE_DIR/vllm-server.log` — server log (redirected here)
- `$EVAL_STATE_DIR/eval-output.log` — eval framework output
- `$EVAL_STATE_DIR/monitor.pid` — background monitor task IDs (Phase 3)

**Recovering `$EVAL_STATE_DIR` in later phases**: every later Bash call starts with

```bash
EVAL_STATE_DIR=$(cat /tmp/eval-state/.last-session-path-$(whoami))
```

or reads the value from the first line of the most recent `current-eval.yaml`. Do not call `date +%Y%m%d_%H%M%S` again — it produces a different timestamp and a different (empty) directory.

Record results in the `env_preflight` section of the state file (capture all probe outputs plus the chosen `runtime`). If neither `docker run` nor a native backend is usable per the Runtime selection rules above → **stop and report**, listing what is missing.

### Phase 1 — Info collection (fully autonomous)

**Do not ask the user any questions during Phase 0, 0.5, or 1.** Collect all information autonomously, apply sensible defaults for anything missing, and present everything in a single unified confirmation at the end. The user makes one decision — approve or adjust.

Defaults for missing info:

- `image`: no model card / recipe specifies one → default by backend: vLLM `vllm/vllm-openai-rocm:latest`; ATOM `rocm/atom-dev:latest`; SGLang `lmsysorg/sglang:v0.5.11-rocm720-mi35x`
- `benchmark`: unspecified and no model card recommendation → default to `gsm8k`
- `backend`: unspecified → default to `vLLM`

**Image availability check**: run `docker images --format '{{.Repository}}:{{.Tag}}'` and check if the chosen image already exists locally. Report the result in the unified confirmation (e.g., "Image: vllm/vllm-openai-rocm:latest (already pulled)" or "Image: vllm/vllm-openai-rocm:latest (needs pull)").

Fetch model card and look for: recommended Docker image, serving / launch command, evaluation command, recommended benchmark.

**Model card fetch**: always use `curl` to get the raw README.md directly — returns raw markdown without needing HTML parsing or JS rendering. Apply the HuggingFace fetch rules above:

```bash
curl -fsSL "${HF_BASE}/<hf_id>/raw/main/README.md" \
  ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"}
```

**Fetch strategy — per-item priority chain with two-stage model card lookup:**

Five pieces of info are collected: `image`, `launch_cmd`, `eval_cmd`, `eval_settings`, and `reference_scores`. Each one independently walks the priority chain below. Finding one item at a given level does NOT skip that level for other items — only skip lower levels for the specific item already found.

`eval_settings` includes: `num_fewshot`, evaluation mode (`base`/`chat`, i.e. `local-completions`/`local-chat-completions`), and prompting strategy (`CoT`/`direct`). These settings determine how the eval command is constructed and are critical for reproducing published results.

Example: model card provides `launch_cmd` but not `eval_cmd` → `launch_cmd` is settled. For `eval_cmd`, continue down the chain until found.

Priority levels:

1. **Quantized model's own model card** — `curl -sL https://huggingface.co/<hf_id>/raw/main/README.md` (e.g., `amd/MiniMax-M2.7-mxfp4`). Highest trust: quant-specific commands verified by the quantized model's author.

2. **Sibling quant variant's model card** — if the quantized model has no HF page or no useful commands, search for sibling quant variants under the same org:

   ```bash
   curl -fsSL "${HF_BASE}/api/models?search=<org>/<base-model-name>&author=<org>" \
     ${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"}
   ```

   If a variant with a different quant strategy is found (e.g., `amd/Model-fp8` when evaluating `amd/Model-mxfp4`), fetch its model card. Commands from a sibling variant share the same architecture and only differ in quantization flags. Record `launch_cmd_source: similar_quant_card` and `reference_model: <sibling_hf_id>`.

3. **Base model's model card** — `curl -sL https://huggingface.co/<base_hf_id>/raw/main/README.md` (e.g., `MiniMaxAI/MiniMax-M2.7`). Commands need adaptation: replace model path, may need quantization-specific adjustments. Record `launch_cmd_source: base_model_card`.

4. **vLLM Recipes** — use `base_hf_id` (base model org/name, not quantized variant). Recipes pages are JS-rendered; try `curl -sL https://recipes.vllm.ai/<org>/<base-model-name>` and extract from `__NEXT_DATA__` or HTML. If not accessible, skip to next level.
5. **ATOM Recipes** / **SGLang Cookbook** — backend-specific sources
6. **Framework repos** / **InferenceX** — eval commands only
7. **Paper / technical report** — for `eval_cmd` and `eval_settings` only. When levels 1–6 do not provide an explicit eval command or eval settings, extract settings from the model's arxiv paper. See `eval-frameworks.md` §Paper Eval Settings Extraction for details. Record `eval_cmd_source: paper` and `eval_settings_source: paper`.
8. **Generic templates** from `eval-frameworks.md` — last resort. Record `launch_cmd_source: template`.

Record each item and its source in the state file (e.g., `launch_cmd_source: quant_model_card`, `eval_cmd_source: template`).

**Reference scores collection**: while fetching model cards and papers in the priority chain above, also collect benchmark reference scores for the target benchmark(s). Walk the lookup chain in `eval-frameworks.md` §Accuracy Comparison — it covers model card, base model card, paper/technical report, Open LLM Leaderboard, and similar-scale fallback. Record the score, source, and **eval setting** (num_fewshot, CoT/direct, base/chat) for each benchmark. This enables the unified confirmation to show expected accuracy and flag setting mismatches before the eval runs.

**Note:** The local path may be a quantized variant (e.g., `/shareddata/amd/Model-mxfp4`), but Recipes uses the base model ID (`base_hf_id`). In Phase 3, replace the base model ID in the command with the local path.

### Phase 2 — Container

**Host runtime variant:** if `runtime=host` (set in Phase 0.5), **skip this phase entirely**. Set `CONTAINER=""` (sentinel) in the state file so Phase 3+ branches detect host mode. Jump straight to Phase 3.

**Docker runtime (default):** Read `backends.md` for the full `docker run` template. Name the container `eval-<hf_id_safe>-<username>-<date>` (username from `whoami`).

### Phase 3 — Pick & adapt launch command

**Starting point — use the highest-priority source found in Phase 1:**

1. Model card command → use as-is (highest trust)
2. Recipes / Cookbook command → use if model card had none
3. No command from any source → minimal template by backend:
   - **vLLM**: `vllm serve <model_path> --tensor-parallel-size <TP>`
   - **ATOM**: `python -m atom.entrypoints.openai_server --model <model_path> --kv_cache_dtype fp8 -tp <TP>`
   - **SGLang**: `python -m sglang.launch_server --model-path <model_path> --tp <TP>`

**Model card command rule**: when a model card provides a launch command, preserve ALL flags from that command. The allowed/forbidden modifications table below only applies to changes YOU make on top of the model card command — do not strip flags that the model card author included (e.g., `--enforce-eager`, `--reasoning-parser`, `--mm-encoder-tp-mode`, env vars like `VLLM_ROCM_USE_AITER=1`).

**Adaptation rules**: see `backends.md` §Cross-Backend Launch Adaptation Rules — applies to all three backends. Only the changes in that table are allowed on top of the source command; reactive flags (`--enforce-eager` etc.) come from `troubleshooting.md` only when an error matches.

**TP priority**: see `model-inspection.md` §5 — recipe TP takes precedence over auto-inferred.

**ATOM-specific notes:** ATOM uses `--kv_cache_dtype fp8` by default for memory efficiency. ATOM auto-detects quantization configs from HuggingFace model configs (FP8, MXFP4, INT8, INT4). For MTP speculative decoding, add `--method mtp --num-speculative-tokens 3`. ATOM also supports running as a vLLM plugin — if `atom` is installed alongside `vllm`, `vllm serve` automatically picks up ATOM's optimized kernels.

**Pre-launch checks (immediately before launching):**

1. GPU index mapping: follow `model-inspection.md` §4 — build the rocm-smi → HIP index mapping via PCIe bus addresses. `rocm-smi` indices and `HIP_VISIBLE_DEVICES` indices are NOT the same.
2. GPU availability: use `rocm-smi --showmeminfo vram` for exact free VRAM, `--showuse` for GPU%, `--showpids` for processes. Identify idle GPUs by rocm-smi index, then convert to HIP indices using the mapping.
3. Select final TP: smallest value from Phase 0 TP candidates that fits the available idle GPU count and whose aggregate free VRAM ≥ model size × 1.3
4. Port selection: default to **8080** (not 8000 — commonly occupied by other services). If model card specifies a port, use that instead. Check with `lsof -i :<port>` — if occupied, try 8081, 8082, etc. Add `--port <port>` to the launch command.
5. Write chosen GPUs (both rocm-smi and HIP indices), TP, and port to state file

**Launch template — server log path is mandatory.** Phase 4/5/7 monitoring all assume the log is reachable at `${EVAL_STATE_DIR}/vllm-server.log`; do not deviate.

Both runtimes write the server log directly to `${EVAL_STATE_DIR}/vllm-server.log` — for docker, this works because the `docker run` template in `backends.md` bind-mounts `${EVAL_STATE_DIR}` into the container at the same path.

**Docker runtime (default):**

```bash
docker exec -d ${CONTAINER} bash -c \
  "HIP_VISIBLE_DEVICES=${GPUS} ${LAUNCH_CMD} > ${EVAL_STATE_DIR}/vllm-server.log 2>&1"
```

**Host runtime variant** (when `runtime=host`):

```bash
# Launch directly on the host; PID captured into the state dir for later signals.
( HIP_VISIBLE_DEVICES=${GPUS} ${LAUNCH_CMD} > "${EVAL_STATE_DIR}/vllm-server.log" 2>&1 ) &
echo "$!" > "${EVAL_STATE_DIR}/server.pid"
```

In host mode, every later step that says `docker exec ${CONTAINER} <cmd>` becomes just `<cmd>` on the host; every "tail the in-container log" becomes `tail -F "${EVAL_STATE_DIR}/vllm-server.log"` (same path works in docker mode too, since the log file is bind-mounted).

Write `$EVAL_STATE_DIR/server-runtime.md` immediately after launch (PID, port, container, image, full launch command).

**Readiness wait (up to 300s):** use the poll template in `backends.md` §Readiness Poll Template — polls health every 5s with process liveness check, breaks immediately on process death instead of waiting 300s.

**Background server monitor**: after the server is ready, launch the monitor via `Bash` with `run_in_background: true` and persist the returned `task_id`:

```bash
# Bash, run_in_background: true. Capture task_id, then:
echo "server_monitor=${task_id}" >> "${EVAL_STATE_DIR}/monitor.pid"
# Both runtimes — log is bind-mounted into the container, so tail from host either way:
tail -F "${EVAL_STATE_DIR}/vllm-server.log" \
  | grep -E --line-buffered "Avg prompt throughput|Avg generation throughput|ERROR|error|OOM|Killed|CUDA|torch.OutOfMemoryError|engine is dead"
```

`TaskStop` every ID in `monitor.pid` at every exit path: Phase 5 retry (before each new launch), Phase 6 early-exit, Phase 7 cleanup. Skipping this leaks a background `tail -F` per session.

### Phase 4 — Smoke test + quality gate

Follow `backends.md` §Smoke Test Templates — verify model name via `/v1/models`, then send a simple inference request (chat or base). Both HTTP 200 and reasonable content are required. Either gate fails → enter Phase 5.

### Phase 5 — Fix if needed

Entered on Phase 3 launch failure or Phase 4 gate failure. Mode comes from `launch_cmd_source` in `current-eval.yaml`.

**Reproduce mode** (`launch_cmd_source ∈ {quant_model_card, similar_quant_card}`) — the command is verified for this arch+quant combo, so do **not** auto-fix:

1. `TaskStop` every ID in `$EVAL_STATE_DIR/monitor.pid`
2. Report: (a) core error line, (b) analysis from `troubleshooting.md`, (c) suggested fix
3. Wait for the user to choose explicitly: **`fall through to bringup`** (re-enter Phase 5 in Bringup mode) or **`abort`** (stop, offer Phase 7 cleanup). Do not silently fall through. Do not retry without instruction.

**Bringup mode** (`launch_cmd_source ∈ {base_model_card, recipes, template}`) — run the fix cycle below.

**Three rules (bringup only):** one fix at a time (no flag stacking); rerun the exact failing step (Phase 3 or 4, no skipping); max 3 attempts then stop and report the full log.

**Fix cycle:**

1. Get server log or traceback, locate the **core error line**
2. Match error signature against `troubleshooting.md`
3. Apply **one** minimal fix:
   - env var / launch flag → apply directly
   - source code / `config.json` → show full diff, wait for user approval
   - no match → search `github.com/vllm-project/vllm/issues`; same approval rule for any source mod
   - kernel / hardware-unsupported → stop and inform user
4. Rerun the exact failing step
5. Record `{attempt, error, fix, result}` in `current-eval.yaml`

### Phase 6 — Eval + report

Three steps, in order. Do not present the final report until all three are done.

**Step 1 — Run eval.** Read `eval-frameworks.md` for command templates, task type classification, and `num_fewshot` lookup. Format the raw eval output as shown in the Outputs section above.

**Step 2 — Accuracy comparison.** Use the `reference_scores` already collected in Phase 1. If Phase 1 found a reference, use it directly. If not, do a fallback lookup per `eval-frameworks.md` §Accuracy Comparison (paper → Open LLM Leaderboard → similar-scale model). Build the comparison table with the eval setting from the reference source. If the reference's eval setting differs from what was actually used, note the difference explicitly. No benchmark may be silently omitted — if no reference exists, write "N/A" with reason.

**Step 3 — Anomaly detection.** Flag any delta > 5% absolute. See `eval-frameworks.md` §Anomaly detection for the diagnostic table.

### Phase 7 — Post-eval cleanup

After the evaluation report is produced, ask the user how to handle the running resources:

0. **Stop background monitors** — for every line in `$EVAL_STATE_DIR/monitor.pid`, call `TaskStop` on the recorded `task_id`. Truncate the file after.
1. **Stop server?**
   - Docker runtime: kill the inference process inside the container (`docker exec ${CONTAINER} pkill -f "vllm serve|sglang.launch_server|atom.entrypoints"`), then verify the port is released via `lsof -i :${PORT}`.
   - Host runtime: stop the PID in `${EVAL_STATE_DIR}/server.pid` following the "Never kill other users' processes" rule above (L72-76) — verify the PID is alive and still our backend, then SIGTERM → wait → SIGKILL only if needed. Verify port with `lsof -i :${PORT}`.
2. **Remove container?** — `docker stop` + `docker rm`. Skip in host runtime.
3. **Rollback file edits?** — if `source_edits` is non-empty in `$EVAL_STATE_DIR/current-eval.yaml`, offer to restore each `.bak.<timestamp>` file: `cp <file>.bak.<timestamp> <file>`
4. **Keep everything** — user may want to re-run or debug

Always ask — never auto-kill. After cleanup (or skip), update `$EVAL_STATE_DIR/server-runtime.md`.

## State Files

`$EVAL_STATE_DIR` is set in Phase 0.5 to `/tmp/eval-state/<user>/<timestamp>/` — unique per session and user. Every later phase **reads** the path from `/tmp/eval-state/.last-session-path-$(whoami)` rather than re-deriving from `whoami` + timestamp.

Primary: `$EVAL_STATE_DIR/current-eval.yaml` — updated after each Phase.
Server: `$EVAL_STATE_DIR/server-runtime.md` — written at Phase 3 launch, kept current.

**Fallback**: if `/tmp/eval-state/` is not writable, use `./.eval-state/<timestamp>/` in the current working directory. Phase 0.5 preflight probes this and writes the resolved path as the first field of `current-eval.yaml` and into the `.last-session-path-<user>` pointer.

## References

| File | When to read |
|------|-------------|
| `model-inspection.md` | Phase 0, for config parsing / GPU probing / TP math; §2 for all HF HTTP requests |
| `backends.md` | Phase 2-3, for docker setup + serving commands (all backends) |
| `troubleshooting.md` | Phase 5, on launch failure or smoke test failure |
| `eval-frameworks.md` | Phase 6, for eval commands, accuracy comparison lookup, and anomaly detection |
| `templates.md` | Phase 1 (pre-run confirmation) and Phase 6 (assembling `eval_report.md`) |
