---
name: hyperloom-workload-optimizer
description: >-
  Autonomously optimizes end-to-end LLM inference throughput on AMD Instinct GPUs
  and reports a validated gain, using the Hyperloom multi-agent optimizer. Given a
  model, framework, workload (TP/EP, concurrency, ISL/OSL, precision), an objective
  and a time budget, it explores per-workload which levers to pull (serving/config
  parameters and env, framework enablement and source patches, and hot GPU-kernel
  rewrites), benchmarks each candidate, and returns the optimization stack that
  produced the gain. Use when the user wants to make a model serve faster, raise
  tokens/sec or throughput, optimize or tune vLLM or SGLang on MI300X/MI325X/MI350X/MI355X,
  run Hyperloom, run the kernel-agent, quantize-then-optimize with Quark, set up
  Hyperloom from scratch, or resume a Hyperloom session. Do not use to stand up a
  server for plain serving, diagnose a broken ROCm install, or run a one-off
  kernel/benchmark or trace analysis without the optimization loop.
---

<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Hyperloom Workload Optimizer

You are the catalog entry point for Hyperloom optimization on AMD Instinct GPUs.
Bootstrap the workspace, prepare the runtime environment, collect workload
parameters, then install, launch, and monitor the optimizer. The packaged `inference_optimizer` skill
(`HYPERLOOM_SKILL_PATH`) is the execution baseline; this catalog skill inlines
the operator workflow so you can run end-to-end without bouncing across files.

Do not manually optimize inside chat unless debugging.

## Prerequisites

- AMD Instinct GPU host (MI300X / MI325X / MI350X / MI355X) with ROCm
- `/dev/kfd` and `/dev/dri` present; `amd-smi` or `rocm-smi` works
- Python 3.10+ and network access to install the Hyperloom wheel
- Anthropic (or compatible) LLM credentials for agent backends
- A dedicated agent workspace directory

The Hyperloom **runtime** ships via `pip install` of the published wheel.

## What Hyperloom runs

The CLI starts a Python Coordinator that coordinates:

- **Orchestration** — baseline, explore, specialist, integrate_patch, sweep
- **Kernel** — trace_analyze, run_optimization, integrate
- **Critic** — proposal review (default `--critic-agent`)
- **Robustness** — health monitoring and RCA (default `--robustness-agent`)

State lives under a **session directory** per run; run-state root is
`$USER_DATA_PATH` (default `/workspace/hyperloom`), independent of the install
directory (`INSTALL_DIR`, where the wheel and `.env` live) and may point to
shared storage. Layout: `$USER_DATA_PATH/runtime/` (install.sh outputs,
`kernel-agent.env.sh`), `logs/`, and `<model_basename>/<UTC_ts>/` per session
holding `manifest.json`, `state.json`, `runs/`, `reports/`, `optimizer_runs/`.

## Workflow overview

Match `hyperloom-custom-advanced` section order — do **not** ask workload
questions while writing `.env` or during `/hyperloom-setup`.

- **Phase 0 Bootstrap** — `pip install`, `/hyperloom-setup` → `.env` (credentials + run mode only)
- **Phase 1 Environment** — custom-advanced §Setup Configuration (baremetal: confirm host; docker: start container + setup inside)
- **Phase 2 Workload intake** — custom-advanced §Advanced Configuration → Model Resolution → show launch plan → user confirms
- **Phase 3 Execute** — install.sh → preflight → launch → monitor → report

Load `hyperloom-custom-advanced` at Phase 1 and follow its sections in order
(discovery: `.cursor/` / `.claude/` / `.agents/skills/hyperloom-custom-advanced/SKILL.md`).
For deeper optimizer behavior read `@${HYPERLOOM_SKILL_PATH}` (`inference_optimizer`);
Iron Rules + CLI reference: [reference.md](reference.md).

## Iron Rules (launcher gates)

Run order is always **IR-2 → IR-1 → launch**. Full text in [reference.md](reference.md).

- **IR-1 — GPU unoccupied.** Before every `optimize` (fresh or `--resume`), every
  visible GPU must have zero foreign serving PIDs (`sglang.launch_server` /
  `vllm.entrypoints` / `Magpie`) and ≲ 500 MiB VRAM in use.
- **IR-2 — install.sh before launch.** Run `install.sh` and source
  `kernel-agent.env.sh` in the **same shell** that spawns `optimize`.
- **Resume carve-out:** `--resume` may skip install only when `install.sh` exited
  0 earlier in the same shell, `kernel-agent.env.sh` is still sourced, and the
  session's `manifest.json` exists. Any failure → re-run `install.sh`.

## Phase discipline (do not skip)

One phase at a time. Each phase asks only its own questions, waits for the
user's answers, completes its exit condition, then moves on. Never batch
questions from different phases into one prompt. In particular, never ask
workload questions (model, framework, TP/EP, precision, ISL/OSL, hours…) during
Phase 0 or Phase 1 — those belong to Phase 2 only.

## Phase 0 — Bootstrap

Skip completed steps (idempotent). Ask only about the install directory and
credentials/run mode here. Do not ask about the model or workload yet.

### Confirm the install directory

The wheel installs into a target directory with `pip install --target <dir>`,
which also holds `.env` and runtime artifacts. Do not silently use the current
directory. Show the resolved current directory (`pwd`) and confirm it with the
user, or let them choose another dedicated path. Wait for the answer, then `cd`
into the chosen directory before installing.

### Install the Hyperloom wheel

Skip when `hyperloom/` (wheel) or `src/hyperloom/` (source) already exists in the
confirmed directory.

Find the latest `hyperloom_inference_optimizer-*-py3-none-any.whl` from
[AMD-AGI/Hyperloom releases](https://github.com/AMD-AGI/Hyperloom/releases).
The repo is public, so resolve the URL with `curl` + `python3` (no extra
tooling). When `gh` and `jq` are installed the one-liner in [setup.md](setup.md)
also works.

```bash
cd "$INSTALL_DIR"   # the directory confirmed above
wheel_url="$(curl -fsSL "https://api.github.com/repos/AMD-AGI/Hyperloom/releases?per_page=20" \
  | python3 -c 'import sys, json
for rel in json.load(sys.stdin):
    for asset in rel.get("assets", []):
        if asset["name"].startswith("hyperloom_inference_optimizer-"):
            print(asset["browser_download_url"]); sys.exit(0)
sys.exit("no matching wheel asset found")')"
python3 -m pip install "$wheel_url" --target .
```

Confirm `hyperloom/inference_optimizer/assets/install.sh` exists. Restart the
agent if wheel skills are not visible.

### Credentials and run mode

Run `/hyperloom-setup` (installed to `.cursor/skills/hyperloom-setup/`). It
writes `.env`, sets `USER_DATA_PATH`, `HYPERLOOM_RUN_MODE`, and
`HYPERLOOM_SKILL_PATH`, and on bare metal runs `install_baremetal.sh`.

**Phase 0 is done when all hold:**

- `hyperloom/inference_optimizer/assets/install.sh` exists
- `.env` exists with non-placeholder LLM secrets
- `USER_DATA_PATH`, `HYPERLOOM_RUN_MODE`, and `HYPERLOOM_SKILL_PATH` are set

More bootstrap detail: [setup.md](setup.md).

## Phase 1 — Environment prep

Load `hyperloom-custom-advanced` and follow its **Setup Configuration**
section only.

**Baremetal (`HYPERLOOM_RUN_MODE=baremetal`):** confirm `install_baremetal.sh`
finished and the serving framework from setup is importable. Do not ask workload
questions yet.

**Docker (`HYPERLOOM_RUN_MODE=docker`):** image choice, `docker run`, and the
in-container setup are owned entirely by custom-advanced Setup Configuration —
follow it, do not restate its commands or flags here. Do not ask workload
questions until the container is up and in-container setup succeeded, and never
run `optimize` on the host.

Phase 1 is done when the target environment (host or container) is ready.

## Phase 2 — Workload intake

Enter only after Phase 0 and Phase 1 exit conditions hold. This is the first and
only phase that asks workload questions.

Now follow custom-advanced **Advanced Configuration**, **Default Values**, and
**Model Resolution**. Use the agent's structured question UI when available.
Never copy API keys into chat output.

| Field | CLI flag | Default | Notes |
|---|---|---|---|
| Model path | `--model` | required | Local dir with `config.json`, or HF cache |
| Framework | `--framework` | `sglang` | or `vllm`; prefer `.env` `FRAMEWORK` when set |
| TP / EP | `--tp` / `--ep` | `1` / `1` | tensor / expert parallel |
| CONC | `--conc` | `64` | client concurrency |
| ISL / OSL | `--isl` / `--osl` | `1024` / `1024` | input / output seq lengths |
| PRECISION | `--precision` | `bf16` | match checkpoint; `fp8` for FP8 models |
| MAX_HOURS | `--max-hours` | CLI `2.0` | skill recommends `8`; `0.5`–`3` for smoke |
| TARGET_GAIN | `--target-gain` | `30` | desired % gain |

**Optional:** `--no-kernel`, `--no-explore`, `--no-framework-agent`,
`--no-enable-conc-sweep`, `--no-enable-roofline`, `--gpu-type`, `--server-args`,
`--compare-against-gpu`, phase budget percentages, `--quantize` prelude.

Infer `PRECISION` from the model name when obvious (e.g. an `FP8` model implies
`--precision fp8`) and confirm it — do not silently keep the `bf16` default.

### Confirmation gate (required before Phase 3)

The Coordinator has no in-loop `setup` / `classify` — a value not asked here is
silently lost to its default. Before running any Phase 3 command, present the
full launch plan (including defaulted fields) and get explicit user confirmation:

```text
Launch plan — please confirm:
  MODEL_PATH    /wekafs/models/Qwen3-14B-FP8
  FRAMEWORK     vllm
  TP=1  EP=1  CONC=64
  ISL=1024  OSL=1024
  PRECISION=fp8
  MAX_HOURS=0.5   TARGET_GAIN=20%
  phases        --no-kernel (smoke)
  RUN_MODE      baremetal
```

Do **not** run `install.sh` or launch `optimize` until the user approves this plan.

### Persist the plan (required — shells do not share exports)

Agent shells do not persist exports between calls, so write the confirmed values
to `$RUN_DIR/workload.env` right after approval. Every Phase 3 block sources it;
without this, launch silently falls back to `${TP:-1}` / `${CONC:-64}` defaults
and `--model ""`. Fill each value from the approved plan.

```bash
export USER_DATA_PATH="${USER_DATA_PATH:?run /hyperloom-setup first}"
export RUN_DIR="${USER_DATA_PATH}/optimizer_runs"
mkdir -p "$RUN_DIR"
# Quoted heredoc (<<'EOF'): values are written literally, so a MODEL_PATH with
# spaces, $, or $(...) is not expanded or executed. Edit each value to the plan.
cat > "$RUN_DIR/workload.env" <<'EOF'
export MODEL_PATH=/wekafs/models/Qwen3-14B-FP8
export FRAMEWORK=vllm
export TP=1
export EP=1
export CONC=64
export ISL=1024
export OSL=1024
export PRECISION=fp8
export MAX_HOURS=0.5
export TARGET_GAIN=20
export OPT_FLAGS="--no-kernel"   # optional Phase 2 flags, space-separated; empty if none
EOF
```

## Phase 3 — Install (IR-2)

Resolve paths for wheel or source layout:

```bash
export REPO_ROOT="$(pwd -P)"
set -a; . "${REPO_ROOT}/.env"; set +a
export USER_DATA_PATH="${USER_DATA_PATH:?USER_DATA_PATH missing}"
. "${USER_DATA_PATH}/optimizer_runs/workload.env"   # confirmed Phase 2 values
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
ulimit -Sn 65536 || true

INSTALL_SH="${REPO_ROOT}/hyperloom/inference_optimizer/assets/install.sh"
[ -f "$INSTALL_SH" ] || INSTALL_SH="${REPO_ROOT}/src/hyperloom/inference_optimizer/assets/install.sh"

bash "$INSTALL_SH"
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH}/runtime/kernel-agent.env.sh}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
```

In Docker mode, run this inside the container.

## Phase 3 — Preflight (IR-1)

`install.sh` exports `$PYTHON`; the fallback below covers agent sandboxes that do
not persist exports between shell calls.

```bash
. "${USER_DATA_PATH}/optimizer_runs/workload.env"   # confirmed Phase 2 values
: "${MODEL_PATH:?MODEL_PATH empty — re-run the Persist the plan step}"
test -d "$MODEL_PATH" || { echo "ERROR: MODEL_PATH not a directory: $MODEL_PATH" >&2; exit 1; }
export PYTHON="${PYTHON:-$(command -v python3)}"
export IR1_VRAM_LIMIT_MIB="${IR1_VRAM_LIMIT_MIB:-500}"

"$PYTHON" - <<'PY'
import os, shutil, subprocess, sys

fail = False

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print("torch_cuda_available=", cuda_ok)
    print("torch_cuda_device_count=", torch.cuda.device_count())
    if not cuda_ok:
        print("gpu_unavailable=true")
        fail = True
except Exception as exc:
    print("torch_check_error=", type(exc).__name__, str(exc)[:300])
    fail = True

patterns = ("hyperloom.inference_optimizer.cli", "Magpie",
            "sglang.launch_server", "vllm.entrypoints")
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read()
    except Exception:
        continue
    text = cmd.replace(b"\0", b" ").decode("utf-8", "ignore")
    if text and any(p in text for p in patterns):
        print(f"existing_process {pid}: {text[:300]}")
        fail = True

# IR-1 VRAM gate: a GPU can hold VRAM without a matching process name above.
limit = int(os.environ.get("IR1_VRAM_LIMIT_MIB", "500"))
used = []
if shutil.which("amd-smi"):
    out = subprocess.run(["amd-smi", "metric", "-m", "--json"],
                         capture_output=True, text=True)
    if out.returncode == 0:
        import json
        for i, g in enumerate(json.loads(out.stdout or "[]")):
            mib = (g.get("mem_usage", {}) or {}).get("used_vram", {}).get("value")
            if mib is not None:
                used.append((i, int(mib)))
elif shutil.which("rocm-smi"):
    out = subprocess.run(["rocm-smi", "--showmeminfo", "vram", "--json"],
                         capture_output=True, text=True)
    if out.returncode == 0:
        import json, re
        for card, info in json.loads(out.stdout or "{}").items():
            raw = next((v for k, v in info.items() if "Used" in k), None)
            if raw is not None:
                used.append((card, int(int(re.sub(r"\D", "", str(raw)) or 0) / 1024 / 1024)))
else:
    print("vram_check_skipped=no_amd_smi_or_rocm_smi")

for dev, mib in used:
    marker = "OVER_LIMIT" if mib > limit else "ok"
    print(f"gpu {dev}: used_vram_mib={mib} ({marker})")
    if mib > limit:
        fail = True

sys.exit(1 if fail else 0)
PY
```

The script exits non-zero when stale serving processes are found or any GPU
holds more than `IR1_VRAM_LIMIT_MIB` (default 500) MiB. Also exit non-zero if
the model path is missing or GPUs are unavailable. If neither `amd-smi` nor
`rocm-smi` is present the VRAM gate is skipped — confirm GPUs are idle manually.
Never print API keys or tokens.

## Phase 3 — Launch

After IR-2 and IR-1 pass, launch with Phase 2 values as CLI flags.
`setsid nohup ... &` is required for runs longer than 5 minutes.

```bash
cd "$REPO_ROOT"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
export RUN_DIR="${USER_DATA_PATH}/optimizer_runs"
. "${RUN_DIR}/workload.env"   # confirmed Phase 2 values — no silent defaults
: "${MODEL_PATH:?MODEL_PATH empty — re-run the Persist the plan step}"
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH}/runtime/kernel-agent.env.sh}"
export PYTHON="${PYTHON:-$(command -v python3)}"
export PATH="$(dirname "$PYTHON"):/usr/local/bin:$PATH"

export RUN_TAG="$(basename "$MODEL_PATH")-$(date +%Y%m%d_%H%M%S)"
export RUN_LOG="$RUN_DIR/run_${RUN_TAG}.log"
export PID_FILE="$RUN_DIR/run_${RUN_TAG}.pid"
export LAUNCH_INFO_FILE="$RUN_DIR/launch_${RUN_TAG}.json"
mkdir -p "$RUN_DIR"

# shellcheck disable=SC2086
setsid nohup "$PYTHON" -m hyperloom.inference_optimizer.cli --verbose optimize \
  --model "$MODEL_PATH" \
  --framework "$FRAMEWORK" \
  --tp "$TP" \
  --ep "$EP" \
  --conc "$CONC" \
  --isl "$ISL" \
  --osl "$OSL" \
  --precision "$PRECISION" \
  --max-hours "$MAX_HOURS" \
  --target-gain "$TARGET_GAIN" \
  --tick-interval-sec 30 \
  --launch-info-file "$LAUNCH_INFO_FILE" \
  ${OPT_FLAGS:-} \
  > "$RUN_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
```

Every value comes from the confirmed `workload.env`; there are no `${VAR:-default}`
fallbacks so a missing value fails loudly instead of silently launching a wrong
config. Put any optional Phase 2 flags (`--no-kernel`, `--no-explore`,
`--gpu-type`, `--model-class`, `--server-args`, `--compare-against-gpu`,
`--quantize`, phase budget flags) into `OPT_FLAGS` in `workload.env`. `OPT_FLAGS`
is word-split (unquoted `${OPT_FLAGS}`), so quote any flag value that contains
spaces, e.g. `export OPT_FLAGS='--server-args "--foo bar"'`.

### Launch health check (30 s after start)

The `$!` recorded at launch is the **setsid wrapper** PID, which exits
immediately — it is NOT the optimizer. Read the real `.pid` (and `.session_dir`)
from the launch-info JSON with a tiny `python3` reader (no `jq` dependency) and
rewrite `$PID_FILE` so the monitor watches the right process.

```bash
sleep 30
read_json() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],''))" "$1" "$2" 2>/dev/null; }

REAL_PID="$(read_json "$LAUNCH_INFO_FILE" pid)"
[ -z "$REAL_PID" ] && REAL_PID="$(pgrep -f 'hyperloom.inference_optimizer.cli .*optimize' | head -1)"
[ -n "$REAL_PID" ] && echo "$REAL_PID" > "$PID_FILE"
test -d "/proc/$REAL_PID" && echo "optimizer_alive=true pid=$REAL_PID"

SESSION_DIR="$(read_json "$LAUNCH_INFO_FILE" session_dir)"
if [ -z "$SESSION_DIR" ]; then
  echo "ERROR: no session_dir yet in $LAUNCH_INFO_FILE; inspect $RUN_LOG" >&2
  exit 1
fi
test -f "$SESSION_DIR/manifest.json" && echo "manifest_present=true session_dir=$SESSION_DIR"
test -f "$SESSION_DIR/state.json" && echo "state_exists=true"
```

Never guess `session_dir` by timestamp — always read it from the launch-info
JSON.

## Phase 3 — Monitor

Poll at most every 5 minutes unless debugging a startup failure.

```bash
# Reuse the launch-info reader instead of hand-filling SESSION_DIR.
read_json() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],''))" "$1" "$2" 2>/dev/null; }
export SESSION_DIR="${SESSION_DIR:-$(read_json "$LAUNCH_INFO_FILE" session_dir)}"
: "${SESSION_DIR:?SESSION_DIR unknown — inspect $LAUNCH_INFO_FILE}"
python3 - <<'PY'
import json, os, pathlib
s = json.loads((pathlib.Path(os.environ["SESSION_DIR"]) / "state.json").read_text())
for k in ("stop_reason", "baseline_tput", "cumulative_gain", "current_best",
          "last_kernel_opt", "phase"):
    print(f"{k}: {s.get(k)}")
print("explore_last_round:", s.get("explore_search", {}).get("last_round"))
PY
```

Report session id + log path, `baseline_tput` / `current_best` /
`cumulative_gain`, explore accepted/rejected, last kernel opt (correctness,
speedup, KEEP/REVERT), and process-alive vs `stop_reason`. See
[reference.md](reference.md) Report fields.

## Resume

Resume runs in a fresh shell, so re-establish the same environment as launch,
re-run the IR-2/IR-1 gates, and resume the **explicit** session — bare
`--resume` auto-picks the latest session and may target the wrong run.

```bash
cd "$REPO_ROOT"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
export RUN_DIR="${USER_DATA_PATH}/optimizer_runs"
. "${RUN_DIR}/workload.env"   # confirmed Phase 2 values
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH}/runtime/kernel-agent.env.sh}"
export PYTHON="${PYTHON:-$(command -v python3)}"
export PATH="$(dirname "$PYTHON"):/usr/local/bin:$PATH"

# Resume the exact session recorded from launch (never guess by timestamp).
: "${SESSION_DIR:?SESSION_DIR unknown — read .session_dir from the launch-info JSON}"

# Re-run IR-2 (install) unless the carve-out holds, then IR-1 (GPU preflight)
# in this shell before resuming — same as a fresh launch.

export RESUME_TAG="resume-$(date +%Y%m%d_%H%M%S)"
export RESUME_LOG="$RUN_DIR/run_${RESUME_TAG}.log"
# shellcheck disable=SC2086
setsid nohup "$PYTHON" -m hyperloom.inference_optimizer.cli --verbose optimize \
  --resume --resume-from "$SESSION_DIR" \
  --tick-interval-sec 30 \
  --launch-info-file "$RUN_DIR/launch_${RESUME_TAG}.json" \
  ${OPT_FLAGS:-} \
  > "$RESUME_LOG" 2>&1 < /dev/null &
```

Resume writes its own log (`run_resume-*.log`) so the original run log is
preserved. Reuse the IR-2 carve-out rules; re-run `install.sh` if the shell or
env changed. Run the same launch health check afterward to capture the real
optimizer `.pid`.

| `stop_reason` | Action |
|---|---|
| `time_exhausted` | `--resume` same session |
| `no_more_leverage` | stop; resume only if user changes strategy |
| `policy_loop` | inspect `policy_denial_history`; clear stale prunes |

## Expected optimizer flow

1. Establish `baseline_tput`.
2. Coordinator runs roofline/profile analysis after baseline.
3. `explore` tests serving parameters incrementally.
4. Kernel-agent runs on hot paths with compile + correctness evidence.
5. `sweep` validates concurrency around the best candidate.
6. Final report under `$SESSION_DIR/reports/`.

## When to defer

- **Plain serving only** — use `serving-llms-on-instinct`.
- **ROCm driver broken** — diagnose the ROCm stack first (e.g. a `rocm-doctor`
  skill if published); do not start the optimizer on a broken driver.
- **Edge cases** — read `@${HYPERLOOM_SKILL_PATH}` for multi-node, atom
  framework (IR-8), critic/robustness backends, cache topology, and the
  full failure matrix.

## Further reading

- Bootstrap detail: [setup.md](setup.md)
- Iron Rules + CLI reference: [reference.md](reference.md)
- Authoritative runtime skill: `hyperloom/inference_optimizer/SKILL.md`
