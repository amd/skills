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
`$USER_DATA_PATH` (default `/workspace/hyperloom`). This is independent of the
install directory (`INSTALL_DIR`, where the wheel and `.env` live) and may point
to shared storage.

```text
$USER_DATA_PATH/
├── runtime/                          # install.sh outputs, kernel-agent.env.sh
├── logs/
└── <model_basename>/
    └── <UTC_YYYYMMDDTHHMMSSZ>/
        ├── manifest.json
        ├── state.json
        ├── runs/
        ├── reports/
        └── optimizer_runs/
```

## Workflow overview

Match `hyperloom-custom-advanced` section order — do **not** ask workload
questions while writing `.env` or during `/hyperloom-setup`.

```
Phase 0   Bootstrap        pip install, /hyperloom-setup → .env (credentials + run mode only)
Phase 1   Environment      custom-advanced §Setup Configuration
                          baremetal: confirm host ready after install_baremetal.sh
                          docker:    start container + run setup inside it
Phase 2   Workload intake  custom-advanced §Advanced Configuration → Model Resolution
                          → show launch plan → user confirms
Phase 3   Execute          install.sh → preflight → launch → monitor → report
```

Load `hyperloom-custom-advanced` at Phase 1 and follow its sections in order.
Phase 3 launch/monitor commands are below; for deeper optimizer behavior read
`@${HYPERLOOM_SKILL_PATH}` (`inference_optimizer`). Iron Rules + CLI reference:
[reference.md](reference.md).

Discovery paths for `hyperloom-custom-advanced`:

- `.cursor/skills/hyperloom-custom-advanced/SKILL.md`
- `.claude/skills/hyperloom-custom-advanced/SKILL.md`
- `.agents/skills/hyperloom-custom-advanced/SKILL.md`

## Iron Rules (launcher gates)

Run order is always **IR-2 → IR-1 → launch**.

**IR-1 — GPU unoccupied.** Before every `optimize` (fresh or `--resume`), verify
every visible GPU has zero foreign serving PIDs and ≲ 500 MiB VRAM in use.
Leftover `sglang.launch_server` / `vllm.entrypoints` / `Magpie` processes
silently degrade the next baseline.

**IR-2 — install.sh before launch.** Run `install.sh` and source
`kernel-agent.env.sh` in the **same shell** that spawns `optimize`. Skipping
install fails after baseline: missing TraceLens/GEAK, hung Ray tasks, or `401`
on kernel-opt gateway calls.

**Resume carve-out:** `--resume` may skip install only when `install.sh` exited 0
earlier in the same shell, `kernel-agent.env.sh` is still sourced, and the
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

**Docker (`HYPERLOOM_RUN_MODE=docker`):** follow custom-advanced to pick the
image, `docker run` a long-running container, and run the setup backend inside
the container (`--install-framework none --yes`). Do not ask workload questions
until the container is up and in-container setup succeeded. Do not run
`optimize` on the host.

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
| MAX_HOURS | `--max-hours` | `8` | `0.5`–`3` for smoke |
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

## Phase 3 — Install (IR-2)

Resolve paths for wheel or source layout:

```bash
export REPO_ROOT="$(pwd -P)"
set -a; . "${REPO_ROOT}/.env"; set +a
export USER_DATA_PATH="${USER_DATA_PATH:?USER_DATA_PATH missing}"
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
export MODEL_PATH=/path/to/model   # from Phase 2
test -d "$MODEL_PATH"
export PYTHON="${PYTHON:-$(command -v python3)}"
export IR1_VRAM_LIMIT_MIB="${IR1_VRAM_LIMIT_MIB:-500}"

"$PYTHON" - <<'PY'
import os, shutil, subprocess, sys

fail = False

try:
    import torch
    print("torch_cuda_available=", torch.cuda.is_available())
    print("torch_cuda_device_count=", torch.cuda.device_count())
except Exception as exc:
    print("torch_check_error=", type(exc).__name__, str(exc)[:300])

patterns = ("hyperloom.inference_optimizer.cli", "Magpie", "sglang.launch_server")
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
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH}/runtime/kernel-agent.env.sh}"
export PYTHON="${PYTHON:-$(command -v python3)}"
export PATH="$(dirname "$PYTHON"):/usr/local/bin:$PATH"

export RUN_TAG="$(basename "$MODEL_PATH")-$(date +%Y%m%d_%H%M%S)"
export RUN_DIR="${USER_DATA_PATH}/optimizer_runs"
export RUN_LOG="$RUN_DIR/run_${RUN_TAG}.log"
export PID_FILE="$RUN_DIR/run_${RUN_TAG}.pid"
export LAUNCH_INFO_FILE="$RUN_DIR/launch_${RUN_TAG}.json"
mkdir -p "$RUN_DIR"

setsid nohup python3 -m hyperloom.inference_optimizer.cli --verbose optimize \
  --model "$MODEL_PATH" \
  --framework "${FRAMEWORK:-sglang}" \
  --tp "${TP:-1}" \
  --ep "${EP:-1}" \
  --conc "${CONC:-64}" \
  --isl "${ISL:-1024}" \
  --osl "${OSL:-1024}" \
  --precision "${PRECISION:-bf16}" \
  --max-hours "${MAX_HOURS:-8}" \
  --target-gain "${TARGET_GAIN:-30}" \
  --tick-interval-sec 30 \
  --launch-info-file "$LAUNCH_INFO_FILE" \
  > "$RUN_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
```

Append optional flags only when Phase 2 resolved them: `--no-kernel`,
`--no-explore`, `--gpu-type`, `--model-class`, `--server-args`,
`--compare-against-gpu`, `--quantize`, phase budget flags, etc.

### Launch health check (30 s after start)

```bash
sleep 30
pid="$(cat "$PID_FILE")"
test -d "/proc/$pid" && echo "optimizer_alive=true pid=$pid"

session_dir="$(jq -r '.session_dir // empty' "$LAUNCH_INFO_FILE" 2>/dev/null)"
if [ -z "$session_dir" ]; then
  echo "ERROR: no session_dir in $LAUNCH_INFO_FILE; inspect $RUN_LOG" >&2
  exit 1
fi
test -f "$session_dir/manifest.json" && echo "manifest_present=true session_dir=$session_dir"
test -f "$session_dir/state.json" && echo "state_exists=true"
```

Never guess `session_dir` by timestamp — always read it from the launch-info
JSON.

## Phase 3 — Monitor

Poll at most every 5 minutes unless debugging a startup failure.

```bash
export SESSION_DIR="<session_dir from launch-info JSON>"
python3 - <<'PY'
import json, os, pathlib
s = json.loads((pathlib.Path(os.environ["SESSION_DIR"]) / "state.json").read_text())
for k in ("stop_reason", "baseline_tput", "cumulative_gain", "current_best",
          "last_kernel_opt", "phase"):
    print(f"{k}: {s.get(k)}")
print("explore_last_round:", s.get("explore_search", {}).get("last_round"))
PY
```

Report to the user:

- session id (`manifest.json`) and log path (`$RUN_LOG`)
- `baseline_tput`, `current_best`, `cumulative_gain`
- explore accepted/rejected summary
- last kernel opt: correctness, speedup, KEEP/REVERT
- process alive vs `stop_reason`

## Resume

When the user asks to resume and workload is unchanged:

```bash
export RESUME_TAG="resume_$(date +%Y%m%d_%H%M%S)"
export RESUME_LOG="$RUN_DIR/run_${RESUME_TAG}.log"
setsid nohup python3 -m hyperloom.inference_optimizer.cli --verbose optimize \
  --resume \
  --tick-interval-sec 30 \
  --launch-info-file "$RUN_DIR/launch_${RESUME_TAG}.json" \
  > "$RESUME_LOG" 2>&1 < /dev/null &
```

Resume writes its own log (`run_resume_*.log`) so the original run log is
preserved for comparison.

`--resume` auto-picks the latest session under `$USER_DATA_PATH/<model>/`.
Reuse IR-2 carve-out rules; re-run `install.sh` if the shell or env changed.

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
- **ROCm driver broken** — use `rocm-doctor` when available.
- **Edge cases** — read `@${HYPERLOOM_SKILL_PATH}` for multi-node, atom
  framework (IR-8), critic/robustness backends, cache topology, and the
  full failure matrix.

## Further reading

- Bootstrap detail: [setup.md](setup.md)
- Iron Rules + CLI reference: [reference.md](reference.md)
- Authoritative runtime skill: `hyperloom/inference_optimizer/SKILL.md`
