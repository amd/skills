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
  tokens/sec or throughput, optimize or tune vLLM or SGLang on MI300X/MI325X/MI355X,
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
parameters, then install, launch, and monitor the optimizer. This skill owns the
orchestration and the launcher gates; environment prep and workload intake are
delegated to the skills the Hyperloom wheel installs, and
`@${HYPERLOOM_SKILL_PATH}` (`inference_optimizer`) is the execution baseline.

Do not manually optimize inside chat unless debugging.

## Prerequisites

- AMD Instinct GPU host (MI300X / MI325X / MI355X) with ROCm
- `/dev/kfd` and `/dev/dri` present; `amd-smi` or `rocm-smi` works
- Python 3.10+ and network access to install the Hyperloom wheel
- Anthropic (or compatible) LLM credentials for agent backends
- A dedicated agent workspace directory

Every command in this skill runs on that GPU host. Confirm the shell you are in
is on it before Phase 0, so a bootstrap does not land on a machine with no GPU.

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
- **Phase 1 Environment** — custom-advanced §Setup Configuration (baremetal: confirm host; docker: start container + setup inside, contract in [setup.md](setup.md))
- **Phase 2 Workload intake** — custom-advanced §Advanced Configuration → Model Resolution → show launch plan → user confirms
- **Phase 3 Execute** — install.sh → preflight → launch → monitor → report

Load `hyperloom-custom-advanced` at Phase 1 and follow its sections in order
(discovery: `.cursor/` / `.claude/` / `.agents/skills/hyperloom-custom-advanced/SKILL.md`).
If it is not on disk, stop and tell the user to restart the agent so the newly
installed skills are picked up — do not improvise the environment or workload
sections from memory, since the wheel is the source of truth for both.
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

The runtime is published to PyPI as `hyperloom-inference-optimizer`. List the
releases, tell the user the newest one, and ask whether to install it or a
version they name.

List with `--pre` so prereleases are visible, and install an exact `==` version
so a later bootstrap installs the same runtime.

```bash
cd "$INSTALL_DIR"   # the directory confirmed above
pip index versions hyperloom-inference-optimizer --pre
pip install hyperloom-inference-optimizer==<version the user approved> --target .
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
| MAX_HOURS | `--max-hours` | CLI `2.0` | offer `3` (quick) or `12` (full); see below |
| TARGET_GAIN | `--target-gain` | `30` | desired % gain |

**Optional:** `--no-explore`, `--no-enable-conc-sweep`, `--gpu-type`,
`--server-args`, `--compare-against-gpu`, `--quantize` prelude.

Infer `PRECISION` from the model name when obvious (e.g. an `FP8` model implies
`--precision fp8`) and confirm it — do not silently keep the `bf16` default.

### Budget and flags — offer these three

Offer all three and let the user pick one. The flags in each are a set: pass them
together, and do not ask for a budget and then ask separately which phases to run.
The two demos take the workload and flags of the Hyperloom demo skill of the same
budget — treat those as given and skip the table above. The user may name their
own model instead of the demo's; for the 3-hour demo keep it at 8B or below.
Confirm everything in the launch plan. Only **Custom** collects workload answers.

**1. 3-hour demo** (`hyperloom-qwen3-8b-3h`) — `Qwen/Qwen3-8B` unless the user
names another 8B-or-smaller model, TP=1, CONC=64, ISL=OSL=1024,
`--precision bf16`, serving and config parameters only, no kernel rewrites.
Resolve the model per custom-advanced Model Resolution; download it from Hugging
Face when it is not already local. Match `--precision` to the chosen checkpoint.
Expect a modest validated gain, or an honest 0% when the workload has no
parameter headroom.

```text
--max-hours 3 --precision bf16
--no-framework-agent --no-kernel --no-enable-conc-sweep --no-enable-roofline
--max-minutes-explore-pct 0.39 --max-minutes-sweep-pct 0.01
--explore-force-exit-budget-pct 0.01 --explore-force-exit-hours-remaining 0.05
```

**2. 12-hour demo** (`hyperloom-qwen3-14b-fp8-12h`) — `Qwen/Qwen3-14B-FP8`
unless the user names another model, TP=1, CONC=64, ISL=OSL=1024,
`--precision fp8` matched to the chosen checkpoint, every lever with kernel
rewrites included. The kernel agent needs room to profile, rewrite and
revalidate, which is where the larger gains come from.

```text
--max-hours 12 --precision fp8
--max-minutes-framework-pct 0.01 --max-minutes-explore-pct 0.42
--max-minutes-kernel-pct 0.42
```

**3. Custom** — the user brings their own model or workload instead of taking a
demo. Walk through the fields in the table above and the phase toggles, one
question at a time, and derive the flags from the answers rather than asking for
flags. Whichever levers they pick, a budget of 3 hours or less keeps the 3-hour
demo's flag set.
Optional flags come from the list above; show the full flag list in the launch
plan either way.

### Confirmation gate (required before Phase 3)

The Coordinator has no in-loop `setup` / `classify` — a value not asked here is
silently lost to its default. Before running any Phase 3 command, present the
full launch plan (including defaulted fields) and get explicit user confirmation.

Print the plan in the reply body as this aligned block:

```text
Launch plan — please confirm:
  MODEL_PATH    /wekafs/models/Qwen3-14B-FP8
  FRAMEWORK     vllm
  TP=1  EP=1  CONC=64
  ISL=1024  OSL=1024
  PRECISION=fp8
  MAX_HOURS=3     TARGET_GAIN=20%
  profile       3-hour demo — no kernel, no framework agent, no roofline
  flags         --no-framework-agent --no-kernel --no-enable-conc-sweep
                --no-enable-roofline
                --max-minutes-explore-pct 0.39 --max-minutes-sweep-pct 0.01
                --explore-force-exit-budget-pct 0.01
                --explore-force-exit-hours-remaining 0.05
  RUN_MODE      baremetal
```

Never put the plan inside the confirmation prompt itself. A prompt renders as one
wrapped paragraph, which collapses the alignment above into an unreadable blob the
user has to search for `MAX_HOURS` in. Keep the prompt to a single short question
such as `Approve this launch plan?`, and if you offer a "change something" option,
name the field to change rather than making the user retype it as free text.

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
export MAX_HOURS=3
export TARGET_GAIN=20
# The whole flag set for the approved profile, space-separated. The 3-hour
# demo is shown; a 12-hour run swaps in its own set.
export OPT_FLAGS="--no-framework-agent --no-kernel --no-enable-conc-sweep --no-enable-roofline --max-minutes-explore-pct 0.39 --max-minutes-sweep-pct 0.01 --explore-force-exit-budget-pct 0.01 --explore-force-exit-hours-remaining 0.05"
EOF
```

## Phase 3 — Install (IR-2)

`INSTALL_DIR` is the directory confirmed in Phase 0, the one holding `hyperloom/`
and `.env`. Every Phase 3 block below rebuilds it from the current directory, so
run them from there; the check refuses a directory that is not it.

Resolve paths for wheel or source layout:

```bash
export INSTALL_DIR="$(pwd -P)"
[ -d "${INSTALL_DIR}/hyperloom" ] || [ -d "${INSTALL_DIR}/src/hyperloom" ] || {
  echo "ERROR: ${INSTALL_DIR} holds no hyperloom/ -- cd to the Phase 0 install directory" >&2; exit 1; }
set -a; . "${INSTALL_DIR}/.env"; set +a
export USER_DATA_PATH="${USER_DATA_PATH:?USER_DATA_PATH missing}"
. "${USER_DATA_PATH}/optimizer_runs/workload.env"   # confirmed Phase 2 values
export PYTHONPATH="${INSTALL_DIR}:${PYTHONPATH:-}"
ulimit -Sn 65536 || true

INSTALL_SH="${INSTALL_DIR}/hyperloom/inference_optimizer/assets/install.sh"
[ -f "$INSTALL_SH" ] || INSTALL_SH="${INSTALL_DIR}/src/hyperloom/inference_optimizer/assets/install.sh"

bash "$INSTALL_SH"
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH}/runtime/kernel-agent.env.sh}"
export PYTHONPATH="${INSTALL_DIR}:${PYTHONPATH:-}"
```

In Docker mode, run this inside the container.

## Phase 3 — Preflight (IR-1)

`install.sh` exports `$PYTHON`; the fallback below covers agent sandboxes that do
not persist exports between shell calls.

```bash
export SKILL_DIR="${SKILL_DIR:?absolute path of the directory holding this SKILL.md}"
. "${USER_DATA_PATH}/optimizer_runs/workload.env"   # confirmed Phase 2 values
export PYTHON="${PYTHON:-$(command -v python3)}"
"$PYTHON" "${SKILL_DIR}/scripts/preflight.py"
```

The gate exits non-zero — do not launch — when `MODEL_PATH` is missing or has no
`config.json`, torch sees no GPU, a foreign serving process still holds a card,
or any GPU holds more than `IR1_VRAM_LIMIT_MIB` (default 500) MiB.

It also blocks when VRAM cannot be read at all: no `amd-smi`/`rocm-smi` on
`PATH`, a probe that exits non-zero, or output it cannot parse. An unreadable
probe cannot rule out a busy GPU, and a foreign process holding VRAM under a
different name would slip through. Confirm the GPUs are idle by hand before
re-running with `IR1_ALLOW_UNVERIFIED_VRAM=1`.

Never print API keys or tokens. `scripts/tests/test_preflight.py` covers the
probe shapes this gate must reject.

## Phase 3 — Launch

After IR-2 and IR-1 pass, launch. `setsid nohup` is required for runs longer than
5 minutes, so the run outlives the agent shell.

```bash
export INSTALL_DIR="$(pwd -P)"
export SKILL_DIR="${SKILL_DIR:?absolute path of the directory holding this SKILL.md}"
bash "${SKILL_DIR}/scripts/launch.sh"
```

Every workload value comes from the confirmed `workload.env`; the script has no
`${VAR:-default}` fallbacks, so a missing value fails loudly instead of launching
a different config. Put any optional Phase 2 flags (`--no-kernel`, `--no-explore`,
`--gpu-type`, `--model-class`, `--server-args`, `--compare-against-gpu`,
`--quantize`, phase budget flags) into `OPT_FLAGS` in `workload.env`. `OPT_FLAGS`
is word-split, so quote any flag value that contains spaces, e.g.
`export OPT_FLAGS='--server-args "--foo bar"'`.

### Launch health check (30 s after start)

Required after every launch and resume. The PID recorded at launch is the
**setsid wrapper**, which exits immediately — it is NOT the optimizer. This reads
the real `.pid` and `.session_dir` from the launch-info JSON, rewrites the PID
file so the monitor watches the right process, and records both in
`$RUN_DIR/last_launch.env` for the later phases.

```bash
export INSTALL_DIR="$(pwd -P)"
export SKILL_DIR="${SKILL_DIR:?absolute path of the directory holding this SKILL.md}"
bash "${SKILL_DIR}/scripts/launch_health.sh"
```

It exits non-zero when the launch-info JSON never appeared, no optimizer process
can be found, or `session_dir` is still unset — inspect the reported run log in
those cases. Never guess `session_dir` from a timestamp; concurrent sessions
share `USER_DATA_PATH`.

## Phase 3 — Monitor

Poll at most every 5 minutes unless debugging a startup failure. Use the state
reader the wheel ships rather than parsing `state.json` by hand — it also prints
the recent lifecycle events.

```bash
export INSTALL_DIR="$(pwd -P)"
. "${USER_DATA_PATH}/optimizer_runs/last_launch.env"   # SESSION_DIR from launch
STATE_TOOL="${INSTALL_DIR}/hyperloom/inference_optimizer/tools/read_optimizer_state.py"
[ -f "$STATE_TOOL" ] || STATE_TOOL="${INSTALL_DIR}/src/hyperloom/inference_optimizer/tools/read_optimizer_state.py"
"${PYTHON:-python3}" "$STATE_TOOL" "$SESSION_DIR"
```

For recent action counts grouped by category, the wheel also ships
`tools/event_counts.py`, invoked the same way.

Report session id + log path, `baseline_tput` / `current_best` /
`cumulative_gain`, explore accepted/rejected, last kernel opt (correctness,
speedup, KEEP/REVERT), and process-alive vs `stop_reason`. See
[reference.md](reference.md) Report fields.

## Resume

Resume runs in a fresh shell. Re-run the IR-2 and IR-1 gates first, exactly as for
a fresh launch — the script does not re-check them.

```bash
export INSTALL_DIR="$(pwd -P)"
export SKILL_DIR="${SKILL_DIR:?absolute path of the directory holding this SKILL.md}"
bash "${SKILL_DIR}/scripts/resume.sh"
bash "${SKILL_DIR}/scripts/launch_health.sh"
```

It resumes the session recorded in `last_launch.env` and always passes
`--resume-from` explicitly, because a bare `--resume` auto-picks the newest
session and can target the wrong run. Resume writes its own log
(`run_resume-*.log`) so the original run log is preserved. Reuse the IR-2
carve-out rules; re-run `install.sh` if the shell or env changed.

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
