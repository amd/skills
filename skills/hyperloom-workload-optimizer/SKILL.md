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

You are the entry point for Hyperloom optimization on AMD Instinct GPUs. Your job
is the bootstrap: confirm the workspace, install the Hyperloom wheel, run
`/hyperloom-setup`, then hand the run to the skill that owns it.

The wheel installs the skills that own everything after setup: `hyperloom-setup`
for credentials and run mode, the demo skills for a workload preset, and
`inference_optimizer` for the launcher gates and monitoring. They ship with the
runtime, so they always match the installed version.

## Out of scope for this skill

- Do not run `python -m hyperloom.inference_optimizer.cli optimize` yourself.
- Do not implement a GPU preflight, launcher gate, or background launch here. The
  installed skills own those, including the Iron Rules and the resume path.
- Do not ask for workload values (model, TP/EP, concurrency, ISL/OSL, precision,
  objective, budget) while installing or while setup is running. They belong to
  the run skill, after setup finishes.
- Do not optimize by hand in chat.

## Prerequisites

- AMD Instinct GPU host (MI300X / MI325X / MI355X) with ROCm, `/dev/kfd` and
  `/dev/dri` present, and `amd-smi` or `rocm-smi` working.
- Python 3.10+ and `pip` on the machine that runs the install.
- LLM credentials: Anthropic API access, or the AMD LLM gateway.
- A dedicated empty directory, opened in the agent as the workspace.

Confirm the shell you are in is on the GPU host before installing. Setup may later
point Docker at a different target host; until it does, everything here runs where
the agent is.

## Step 1: Confirm the workspace

The current directory is both the install target and the agent workspace. Confirm
with the user that it is a dedicated directory before installing: setup creates or
updates `.env` in it. Do not switch to another directory on your own, and do not
install into an existing project unless the user accepts the `.env` change.

## Step 2: Install the Hyperloom wheel

```bash
pip install hyperloom-inference-optimizer --target .
```

Install the current release unless the user asks for a specific version. It is
normal for the directory to hold many Python package folders afterwards; the user
does not need to inspect them.

## Step 3: Run `/hyperloom-setup`

The wheel installs `hyperloom-setup` into the agent's skill directories
(`.agents/skills/`, `.claude/skills/`, `.cursor/skills/`). Run it:

```text
/hyperloom-setup
```

It is interactive and owns credentials, `USER_DATA_PATH`, the run mode
(`docker` recommended, or `baremetal`), the Docker target host, and the bare-metal
framework install. It writes `.env` and stops before any optimization. Run it once
per workspace; the run skills reuse those values.

Let setup ask its own questions. Do not preempt them, do not restate its option
lists, and never ask the user to paste an API key into chat.

If the agent does not list `hyperloom-setup` after the install, the skill
directories were written after the agent scanned them. Tell the user to restart
the agent, then run it again. Do not substitute your own setup steps.

## Step 4: Hand off to a run skill

Setup ends by offering a run and loading the matching skill, so normally you just
follow it. When the user asks for a run directly, load the skill by name and follow
its instructions instead of this one:

- `hyperloom-qwen3-8b-3h` — short no-kernel Qwen3-8B run; best first end-to-end check.
- `hyperloom-qwen3-14b-fp8-12h` — medium-length Qwen3-14B-FP8 run.
- `hyperloom-qwen3-14b-fp8-12h-forge` — the same run on the KernelForge kernel backend.
- `hyperloom-custom-advanced` — explicit model, framework, workload, budget, and phase toggles.

A preset keeps its workload even if the user supplies their own `MODEL_PATH`;
tensor parallelism, concurrency, sequence lengths, precision, and budget are not
retuned for that model. When those need to change, use `hyperloom-custom-advanced`.

To resume a stopped session, follow the optimizer skill at `.env`
`HYPERLOOM_SKILL_PATH`; it owns the resume path and the gates a relaunch still has
to clear.

## What to expect during a run

Optimization runs for hours in the background. Do not stream the log.

Before launch the run skill shows a plan: resolved model path, run mode, framework,
TP, concurrency, ISL/OSL, precision, budget, and `USER_DATA_PATH`. Get the user's
go-ahead on that plan before the optimizer starts — it then owns the GPU for hours.
After launch it reports the optimizer PID, run log, launch-info JSON, session
directory, `state.json`, and the first health check.

During the run, report a short status about every 300 seconds: process alive,
current phase, `stop_reason`, baseline and current best throughput, cumulative
gain, the latest benchmark or candidate decision, and the most relevant log lines.
Never print API keys, tokens, or custom headers.

## Troubleshooting

- Many package folders in the workspace after `pip install --target .` is expected.
- `/hyperloom-setup` not listed: the install landed after the agent scanned for
  skills. Restart the agent and check `.claude/skills/hyperloom-setup/` exists.
- `ImportError: libamdhip64.so.7` or `libhipblas.so.3`: the framework torch wheel
  wants different ROCm user-space libraries; align `ROCM_PATH` and
  `LD_LIBRARY_PATH`.
- `hipDeviceAttributePciChipId` missing during an AITER build: `hipcc` is using
  older ROCm headers; put the matching ROCm `bin` first on `PATH`.
- Anything past setup (preflight failures, launch, phases, gains) belongs to the
  installed run and optimizer skills. Read those rather than reproducing their
  checks here.
