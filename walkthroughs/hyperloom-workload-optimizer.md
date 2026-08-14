# AMD Skills Walkthroughs: `hyperloom-workload-optimizer`

This skill teaches your AI agent to set up a Hyperloom workspace and autonomously
optimize end-to-end LLM inference throughput on AMD Instinct GPUs (MI300X /
MI308X / MI325X / MI355X).

**What you'll end up with:** a running Hyperloom optimization session with
`manifest.json`, `state.json`, benchmark runs under `runs/`, and a final report
under `reports/` showing validated throughput gain over baseline.

Step 3 gives three ways to run: a 3-hour demo over serving parameters, a 12-hour
demo that adds kernel rewrites, and a custom run for any other budget or mix.
Each comes as a ready-to-paste prompt with its flags.

## Prerequisites

**Hardware**

- AMD Instinct GPU with ROCm (`/dev/kfd`, `/dev/dri`)
- Sufficient VRAM for the target model at the chosen TP degree

**Software**

- Python 3.10+
- An agentic runner: **Cursor**, **Claude Code**, or **Codex**
- Anthropic API access (or AMD LLM gateway) for Hyperloom agent backends
- Docker (recommended) or bare-metal ROCm + serving framework

**Workspace**

- A dedicated empty directory opened as the agent workspace

## Step 1 — Enable the skill

**Claude Code:**

```bash
npx skills add amd/skills --skill hyperloom-workload-optimizer --agent claude-code
```

**Cursor:** install the `amd-skills` plugin from the AMD skills marketplace, or
copy `skills/hyperloom-workload-optimizer/` into your project's
`.cursor/skills/` directory.

Confirm the skill is visible:

```text
Which skills do you see?
```

You should see `hyperloom-workload-optimizer` in the list.

## Step 2 — Bootstrap the workspace

In the dedicated workspace, ask the agent:

```text
Set up Hyperloom and prepare to optimize vLLM throughput on MI300X.
```

The agent works through four phases in order and should not ask workload
questions before the environment is ready:

1. **Phase 0 — Bootstrap:** confirm the install directory, install a pinned
   `hyperloom-inference-optimizer` release from PyPI with `pip install --target .`,
   then run `/hyperloom-setup` to write `.env` (credentials + run mode only).
2. **Phase 1 — Environment prep:** load `hyperloom-custom-advanced` Setup
   Configuration. On bare metal, confirm the host is ready; in Docker, start a
   long-running container and run the in-container setup first.
3. **Phase 2 — Workload intake:** only now ask for workload parameters (model,
   framework, TP, conc, ISL/OSL, precision, budget) and confirm a launch plan.
4. **Phase 3 — Execute:** run `install.sh` (IR-2) and the GPU preflight (IR-1),
   then launch and monitor.

Verify:

```bash
ls hyperloom/inference_optimizer/assets/install.sh
test -f .env && grep -q HYPERLOOM_SKILL_PATH .env
```

## Step 3 — Launch an optimization

There are three ways to run. Paste the prompt that matches, with your own model
path — the flags in each are a set, so pass them together.

**1. 3-hour demo.** Serving and config parameters only; the kernel agent is off.
Expect a modest validated gain, or an honest 0% when the workload has none.

```text
Optimize /path/to/Qwen3-8B with vLLM on MI300X: TP=1, conc=64, ISL=1024, OSL=1024,
max-hours 3, serving parameters only: --no-kernel --no-framework-agent
--no-enable-roofline --explore-force-exit-hours-remaining 0.05
--explore-force-exit-budget-pct 0.01 --max-minutes-explore-pct 0.46
--max-minutes-sweep-pct 0.01. Launch and monitor.
```

**2. 12-hour demo.** Every lever, kernel rewrites included. The kernel agent
needs room to profile, rewrite and revalidate hot kernels, which is where the
larger gains come from.

```text
Optimize /path/to/Qwen3-8B with vLLM on MI300X: TP=1, conc=64, ISL=1024, OSL=1024,
max-hours 12, all components enabled, with --max-minutes-framework-pct 0.01
--max-minutes-explore-pct 0.42 --max-minutes-kernel-pct 0.42. Launch and monitor.
```

**3. Custom.** Any other budget or mix of levers. Say what you want and the agent
starts from whichever demo matches the levers, changing only the hours:

```text
Optimize /path/to/Qwen3-8B with vLLM on MI300X: TP=1, conc=64, ISL=1024, OSL=1024,
max-hours 6, serving parameters only. Launch and monitor.
```

Model, framework, TP, concurrency, sequence lengths, precision and target gain
are asked for during intake in every case, so change those to suit your workload.

Once workload values are resolved, the agent should run `install.sh` (IR-2),
the GPU preflight (IR-1), launch `hyperloom.inference_optimizer.cli optimize`
with those values as CLI flags, then poll `state.json`.

## Step 4 — Read results

When the session stops or the budget expires, ask:

```text
Report Hyperloom status: baseline, current best, cumulative gain, stop_reason.
```

Check artifacts:

```bash
ls "$USER_DATA_PATH"/*/*/manifest.json
ls "$USER_DATA_PATH"/*/*/reports/
```

## Troubleshooting

- **`/hyperloom-setup` not found** — confirm `pip install --target .` ran in the
  workspace and restart the agent.
- **`install.sh` fails** — check network access for Magpie / TraceLens clones;
  see the [Hyperloom install guide](https://github.com/AMD-AGI/Hyperloom/blob/main/docs/install/install.md).
- **GPU occupied** — kill stale `vllm` / `sglang` / `Magpie` processes (IR-1).
- **Plain serving request** — use `serving-llms-on-instinct` instead.

## Next steps

- Resume: `Resume the latest Hyperloom session for <model>.`
- Ran the quick profile and want kernel rewrites? Start the 12-hour run from
  Step 3 rather than raising `--max-hours` on the quick flags.
- Advanced flags: see `reference.md` in the skill folder.
