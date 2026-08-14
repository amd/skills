# AMD Skills Walkthroughs: `hyperloom-workload-optimizer`

This skill teaches your AI agent to set up a Hyperloom workspace and autonomously
optimize end-to-end LLM inference throughput on AMD Instinct GPUs (MI300X / MI325X / MI355X).

**What you'll end up with:** a running Hyperloom optimization session with
`manifest.json`, `state.json`, benchmark runs under `runs/`, and a final report
under `reports/` showing validated throughput gain over baseline.

Step 3 gives three ways to run: a 3-hour demo over serving parameters, a 12-hour
demo that adds kernel rewrites, and a custom run for any other budget or mix.

## Prerequisites

Run the agent on the target GPU host. The shell where Cursor, Claude Code runs must be able to see the AMD Instinct GPU and ROCm devices.

Check the things that must already be true:

```bash
test -e /dev/kfd && test -e /dev/dri
(amd-smi || rocm-smi) >/dev/null
python3 --version
node --version
```

You also need:

- AMD Instinct GPU hardware, such as MI300X / MI325X / MI355X
- Python 3.10+ for the Hyperloom runtime; Python 3.12 when the setup flow will
  install vLLM on bare metal
- Node.js, for the `npx skills add ...` install path
- An agentic runner: **Cursor** or **Claude Code**
- Anthropic API access, or AMD LLM gateway access, for Hyperloom agent backends
- A dedicated empty directory opened as the agent workspace

Use these values for the placeholders in the prompts below:

- `<framework>`: `vllm` or `sglang`
- `<gpu-type>`: `MI300X`, `MI325X`, or `MI355X`

You do **not** need to decide these before Step 2:

- Docker vs. bare metal — `/hyperloom-setup` asks for the run mode and explains
  the tradeoff.
- Model path, TP/EP, concurrency, ISL/OSL, precision, or target gain — those are
  workload choices collected later.
- Exact credential variable names — setup writes `.env` and tells you which
  secret value to fill in.

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
Install Hyperloom and set up the execution environment for <framework> on <gpu-type>.
```

This step prepares the workspace and execution environment only:

1. **Phase 0 — Bootstrap:** confirm the install directory, install a pinned
   `hyperloom-inference-optimizer` release from PyPI with `pip install --target .`,
   then run `/hyperloom-setup` to write `.env` (credentials + run mode only).
2. **Phase 1 — Environment prep:** choose Docker or bare metal, then prepare
   that environment. On bare metal, confirm the host stack; in Docker, start a
   long-running container and run the in-container setup first.

Choose **Docker** when you want the validated, reproducible ROCm + framework
stack and can run containers with GPU devices mapped in. Choose **bare metal**
when the host already has the ROCm/framework stack you want to use, or when
containers are unavailable. Bare metal is more sensitive to host packages and
can modify the environment, so prefer Docker for first-time walkthroughs when it
is available.

Save model selection, workload choices and launch approval for Step 3.

Verify the setup handoff:

```bash
ls hyperloom/inference_optimizer/assets/install.sh
test -f .env
grep -E '^(USER_DATA_PATH|HYPERLOOM_RUN_MODE)=' .env
```


## Step 3 — Launch an optimization

Start this step only after Step 2 has written `.env` and prepared the execution
environment. Step 3 reuses the Docker or bare-metal run mode recorded in `.env`;
do not choose it again here.

There are three ways to run. Paste the prompt that matches, replacing
`<framework>` and `<gpu-type>` with your target — the flags in each are a set, so
pass them together. The two demos carry the same workload and flags as the
matching Hyperloom demo skill; you can point either one at your own model, and
**Custom** is for changing the workload itself.

**1. 3-hour demo.** Serving and config parameters only; the kernel agent is
off. Swap in your own model if you want; with a 3-hour budget keep it at 8B or
below.

```text
Optimize Qwen/Qwen3-8B with <framework> on <gpu-type>: TP=1, conc=64, ISL=1024,
OSL=1024, precision bf16, target-gain 30, max-hours 3, serving parameters only:
--no-framework-agent --no-kernel --no-enable-conc-sweep --no-enable-roofline
--max-minutes-explore-pct 0.39 --max-minutes-sweep-pct 0.01
--explore-force-exit-budget-pct 0.01 --explore-force-exit-hours-remaining 0.05.
Launch and monitor.
```

**2. 12-hour demo.** Every lever, kernel rewrites included. The kernel agent
needs room to profile, rewrite and revalidate hot kernels, which is where the
larger gains come from.

```text
Optimize Qwen/Qwen3-14B-FP8 with <framework> on <gpu-type>: TP=1, conc=64,
ISL=1024, OSL=1024, precision fp8, target-gain 30, max-hours 12, all components
enabled:
--max-minutes-framework-pct 0.01 --max-minutes-explore-pct 0.42
--max-minutes-kernel-pct 0.42. Launch and monitor.
```

**3. Custom.** Ask for a run and let the agent take you through the choices —
model, framework, TP/EP, concurrency, sequence lengths, precision, budget and
which phases to allow. You do not supply flags; the agent derives them from your
answers.

```text
Optimize a model with Hyperloom on this host. Walk me through the choices.
```

Whichever of the three you use, the agent shows the full launch plan — every
resolved value and every flag — and waits for your approval before it starts.

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
- Ran the 3-hour demo and want kernel rewrites? Start the 12-hour run from
  Step 3 rather than raising `--max-hours` on the 3-hour demo's flags.
- Advanced flags: see `reference.md` in the skill folder.
