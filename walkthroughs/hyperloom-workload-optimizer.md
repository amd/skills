# AMD Skills walkthroughs: `hyperloom-workload-optimizer`

This skill teaches your AI agent to set up a Hyperloom workspace and autonomously
optimize end-to-end large language model (LLM) inference throughput on AMD Instinct GPUs (MI300X, MI325X, or MI355X).

**What you'll end up with:** a running Hyperloom optimization session with
`manifest.json`, `state.json`, benchmark runs under `runs/`, and a final report
under `reports/` showing validated throughput gain over baseline.

Step 3 gives three ways to run: a 3-hour demo over serving parameters, a 12-hour
demo that adds kernel rewrites, and a custom run for any other budget or mix.

## Prerequisites

Run the agent on the target GPU host. Before you start, confirm the following.

The shell where Cursor or Claude Code runs must be able to see the AMD Instinct GPU and ROCm devices.

Check the things that must already be true:

```bash
test -e /dev/kfd && test -e /dev/dri
(amd-smi || rocm-smi) >/dev/null
python3 --version
node --version
```

You also need:

- AMD Instinct GPU hardware, such as MI300X, MI325X, or MI355X
- Python 3.10+ for the Hyperloom runtime; Python 3.12 when the setup flow will
  install vLLM on bare metal
- Node.js, for the `npx skills add ...` install path
- An agentic runner: **Cursor** or **Claude Code**
- Anthropic API access, or AMD LLM gateway access, for Hyperloom agent backends
- A dedicated empty directory opened as the agent workspace

Use these values for the placeholders in the prompts below:

- `<framework>`: `vllm` or `sglang`
- `<gpu_type>`: `MI300X`, `MI325X`, or `MI355X`
- `<model_path>`: absolute path to a local model directory holding `config.json`.
  Drop that sentence from a demo prompt to run the model the demo names.

You do **not** need to decide these before Step 2:

- Docker or bare metal. `/hyperloom-setup` asks for the run mode and explains
  the tradeoff.
- Model path, tensor parallel (TP) or expert parallel (EP), concurrency, input sequence length (ISL) or output sequence length (OSL), precision, or target gain. Those are
  workload choices collected later.
- Exact credential variable names. Setup writes `.env` and tells you which
  secret value to fill in.

## Step 1: Enable the skill

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

## Step 2: Bootstrap the workspace

In the dedicated workspace, ask the agent:

```text
Install Hyperloom and set up the execution environment for <framework> on <gpu_type>.
```

This step prepares the workspace and execution environment only:

1. **Phase 0, Bootstrap:** confirm the install directory, install the
   `hyperloom-inference-optimizer` release from PyPI with `pip install --target .`,
   then run `/hyperloom-setup` to write `.env` (credentials + run mode only).
2. **Phase 1, Environment prep:** choose Docker or bare metal, then prepare
   that environment. On bare metal, confirm the host stack; in Docker, start a
   long-running container and run the in-container setup first.

Choose **Docker** when you want the validated, reproducible ROCm and framework
stack and can run containers with GPU devices mapped in. Choose **bare metal**
when the host already has the ROCm or framework stack you want to use, or when
containers are unavailable. Bare metal is more sensitive to host packages and
can modify the environment, so prefer Docker for first-time walkthroughs when it
is available.

Save model selection, workload choices and launch approval for Step 3.

Verify the setup handoff:

```bash
test -f .env
grep -E '^(USER_DATA_PATH|HYPERLOOM_RUN_MODE)=' .env
```


## Step 3: Launch an optimization

Start this step only after Step 2 has written `.env` and prepared the execution
environment. Step 3 reuses the Docker or bare-metal run mode recorded in `.env`;
do not choose it again here.

There are three ways to run. Ask for the one you want; the agent loads the
matching demo skill the wheel installed, and that skill owns the workload preset,
the budget and every optimizer flag. Nothing here restates them, so a change to
the CLI reaches you through the wheel rather than through this page.

Each demo names the model it was tuned around, and either can run yours instead —
give the path, or drop that sentence to take the demo's own. The preset workload
does not change with the model, so if yours is much larger or a different
architecture, use **Custom** and set the values yourself.

**1. 3-hour demo (Qwen3-8B).** Serving and config parameters only; the kernel
agent is off. The shortest end-to-end check; with a 3-hour budget keep the model
at 8B or below.

```text
Run the 3-hour Hyperloom Qwen3-8B demo with <framework> on <gpu_type>. Use the
model at <model_path> instead of the demo default. Launch and monitor.
```

**2. 12-hour demo (Qwen3-14B-FP8).** Every lever, kernel rewrites included. The
kernel agent needs room to profile, rewrite and revalidate hot kernels, which is
where the larger gains come from.

```text
Run the 12-hour Hyperloom Qwen3-14B-FP8 demo with <framework> on <gpu_type>. Use
the model at <model_path> instead of the demo default. Launch and monitor.
```

**3. Custom.** Ask for a run and let the agent take you through the choices:
model, framework, TP or EP, concurrency, sequence lengths, precision, budget and
which phases to allow. You do not supply flags; the agent derives them from your
answers.

```text
Optimize a model with Hyperloom on this host. Walk me through the choices.
```

Whichever of the three you use, the agent shows the full launch plan (every
resolved value and every flag) and waits for your approval before it starts.

Once the values are resolved, the agent clears the runtime install and GPU
preflight gates the optimizer skill owns, starts the run in the background, then
polls the session state.

## Step 4: Read results

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

Use these entries when a step fails.

- **`/hyperloom-setup` not found:** Confirm `pip install --target .` ran in the
  workspace and restart the agent.
- **`install.sh` fails:** Check network access for Magpie or TraceLens clones;
  see the [Hyperloom install guide](https://github.com/AMD-AGI/Hyperloom/blob/main/docs/install/install.md).
- **GPU occupied:** Kill stale `vllm`, `sglang`, or `Magpie` processes (IR-1).
- **Plain serving request:** Use `serving-llms-on-instinct` instead.

## Next steps

- Resume: `Resume the most recent Hyperloom session for <model>.`
- Ran the 3-hour demo and want kernel rewrites? Start the 12-hour run from
  Step 3 rather than stretching the 3-hour demo's budget.
- Advanced flags: use **Custom** in Step 3 and let the agent derive them, or read
  the [optimizer skill](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/inference_optimizer/SKILL.md)
  the wheel installs.
