# AMD Skills Walkthroughs: `tracelens-analysis-orchestrator`

The goal of this skill is to teach your AI agent to run the TraceLens agentic
analysis workflow on a PyTorch trace and produce a prioritized stakeholder
report (`analysis.md`) with recommendations for optimizing individual kernels, 
kernel fusion opportunities, and system-level optimizations.

**What you'll end up with:** a single user-facing report at
`<output_dir>/analysis.md` — an executive summary, ranked P-items by impact score,
and a detailed analysis section with evidence tables and resolution guidance.
Expect this process to take 5–30 minutes depending on trace size.

## Analysis modes at a glance

| Mode | Traces | Best for |
|---|---|---|
| **Standalone** | 1 | Roofline analysis on a single trace (default) |
| **Comparative** | 2 | Gap analysis: primary trace vs. a faster reference trace |

| Workload | Standalone | Comparative |
|---|---|---|
| Eager (training / eager inference) | Yes | Yes |
| Graph + capture (vLLM / SGLang) | Yes | No |
| Graph only | No | No |

For vLLM / SGLang inference traces, the canonical collection guide in
[TraceLens Inference Analysis](https://github.com/AMD-AGI/TraceLens/blob/main/docs/Inference_analysis.md)
provides detailed instructions about collecting traces.
Including capture mode graphs will produce better results but may require patching vLLM or SGLang.

## Prerequisites

**Software**

- NVM 22 or later is installed to enable NPX skills
- An agentic runner: **Cursor** (chat or `agent` CLI) or **Claude Code**
- Model: **Claude Opus 4.7 High** (`claude-opus-4-7-high`) — the orchestrator and
  all 13 sub-agents are tuned for this model

**Data**

- A PyTorch profiler trace (`.json` or `.json.gz`) from a representative
  steady-state window (post-warmup). A single rank's trace is sufficient for
  per-rank analysis.
- The **platform** of the first trace (e.g. `MI300X`, `MI325X`) — used for
  roofline limits and hardware reference in the report

## Step 1 - Understanding which skills are available

* Run `claude "which skills do you see?"`. You should see a list of skills that should not include anythink related to TraceLens.
* Make sure there is no `AGENTS.md` file on your local folder.

## Step 2 — Enabling your agent to see `tracelens-analysis-orchestrator`

**Claude Code** — install with the [`skills` CLI](https://github.com/vercel-labs/skills):

```bash
npx skills add amd/skills --skill tracelens-analysis-orchestrator --agent claude-code
```

**Cursor** — install the AMD Skills plugin or add the skill manually:

```bash
npx skills add amd/skills --skill tracelens-analysis-orchestrator --agent cursor
```

Confirm the skill is visible:

```bash
claude "Which skills can you see?" --model opus
```

You should now see `tracelens-analysis-orchestrator` in the list.

## Step 5 — Running standalone analysis (interactive)

This is the recommended first run: one trace, default analysis mode (training and
non-vLLM/SGLang eager inference).

### Cursor chat

Open a Cursor chat with **Claude Opus 4.7 High** and send:

```
Follow the analysis orchestrator installed with TraceLens and run the full agentic
analysis workflow on <path_to_trace.json>
```

When prompted, provide:

| Input | Example |
|---|---|
| Platform | `MI300X` |
| Analysis mode | `default` (training / eager inference) |
| Environment | local, local+venv, cluster, or cluster+container |
| Node / container / venv | only if running remotely |
| Output directory | `./analysis_output` |

For **vLLM / SGLang** traces, choose analysis mode `inference` and specify execution
mode (`eager` or `graph replay + capture`). Graph replay + capture also requires a
capture folder path.

### Claude Code

```bash
claude --model opus
```

Then prompt:

```
Follow the analysis orchestrator and run the full agentic analysis workflow on
<path_to_trace.json> with platform MI300X, analysis mode default, output to
./analysis_output
```

Adapt the environment fields if TraceLens runs on a remote node or inside a
container.

The orchestrator will:

1. Generate a TraceLens performance report
2. Prepare per-category data (GEMM, SDPA, norm, etc.)
3. Launch system-level and compute-kernel sub-agents **in parallel**
4. Validate, aggregate, and write `analysis.md`

This may take a while — the workflow invokes 13 specialized sub-agents.

## Step 6 — Review the output

The only artifact intended for end-user review is **`analysis.md`**. Everything
else under the output directory is agent internals.

```
analysis_output/
├── analysis.md              # Stakeholder report (read this)
├── perf_report.xlsx         # Internal
├── perf_report_csvs/        # Internal
├── category_data/           # Internal
├── system_findings/         # Internal
├── category_findings/       # Internal
└── metadata/                # Internal
```

Open `analysis.md` and check the structure:

1. **Executive Summary** — workload characterization, metrics table, improvement chart
2. **Compute Kernel Optimizations** — top operations and ranked P-items
3. **Kernel Fusion Opportunities (Experimental)**
4. **System-Level Optimizations (Experimental)**
5. **Detailed Analysis** — per-P-item drill-down with identification, data tables,
   reasoning, and resolution

Each P-item includes `impact_score` bounds in HTML comment markers
(`<!-- impact-begin ... -->`) that downstream optimization tools can parse
programmatically.

## Step 7 — (Optional) Comparative analysis

When you have two traces from the same framework (e.g. both vLLM, both SGLang),
run a gap analysis. Always pass the **baseline** trace as trace 1.

**Cursor chat:**

```
Follow the analysis orchestrator installed with TraceLens and run the full agentic
analysis workflow on <path_to_baseline_trace.json> and <path_to_primary_trace.json>
```

**Claude Code:**

```
Follow the analysis orchestrator and run the full agentic analysis workflow on
<path_to_baseline_trace.json> and <path_to_primary_trace.json> with platform MI300X,
analysis mode default, output to ./analysis_output_comparative
```

Comparative output adds per-trace perf reports (`perf_report_trace1.xlsx`,
`perf_report_trace2.xlsx`) and gap-based impact estimates in `category_data/`.

> Cross-framework comparisons (e.g. vLLM vs. SGLang) may produce misleading gap
> estimates due to structural differences in operation call stacks.

## Step 8 — (Optional) Headless runs with the Cursor `agent` CLI

For batch runs or CI, install the Cursor CLI and pass all parameters inline so no
interactive prompts are needed.

```bash
curl https://cursor.com/install -fsS | bash
```

**Local — standalone, default mode:**

```bash
agent --model claude-opus-4-7-high --print --force --trust \
  "Follow the analysis orchestrator installed with the TraceLens pip package (look under TraceLens/Agent/Analysis/.cursor/skills/ in the package installation directory) and run the full agentic analysis workflow on <path_to_trace.json> with platform <platform>, analysis mode default, output to <output_dir>"
```

**Cluster + container — inference (vLLM/SGLang eager):**

```bash
agent --model claude-opus-4-7-high --print --force --trust \
  "Follow the analysis orchestrator installed with the TraceLens pip package (look under TraceLens/Agent/Analysis/.cursor/skills/ in the package installation directory) and run the full agentic analysis workflow on <path_to_trace.json> with platform <platform>, analysis mode inference, execution mode eager, node <node>, container <container>, output to <output_dir>"
```

See
[TraceLens Agent Analysis README](https://github.com/AMD-AGI/TraceLens/blob/main/TraceLens/Agent/Analysis/README.md)
for graph replay + capture and additional CLI variants. The eval script
`agent_evals/Analysis/eval_scripts/generate_ref.sh` in the TraceLens repo is a
reference for batch reference generation.

## Step 9 — (Optional) Try to get things done without AMD Skills

Remove the added skill from your agent's skills directory and rerun the experiment
above. Without the orchestrator skill you should see higher variance in execution
length, token usage, and report quality. Common issues without the skill include:

- Agent improvising ad-hoc trace parsing instead of running TraceLens CLI tools
- Missing or inconsistent sub-agent workflow (system-level vs. compute-kernel tiers)
- Report that lacks prioritized P-items, impact scores, or the three-tier structure
- Skipped validation steps or incomplete `category_data/` preparation
- Agent providing a generic performance essay instead of a structured `analysis.md`

## Going further

- **Inference profiling:** Use the TraceLens profiling skill
  ([`magpie-benchmark-profiling`](https://github.com/AMD-AGI/TraceLens/blob/main/TraceLens/Agent/Profiling/README.md))
  or the [`magpie-kernel-evaluator`](../skills/magpie-kernel-evaluator/SKILL.md)
  skill to collect vLLM/SGLang traces before analysis.
- **Optimization loop:** Parse `analysis.md` P-items and impact markers to drive
  kernel tuning, fusion, batching, or precision narrowing, then re-profile and
  re-run analysis to measure improvement.
- **Upstream docs:** Full architecture, sub-agent reference, and output layout are
  documented in the
  [TraceLens Agent Analysis README](https://github.com/AMD-AGI/TraceLens/blob/main/TraceLens/Agent/Analysis/README.md).
