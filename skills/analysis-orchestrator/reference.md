<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Analysis orchestrator — reference

This document is the detailed specification for the TraceLens **analysis-orchestrator** skill ([SKILL.md](SKILL.md)). Read it when executing the workflow: step-by-step user prompts, CLI commands, subagent contracts, validation, report assembly, and trace diagnostics.

## Workflow overview

The orchestrator runs a staged pipeline (Steps 0–11): collect inputs and environment prefix, generate perf reports, prepare category data via `orchestrator_prepare.py`, run system-level and compute-kernel subagents in parallel, aggregate and validate findings, identify the model, render plots when no extension is present, and write `analysis.md` via remote `tee` heredocs. Only Steps 6, 7, and 9 delegate to Task subagents; all other steps run in the main agent.

---

## Language Guidelines

Use vendor-agnostic terminology throughout such as GPU kernels, collective communication, vendor GEMM library, DNN primitives, GPU graph, etc. Focus on operation semantics, not vendor implementation details

**Exception:** When quoting kernel names from traces, it's acceptable to include the actual name for identification.
 
---

## Workflow Steps

```
0. Query User Inputs (Platform, Trace Path(s), Analysis Mode, Environment Setup)
1. Generate Performance Report (branches on analysis mode: training vs inference then, comparison scope)
2-5. Prepare Category Data (GPU Util, Top Ops, Tree Data, Multi-Kernel Data, Category Filtering)
6. System-Level Analysis (PARALLEL, CPU/Idle + Kernel Fusion + Multi-Kernel) → system_findings/
7. Invoke Compute Kernel Subagents (PARALLEL, read category_findings[] from _metrics.json) → category_findings/
   7.5. Aggregate per-category category_findings[] → priority_data.json::findings[] (globally sorted)
8. Validate Subagent Outputs (system_findings/ + category_findings/)
9. Prepare Report Data (load_findings) + Model Identification (subagent) → metadata/model_info.json
10. Render performance PNG IF agent_extension.py is absent
11. Generate Final Report (composable System + Compute sections), validate it,
    optionally invoke agent_extension.py (when present), then embed the PNG into the report.
```

**Subagent usage:** Only invoke Task subagents in steps that explicitly say "subagent" (Steps 6, 7, 9). All other steps (including Step 7.5) must be performed directly by the orchestrator using the command prefix.

---

## Step 0: Query User Inputs

**When this skill is invoked, immediately ask the user for:**

### Required Information:

1. **Comparison scope** → `<comparison_scope>`
   - Set from the user’s intent **before** deep-diving on paths:
     - **`comparative`** if the skill was triggered by **“comparative analysis”**, **“compare two traces”**, or the user supplies **two** trace paths / explicitly asks to compare trace A vs B.
     - **`standalone`** otherwise (including triggers **“standalone analysis”**, **“analyze trace standalone”**, single trace only).

2. **Trace File Path(s)**
   - **`standalone`:** **Trace File Path** → `<trace_path>`
     - Ask: "Please provide the full path to your PyTorch trace file (.json or .json.gz)"
   - **`comparative`:** ask for both:
     - **Primary trace (trace1)** → `<trace_path>`
     - **Comparison trace (trace2)** → `<trace2_path>`
     - Ask: "Please provide the full path to your primary trace file and your comparison trace file (.json or .json.gz)"

3. **Platform** → `<platform>`
   **`standalone`**: Ask: "Which platform are you analyzing?"
   **`comparative`**: Ask: "Which platform is baseline trace (trace1)?"
   - Options:
     1. **MI300X**
     2. **MI325X**
     3. **MI350X**
     4. **MI355X**
     5. **MI455X**
   **`comparative`:** Ask: "Which platform is target trace (trace2)?" Assign `<platform2>` (`<platform2>` does not need to be one of the platform options)

4. **Analysis Mode** → `<analysis_mode>`
   - If the user's prompt explicitly specifies an analysis mode or mentions inference/vLLM/SGLang, use that. Otherwise, default to `default` without asking.
   - Options:
     1. **Default (training and non-vLLM/SGLang eager inference)** (`<analysis_mode>` = `default`) — uses `TraceLens_generate_perf_report_pytorch`
     2. **Inference analysis (vLLM/SGLang)** (`<analysis_mode>` = `inference`) — uses `TraceLens_generate_perf_report_pytorch_inference`
   - If **Inference (vLLM/SGLang)** is selected, ask **Execution Mode** → `<inference_exec_mode>`:
     1. **Eager mode** (`<inference_exec_mode>` = `eager`) — only the trace file is needed
     2. **Graph replay + capture** (`<inference_exec_mode>` = `graph_capture`) — also requires a capture folder path
   - If **Graph replay + capture**, ask for **Capture Folder Path** → `<capture_folder_path>`:
     - Ask: "Please provide the full path to the graph capture traces folder"
   - **Unsupported combination:** If `<inference_exec_mode>` = `graph_capture` **and** `<comparison_scope>` = `comparative`, stop immediately. Inform the user: "Graph replay + capture mode is not yet supported for comparative analysis. Please provide eager mode traces instead." Do not misinterpret as two standalone analyses. Do **not** proceed to Step 1 or beyond.

5. **Environment Setup**
   - Ask: "Are you running locally or on a cluster?"
     - If **local**: No further environment questions — prefix is blank (commands run directly).
     - If **cluster**:
       - Ask "Which node should we use?" → `<node>`
       - Ask "Are you working in a containerized environment (e.g. Docker)?" → if yes, ask for container name → `<container>`
       - Ask "Are you using a virtual environment?" → if yes, ask for venv path → `<venv_path>`

6. **Output Directory** (Optional)
   - Ask: "Where should we save analysis results? (Press Enter for default: <trace_directory>/analysis_output)"
   - Default: Same directory as trace file, in `analysis_output/` subdirectory

7. **Extension File** (Optional) → `<extension_file>`
   - Ask: "Do you have a TraceLens extension file to apply? Press Enter to skip."
   - If provided, resolve to an absolute path and assign to `<extension_file>`.
   - If skipped, set `<extension_file>` to empty (no `--extension_file` flag is added to any command).

### Build and Cache Command Prefix

After collecting inputs, build a command template and save it to `<output_dir>/cache/cmd_prefix.txt`. Create the directory with `mkdir -p <output_dir>/cache`.

The template uses `{CMD}` as a placeholder for the actual command.

**Cluster:** Before building the prefix, locate the TraceLens project root on the remote environment.

Run the following command (adjust for container if applicable):

```bash
# Without container:
ssh <node> "find / -maxdepth 5 -type d -name 'TraceLens' 2>/dev/null | head -5"

# With container:
ssh <node> "docker exec <container> bash -c 'find / -maxdepth 5 -type d -name TraceLens 2>/dev/null | head -5'"
```

Pick the result containing `Agent/` and strip the trailing `/TraceLens` to get `<tracelens_dir>`.

Build the cluster prefix using this lookup:

| Container | Venv | Template |
|-----------|------|----------|
| No | No | `ssh <node> "cd <tracelens_dir> && {CMD}"` |
| Yes | No | `ssh <node> "docker exec <container> bash -c 'cd <tracelens_dir> && {CMD}'"` |
| No | Yes | `ssh <node> "bash -c 'source <venv_path>/bin/activate && cd <tracelens_dir> && {CMD}'"` |

Write the resolved template to `<output_dir>/cache/cmd_prefix.txt`. Then validate it works:

```bash
<prefix> python3 -c "import TraceLens; print('PREFIX_OK')"
```

If this fails, inform the user with `[DIAG:pipeline:PREFIX_FAIL]` and check that `<tracelens_dir>` is the **parent** of TraceLens (not the repo root itself), verify the container/venv is accessible, rebuild, and retry. Do NOT proceed to Step 1 until validation passes.

### Command Execution Pattern

**Before executing any command**, read `<output_dir>/cache/cmd_prefix.txt`. It contains a template with a `{CMD}` placeholder. Substitute `{CMD}` with the actual command. All commands below use `<prefix>` to represent this resolved template.

---

## Step 1: Generate Performance Report

Use **`<analysis_mode>`** to determine which CLI tool to run and then **`<comparison_scope>`** to determine arguments.

For all of these scripts below, look at the environment variable TL_EXTENSION to recursively search for a file called <platform>.json. Do not look for <platform2>.json; it is not needed.
If it is not found also look in TraceLens/Agent/Analysis/utils/arch/<platform>.json.
Use <platform_file> to represent the location of this file

**CLI call count:**
- **`standalone`**: one TraceLens CLI call (for `<trace_path>`)
- **`comparative`**: one TraceLens CLI call per trace (for `<trace_path>` and `<trace2_path>`)

All commands below append `<suffix_1>` and `<suffix_2>`, resolved by `<comparison_scope>`:

**`<suffix_1>`** — output paths:

| scope | value |
|-------|-------|
| `standalone` | `--output_xlsx_path <output_dir>/perf_report.xlsx --output_csvs_dir <output_dir>/perf_report_csvs` |
| `comparative` trace1 | `--output_xlsx_path <output_dir>/perf_report_trace1.xlsx --output_csvs_dir <output_dir>/perf_report_trace1_csvs` |
| `comparative` trace2 | `--profile_json_path <trace2_path> --output_xlsx_path <output_dir>/perf_report_trace2.xlsx --output_csvs_dir <output_dir>/perf_report_trace2_csvs` |

**`<suffix_2>`** — extension flags:

| scope | value |
|-------|-------|
| `standalone` | none |
| `comparative` trace1 | `--comparison_json_path <trace2_path>` |
| `comparative` trace2 | none |

**`<suffix_ext>`** — user extension file:

| condition | value |
|-----------|-------|
| `<extension_file>` provided | `--extension_file <extension_file>` |
| not provided | none |

---

**Default (training and non-vLLM/SGLang eager inference)** (`<analysis_mode>` = `default`):

```bash
<prefix> TraceLens_generate_perf_report_pytorch \
  --profile_json_path <trace_path> \
  --gpu_arch_json_path <platform_file> \
  --enable_pseudo_ops \
  --group_by_num_kernels \
  --include_call_stack \
  <suffix_1> \
  <suffix_2> \
  <suffix_ext>
```

**Inference eager mode** (`<analysis_mode>` = `inference`, `<inference_exec_mode>` = `eager`):

```bash
<prefix> TraceLens_generate_perf_report_pytorch_inference \
  --profile_json_path <trace_path> \
  --gpu_arch_json_path <platform_file> \
  --group_by_parent_module \
  --enable_pseudo_ops \
  --group_by_num_kernels \
  --include_call_stack \
  <suffix_1> \
  <suffix_2> \
  <suffix_ext>
```

**Inference graph replay + capture mode** (`<analysis_mode>` = `inference`, `<inference_exec_mode>` = `graph_capture`):

```bash
<prefix> TraceLens_generate_perf_report_pytorch_inference \
  --profile_json_path <trace_path> \
  --capture_folder <capture_folder_path> \
  --gpu_arch_json_path <platform_file> \
  --group_by_parent_module \
  --enable_pseudo_ops \
  --group_by_num_kernels \
  --include_call_stack \
  <suffix_1> \
  <suffix_2> \
  <suffix_ext>
```

---

## Steps 2-5: Prepare Category Data

Execute the TraceLens Agentic Mode orchestrator preparation script:

```bash
<prefix> python3 \
  TraceLens/Agent/Analysis/utils/orchestrator_prepare.py \
  --trace-path <trace_path> \
  --platform <platform> \
  --output-dir <output_dir> \
  --comparison-scope <comparison_scope>
```

This script performs:
- **Step 2:** Assess GPU utilization (computation, idle, communication times)
- **Step 3:** Identify top 10 operations by GPU time
- **Step 4:** Pre-compute tree data for bottleneck operations (load trace ONCE)
- **Step 4.5:** Pre-compute multi-kernel issue data (memcpy by direction, NCCL events, overlap metrics)
- **Step 5:** Filter and export category-specific data

**Outputs:**
- `category_data/<category>_ops.csv` - Filtered operations per category
- `metadata/<category>_metadata.json` - Platform specs, GPU utilization, config
- `category_data/multi_kernel_data.json` - Memcpy/NCCL/overlap pre-computed data
- `category_data/category_manifest.json` - Workflow metadata with categories (includes `tier` field: `system` or `compute_kernel`)
- `system_findings/` - Directory for system-level analysis outputs
- `category_findings/` - Directory for compute kernel analysis outputs

---

## Step 6: System-Level Analysis (PARALLEL)

System-level analysis examines issues that affect the GPU pipeline as a whole -- idle time, memory transfer patterns, and communication/compute overlap. These are **not** about individual kernel efficiency.

**Output directory:** `system_findings/`

### 6.1 Read Manifest and Identify System-Level Subagents

```bash
<prefix> python3 -c \"
import sys
from TraceLens.Agent.Analysis.utils.report_utils import load_manifest_categories
load_manifest_categories(sys.argv[1])
\" '<output_dir>'"
```

This prints `system_categories` and `compute_categories` lists. Use `system_categories` for Step 6 and `compute_categories` for Step 7.

### 6.2 Launch System-Level Subagents in PARALLEL

Launch system-level sub-agents simultaneously using the Task tool. Do NOT wait between invocations.

**System-Level Agent File Map:**

**Base path:** `TraceLens/Agent/Analysis/skills/analysis-orchestrator/agents/`

| Category | Agent file |
|----------|-----------|
| `cpu_idle` | `cpu-idle-analyzer.md` |
| `multi_kernel` | `multi-kernel-analyzer.md` |
| `kernel_fusion` | `kernel-fusion-analyzer.md` |

**Invocation conditions:**
- **CPU/Idle**: Read `category_data/category_manifest.json` and check `gpu_utilization.idle_time_percent`. Only invoke the subagent if `idle_time_percent > 15`. Skip otherwise -- the deterministic script already captured the factual data.
- **Multi-Kernel**: `multi_kernel` category exists in manifest OR `gpu_util['exposed_comm_time_percent'] > 0` OR `gpu_util['exposed_memcpy_time_percent'] > 0`
- **Kernel Fusion**: `kernel_fusion` category exists in manifest

**Task prompt structure for each system-level subagent:**

The subagent reads its own agent file — the orchestrator does NOT read or paste agent file contents.

```
Read and follow the FULL instructions in:
  TraceLens/Agent/Analysis/skills/analysis-orchestrator/agents/<agent-file>.md

**Execution Context:**
- Comparison scope: `<comparison_scope>`
- Output directory: <output_dir>
- Command prefix: read `<output_dir>/cache/cmd_prefix.txt` — contains a template
  with `{CMD}` placeholder; substitute `{CMD}` with the actual command
- Input files: <list from agent file's "Input files" section>
- Output file: <from agent file's "Output file" section>

Execute every step in the agent file. Return "DONE" when complete.
```

**CRITICAL:** The orchestrator does NOT read agent files or run analysis scripts. Each sub-agent is responsible for:
1. Reading its own agent `.md` file
2. Running its Python script using the command prefix
3. Reading the metrics JSON output
4. Identifying issues and generating findings

### 6.3 Wait for System-Level Subagents to Complete

The three subagents must complete before proceeding to Step 6.4.
Each writes findings to `system_findings/<name>_findings.md`.

### 6.4 Verify System Outputs and Retry Failures (up to 1 retry per subagent)

After all system-level subagents complete:

1. For each expected system category from the manifest, check:
   - Does `system_findings/<category>_findings.md` exist?
   - If it exists, does it contain "Status: ERROR"?
2. Collect a list of **failed** categories (missing file OR Status: ERROR).
3. **Retry each failed category exactly once** by re-launching its subagent with the same prompt from Step 6.2. Wait for all retries to complete before proceeding.
4. After retries, re-check outputs. Any category that still fails is excluded from aggregation.
5. **CRITICAL: Do NOT attempt manual analysis of failed system checks — only automated subagent retry is allowed.**

---

## Step 7: Invoke Compute Kernel Subagents (PARALLEL)

Compute kernel analysis examines individual operation category efficiency.

**Output directory:** `category_findings/`

### 7.1 Read Manifest and Identify Compute Kernel Categories

Use `compute_categories` from the `load_manifest_categories()` call in Step 6.1.

### 7.2 Launch Compute Kernel Subagents in PARALLEL

For each entry in `compute_categories` (loaded in Step 6.1), resolve `{agent_file}` as `{entry.skill}.md` and launch a subagent with agent file `TraceLens/Agent/Analysis/skills/analysis-orchestrator/agents/{agent_file}`. Fall back to `generic-op-analyzer.md` if the file is absent.

Launch all subagents simultaneously in a single parallel batch.

---

#### Shared Compute Kernel Preamble

Include this block in every compute kernel subagent prompt:

<Shared Compute Kernel Preamble>:
```
comparison_scope: {comparison_scope}

**CRITICAL - READ FIRST:**
- Use GPU kernel time (not CPU duration) for all bottleneck analysis
- `efficiency_percent` semantics differ by mode:
  - **Standalone:** % of roofline. Flag > 100% as "[ANOMALY] - verify measurement".
  - **Comparative:** `100 × (Trace 2 kernel time) / (Trace 1 kernel time)`.
    - **< 100%** → Trace 1 is slower than Trace 2. **This is an optimization opportunity — flag it.**
    - **> 100%** → Trace 2 is slower than Trace 1. **NOT an anomaly; no Trace-1 optimization needed.**

**CRITICAL CONSTRAINTS:**
1. **Standalone:** Any efficiency > 100% → `[ANOMALY] - verify measurement`. **Comparative:** efficiency > 100% means Trace 2 is slower — NOT an anomaly; efficiency < 100% means Trace 1 is slower — flag as optimization opportunity.
2. Status must be SUCCESS or ERROR; times in ms; efficiencies as percentages
3. Operations with `fusion_flagged: true` in the metrics JSON are already covered by
   a high-confidence kernel fusion candidate — do NOT flag them as bottlenecks or write
   kernel_tuning recommendations. The analysis scripts already exclude them from `impact_estimates`.

**Execution Context:**
- Output directory: <output_dir>
- Command prefix: read `<output_dir>/cache/cmd_prefix.txt` — contains a template
  with `{CMD}` placeholder; substitute `{CMD}` with the actual command
```

---

#### Compute Kernel Subagent Prompt

For each category, launch a Task (subagent_type: generalPurpose):

```
You are analyzing {category} operations for a PyTorch trace on {platform}.

<Shared Compute Kernel Preamble>

Read and follow the FULL instructions in:
  TraceLens/Agent/Analysis/skills/analysis-orchestrator/agents/{agent_file}

- Category: {category}
- Input files: category_data/{category}_ops.csv, metadata/{category}_metadata.json,
  category_data/{category}_metrics.json (P-items come from `category_findings[]`; `operations[i].module_chain` provides model layer context)
- Output file: category_findings/{category}_findings.md

Execute every step in the agent file. Return "DONE" when complete.
```

### 7.3 Wait for All Compute Kernel Subagents to Complete

All subagents must complete before proceeding to Step 7.4.
Each subagent writes its findings to `category_findings/<category>_findings.md`.

### 7.4 Verify Outputs and Retry Failures (up to 1 retry per subagent)

After all compute kernel subagents complete:

1. For each category in the manifest with `tier: compute_kernel`, check:
   - Does `category_findings/<category>_findings.md` exist?
   - If it exists, does it contain "Status: ERROR"?
2. Collect a list of **failed** categories (missing file OR Status: ERROR).
3. **Retry each failed category exactly once** by re-launching its subagent with the same prompt structure from Step 7.2. Launch all retries in parallel and wait for completion.
4. After retries, re-check outputs. Any category that still fails is excluded from aggregation and recommendations.
5. **CRITICAL: Do NOT attempt to manually analyze failed categories — only automated subagent retry is allowed.**

### 7.5 Aggregate findings → priority_data.json

After all compute sub-agent `_metrics.json` files exist (each carrying its own `category_findings[]`), concatenate them into a globally-sorted `priority_data.json::findings[]` for the report template.

```bash
<prefix> python3 -c \"
import sys
from TraceLens.Agent.Analysis.utils.report_utils import generate_priority_data
generate_priority_data(sys.argv[1])
\" '<output_dir>'
```

---

## Step 8: Validate Subagent Outputs

Before aggregating results, validate outputs from **both** tiers (system_findings/ and category_findings/).

```bash
<prefix> python3 -c \"
import sys
from TraceLens.Agent.Analysis.utils.validation_utils import validate_subagent_outputs
validate_subagent_outputs(sys.argv[1])
\" '<output_dir>'"
```

This runs four checks:
1. **Time Sanity** -- category GPU kernel time sum vs computation time (WARN if >15% discrepancy)
2. **Efficiency Anomalies** -- findings with efficiency >100% (measurement issues) when `<comparison_scope>` = `standalone`
3. **Coverage** -- all expected system and compute findings present
4. **Priority Consistency** -- `priority_data.json` invariants: `findings[]` sorted desc by `impact_score`, contiguous `global_rank` / `priorities[].rank`, and per-category `priorities[].impact_score` ≈ `sum(findings[].impact_score)`

---

## Step 9: Prepare Report Data + Model Identification

```bash
<prefix> python3 -c \"
import sys
from TraceLens.Agent.Analysis.utils.report_utils import load_findings
load_findings(sys.argv[1])
\" '<output_dir>'"
```

### 9.1 Model Identification (Subagent, retry once on failure)

Launch a Task subagent (generalPurpose) that reads and follows `TraceLens/Agent/Analysis/skills/analysis-orchestrator/agents/model-identification-agent.md` with context: <output_dir>. Wait for completion.

**On failure (subagent error, timeout, or `model_info.json` not written):**
1. **Retry exactly once** by re-launching the same subagent with the same prompt.
2. If the retry also fails, write fallback `metadata/model_info.json` with all four fields set to `"Cannot be inferred from trace"`.

Assign <Model> to model value in `<output_dir>/metadata/model_info.json` or "Workload" if model is "Cannot be inferred from trace".

---

## Step 10: Render Plot (conditional)

**Important:** Plot data is sourced from `priority_data.json` (written in Step 7.5). This step only renders the PNG when `agent_extension.py` is absent.
Look at the environment variable TL_EXTENSION to find python packages and directories to recursively search for `agent_extension.py`.
If this environment variable is not present or the it is not found look in TraceLens/Agent/Analysis/utils/.
If the file is present, **skip this step** — Step 11.2 will produce `perf_improvement.png` and Step 11.3 will embed it.
Use <agent_extension_file> to represent the location of this file.

```bash
EXT='<agent_extension_file>'
if [ ! -f "$EXT" ]; then
  <prefix> python3 -c \"
import sys
from TraceLens.Agent.Analysis.utils.plot_utils import generate_perf_plot
generate_perf_plot(sys.argv[1], sys.argv[2])
\" '<output_dir>' '<Model> on <Platform> — Performance Breakdown'
fi
```

If the plot fails (extension-absent branch), retry once. If still failing, proceed to Step 11 without the plot.

---

## Step 11: Generate Final Report (<output_dir>/analysis.md)

**CRITICAL: Do NOT delegate Step 11 to a Task subagent.** The orchestrator must write the report directly.

1. **Read** the report template: `TraceLens/Agent/Analysis/skills/analysis-orchestrator/templates/analysis_template.md`
2. **Write the report in sections** to `<output_dir>/analysis.md` using **only** `<prefix> tee` / `<prefix> tee -a` with single-quoted heredoc delimiters (see write order below). You MUST NOT use the IDE Write tool, Edit tool, StrReplace tool, `cat >`, `echo >`, `>>` redirect, or any other write method for `analysis.md` unless tee fails.
3. **Fill in** each section by substituting placeholders with actual data. Never retain template placeholders (`<Brief Title>`, `X ms`, `Y%`, `<platform>`, `<model>`) — every field must contain actual data.

**Write order (one heredoc per step):**

   a. **Initialize** — truncate and write the title line + `## Executive Summary` (metrics table, `{{PERF_PLOT}}` placeholder). Use `<prefix> tee <output_dir>/analysis.md << 'SECTION_EOF'` (truncating `tee`, not append) for this first write only.
      - Data sources: `category_data/category_manifest.json` (`gpu_utilization` keys), `priority_data.json` (top bottleneck).

   b. **Compute Kernel Optimizations** — append `## Compute Kernel Optimizations` with `### Top Operations` table and P-item cards. Use `<prefix> tee -a <output_dir>/analysis.md << 'SECTION_EOF'`.
      - Data sources: `priority_data.json` — P1 = `findings[0]`, P2 = `findings[1]`, ... ; each card joins its sub-agent's Detailed Analysis block by `(findings[i].category, findings[i].category_rank)`. The Top Operations table materializes `priorities[]` verbatim (one row per entry, array order, no re-sorting).
      - `category_findings/*.md` — for each findings file, copy its `## Recommendations` P-items into the report card slots. **Copy table cells verbatim** from the source `category_findings/<cat>_findings.md`.
      - Heuristic findings (`findings[i].estimate_method == "heuristic"`) carry a numeric estimated impact and sort by `impact_score` like any other compute finding — render them (do NOT skip them) per `sub_agent_spec.md § Heuristic findings`.

   c. **Kernel Fusion** — append `## Kernel Fusion Opportunities (Experimental)`. Use `<prefix> tee -a <output_dir>/analysis.md << 'SECTION_EOF'`.
      - Data source: `system_findings/kernel_fusion_findings.md`.

   d. **System-Level** — append `## System-Level Optimizations`. Use `<prefix> tee -a <output_dir>/analysis.md << 'SECTION_EOF'`.
      - Data sources: remaining `system_findings/*.md` (cpu_idle, multi_kernel).

   e. **Detailed Analysis** — append `## Detailed Analysis` with `### Compute Kernel Insights`, `### Kernel Fusion Insights`, `### System-Level Insights` subsections. Use `<prefix> tee -a <output_dir>/analysis.md << 'SECTION_EOF'`.
      - Data sources: copy the `## Detailed Analysis` blocks verbatim from each `*_findings.md` file. Follow the template for formatting.
      - `category_data/*_metrics.json` (per-op tables, impact estimates).

   f. **Appendix** — append `## Appendix` with `### Model Architecture` and `### Hardware Reference`. Use `<prefix> tee -a <output_dir>/analysis.md << 'SECTION_EOF'`.
      - `metadata/model_info.json` — substitute `<model>`, `<architecture>`, `<scale>`, `<precision>` with the four field values.
      - Platform arch file — read `platform` from `category_manifest.json`, then read `TraceLens/Agent/Analysis/utils/arch/<platform>.json`. For `### Hardware Reference`: substitute `<platform>`, Peak HBM BW = `mem_bw_gbps / 1000` TB/s, Peak MAF (BF16) = `max_achievable_tflops.matrix_bf16` TFLOPS, Peak MAF (FP8) = `max_achievable_tflops.matrix_fp8` TFLOPS if present.

**Failure exclusion:** Skip any category listed in `load_findings()` output as `failed_system` or `failed_compute`. Include a `## Warnings` section (between Executive Summary and Compute Kernel Optimizations) only if failures exist.

The report at `<output_dir>/analysis.md` must use these exact `##` headers — do NOT rename them:
1. `## Executive Summary`
2. `## Compute Kernel Optimizations`
3. `## Kernel Fusion Opportunities (Experimental)`
4. `## System-Level Optimizations`
5. `## Detailed Analysis`
6. `## Appendix`


### 11.1 Validate Report Structure (Retry up to 2x)

After writing `analysis.md`, validate that the report contains all required `##` section headers. If validation fails, modify the report with the missing sections.

**Validation procedure:**

```bash
<prefix> python3 -c \"
import sys
from TraceLens.Agent.Analysis.utils.validation_utils import validate_report
passed, missing = validate_report(sys.argv[1], comparison_scope=sys.argv[2])
if not passed:
    print('FAIL:')
    for m in missing:
        print('  - ' + m)
    sys.exit(1)
print('PASS: All required sections present')
\" '<output_dir>' '<comparison_scope>'
```

**If validation fails (exit code 1):**

1. Read the FAIL output to identify the issue. Fix in-place, do NOT rewrite the report from scratch. Edit sections in place and not regenerate the entire output.
a. Check if the report contains similar but incorrectly named headers and rename them to match the exact required names. 
b. If sections are entirely absent, add them with the correct `##` headers, keeping existing content.
c. For "Missing metrics row" errors: add the row to the Executive Summary table using values from `category_data/category_manifest.json` (`gpu_utilization` keys) and `priority_data.json` (top bottleneck).
d. For placeholder values (`X ms`, `Y%`, `Z%`, `W%`) in the Executive Summary metrics table: replace each with the actual value from `category_manifest.json` -> `gpu_utilization`.
e. For unfilled `<Brief Title>` / `<Library>` / `<platform>` placeholders: substitute the real title/backend/platform from the corresponding findings file or `metadata/*_metadata.json`.
f. For Args cell mismatches: copy the matching `operations[].args` value verbatim (preserving `<br>`) from the corresponding `category_data/<cat>_metrics.json` and string-replace the bad cell.
g. For marker errors: restore or add the missing/broken marker in place — never delete a card or block to silence an error. Source numeric values from `priority_data.json` (P-items) or `<cat>_metrics.json::impact_estimates[]` (detail estimates); use `null` or the sentinel `not quantifiable from trace data` for non-quantifiable items.
h. For priority-consistency errors (R1 P-item count mismatch, R2 P-item category-order mismatch, R3 marker numeric mismatch, R4 Top Ops row-count mismatch): re-render the affected card(s) by re-reading `priority_data.json::findings[N-1]` for `category`, `low` (impact_score_low), `mid` (impact_score), `high` (impact_score_high), and `priorities[]` for the Top Operations table rows (one row per entry, in array order).
2. Run validation again.
3. Maximum 2 retry attempts. If still failing after retry, proceed with a warning.

---

### 11.2 Optional extension (auto-detected)

If `<agent_extension_file>` exists, run it as shown below. Its behavior is documented in the extension itself; the orchestrator does not need to inspect or reason about it.

If the file is absent, skip this step silently. The analysis is complete; the simple plot from Step 10 stays in place.

```bash
EXT='<agent_extension_file>'
if [ -f "$EXT" ]; then
  <prefix> python3 "$EXT" --output-dir '<output_dir>' --title '<Model> on <Platform> — Kernel Tuning Potential' --comparison-scope <comparison_scope>
fi
```

This step is a hook for an optional extension; if `agent_extension.py` is not present, skip it.

**Do NOT re-run `validate_report` after this step.**

---

### 11.3 Embed Performance Improvement Plot

The PNG (`perf_improvement.png`) is already on disk from either Step 10.3 or Step 11.2 (whichever ran). This step only embeds its base64 sidecar into the report at the `{{PERF_PLOT}}` placeholder.

```bash
<prefix> python3 -c \"
import sys
from TraceLens.Agent.Analysis.utils.plot_utils import embed_plot_in_report
embed_plot_in_report(sys.argv[1])
\" '<output_dir>'
```

If the plot is skipped, the `{{PERF_PLOT}}` placeholder is removed so the report remains clean.
---

## Trace Feature Detection

If Steps 1 or many of Steps 2-5 fail or produce unexpected results, check whether the trace uses the following features before retrying:
- **GPU Graph Replay**: raw trace JSON contains `hipGraphLaunch` or `cudaGraphLaunch`.
  - **Default mode** (analysis_mode = `default`): Inform the user with `[DIAG:trace_quality:GPU_GRAPH_REPLAY]` that GPU graph replay was detected and that the default analysis mode supports typical PyTorch traces. **Abort** -- do not retry or continue.
  - **Inference mode** (analysis_mode = `inference`): Graph launches are expected and supported if graph capture folder is provided, do not abort. If inference_exec_mode is `eager` (no capture folder was provided), continue.
