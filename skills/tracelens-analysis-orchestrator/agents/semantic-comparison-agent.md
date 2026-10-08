<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: semantic-comparison-agent
description: End-to-end semantic comparison of two graph-mode GPU traces. Runs deterministic breakdown per trace (extraction + classification + pattern finding + label assembly), then a name-first LLM kernel-name unification pass that establishes cross-trace matching anchors in the semantic_block field, followed by a comparison pipeline.
model: claude-opus-4-8
---

# Semantic Comparison

Orchestrate end-to-end semantic comparison of two GPU traces. The user
provides two raw trace files and the orchestrator handles everything:
deterministic parallel breakdown (no LLM), a name-first LLM kernel-name
unification pass that establishes cross-trace matching anchors, and the
comparison pipeline.

**Why name-first.** In graph mode the CPU->GPU call stack collapses under
`hip/cudaGraphLaunch`, so `nn_module` / `cpu_op` context is unavailable and
block-alignment harmonization cannot work. The only reliable cross-trace
signal is the raw GPU **kernel name**. This workflow unifies kernel names
across the two traces (e.g. `moe_attn_vllm` and `sglang_moe_attention` ->
`moe_attn`) and writes the unified name into each kernel's `semantic_block`
field, which the downstream comparison uses as its matching key.

Use vendor-agnostic terminology (GPU kernels, vendor GEMM library, etc.)
except when quoting actual kernel names from traces.

**Command prefix.** All commands below run through `<prefix>`: read the command
prefix path from your execution context (`<output_dir>/cache/cmd_prefix.txt`) and
substitute the command for `{CMD}`. `<prefix>` is blank for local runs and wraps
commands into the target environment otherwise. Shell control flow (`for`, `if`,
`wait`) and variable assignments run in the driving shell and are not prefixed.

---

## Workflow Steps

```
0.   Query User Inputs
1.   Semantic Breakdown (PARALLEL shell commands, one per trace)
2.   Kernel-Name Unification (name-first anchors + coherence refinement, LLM)
3.   Generate TraceDiff Output (script)
4.   Generate Comparison CSV (script)
```

---

## Step 0: Query User Inputs

Ask the user for:

**Required:**
- Trace A path (.json or .json.gz)
- Trace B path (.json or .json.gz)
- Short labels for each trace (e.g., MI355 / B200)

**Optional:**
- Output directory (default: `comparison_output/`)

**vLLM / annotated traces** are auto-detected by `extract_trace_data.py`.
No special flag is needed.

---

## Step 1: Semantic Breakdown (Deterministic, PARALLEL)

Breakdown is fully deterministic -- no LLM calls. Run both traces as
parallel shell commands.

### 1.1 Per-trace Pipeline

Run the full breakdown for **both traces in a single shell call** using
background jobs + `wait`.

```bash
SCRIPTS=TraceLens/Agent/Analysis/semantic_analyses
CLASSIFY=TraceLens/Agent/Analysis/utils/classify_kernels.py
DIR_A=<output_dir>/work/<name_a>
DIR_B=<output_dir>/work/<name_b>
<prefix> mkdir -p $DIR_A $DIR_B

run_breakdown() {
    local TRACE=$1 DIR=$2

    # Extract (auto-splits vLLM traces into region subdirs)
    <prefix> python3 $SCRIPTS/extract_trace_data.py $TRACE -o $DIR/

    # Check whether extraction produced region subdirs or a flat file.
    # gpu_op_uid is stamped by extract_trace_data.py (raw-index UID aligned
    # with the perf report); no trace-tree build is needed.
    if <prefix> ls $DIR/*/extracted.json >/dev/null 2>&1; then
        for REGION in $DIR/*/; do
            <prefix> python3 $SCRIPTS/pattern_finder.py $REGION/extracted.json -o $REGION/pattern.json &
            <prefix> python3 $CLASSIFY $REGION/extracted.json -o $REGION/classified.json &
        done
        wait
        for REGION in $DIR/*/; do
            <prefix> python3 $SCRIPTS/build_semantic_labels.py \
                $REGION/extracted.json $REGION/classified.json $REGION/pattern.json \
                -o $REGION/semantic_labels.json
        done
    else
        <prefix> python3 $SCRIPTS/pattern_finder.py $DIR/extracted.json -o $DIR/pattern.json &
        <prefix> python3 $CLASSIFY $DIR/extracted.json -o $DIR/classified.json &
        wait
        <prefix> python3 $SCRIPTS/build_semantic_labels.py \
            $DIR/extracted.json $DIR/classified.json $DIR/pattern.json \
            -o $DIR/semantic_labels.json
    fi
}

run_breakdown <trace_a_path> $DIR_A &
run_breakdown <trace_b_path> $DIR_B &
wait
```

**Output directories:**
- Trace A: `<output_dir>/work/<name_a>/`
- Trace B: `<output_dir>/work/<name_b>/`

For multi-region traces, each directory contains per-region subdirs
(e.g., `decode_only_3/`, `prefill_only_1024/`).

### 1.2 Verify Breakdown Outputs

**CRITICAL: DO NOT proceed to Step 2 until both breakdowns have
completed and outputs are verified.**

After both breakdowns complete, verify that `semantic_labels.json` exists
(in each region subdir for multi-region traces, or directly in the trace
directory for single-trace). If either trace failed, report the error
and stop.

---

## Step 2: Kernel-Name Unification (Name-First, LLM)

The LLM unifies raw **kernel names** across the two traces. The unified name is written into each kernel's `semantic_block` field. Kernel names already identical in both traces unify by default (no map entry needed); the LLM only maps names that differ but denote the same operation

Scripts: `TraceLens/Agent/Analysis/semantic_analyses/kernel_unification.py`

### 2.1 Prepare Unification Context

```bash
<prefix> python3 TraceLens/Agent/Analysis/semantic_analyses/kernel_unification.py prepare-context \
    --labels-a <output_dir>/work/<name_a>/semantic_labels.json \
    --labels-b <output_dir>/work/<name_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/work/kernel_unification_context.json
```

### 2.2 Stem Preprocessing (conditional, only if flagged)

If Step 2.1 prints `STEM PREPROCESSING NEEDED` (combined unique names exceed
the threshold, default 5000), the raw name set is too large for the LLM.
Launch the subagent `TraceLens/Agent/Analysis/skills/analysis-orchestrator/agents/kernel-stem-preprocessing-agent.md`, then:

```bash
<prefix> python3 TraceLens/Agent/Analysis/semantic_analyses/kernel_unification.py apply-stem-rules \
    --labels-a <output_dir>/work/<name_a>/semantic_labels.json \
    --labels-b <output_dir>/work/<name_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    --rules <output_dir>/work/stem_rules.json \
    --raw-to-stem <output_dir>/work/raw_to_stem.json \
    -o <output_dir>/work/kernel_unification_context.json
```

Re-run until the printed stem count is within budget.

### 2.3 Launch Kernel Unification Agent

Read `TraceLens/Agent/Analysis/skills/analysis-orchestrator/agents/kernel-unification-agent.md` and
launch it with `kernel_unification_context.json` inline. For multi-region vLLM: run once per matching region pair.

### 2.4 Apply the Map

```bash
<prefix> python3 TraceLens/Agent/Analysis/semantic_analyses/kernel_unification.py apply-map \
    --labels-a <output_dir>/work/<name_a>/semantic_labels.json \
    --labels-b <output_dir>/work/<name_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    --map <output_dir>/work/kernel_unification_map.json \
    --raw-to-stem <output_dir>/work/raw_to_stem.json   # only if 2.2 ran
```

### 2.5 Verify Unification

Check that `kernel_unification_context.json` and `kernel_unification_map.json`
exist in `<output_dir>/work/`, and that `apply-map` reported a non-empty
shared vocabulary.

### 2.6 Coherence Pass (second pass, LLM)

The first pass leaves **one-sided** buckets -- kernels whose unified name
appears in only one trace. This pass uses
the first-pass **shared** buckets as cross-trace positional anchors and
re-labels one-sided buckets by their shared-neighbor context, which (a)
pairs GEMMs across vendors by position and (b) splits a name that occurs in
different contexts. Skip it only if `apply-map` already reported no
meaningful one-sided buckets.

### 2.6a Prepare coherence context

```bash
KC=TraceLens/Agent/Analysis/semantic_analyses/kernel_coherence.py
<prefix> python3 $KC prepare-context \
    --labels-a <output_dir>/work/<name_a>/semantic_labels.json \
    --labels-b <output_dir>/work/<name_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    --neighbor-radius 1 \
    -o <output_dir>/work/kernel_coherence_context.json
```

### 2.6b Launch coherence agent

Read
`TraceLens/Agent/Analysis/skills/analysis-orchestrator/agents/kernel-coherence-agent.md` and launch
it with `kernel_coherence_context.json` inline. It writes
`<output_dir>/work/kernel_coherence_decisions.json` (`context_renames` +
`fallback_remap_a` / `fallback_remap_b`), pairing same-context one-sided buckets
across traces into new shared names.

### 2.6c Apply

```bash
<prefix> python3 $KC apply \
    --context <output_dir>/work/kernel_coherence_context.json \
    --decisions <output_dir>/work/kernel_coherence_decisions.json \
    --audit-csv-a <output_dir>/work/per_kernel_final_<name_a>.csv \
    --audit-csv-b <output_dir>/work/per_kernel_final_<name_b>.csv
```

`apply` rewrites `semantic_block` in place and prints residual one-sided symbols.
Revise decisions and re-run 2.6b--2.6c if meaningful (non-singleton) symbols remain (singleton setup/copy kernels may be accepted); raise `--neighbor-radius` in 2.6a for ambiguous contexts.

---

## Step 3: Generate TraceDiff Output

```bash
<prefix> python3 TraceLens/Agent/Analysis/semantic_analyses/generate_semantic_diff.py \
    <output_dir>/work/<name_a>/semantic_labels.json \
    <output_dir>/work/<name_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    -o <output_dir>/tracediff_output/
```

Produces in `<output_dir>/tracediff_output/`:
- `diff_stats.csv` -- per-kernel rows matching TraceDiff schema
- `diff_stats_unique_args_summary.csv` -- aggregated by semantic block
- `cpu_op_map.json`, `cpu_op_map_trace1.json`, `cpu_op_map_trace2.json`
- `merged_tree_output.txt`

This is a **final deliverable** directory for downstream TraceDiff consumers.

**Perf-report enrichment compatibility:** `diff_stats.csv` carries per-kernel `gpu_op_uid` and per-LCA `busy_time`, consumable by `enrich_perf_report_dict_inplace` in `TraceLens/Reporting/tracediff_comparison_extension.py`.

---

## Key Principles

1. **Conservative anchors** -- map only certain equivalences; preserve
   granularity and leave uncertain names unmapped
2. **No script creation** -- subagents use only existing scripts (the stem
   preprocessing authors regex *rules*, not new scripts)

---

## Final Deliverables

- `<output_dir>/work/` -- per-trace `semantic_labels.json`, the unification/coherence JSON artifacts, `per_kernel_final_<name>.csv`
- `<output_dir>/tracediff_output/` -- TraceDiff deliverables (see Step 3)
