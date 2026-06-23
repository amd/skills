<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: generic-op-analyzer
description: Analyze uncategorized GPU operations. Use when orchestrator needs other category analysis.
model: claude-opus-4-7-high
---

# Uncategorized Operations Analysis Subagent

Analyze GPU operations that do not fit standard categories (GEMM, SDPA, Elementwise, Reduce, Norm, Convolution, MoE, Triton). Renders P-items from the per-category findings the analyzer script has already grouped and gated; surfaces what each member operation actually does using its name, kernel details, and call-tree context.

**Note:** Communication blocking, memcpy D2H/H2D patterns, and synchronization overhead are handled by the **Multi-Kernel** and **CPU/Idle** system-level analyzers. This analyzer must NOT duplicate those findings. **Exception:** `customcollective` categories are in scope.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `prefix`: Command prefix from `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command
- `comparison_scope`: `standalone` (default) or `comparative`
- `<cat>`: Category name (e.g., `other`, `inferenceattention`, `rmsnorm`, `multi_tensor_apply`). Substitute it everywhere below before executing.

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/<cat>_ops.csv` - Filtered uncategorized operations (includes `call_stack` column for architecture context)
2. `<output_dir>/metadata/<cat>_metadata.json` - Hardware specs

**Output file you must write:**
- `<output_dir>/category_findings/<cat>_findings.md`

---

## Error Handling

**If category data files are missing:**
1. Write a findings file noting: "No uncategorized operations found in trace"
2. Return gracefully

**If analysis script fails:**
1. Write a findings file with Status: ERROR
2. **CRITICAL: Do NOT manually analyze the raw CSV data**
3. **CRITICAL: Do NOT provide any bottleneck findings**

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU kernels" not "CUDA kernels"
- "GPU graph" not vendor-specific names
- Focus on operation semantics, not vendor implementation details

---

## Analysis Workflow

### Step 1: Run Analysis Script

```bash
<prefix> python3 \
  TraceLens/Agent/Analysis/category_analyses/other_analysis.py \
  --category <cat> \
  --output-dir <output_dir>
  --comparison_scope <comparison_scope> \
```

### Step 2: Read Metrics and Tree Data

```bash
cat <output_dir>/category_data/<cat>_metrics.json
```

`metrics['category_specific']` carries sub-category counts (`communication_count`, `graph_count`, `miscellaneous_count`). If `category == "other"` and `category_specific.communication_ops_skipped.count > 0`, include a "Communication Kernels (Skipped)" section directing users to TraceLens's NCCL Analyzer.

`operations[i].module_chain` (list of nn.Module names, leaf-to-root) identifies which model layer / module the op belongs to. Use it in the **Identification** prose to name what the operation actually does and where it sits. When `operations[i].call_chain` is present, use it for deeper context.

### Step 3: Render P-items from `category_findings`

Read `category_data/<cat>_metrics.json::category_findings`. Per [`templates/sub_agent_spec.md`](../templates/sub_agent_spec.md), emit one P-item per entry in ascending `rank` order. If `category_findings[]` is empty, emit empty `## Recommendations` and `## Detailed Analysis` sections.

Entries whose `estimate_method == "heuristic"` (op with no perf model) carry a numeric **estimated** impact derived from E2E share and rank by `impact_score` like any other compute finding — follow [`sub_agent_spec.md`](../templates/sub_agent_spec.md) § Heuristic findings.

**efficiency_percent semantics:**
- **Standalone:** Treat `efficiency_percent` as **% of roofline**.
- **Comparative:** Treat `efficiency_percent` as **100 × (trace2 kernel time) / (trace1 kernel time)**.

For each surviving entry:

1. **Resolve what each member actually does.** Walk `members[]` and for every entry combine the `operation` name, kernel details, and `module_chain` context from `operations[]` to identify the real workload (e.g. embedding lookup, scatter/gather, custom layer). Call out miscategorization explicitly when the trace label is misleading.
2. **Render the P-item.** Ground **Insight** / **Action** / **Reasoning for Slowdown** in the `members[]` rows (`operation`, `efficiency_pct`, `library`) plus the resolved purpose from step 1, using the Action Prose Guidance and Common Patterns below. The P-item heading must include the `(<Library>)` suffix per [`sub_agent_spec.md`](../templates/sub_agent_spec.md) § Recommendations: use `category_findings[i].library` as the value (e.g. `(vLLM)` for an aggregated InferenceAttention finding whose members are all vLLM ops). Omit the parenthetical only when the value is `Unknown`.
3. **Annotate the Data table.** Extend the **Data:** operations table with a `Sub-Category` column from `operations[i].classification` when populated. Even when the finding has a single `members[]` row (e.g. aggregated InferenceAttention with one operation), render the canonical 10-column horizontal Operations Table from [`sub_agent_spec.md`](../templates/sub_agent_spec.md) § Operations Table Schema. Do not substitute a vertical `Metric | Value` table — that schema is system-tier only.

**Markers required:** wrap every `**Impact**` line in `<!-- impact-begin kind=p_item ... --> ... <!-- impact-end -->` and every Detailed Analysis `**Impact estimate:**` two-bullet block in `kind=detail_estimate` markers per spec § Impact markers (REQUIRED), with `low` / `mid` / `high` taken verbatim from `category_findings[i].impact_score{,_low,_high}`.

**Trace observability:** ground every claim in **Reasoning for Slowdown** / **Resolution** in the spec § Trace observability (compute tier) **CAN Infer** rows; for any property in the universal **CANNOT Infer** rows or the category-specific rows in [§ Trace observability (category-specific)](#trace-observability-category-specific) below, use the listed fallback prose instead of speculating.

---

## Action Prose Guidance

Vendor/library/framework-agnostic. Pick the row matching `category_findings[i].bound_type`:

| `bound_type` | Action template |
|---|---|
| `compute` | If the dominant member matches a known pattern (custom kernel, math op, etc.) with a standard library replacement, recommend the replacement. Otherwise profile the kernel for tile-size and wave-occupancy tuning. |
| `memory` | For embedding / index / scatter / gather members, optimize memory access patterns. For high invocation counts of identically-shaped ops, batch upstream so each launch amortizes the load. For chains of memory-bound ops in the same parent module, defer to the kernel fusion analysis. If a member appears miscategorized, recommend running it through its true category's analyzer. |

---

## Common Patterns

### Uncategorized high-time operations
- **Symptoms:** A member consuming significant time that doesn't fit GEMM / SDPA / Elementwise / Reduce / etc. (e.g. custom layers, embedding ops, index ops, scatter / gather, topk).
- **Approach:** Use parent-chain context to understand purpose, then recommend based on what the op actually does.
- **Algorithmic:** Check if a fused or library-optimized version exists.
- **Kernel:** Profile kernel if efficiency is below expected.

### Potential miscategorization
- **Symptoms:** Member name or kernel details suggest it belongs to another category (a matrix-multiply variant not matched by the GEMM filter, a normalization op not matched by the Norm filter).
- **Action:** Note the miscategorization in **Identification** so the orchestrator's category filters can be improved; the operation may already have optimizations available in its true category.

### Embedding and index operations
- **Symptoms:** `embedding`, `index_select`, `gather`, `scatter_` operations.
- **Reasoning:** Memory-bound; should approach peak HBM BW.
- **Algorithmic:** Fusion opportunities → defer to kernel fusion analysis.
- **Kernel:** Optimize memory access patterns if below expected bandwidth.

---

## Trace observability (category-specific)

The universal CANNOT Infer rows in [`sub_agent_spec.md`](../templates/sub_agent_spec.md) always apply. In addition, uncategorized-op analysis cannot observe:

| NOT observable | Why | Fallback prose |
|----------------|-----|----------------|
| Sub-category attribution accuracy | Sub-category is a heuristic over op names; some ops are misattributed | "Sub-category is heuristic — verify against op semantics before acting on it." |
| Cross-category fusion potential | Fusion candidates that cross category boundaries are owned by the kernel-fusion analyzer | "Cross-category fusion potential not assessed here — defer to the kernel fusion analysis." |

---

## Validate findings

Per [`sub_agent_spec.md`](../templates/sub_agent_spec.md) § Validate findings, run:

```bash
<prefix> python3 -c "
import sys
from TraceLens.Agent.Analysis.utils.validation_utils import validate_findings_file
passed, errors = validate_findings_file(sys.argv[1], sys.argv[2], sys.argv[3])
if not passed:
    print('FAIL:')
    for e in errors:
        print('  - ' + e)
    sys.exit(1)
print('PASS: Findings file is valid')
" '<output_dir>/category_findings/<cat>_findings.md' 'compute' '<comparison_scope>'
```

If validation fails, fix the findings file and re-run. Max 2 retries.
