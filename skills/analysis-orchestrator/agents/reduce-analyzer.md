<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: reduce-analyzer
description: Analyze reduce operations for performance bottlenecks and optimization opportunities. Use when orchestrator needs reduce category analysis.
model: claude-opus-4-7-high
---

# Reduce Analysis Subagent

Analyze reduce operations (softmax, sum, mean, max, min) for memory-bandwidth efficiency. Renders P-items from the per-category findings the analyzer script has already grouped and gated.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `prefix`: Command prefix from `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command
- `comparison_scope`: `standalone` (default) or `comparative`

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/reduce_ops.csv` - Filtered reduce operations (includes `call_stack` column for architecture context)
2. `<output_dir>/metadata/reduce_metadata.json` - Hardware specs

**Output file you must write:**
- `<output_dir>/category_findings/reduce_findings.md`

---

## Error Handling

**If category data files are missing:**
1. Write a findings file noting: "No reduce operations found in trace"
2. Return gracefully

**If analysis script fails:**
1. Write a findings file with Status: ERROR
2. **CRITICAL: Do NOT manually analyze the raw CSV data**
3. **CRITICAL: Do NOT provide any bottleneck findings**

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU kernels" not "CUDA kernels"
- "memory bandwidth" not vendor-specific terms
- Focus on operation semantics, not vendor implementation details

---

## Analysis Workflow

### Step 1: Run Analysis Script

```bash
<prefix> python3 \
  TraceLens/Agent/Analysis/category_analyses/reduce_analysis.py \
  --output-dir <output_dir>
  --comparison_scope <comparison_scope>
```

### Step 2: Read metrics

```bash
cat <output_dir>/category_data/reduce_metrics.json
```

`category_specific.softmax_count` flags attention-pattern reductions; reference it in **Identification** when softmax dominates a finding.

### Step 3: Classify members by name

Each `category_findings[i].members[j].operation` carries a torch op name (e.g. `aten::softmax`, `aten::sum`, `aten::mean`). Classify each member semantically when describing the finding:

- **Softmax**: `softmax` (attention activation; common in Transformer attention layers).
- **Sum**: `sum` (element summation across dimensions; common in loss / gradient accumulation).
- **Mean**: `mean`, `avg` (average reduction; used in pooling and normalization).
- **Max**: `max` (maximum-value reduction; used in argmax and pooling).
- **Min**: `min` (minimum-value reduction; used in clamping and threshold logic).
- **Other**: anything not matching the above.

### Step 4: Render P-items from `category_findings`

Per [`utils/templates/sub_agent_spec.md`](../utils/templates/sub_agent_spec.md), emit one P-item per entry in ascending `rank` order; ground **Insight** / **Action** / **Reasoning for Slowdown** in the `members[]` rows (their `operation`, `efficiency_pct`, `library`) using the Action Prose Guidance and Common Patterns below. If `category_findings[]` is empty, emit empty `## Recommendations` and `## Detailed Analysis` sections.

**efficiency_percent semantics:**
- **Standalone:** Treat `efficiency_percent` as **% of roofline**.
- **Comparative:** Treat `efficiency_percent` as **100 × (trace2 kernel time) / (trace1 kernel time)**.

**Markers required:** wrap every `**Impact**` line in `<!-- impact-begin kind=p_item ... --> ... <!-- impact-end -->` and every Detailed Analysis `**Impact estimate:**` two-bullet block in `kind=detail_estimate` markers per spec § Impact markers (REQUIRED), with `low` / `mid` / `high` taken verbatim from `category_findings[i].impact_score{,_low,_high}`.

**Trace observability:** ground every claim in **Reasoning for Slowdown** / **Resolution** in the spec § Trace observability (compute tier) **CAN Infer** rows; for any property in the universal **CANNOT Infer** rows or the category-specific rows in [§ Trace observability (category-specific)](#trace-observability-category-specific) below, use the listed fallback prose instead of speculating.

---

## Action Prose Guidance

Vendor/library/framework-agnostic. Pick the row matching `category_findings[i].bound_type`:

| `bound_type` | Action template |
|---|---|
| `memory` | Optimize memory access patterns of the dominant member kernels. For softmax members in an attention parent chain, the unfused softmax indicates a fusion opportunity — defer to the kernel fusion analysis. For chains of memory-bound reductions in the same parent module (norm + reduce + scale), defer to the kernel fusion analysis. |
| `compute` | Rare for reductions; if it occurs, profile the kernel for wave-occupancy tuning. |

---

## Common Patterns

### Standalone reductions
- **Symptoms:** `sum`, `mean`, `max` operations in isolation (no fusion candidate above).
- **Reasoning:** Memory-bound reductions should approach peak HBM BW for simple cases.
- **Kernel:** Investigate kernel-level memory access patterns if well below the band.

---

## Trace observability (category-specific)

The universal CANNOT Infer rows in [`sub_agent_spec.md`](../utils/templates/sub_agent_spec.md) always apply. In addition, reduce analysis cannot observe:

| NOT observable | Why | Fallback prose |
|----------------|-----|----------------|
| Reduction algorithm (tree vs. block-shuffle vs. atomic) | The strategy is internal to the reduce kernel | "Reduction strategy not visible — profile the kernel to identify the variant." |

---

## Validate findings

Per [`sub_agent_spec.md`](../utils/templates/sub_agent_spec.md) § Validate findings, run:

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
" '<output_dir>/category_findings/reduce_findings.md' 'compute' '<comparison_scope>'
```

If validation fails, fix the findings file and re-run. Max 2 retries.
