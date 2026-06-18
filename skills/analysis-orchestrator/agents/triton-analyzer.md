<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: triton-analyzer
description: Analyze Triton (torch.compile fused) kernels for roofline efficiency. Use when orchestrator needs Triton category analysis.
model: claude-opus-4-7-high
---

# Triton Analysis Subagent

Analyze Triton (torch.compile / inductor) fused GPU kernels for roofline efficiency. Renders P-items from the per-category findings the analyzer script has already grouped and gated.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `prefix`: Command prefix from `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command
- `comparison_scope`: `standalone` (default) or `comparative`

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/triton_ops.csv` - Filtered Triton operations (includes `call_stack` column for architecture context)
2. `<output_dir>/metadata/triton_metadata.json` - Hardware specs

**Output file you must write:**
- `<output_dir>/category_findings/triton_findings.md`

---

## Error Handling

**If category data files are missing:**
1. Write a findings file noting: "No Triton operations found in trace"
2. Return gracefully

**If analysis script fails:**
1. Write a findings file with Status: ERROR
2. **CRITICAL: Do NOT manually analyze the raw CSV data**
3. **CRITICAL: Do NOT provide any bottleneck findings**

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU kernels" not "CUDA kernels"
- "Triton fused kernels" or "torch.compile fused kernels" for the category
- Focus on operation semantics, not vendor implementation details

---

## Analysis Workflow

### Step 1: Run Analysis Script

```bash
<prefix> python3 \
  TraceLens/Agent/Analysis/category_analyses/triton_analysis.py \
  --output-dir <output_dir>
  --comparison_scope <comparison_scope>
```

### Step 2: Read metrics

```bash
cat <output_dir>/category_data/triton_metrics.json
```

`category_specific.pointwise_count`, `reduction_count`, and `persistent_count` indicate the inductor kernel-type mix; reference them in **Identification** when one type dominates a finding.

### Step 3: Classify members by name

Each `category_findings[i].members[j].operation` carries a torch.compile kernel name (e.g. `triton_poi_fused_add_gelu_1`, `triton_red_fused_sum_36`). Classify each member by its inductor prefix when describing the finding:

- **Pointwise**: `triton_poi_` (elementwise fusions — add, mul, gelu, sigmoid, etc.).
- **Reduction**: `triton_red_` (reduction fusions — sum, mean, norm backward, etc.).
- **Persistent**: `triton_per_` (persistent-reduction fusions — layer_norm, etc.).
- **Other**: anything not matching the above.

The fused ATen ops are encoded in the kernel name after the prefix (e.g. `triton_red_fused_add_native_layer_norm_backward_20` fuses `add` + `native_layer_norm_backward`). Use them to describe the dominant computation in prose.

### Step 4: Render P-items from `category_findings`

**efficiency_percent semantics:**
- **Standalone:** Treat `efficiency_percent` as **% of roofline**.
- **Comparative:** Treat `efficiency_percent` as **100 × (trace2 kernel time) / (trace1 kernel time)**.

Per [`utils/templates/sub_agent_spec.md`](../utils/templates/sub_agent_spec.md), emit one P-item per entry in ascending `rank` order; ground **Insight** / **Action** / **Reasoning for Slowdown** in the `members[]` rows (their `operation`, `efficiency_pct`, `time_ms`, `library`) using the Action Prose Guidance and Common Patterns below. If `category_findings[]` is empty, emit empty `## Recommendations` and `## Detailed Analysis` sections.

**Markers required:** wrap every `**Impact**` line in `<!-- impact-begin kind=p_item ... --> ... <!-- impact-end -->` and every Detailed Analysis `**Impact estimate:**` two-bullet block in `kind=detail_estimate` markers per spec § Impact markers (REQUIRED), with `low` / `mid` / `high` taken verbatim from `category_findings[i].impact_score{,_low,_high}`.

**Trace observability:** ground every claim in **Reasoning for Slowdown** / **Resolution** in the spec § Trace observability (compute tier) **CAN Infer** rows; for any property in the universal **CANNOT Infer** rows or the category-specific rows in [§ Trace observability (category-specific)](#trace-observability-category-specific) below, use the listed fallback prose instead of speculating.

---

## Action Prose Guidance

Vendor/library/framework-agnostic. Pick the row matching `category_findings[i].bound_type`:

| `bound_type` | Action template |
|---|---|
| `memory` | Optimize memory access patterns of the dominant member kernels. For chains of memory-bound fused ops in the same parent module, defer to the kernel fusion analysis. |
| `compute` | Rare for fused Triton kernels; if it occurs, profile the kernel for tile-size and wave-occupancy tuning. |

---

## Common Patterns

### Low-efficiency fused kernels (<30% roofline)
- **Symptoms:** Fused kernels with norm or reduction ops at <30% of peak HBM BW.
- **Reasoning:** Fused norm+backward or small-reduction kernels can have suboptimal memory access patterns.
- **Kernel:** Profile the fused kernel; consider dedicated kernel libraries for the dominant op.

### Many small fused kernels
- **Symptoms:** High aggregate count of small Triton kernels with low individual time.
- **Reasoning:** torch.compile may generate many narrow fusions instead of one broad fusion.
- **Kernel:** Review compilation strategy for broader fusion scope.

---

## Trace observability (category-specific)

The universal CANNOT Infer rows in [`sub_agent_spec.md`](../utils/templates/sub_agent_spec.md) always apply. In addition, Triton fused-kernel analysis cannot observe:

| NOT observable | Why | Fallback prose |
|----------------|-----|----------------|
| Per-sub-op breakdown within a fused kernel | Trace only captures the fused kernel as a single event | "Individual sub-op timings within the fused kernel are not separable from the trace." |
| Torch.compile fusion strategy | The inductor fusion decisions are not recorded in the trace | "Fusion strategy not visible — review torch.compile settings if kernels appear under-fused." |

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
" '<output_dir>/category_findings/triton_findings.md' 'compute' '<comparison_scope>'
```

If validation fails, fix the findings file and re-run. Max 2 retries.
