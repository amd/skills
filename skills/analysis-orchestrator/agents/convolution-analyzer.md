<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: convolution-analyzer
description: Analyze Convolution operations for compute efficiency and layout optimization. Use when orchestrator needs Convolution category analysis.
model: claude-opus-4-7-high
---

# Convolution Analysis Subagent

Analyze Convolution operations for compute efficiency and memory-layout optimization. Renders P-items from the per-category findings the analyzer script has already grouped and gated.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `prefix`: Command prefix from `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command
- `cat`: `conv_fwd` or `conv_bwd`
- `comparison_scope`: `standalone` (default) or `comparative`

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/<cat>_ops.csv` - Filtered Convolution operations (includes `call_stack` column for architecture context)
2. `<output_dir>/metadata/<cat>_metadata.json` - Hardware specs

**Output file you must write:**
- `<output_dir>/category_findings/<cat>_findings.md`

---

## Error Handling

**If category data files are missing:**
1. Write a findings file noting: "No Convolution operations found in trace"
2. Return gracefully

**If analysis script fails:**
1. Write a findings file with Status: ERROR
2. **CRITICAL: Do NOT manually analyze the raw CSV data**
3. **CRITICAL: Do NOT provide any bottleneck findings**

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU kernels" not "CUDA kernels"
- "DNN library" not vendor-specific names
- Focus on operation semantics, not vendor implementation details

---

## Analysis Workflow

### Step 1: Run Analysis Script

```bash
<prefix> python3 \
  TraceLens/Agent/Analysis/category_analyses/convolution_analysis.py \
  --output-dir <output_dir> \
  --category <cat> \
  --comparison_scope <comparison_scope>
```

### Step 2: Read metrics

```bash
cat <output_dir>/category_data/<cat>_metrics.json
```

`category_specific.transpose_overhead_percent` flags memory-layout mismatch (NCHW vs NHWC); reference it in **Identification** for any memory-bound finding when it exceeds ~10%.

### Step 3: Classify members by name

Each `category_findings[i].members[j].operation` carries a torch op name (e.g. `aten::conv2d`, `aten::conv_transpose2d`). Classify each member semantically when describing the finding:

- **Standard 2D**: `conv2d` operations (most common in CNNs).
- **1D**: `conv1d` operations (sequence/audio models).
- **3D**: `conv3d` operations (video/volumetric models).
- **Depthwise**: depthwise / channel-wise convolutions (low parallelism, expect lower efficiency).
- **Transpose / Deconv**: transpose convolutions, deconvolutions (also signals potential layout mismatch — cross-reference with `category_specific.transpose_overhead_percent`).
- **Other**: anything not matching the above.

These are guidelines; if a member doesn't fit neatly, classify it semantically.

### Step 4: Render P-items from `category_findings`

**efficiency_percent semantics:**
- **Standalone:** Treat `efficiency_percent` as **% of roofline**.
- **Comparative:** Treat `efficiency_percent` as **100 × (trace2 kernel time) / (trace1 kernel time)**.

Per [`utils/templates/sub_agent_spec.md`](../utils/templates/sub_agent_spec.md), emit one P-item per entry in ascending `rank` order; ground **Insight** / **Action** / **Reasoning for Slowdown** in the `members[]` rows (their `operation`, `efficiency_pct`, `time_ms`, `library`) using the Action Prose Guidance, Expected Efficiency, and Common Patterns below. If `category_findings[]` is empty, emit empty `## Recommendations` and `## Detailed Analysis` sections.

**Markers required:** wrap every `**Impact**` line in `<!-- impact-begin kind=p_item ... --> ... <!-- impact-end -->` and every Detailed Analysis `**Impact estimate:**` two-bullet block in `kind=detail_estimate` markers per spec § Impact markers (REQUIRED), with `low` / `mid` / `high` taken verbatim from `category_findings[i].impact_score{,_low,_high}`.

**Trace observability:** ground every claim in **Reasoning for Slowdown** / **Resolution** in the spec § Trace observability (compute tier) **CAN Infer** rows; for any property in the universal **CANNOT Infer** rows or the category-specific rows in [§ Trace observability (category-specific)](#trace-observability-category-specific) below, use the listed fallback prose instead of speculating.

---

## Action Prose Guidance

Vendor/library/framework-agnostic. Pick the row matching `category_findings[i].bound_type`:

| `bound_type` | Action template |
|---|---|
| `compute` | Profile the dominant member kernels for tile-size and wave-occupancy tuning. Depthwise members will naturally show lower efficiency due to limited parallelism — call that out in **Identification** before recommending tuning. |
| `memory` | If `transpose_overhead_percent` > 10%, recommend converting to channels-last layout (`model.to(memory_format=torch.channels_last)`) to eliminate transpose overhead. Otherwise optimize memory access patterns of the dominant member kernels. |

---

## Expected efficiency per operation type

| Convolution type | Expected efficiency | Bound type |
|------------------|---------------------|------------|
| Large kernels (5×5+) | >70% of peak TFLOPS | compute-bound |
| Standard 3×3 | >70% of peak TFLOPS | compute-bound |
| 1×1 (pointwise) | >60% of peak HBM BW | memory-bound |
| Depthwise | >50% (low parallelism) | varies |

**Transpose overhead bands:**
- `>20%`: high — strongly recommend channels-last.
- `10–20%`: moderate — consider channels-last.
- `<10%`: acceptable.

---

## Common Patterns

### Transpose overhead (layout mismatch)
- **Symptoms:** Many `batched_transpose` kernels; 30–45% of convolution time.
- **Cause:** PyTorch defaults to NCHW; vendor DNN libraries prefer NHWC.
- **Algorithmic (primary):** `model.to(memory_format=torch.channels_last)`.

### Large-kernel convolutions
- **Symptoms:** Kernel size > 3×3, compute-bound.
- **Algorithmic:** Limited — these are typically well-optimized.
- **Kernel:** Profile if efficiency below expected band.

### Small-kernel convolutions (1×1, 3×3)
- **Symptoms:** Common in modern architectures.
- **Algorithmic:** Fusion opportunities → defer to kernel fusion analysis.
- **Kernel:** Optimize memory access patterns.

### Depthwise convolutions
- **Symptoms:** Low efficiency due to limited parallelism.
- **Algorithmic:** Limited optimization potential.
- **Kernel:** Specialized depthwise kernels.

---

## Trace observability (category-specific)

The universal CANNOT Infer rows in [`sub_agent_spec.md`](../utils/templates/sub_agent_spec.md) always apply. In addition, Convolution analysis cannot observe:

| NOT observable | Why | Fallback prose |
|----------------|-----|----------------|
| Per-op layout (NCHW vs. NHWC) | Only the aggregate `category_specific.transpose_overhead_percent` is exposed, not per-op layout | "Per-op layout not visible — refer to aggregate `transpose_overhead_percent`." |

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
" '<output_dir>/category_findings/<cat>_findings.md' 'compute' '<comparison_scope>'
```

If validation fails, fix the findings file and re-run. Max 2 retries.
