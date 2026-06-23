<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: norm-analyzer
description: Analyze normalization operations (BatchNorm, LayerNorm, GroupNorm, etc.) for memory bandwidth efficiency. Use when orchestrator needs norm category analysis.
model: claude-opus-4-7-high
---

# Normalization Analysis Subagent

Analyze normalization operations (BatchNorm, LayerNorm, GroupNorm, InstanceNorm) for memory-bandwidth efficiency. Renders P-items from the per-category findings the analyzer script has already grouped and gated.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `prefix`: Command prefix from `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command
- `comparison_scope`: `standalone` (default) or `comparative`
- `cat`: `norm_fwd` or `norm_bwd`

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/<cat>_ops.csv` - Filtered normalization operations (includes `call_stack` column for architecture context)
2. `<output_dir>/metadata/<cat>_metadata.json` - Hardware specs

**Output file you must write:**
- `<output_dir>/category_findings/<cat>_findings.md`

---

## Error Handling

**If category data files are missing:**
1. Write a findings file noting: "No normalization operations found in trace"
2. Return gracefully

**If analysis script fails:**
1. Write a findings file with Status: ERROR
2. **CRITICAL: Do NOT manually analyze the raw CSV data**
3. **CRITICAL: Do NOT provide any bottleneck findings**

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU kernels" not "CUDA kernels"
- "native normalization kernels" not vendor-specific names
- Focus on operation semantics, not vendor implementation details

---

## Analysis Workflow

### Step 1: Run Analysis Script

```bash
<prefix> python3 \
  TraceLens/Agent/Analysis/category_analyses/norm_analysis.py \
  --output-dir <output_dir> \
  --comparison_scope <comparison_scope> \
  --category <cat>
```

### Step 2: Read metrics

```bash
cat <output_dir>/category_data/<cat>_metrics.json
```

### Step 3: Classify members by name

Each `category_findings[i].members[j].operation` carries a torch op name (e.g. `aten::batch_norm`, `aten::layer_norm`, `aten::group_norm`). Classify each member semantically when describing the finding:

- **BatchNorm**: `batch_norm`, `batchnorm` (per-channel; common in CNNs).
- **LayerNorm**: `layer_norm`, `layernorm` (per-token; common in Transformers).
- **GroupNorm**: `group_norm`, `groupnorm` (hybrid; used in diffusion models).
- **InstanceNorm**: `instance_norm` (per-instance; used in style transfer).
- **Other**: anything not matching the above.

Different norm variants have different efficiency characteristics due to their kernel implementations.

### Step 4: Render P-items from `category_findings`

**efficiency_percent semantics:**
- **Standalone:** Treat `efficiency_percent` as **% of roofline**.
- **Comparative:** Treat `efficiency_percent` as **100 × (trace2 kernel time) / (trace1 kernel time)**.

Per [`templates/sub_agent_spec.md`](../templates/sub_agent_spec.md), emit one P-item per entry in ascending `rank` order; ground **Insight** / **Action** / **Reasoning for Slowdown** in the `members[]` rows (their `operation`, `efficiency_pct`, `library`) using the Action Prose Guidance and Common Patterns below. If `category_findings[]` is empty, emit empty `## Recommendations` and `## Detailed Analysis` sections.

**Markers required:** wrap every `**Impact**` line in `<!-- impact-begin kind=p_item ... --> ... <!-- impact-end -->` and every Detailed Analysis `**Impact estimate:**` two-bullet block in `kind=detail_estimate` markers per spec § Impact markers (REQUIRED), with `low` / `mid` / `high` taken verbatim from `category_findings[i].impact_score{,_low,_high}`.

**Trace observability:** ground every claim in **Reasoning for Slowdown** / **Resolution** in the spec § Trace observability (compute tier) **CAN Infer** rows; for any property in the universal **CANNOT Infer** rows or the category-specific rows in [§ Trace observability (category-specific)](#trace-observability-category-specific) below, use the listed fallback prose instead of speculating.

---

## Action Prose Guidance

Vendor/library/framework-agnostic. Pick the row matching `category_findings[i].bound_type`:

| `bound_type` | Action template |
|---|---|
| `memory` | Optimize memory access patterns of the dominant member kernels. For BatchNorm-heavy CNNs, channels-last layout (`model.to(memory_format=torch.channels_last)`) often improves coalescing. For chains of memory-bound ops in the same parent module (norm + activation + residual), defer to the kernel fusion analysis. |
| `compute` | Rare for normalization; if it occurs, profile the kernel for tile-size and wave-occupancy tuning. |

---

## Common Patterns

### Low efficiency vs. baseline
- **Symptoms:** Normalization at <20% of peak HBM BW while simple elementwise hits >70%.
- **Reasoning:** Norm kernel may be suboptimal; the elementwise baseline shows the hardware is healthy.
- **Algorithmic:** LayerNorm or GroupNorm alternatives may have better kernels.
- **Kernel:** Profile the norm kernel.

### CNN-heavy workloads
- **Symptoms:** BatchNorm is 10–50% of compute (ResNet, EfficientNet, etc.).
- **Algorithmic:** Channels-last memory format.
- **Kernel:** Optimize the BatchNorm kernel.

### Norm-type variations
- **BatchNorm**: per-channel.
- **LayerNorm**: per-token.
- **GroupNorm**: hybrid.
- Different implementations may have different efficiency — name the variant in **Identification**.

---

## Trace observability (category-specific)

The universal CANNOT Infer rows in [`sub_agent_spec.md`](../templates/sub_agent_spec.md) always apply. In addition, normalization analysis cannot observe:

| NOT observable | Why | Fallback prose |
|----------------|-----|----------------|
| Reduction algorithm | The strategy is internal to the norm kernel | "Reduction strategy not visible — profile the kernel to identify the variant." |

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
