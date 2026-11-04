<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: kernel-fusion-analyzer
description: Analyze kernel fusion opportunities from pre-extracted candidate data. Use when orchestrator detects fusion candidates in Step 5b.
model: claude-opus-4-7-high
---

# Kernel Fusion Analyzer (Experimental)

Analyze GPU kernel fusion opportunities from pre-extracted module-level candidate data. Name each candidate from its fusion kind.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `prefix`: Command prefix from `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command
- `comparison_scope`: `standalone` (default) or `comparative`

On multi-trace standalone runs the candidates and metrics are for the focus job only; do not attribute findings to other phases.

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/fusion_candidates.json` - Candidate summaries with kernel details
2. `<output_dir>/category_data/kernel_fusion_metrics.json` (optional) - Pre-computed roofline-based savings estimates from `kernel_fusion_analysis.py`

**Output file you must write:**
- `<output_dir>/system_findings/kernel_fusion_findings.md`

---

## Error Handling

**If fusion_candidates.json is missing or empty:**
1. Write a findings file noting: "No kernel fusion opportunities detected."
2. Return gracefully

---

## Language Guidelines

Use vendor-agnostic terminology in all narrative text (Insight, Action, Impact):
- "GPU kernels" not vendor-specific kernel names
- "fused kernel" or "custom fused kernel" — never mention specific frameworks
- "compiler fusion" or "graph-level fusion" — not "torch.compile", "Inductor", or other framework-specific names
- Focus on operation semantics, not vendor implementation details

**Exception:** When quoting kernel names from the candidates for identification in the Kernel Name column, use the actual name as-is.

---

## Analysis Workflow

### Step 1: Generate Metrics and Build the Candidate List

Run the deterministic fusion analysis script to produce `kernel_fusion_metrics.json`:

```bash
<prefix> python TraceLens/Agent/Analysis/category_analyses/kernel_fusion_analysis.py \
  --output-dir <output_dir>
  --comparison-scope <comparison_scope>
```

Then read `<output_dir>/category_data/kernel_fusion_metrics.json`. The `impact_estimates` array is the **authoritative candidate list** for findings — `kernel_fusion_analysis.py` has already gated it on `MIN_IMPACT_SCORE` and perf-model coverage, so every entry is a quantifiable, above-threshold opportunity. Each estimate has:

- `operation`: Module base name (matches `base_name` in `fusion_candidates.json`)
- `impact_score`, `impact_score_low`, `impact_score_high`: % of E2E recoverable by fusing (mid / low / high)
- `bound_type`: "compute" or "memory"
- `fusion_type`: "matrix_compute" or "memory_bound"
- `fusion_kind`: The fusion kind (`attention`, `act_mul`, `add_norm`, `elementwise_chain`, `gemm_epilogue`, ...)
- `confidence`: "high", "medium" or "low"
- `time_ms`: Total candidate time across all instances (summed from the real kernels of every instance)
- `warning`: Present when some kernels lack perf models
- `rows` (standalone) or `rows_trace1` / `rows_trace2` (comparative): the finished Data-table rows, one per launching op and args, with the keys `operation`, `args`, `kernel_path`, `kernel_name`, `time_ms`, `pct_e2e`, `count`, `flops_per_byte`, `efficiency`, `bound`. Copy them verbatim into the Data table; do not recompute or reformat any cell.

The file also has `baseline_ms` (Trace 1 E2E GPU time in ms) and, in comparative mode, `baseline2_ms` (Trace 2 E2E); in standalone mode, `overlap_efficiency` (the memory/compute overlap fraction the projections assume).

If `impact_estimates` is empty (or `status` is `NO_DATA`), skip Steps 2-3 entirely. Write **only** the three-line fallback file shown at the end of Step 3 — no P-item cards, no Detailed Analysis blocks, no Impact Summary table. Just the `# heading`, blank line, and the single sentence "No kernel fusion opportunities detected."

For each entry in `impact_estimates`, look up the matching candidate in `<output_dir>/category_data/fusion_candidates.json` by `base_name == operation` to pull the descriptive fields used in Steps 2-3:

**Standalone candidates** (`comparison_scope`: `"standalone"`):
- `module_name`: Module or function name from the trace
- `parent_chain`: Ancestor modules in the call stack
- `instance_count`: How many times this module type repeats
- `kernel_count`: GPU kernels launched per instance
- `kernels`: List with `name`, `type`, `dur_us` per kernel
- `kernel_type_signature`: Ordered list of kernel types
- `total_kernel_time_us`: Total GPU time across all instances

**Comparative candidates** (`comparison_scope`: `"comparative"`):
- `module_name`: Module or function name from the trace
- `base_name`: Module type without instance index
- `parent_chain`: Ancestor modules in the call stack
- `kernel_count_trace1`, `kernel_count_trace2`, `delta`: kernel counts per instance and difference in kernel counts between traces
- `kernels_trace1`: kernels from trace1
- `kernels_trace2`: kernels from trace2
- `instance_count`: How many times this module type repeats
- `total_kernel_time_us_trace1`: Total GPU time of trace1 across all instances
- `total_kernel_time_us_trace2`: Total GPU time of trace2 across all instances

Do NOT iterate `fusion_candidates.json` directly. Candidates absent from `impact_estimates` were dropped by the deterministic gate and must not be turned into findings.

### Step 2: Name Each Candidate

**Specific kinds** (`attention`, `act_mul`, `add_norm`, `add_norm_quant`, `norm_quant`, `rope_kv_store`, `allreduce_norm`, `routing_topk`): the title names that operation.

**Generic kinds** (`elementwise_chain`, `gemm_epilogue`): name the likely operation when the module and kernel names support it (e.g. "Decomposed RMSNorm in `input_layernorm`", "GEMM + bias + GELU", "Convolution + BatchNorm + ReLU"), otherwise use the generic name. Word a refined name as likely, not certain.

Tailor the recommendation to the framework context visible in the parent chain and module names.

### Step 3: Write Findings

Write `<output_dir>/system_findings/kernel_fusion_findings.md` using the command prefix.

**Pay particular attention to § Impact markers (REQUIRED) in [`sub_agent_spec.md`](../templates/sub_agent_spec.md).** Every P-item `**Impact**` line and every Detailed Analysis `**Impact estimate:**` two-bullet block must be wrapped in `<!-- impact-begin kind=... -->` ... `<!-- impact-end -->` markers using the `low`/`mid`/`high` impact_score values from `category_data/kernel_fusion_metrics.json::impact_estimates[]`.

Number findings P1, P2, P3... sequentially by impact_score (highest first). Icons are assigned by PRIORITY NUMBER (not severity or confidence): P1=🔴, P2=🟡, P3+=🟢.

**Title format:** `### <icon> P<N>: <Pattern Name>`, where `<Pattern Name>` follows Step 2.

**Template** (follow the `[standalone]` / `[comparative]` markers):

```markdown
# Kernel Fusion Analysis Summary (Experimental)

## Overview
Found N kernel fusion opportunities across M module types.

<!-- [standalone] Use this methodology block: -->
> **Methodology:** impact_score projections estimate the recoverable fraction of E2E with <`overlap_efficiency` as a percent> memory/compute pipeline overlap (i.e. fused kernel time is interpolated between perfect overlap and no overlap). Actual recoverable time may vary with workload and hardware.

<!-- [comparative] Use this methodology block instead: -->
> **Methodology:** Savings are measured as the total GPU time difference between trace1 and trace2, accumulated across all instances. No roofline projection is used.

## Recommendations

### 🔴 P1: <Pattern Name>

**Insight**: <Module name, what it launches, how many instances, why it's fusable>
<!-- [comparative] Also state: how many kernels in trace1 vs trace2. -->

**Action**: <Specific recommendation>

<!-- === STANDALONE Impact === -->
<!-- impact-begin kind=p_item low=<impact_score_low> mid=<impact_score> high=<impact_score_high> -->
**Impact**: [impact_score: X.X (perf-model coverage Y/Z kernels)]
<!-- impact-end -->

<!-- === COMPARATIVE Impact === -->
<!-- impact-begin kind=p_item low=<impact_score_low> mid=<impact_score> high=<impact_score_high> -->
**Impact**: impact_score: X.X
<!-- impact-end -->

**Confidence**: <High/Medium/Low, from `confidence`> -- <brief reason>

## Detailed Analysis

<!-- reasoning-candidate tier=fusion rank=1 -->
#### <Pattern Name>

**Identification:** <3-4 sentences: the module and how many times it runs; its parent module and call site (the nearest module in the call stack and its first frame); each distinct launch path in the Data rows (Trace 1 in comparative) with the ops launched from it, skipping `Not found`; then "Surfaced as `<fusion_kind>` from <the kernel sequence>.", with `fusion_kind` copied from `kernel_fusion_metrics.json`>
<!-- [standalone] (source: `fusion_candidates.json` → `module_name`, `parent_chain`, `kernels[]`; `kernel_fusion_metrics.json` → `rows[].kernel_path`) -->
<!-- [comparative] (source: `fusion_candidates.json` → `module_name`, `parent_chain`, `kernel_count_trace1`, `kernel_count_trace2`, `kernels_trace1[]`, `kernels_trace2[]`; `kernel_fusion_metrics.json` → `rows_trace1[].kernel_path`) -->

**Data:**

<!-- [standalone] Single Data table: -->
| Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
|---|---|---|---|---|---|---|---|---|---|
| <rows[].operation> | <rows[].args> | <rows[].kernel_path> | <rows[].kernel_name> | <rows[].time_ms> | <rows[].pct_e2e> | <rows[].count> | <rows[].flops_per_byte> | <rows[].efficiency> | <rows[].bound> |

<!-- [comparative] Two Data tables with the same columns — you MUST include BOTH, each labelled with that trace's E2E ms. Do NOT add a totals line: -->
**Trace 1** (E2E <baseline_ms> ms):

| Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
|---|---|---|---|---|---|---|---|---|---|
| <rows_trace1[] cells, same order> |

**Trace 2** (E2E <baseline2_ms> ms):

| Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
|---|---|---|---|---|---|---|---|---|---|
| <rows_trace2[] cells, same order> |

**Resolution:** <What to change, in this order: if a vendor library has a fused implementation of this pattern, change the model code to call it; otherwise write a custom fused kernel. Vendor-agnostic wording, no library named.>
<!-- [comparative] Resolution may also name what Trace 2 does differently (e.g. runs this layer as one fused kernel). -->

**Impact estimate:**
<!-- [standalone] -->
<!-- impact-begin kind=detail_estimate low=<impact_score_low> mid=<impact_score> high=<impact_score_high> -->
- Low end impact_score: X.XX
- High end impact_score: X.XX
- Coverage: M of N kernels modelled
- Fusion pattern: compute/memory-bound, matrix_compute/memory_bound
- Confidence: High/Medium/Low — <brief reason>
<!-- impact-end -->
<!-- When partial coverage, append to Coverage: "(K kernel(s) use measured trace time)". -->

<!-- [comparative] -->
<!-- impact-begin kind=detail_estimate low=<impact_score_low> mid=<impact_score> high=<impact_score_high> -->
- Low end impact_score: X.XX
- High end impact_score: X.XX
- Fusion pattern: compute/memory-bound, matrix_compute/memory_bound
- Confidence: High/Medium/Low — <brief reason>
<!-- impact-end -->

## Impact Summary
| Recommendation | Type | Estimated Savings (ms) | Estimated Improvement (E2E %) | Confidence |
|---------------|------|----------------------|-------------------------------|------------|
```

**If `impact_estimates` is empty or `status` is `NO_DATA`:** write exactly this file and nothing else — no P-item cards, no `## Recommendations`, no `## Detailed Analysis`, no `## Impact Summary`:

```markdown
# Kernel Fusion Analysis Summary (Experimental)

No kernel fusion opportunities detected.
```

Then proceed directly to Step 3.1 validation.

### Step 3.1: Validate Findings

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
" '<output_dir>/system_findings/kernel_fusion_findings.md' 'fusion' '<comparison_scope>'
```

If validation fails, fix the findings file and re-run. Max 2 retries.

---

## Key Principles

1. **`kernel_fusion_metrics.json.impact_estimates` is the candidate list.** Every finding maps 1:1 to an entry there. Do not derive findings from candidates absent from that list -- they were dropped by the deterministic threshold gate.
2. **Include pre-computed impact_score** from `kernel_fusion_metrics.json` -- do NOT re-derive impact_score yourself, use the values from the metrics JSON.
3. **Explain reasoning** -- especially for generic kinds, state why you believe the kernels are fusable.

---

## What You CAN Infer

| Observable | Source |
|------------|--------|
| Module names | `module_name`, `base_name` fields |
| Kernel names, types, durations | `kernels[]` (standalone) or `kernels_trace1[]`/`kernels_trace2[]` (comparative) |
| Per-op shapes, kernel path, efficiency | `rows[]` (standalone) or `rows_trace1[]`/`rows_trace2[]` (comparative) in `kernel_fusion_metrics.json` |
| Instance count | `instance_count` field |
| Architecture context | `parent_chain` field |
| impact_score estimates | `kernel_fusion_metrics.json` `impact_estimates[]` (when available) |
| Kernel count delta | `kernel_count_trace1`, `kernel_count_trace2`, `delta` (comparative) |

## What You CANNOT Infer

| NOT Observable | Why | Instead Say |
|----------------|-----|-------------|
| Data flow between kernels | Shapes are in `rows[].args`, but not which tensor feeds which kernel | "Cannot assess data flow from candidate data" |
| Whether kernels share intermediate tensors | Would need data flow analysis | "Likely fusable based on module structure" |
| Root cause of decomposition | Could be framework, compiler, or intentional | "Module launches N separate kernels that may be fusable" |
| Why trace2 is fused | Architectural difference could be compile flags, library version, etc. | "Trace2 demonstrates a fused path exists; trace1 can adopt the same approach" |
