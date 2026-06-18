<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: kernel-fusion-analyzer
description: Analyze kernel fusion opportunities from pre-extracted candidate data. Use when orchestrator detects fusion candidates in Step 4b.
model: claude-opus-4-7-high
---

# Kernel Fusion Analyzer (Experimental)

Analyze GPU kernel fusion opportunities from pre-extracted module-level candidate data. Classify candidates as known patterns, novel patterns, or not fusable.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `prefix`: Command prefix from `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command
- `comparison_scope`: `standalone` (default) or `comparative`

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

**Exception:** When quoting kernel names from the candidates for identification in the Kernels table, use the actual name as-is.

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
- `confidence`: "high" or "medium"
- `time_ms`: Total candidate time across all instances
- `warning`: Present when some kernels lack perf models

If `impact_estimates` is empty (or `status` is `NO_DATA`), skip Steps 2-4 entirely. Write **only** the three-line fallback file shown at the end of Step 4 — no P-item cards, no Detailed Analysis blocks, no Impact Summary table. Just the `# heading`, blank line, and the single sentence "No kernel fusion opportunities detected."

For each entry in `impact_estimates`, look up the matching candidate in `<output_dir>/category_data/fusion_candidates.json` by `base_name == operation` to pull the descriptive fields used in Steps 2-4:

**Standalone candidates** (`comparison_scope`: `"standalone"`):
- `module_name`: Module or function name from the trace
- `parent_chain`: Ancestor modules in the call stack
- `instance_count`: How many times this module type repeats
- `kernel_count`: GPU kernels launched per instance
- `kernels`: List with `name`, `type`, `dur_us` per kernel
- `kernel_type_signature`: Ordered list of kernel types
- `has_fused_kernel`: Whether subtree contains a fused kernel
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

### Step 2: Classify Each Candidate

For each candidate, make three decisions:

**Decision 1 -- Is this a fusion opportunity?** Reject candidates where:
**Standalone only**:
- The kernels are genuinely independent operations (e.g., separate projection GEMMs reading different weight matrices)
- The module is a container (Sequential, ModuleList, full decoder/encoder layer)
- All kernels are GEMMs
- The non-GEMM kernels are all normalization ops (GEMM + LayerNorm/Norm sequences are not fusable)
- Any kernel is a Triton-compiled fused kernel (`triton_` prefix)
- The module already contains a fused kernel (`has_fused_kernel: true`)

**Decision 2 -- What pattern?** Check known patterns first:

| Pattern | Kernel composition | Module name hints |
|---------|-------------------|-------------------|
| Unfused attention | >= 2 GEMM + Softmax, no fused attention kernel | "attention", "sdpa", "self_attn" |
| Unfused RMSNorm | rsqrt + mean or pow + mul | "rmsnorm", "rms_norm" |
| Unfused LayerNorm | rsqrt + mean + sub + mul | "layernorm", "layer_norm" |
| Unfused BatchNorm | mul + add (precomputed scale+shift) | "batchnorm", "batch_norm", "FrozenBatchNorm" |
| Unfused RoPE | neg + cat + mul + add | "rotary", "rope", "apply_rotary" |
| Unfused SiGLU/SwiGLU | SiLU + Mul (may have GEMMs between) | "silu", "swiglu", MLP context |
| Unfused GELU | Multiple GELU component kernels | "gelu" |
| GEMM epilogue | GEMM + 1-2 elementwise as separate kernels | "linear", "conv2d", "addmm" |

Then look for novel patterns:
- Multiple elementwise kernels under one module
- Reduction + elementwise sequences
- Dropout + residual add + normalization under one module
- Repeated small kernels suggesting a decomposed operation

**Decision 3 -- What recommendation?** Tailor to framework context visible in the parent chain and module names.

### Step 3: Assign Confidence

Use the `confidence` from `kernel_fusion_metrics.json` when available. Otherwise:

- **high**: Module name matches a known pattern AND kernel composition confirms it
- **medium**: Module name OR kernel composition suggests a pattern, but not both
- **low**: Speculative - structural analysis suggests fusion is possible

### Step 4: Write Findings

Write `<output_dir>/system_findings/kernel_fusion_findings.md` using the command prefix.

**Pay particular attention to § Impact markers (REQUIRED) in [`sub_agent_spec.md`](../utils/templates/sub_agent_spec.md).** Every P-item `**Impact**` line and every Detailed Analysis `**Impact estimate:**` two-bullet block must be wrapped in `<!-- impact-begin kind=... -->` ... `<!-- impact-end -->` markers using the `low`/`mid`/`high` impact_score values from `category_data/kernel_fusion_metrics.json::impact_estimates[]`.

Number findings P1, P2, P3... sequentially by impact_score (highest first). The icon is set ONLY by the `confidence` field in `kernel_fusion_metrics.json`:

| Confidence | Icon |
|------------|------|
| high       | 🔴   |
| medium     | 🟡   |
| low        | 🟢   |

Example: if the highest-savings finding has LOW confidence, write `### 🟢 P1:`. Two HIGH findings in a row are `### 🔴 P1:` and `### 🔴 P2:` (both red).

**Title format:** `### <icon> P<N>: <Pattern Name>`

**Template** (follow the `[standalone]` / `[comparative]` markers):

```markdown
# Kernel Fusion Analysis Summary (Experimental)

## Overview
Found N kernel fusion opportunities across M module types.

<!-- [standalone] Use this methodology block: -->
> **Methodology:** impact_score projections estimate the recoverable fraction of E2E with 85% memory/compute pipeline overlap (i.e. fused kernel time is interpolated between perfect overlap and no overlap). Actual recoverable time may vary with workload and hardware.

<!-- [comparative] Use this methodology block instead: -->
> **Methodology:** Savings are measured as the total GPU time difference between trace1 and trace2, accumulated across all instances. No roofline projection is used.

## Recommendations

### 🔴 P1: <Pattern Name> (<time_ms> ms, <instance_count> instances)

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

**Confidence**: High/Medium/Low -- <brief reason>

## Detailed Analysis

<!-- reasoning-candidate tier=fusion rank=1 -->
#### <Pattern Name> (<time_ms> ms, <instance_count> instances)

**Identification:** <1-2 sentences: how this fusion candidate was surfaced>
<!-- [standalone] (source: `fusion_candidates.json` → `module_name`, `has_fused_kernel`, `kernels[]`) -->
<!-- [comparative] (source: `fusion_candidates.json` → `module_name`, `kernel_count_trace1`, `kernel_count_trace2`, `kernels_trace1[]`, `kernels_trace2[]`) -->

<!-- [standalone] Single kernel table: -->
**Data:**

| Kernel | Type | Duration (us) | Perf model |
|--------|------|--------------|------------|
| <kernel name (truncated to ~60 chars)> | <type> | X.X | Yes/No |

<!-- [comparative] Two kernel tables — you MUST include BOTH: -->
**Trace1 kernels:**

| Kernel | Type | Duration (us) |
|--------|------|--------------|
| <kernel name (truncated to ~60 chars)> | <type> | X.X |

**Trace2 kernels:**

| Kernel | Type | Duration (us) |
|--------|------|--------------|
| <kernel name (truncated to ~60 chars)> | <type> | X.X |

**Impact estimate:**
<!-- [standalone] -->
<!-- impact-begin kind=detail_estimate low=<impact_score_low> high=<impact_score_high> -->
- Low end impact_score: X.XX
- High end impact_score: X.XX
- Coverage: M of N kernels modelled
- Fusion pattern: compute/memory-bound, matrix_compute/memory_bound
- Confidence: High/Medium/Low — <brief reason>
<!-- impact-end -->
<!-- When partial coverage, append to Coverage: "(K kernel(s) use measured trace time)". -->

<!-- [comparative] -->
<!-- impact-begin kind=detail_estimate low=<impact_score_low> high=<impact_score_high> -->
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

Then proceed directly to Step 4.1 validation.

### Step 4.1: Validate Findings

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
" '<output_dir>/system_findings/kernel_fusion_findings.md' 'fusion' '<comparison_scope>'
```

If validation fails, fix the findings file and re-run. Max 2 retries.

---

## Key Principles

1. **`kernel_fusion_metrics.json.impact_estimates` is the candidate list.** Every finding maps 1:1 to an entry there. Do not derive findings from candidates absent from that list -- they were dropped by the deterministic threshold gate.
2. **Include pre-computed impact_score** from `kernel_fusion_metrics.json` -- do NOT re-derive impact_score yourself, use the values from the metrics JSON.
3. **Let the data speak** -- classify based on module names AND kernel composition, not just one signal.
4. **Reject confidently** -- not every multi-kernel module is a fusion opportunity; independent operations under a container module are not fusable. Use Step 2's Decision 1 to drop candidates from `impact_estimates` that turn out to be containers, all-GEMM groups, or already-fused subtrees.
5. **Explain reasoning** -- especially for novel patterns, state why you believe the kernels are fusable.
6. Use the **module name** to determine the user-facing operation name. If the module is `aten::conv2d` or `Conv2d`, call it "Convolution" in the finding title, not "GEMM" -- even though convolutions are implemented as GEMMs internally.

---

## What You CAN Infer

| Observable | Source |
|------------|--------|
| Module names | `module_name`, `base_name` fields |
| Kernel names, types, durations | `kernels[]` (standalone) or `kernels_trace1[]`/`kernels_trace2[]` (comparative) |
| Instance count | `instance_count` field |
| Architecture context | `parent_chain` field |
| Already-fused status | `has_fused_kernel` field |
| impact_score estimates | `kernel_fusion_metrics.json` `impact_estimates[]` (when available) |
| Kernel count delta | `kernel_count_trace1`, `kernel_count_trace2`, `delta` (comparative) |

## What You CANNOT Infer

| NOT Observable | Why | Instead Say |
|----------------|-----|-------------|
| Tensor shapes | Not in candidate JSON | "Cannot assess data flow from candidate data" |
| Whether kernels share intermediate tensors | Would need data flow analysis | "Likely fusable based on module structure" |
| Root cause of decomposition | Could be framework, compiler, or intentional | "Module launches N separate kernels that may be fusable" |
| Why trace2 is fused | Architectural difference could be compile flags, library version, etc. | "Trace2 demonstrates a fused path exists; trace1 can adopt the same approach" |
