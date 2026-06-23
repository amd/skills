<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

<!--
=== FORMATTING RULES (for the agent filling in this template) ===

=== MODE SELECTION ===
This template supports two modes determined by `comparison_scope`:
  - **standalone**: Single-trace roofline analysis (default). Use sections marked STANDALONE.
  - **comparative**: Two-trace analysis (Trace 1 =  primary, Trace 2 = target). Use sections marked COMPARATIVE.
When filling in this template, select the block matching the active `comparison_scope` for each
section that has STANDALONE / COMPARATIVE variants. Delete the unused variant.

=== COMPARATIVE TERMINOLOGY ===
  - **Trace 1** =  trace (primary). **Trace 2** = trace (target/comparison).
  - Impact semantics: standalone uses roofline gap; comparative uses
    trace 2 kernel time as the optimization target (gap = trace1 time − trace2 time).
  - Comparative speed semantics: express as "X% faster" or "X% slower" relative to Trace 1. If t2 < t1 → "X% faster"; if t2 > t1 → "X% slower".
    Standalone efficiency semantics: % of roofline (unchanged).

=== GENERAL RULES ===
1. Warnings section: Only include if there were errors or high-variance operations; omit entirely if all succeeded and no variance flags.
2. Executive Summary: Max ~20 lines.
3. Performance plot: The {{PERF_PLOT}} placeholder is replaced by Step 11.3 with a base64-embedded
   PNG data URI (![Performance Breakdown](data:image/png;base64,...)) of a single horizontal stacked
   bar showing the run's compute-time breakdown by kernel category. The plot is purely descriptive
   (no error bars, no throughput cone, no savings estimates). If the plot was not generated
   (Step 10.3 / Step 11.2 failed), the placeholder is removed.
4. Compute Kernel Optimizations: One P-item per entry in `priority_data.json::findings[]`,
   numbered P1, P2, ... in `findings[]` order (already globally sorted by `impact_score`). The
   P-item Impact line uses the canonical mid `impact_score` value; low/high values appear only
   in Detailed Analysis. Cards join the corresponding sub-agent's Detailed Analysis block by
   `(findings[i].category, findings[i].category_rank)`.
5. System-Level Optimizations: If all system-level analyses report no actionable issues
   (NONE/N/A severity), use a single "✅ No system-level bottlenecks detected" summary instead of
   P1/P2/P3 recommendations. Only generate numbered priorities when at least one actionable issue
   exists (number sequentially from P1, including CPU/Idle first if invoked).
6. Each section is independently composable -- can be shared standalone.
7. All three tiers (Compute, Kernel Fusion, System) use separate sequential P1/P2/P3 numbering (no gaps).
8. Priority icons are assigned by PRIORITY NUMBER, not severity:
   - Compute Kernel: 🔴 P1 → 🟡 P2 → 🟢 P3 → 🟢 P4 ...
   - Kernel Fusion: icon by confidence (🔴 high → 🟡 medium → 🟢 low), not priority number
   - System-Level: 🔴 P1 → 🟡 P2 → 🟢 P3 → 🟢 P4 ... (only when actionable issues exist)
9. Field labels — each section uses EXACTLY these labels:

   OPTIMIZATION CARDS (§Compute Kernel Optimizations, §Kernel Fusion, §System-Level):
   - Compute Kernel P-items: **Insight** / **Action** / **Impact**
   - Kernel Fusion P-items:  **Insight** / **Action** / **Impact** / **Confidence**
   - System-Level P-items:   **Insight** / **Action** / **Impact**

   DETAILED ANALYSIS (§Detailed Analysis only):
   - Compute / System blocks: **Identification:** / **Data:** / **Reasoning for Slowdown:** / **Resolution:** / **Impact estimate:**
   - Kernel Fusion blocks:    **Identification:** / **Data:** / **Impact estimate:**

10. Detailed Analysis: three subsections (`### Compute Kernel Insights`, `### Kernel Fusion Insights`, `### System-Level Insights`) with `#### 🔴/🟡/🟢 Pn: <Brief Title>` blocks matching card titles and order.
11. Model and appendix: Use `model_info["model"]` from `metadata/model_info.json` for the
    report title (fall back to "Workload" if "Cannot be inferred from trace"). Fill Appendix
    **Model Architecture** with the raw `model`, `architecture`, `scale`, `precision` values.
12. Library parenthetical: Compute Kernel card titles and Detailed Analysis headings must include
    the library name(s) in parentheses when present in the sub-agent findings. Omit when no
    library is identified. System-Level and Kernel Fusion titles do NOT include a library
    parenthetical.
-->

<!-- === STANDALONE title === -->
# <Model> - <Platform> Standalone Analysis

<!-- === COMPARATIVE title === -->
# <Model> - Comparative Analysis: <Platform1> vs <Platform2>

## Executive Summary

<!-- === STANDALONE Executive Summary === -->
[1 paragraph overview + key metrics table]

<!-- MANDATORY: This table must contain exactly these 5 rows:
     Total Time | Compute % | Idle % | Exposed Communication % | Top Bottleneck Category
     Top Bottleneck Category V% = gpu_kernel_time_ms of top category / (gpu_utilization.total_time_ms * computation_time_percent / 100) -->
| Metric | Value |
|--------|-------|
| Total Time | X ms |
| Compute % | Y% |
| Idle % | Z% |
| Exposed Communication % | W% |
| Top Bottleneck Category | Category (V%) |

<!-- === COMPARATIVE Executive Summary === -->
[1 paragraph comparative overview: summarize which trace is faster overall, by how much, and the dominant gap categories]

<!-- Top Bottleneck Category X% = top category's gpu_kernel_time_ms / (manifest.gpu_utilization.total_time_ms * manifest.gpu_utilization.computation_time_percent / 100)
     Top Bottleneck Category Y% = top category's gpu_kernel_time_ms / (manifest.trace2_gpu_utilization.total_time_ms * manifest.trace2_gpu_utilization.computation_time_percent / 100)
     Difference = Trace 2 value − Trace 1 value -->
| Metric | Trace 1 - (<Platform1>) | Trace 2 - (<Platform2>) | Difference |
|--------|----------------------------|-------------------------------|------------|
| Total Time | X ms | Y ms | +/-Z ms (+/-W%) |
| Compute % | X% | Y% | +/-Z% |
| Idle % | X% | Y% | +/-Z% |
| Exposed Communication % | X% | Y% | +/-Z% |
| Top Bottleneck Category | Category (X%) | Category (Y%) | — |

{{PERF_PLOT}}

## Warnings

**Include this section ONLY if any subagent failed OR any operation has high_variance: true in *_metrics.json:**

<!-- Subagent failures (if any): -->
The following analyses could not be completed due to script failures:

| Analysis | Tier | Error Summary |
|----------|------|---------------|
| <name> | System / Compute Kernel | <brief error description> |

These are excluded from the recommendations below.

<!-- Data quality warnings (if any operation has high_variance: true in *_metrics.json): -->
**Data Quality:** The following operations have unreliable kernel time measurements (CoV > 1.0, indicating extreme variance across instances — likely a profiler timing artifact):

| Operation | Category | CoV | Reported Time (ms) |
|-----------|----------|-----|-------------------|
| <name> | <category> | X.X | Y.Y |

---

## Compute Kernel Optimizations

Findings from per-category kernel analysis (GEMM, SDPA, elementwise, etc.).
Summaries of recommendations from Step 7 sub-agents, focused on individual kernel efficiency.

### Top Operations

One row per entry in `priority_data.json::priorities[]`, in array order (no manifest-sort, no extra rows). For row N (= `priorities[N-1]`): `Rank`/`Category` = `rank`/`display_name`; `Time (ms)` = matching `manifest.categories[].gpu_kernel_time_ms` (verbatim); `Ops` = matching `manifest.categories[].ops_count`; `% of Compute Time` = `Time (ms) / (gpu_utilization.total_time_ms * computation_time_percent / 100)`; trailer `low`/`high` = `priorities[N-1].impact_score_low`/`impact_score_high` (use `null` for `source: "manifest_fallback"`). Wrap the whole block (header + separator + rows) in the `kind=top_ops` marker.

<!-- === STANDALONE Top Operations === -->
<!-- impact-begin kind=top_ops -->
| Rank | Category | Time (ms) | % of Compute Time | Ops |
|------|----------|-----------|-------------------|-----|
| 1 | ... | ... | ... | ... | <!-- top-ops-row low=<impact_score_low> high=<impact_score_high> -->
<!-- impact-end -->

<!-- === COMPARATIVE Top Operations === -->
`Trace 2 Time (ms)` = matching `manifest.trace2_ops_summary_by_category[]["total_direct_kernel_time_ms"]` where `"op category"` matches the row Category **case-insensitively**; use — if no match.
`Difference (ms)` = Trace 2 Time − Trace 1 Time.
<!-- impact-begin kind=top_ops -->
| Rank | Category | Trace 1 Time (ms) | Trace 2 Time (ms) | % of Compute Time | Ops | Difference (ms) |
|------|----------|-------------------|-------------------|-------------------|-----|-----------------|
| 1 | ... | ... | ... | ... | ... | +/-X.X or — | <!-- top-ops-row low=<impact_score_low> high=<impact_score_high> -->
<!-- impact-end -->

<!-- === NO ACTIONABLE FINDINGS (all quantified compute categories have empty category_findings[] in *_metrics.json) === -->
<!-- Use when priority_data / per-category metrics show no compute P-items to render (category_findings[] empty everywhere that applies). -->
✅ No compute kernel optimization opportunities identified. All categories are within target performance bounds.

<!-- === ACTIONABLE FINDINGS (at least one compute category has P-items) === -->
<!-- Icon mapping by PRIORITY NUMBER (not severity): P1=🔴, P2=🟡, P3+=🟢 -->
<!-- One card per entry in priority_data.findings[] in array order. Title uses the entry's category and library; Action text is category-appropriate. Do NOT recommend "fuse the SDPA kernel" (already fused — defer upstream/downstream fusion to Kernel Fusion section). -->
<!-- Skip categories that have empty category_findings[] in category_data/<cat>_metrics.json (no P-items for that category). -->
<!-- Heuristic findings (priority_data.findings[i].estimate_method == "heuristic") carry a numeric estimated impact and sort by impact_score like any other compute finding; render per sub_agent_spec.md § Heuristic findings. -->

### 🔴 P1: <Brief Title> (<Library>)

**Insight**: [1 sentence - what's wrong]

**Action**: [1-2 sentences - category-appropriate: GEMM fusion/tile/library; SDPA tile/backend; elementwise fusion; etc.]

<!-- impact-begin kind=p_item category=<priority_data.findings[0].category> low=<priority_data.findings[0].impact_score_low> mid=<priority_data.findings[0].impact_score> high=<priority_data.findings[0].impact_score_high> -->
**Impact**: [impact_score: X.X, OR "Not quantifiable from trace data"]
<!-- impact-end -->

→ *See [Detailed Analysis: Compute kernel insights > P1](#detailed-analysis-compute-p1) for details*

---

### 🟡 P2: <Brief Title> (<Library>)

**Insight**: [1 sentence]

**Action**: [1-2 sentences]

<!-- impact-begin kind=p_item category=<priority_data.findings[1].category> low=<priority_data.findings[1].impact_score_low> mid=<priority_data.findings[1].impact_score> high=<priority_data.findings[1].impact_score_high> -->
**Impact**: [impact_score: X.X, OR "Not quantifiable from trace data"]
<!-- impact-end -->

→ *See [Detailed Analysis: Compute kernel insights > P2](#detailed-analysis-compute-p2) for details*

---

### 🟢 P3: <Brief Title> (<Library>)

**Insight**: [1 sentence]

**Action**: [1-2 sentences]

<!-- impact-begin kind=p_item category=<priority_data.findings[2].category> low=<priority_data.findings[2].impact_score_low> mid=<priority_data.findings[2].impact_score> high=<priority_data.findings[2].impact_score_high> -->
**Impact**: [impact_score: X.X, OR "Not quantifiable from trace data"]
<!-- impact-end -->

→ *See [Detailed Analysis: Compute kernel insights > P3](#detailed-analysis-compute-p3) for details*

<!-- All additional P-items (P4, P5, ...) follow the same pattern, sourcing markers from priority_data.findings[N-1]. Detailed Analysis links: → *See [Detailed Analysis: Compute kernel insights > PN](#detailed-analysis-compute-pN) for details* -->

---

## Kernel Fusion Opportunities (Experimental)
<!-- === STANDALONE Kernel Fusion === -->
> **Note:** Kernel fusion analysis is experimental. impact_score projections estimate the recoverable fraction of E2E with 85% memory/compute pipeline overlap. Kernels without perf models use their measured trace time as-is. Candidates where fewer than 75% of kernels have perf models are not reported. Actual recoverable time depends on implementation feasibility and interaction effects.
<!-- === COMPARATIVE Kernel Fusion === -->
> **Note:** Kernel fusion analysis is experimental.

<!-- Populate from system_findings/kernel_fusion_findings.md if kernel_fusion category exists in manifest. -->
<!-- Each finding uses Insight / Action / Impact / Confidence format, with Impact from kernel_fusion_metrics.json. -->
<!-- P1/P2/P3+ ordered by confidence then kernel time. -->
<!-- Icon mapping by CONFIDENCE (not priority number): 🔴 high → 🟡 medium → 🟢 low. -->
<!-- If no findings or kernel_fusion category not in manifest, replace the cards below with: "No kernel fusion opportunities detected." -->

### 🔴 P1: <Candidate Name>

**Insight**: [1 sentence - what fusion pattern was detected]

**Action**: [1-2 sentences - which kernels to fuse and how]

<!-- impact-begin kind=p_item low=<kernel_fusion_metrics.impact_estimates[0].impact_score_low> mid=<kernel_fusion_metrics.impact_estimates[0].impact_score> high=<kernel_fusion_metrics.impact_estimates[0].impact_score_high> -->
**Impact**: [impact_score: X.X (perf-model coverage Y/Z kernels)]
<!-- impact-end -->

**Confidence**: [high / medium / low - fusion pattern quality]

→ *See [Detailed Analysis: Kernel fusion insights > P1](#detailed-analysis-fusion-P1) for details*

---

### 🟡 P2: <Candidate Name>

**Insight**: [1 sentence]

**Action**: [1-2 sentences]

<!-- impact-begin kind=p_item low=<kernel_fusion_metrics.impact_estimates[1].impact_score_low> mid=<kernel_fusion_metrics.impact_estimates[1].impact_score> high=<kernel_fusion_metrics.impact_estimates[1].impact_score_high> -->
**Impact**: [impact_score: X.X (perf-model coverage Y/Z kernels)]
<!-- impact-end -->

**Confidence**: [high / medium / low]

→ *See [Detailed Analysis: Kernel fusion insights > P2](#detailed-analysis-fusion-P2) for details*

---

### 🟢 P3: <Candidate Name>

**Insight**: [1 sentence]

**Action**: [1-2 sentences]

<!-- impact-begin kind=p_item low=<kernel_fusion_metrics.impact_estimates[2].impact_score_low> mid=<kernel_fusion_metrics.impact_estimates[2].impact_score> high=<kernel_fusion_metrics.impact_estimates[2].impact_score_high> -->
**Impact**: [impact_score: X.X (perf-model coverage Y/Z kernels)]
<!-- impact-end -->

**Confidence**: [high / medium / low]

→ *See [Detailed Analysis: Kernel fusion insights > P3](#detailed-analysis-fusion-P3) for details*

<!-- All additional fusion P-items (P4, P5, ...) follow the same pattern with Detailed Analysis links: → *See [Detailed Analysis: Kernel fusion insights > PN](#detailed-analysis-fusion-PN) for details* -->

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

<!-- === COMPARATIVE system-level note === -->
<!-- In comparative mode, add this note immediately after the blockquote above: -->
<!-- > **Comparative Note:** System-level analysis is performed on the primary trace (Trace 1) only. Cross-trace system-level comparison is not yet supported. -->

Findings from system-level analysis (GPU utilization, memory transfer patterns,
communication/compute overlap). These affect the GPU pipeline as a whole.

<!-- CONDITIONAL: If NO actionable system-level issues found (idle <= 15% and all multi-kernel assessments flagged: false), use Template A. -->
<!-- Otherwise, number priorities sequentially: CPU/Idle first (if idle > 15%), then multi-kernel issues by severity. -->
<!-- Icon mapping by PRIORITY NUMBER (not severity): P1=🔴, P2=🟡, P3+=🟢 -->
<!-- Title format: Descriptive name only. -->
<!-- System-level recommendations always include **Impact**: "Not quantifiable from trace data" with null markers. -->
<!-- De-dup rule: If CPU/Idle and Multi-Kernel propose the same mechanism/action, keep one merged system card with combined evidence (do not render two near-duplicate cards). -->

<!-- === TEMPLATE A: No actionable system-level issues === -->
<!-- Use this when idle <= 15% and all multi-kernel assessments have flagged: false -->

✅ No system-level bottlenecks detected. GPU activity breakdown shows X% computation, with negligible memcpy and communication overhead.

<!-- === TEMPLATE B: Actionable issues found === -->
<!-- Use this when idle > 15% or at least one multi-kernel assessment has flagged: true -->

### 🔴 P1: <CPU/Idle Title OR Multi-Kernel Issue Title>

**Insight**: [1-2 sentences - what's wrong]

**Action**: [1-2 sentences - what to do]

<!-- impact-begin kind=p_item low=null mid=null high=null -->
**Impact**: Not quantifiable from trace data
<!-- impact-end -->

→ *See [Detailed Analysis: System-level insights > P1](#detailed-analysis-system-p1) for details*

---

### 🟡 P2: <Multi-Kernel Issue Title>

**Insight**: [1 sentence - what's wrong]

**Action**: [1-2 sentences - what to do]

<!-- impact-begin kind=p_item low=null mid=null high=null -->
**Impact**: Not quantifiable from trace data
<!-- impact-end -->

→ *See [Detailed Analysis: System-level insights > P2](#detailed-analysis-system-p2) for details*

---

### 🟢 P3: <Next Multi-Kernel Issue>

**Insight**: [1 sentence]

**Action**: [1-2 sentences]

<!-- impact-begin kind=p_item low=null mid=null high=null -->
**Impact**: Not quantifiable from trace data
<!-- impact-end -->

→ *See [Detailed Analysis: System-level insights > P3](#detailed-analysis-system-p3) for details*

<!-- All additional system P-items follow the same pattern with Detailed Analysis links -->

---

## Detailed Analysis

<!-- Paste reasoning blocks from sub-agent findings, augment headings with P-numbers, icons, and HTML anchors. Everything else should be copied verbatim-->
<!-- Detailed Analysis labels per rule 9 — do not use these labels in optimization cards above -->
<!-- Impact estimate bullets are rendered by each sub-agent from metadata/*.json → impact_estimates (same source as card Impact). -->
<!-- MARKER CONTRACT: Every #### P<N>: heading in Detailed Analysis MUST be
     preceded by  <!-- reasoning-candidate tier=<TIER> rank=<R> --> where TIER = compute | fusion | system (matching the ### subsection), R= 1, 2, 3, … incrementing per tier (rank=1 for first item, rank=2 for second, etc.). -->

### Compute Kernel Insights

<!-- One #### 🔴/🟡/🟢 Pn: <title> block per entry in priority_data.findings[], in array order. -->
<!-- Source the body block from the sub-agent's findings.md by joining on (findings[i].category, findings[i].category_rank): the sub-agent emits its P-items ordered by intra-category rank, so its rank-N block becomes this report's PN where N matches the position in priority_data.findings[]. -->
<!-- Each block has an HTML anchor: <a id="detailed-analysis-compute-pN"></a> -->

<!-- === STANDALONE Compute Kernel Data table === Use this schema for standalone mode ONLY. Use these 10 exact columns (must match sub_agent_spec.md § Operations Table Schema) -->

<a id="detailed-analysis-compute-p1"></a>
<!-- reasoning-candidate tier=compute rank=1 -->
#### 🔴 P1: <Brief Title> (<Library>)
**Identification:**
**Data:**

| Operation | Args | Kernel Path | Kernel Name | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
|-----------|------|-------------|-------------|-----------|------|-------|------------|------------|-------|
| ...       | ...  | ...         | ...         | ...       | ...  | ...   | ...        | ...        | ...   |

**Reasoning for Slowdown:**
**Resolution:**
**Impact estimate:**

<!-- === COMPARATIVE Compute Kernel Data table === Use this schema for comparative mode ONLY. Use these 8 exact columns (Kernel Name/Path are omitted in comparative mode) -->
<!-- Trace 1 ms = operations[i].time_ms. Trace 2 ms = operations[i].t2_time_ms.
     Count T1/T2 = operations[i].count / operations[i].count_trace2 when present.
     Difference (ms) = operations[i].difference_ms (negative ⇒ Trace 1 slower), or —. -->

<a id="detailed-analysis-compute-p1"></a>
<!-- reasoning-candidate tier=compute rank=1 -->
#### 🔴 P1: <Brief Title>
**Identification:** [1-2 sentences - How this opportunity was surfaced relative to the target trace. Must end with (source: <artifact> → <keys>).]
**Data:** [1 sentence summary of table]

| Operation | Args (T1) | Trace 1 Time (ms) | Trace 2 Time (ms) | Count (T1/T2) | Difference (ms) | FLOPS/Byte (T1) | Bound (T1) |
|-----------|-----------|-------------------|-------------------|---------------|-----------------|-----------------|------------|
| ...       | ...       | ...               | ...               | .../...       | ...             | ...             | ...        |

**Reasoning for Slowdown:** [2-3 sentences - Why Trace 1 is slower than Trace 2 for these operations as the traces show. No micro-architecture speculation.]
**Resolution:** [1-2 sentences - Why the suggested optimization helps close the gap — not merely restating what to do.]
**Impact estimate:** [Rendered from metadata → impact_estimates]

<a id="detailed-analysis-compute-p2"></a>
<!-- reasoning-candidate tier=compute rank=2 -->
#### 🟡 P2: <Brief Title>
**Identification:**
**Data:**
**Reasoning for Slowdown:**
**Resolution:**
**Impact estimate:**

<a id="detailed-analysis-compute-p3"></a>
<!-- reasoning-candidate tier=compute rank=3 -->
#### 🟢 P3: <Brief Title>
**Identification:**
**Data:**
**Reasoning for Slowdown:**
**Resolution:**
**Impact estimate:**

### Kernel Fusion Insights
<!-- === STANDALONE Kernel Fusion === -->
> **Note:** Kernel fusion analysis is experimental. impact_score projections estimate the recoverable fraction of E2E with 85% memory/compute pipeline overlap. Kernels without perf models use their measured trace time as-is. Actual recoverable time depends on implementation feasibility and interaction effects.
<!-- === COMPARATIVE Kernel Fusion === -->
> **Note:** Kernel fusion analysis is experimental.

<!-- Paste reasoning blocks from kernel_fusion_findings.md, ordered by confidence then kernel time (matching card order). -->
<!-- Each block uses three required labels: **Identification:**, **Data:**, **Impact estimate:** -->
<!-- If kernel_fusion category is not in the manifest or findings are empty, show "No fusion impact estimates available." -->

<a id="detailed-analysis-fusion-P1"></a>
<!-- reasoning-candidate tier=fusion rank=1 -->
#### 🔴/🟡/🟢 P1: <Candidate Name> (<time_ms> ms, <instance_count> instances)

**Identification:**

**Data:**

| Kernel | Type | Duration (us) | Perf model |
|--------|------|--------------|------------|
| <kernel name (truncated to ~60 chars)> | <type> | X.X | Yes/No |

**Impact estimate:**

<a id="detailed-analysis-fusion-P2"></a>
<!-- reasoning-candidate tier=fusion rank=2 -->
#### 🔴/🟡/🟢 P2: <Candidate Name> (<time_ms> ms, <instance_count> instances)

**Identification:**

**Data:**

| Kernel | Type | Duration (us) | Perf model |
|--------|------|--------------|------------|
| <kernel name (truncated to ~60 chars)> | <type> | X.X | Yes/No |

**Impact estimate:**

<a id="detailed-analysis-fusion-P3"></a>
<!-- reasoning-candidate tier=fusion rank=3 -->
#### 🔴/🟡/🟢 P3: <Candidate Name> (<time_ms> ms, <instance_count> instances)

*Repeat the same Identification + Data + Impact estimate format for each candidate, with anchors `detailed-analysis-fusion-PN`.*

### System-Level Insights

<!-- One #### 🔴/🟡/🟢 Pn: <title> block per promoted system P-item, in priority order. -->
<!-- Each block has an HTML anchor: <a id="detailed-analysis-system-pN"></a> -->
<!-- System-level detailed analysis uses the same format for both standalone and comparative modes.
     In comparative mode, system-level analysis covers Trace 1 () only. -->

<a id="detailed-analysis-system-p1"></a>
<!-- reasoning-candidate tier=system rank=1 -->
#### 🔴 P1: <Brief Title>
**Identification:**
**Data:**
**Reasoning for Slowdown:**
**Resolution:**
**Impact estimate:**

<a id="detailed-analysis-system-p2"></a>
<!-- reasoning-candidate tier=system rank=2 -->
#### 🟡 P2: <Brief Title>
**Identification:**
**Data:**
**Reasoning for Slowdown:**
**Resolution:**
**Impact estimate:**

<a id="detailed-analysis-system-p3"></a>
<!-- reasoning-candidate tier=system rank=3 -->
#### 🟢 P3: <Brief Title>
**Identification:**
**Data:**
**Reasoning for Slowdown:**
**Resolution:**
**Impact estimate:**

---

## Appendix

### Model Architecture
- **Model**: <model>
- **Architecture**: <architecture>
- **Scale**: <scale>
- **Precision**: <precision>

### Hardware Reference
- **Platform**: <platform>
- **Peak HBM BW**: X TB/s
- **Peak MAF (BF16)**: Y TFLOPS
- **Peak MAF (FP8)**: Z TFLOPS (if supported)
- **Peak MAF (FP4)**: W TFLOPS (if supported)
