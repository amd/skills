<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens Eval Rubrics

Single-source reference for every eval check, its pass/fail criteria, tolerances,
root-cause categories, and where the implementation lives.

> **Canonical location:** `TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/EVAL_RUBRICS.md`
> Update this file whenever an eval is added, removed, or its criteria change.

---

## Overview

| Range | Type | Count | Engine |
|-------|------|:-----:|--------|
| workflow\_eval\_1–8 | Scripted (file/dir existence) | 8 | `workflow_scripted_evals.py` |
| workflow\_eval\_9 | Scripted (per-header) | 7 sub-indices | `workflow_scripted_evals.py` |
| workflow\_eval\_10 | Scripted (per-row + CSV cross-check) | 5 sub-indices | `workflow_scripted_evals.py` |
| workflow\_eval\_11 | Scripted (per-P-item) | Dynamic | `workflow_scripted_evals.py` |
| workflow\_eval\_12 | LLM (multi-dimensional scoring) | 1 | `skills/workflow-llm-eval/` |
| workflow\_eval\_13 | Scripted (per-category) | Dynamic | `workflow_scripted_evals.py` |
| quality\_eval\_1 | Scripted (CSV alignment) | 1 (standalone) / 2 (comparative) | `quality_scripted_evals.py` |
| quality\_eval\_2 | LLM (multi-dimensional scoring) | 1 | `skills/quality-llm-eval/` |
| quality\_eval\_3 | LLM (multi-dimensional scoring) | 1 | `skills/quality-llm-eval/` |
| marker\_eval\_1 | Scripted (structural) | 1 | `workflow_scripted_evals.py` |
| marker\_eval\_2 | Scripted (per-P-item) | Dynamic | `workflow_scripted_evals.py` |
| marker\_eval\_3 | Scripted (per-P-item) | Dynamic | `workflow_scripted_evals.py` |

**Total:** 15 scripted evals (expanding to ~30+ sub-indices) + 3 LLM evals = 18 logical evals.

---

## Pre-Check Gates

Before any eval runs, hard gates are checked in order. If any trips, **all**
workflow evals are set to FAIL with `root_cause=pipeline`.

| Gate | Condition | Implemented in |
|------|-----------|----------------|
| 1 | `output_dir` does not exist | `workflow_scripted_evals.py` |
| 2 | `analysis.md` missing or < 100 bytes | `workflow_scripted_evals.py` |
| 3 | `analysis.md` is garbled (> 50% non-ASCII) | `workflow_scripted_evals.py` |
| 4 | `analysis.md` contains raw JSON instead of markdown | `workflow_scripted_evals.py` |
| 5 | `output_dir` or `reference_dir` missing | `quality_scripted_evals.py` |
| 6 | `perf_report_csvs/` (standalone) or `perf_report_trace1_csvs/` (comparative) missing in generated or reference | `quality_scripted_evals.py` |

---

## Workflow Evals 1–8: File & Directory Existence

All binary PASS/FAIL. Root cause is always `pipeline`.

| Index | Summary | Pass Criteria | Key Details |
|-------|---------|---------------|-------------|
| `workflow_eval_1` | Directory structure created | `metadata/`, `category_data/`, `system_findings/`, `category_findings/` all exist | — |
| `workflow_eval_2` | Metadata files exist on disk | Every `metadata_file` listed in `category_manifest.json` exists on disk | Requires manifest |
| `workflow_eval_3` | Model info JSON exists and valid | `metadata/model_info.json` exists, is valid JSON, has keys `{model, architecture, scale, precision}`, none empty | — |
| `workflow_eval_4` | Unified perf report exists | `perf_report_csvs/unified_perf_summary.csv` exists (standalone) or `perf_report_trace1_csvs/unified_perf_summary.csv` (comparative) | — |
| `workflow_eval_5` | Tree data files exist on disk | Every `tree_data_file` listed in `category_manifest.json` exists on disk | Requires manifest |
| `workflow_eval_6` | Categorical findings .md files exist | For every category in manifest (except `cpu_idle` when idle ≤ 15%), the corresponding `*_findings.md` exists in the correct tier directory | — |
| `workflow_eval_7` | All findings correctly placed | No findings file is in the wrong tier directory (`system_findings/` vs `category_findings/`) | — |
| `workflow_eval_8` | Plot generated on disk | `perf_improvement.png` exists **OR** no kernel tuning recommendations exist (legitimately skipped) | Checks `plot_data.json` and `*_metrics.json` for zero recommendations |

---

## Workflow Eval 9: Report Template Rendering

**Type:** Scripted, per-header. **Root cause on fail:** `template`.

Checks `analysis.md` for required `##` section headers and a metrics table.

| Sub-index | Pass Criteria |
|-----------|---------------|
| `workflow_eval_9_executive_summary` | `## Executive Summary` header present (regex: `^## Executive Summary`) |
| `workflow_eval_9_compute` | `## Compute Kernel Optimizations` header present |
| `workflow_eval_9_system` | `## System-Level Optimizations` header present |
| `workflow_eval_9_detailed` | `## Detailed Analysis` header present |
| `workflow_eval_9_appendix` | `## Appendix` header present |
| `workflow_eval_9_metrics_table` | Executive Summary section contains at least one markdown table row (`\|.*\|`) |
| `workflow_eval_9_kernel_fusion` | `## Kernel Fusion Opportunities (Experimental)` header present |

---

## Workflow Eval 10: Executive Summary Metrics Table

**Type:** Scripted, per-row + CSV cross-check. **Root cause on fail:** `template`.

Parses the metrics table in `## Executive Summary` and optionally cross-checks numeric values against `gpu_timeline.csv`.

### Standalone (3-column table: Metric | Value)

Cross-checks against `perf_report_csvs/gpu_timeline.csv`.

| Sub-index | Row Labels (synonyms) | CSV Cross-Check | Tolerance |
|-----------|----------------------|-----------------|-----------|
| `workflow_eval_10_total_time` | "Total Compute Time", "Total Time" | — | — |
| `workflow_eval_10_compute_pct` | "Computation", "Compute %", "Compute" | `computation_time` percent | ±1.0% absolute |
| `workflow_eval_10_idle_pct` | "Idle Time", "Idle %", "Idle" | `idle_time` percent | ±1.0% absolute |
| `workflow_eval_10_comm_pct` | "Exposed Communication", "Exposed Communication %" | — | — |
| `workflow_eval_10_bottleneck` | "Top Bottleneck Category", "Top Bottleneck" | — | — |

### Comparative (4-column table: Metric | T1 | T2 | Difference)

Cross-checks T1 values against `perf_report_trace1_csvs/gpu_timeline.csv` and T2 values against `perf_report_trace2_csvs/gpu_timeline.csv`. Same 5 sub-indices as standalone apply, but fail if a T1 or T2 cell is empty for a row that has a CSV cross-check.

Additionally, for every numeric row: verifies `|reported_diff − (T1 − T2)| / max(|T1|, |T2|) ≤ 0.01`. Non-numeric rows (e.g., Top Bottleneck Category) are skipped.

**Pass:** Row found in table. If CSV cross-check applies, report value within tolerance of CSV value.
**Fail:** Row not found, or numeric value exceeds tolerance.

---

## Workflow Eval 11: Issue Template Rendering

**Type:** Scripted, per-P-item. **Root cause on fail:** `template`.

Finds every priority item (`### ...P{N}:` headers) and checks for required bold fields.

| Section | Required Fields | Example Sub-index |
|---------|----------------|-------------------|
| `## Compute Kernel Optimizations` | `**Insight**` or `**Issue**`, `**Action**`, `**Impact**` | `workflow_eval_11_compute_P1` |
| `## System-Level Optimizations` | `**Insight**` or `**Issue**`, `**Action**` (no Impact) | `workflow_eval_11_system_P1` |

**Pass:** All required bold fields present in the P-item block.
**Fail:** Any required field missing. Details list which fields are absent.

**Note:** Either `**Insight**` or `**Issue**` is acceptable — both are valid as the first field.

---

## Workflow Eval 12: Hardware Reference in Appendix

**Type:** LLM, multi-dimensional weighted scoring. **Root cause on fail:** `template`.

**Implementation:** `TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/skills/workflow-llm-eval/SKILL.md` (see `reference.md` for rubric detail)

Checks the `## Appendix` section for hardware reference values (only for trace1 if in comparative mode):
- Platform name (e.g., MI300X)
- Peak HBM BW value (e.g., 5.3 TB/s)
- At least one Peak MAF value (e.g., 708 TFLOPS)

### Scoring Dimensions

| Dimension | Weight | Scale |
|-----------|--------|-------|
| **correctness** | 50% | 10 = all values correct and plausible, 7 = present but imprecise, 0 = wrong/invented |
| **completeness** | 50% | 10 = all 3 items present, 7 = 2 of 3, 4 = 1 of 3, 0 = none |

### Pass/Fail

```
overall_score = correctness × 0.50 + completeness × 0.50
```

- **FAIL** if correctness = 0
- **FAIL** if overall\_score < 7.0
- **PASS** otherwise

**Details format:** `correctness=N/10 completeness=N/10 overall=N.N | <explanation>`

---

## Workflow Eval 13: Model Identification in Report

**Type:** Scripted, per-field. **Root cause on fail:** `template`.

Reads `metadata/model_info.json` and checks that each field value appears
(case-insensitive substring match) in the `## Appendix` section of the report.

| Sub-index | JSON Field | Pass Criteria |
|-----------|-----------|---------------|
| `workflow_eval_13_model` | `model` | Value appears in Appendix text |
| `workflow_eval_13_architecture` | `architecture` | Value appears in Appendix text |
| `workflow_eval_13_scale` | `scale` | Value appears in Appendix text |
| `workflow_eval_13_precision` | `precision` | Value appears in Appendix text |

**Special case:** If a field value is empty or `"Cannot be inferred from trace"`,
the check is **skipped (PASS)** with a note in details.

---

## Quality Eval 1: Perf Report CSV Alignment

**Type:** Scripted. **Root cause on fail:** `data`.

**Implementation:** `quality_scripted_evals.py`

Compares CSVs against the reference directory.

| Check | Criteria |
|-------|----------|
| File presence | Every reference CSV must exist in generated output |
| Column presence | All non-optional reference columns must exist (optional prefixes: `Pct Roofline`, `Roofline Time`) |
| Row count | Generated row count must match reference |
| Numeric columns | Relative diff ≤ 1% **AND** absolute diff ≤ 0.05 (both must exceed to fail) |
| String columns | Exact match after stripping numpy type wrappers (e.g., `np.int64(135)` → `135`) |

**Pass:** All reference CSVs match within tolerances.
**Fail:** Details list up to 5 mismatches.

---

## Quality Eval 2: Compute Issue Title Alignment

**Type:** LLM, multi-dimensional weighted scoring. **Root cause on fail:** `data`.

**Implementation:** `TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/skills/quality-llm-eval/SKILL.md` (see `reference.md` for rubric detail)

Compares P-item titles in the generated report against the reference report.
This is a **semantic** comparison — not a string match.

### Scoring Dimensions

| Dimension | Weight | Scale |
|-----------|--------|-------|
| **correctness** | 40% | 10 = same bottleneck identified (e.g., both say "GEMM low CU occupancy"), 7 = same category and root cause but framed differently (e.g., ref says "low CU occupancy due to small tiles" vs generated says "insufficient tile coverage on CUs"), 4 = same category but different root cause (e.g., ref says "occupancy" vs generated says "memory bandwidth"), 0 = wrong category or fabricated bottleneck |
| **completeness** | 30% | 10 = all reference P-items matched, 7 = one miss, 4 = several misses, 0 = most unmatched |
| **precision** | 30% | 10 = same priority level, 7 = off by 1, 4 = off by 2+, 0 = completely different |

### Pass/Fail

```
overall_score = correctness × 0.40 + completeness × 0.30 + precision × 0.30
```

- **FAIL** if correctness = 0
- **FAIL** if overall\_score < 7.0
- **PASS** otherwise

**Details format:** `correctness=N/10 completeness=N/10 precision=N/10 overall=N.N | <explanation>`

---

## Quality Eval 3: Compute Issue Content Alignment

**Type:** LLM, multi-dimensional weighted scoring. **Root cause on fail:** `data`.

**Implementation:** `TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/skills/quality-llm-eval/SKILL.md` (see `reference.md` for rubric detail)

For each matched compute P-item pair (from eval 2), compares content values.
**Only compares Compute Kernel P-items** — System-Level P-items are skipped entirely.

### Value Tolerances

| Value Type | Tolerance | Special Rules |
|-----------|-----------|---------------|
| Performance numbers (kernel time, efficiency %) | 2% relative | — |
| Shapes (matrix dimensions, batch sizes) | Exact match | — |
| Gap to roofline (efficiency %) | 2% relative | — |
| Estimated savings (ms) | 2% relative | If savings < 5 ms, accept ≤ 1 ms absolute diff |
| Non-numeric gains ("Not quantifiable") | Semantic match | Mismatch only if one side numeric, other not |
| Diff values — cross-trace deltas (comparative only) | 2% relative | e.g., "−12.3 ms" or "2.1× slower" |

### Scoring Dimensions

| Dimension | Weight | Scale |
|-----------|--------|-------|
| **correctness** | 40% | 10 = all values match, 7 = minor discrepancies, 0 = wrong values |
| **completeness** | 30% | 10 = all fields present (time, efficiency, shapes, gains), 7 = one missing, 0 = most missing |
| **precision** | 30% | See precision scale below |

**Precision scale** (based on the [Value Tolerances](#value-tolerances) above):

| Score | Meaning |
|-------|---------|
| 10 | All numeric values within 2% relative tolerance |
| 7 | Most values within 2%; 1–2 values between 2–5% off |
| 4 | Several values 2–5% off, or 1–2 values >5% off |
| 0 | Most values >5% off, or systematically wrong |

### Pass/Fail

```
overall_score = correctness × 0.40 + completeness × 0.30 + precision × 0.30
```

- **FAIL** if correctness = 0
- **FAIL** if overall\_score < 7.0
- **PASS** otherwise

**Details format:** `correctness=N/10 completeness=N/10 precision=N/10 overall=N.N | <explanation>`

---

## Marker Identification Evals

**Type:** Scripted (structural / deterministic). **Root cause on fail:** `template`.

**Implementation:** `TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/eval_utils/workflow_scripted_evals.py`

Validates the structural correctness of impact markers (`<!-- impact-begin ... -->` /
`<!-- impact-end -->`) in generated `analysis.md` files. These checks are purely
regex-based and do not validate numeric correctness of marker attribute values.

### marker\_eval\_1: Top Operations Markers

Checks `analysis.md` for:

| Check | Pass Criteria |
|-------|---------------|
| `kind=top_ops` wrapper | A `<!-- impact-begin kind=top_ops -->` / `<!-- impact-end -->` pair exists |
| Inline row markers | Each table row within the wrapper has a `<!-- top-ops-row low=... high=... -->` marker |
| Row marker attributes | Each `top-ops-row` marker contains both `low` and `high` attributes |

**Pass:** All structural checks satisfied.
**Fail:** Any marker missing or malformed. Details list specific issues.

### marker\_eval\_2: P-item Impact Markers

**Sub-indices:** `marker_eval_2_P{N}` (one per compute P-item)

For each P-item (`### ...P{N}:`) under `## Compute Kernel Optimizations`:

| Check | Pass Criteria |
|-------|---------------|
| `kind=p_item` marker present | A `<!-- impact-begin kind=p_item ... -->` / `<!-- impact-end -->` pair exists |
| Required attributes | Marker contains `category`, `low`, `mid`, `high` attributes |
| Pairing | Every `impact-begin` has a matching `impact-end` |

**Pass:** All structural checks satisfied for the P-item.
**Fail:** Marker missing, attributes missing, or unpaired. Details list specific issues.

### marker\_eval\_3: Detail Estimate Markers

**Sub-indices:** `marker_eval_3_P{N}` (one per compute P-item in Detailed Analysis)

For each compute P-item section under `## Detailed Analysis`:

| Check | Pass Criteria |
|-------|---------------|
| `kind=detail_estimate` marker or sentinel | Either a `<!-- impact-begin kind=detail_estimate ... -->` / `<!-- impact-end -->` pair exists, or the text "not quantifiable from trace data" is present |
| Required attributes | If marker present, it contains `low` and `high` attributes |
| Pairing | Every `impact-begin` has a matching `impact-end` |

**Pass:** Marker with correct attributes present, or not-quantifiable sentinel present.
**Fail:** Neither marker nor sentinel found, or marker has missing attributes. Details list specific issues.

---

## Stability Classification

After multiple repeated runs, the aggregation script (`aggregate_repeatability.py`)
classifies each (trace, eval\_index) pair:

| Classification | Criteria |
|----------------|----------|
| `STABLE_PASS` | 100% of runs passed |
| `FLAKY_PASS` | > 50% passed but not all |
| `FLAKY_FAIL` | > 0% passed but ≤ 50% |
| `STABLE_FAIL` | 0% of runs passed |

---

## Root Cause Categories

Every FAIL row includes a `root_cause` field for triage:

| Root Cause | Meaning | Typical Fix |
|-----------|---------|-------------|
| `pipeline` | Analysis pipeline didn't produce expected output | Re-run pipeline step or fix subagent |
| `template` | Report formatting / structure issue | Fix report assembly logic |
| `data` | Generated data doesn't match reference | Regenerate golden refs or fix data pipeline |

---

## CSV Output Schema

All eval results use the same 7-column schema:

```
index,category,issue_summary,result,details,root_cause,recommended_fix
```

| Column | Description |
|--------|-------------|
| `index` | Eval identifier (e.g., `workflow_eval_9_compute`, `quality_eval_2`) |
| `category` | `Workflow`, `Quality`, or `Marker Identification` |
| `issue_summary` | Human-readable name of the check |
| `result` | `PASS` or `FAIL` |
| `details` | Failure specifics, scoring breakdown, or match confirmation |
| `root_cause` | `pipeline`, `template`, `data`, or empty if PASS |
| `recommended_fix` | Actionable fix suggestion, or empty if PASS |
