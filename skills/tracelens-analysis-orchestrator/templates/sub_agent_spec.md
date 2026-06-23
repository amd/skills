<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Sub-Agent Findings Specification

Canonical reference for the output that sub-agents write into their findings
files. The orchestrator extracts these sections when composing the final
`analysis.md` report.

> **Usage:** Link here from every `*-analyzer.md` instead of duplicating the
> schema. Replace `<category>` with the actual category name.
> This spec receives `comparison_scope`: `standalone` (default) or `comparative`.
---

## Orchestrator-consumed sections

Every findings file must end with these two sections, in this order:

1. `## Recommendations`
2. `## Detailed Analysis`

Applies to both tiers (compute → `category_findings/`, system → `system_findings/`). Agents may include any other sections (Overview, Operations Breakdown, Key Bottlenecks, …) before them — those are agent-internal and not parsed by the orchestrator.

---

## No actionable findings

**Compute tier:** There is no actionable bottleneck when the analyzer left
`category_data/<category>_metrics.json::category_findings` as an **empty array**
`[]`. In that case emit **empty** `## Recommendations` and **empty**
`## Detailed Analysis` exactly as in § Empty category_findings.

**System tier:** Follow the structured output your analyzer JSON supports.

---

## Recommendations

Each P-item maps 1:1 to a `## Detailed Analysis` reasoning candidate at the same rank.

```markdown
### P1: <Brief Title> (<Library>)            <!-- (<Library>) only on compute tier -->
**Insight**: [1 sentence — what's wrong]
**Action**: [1-2 sentences — what to do]
<!-- impact-begin kind=p_item low=<impact_score_low> mid=<impact_score> high=<impact_score_high> -->
**Impact**: [impact_score: X.X, OR "Not quantifiable from trace data"]
<!-- impact-end -->
```

- **Compute tier**: include all three fields. Pull `**Impact**` from `category_data/<category>_metrics.json::category_findings[i]`, ordered by `rank` (one card per entry).
- **System tier**: omit the `(<Library>)` title suffix. Always emit `**Impact**: Not quantifiable from trace data` wrapped in `kind=p_item` markers with `low=null mid=null high=null`.
- **Field labels are exact** — `**Insight**`, `**Action**`, `**Impact**`.
- **`(<Library>)` suffix**: the single `category_findings[i].library` for this card (one library per finding by construction). Omit the parenthetical when the value is `Unknown`.
- **Marker required** — see § Impact markers (REQUIRED). The `low`/`mid`/`high` attributes carry the raw `impact_score_low/impact_score/impact_score_high` values from `category_findings[i]`. For non-quantifiable cards (system tier) use `low=null mid=null high=null`.

---

## Detailed Analysis block schema

Each candidate block lives inside a `## Detailed Analysis` section. It starts
with an HTML comment and an `####` heading:

```markdown
## Detailed Analysis

<!-- reasoning-candidate tier=<compute|system> rank=<N> -->
#### <insight_title>
**Identification:** …

**Data:** …

**Reasoning for Slowdown:** …

**Resolution:** …

**Impact estimate:** …
```

### HTML comment fields

| Field | Values | Meaning |
|-------|--------|---------|
| `tier` | `compute` \| `system` | Must match the findings directory (`category_findings/` → compute, `system_findings/` → system). |
| `rank` | Integer ≥ 1 | Compute tier: `category_findings[i].rank`. System tier: agent-local priority within this file (1 = highest). |

### Required labels

The five labels below must appear **in this order**, each on its own line with a
blank line between them. The validator checks for these as substring matches.

| Label | Purpose |
|-------|---------|
| `**Identification:**` | Why these operations were flagged. Body text must be plain language — JSON keys, dotted paths, and internal variable names belong **only** in the closing `(source: \`artifact\` → \`keys\`)` parenthetical (artifact + keys backticked, e.g. `(source: \`<cat>_metrics.json\` → \`operations[].efficiency.efficiency_percent\` < 70)`). When any flagged op has a non-null `library` (e.g. `Tensile`, `CK`, `AITER`, `Triton`, `rocBLAS`), state the backend in prose and include `operations[].library` in the `(source:)` parenthetical. When `operations[i].module_chain` is non-empty, name the model layer the ops belong to. When `operations[i].call_chain` is present, use it for deeper context. |
| `**Data:**` | **Compute** (`tier=compute`): exactly one trace-grounded kernel breakdown table (see § Operations Table Schema). **All columns in the schema are mandatory — never drop a column.** Use `—` for any individual cell whose value is missing or null. **System** (`tier=system`): **must not** include kernel breakdown tables. include metric table (see § Metric Table Schema). |
| `**Reasoning for Slowdown:**` | Why the workload is slow *as the trace shows*: **Standalone:** low % of roofline, low arithmetic intensity, unfused patterns, etc. **Comparative:** how Trace 1 is slower than Trace 2 for these operations — express speed differences as "X% faster" or "X% slower", plus absolute time gaps. Never use raw efficiency ratios or `efficiency_percent` values in prose. **Forbidden:** micro-architecture speculation (bank conflicts, L1 miss rates, etc.). |
| `**Resolution:**` | **Why** the suggested optimization helps — not merely restating *what* to do. Must align with the P-item **Action** on the card. **Forbidden tautologies:** Do not restate the roofline definition (e.g. "raising bandwidth toward the roofline reduces kernel time"). Instead, explain the **mechanism** (e.g. "fusion eliminates the intermediate write-back, cutting bytes moved per invocation in half"). If the mechanism is not inferable from the trace, state only the action. |
| `**Impact estimate:**` | Compute tier: rendered from `category_findings[i]` (matched by `rank`), two-bullet low/high `impact_score` format (see § Impact estimate rendering). System tier: `Impact estimate is not quantifiable from trace data.` |

### Sentence quality

- Each sentence should convey **one main idea**. Do not chain independent
  observations with em-dashes, semicolons, or "while" bridges. Avoid run-on
  sentences.

### Trace observability (compute tier)

This is the single source of truth for what compute-tier sub-agents can and
cannot infer from a kernel-level PyTorch trace. Ground every claim in
**Reasoning for Slowdown** / **Resolution** in a **CAN Infer** row; for any
property in the **CANNOT Infer** rows, use the listed fallback prose instead
of speculating.

#### CAN Infer (universal — all compute categories)

| Observable | Source |
|------------|--------|
| Kernel names | `trunc_kernel_details` column |
| Kernel durations | Trace events |
| Achieved TFLOPS/s or TB/s | Calculated from duration + FLOPs/bytes |
| Efficiency % vs roofline | Achieved / resolved peak (MAF or HBM BW) |
| Invocation counts | Number of trace events per signature |
| Library / backend | `library` column / kernel-name heuristics |
| Bound type | `efficiency.bound_type` (compute / memory) |
| Input shape dimensions | `Input Dims` column (semantics per category — e.g. M/N/K for GEMM, B/H/S/D for SDPA, expert/token counts for MoE, NCHW for convolution) |

#### CANNOT Infer (universal — all compute categories)

These require hardware counters or profiler tools, not a trace.

| NOT Observable | Why | Fallback prose |
|----------------|-----|----------------|
| Bank conflicts | Requires hardware counters | "Low efficiency — profile with hardware counters to diagnose." |
| Cache hit rates | Requires hardware counters | "Large working set may exceed cache." |
| Wave / SM occupancy | Requires hardware counters | "Kernel running slower than expected — profile occupancy with hardware counters." |
| Shared-memory / LDS usage | Requires hardware counters | "Shared-memory usage not visible — profile with hardware counters." |
| Intra-warp shuffle efficiency | Requires hardware counters | "Warp-shuffle efficiency not visible — profile with hardware counters." |
| Root causes generally | Traces show WHAT, not WHY | "Bottleneck identified — generate reproducer for kernel team." |

#### CANNOT Infer (category-specific)

Each analyzer owns its own category-specific blind spots under a
`## Trace observability (category-specific)` section in its `*-analyzer.md` file.
The universal rows above always apply on top of those.

---

## Operations Table Schema (compute tier)

Standard column schema for operations breakdown tables and the `**Data:**` table
inside `## Detailed Analysis` blocks.

### Standalone (`comparison_scope` = `standalone`)

```markdown
| Operation |  Args  |            Kernel Path                  | Kernel Name | Time (ms) | %E2E | Count |FLOPS/Byte| Efficiency | Bound |
|-----------|--------|-----------------------------------------|-------------|-----------|------|-------|----------|------------|-------|
```

**All ten columns above are mandatory.** Never drop a column because some or all of its values are missing — render `—` in any cell whose value is null/absent and keep the column. The header row of every `**Data:**` table must contain exactly these ten column names in this order. (Agents may append extra columns at the end when needed, e.g. `Sub-Category` in the generic-op analyzer, but must not remove or reorder the ten standard columns.)

**Column mappings** (source: `metrics['operations']`):
- **Operation**: `operations[i].name`. Bare op name only — shape/dtype go in Args. Allowed suffix: `(decode)`/`(prefill)` to disambiguate the same op at multiple shapes.
- **Args**: `operations[i].args`. Pre-rendered shape/dtype string, already joined with `<br>` — paste verbatim, do not reformat or re-join. `—` when absent.
- **Kernel Path**: `operations[i].launcher_path`. Relative Python path that launched the kernel (e.g. `sglang/srt/layers/quantization/fp8_utils.py(549): aiter_w8a8_block_fp8_linear`). **Copy the value exactly as-is — do NOT truncate, shorten, or extract just the function name.** `—` when absent.
- **Kernel Name**: `operations[i].kernel_name_trunc`. Truncated GPU kernel name(s) launched by this operation. For multi-kernel ops, formatted as `Kernel 1: <name><br>Kernel 2: <name>`. **Copy the value exactly as-is.** `—` when absent. (The full untruncated name is available in `operations[i].kernel_name` if needed for identification.)
- **Time (ms)**: `operations[i].time_ms` — kernel time in milliseconds.
- **%E2E**: `operations[i].percent_of_total` — kernel time as % of E2E GPU time. `—` when null. (`percent_of_category` is still in the JSON for screening thresholds but no longer rendered.)
- **Count**: `operations[i].count` — total invocations, not unique signatures. `—` when absent.
- **FLOPS/Byte**: `operations[i].efficiency.flops_per_byte` — note the nested path under `efficiency`, NOT a top-level field. `—` when null.
- **Efficiency**: `operations[i].efficiency.efficiency_percent`, formatted by `bound_type`:
  - `compute-bound`: `X.XX% of Y TFLOPS` (Y = `resolved_peak_maf`)
  - `memory-bound`: `X.XX% of Y TB/s` (Y = `resolved_peak_hbm_bw`)
- **Bound**: `operations[i].efficiency.bound_type` + `-bound` suffix (e.g., `memory-bound`). Must reflect compute/memory bound type — never use `classification.gemm_type` or similar.

### Comparative (`comparison_scope` = `comparative`)

```markdown
| Operation | Args (T1) | Trace 1 Time (ms) | Trace 2 Time (ms) | Count (T1/T2) | Difference (ms) | FLOPS/Byte (T1) | Bound (T1) |
|-----------|-----------|-------------------|-------------------|---------------|-----------------|-----------------|------------|
```

**Column mappings** (all sourced from `metrics['operations']`; do **not** re-join the CSV):
- **Operation**: `operations[i].name`. Bare op name only.
- **Args (T1)**: `operations[i].args`. Pre-rendered shape/dtype string, already joined with `<br>` — paste verbatim. `—` when absent.
- **Trace 1 Time (ms)**: `operations[i].time_ms`
- **Trace 2 Time (ms)**: `operations[i].t2_time_ms`. `—` when absent.
- **Count (T1/T2)**: T1 = `operations[i].count`; T2 = `operations[i].count_trace2`. Format `T1 / T2` (use `—` for missing T2).
- **Difference (ms)**: `operations[i].difference_ms`. `—` when absent.
- **FLOPS/Byte (T1)**: `operations[i].efficiency.flops_per_byte`
- **Bound (T1)**: `operations[i].efficiency.bound_type` with a `-bound` suffix

Agents may add extra columns when needed (e.g. `Sub-Category` in the generic-op analyzer).

---

## Metric Table Schema (system tier)

Standard schema for the `**Data:**` table inside system-tier `## Detailed Analysis` blocks. In comparative mode, report Trace 1 metrics only — do not add Trace 2 columns or comparisons.

```markdown
| Metric | Value | Flagged |
|--------|-------|---------|
```

**Column rules:**
- **Metric**: Copy metric label directly from earlier findings sections — do not rename or reformat.
- **Value**: `X.X ms` or `X.X%` or `X.X ms (X.X%)`
- **Flagged**: `true` when the metric's threshold is exceeded (issue present); `false` otherwise.

---

## Peak Reference (compute tier)

When citing peak performance for a bottleneck, select the correct peak based on
`operations[i].efficiency.bound_type`:
- **compute-bound**: Use `operations[i].efficiency.resolved_peak_maf` (TFLOPS).
  Report achieved TFLOPS/s vs peak TFLOPS.
- **memory-bound**: Use `operations[i].efficiency.resolved_peak_hbm_bw` (TB/s).
  Report achieved TB/s vs peak TB/s.

Do not look up peaks independently from the metadata dict.

---

## Impact estimate rendering

Compute-tier sub-agents READ their P-items from `category_data/<category>_metrics.json::category_findings[]`, one card per entry ordered by `rank`. The analyzer script has already grouped per-op estimates by `(bound_type, library, eff_bucket)` in standalone mode (or `(bound_type, library)` in comparative mode), summed impact, and dropped sub-threshold groups; the sub-agent renders one card per surviving entry.

The set of P-items is decided by `category_findings[]` alone — `MIN_PITEM_IMPACT_SCORE` already gated upstream. **Per-category efficiency tables, expected-band thresholds, and Common Patterns in analyzer files are interpretation context for the prose** (cite in **Reasoning for Slowdown** when a member matches the band/symptom); they MUST NOT be used to add or drop P-items.

### Reading category_findings[]

| Field | Use |
|-------|-----|
| `rank` | Card order within your category (1 = highest impact). Also the `rank=` value in `<!-- reasoning-candidate -->`. |
| `bound_type` | `compute` \| `memory`. Selects the matching Action Prose Guidance row. |
| `library` | One per finding. Drives the `(<Library>)` title suffix. |
| `eff_bucket` | Roofline-efficiency band: `"0-30"`, `"30-60"`, `"60-100"`, or `"unknown"` (standalone); `"all"` (comparative). Members within a finding share the same band. |
| `impact_score` / `_low` / `_high` | Group-summed % of E2E. Render verbatim into `kind=p_item` and `kind=detail_estimate` markers. |
| `estimate_method` | `"quantified"` (impact from a perf model — standalone roofline gap or comparative t2/t1 ratio) or `"heuristic"` (op has no perf model; impact estimated from E2E share — see below). |
| `percent_of_total` | Heuristic findings only: the op's combined E2E GPU-time share (summed across shapes). Drives the warning line in § Heuristic findings. Absent on quantified findings. |
| `member_count`, `members[]` | Underlying per-op estimate rows (operation, time_ms, efficiency_pct, `type`, …) — rows of the `**Data:**` table. `members[].type == "unmodeled_significant"` marks a heuristic finding; `"kernel_tuning"` is a quantified (perf-modeled) finding. |

### Empty category_findings

If `category_findings[]` is empty, emit `## Recommendations` with no P-items
and `## Detailed Analysis` with no candidates. Do not manufacture sub-threshold
cards to fill the section — that is the honest "no actionable issues" answer.

### Heuristic findings (`estimate_method == "heuristic"`)

An op with no perf model: `impact_score` / `_low` / `_high` are estimated from a
recoverable fraction of its E2E share (`percent_of_total`, summed across shapes)
and ranked alongside quantified findings.

Render like a normal P-item card (numeric `low`/`mid`/`high` on `kind=p_item`,
normal `kind=detail_estimate` bullets) with two additions:
- Immediately **after** the `kind=p_item` impact marker block (i.e. *outside* the `impact-begin`/`impact-end` markers), add a warning line, substituting the finding's `percent_of_total`:

```markdown
> **Warning — estimated:** No performance model for this op; impact is derived from its E2E GPU-time share (<percent_of_total>%), not a gap projection.
```
- In the `## Detailed Analysis` `**Data:**` row, render `—` for `Efficiency` and `Bound`.

### Rendering in `## Detailed Analysis` (compute tier)

Two bullets — low and high. Wrap in `kind=detail_estimate` markers (see
§ Impact markers).

```markdown
<!-- impact-begin kind=detail_estimate low=<impact_score_low> high=<impact_score_high> -->
- Low end impact_score: <impact_score_low>
- High end impact_score: <impact_score_high>
<!-- impact-end -->
```

## Impact markers (REQUIRED)

Every block whose contents depend on `impact_score*` values must be wrapped in
a paired HTML-comment marker. The markers carry the underlying numeric data as
key=value attributes so that optional downstream tooling can re-process the
block deterministically without re-parsing prose.

### Marker shape

```
<!-- impact-begin kind=KIND attr1=value1 attr2=value2 -->
...rendered markdown content for this block...
<!-- impact-end -->
```

The block between them is exactly the `impact_score`-based markdown you would otherwise emit.

### `kind` values you must emit

| `kind` | Where | Required attributes | Optional attributes |
|--------|-------|--------------------|---------------------|
| `p_item` | Around every P-item `**Impact**` line in `## Recommendations`. | `low`, `mid`, `high` (all three; use `null` only for system-tier non-quantifiable). | `category` is reserved for the orchestrator template; sub-agents do **not** emit it. |
| `detail_estimate` | Around the two-bullet `Low end ... / High end ...` block under `**Impact estimate:**` in each `## Detailed Analysis` candidate. Skip only for system-tier non-quantifiable estimates. | `low`, `high` (impact_score values, % of E2E). | none |

### Value-source rule

The numbers in marker attributes (`low`, `mid`, `high`) **must** be transcribed
verbatim from `category_data/<category>_metrics.json::category_findings[i]`
(`impact_score_low`, `impact_score`, `impact_score_high` respectively), matched
by `rank`. Do not re-derive, round, or scale them. Do not pull them from any
other source.

---

## Validate findings (required before returning)

After writing the findings file and impact estimates, run the programmatic
validator. This replaces the previous manual self-check. The validator also
enforces marker structure (pairing, `kind=`, per-kind required attrs,
mandatory `kind=p_item` for category/system findings unless exempt) per
§ Impact markers above.

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
" '<output_dir>/<subdir>/<category>_findings.md' '<tier>' '<comparison_scope>'
```

Where `<tier>` is `compute` or `system`, `<subdir>` is `category_findings`
or `system_findings` respectively, and `<comparison_scope>` is `standalone` or
`comparative`.

**If validation fails (exit code 1):**

1. Read the FAIL output — error messages are self-explanatory and include the fix hint.
2. Fix the findings file accordingly. Edit sections in place and not regenerate the entire output.
3. When fixing table cells (Args, Kernel Path, or any multi-row issue), **re-emit the entire table in one edit** — do NOT patch rows individually. Batch together edits together.
4. Re-run validation. Maximum 2 retry attempts; if still failing, return with a warning.
