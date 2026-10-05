<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Quality LLM eval — reference

Detailed rubrics and CSV output format for **quality-llm-eval** ([SKILL.md](SKILL.md)) — evals 2 and 3.

---

## Inputs

When triggered, the prompt will specify:
- **output_dir**: path to the generated `analysis_output/` directory
- **reference_dir**: path to the `analysis_output_ref/` directory
- **results_path**: path to write the results CSV
- **comparison_scope**: `standalone` (default) or `comparative`

## Files to Read

Read ALL of these before evaluating:

- `<output_dir>/analysis.md` (generated report)
- `<reference_dir>/analysis.md` (reference report)
- `<output_dir>/category_findings/*_findings.md` (generated findings)
- `<reference_dir>/category_findings/*_findings.md` (reference findings)

## Scoring Rubric

Each eval below uses **multi-dimensional weighted scoring** (adopted from the Gaia eval framework). Score each dimension on a 0–10 scale, compute a weighted overall, and apply the pass/fail threshold.

### Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **correctness** | 40% | Do the generated values factually match the reference? 10=exact match, 7=minor drift, 4=partial, 0=wrong/hallucinated |
| **completeness** | 30% | Are all reference items covered? 10=all present, 7=1 missing, 4=several missing, 0=most missing |
| **precision** | 30% | Are numeric values within tolerance bands? 10=all within tolerance, 7=1-2 slightly off, 4=several off, 0=systematically wrong |

### Pass/Fail Rules (apply in order)

1. **FAIL** if correctness = 0
2. **FAIL** if overall_score < 7.0
3. **PASS** otherwise

`overall_score = correctness * 0.40 + completeness * 0.30 + precision * 0.30`

## Evals

Evaluate BOTH checks below. Write BOTH rows to the results CSV.

### quality_eval_2: Compute Issue Title Alignment

**Category:** Quality
**Issue Summary:** Compute Issue Title Alignment

Compare the P-item titles in the generated `analysis.md` against the reference report. For each P-item in the reference (lines matching `### ... P1:`, `### ... P2:`, etc.):

- Check if the generated report identifies the **same bottleneck** at the same or similar priority level
- This is a **semantic** comparison, not a string match
- A P-item in the reference that has no semantic match in the generated report is a miss

**comparison_scope=comparative note:** P-items in a comparative report describe **deltas between traces**. Correctness means the same cross-trace delta or bottleneck is identified — not an absolute single-trace bottleneck. Apply the scoring guide with this context in mind.

**Scoring guide:**
- **correctness**: Do matched titles describe the same bottleneck? (10=same bottleneck e.g. both say "GEMM low CU occupancy", 7=same category and root cause but framed differently e.g. "low CU occupancy due to small tiles" vs "insufficient tile coverage on CUs", 4=same category but different root cause e.g. "occupancy" vs "memory bandwidth", 0=wrong category or fabricated bottleneck)
- **completeness**: What fraction of reference P-items have a match? (10=all matched, 7=one miss, 4=several misses, 0=most unmatched)
- **precision**: Are priority levels aligned? (10=same priority, 7=off by 1 level, 4=off by 2+, 0=completely different)

**PASS** if overall_score >= 7.0 and correctness >= 4. **FAIL** listing unmatched reference P-items and per-dimension scores.

### quality_eval_3: Compute Issue Content Alignment

**Category:** Quality
**Issue Summary:** Compute Issue Content Alignment

For each matched P-item pair (from eval 2), compare the content values. **Only compare Compute Kernel P-items** (under `## Compute Kernel Optimizations`). **Skip System-Level P-items** (under `## System-Level Optimizations`) entirely — system P-items have no `**Impact**` field and no numeric gain to compare.

For each matched compute P-item pair:

- **Performance numbers**: kernel time, efficiency percentage, achieved bandwidth/TFLOPS. Numeric tolerance: 2% relative difference.
- **Shapes**: matrix dimensions, batch sizes. Must match exactly.
- **Gap to roofline**: the efficiency percentage or fraction of peak. Tolerance: 2%.
- **Estimated gain**: savings in ms or speedup factor. Tolerance: 2% relative. **For estimated savings values < 5 ms, accept differences up to 1 ms absolute regardless of relative percentage**. When both reference and generated P-items have no numeric estimated gain (e.g., both say "Not quantifiable" or equivalent non-numeric text), treat as aligned. Flag a mismatch only when one side has a numeric gain and the other does not.
- **Diff values** (comparison_scope=comparative only): cross-trace deltas expressed as "−12.3 ms" or "2.1× slower". Treat these like performance numbers with 2% relative tolerance.

When comparing Impact/savings fields, accept format variants as equivalent: `**Estimated Savings**` tables, `**Impact** kernel_tuning` inline text, and `~X.X ms savings (pre-computed)` patterns all convey the same information. Compare the numeric values regardless of formatting. Do not flag a mismatch solely because the label or structure differs between reference and generated.

Check the category findings files for detailed values if the top-level report summarizes them.

**Scoring guide:**
- **correctness**: Do the content values factually align? (10=all values match, 7=minor discrepancies, 0=wrong values)
- **completeness**: Are all expected fields (time, efficiency, shapes, gains) present? (10=all present, 7=one field missing, 0=most missing)
- **precision**: Are numeric values within tolerance? (10=all within 5%, 7=most within 5% but 1-2 values 5-10% off, 4=several values 5-10% off or 1-2 values >10% off, 0=most values >10% off or systematically wrong)

**PASS** if overall_score >= 7.0 and correctness >= 4. **FAIL** listing specific mismatches with expected vs actual values and per-dimension scores.

## Output

Write a CSV to the specified `results_path` with exactly these 5 columns and 2 data rows:

`index,category,issue_summary,result,details`

Use `quality_eval_2` and `quality_eval_3` as the `index` values.

In the `details` column, include the scoring breakdown in the format:
`correctness=N/10 completeness=N/10 precision=N/10 overall=N.N | <explanation>`

**Replace all commas in the `details` explanation text with semicolons** to avoid breaking CSV parsing.
Do not add any other columns.
