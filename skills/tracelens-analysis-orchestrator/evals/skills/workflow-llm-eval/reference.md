<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Workflow LLM eval — reference

Detailed rubric and output format for **workflow-llm-eval** ([SKILL.md](SKILL.md)) — eval 12 (Appendix hardware reference) only.

---

## Inputs

When triggered, the prompt will specify:
- **output_dir**: path to the `analysis_output/` directory to evaluate
- **results_path**: path to write the results CSV
- **comparison_scope**: `standalone` (default) or `comparative`

## Files to Read

Read these before evaluating:

- `<output_dir>/analysis.md`

## Scoring Rubric

Eval 12 uses **multi-dimensional weighted scoring** (adopted from the Gaia eval framework).

### Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| **correctness** | 50% | Are the hardware values present and plausible? 10=all correct, 7=minor issue, 0=wrong/missing |
| **completeness** | 50% | Are all three items (platform, HBM BW, MAF) present? 10=all three, 7=two of three, 4=one, 0=none |

### Pass/Fail

`overall_score = correctness * 0.50 + completeness * 0.50`
- **FAIL** if correctness = 0
- **FAIL** if overall_score < 7.0
- **PASS** otherwise

## Eval

### workflow_eval_12: Hardware Reference in Appendix

**Category:** Workflow
**Issue Summary:** Hardware Reference in Appendix

In `analysis.md`, find the `## Appendix` section. Verify it contains:

- Platform name
- Peak HBM BW value
- At least one Peak MAF value

**Scoring guide (multi-dimensional):**
- **correctness**: Are the hardware values plausible and correctly stated? (10=correct, 7=present but imprecise, 0=wrong/invented)
- **completeness**: How many of the three items are present? (10=all three, 7=two of three, 4=one, 0=none)

**PASS** if overall_score >= 7.0 and correctness >= 4. **FAIL** with what's missing.

In the `details` column, include: `correctness=N/10 completeness=N/10 overall=N.N | <explanation>`

## Output

Write a CSV to the specified `results_path` with exactly these columns and **1** data row:

`index,category,issue_summary,result,details,root_cause,recommended_fix`

Use `workflow_eval_12` as the `index` value.

Include scoring breakdown in the `details` column:
`correctness=N/10 completeness=N/10 overall=N.N | <explanation>`

Set `root_cause` to `template` and `recommended_fix` to a specific fix suggestion if the result is FAIL. Leave both empty if PASS.

**Replace all commas in the `details` and `recommended_fix` text with semicolons** to avoid breaking CSV parsing.
