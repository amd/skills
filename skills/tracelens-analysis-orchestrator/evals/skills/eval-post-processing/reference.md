<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Eval post processing — reference

Full specification for the **eval-post-processing** skill ([SKILL.md](SKILL.md)). Follow for key=value inputs, aggregation, classification, report templates, reproducer tarballs, and `eval_reports/latest/` copy.

---

## Inputs

The prompt provides these key=value parameters:

- **results_root**: path to the repeatability results tree (contains `<trace_id>/run_<n>/` directories)
- **suite**: `unit` or `e2e`
- **test_traces_csv**: path to the trace CSV used (e.g. `TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/analysis_tests/combined_traces_standalone.csv`)
- **report_dir**: where to write output reports
- **container**: Docker container name (used in reproducer commands)

## Step 4 — Aggregate

Run `aggregate_repeatability.py` to merge all per-run `eval_summary.csv` files and parse stream logs:

```bash
RESULTS_ROOT=<results_root> OUTPUT_DIR=<report_dir>/aggregates \
  python3 TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/eval_utils/aggregate_repeatability.py
```

This produces six files in `<report_dir>/aggregates/`:

- `aggregated_results.csv` — columns: `trace_id,run_id,eval_index,eval_category,issue_summary,result,details,root_cause,recommended_fix`
- `pass_rate_summary.csv` — columns: `trace_id,<eval_index>,...,overall_pass_rate` (values like `13/15` or `N/A`)
- `stability_summary.csv` — columns: `trace_id,<eval_index>,...` (values: `STABLE_PASS`, `FLAKY_PASS`, `FLAKY_FAIL`, `STABLE_FAIL`, or `N/A`)
- `stream_diagnostics.csv` — columns: `trace_id,run_id,outcome,duration_ms,input_tokens,output_tokens,cache_read_tokens,turns,tool_calls,report_written,report_headers,last_step_reached`
- `run_level_summary.csv` — columns: `trace_id,run_id,pass,fail,total,is_catastrophic` — per-run failure counts; `is_catastrophic=True` when >50% of checks fail (indicates agent crash / no `analysis.md`)
- `failure_nature_summary.csv` — columns: `catastrophic_pipeline,stable,flaky,total` — classifies all failures by nature

Verify the script exits 0 and all six files exist before proceeding.

## Step 5 — Read and Classify

Read these files:

1. `<report_dir>/aggregates/aggregated_results.csv`
2. `<report_dir>/aggregates/pass_rate_summary.csv`
3. `<report_dir>/aggregates/stability_summary.csv`
4. `<report_dir>/aggregates/stream_diagnostics.csv`
5. `<report_dir>/aggregates/run_level_summary.csv`
6. `<report_dir>/aggregates/failure_nature_summary.csv`
7. `TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/eval_utils/report_section_rules.yaml` — classification guide (YAML format, load with `yaml.safe_load()`)
8. `<test_traces_csv>` — for trace metadata (id, sub_category, platform, trace_path)

Note: `aggregated_results.csv` now includes `root_cause` and `recommended_fix` columns from scripted evals. Use these when available instead of re-classifying failure modes from scratch. The `stability_summary.csv` classifies each (trace, eval) pair as `STABLE_PASS`, `FLAKY_PASS`, `FLAKY_FAIL`, or `STABLE_FAIL`.

### Splitting Unit Test vs E2E Test Cases

Use the `sub_category` column from `<test_traces_csv>` to classify each trace:

- **Unit test cases**: `sub_category` is NOT `full_model` (e.g., gemm, conv, attention, elementwise, moe)
- **E2E test cases**: `sub_category` is `full_model`

All subsequent metrics, tables, and failure analysis must be computed and reported **separately** for unit tests and e2e tests.

From `aggregated_results.csv`, compute:

- **Overall metrics**: count PASS, FAIL, MISSING rows; compute pass rate
- **Unit test metrics**: same counts for unit test trace_ids only
- **E2E test metrics**: same counts for e2e test trace_ids only
- **Top failure issues**: group FAIL rows by `issue_summary`, count, sort descending — compute overall AND per-split
- **Per-trace failure counts**: group FAIL rows by `trace_id`, count, sort descending — compute per-split

From `stream_diagnostics.csv`, compute per-split:

- **Success rate**: successful runs / total runs
- **Average duration**: mean of `duration_ms` for successful runs

### Classification

Use `report_section_rules.yaml` as a guide to classify each FAIL row into:

- **Base sections** (broad categories like "Reasoning", "Kernel Fusion", "Others"): match `eval_index`, `eval_category`, `issue_summary`, or `details` against `base_section_rules`. Unmatched rows go to "Others".
- **Standalone sections** (report sections like "Detailed Analysis", "Executive Summary"): match against `standalone_section_rules`. Unmatched rows go to "Detailed Analysis".
- **Failure modes** (likely cause + suggested fix): match against `failure_mode_rules`. You may improve on the suggested causes and fixes using your own judgment based on the failure details — the rules file is a starting point, not a rigid contract.

## Step 6 — Write Reports

Write two markdown files. Follow the exact structure below.

### `<report_dir>/pr_report.md`

```markdown
# Automated Eval Report (PR)

Generated at: `<ISO 8601 timestamp>`

| Metric | Overall | Unit Tests | E2E Tests |
|---|---|---|---|
| PASS | <overall_pass> | <unit_pass> | <e2e_pass> |
| FAIL | <overall_fail> | <unit_fail> | <e2e_fail> |
| MISSING | <overall_missing> | <unit_missing> | <e2e_missing> |
| Pass rate | <overall_rate>% | <unit_rate>% | <e2e_rate>% |

## Failure Nature Summary

| Nature | Count | % of Failures | Description |
|---|---|---|---|
| Catastrophic pipeline | <count> | <pct>% | Agent crashed — entire run fails |
| Stable | <count> | <pct>% | Consistent failures — real bugs |
| Flaky | <count> | <pct>% | Intermittent — agent non-determinism |

Catastrophic runs: <N> (list trace_id/run_id pairs if any, otherwise "None")

## Failure Sections

| Section | Failures |
|---|---|
| <base_section> | <count> |
...

## Top Failure Issues (Overall)

| Issue | Count |
|---|---|
| <issue_summary> | <count> |
...top 10, sorted descending

---

## Unit Test Cases (<N> cases, <unit_rate>% pass rate)

### Per-Case Results

| Case | Category | Platform | PASS | FAIL | MISSING | Pass Rate | Runs | Avg Duration |
|---|---|---|---|---|---|---|---|---|
| <trace_id> | <sub_category> | <platform> | <pass> | <fail> | <missing> | <pass_rate> | <success>/<total> | <avg_dur>s |
...sorted by FAIL descending (worst first)

### Top Failures (Unit Tests)

| Issue | Count |
|---|---|
| <issue_summary> | <count> |
...top 10 for unit test cases only

---

## E2E Test Cases (<N> cases, <e2e_rate>% pass rate)

### Per-Case Results

| Case | Category | Platform | PASS | FAIL | MISSING | Pass Rate | Runs | Avg Duration |
|---|---|---|---|---|---|---|---|---|
| <trace_id> | <sub_category> | <platform> | <pass> | <fail> | <missing> | <pass_rate> | <success>/<total> | <avg_dur>s |
...sorted by FAIL descending (worst first)

### Top Failures (E2E Tests)

| Issue | Count |
|---|---|
| <issue_summary> | <count> |
...top 10 for e2e test cases only
```

Column definitions for the Per-Case Results table:
- **Category**: the `sub_category` from the test traces CSV (e.g., gemm, conv, attention, moe, full_model)
- **Pass Rate**: format as `<pass>/<total> (<pct>%)` matching `pass_rate_summary.csv`
- **Runs**: `<successful_runs>/<total_runs>` from `stream_diagnostics.csv`
- **Avg Duration**: mean of `duration_ms` (in seconds) for successful runs from `stream_diagnostics.csv`

### `<report_dir>/fix_ticket_report.md`

```markdown
# Automated Eval Report (Fix Ticket)

Generated at: `<ISO 8601 timestamp>`

| Metric | Overall | Unit Tests | E2E Tests |
|---|---|---|---|
| PASS | <overall_pass> | <unit_pass> | <e2e_pass> |
| FAIL | <overall_fail> | <unit_fail> | <e2e_fail> |
| MISSING | <overall_missing> | <unit_missing> | <e2e_missing> |
| Pass rate | <overall_rate>% | <unit_rate>% | <e2e_rate>% |

## Failure Nature Analysis

Classifies all failures by their nature using `run_level_summary.csv` and `failure_nature_summary.csv`:

| Nature | Count | % of Failures | Description |
|---|---|---|---|
| Catastrophic pipeline | <count> | <pct>% | Agent crashed / no analysis.md — entire run fails (>50% of checks) |
| Stable | <count> | <pct>% | Eval fails consistently in every run for a given trace — real bug |
| Flaky | <count> | <pct>% | Eval fails in some runs but not others — agent non-determinism |

### Catastrophic Runs

List runs from `run_level_summary.csv` where `is_catastrophic=True`. If none, write "No catastrophic runs detected."

| Case | Run | Pass | Fail | Total |
|---|---|---|---|---|
| <trace_id> | run_<run_id> | <pass> | <fail> | <total> |
...only catastrophic runs

### Per-Case Run Pattern

Show the fail count per run for each case, sorted by total failures descending. Use `run_level_summary.csv`. Flag catastrophic runs with (**crash**).

| Case | run_0 | run_1 | run_2 | run_3 | run_4 | Total Fails | Pattern |
|---|---|---|---|---|---|---|---|
| <trace_id> | <fail>/<total> | <fail>/<total> | ... | ... | ... | <sum_fails> | <stable/flaky/catastrophic> |
...all cases, sorted by Total Fails descending

Pattern classification per case:
- **catastrophic**: at least one run has >50% failure rate
- **stable**: same number of failures in every run (±1)
- **flaky**: varying failure counts but no catastrophic run

---

## Unit Test Cases (<N> cases, <unit_rate>% pass rate)

### Per-Case Results

| Case | Category | Platform | PASS | FAIL | MISSING | Pass Rate | Runs | Avg Duration |
|---|---|---|---|---|---|---|---|---|
| <trace_id> | <sub_category> | <platform> | <pass> | <fail> | <missing> | <pass_rate> | <success>/<total> | <avg_dur>s |
...sorted by FAIL descending

### Top Failures (Unit Tests)

| Issue | Count |
|---|---|
| <issue_summary> | <count> |
...all issues for unit test cases, sorted descending

### Failure Modes (Unit Tests)

| Issue | Count | Likely cause | Suggested fix |
|---|---|---|---|
| <issue_summary> | <count> | <cause> | <fix> |
...top 8 unit test issues with cause/fix analysis

### Top Reproducers (Unit Tests)

| Trace/Case | Failures | Platform | Reproducer command |
|---|---|---|---|
| <trace_id> | <count> | <platform> | `CONTAINER=<container> TEST_IDS="<trace_id>" TEST_TRACES_CSV="<test_traces_csv_relative>" bash TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/eval_scripts/run_repeatability_parallel.sh` |
...top 5 unit test traces by failure count

---

## E2E Test Cases (<N> cases, <e2e_rate>% pass rate)

### Per-Case Results

| Case | Category | Platform | PASS | FAIL | MISSING | Pass Rate | Runs | Avg Duration |
|---|---|---|---|---|---|---|---|---|
| <trace_id> | <sub_category> | <platform> | <pass> | <fail> | <missing> | <pass_rate> | <success>/<total> | <avg_dur>s |
...sorted by FAIL descending

### Top Failures (E2E Tests)

| Issue | Count |
|---|---|
| <issue_summary> | <count> |
...all issues for e2e test cases, sorted descending

### Failure Modes (E2E Tests)

| Issue | Count | Likely cause | Suggested fix |
|---|---|---|---|
| <issue_summary> | <count> | <cause> | <fix> |
...top 8 e2e test issues with cause/fix analysis

### Top Reproducers (E2E Tests)

| Trace/Case | Failures | Platform | Reproducer command |
|---|---|---|---|
| <trace_id> | <count> | <platform> | `CONTAINER=<container> TEST_IDS="<trace_id>" TEST_TRACES_CSV="<test_traces_csv_relative>" bash TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/eval_scripts/run_repeatability_parallel.sh` |
...top 5 e2e test traces by failure count
```

For the reproducer commands:
- Use the `container` value from inputs
- Use the **relative** path of `test_traces_csv` (e.g. `TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/analysis_tests/combined_traces_standalone.csv`) — never embed absolute or user-specific paths
- Omit `NUM_REPEATS` and `MAX_PARALLEL` so the script defaults (5 repeats, 5 parallel) are used, matching a standard eval run
- The `platform` comes from the test traces CSV

### Handling single-split runs

If the test traces CSV contains **only unit tests** or **only e2e tests** (no `full_model` entries, or all `full_model` entries), still use the two-section format but note the empty section:

```markdown
## E2E Test Cases (0 cases)

No e2e test cases in this run.
```

## Step 7 — Build Reproducer Packages

Create a self-contained, assignable reproducer folder for each unique failure issue. These let a developer load the stream JSON into Cursor and debug the issue directly.

**For each unique `issue_summary` in the FAIL rows:**

1. Create a folder: `<report_dir>/reproducers/<sanitized_issue_name>/`
   - Sanitize the name for filesystem use (lowercase, replace spaces/special chars with underscores, truncate to 80 chars)

2. Write a `README.md` inside the folder with:
   - Issue name and total failure count
   - Table of affected traces (trace_id, run_id, platform, details snippet)
   - The reproducer shell command for the worst-affected trace

3. For each (trace_id, run_id) pair that hit this issue (limit to 3 representative examples):
   - Copy the `analysis_stream.ndjson` from `<results_root>/<trace_id>/run_<run_id>/analysis_stream.ndjson` into the folder as `<trace_id>_run_<run_id>.ndjson`
   - **Sanitize identifying paths**: replace the absolute repo root path with `$REPO_ROOT/` so the stream is portable
   - Copy the `eval_summary.csv` from that run as `<trace_id>_run_<run_id>_eval_summary.csv`

4. Compress the folder:
   ```bash
   tar -czf <report_dir>/reproducers/<sanitized_issue_name>.tar.gz \
     -C <report_dir>/reproducers <sanitized_issue_name>/
   ```

The path sanitization command for each NDJSON (replaces the full repo root path):
```bash
REPO_ABS=$(cd "$(git rev-parse --show-toplevel)" && pwd)
sed "s|${REPO_ABS}|\$REPO_ROOT|g" <source> > <dest>
```

After all issues are processed, print the count:
```
Built <N> reproducer packages in <report_dir>/reproducers/
```

## Step 8 — Save and Summarize

1. Copy `<report_dir>` to `TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/eval_reports/latest/` (remove existing `latest/` first if present):
   ```bash
   rm -rf TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/eval_reports/latest
   cp -r <report_dir> TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/eval_reports/latest
   ```

2. Print a summary to the user:
   - Overall pass rate and PASS/FAIL/MISSING counts
   - Top 3 worst-performing traces
   - Top 3 failure issues
   - **Explicitly print the full paths** to the generated outputs:
     ```
     PR report:          <report_dir>/pr_report.md
     Fix-ticket report:  <report_dir>/fix_ticket_report.md
     Reproducer packages: <report_dir>/reproducers/ (<N> issues)
     Latest copy:        TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/eval_reports/latest/
     ```
