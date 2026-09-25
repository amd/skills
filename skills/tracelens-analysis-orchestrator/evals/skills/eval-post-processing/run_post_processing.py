#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Eval post-processing driver: aggregate + classify + reports + reproducers + save.

Reads source per-run *_results.csv files directly (which preserves correct
column alignment for LLM eval rows that contain embedded newlines),
re-aggregates, classifies failures, writes PR + fix-ticket reports, builds
reproducer packages, and copies the report tree to eval_reports/latest/.
"""

import csv
import glob
import logging
import os
import re
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_UTILS_DIR = os.path.join(_SCRIPT_DIR, "..", "..", "eval_utils")
if _EVAL_UTILS_DIR not in sys.path:
    sys.path.insert(0, _EVAL_UTILS_DIR)
from aggregate_repeatability import find_runs, parse_ndjson_stream

REPO_ROOT = "/workspace/TraceLens"
ANALYSIS_DIR = os.path.join(
    REPO_ROOT,
    "TraceLens",
    "Agent",
    "Analysis",
    "skills",
    "analysis-orchestrator",
    "evals",
)
RESULTS_ROOT = "/workspace/TraceLens/TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/repeatability_results_combined"
REPORT_DIR = "/workspace/TraceLens/TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/reports"
TEST_TRACES_CSV = "/workspace/TraceLens/TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/reports/combined_traces.csv"
SUITE = "eval"
CONTAINER = ""  # local host, no container

AGG_DIR = os.path.join(REPORT_DIR, "aggregates")
REPRODUCERS_DIR = os.path.join(REPORT_DIR, "reproducers")
LATEST_DIR = os.path.join(ANALYSIS_DIR, "eval_reports", "latest")
RULES_PATH = os.path.join(ANALYSIS_DIR, "eval_utils", "report_section_rules.yaml")

CSV_COLUMNS = [
    "index",
    "category",
    "issue_summary",
    "result",
    "details",
    "root_cause",
    "recommended_fix",
]
EVAL_OUTPUT_COLS = [
    "trace_id",
    "run_id",
    "eval_index",
    "eval_category",
    "issue_summary",
    "result",
    "details",
    "root_cause",
    "recommended_fix",
]


def normalize_row(row):
    """Map a source-CSV row to the canonical 7-column schema."""
    out = {col: "" for col in CSV_COLUMNS}
    if len(row) == 7:
        for i, col in enumerate(CSV_COLUMNS):
            out[col] = row[i]
    elif len(row) == 5:
        legacy = ["index", "category", "issue_summary", "result", "details"]
        for i, col in enumerate(legacy):
            out[col] = row[i]
    elif len(row) > 7:
        for i, col in enumerate(CSV_COLUMNS):
            out[col] = row[i] if i < len(row) else ""
        out["details"] = ",".join(row[4:])
    else:
        for i, col in enumerate(CSV_COLUMNS):
            out[col] = row[i] if i < len(row) else ""
    return out


def reaggregate_from_sources(results_root):
    """Read per-run *_results.csv files directly to preserve column alignment."""
    rows = []
    for trace_id in sorted(os.listdir(results_root)):
        td = os.path.join(results_root, trace_id)
        if not os.path.isdir(td):
            continue
        for run_dir in sorted(os.listdir(td)):
            if not run_dir.startswith("run_"):
                continue
            run_id = int(run_dir.split("_")[1])
            rd = os.path.join(td, run_dir)
            for src in sorted(glob.glob(os.path.join(rd, "*_results.csv"))):
                with open(src, newline="") as f:
                    reader = csv.reader(f)
                    try:
                        next(reader)  # skip header
                    except StopIteration:
                        continue
                    for row in reader:
                        if not any(row):
                            continue
                        norm = normalize_row(row)
                        rows.append(
                            {
                                "trace_id": trace_id,
                                "run_id": run_id,
                                "eval_index": norm["index"],
                                "eval_category": norm["category"],
                                "issue_summary": norm["issue_summary"],
                                "result": norm["result"],
                                "details": norm["details"],
                                "root_cause": norm["root_cause"],
                                "recommended_fix": norm["recommended_fix"],
                            }
                        )
    return rows


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def classify_stability(pass_count, total):
    if total == 0:
        return "N/A"
    rate = pass_count / total
    if rate == 1.0:
        return "STABLE_PASS"
    if rate > 0.5:
        return "FLAKY_PASS"
    if rate > 0.0:
        return "FLAKY_FAIL"
    return "STABLE_FAIL"


def build_pass_rate_summary(agg_rows, trace_ids):
    """Returns dict[trace_id] -> dict[eval_index] -> str (pass_rate)."""
    rows = []
    by_trace_eval = defaultdict(dict)  # trace_id -> eval_index -> {PASS, FAIL, total}
    for r in agg_rows:
        if r["trace_id"] not in trace_ids:
            continue
        ei = r["eval_index"] or "_unknown"
        res = r["result"]
        if res not in ("PASS", "FAIL"):
            continue
        d = by_trace_eval[r["trace_id"]]
        d.setdefault(ei, {"PASS": 0, "FAIL": 0, "total": 0})
        d[ei][res] += 1
        d[ei]["total"] += 1

    for trace_id in sorted(trace_ids):
        row = {"trace_id": trace_id}
        evals = by_trace_eval.get(trace_id, {})
        pass_sum = fail_sum = total_sum = 0
        for ei, counts in evals.items():
            c_total = counts["total"]
            row[ei] = f"{counts['PASS']}/{c_total}" if c_total > 0 else "N/A"
            pass_sum += counts["PASS"]
            fail_sum += counts["FAIL"]
            total_sum += c_total
        row["overall_pass_rate"] = (
            f"{pass_sum}/{total_sum} ({round(100*pass_sum/total_sum, 1)}%)"
            if total_sum
            else "N/A"
        )
        rows.append(row)
    return rows


def build_stability_summary(agg_rows, trace_ids):
    rows = []
    by_trace_eval = defaultdict(dict)
    for r in agg_rows:
        if r["trace_id"] not in trace_ids:
            continue
        ei = r["eval_index"] or "_unknown"
        res = r["result"]
        if res not in ("PASS", "FAIL"):
            continue
        d = by_trace_eval[r["trace_id"]]
        d.setdefault(ei, {"PASS": 0, "FAIL": 0, "total": 0})
        d[ei]["total"] += 1
        if res == "PASS":
            d[ei]["PASS"] += 1
    for trace_id in sorted(trace_ids):
        row = {"trace_id": trace_id}
        for ei, counts in by_trace_eval.get(trace_id, {}).items():
            row[ei] = classify_stability(counts["PASS"], counts["total"])
        rows.append(row)
    return rows


def build_run_level_summary(agg_rows):
    """Returns list of dicts: trace_id, run_id, pass, fail, total, is_catastrophic."""
    by_run = defaultdict(lambda: {"PASS": 0, "FAIL": 0})
    for r in agg_rows:
        if r["result"] not in ("PASS", "FAIL"):
            continue
        key = (r["trace_id"], r["run_id"])
        by_run[key][r["result"]] += 1
    rows = []
    for (trace_id, run_id), counts in sorted(by_run.items()):
        total = counts["PASS"] + counts["FAIL"]
        is_cat = total > 0 and (counts["FAIL"] / total) > 0.5
        rows.append(
            {
                "trace_id": trace_id,
                "run_id": run_id,
                "pass": counts["PASS"],
                "fail": counts["FAIL"],
                "total": total,
                "is_catastrophic": str(is_cat),
            }
        )
    return rows


def build_failure_nature(run_level_rows, stability_rows, all_fail_rows):
    """Count failures by nature: catastrophic_pipeline, stable, flaky."""
    # Per-trace-per-eval stability: STABLE_FAIL = stable, FLAKY_FAIL = flaky
    for r in all_fail_rows:
        # all_fail_rows is the aggregated results with results PASS/FAIL only
        pass  # not used; we'll classify via stability
    stable = flaky = 0
    eval_pairs_total = 0
    for row in stability_rows:
        for k, v in row.items():
            if k == "trace_id":
                continue
            if v in ("STABLE_FAIL",):
                stable += 1
                eval_pairs_total += 1
            elif v in ("FLAKY_FAIL",):
                flaky += 1
                eval_pairs_total += 1
    catastrophic = sum(1 for r in run_level_rows if r["is_catastrophic"] == "True")
    return {
        "catastrophic_pipeline": catastrophic,
        "stable": stable,
        "flaky": flaky,
        "total": stable + flaky,
    }


def parse_stream_diagnostics(results_root):
    rows = []
    for trace_id, run_id, run_dir in find_runs(results_root):
        parsed = parse_ndjson_stream(os.path.join(run_dir, "analysis_stream.ndjson"))
        rows.append(
            {
                "trace_id": trace_id,
                "run_id": run_id,
                "outcome": parsed["outcome"],
                "duration_ms": parsed["duration_ms"],
                "input_tokens": parsed["input_tokens"],
                "output_tokens": parsed["output_tokens"],
                "cache_read_tokens": parsed["cache_read_tokens"],
                "turns": str(parsed["turns"]),
                "tool_calls": str(parsed["tool_calls"]),
                "report_written": str(parsed["report_written"]),
                "report_headers": parsed["report_headers"],
                "last_step_reached": parsed["last_step_reached"],
            }
        )
    return rows


def safe_name(s, max_len=80):
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:max_len] if len(s) > max_len else s


def pct(num, denom):
    if denom == 0:
        return 0.0
    return round(100.0 * num / denom, 1)


def fmt_pass_rate(p, t):
    if t == 0:
        return "0/0 (0%)"
    return f"{p}/{t} ({pct(p, t)}%)"


def pattern_label(fail_counts, total_per_run):
    if not fail_counts:
        return "stable"
    catastrophic = any(f / t > 0.5 for f, t in zip(fail_counts, total_per_run) if t > 0)
    if catastrophic:
        return "catastrophic"
    spread = max(fail_counts) - min(fail_counts)
    if spread <= 1:
        return "stable"
    return "flaky"


def classify_row(row, rules):
    eval_index = row.get("eval_index", "") or ""
    issue_summary = row.get("issue_summary", "") or ""
    details = row.get("details", "") or ""
    base_section = "Others"
    for r in rules["base_section_rules"]:
        if "eval_index_regex" in r and re.search(r["eval_index_regex"], eval_index):
            base_section = r["section"]
            break
        if "issue_summary_regex" in r and re.search(
            r["issue_summary_regex"], issue_summary
        ):
            base_section = r["section"]
            break
    standalone_section = "Detailed Analysis"
    for r in rules["standalone_section_rules"]:
        if "eval_index_regex" in r and re.search(r["eval_index_regex"], eval_index):
            standalone_section = r["section"]
            break
        if "issue_summary_regex" in r and re.search(
            r["issue_summary_regex"], issue_summary
        ):
            standalone_section = r["section"]
            break
    haystack = f"{eval_index} {issue_summary} {details}"
    likely_cause = rules["defaults"]["likely_cause"]
    suggested_fix = rules["defaults"]["suggested_fix"]
    for r in rules["failure_mode_rules"]:
        if "match_regex" in r and re.search(r["match_regex"], haystack):
            likely_cause = r["likely_cause"]
            suggested_fix = r["suggested_fix"]
            break
    return base_section, standalone_section, likely_cause, suggested_fix


def load_yaml_rules():
    with open(RULES_PATH) as f:
        return yaml.safe_load(f)


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_trace_meta(csv_path):
    rows = load_csv(csv_path)
    return {
        r["id"]: {"sub_category": r["sub_category"], "platform": r["platform"]}
        for r in rows
    }


def split_unit_e2e(trace_meta):
    unit = {
        tid: m for tid, m in trace_meta.items() if m["sub_category"] != "full_model"
    }
    e2e = {tid: m for tid, m in trace_meta.items() if m["sub_category"] == "full_model"}
    return unit, e2e


def compute_per_split(trace_ids, agg_rows, run_level_rows, stream_rows, trace_meta):
    rows = [r for r in agg_rows if r["trace_id"] in trace_ids]
    p = sum(1 for r in rows if r["result"] == "PASS")
    f = sum(1 for r in rows if r["result"] == "FAIL")
    m = sum(1 for r in rows if r["result"] not in ("PASS", "FAIL"))
    total = p + f + m
    rate = pct(p, total)
    issue_counter = Counter(r["issue_summary"] for r in rows if r["result"] == "FAIL")
    per_trace_pass = Counter()
    per_trace_fail = Counter()
    per_trace_missing = Counter()
    for r in rows:
        tid = r["trace_id"]
        if r["result"] == "PASS":
            per_trace_pass[tid] += 1
        elif r["result"] == "FAIL":
            per_trace_fail[tid] += 1
        else:
            per_trace_missing[tid] += 1
    per_case = []
    for tid in sorted(trace_ids):
        ps, fs, ms = per_trace_pass[tid], per_trace_fail[tid], per_trace_missing[tid]
        case_total = ps + fs + ms
        meta = trace_meta.get(tid, {"sub_category": "unknown", "platform": "unknown"})
        s_rows = [s for s in stream_rows if s["trace_id"] == tid]
        successful_runs = sum(
            1
            for s in s_rows
            if str(s.get("outcome", "")).lower() not in ("", "no_result_record")
        )
        total_runs = len(s_rows)
        durations = [
            int(s["duration_ms"])
            for s in s_rows
            if str(s.get("duration_ms", "")).strip() not in ("",)
            and int(s["duration_ms"] or 0) > 0
        ]
        avg_dur_s = (
            round(sum(durations) / len(durations) / 1000.0, 1) if durations else 0.0
        )
        per_case.append(
            {
                "trace_id": tid,
                "sub_category": meta["sub_category"],
                "platform": meta["platform"],
                "pass": ps,
                "fail": fs,
                "missing": ms,
                "pass_rate": fmt_pass_rate(ps, case_total),
                "runs": f"{successful_runs}/{total_runs}",
                "avg_dur": avg_dur_s,
            }
        )
    per_case.sort(key=lambda x: -x["fail"])
    per_trace_fail_count = sorted(
        [(tid, per_trace_fail.get(tid, 0)) for tid in trace_ids],
        key=lambda x: -x[1],
    )
    return {
        "pass": p,
        "fail": f,
        "missing": m,
        "total": total,
        "rate": rate,
        "top_issues": issue_counter.most_common(10),
        "per_case": per_case,
        "per_trace_fail_count": per_trace_fail_count,
    }


def top_failure_modes(classified_failures, trace_ids, n=8):
    rows = [r for r in classified_failures if r["trace_id"] in trace_ids]
    issue_counter = Counter(r["issue_summary"] for r in rows)
    out = []
    for issue, cnt in issue_counter.most_common(n):
        pair = next(
            (r for r in rows if r["issue_summary"] == issue),
            {"likely_cause": "", "suggested_fix": ""},
        )
        out.append((issue, cnt, pair["likely_cause"], pair["suggested_fix"]))
    return out


# ----------------- Report builders -----------------


def metrics_table(overall, unit, e2e):
    return (
        "| Metric | Overall | Unit Tests | E2E Tests |\n"
        "|---|---|---|---|\n"
        f"| PASS | {overall['pass']} | {unit['pass']} | {e2e['pass']} |\n"
        f"| FAIL | {overall['fail']} | {unit['fail']} | {e2e['fail']} |\n"
        f"| MISSING | {overall['missing']} | {unit['missing']} | {e2e['missing']} |\n"
        f"| Pass rate | {overall['rate']}% | {unit['rate']}% | {e2e['rate']}% |\n"
    )


def failure_nature_table(fn):
    t = fn["total"] or 1
    return (
        "| Nature | Count | % of Failures | Description |\n"
        "|---|---|---|---|\n"
        f"| Catastrophic pipeline | {fn['catastrophic_pipeline']} | {pct(fn['catastrophic_pipeline'], t)}% | Agent crashed — entire run fails |\n"
        f"| Stable | {fn['stable']} | {pct(fn['stable'], t)}% | Consistent failures — real bugs |\n"
        f"| Flaky | {fn['flaky']} | {pct(fn['flaky'], t)}% | Intermittent — agent non-determinism |\n"
    )


def failure_sections_table(section_counter):
    rows = ["| Section | Failures |\n|---|---|\n"]
    for section, cnt in sorted(section_counter.items(), key=lambda x: -x[1]):
        rows.append(f"| {section} | {cnt} |\n")
    return "".join(rows)


def top_issues_table(top_issues, label="Overall", limit=10):
    rows = [f"### Top Failure Issues ({label})\n\n", "| Issue | Count |\n|---|---|\n"]
    if not top_issues:
        rows.append("| (no failures) | 0 |\n")
    else:
        for issue, cnt in top_issues[:limit]:
            rows.append(f"| {issue} | {cnt} |\n")
    return "".join(rows)


def per_case_table(per_case, label):
    if not per_case:
        return f"### Per-Case Results ({label})\n\nNo {label.lower()} test cases in this run.\n"
    rows = [
        f"### Per-Case Results ({label})\n\n",
        "| Case | Category | Platform | PASS | FAIL | MISSING | Pass Rate | Runs | Avg Duration |\n",
        "|---|---|---|---|---|---|---|---|---|\n",
    ]
    for c in per_case:
        rows.append(
            f"| {c['trace_id']} | {c['sub_category']} | {c['platform']} | "
            f"{c['pass']} | {c['fail']} | {c['missing']} | {c['pass_rate']} | "
            f"{c['runs']} | {c['avg_dur']}s |\n"
        )
    return "".join(rows)


def failure_modes_table(modes, label):
    if not modes:
        return f"### Failure Modes ({label})\n\nNo failures in {label.lower()} cases.\n"
    rows = [
        f"### Failure Modes ({label})\n\n",
        "| Issue | Count | Likely cause | Suggested fix |\n|---|---|---|---|\n",
    ]
    for issue, cnt, cause, fix in modes:
        rows.append(f"| {issue} | {cnt} | {cause} | {fix} |\n")
    return "".join(rows)


def top_reproducers_table(
    per_trace_fail_count, trace_meta, test_traces_csv_rel, container, n=5
):
    rows = [
        f"### Top Reproducers\n\n",
        "| Trace/Case | Failures | Platform | Reproducer command |\n|---|---|---|---|\n",
    ]
    top_n = [t for t in per_trace_fail_count if t[1] > 0][:n]
    if not top_n:
        rows.append("| (none) | 0 | - | - |\n")
        return "".join(rows)
    for tid, fcount in top_n:
        platform = trace_meta.get(tid, {}).get("platform", "unknown")
        container_kv = f'CONTAINER="{container}"' if container else 'CONTAINER=""'
        cmd = (
            f'{container_kv} TEST_IDS="{tid}" TEST_TRACES_CSV="{test_traces_csv_rel}" '
            f"bash TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/eval_scripts/run_repeatability_parallel.sh"
        )
        rows.append(f"| {tid} | {fcount} | {platform} | `{cmd}` |\n")
    return "".join(rows)


def catastrophic_table(catastrophic_runs):
    if not catastrophic_runs:
        return "### Catastrophic Runs\n\nNo catastrophic runs detected.\n"
    rows = [
        "### Catastrophic Runs\n\n",
        "| Case | Run | Pass | Fail | Total |\n|---|---|---|---|---|\n",
    ]
    for r in catastrophic_runs:
        rows.append(
            f"| {r['trace_id']} | run_{r['run_id']} | {r['pass']} | {r['fail']} | {r['total']} |\n"
        )
    return "".join(rows)


def per_case_pattern_table(per_case_pattern, num_runs=5):
    header_cells = [f"run_{i}" for i in range(num_runs)]
    rows = [
        "### Per-Case Run Pattern\n\n",
        "| Case | "
        + " | ".join(header_cells)
        + " | Total Fails | Pattern |\n|"
        + "|".join(["---"] * (num_runs + 2))
        + "|\n",
    ]
    for entry in per_case_pattern:
        runs = entry["runs"]
        cells = []
        for run_idx in range(num_runs):
            run_row = next((r for r in runs if int(r["run_id"]) == run_idx), None)
            if run_row:
                label = (
                    " (**crash**)"
                    if str(run_row.get("is_catastrophic", "")).lower() == "true"
                    else ""
                )
                cells.append(f"{run_row['fail']}/{run_row['total']}{label}")
            else:
                cells.append("-")
        rows.append(
            f"| {entry['trace_id']} | "
            + " | ".join(cells)
            + f" | {entry['total_fails']} | {entry['label']} |\n"
        )
    return "".join(rows)


def main():
    rules = load_yaml_rules()
    trace_meta = build_trace_meta(TEST_TRACES_CSV)
    unit_ids, e2e_ids = split_unit_e2e(trace_meta)
    print(f"Unit test cases: {len(unit_ids)}, E2E test cases: {len(e2e_ids)}")

    # ---------- Step 4: re-aggregate from source files ----------
    print("\nRe-aggregating from source *_results.csv files...")
    agg_rows = reaggregate_from_sources(RESULTS_ROOT)
    pass_rows = [r for r in agg_rows if r["result"] == "PASS"]
    fail_rows = [r for r in agg_rows if r["result"] == "FAIL"]
    miss_rows = [r for r in agg_rows if r["result"] not in ("PASS", "FAIL")]
    print(
        f"  PASS={len(pass_rows)}, FAIL={len(fail_rows)}, MISSING={len(miss_rows)}, total={len(agg_rows)}"
    )

    # Write aggregated_results.csv
    write_csv(
        os.path.join(AGG_DIR, "aggregated_results.csv"), agg_rows, EVAL_OUTPUT_COLS
    )
    print(f"  Wrote: {AGG_DIR}/aggregated_results.csv")

    # Pass rate summary
    pr_rows = build_pass_rate_summary(agg_rows, set(trace_meta.keys()))
    cols = sorted({k for r in pr_rows for k in r.keys()})
    write_csv(os.path.join(AGG_DIR, "pass_rate_summary.csv"), pr_rows, cols)
    print(f"  Wrote: {AGG_DIR}/pass_rate_summary.csv")

    # Stability summary
    st_rows = build_stability_summary(agg_rows, set(trace_meta.keys()))
    cols = sorted({k for r in st_rows for k in r.keys()})
    write_csv(os.path.join(AGG_DIR, "stability_summary.csv"), st_rows, cols)
    print(f"  Wrote: {AGG_DIR}/stability_summary.csv")

    # Run-level summary
    rl_rows = build_run_level_summary(agg_rows)
    write_csv(
        os.path.join(AGG_DIR, "run_level_summary.csv"),
        rl_rows,
        ["trace_id", "run_id", "pass", "fail", "total", "is_catastrophic"],
    )
    print(f"  Wrote: {AGG_DIR}/run_level_summary.csv")

    # Failure nature summary
    fn = build_failure_nature(rl_rows, st_rows, fail_rows)
    fn_rows = [
        {
            "catastrophic_pipeline": fn["catastrophic_pipeline"],
            "stable": fn["stable"],
            "flaky": fn["flaky"],
            "total": fn["total"],
        }
    ]
    write_csv(
        os.path.join(AGG_DIR, "failure_nature_summary.csv"),
        fn_rows,
        ["catastrophic_pipeline", "stable", "flaky", "total"],
    )
    print(f"  Wrote: {AGG_DIR}/failure_nature_summary.csv")
    print(f"  Failure nature: {fn}")

    # Stream diagnostics
    sd_rows = parse_stream_diagnostics(RESULTS_ROOT)
    write_csv(
        os.path.join(AGG_DIR, "stream_diagnostics.csv"),
        sd_rows,
        [
            "trace_id",
            "run_id",
            "outcome",
            "duration_ms",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "turns",
            "tool_calls",
            "report_written",
            "report_headers",
            "last_step_reached",
        ],
    )
    print(f"  Wrote: {AGG_DIR}/stream_diagnostics.csv")

    # ---------- Step 5: classify ----------
    classified_failures = []
    for row in fail_rows:
        bs, ss, lc, sf = classify_row(row, rules)
        classified_failures.append(
            {
                **row,
                "base_section": bs,
                "standalone_section": ss,
                "likely_cause": lc,
                "suggested_fix": sf,
            }
        )

    classified_path = os.path.join(AGG_DIR, "classified_failures.csv")
    if classified_failures:
        cols = list(classified_failures[0].keys())
        write_csv(classified_path, classified_failures, cols)
        print(f"  Wrote: {classified_path}")

    section_counter = Counter(f["base_section"] for f in classified_failures)

    # Compute metrics
    overall = compute_per_split(
        set(trace_meta.keys()), agg_rows, rl_rows, sd_rows, trace_meta
    )
    unit_metrics = compute_per_split(
        set(unit_ids.keys()), agg_rows, rl_rows, sd_rows, trace_meta
    )
    e2e_metrics = compute_per_split(
        set(e2e_ids.keys()), agg_rows, rl_rows, sd_rows, trace_meta
    )

    unit_modes = top_failure_modes(classified_failures, set(unit_ids.keys()), n=8)
    e2e_modes = top_failure_modes(classified_failures, set(e2e_ids.keys()), n=8)

    # Per-case pattern
    run_by_trace = defaultdict(list)
    for r in rl_rows:
        run_by_trace[r["trace_id"]].append(r)
    per_case_pattern = []
    for tid in sorted(trace_meta.keys()):
        runs = sorted(run_by_trace.get(tid, []), key=lambda x: int(x["run_id"]))
        if not runs:
            continue
        fc = [int(r["fail"]) for r in runs]
        totals = [int(r["total"]) for r in runs]
        total_fails = sum(fc)
        label = pattern_label(fc, totals)
        per_case_pattern.append(
            {"trace_id": tid, "runs": runs, "total_fails": total_fails, "label": label}
        )
    per_case_pattern.sort(key=lambda x: -x["total_fails"])

    catastrophic_runs = [
        r for r in rl_rows if str(r.get("is_catastrophic", "")).lower() == "true"
    ]
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    test_traces_csv_rel = os.path.relpath(TEST_TRACES_CSV, REPO_ROOT)

    # ---------- Step 6: PR report ----------
    pr = []
    pr.append("# Automated Eval Report (PR)\n\n")
    pr.append(f"Generated at: `{timestamp}`\n\n")
    pr.append(metrics_table(overall, unit_metrics, e2e_metrics))
    pr.append("\n## Failure Nature Summary\n\n")
    pr.append(failure_nature_table(fn))
    pr.append(
        f"\nCatastrophic runs: {fn['catastrophic_pipeline']}\n\n"
        if fn["catastrophic_pipeline"]
        else "\nCatastrophic runs: None\n\n"
    )
    pr.append("## Failure Sections\n\n")
    pr.append(failure_sections_table(section_counter))
    pr.append("\n")
    pr.append(top_issues_table(overall["top_issues"], "Overall"))
    pr.append("\n---\n\n")
    pr.append(
        f"## Unit Test Cases ({len(unit_ids)} cases, {unit_metrics['rate']}% pass rate)\n\n"
    )
    pr.append(per_case_table(unit_metrics["per_case"], "Unit"))
    pr.append("\n")
    pr.append(top_issues_table(unit_metrics["top_issues"], "Unit Tests"))
    pr.append("\n---\n\n")
    pr.append(
        f"## E2E Test Cases ({len(e2e_ids)} cases, {e2e_metrics['rate']}% pass rate)\n\n"
    )
    pr.append(per_case_table(e2e_metrics["per_case"], "E2E"))
    pr.append("\n")
    pr.append(top_issues_table(e2e_metrics["top_issues"], "E2E Tests"))

    pr_path = os.path.join(REPORT_DIR, "pr_report.md")
    with open(pr_path, "w") as f:
        f.write("".join(pr))
    print(f"\nWrote: {pr_path}")

    # ---------- Fix-ticket report ----------
    fx = []
    fx.append("# Automated Eval Report (Fix Ticket)\n\n")
    fx.append(f"Generated at: `{timestamp}`\n\n")
    fx.append(metrics_table(overall, unit_metrics, e2e_metrics))
    fx.append("\n## Failure Nature Analysis\n\n")
    fx.append(
        "Classifies all failures by their nature using `run_level_summary.csv` and "
        "`failure_nature_summary.csv`:\n\n"
    )
    fx.append("| Nature | Count | % of Failures | Description |\n|---|---|---|---|\n")
    fx.append(
        f"| Catastrophic pipeline | {fn['catastrophic_pipeline']} | {pct(fn['catastrophic_pipeline'], fn['total'])}% | "
        "Agent crashed / no analysis.md — entire run fails (>50% of checks) |\n"
    )
    fx.append(
        f"| Stable | {fn['stable']} | {pct(fn['stable'], fn['total'])}% | "
        "Eval fails consistently in every run for a given trace — real bug |\n"
    )
    fx.append(
        f"| Flaky | {fn['flaky']} | {pct(fn['flaky'], fn['total'])}% | "
        "Eval fails in some runs but not others — agent non-determinism |\n"
    )
    fx.append("\n")
    fx.append(catastrophic_table(catastrophic_runs))
    fx.append("\n")
    fx.append(per_case_pattern_table(per_case_pattern, num_runs=5))
    fx.append("\n---\n\n")
    fx.append(
        f"## Unit Test Cases ({len(unit_ids)} cases, {unit_metrics['rate']}% pass rate)\n\n"
    )
    fx.append(per_case_table(unit_metrics["per_case"], "Unit"))
    fx.append("\n")
    fx.append(top_issues_table(unit_metrics["top_issues"], "Unit Tests", limit=999))
    fx.append("\n")
    fx.append(failure_modes_table(unit_modes, "Unit Tests"))
    fx.append("\n")
    fx.append(
        top_reproducers_table(
            unit_metrics["per_trace_fail_count"],
            trace_meta,
            test_traces_csv_rel,
            CONTAINER,
            n=5,
        )
    )
    fx.append("\n---\n\n")
    fx.append(
        f"## E2E Test Cases ({len(e2e_ids)} cases, {e2e_metrics['rate']}% pass rate)\n\n"
    )
    fx.append(per_case_table(e2e_metrics["per_case"], "E2E"))
    fx.append("\n")
    fx.append(top_issues_table(e2e_metrics["top_issues"], "E2E Tests", limit=999))
    fx.append("\n")
    fx.append(failure_modes_table(e2e_modes, "E2E Tests"))
    fx.append("\n")
    fx.append(
        top_reproducers_table(
            e2e_metrics["per_trace_fail_count"],
            trace_meta,
            test_traces_csv_rel,
            CONTAINER,
            n=5,
        )
    )

    fx_path = os.path.join(REPORT_DIR, "fix_ticket_report.md")
    with open(fx_path, "w") as f:
        f.write("".join(fx))
    print(f"Wrote: {fx_path}")

    # ---------- Step 7: reproducer packages ----------
    by_issue = defaultdict(list)
    for f in classified_failures:
        by_issue[f["issue_summary"]].append(f)

    if os.path.isdir(REPRODUCERS_DIR):
        shutil.rmtree(REPRODUCERS_DIR)
    os.makedirs(REPRODUCERS_DIR, exist_ok=True)

    issue_names = []
    for issue, failures in by_issue.items():
        safe = safe_name(issue)
        orig_safe = safe
        n = 2
        while safe in issue_names:
            safe = f"{orig_safe}_{n}"
            n += 1
        issue_names.append(safe)
        folder = os.path.join(REPRODUCERS_DIR, safe)
        os.makedirs(folder, exist_ok=True)

        unique_pairs = []
        seen = set()
        for f in failures:
            key = (f["trace_id"], f["run_id"])
            if key not in seen:
                seen.add(key)
                unique_pairs.append(key)
            if len(unique_pairs) >= 3:
                break

        readme = [f"# Reproducer: {issue}\n\n"]
        readme.append(f"**Total failures across all runs:** {len(failures)}\n\n")
        readme.append("## Affected traces\n\n")
        readme.append(
            "| Trace | Run | Platform | Details snippet |\n|---|---|---|---|\n"
        )
        for f in failures[:30]:
            platform = trace_meta.get(f["trace_id"], {}).get("platform", "unknown")
            details = (
                (f.get("details") or "")[:200].replace("|", "\\|").replace("\n", " ")
            )
            readme.append(
                f"| {f['trace_id']} | run_{f['run_id']} | {platform} | {details} |\n"
            )

        worst_trace = max(
            (f["trace_id"] for f in failures),
            key=lambda t: sum(1 for f in failures if f["trace_id"] == t),
        )
        container_kv = f'CONTAINER="{CONTAINER}"' if CONTAINER else 'CONTAINER=""'
        repro_cmd = (
            f'{container_kv} TEST_IDS="{worst_trace}" TEST_TRACES_CSV="{test_traces_csv_rel}" '
            f"bash TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/eval_scripts/run_repeatability_parallel.sh"
        )
        readme.append("\n## Reproducer command (worst-affected trace)\n\n```bash\n")
        readme.append(repro_cmd + "\n```\n\n")
        if failures:
            readme.append("## Failure mode\n\n")
            readme.append(f"**Likely cause:** {failures[0]['likely_cause']}\n\n")
            readme.append(f"**Suggested fix:** {failures[0]['suggested_fix']}\n")

        with open(os.path.join(folder, "README.md"), "w") as f:
            f.write("".join(readme))

        repo_abs = REPO_ROOT
        for tid, rid in unique_pairs:
            stream_src = os.path.join(
                RESULTS_ROOT, tid, f"run_{rid}", "analysis_stream.ndjson"
            )
            stream_dst = os.path.join(folder, f"{tid}_run_{rid}.ndjson")
            if os.path.isfile(stream_src):
                try:
                    with open(stream_src) as src, open(stream_dst, "w") as dst:
                        dst.write(src.read().replace(repo_abs, "$REPO_ROOT"))
                except Exception:
                    logger.warning(
                        "Failed to sanitize stream log for %s run_%s",
                        tid,
                        rid,
                        exc_info=True,
                    )
            eval_src = os.path.join(RESULTS_ROOT, tid, f"run_{rid}", "eval_summary.csv")
            eval_dst = os.path.join(folder, f"{tid}_run_{rid}_eval_summary.csv")
            if os.path.isfile(eval_src):
                shutil.copyfile(eval_src, eval_dst)

        tar_path = os.path.join(REPRODUCERS_DIR, f"{safe}.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(folder, arcname=safe)

    print(f"Built {len(by_issue)} reproducer packages in {REPRODUCERS_DIR}/")

    # ---------- Step 8: save to eval_reports/latest/ ----------
    if os.path.isdir(LATEST_DIR):
        shutil.rmtree(LATEST_DIR)
    shutil.copytree(REPORT_DIR, LATEST_DIR)
    print(f"Copied report to: {LATEST_DIR}")

    # ---------- Summary ----------
    print()
    print("=" * 70)
    print("POST-PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Suite: {SUITE}")
    print(f"Test traces CSV: {TEST_TRACES_CSV}")
    print(f"Results root: {RESULTS_ROOT}")
    print(
        f"Overall pass rate: {overall['rate']}% "
        f"(PASS={overall['pass']}, FAIL={overall['fail']}, MISSING={overall['missing']})"
    )
    print(f"  Unit:  {unit_metrics['rate']}% ({len(unit_ids)} cases)")
    print(f"  E2E:   {e2e_metrics['rate']}% ({len(e2e_ids)} cases)")
    print()
    print("Top 3 worst-performing traces (by FAIL count):")
    for tid, fc in overall["per_trace_fail_count"][:3]:
        if fc > 0:
            print(f"  - {tid}: {fc} failures")
    print()
    print("Top 3 failure issues:")
    for issue, cnt in overall["top_issues"][:3]:
        print(f"  - {issue}: {cnt}")
    print()
    print(f"PR report:           {REPORT_DIR}/pr_report.md")
    print(f"Fix-ticket report:   {REPORT_DIR}/fix_ticket_report.md")
    print(f"Reproducer packages: {REPRODUCERS_DIR}/ ({len(by_issue)} issues)")
    print(f"Latest copy:         {LATEST_DIR}/")


if __name__ == "__main__":
    main()
