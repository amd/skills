###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Aggregate the semantic-purity regression gate across repeated runs.

semantic_partition_scripted_evals.py records per-run LCA-partition purity
metrics but does not gate on them (see that module's docstring) -- the
semantic-bucketing method has real run-to-run variance, so a single run is not a reliable regression signal.

This script is the actual gate. For each semantic-purity test case it:
  1. Scans results_root/<test_id>/run_*/semantic_purity_results.csv,
     parsing out each run's strict_forward value (the "strict forward
     consistency" metric from compare_lca_partitions.py: the fraction of
     gold groups that map, in their entirety, to a single candidate LCA).
     strict_forward is used as the sole gating metric -- it is more
     discriminating than forward_purity because it does not give partial
     credit to a majority-only match.
  2. Averages strict_forward across however many runs were found.
  3. Compares that average against MIN_STRICT_FORWARD_AVG[test_id] -- a
     floor set BELOW the worst single independently-observed run (with
     margin), so that even a quick NUM_REPEATS=1 check does not
     false-alarm on ordinary variance, while still sitting far above the
     random-shuffle baseline (Baseline 1 in compare_lca_partitions.py,
     ~0.04-0.06 strict_forward for both models). The intent is regression
     detection ("don't get materially worse than today"), not an absolute
     quality bar.

Writes one verdict row (7-column eval schema) per test id to
results_root/<test_id>/semantic_purity_aggregate_verdict.csv, and prints a
summary table.

Usage: python3 semantic_purity_aggregate.py --results-root <RESULTS_ROOT>
"""

import argparse
import csv
import glob
import os
import re
import statistics
import sys

CSV_COLUMNS = [
    "index",
    "category",
    "issue_summary",
    "result",
    "details",
    "root_cause",
    "recommended_fix",
]

# Floors are set below the worst single strict_forward run observed during
# decode-only method validation (with margin), NOT at the mean -- so a
# one-off NUM_REPEATS=1 quick check cannot false-alarm on ordinary
# variance. Averaging across repeats leaves even more headroom.
#
# Observed decode-only strict_forward (compare_lca_partitions.py vs the
# pre-baked gold):
#   deepseek_r1:   0.7280, 0.9780, 0.9748, 0.9748, 0.9780   (worst 0.7280)
#   qwen3_30b_a3b: 0.3710, 0.5671, 0.4677, 0.7304, 0.5691    (worst 0.3710)
# Both floors sit far above the random-shuffle baseline (~0.04-0.06). If the
# method is intentionally improved, raise these; do not lower them without a
# documented reason.
MIN_STRICT_FORWARD_AVG = {
    "semantic_purity_deepseek_r1": 0.60,
    "semantic_purity_qwen3_30b_a3b": 0.30,
}

# Degenerate-collapse guard. strict_forward saturates to ~1.0 when the candidate
# partition collapses toward a single LCA bucket (every gold group is trivially
# homogeneous inside one bucket), so the floor above cannot catch total collapse.
# Distinct candidate LCA buckets on the matched set of observed healthy runs
# ranged 26-32 for both models; total collapse is 1. A floor of 10 sits ~2.6x
# below the worst healthy run and ~10x above collapse, so ordinary bucketing
# variance never false-alarms while any real collapse fails hard.
MIN_SHARED_BUCKETS = {
    "semantic_purity_deepseek_r1": 10,
    "semantic_purity_qwen3_30b_a3b": 10,
}

DETAILS_RE = re.compile(r"strict_forward=([0-9.]+)")
FORWARD_RE = re.compile(r"forward_purity=([0-9.]+)")
BUCKETS_RE = re.compile(r"shared_buckets=([0-9]+)")


def find_run_csvs(results_root: str, test_id: str) -> list[str]:
    pattern = os.path.join(
        results_root, test_id, "run_*", "semantic_purity_results.csv"
    )
    return sorted(glob.glob(pattern))


def parse_run(csv_path: str):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None  # self-gated no-op for this test id (shouldn't happen here)
    row = rows[0]
    details = row.get("details", "")
    m_sfwd = DETAILS_RE.search(details)
    m_fwd = FORWARD_RE.search(details)
    m_buckets = BUCKETS_RE.search(details)
    if not m_sfwd:
        return None  # pipeline failure row, no metrics to parse
    return {
        "strict_forward": float(m_sfwd.group(1)),
        "forward_purity": float(m_fwd.group(1)) if m_fwd else None,
        "shared_buckets": int(m_buckets.group(1)) if m_buckets else None,
        "source": csv_path,
    }


def aggregate_one(results_root: str, test_id: str) -> dict:
    run_csvs = find_run_csvs(results_root, test_id)
    parsed = [p for p in (parse_run(c) for c in run_csvs) if p is not None]

    if not parsed:
        return {
            "index": "semantic_purity_aggregate",
            "category": "Quality",
            "issue_summary": f"Semantic-purity regression gate ({test_id})",
            "result": "FAIL",
            "details": f"No usable runs found under {results_root}/{test_id}/run_*/semantic_purity_results.csv",
            "root_cause": "pipeline",
            "recommended_fix": "Ensure the run(s) completed for this test id before aggregating",
        }

    values = [p["strict_forward"] for p in parsed]
    avg = statistics.mean(values)
    floor = MIN_STRICT_FORWARD_AVG.get(test_id)
    if floor is None:
        return {
            "index": "semantic_purity_aggregate",
            "category": "Quality",
            "issue_summary": f"Semantic-purity regression gate ({test_id})",
            "result": "FAIL",
            "details": f"No floor configured for test id {test_id}; add one to MIN_STRICT_FORWARD_AVG",
            "root_cause": "config",
            "recommended_fix": "Add this test id to MIN_STRICT_FORWARD_AVG in semantic_purity_aggregate.py",
        }

    passed = avg >= floor

    # Collapse guard: fail hard if the distinct candidate LCA count on the
    # matched set drops below the floor, which would saturate strict_forward.
    bucket_floor = MIN_SHARED_BUCKETS.get(test_id)
    bucket_vals = [
        p["shared_buckets"] for p in parsed if p.get("shared_buckets") is not None
    ]
    min_buckets = min(bucket_vals) if bucket_vals else None
    collapsed = (
        bucket_floor is not None
        and min_buckets is not None
        and min_buckets < bucket_floor
    )
    passed = passed and not collapsed

    n_note = (
        ""
        if len(parsed) > 1
        else " (single run; a multi-run average is a stronger estimate)"
    )
    bucket_note = (
        f" min_shared_buckets={min_buckets} bucket_floor={bucket_floor}"
        if bucket_floor is not None
        else ""
    )
    details = (
        f"n_runs={len(parsed)} strict_forward_values={[round(v, 4) for v in values]} "
        f"avg_strict_forward={avg:.4f} floor={floor}{bucket_note}{n_note}"
    )
    if collapsed:
        details += (
            f" -- FAIL: candidate partition collapsed toward a single LCA bucket "
            f"(min {min_buckets} < {bucket_floor}); strict_forward is unreliable"
        )

    if passed:
        recommended_fix = ""
    elif collapsed:
        recommended_fix = (
            "Candidate LCA partition collapsed (near-single-bucket); strict_forward "
            "is saturated and meaningless. Investigate the clustering/coherence step."
        )
    else:
        recommended_fix = (
            "Semantic-bucketing quality regressed below the observed-performance "
            "floor; investigate the clustering change"
        )

    return {
        "index": "semantic_purity_aggregate",
        "category": "Quality",
        "issue_summary": f"Semantic-purity regression gate ({test_id})",
        "result": "PASS" if passed else "FAIL",
        "details": details,
        "root_cause": "" if passed else "quality",
        "recommended_fix": recommended_fix,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate semantic-purity gate across repeats"
    )
    parser.add_argument("--results-root", required=True)
    parser.add_argument(
        "--test-ids",
        default=",".join(MIN_STRICT_FORWARD_AVG.keys()),
        help="Comma-separated test ids to aggregate (default: all known semantic-purity test ids)",
    )
    args = parser.parse_args()

    test_ids = [t.strip() for t in args.test_ids.split(",") if t.strip()]
    overall_pass = True
    for test_id in test_ids:
        case_dir = os.path.join(args.results_root, test_id)
        if not os.path.isdir(case_dir):
            print(f"[{test_id}] skipped (no results directory found)")
            continue
        result = aggregate_one(args.results_root, test_id)
        out_path = os.path.join(case_dir, "semantic_purity_aggregate_verdict.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerow(result)
        print(f"[{test_id}] {result['result']}: {result['details']}")
        print(f"  -> {out_path}")
        if result["result"] != "PASS":
            overall_pass = False

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
