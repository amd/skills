###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Scripted eval: LCA-partition purity vs. a with-capture (gold) reference.

Self-gating: only test cases whose reference_dir contains a
``semantic_purity_gold_diff_stats.csv`` (which ships pre-baked in each
partial-test fixture; regenerated only rarely by
``generate_gold_ref.sh``) are scored. Any other test case
produces zero rows here, so this eval is inert unless a gold reference is
present.

For gated test cases, this reuses compare_lca_partitions.py's forward/reverse
purity and strict forward/reverse consistency metrics, computed between the
gold partition and the candidate's semantic-bucketing output
(``tracediff_output/diff_stats.csv``, produced by the semantic-comparison
workflow through its "Generate TraceDiff Output" step), and records them.

IMPORTANT: this per-run result is informational only. Because the semantic
method has real run-to-run variance, a single
run's metrics are not a reliable regression signal by themselves -- the
actual pass/fail decision is made by semantic_purity_aggregate.py, which
averages strict_forward across NUM_REPEATS runs and compares against a
floor derived from currently-observed performance. See that script and the
partial_tests/README.md "Semantic-purity quality gate" section.

Consequently the "result" written here is PASS whenever the metrics were
computed at all; it only turns FAIL for genuine pipeline errors (candidate
output missing, no matched keys).
"""

import argparse
import csv
import os
import sys

# compare_lca_partitions lives two dirs up in eval_utils/ (not a package).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ANALYSIS_EVAL_UTILS = os.path.abspath(os.path.join(_HERE, "..", "..", "eval_utils"))
sys.path.insert(0, _ANALYSIS_EVAL_UTILS)
from compare_lca_partitions import (
    LCA_COL,
    both_purities,
    both_strict,
    load,
)  # noqa: E402

CSV_COLUMNS = [
    "index",
    "category",
    "issue_summary",
    "result",
    "details",
    "root_cause",
    "recommended_fix",
]

GOLD_FILENAME = "semantic_purity_gold_diff_stats.csv"
CANDIDATE_RELPATH = os.path.join("tracediff_output", "diff_stats.csv")


def _write(results_path: str, rows: list[dict]) -> None:
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run(output_dir: str, reference_dir: str, results_path: str) -> list[dict]:
    gold_path = os.path.join(reference_dir, GOLD_FILENAME)
    if not os.path.isfile(gold_path):
        # No gold reference here; self-gate to a no-op.
        _write(results_path, [])
        return []

    candidate_path = os.path.join(output_dir, CANDIDATE_RELPATH)
    if not os.path.isfile(candidate_path):
        rows = [
            {
                "index": "semantic_purity_1",
                "category": "Quality",
                "issue_summary": "LCA-partition purity vs with-capture gold",
                "result": "FAIL",
                "details": f"Candidate semantic diff-stats not found: {candidate_path}",
                "root_cause": "pipeline",
                "recommended_fix": "Semantic workflow did not produce tracediff_output/diff_stats.csv; check the run log",
            }
        ]
        _write(results_path, rows)
        return rows

    gold = load(gold_path)
    cand = load(candidate_path)

    merged = (
        gold[["key", LCA_COL]]
        .rename(columns={LCA_COL: "lca_gold"})
        .merge(
            cand[["key", LCA_COL]].rename(columns={LCA_COL: "lca_cand"}),
            on="key",
        )
    )

    if merged.empty:
        rows = [
            {
                "index": "semantic_purity_1",
                "category": "Quality",
                "issue_summary": "LCA-partition purity vs with-capture gold",
                "result": "FAIL",
                "details": "No matched (source, gpu_op_uid) keys between gold and candidate",
                "root_cause": "data",
                "recommended_fix": "Check that candidate and gold were generated from the same DECODE trace pair",
            }
        ]
        _write(results_path, rows)
        return rows

    gold_lbl = merged["lca_gold"].to_numpy()
    cand_lbl = merged["lca_cand"].to_numpy()
    fwd, rev = both_purities(gold_lbl, cand_lbl)
    sfwd, srev = both_strict(gold_lbl, cand_lbl)

    # Informational only -- see module docstring. The regression gate is
    # applied post-hoc, across repeats, by semantic_purity_aggregate.py.
    details = (
        f"matched={len(merged)} forward_purity={fwd:.4f} reverse_purity={rev:.4f} "
        f"strict_forward={sfwd:.4f} strict_reverse={srev:.4f} "
        f"shared_buckets={merged['lca_cand'].nunique()} "
        f"(informational only -- see semantic_purity_aggregate.py for the actual gate)"
    )
    rows = [
        {
            "index": "semantic_purity_1",
            "category": "Quality",
            "issue_summary": "LCA-partition purity vs with-capture gold",
            "result": "PASS",
            "details": details,
            "root_cause": "",
            "recommended_fix": "",
        }
    ]
    _write(results_path, rows)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="LCA-partition purity vs with-capture gold"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    rows = run(args.output_dir, args.reference_dir, args.results)
    if not rows:
        sys.exit(0)
    passed = sum(1 for r in rows if r["result"] == "PASS")
    sys.exit(0 if passed == len(rows) else 1)


if __name__ == "__main__":
    main()
