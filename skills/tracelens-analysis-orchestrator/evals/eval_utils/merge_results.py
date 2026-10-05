###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Merge per-test-case eval CSVs into a single summary."""

import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge eval results")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    expected_csvs = [
        "workflow_scripted_results.csv",
        "workflow_llm_results.csv",
        "quality_scripted_results.csv",
        "quality_llm_results.csv",
    ]
    expected_cols = [
        "index",
        "category",
        "issue_summary",
        "result",
        "details",
        "root_cause",
        "recommended_fix",
    ]
    out = args.output or os.path.join(args.results_dir, "eval_summary.csv")

    frames = []
    missing_csvs = []
    csv_pattern = os.path.join(args.results_dir, "*_results.csv")
    found_files = {os.path.basename(p) for p in glob.glob(csv_pattern)}

    for csv_path in sorted(glob.glob(csv_pattern)):
        try:
            df = pd.read_csv(csv_path, on_bad_lines="warn")
        except Exception:
            try:
                df = pd.read_csv(
                    csv_path,
                    sep=None,
                    engine="python",
                    on_bad_lines="warn",
                )
            except Exception as exc:
                print(f"Warning: could not parse {csv_path}: {exc}")
                continue
        frames.append(df)

    for expected in expected_csvs:
        if expected not in found_files:
            missing_csvs.append(expected)

    if not frames:
        merged = pd.DataFrame(
            [
                {
                    "index": "merge_error",
                    "category": "Merge",
                    "issue_summary": "No eval results produced",
                    "result": "FAIL",
                    "details": f"Missing: {', '.join(missing_csvs) if missing_csvs else 'all result CSVs'}",
                    "root_cause": "pipeline",
                    "recommended_fix": "Re-run eval pipeline to produce result CSVs",
                }
            ]
        )
        merged.to_csv(out, index=False)
        print(f"No result CSVs found. Missing: {', '.join(missing_csvs)}")
        print(f"Partial summary written to {out}")
        return

    merged = pd.concat(frames, ignore_index=True)
    merged = merged[[c for c in expected_cols if c in merged.columns]]
    merged.to_csv(out, index=False)

    total = len(merged)
    passed = (merged["result"] == "PASS").sum()
    if missing_csvs:
        print(f"Warning: missing eval CSVs: {', '.join(missing_csvs)}")
    print(f"Summary: {passed}/{total} PASS  |  Written to {out}")


if __name__ == "__main__":
    main()
