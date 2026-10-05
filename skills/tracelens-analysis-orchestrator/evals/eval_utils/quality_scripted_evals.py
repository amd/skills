###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pythonic quality evals.

Compares generated perf report CSVs against reference CSVs
with numeric tolerance.
"""

import argparse
import csv
import os
import re
import sys

import pandas as pd

CSV_COLUMNS = [
    "index",
    "category",
    "issue_summary",
    "result",
    "details",
    "root_cause",
    "recommended_fix",
]
NUMERIC_TOLERANCE = 0.01
ABS_TOLERANCE = 0.05
OPTIONAL_COLUMN_PREFIXES = ("Pct Roofline", "Roofline Time")
OPTIONAL_COLUMNS = {"num_kernels"}

_NUMPY_TYPE_RE = re.compile(r"np\.\w+\(([^)]+)\)")


def _normalize_numpy_reprs(s: str) -> str:
    """Strip numpy type wrappers so np.int64(135) becomes 135, etc."""
    return _NUMPY_TYPE_RE.sub(r"\1", s)


def _check_csv_alignment_dirs(gen_dir: str, ref_dir: str) -> tuple[str, str]:
    """Compare all CSVs in gen_dir against ref_dir. Returns (result, details)."""
    if not os.path.isdir(ref_dir):
        return "FAIL", f"Reference CSV directory not found: {ref_dir}"
    if not os.path.isdir(gen_dir):
        return "FAIL", f"Generated CSV directory not found: {gen_dir}"

    ref_files = {f for f in os.listdir(ref_dir) if f.endswith(".csv")}
    if not ref_files:
        return "FAIL", "No reference CSVs found"

    mismatches = []
    for fname in sorted(ref_files):
        gen_path = os.path.join(gen_dir, fname)
        if not os.path.isfile(gen_path):
            mismatches.append(f"{fname}: missing")
            continue

        try:
            ref_df = pd.read_csv(os.path.join(ref_dir, fname), on_bad_lines="warn")
            gen_df = pd.read_csv(gen_path, on_bad_lines="warn")
        except Exception as exc:
            mismatches.append(f"{fname}: parse error: {exc}")
            continue

        missing_cols = set(ref_df.columns) - set(gen_df.columns)
        missing_cols = {
            c
            for c in missing_cols
            if not any(c.startswith(p) for p in OPTIONAL_COLUMN_PREFIXES)
            and c not in OPTIONAL_COLUMNS
        }
        if missing_cols:
            mismatches.append(f"{fname}: missing required columns: {missing_cols}")
            continue
        compare_cols = [c for c in ref_df.columns if c in gen_df.columns]
        gen_df = gen_df[compare_cols]
        if len(ref_df) != len(gen_df):
            mismatches.append(f"{fname}: row count {len(gen_df)} vs ref {len(ref_df)}")
            continue

        for col in compare_cols:
            if pd.api.types.is_bool_dtype(ref_df[col]) or pd.api.types.is_bool_dtype(
                gen_df[col]
            ):
                if not ref_df[col].equals(gen_df[col]):
                    n_diff = (ref_df[col] != gen_df[col]).sum()
                    mismatches.append(f"{fname}:{col} {n_diff} bool value(s) differ")
            elif pd.api.types.is_numeric_dtype(ref_df[col]):
                if not ref_df[col].equals(gen_df[col]):
                    diff = (ref_df[col] - gen_df[col]).abs()
                    denom = ref_df[col].abs().replace(0, 1)
                    rel_diff = (diff / denom).max()
                    abs_diff = diff.max()
                    if rel_diff > NUMERIC_TOLERANCE and abs_diff > ABS_TOLERANCE:
                        mismatches.append(
                            f"{fname}:{col} max relative diff {rel_diff:.4f}"
                        )
            else:
                ref_norm = (
                    ref_df[col].fillna("").astype(str).map(_normalize_numpy_reprs)
                )
                gen_norm = (
                    gen_df[col].fillna("").astype(str).map(_normalize_numpy_reprs)
                )
                mask = ref_norm != gen_norm
                if mask.any():
                    rows = list(mask[mask].index[:3])
                    mismatches.append(f"{fname}:{col} differs at rows {rows}")

    if mismatches:
        return "FAIL", "; ".join(mismatches[:5])
    return "PASS", ""


def _pre_check_gates(
    output_dir: str, reference_dir: str, comparison_scope: str
) -> str | None:
    """Return a failure reason if a hard pre-check gate trips, else None."""
    if not os.path.isdir(output_dir):
        return "generated output directory does not exist"
    if not os.path.isdir(reference_dir):
        return "reference directory does not exist"

    if comparison_scope == "comparative":
        csv_dir = "perf_report_trace1_csvs"
    else:
        csv_dir = "perf_report_csvs"

    if not os.path.isdir(os.path.join(output_dir, csv_dir)):
        return f"generated {csv_dir}/ directory not found"
    if not os.path.isdir(os.path.join(reference_dir, csv_dir)):
        return f"reference {csv_dir}/ directory not found"

    if comparison_scope == "comparative":
        if not os.path.isdir(os.path.join(output_dir, "perf_report_trace2_csvs")):
            return "generated perf_report_trace2_csvs/ directory not found"
        if not os.path.isdir(os.path.join(reference_dir, "perf_report_trace2_csvs")):
            return "reference perf_report_trace2_csvs/ directory not found"

    return None


def run(
    output_dir: str,
    reference_dir: str,
    results_path: str,
    comparison_scope: str = "standalone",
) -> list[dict]:
    rows = []

    gate_fail = _pre_check_gates(output_dir, reference_dir, comparison_scope)
    if gate_fail is not None:
        index = (
            "quality_eval_1"
            if comparison_scope == "standalone"
            else "quality_eval_1_trace1"
        )
        rows.append(
            {
                "index": index,
                "category": "Quality",
                "issue_summary": "TraceLens Perf report CSVs alignment",
                "result": "FAIL",
                "details": f"Pre-check gate: {gate_fail}",
                "root_cause": "pipeline",
                "recommended_fix": "Fix pre-check gate failure before running evals",
            }
        )
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return rows

    if comparison_scope == "standalone":
        gen_dir = os.path.join(output_dir, "perf_report_csvs")
        ref_dir = os.path.join(reference_dir, "perf_report_csvs")
        result, details = _check_csv_alignment_dirs(gen_dir, ref_dir)
        rows.append(
            {
                "index": "quality_eval_1",
                "category": "Quality",
                "issue_summary": "TraceLens Perf report CSVs alignment",
                "result": result,
                "details": details,
                "root_cause": "data" if result == "FAIL" else "",
                "recommended_fix": (
                    "Regenerate golden refs with current TraceLens version"
                    if result == "FAIL"
                    else ""
                ),
            }
        )
    else:
        # Comparative: check trace1 and trace2 CSVs separately
        for trace_num in (1, 2):
            gen_dir = os.path.join(output_dir, f"perf_report_trace{trace_num}_csvs")
            ref_dir = os.path.join(reference_dir, f"perf_report_trace{trace_num}_csvs")
            result, details = _check_csv_alignment_dirs(gen_dir, ref_dir)
            rows.append(
                {
                    "index": f"quality_eval_1_trace{trace_num}",
                    "category": "Quality",
                    "issue_summary": f"TraceLens Perf report CSVs alignment (trace{trace_num})",
                    "result": result,
                    "details": details,
                    "root_cause": "data" if result == "FAIL" else "",
                    "recommended_fix": (
                        "Regenerate golden refs with current TraceLens version"
                        if result == "FAIL"
                        else ""
                    ),
                }
            )

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Pythonic quality evals")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument(
        "--comparison-scope",
        choices=["standalone", "comparative"],
        default="standalone",
        help="Analysis mode (default: standalone)",
    )
    args = parser.parse_args()

    rows = run(args.output_dir, args.reference_dir, args.results, args.comparison_scope)
    passed = sum(1 for r in rows if r["result"] == "PASS")
    sys.exit(0 if passed == len(rows) else 1)


if __name__ == "__main__":
    main()
