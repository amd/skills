###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pythonic workflow evals.

Checks directory structure, file existence, and output completeness
against the category_manifest.json contract. Also includes deterministic
marker identification checks (top_ops, p_item, detail_estimate).

Supports both standalone and comparative analysis modes via --comparison-scope flag.
"""

import argparse
import csv
import json
import os
import re
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
_REQUIRED_MODEL_KEYS = {"model", "architecture", "scale", "precision"}

_MIN_REPORT_BYTES = 100
_GARBLED_THRESHOLD = 0.50

# Marker validation regexes — identical to those in MarkerValidator
# (TraceLens/Agent/Analysis/utils/validation_utils.py). Defined here directly
# because validation_utils.py has relative imports that prevent standalone loading.
_MV_BEGIN_RE = re.compile(r"<!--\s*impact-begin\s+([^>]*?)-->", re.DOTALL)
_MV_END_RE = re.compile(r"<!--\s*impact-end\s*-->")
_MV_KIND_ATTR_RE = re.compile(r"\bkind=(\w+)\b")
_MV_ATTR_RE = re.compile(r"\b(\w+)=([^\s]+)")
_TOP_OPS_ROW_RE = re.compile(r"<!--\s*top-ops-row\s+([^>]*?)-->")
_DETAIL_P_HEADER_RE = re.compile(r"^####\s+.+P(\d+):", re.MULTILINE)
_DETAIL_P_HEADER_FALLBACK_RE = re.compile(r"^(?:####|###)\s+.+P(\d+):", re.MULTILINE)
_REASONING_CANDIDATE_RE = re.compile(
    r"<!--\s*reasoning-candidate\s+tier=(\w+)\s+rank=(\d+)\s*-->"
)
_NOT_QUANTIFIABLE_RE = re.compile(
    r"not\s+quantifiable\s+from\s+trace\s+data", re.IGNORECASE
)


def _pre_check_gates(output_dir: str) -> str | None:
    """Return a failure reason if a hard pre-check gate trips, else None.

    Gates (applied in order, first failure wins):
      1. output_dir does not exist
      2. analysis.md missing or too small (< 100 bytes)
      3. analysis.md is garbled (> 50% non-ASCII or looks like JSON)
    """
    if not os.path.isdir(output_dir):
        return "output directory does not exist"

    report = os.path.join(output_dir, "analysis.md")
    if not os.path.isfile(report):
        return "analysis.md not found"
    size = os.path.getsize(report)
    if size < _MIN_REPORT_BYTES:
        return f"analysis.md too small ({size} bytes)"

    with open(report, encoding="utf-8", errors="replace") as f:
        content = f.read()
    if not content.strip():
        return "analysis.md is empty"

    non_ascii = sum(1 for c in content if ord(c) > 127)
    if len(content) > 0 and non_ascii / len(content) > _GARBLED_THRESHOLD:
        return (
            f"analysis.md appears garbled ({non_ascii}/{len(content)} non-ASCII chars)"
        )

    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return "analysis.md contains raw JSON instead of markdown"

    return None


def _load_manifest(output_dir: str) -> dict | None:
    path = os.path.join(output_dir, "category_data", "category_manifest.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _rebase_path(manifest_path: str, output_dir: str) -> str:
    """Rebase an absolute manifest path to the actual output_dir.

    The manifest stores absolute paths using the original output_dir.
    When evaluating against a different directory (e.g. analysis_output_ref),
    we extract the relative suffix and resolve it against the actual dir.
    """
    for marker in (
        "category_data/",
        "metadata/",
        "system_findings/",
        "category_findings/",
    ):
        idx = manifest_path.find(marker)
        if idx != -1:
            return os.path.join(output_dir, manifest_path[idx:])
    return manifest_path


def _check_directories(output_dir: str) -> tuple[str, str]:
    required = ["metadata", "category_data", "system_findings", "category_findings"]
    missing = [d for d in required if not os.path.isdir(os.path.join(output_dir, d))]
    if missing:
        return "FAIL", f"Missing directories: {', '.join(missing)}"
    return "PASS", ""


def _check_metadata_files(output_dir: str) -> tuple[str, str]:
    manifest = _load_manifest(output_dir)
    if manifest is None:
        return "FAIL", "category_manifest.json not found"
    missing = []
    for cat in manifest.get("categories", []):
        mf = cat.get("metadata_file", "")
        if mf and not os.path.isfile(_rebase_path(mf, output_dir)):
            missing.append(os.path.basename(mf))
    if missing:
        return "FAIL", f"Missing metadata files: {', '.join(missing)}"
    return "PASS", ""


def _check_model_info(output_dir: str) -> tuple[str, str]:
    path = os.path.join(output_dir, "metadata", "model_info.json")
    if not os.path.isfile(path):
        return "FAIL", "metadata/model_info.json not found"
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return "FAIL", f"Invalid JSON: {e}"
    missing = _REQUIRED_MODEL_KEYS - data.keys()
    if missing:
        return "FAIL", f"Missing keys: {', '.join(missing)}"
    empty = [k for k in _REQUIRED_MODEL_KEYS if not str(data[k]).strip()]
    if empty:
        return "FAIL", f"Empty values: {', '.join(empty)}"
    return "PASS", ""


def _check_unified_perf_report(
    output_dir: str, comparison_scope: str = "standalone"
) -> tuple[str, str]:
    if comparison_scope == "comparative":
        path = os.path.join(
            output_dir, "perf_report_trace1_csvs", "unified_perf_summary.csv"
        )
        label = "perf_report_trace1_csvs/unified_perf_summary.csv"
    else:
        path = os.path.join(output_dir, "perf_report_csvs", "unified_perf_summary.csv")
        label = "perf_report_csvs/unified_perf_summary.csv"
    if not os.path.isfile(path):
        return "FAIL", f"{label} not found"
    return "PASS", ""


def _check_tree_data_files(output_dir: str) -> tuple[str, str]:
    manifest = _load_manifest(output_dir)
    if manifest is None:
        return "FAIL", "category_manifest.json not found"
    missing = []
    for cat in manifest.get("categories", []):
        tf = cat.get("tree_data_file")
        if tf and not os.path.isfile(_rebase_path(tf, output_dir)):
            missing.append(os.path.basename(tf))
    if missing:
        return "FAIL", f"Missing tree data files: {', '.join(missing)}"
    return "PASS", ""


def _check_findings_exist(output_dir: str) -> tuple[str, str]:
    manifest = _load_manifest(output_dir)
    if manifest is None:
        return "FAIL", "category_manifest.json not found"
    gpu_util = manifest.get("gpu_utilization", {})
    idle_pct = gpu_util.get("idle_time_percent", 0)
    missing = []
    for cat in manifest.get("categories", []):
        name = cat["name"]
        if name == "cpu_idle" and idle_pct <= 15:
            continue
        tier = cat.get("tier", "compute_kernel")
        subdir = "system_findings" if tier == "system" else "category_findings"
        path = os.path.join(output_dir, subdir, f"{name}_findings.md")
        if not os.path.isfile(path):
            missing.append(f"{subdir}/{name}_findings.md")
    if missing:
        return "FAIL", f"Parallel subagent failed to invoke for: {', '.join(missing)}"
    return "PASS", ""


def _check_findings_placement(output_dir: str) -> tuple[str, str]:
    manifest = _load_manifest(output_dir)
    if manifest is None:
        return "FAIL", "category_manifest.json not found"
    gpu_util = manifest.get("gpu_utilization", {})
    idle_pct = gpu_util.get("idle_time_percent", 0)
    misplaced = []
    for cat in manifest.get("categories", []):
        name = cat["name"]
        if name == "cpu_idle" and idle_pct <= 15:
            continue
        tier = cat.get("tier", "compute_kernel")
        correct_dir = "system_findings" if tier == "system" else "category_findings"
        wrong_dir = "category_findings" if tier == "system" else "system_findings"
        correct = os.path.join(output_dir, correct_dir, f"{name}_findings.md")
        wrong = os.path.join(output_dir, wrong_dir, f"{name}_findings.md")
        if not os.path.isfile(correct) and os.path.isfile(wrong):
            misplaced.append(f"{name} in {wrong_dir}/ instead of {correct_dir}/")
    if misplaced:
        return "FAIL", f"Misplaced findings: {'; '.join(misplaced)}"
    return "PASS", ""


def _check_plot(output_dir: str) -> tuple[str, str]:
    plot_path = os.path.join(output_dir, "perf_improvement.png")
    plot_data_path = os.path.join(output_dir, "priority_data.json")

    if os.path.isfile(plot_path):
        return "PASS", ""

    if os.path.isfile(plot_data_path):
        with open(plot_data_path) as f:
            plot_data = json.load(f)
        recs = plot_data.get("recommendations", [])
        if not recs:
            return "PASS", "No kernel tuning recommendations — plot correctly skipped"

    cat_data_dir = os.path.join(output_dir, "category_data")
    if os.path.isdir(cat_data_dir):
        total_estimates = 0
        for f in os.listdir(cat_data_dir):
            if f.endswith("_metrics.json"):
                with open(os.path.join(cat_data_dir, f)) as fh:
                    metrics = json.load(fh)
                total_estimates += len(metrics.get("impact_estimates", []))
        if total_estimates == 0:
            return (
                "PASS",
                "No kernel tuning recommendations in any metrics — plot correctly skipped",
            )

    return "FAIL", "perf_improvement.png not found"


# Each entry: (summary, func, root_cause, recommended_fix, mode_aware)
# mode_aware=True  → called as func(output_dir, comparison_scope)
# mode_aware=False → called as func(output_dir)
EVAL_REGISTRY = [
    (
        "Directory structure created",
        _check_directories,
        "pipeline",
        "Re-run analysis pipeline from Step 1",
        False,
    ),
    (
        "Metadata files exist on disk",
        _check_metadata_files,
        "pipeline",
        "Re-run orchestrator_prepare.py to regenerate metadata",
        False,
    ),
    (
        "Model info JSON exists and valid",
        _check_model_info,
        "pipeline",
        "Check model-identification subagent; add fallback skeleton in prepare step",
        False,
    ),
    (
        "Unified perf. report exists",
        _check_unified_perf_report,
        "pipeline",
        "Re-run TraceLens_generate_perf_report_pytorch (Step 1)",
        True,
    ),
    (
        "Tree data files exist on disk",
        _check_tree_data_files,
        "pipeline",
        "Re-run orchestrator_prepare.py to regenerate tree data",
        False,
    ),
    (
        "Categorical findings .md files exist",
        _check_findings_exist,
        "pipeline",
        "Retry failed subagent(s) for missing findings files",
        False,
    ),
    (
        "All findings exist",
        _check_findings_placement,
        "pipeline",
        "Fix tier assignment in orchestrator_prepare.py",
        False,
    ),
    (
        "Plot generated on disk",
        _check_plot,
        "pipeline",
        "Check priority_data.json; add fallback empty plot_data.json",
        False,
    ),
]


# ---------------------------------------------------------------------------
# Helpers for per-item deterministic checks (evals 9-14, excl. 12 = LLM)
# ---------------------------------------------------------------------------


def _make_row(index, summary, result, details, root_cause, fix, category="Workflow"):
    return {
        "index": index,
        "category": category,
        "issue_summary": summary,
        "result": result,
        "details": details,
        "root_cause": root_cause,
        "recommended_fix": fix,
    }


def _read_report(output_dir):
    path = os.path.join(output_dir, "analysis.md")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _extract_section(content, header):
    """Extract text between a ## header and the next ## header or end of file."""
    escaped = re.escape(header)
    m = re.search(
        rf"^{escaped}[^\n]*\n(.*?)(?=^## |\Z)",
        content,
        re.MULTILINE | re.DOTALL,
    )
    return m.group(1).strip() if m else None


def _find_table_row(section_text, label_synonyms):
    """Find a markdown table row matching any synonym. Returns value cell or None."""
    for line in section_text.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if all(set(c) <= {"-", ":", " "} for c in cells):
            continue
        label = cells[0]
        for synonym in label_synonyms:
            if synonym.lower() in label.lower():
                return cells[1] if len(cells) > 1 else ""
    return None


def _find_table_row_all_cells(section_text, label_synonyms):
    """Find a markdown table row matching any synonym. Returns all cells or None."""
    for line in section_text.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        if all(set(c) <= {"-", ":", " "} for c in cells):
            continue
        label = cells[0]
        for synonym in label_synonyms:
            if synonym.lower() in label.lower():
                return cells  # [label, val1, val2, ...]
    return None


def _extract_percent(value_str):
    """Extract a percentage value from a string like '82.2%'."""
    m = re.search(r"([\d.]+)\s*%", value_str)
    return float(m.group(1)) if m else None


def _extract_numeric(value_str):
    """Extract first numeric value from a string (handles %, ms, ±, sign prefix)."""
    # Strip commas used as thousands separators
    s = value_str.replace(",", "")
    # Match optional leading sign then digits
    m = re.search(r"([-+]?\s*[\d.]+)", s)
    if m:
        return float(m.group(1).replace(" ", ""))
    return None


def _load_gpu_timeline(output_dir, comparison_scope="standalone", trace_num=1):
    """Load gpu_timeline.csv as {type_name: percent}."""
    if comparison_scope == "standalone":
        path = os.path.join(output_dir, "perf_report_csvs", "gpu_timeline.csv")
    else:
        path = os.path.join(
            output_dir, f"perf_report_trace{trace_num}_csvs", "gpu_timeline.csv"
        )
    if not os.path.isfile(path):
        return None
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data[row["type"]] = float(row["percent"])
            except (KeyError, ValueError):
                continue
    return data


def _extract_p_items(section_text):
    """Extract P-items from a section. Returns [(p_number, text_block), ...]."""
    p_pattern = re.compile(r"^### .+P(\d+):", re.MULTILINE)
    matches = list(p_pattern.finditer(section_text))
    items = []
    for i, m in enumerate(matches):
        pnum = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(section_text)
        items.append((pnum, section_text[start:end]))
    return items


def _check_p_item_fields(p_text, is_compute):
    """Check P-item for required bold fields. Returns list of missing field names."""
    missing = []
    if not re.search(r"\*\*Insight\*\*", p_text) and not re.search(
        r"\*\*Issue\*\*", p_text
    ):
        missing.append("**Insight** or **Issue**")
    if not re.search(r"\*\*Action\*\*", p_text):
        missing.append("**Action**")
    if is_compute and not re.search(r"\*\*Impact\*\*", p_text):
        missing.append("**Impact**")
    return missing


# ---------------------------------------------------------------------------
# Per-item check functions (evals 9, 10, 11, 13, 14)
# ---------------------------------------------------------------------------

_REPORT_HEADERS = {
    "executive_summary": "Executive Summary",
    "compute": "Compute Kernel Optimizations",
    "kernel_fusion": "Kernel Fusion Opportunities (Experimental)",
    "system": "System-Level Optimizations",
    "detailed": "Detailed Analysis",
    "appendix": "Appendix",
}


def _check_report_template(output_dir, comparison_scope="standalone"):
    """Eval 9 — per-header check for required section headers + table presence."""
    content = _read_report(output_dir)
    rows = []
    for key, header_text in _REPORT_HEADERS.items():
        found = bool(re.search(rf"^## {re.escape(header_text)}", content, re.MULTILINE))
        rows.append(
            _make_row(
                f"workflow_eval_9_{key}",
                f"Report header: {header_text}",
                "PASS" if found else "FAIL",
                "" if found else f"Missing '## {header_text}' header",
                "template" if not found else "",
                "Add missing section header to report" if not found else "",
            )
        )

    exec_section = _extract_section(content, "## Executive Summary")
    has_table = bool(re.search(r"\|.*\|", exec_section)) if exec_section else False
    rows.append(
        _make_row(
            "workflow_eval_9_metrics_table",
            "Metrics table in Executive Summary",
            "PASS" if has_table else "FAIL",
            "" if has_table else "No markdown table in Executive Summary",
            "template" if not has_table else "",
            "Add metrics table to Executive Summary" if not has_table else "",
        )
    )
    return rows


_EXEC_SUMMARY_CHECKS_STANDALONE = [
    ("total_time", ["Total Compute Time", "Total Time"], None, None),
    ("compute_pct", ["Computation", "Compute %", "Compute"], "computation_time", 1.0),
    ("idle_pct", ["Idle Time", "Idle %", "Idle"], "idle_time", 1.0),
    ("comm_pct", ["Exposed Communication", "Exposed Communication %"], None, None),
    ("bottleneck", ["Top Bottleneck Category", "Top Bottleneck"], None, None),
]

# comparative: (key, labels, csv_type_t1, tol_t1, csv_type_t2, tol_t2)
# None csv_type means no CSV cross-check for that trace
_EXEC_SUMMARY_CHECKS_COMPARATIVE = [
    ("total_time", ["Total Compute Time", "Total Time"], None, None, None, None),
    (
        "compute_pct",
        ["Computation", "Compute %", "Compute"],
        "computation_time",
        1.0,
        "computation_time",
        1.0,
    ),
    ("idle_pct", ["Idle Time", "Idle %", "Idle"], "idle_time", 1.0, "idle_time", 1.0),
    (
        "comm_pct",
        ["Exposed Communication", "Exposed Communication %"],
        None,
        None,
        None,
        None,
    ),
    (
        "bottleneck",
        ["Top Bottleneck Category", "Top Bottleneck"],
        None,
        None,
        None,
        None,
    ),
]


def _check_exec_summary(output_dir, comparison_scope="standalone"):
    """Eval 10 — per-row check for exec summary metrics + CSV cross-validation."""
    content = _read_report(output_dir)
    exec_section = _extract_section(content, "## Executive Summary")
    if not exec_section:
        return [
            _make_row(
                "workflow_eval_10",
                "Executive Summary has metrics table",
                "FAIL",
                "Executive Summary section not found",
                "template",
                "Add Executive Summary section",
            )
        ]

    if comparison_scope == "standalone":
        return _check_exec_summary_standalone(output_dir, exec_section)
    return _check_exec_summary_comparative(output_dir, exec_section)


def _check_exec_summary_standalone(output_dir, exec_section):
    csv_data = _load_gpu_timeline(output_dir, comparison_scope="standalone")
    rows = []
    for key, labels, csv_type, tolerance in _EXEC_SUMMARY_CHECKS_STANDALONE:
        value_str = _find_table_row(exec_section, labels)
        if value_str is None:
            rows.append(
                _make_row(
                    f"workflow_eval_10_{key}",
                    f"Exec Summary row: {labels[0]}",
                    "FAIL",
                    f"Row not found (looked for: {', '.join(labels)})",
                    "template",
                    "Add missing row to Executive Summary table",
                )
            )
            continue

        result, detail = "PASS", ""
        if csv_type and csv_data and tolerance:
            csv_val = csv_data.get(csv_type)
            if csv_val is not None:
                report_val = _extract_percent(value_str)
                if report_val is not None:
                    diff = abs(report_val - csv_val)
                    if diff > tolerance:
                        result = "FAIL"
                        detail = (
                            f"Value mismatch: report={report_val:.1f}%, "
                            f"csv={csv_val:.2f}%, diff={diff:.2f}% "
                            f"(tolerance={tolerance}%)"
                        )
                    else:
                        detail = (
                            f"Matches CSV (report={report_val:.1f}%, "
                            f"csv={csv_val:.2f}%)"
                        )

        rows.append(
            _make_row(
                f"workflow_eval_10_{key}",
                f"Exec Summary row: {labels[0]}",
                result,
                detail,
                "template" if result == "FAIL" else "",
                "Fix value in Executive Summary table" if result == "FAIL" else "",
            )
        )
    return rows


def _check_exec_summary_comparative(output_dir, exec_section):
    """Comparative exec summary: 4-column table (Metric | T1 | T2 | Difference)."""
    csv_t1 = _load_gpu_timeline(output_dir, comparison_scope="comparative", trace_num=1)
    csv_t2 = _load_gpu_timeline(output_dir, comparison_scope="comparative", trace_num=2)
    rows = []

    for (
        key,
        labels,
        csv_type_t1,
        tol_t1,
        csv_type_t2,
        tol_t2,
    ) in _EXEC_SUMMARY_CHECKS_COMPARATIVE:
        cells = _find_table_row_all_cells(exec_section, labels)
        if cells is None:
            rows.append(
                _make_row(
                    f"workflow_eval_10_{key}",
                    f"Exec Summary row: {labels[0]}",
                    "FAIL",
                    f"Row not found (looked for: {', '.join(labels)})",
                    "template",
                    "Add missing row to Executive Summary table",
                )
            )
            continue

        # Expect [label, t1_val, t2_val, diff_val]
        t1_str = cells[1] if len(cells) > 1 else ""
        t2_str = cells[2] if len(cells) > 2 else ""

        result, detail_parts = "PASS", []

        if csv_type_t1 and not t1_str:
            result = "FAIL"
            detail_parts.append("T1 cell missing or empty")
        if csv_type_t2 and not t2_str:
            result = "FAIL"
            detail_parts.append("T2 cell missing or empty")

        # Cross-check T1
        if csv_type_t1 and csv_t1 and tol_t1:
            csv_val = csv_t1.get(csv_type_t1)
            if csv_val is not None:
                report_val = _extract_percent(t1_str)
                if report_val is not None:
                    diff = abs(report_val - csv_val)
                    if diff > tol_t1:
                        result = "FAIL"
                        detail_parts.append(
                            f"T1 mismatch: report={report_val:.1f}%, csv={csv_val:.2f}%, diff={diff:.2f}%"
                        )
                    else:
                        detail_parts.append(
                            f"T1 OK ({report_val:.1f}% vs csv {csv_val:.2f}%)"
                        )

        # Cross-check T2
        if csv_type_t2 and csv_t2 and tol_t2:
            csv_val = csv_t2.get(csv_type_t2)
            if csv_val is not None:
                report_val = _extract_percent(t2_str)
                if report_val is not None:
                    diff = abs(report_val - csv_val)
                    if diff > tol_t2:
                        result = "FAIL"
                        detail_parts.append(
                            f"T2 mismatch: report={report_val:.1f}%, csv={csv_val:.2f}%, diff={diff:.2f}%"
                        )
                    else:
                        detail_parts.append(
                            f"T2 OK ({report_val:.1f}% vs csv {csv_val:.2f}%)"
                        )

        # Check Difference column exists
        if len(cells) < 4:
            result = "FAIL"
            detail_parts.append("Difference column missing (expected 4-column table)")
        else:
            # Arithmetic check: |Difference| ≈ |T1 - T2| within 1%
            diff_str = cells[3]
            t1_val = _extract_numeric(t1_str)
            t2_val = _extract_numeric(t2_str)
            diff_val = _extract_numeric(diff_str)
            if t1_val is not None and t2_val is not None and diff_val is not None:
                expected_magnitude = abs(t1_val - t2_val)
                denom = max(abs(t1_val), abs(t2_val), 1.0)
                rel_err = abs(abs(diff_val) - expected_magnitude) / denom
                if rel_err > 0.01:
                    result = "FAIL"
                    detail_parts.append(
                        f"Difference arithmetic error: reported={diff_str}, "
                        f"expected |diff|≈{expected_magnitude:.2f}, rel_err={rel_err:.3f}"
                    )

        rows.append(
            _make_row(
                f"workflow_eval_10_{key}",
                f"Exec Summary row: {labels[0]}",
                result,
                "; ".join(detail_parts),
                "template" if result == "FAIL" else "",
                "Fix value in Executive Summary table" if result == "FAIL" else "",
            )
        )
    return rows


def _check_issue_template(output_dir, comparison_scope="standalone"):
    """Eval 11 — per-P-item check for correct bold fields in each priority item."""
    content = _read_report(output_dir)
    compute_section = _extract_section(content, "## Compute Kernel Optimizations")
    system_section = _extract_section(content, "## System-Level Optimizations")

    rows = []
    if compute_section:
        for pnum, p_text in _extract_p_items(compute_section):
            missing = _check_p_item_fields(p_text, is_compute=True)
            result = "PASS" if not missing else "FAIL"
            rows.append(
                _make_row(
                    f"workflow_eval_11_compute_P{pnum}",
                    f"Compute P{pnum} template",
                    result,
                    f"Missing: {', '.join(missing)}" if missing else "",
                    "template" if result == "FAIL" else "",
                    "Add missing fields to P-item" if result == "FAIL" else "",
                )
            )

    if system_section:
        for pnum, p_text in _extract_p_items(system_section):
            missing = _check_p_item_fields(p_text, is_compute=False)
            result = "PASS" if not missing else "FAIL"
            rows.append(
                _make_row(
                    f"workflow_eval_11_system_P{pnum}",
                    f"System P{pnum} template",
                    result,
                    f"Missing: {', '.join(missing)}" if missing else "",
                    "template" if result == "FAIL" else "",
                    "Add missing fields to P-item" if result == "FAIL" else "",
                )
            )

    if not rows:
        all_sections = (compute_section or "") + (system_section or "")
        intentionally_empty = (
            "No compute kernel optimization opportunities identified" in all_sections
            or "No system-level bottlenecks detected" in all_sections
        )
        rows.append(
            _make_row(
                "workflow_eval_11",
                "Issue Template rendering",
                "PASS" if intentionally_empty else "FAIL",
                "" if intentionally_empty else "No P-items found in report",
                "" if intentionally_empty else "template",
                "" if intentionally_empty else "Ensure report contains priority items",
            )
        )
    return rows


_MODEL_INFO_FIELDS = ["model", "architecture", "scale", "precision"]


def _check_model_id(output_dir, comparison_scope=None):
    """Eval 13 — per-field check for model_info.json values in Appendix."""
    model_info_path = os.path.join(output_dir, "metadata", "model_info.json")
    report_path = os.path.join(output_dir, "analysis.md")

    if not os.path.isfile(model_info_path):
        return [
            _make_row(
                "workflow_eval_13",
                "Model identification in report",
                "FAIL",
                "metadata/model_info.json not found",
                "pipeline",
                "Run model identification subagent",
            )
        ]
    if not os.path.isfile(report_path):
        return [
            _make_row(
                "workflow_eval_13",
                "Model identification in report",
                "FAIL",
                "analysis.md not found",
                "pipeline",
                "Re-run report generation",
            )
        ]

    try:
        with open(model_info_path) as f:
            model_info = json.load(f)
    except (json.JSONDecodeError, OSError):
        return [
            _make_row(
                "workflow_eval_13",
                "Model identification in report",
                "FAIL",
                "Invalid model_info.json",
                "pipeline",
                "Fix model identification subagent",
            )
        ]

    with open(report_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    appendix = _extract_section(content, "## Appendix")
    if not appendix:
        return [
            _make_row(
                "workflow_eval_13",
                "Model identification in report",
                "FAIL",
                "Appendix section not found in report",
                "template",
                "Add Appendix section to report",
            )
        ]

    appendix_lower = appendix.lower()
    rows = []
    for field in _MODEL_INFO_FIELDS:
        value = str(model_info.get(field, "")).strip()
        if not value or value.lower() == "cannot be inferred from trace":
            rows.append(
                _make_row(
                    f"workflow_eval_13_{field}",
                    f"Model ID: {field}",
                    "PASS",
                    f"Field not determined ('{value}') — skipped",
                    "",
                    "",
                )
            )
            continue

        found = value.lower() in appendix_lower
        rows.append(
            _make_row(
                f"workflow_eval_13_{field}",
                f"Model ID: {field}",
                "PASS" if found else "FAIL",
                "" if found else f"'{value}' not found in Appendix",
                "template" if not found else "",
                "Add model info field to Appendix" if not found else "",
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Marker Identification checks (deterministic, structural)
# ---------------------------------------------------------------------------


def _marker_attrs_from_inner(inner):
    """Parse key=value pairs from a marker's inner content."""
    return dict(_MV_ATTR_RE.findall(inner))


def _make_marker_row(index, summary, result, details, root_cause="", fix=""):
    return _make_row(
        index,
        summary,
        result,
        details,
        root_cause,
        fix,
        category="Marker Identification",
    )


def _check_marker_top_ops(output_dir, comparison_scope=None):
    """Marker eval 1: top_ops wrapper + inline top-ops-row markers."""
    content = _read_report(output_dir)
    if content is None:
        return [
            _make_marker_row(
                "marker_eval_1",
                "Top Operations markers (kind=top_ops)",
                "FAIL",
                "analysis.md not found",
                "pipeline",
                "Re-run report generation",
            )
        ]

    errors = []
    top_ops_begin = None
    for m in _MV_BEGIN_RE.finditer(content):
        inner = m.group(1)
        km = _MV_KIND_ATTR_RE.search(inner)
        if km and km.group(1) == "top_ops":
            top_ops_begin = m
            break

    if top_ops_begin is None:
        errors.append("No <!-- impact-begin kind=top_ops --> marker found")
    else:
        end_after = _MV_END_RE.search(content, top_ops_begin.end())
        if end_after is None:
            errors.append("kind=top_ops marker has no matching <!-- impact-end -->")
        else:
            block = content[top_ops_begin.end() : end_after.start()]
            row_markers = list(_TOP_OPS_ROW_RE.finditer(block))
            if not row_markers:
                errors.append(
                    "No <!-- top-ops-row ... --> markers found inside top_ops block"
                )
            else:
                for i, rm in enumerate(row_markers, start=1):
                    inner = rm.group(1)
                    attrs = _marker_attrs_from_inner(inner)
                    missing_attrs = []
                    if "low" not in attrs:
                        missing_attrs.append("low")
                    if "high" not in attrs:
                        missing_attrs.append("high")
                    if missing_attrs:
                        errors.append(
                            f"top-ops-row #{i} missing attributes: "
                            f"{', '.join(missing_attrs)}"
                        )

    result = "PASS" if not errors else "FAIL"
    details = "; ".join(errors) if errors else ""
    return [
        _make_marker_row(
            "marker_eval_1",
            "Top Operations markers (kind=top_ops)",
            result,
            details,
            "template" if result == "FAIL" else "",
            "Add or fix top_ops markers in analysis.md" if result == "FAIL" else "",
        )
    ]


def _check_marker_p_items(output_dir, comparison_scope=None):
    """Marker eval 2: kind=p_item markers for each compute P-item."""
    content = _read_report(output_dir)
    if content is None:
        return [
            _make_marker_row(
                "marker_eval_2",
                "P-item markers (kind=p_item)",
                "FAIL",
                "analysis.md not found",
                "pipeline",
                "Re-run report generation",
            )
        ]

    compute_section = _extract_section(content, "## Compute Kernel Optimizations")
    if compute_section is None:
        return [
            _make_marker_row(
                "marker_eval_2",
                "P-item markers (kind=p_item)",
                "FAIL",
                "Compute Kernel Optimizations section not found",
                "template",
                "Ensure report contains Compute Kernel Optimizations section",
            )
        ]

    p_items = _extract_p_items(compute_section)
    if not p_items:
        return [
            _make_marker_row(
                "marker_eval_2",
                "P-item markers (kind=p_item)",
                "FAIL",
                "No P-items found in Compute Kernel Optimizations",
                "template",
                "Ensure report contains P-items",
            )
        ]

    rows = []
    for pnum, p_text in p_items:
        errors = []
        p_item_markers = []
        for m in _MV_BEGIN_RE.finditer(p_text):
            inner = m.group(1)
            km = _MV_KIND_ATTR_RE.search(inner)
            if km and km.group(1) == "p_item":
                p_item_markers.append(inner)

        if not p_item_markers:
            errors.append("No <!-- impact-begin kind=p_item ... --> marker found")
        else:
            for inner in p_item_markers:
                attrs = _marker_attrs_from_inner(inner)
                required = ("category", "low", "mid", "high")
                missing = [a for a in required if a not in attrs]
                if missing:
                    errors.append(
                        f"kind=p_item marker missing attributes: "
                        f"{', '.join(missing)}"
                    )

            end_count = len(_MV_END_RE.findall(p_text))
            begin_count = len(_MV_BEGIN_RE.findall(p_text))
            if begin_count > end_count:
                errors.append("Unpaired impact-begin (missing impact-end)")

        result = "PASS" if not errors else "FAIL"
        details = "; ".join(errors) if errors else ""
        rows.append(
            _make_marker_row(
                f"marker_eval_2_P{pnum}",
                f"Compute P{pnum} impact marker (kind=p_item)",
                result,
                details,
                "template" if result == "FAIL" else "",
                f"Fix kind=p_item marker for P{pnum}" if result == "FAIL" else "",
            )
        )
    return rows


def _check_marker_detail_estimates(output_dir, comparison_scope=None):
    """Marker eval 3: kind=detail_estimate markers in Detailed Analysis."""
    content = _read_report(output_dir)
    if content is None:
        return [
            _make_marker_row(
                "marker_eval_3",
                "Detail estimate markers (kind=detail_estimate)",
                "FAIL",
                "analysis.md not found",
                "pipeline",
                "Re-run report generation",
            )
        ]

    detailed_section = _extract_section(content, "## Detailed Analysis")
    if detailed_section is None:
        return [
            _make_marker_row(
                "marker_eval_3",
                "Detail estimate markers (kind=detail_estimate)",
                "FAIL",
                "Detailed Analysis section not found",
                "template",
                "Ensure report contains Detailed Analysis section",
            )
        ]

    m = re.search(
        r"^### Compute Kernel Insights[^\n]*\n(.*?)(?=^### (?!.*P\d)|^## |\Z)",
        detailed_section,
        re.MULTILINE | re.DOTALL,
    )
    compute_subsection = m.group(0) if m else detailed_section

    matches = list(_DETAIL_P_HEADER_RE.finditer(compute_subsection))
    if not matches:
        matches = list(_DETAIL_P_HEADER_FALLBACK_RE.finditer(compute_subsection))

    if not matches:
        return [
            _make_marker_row(
                "marker_eval_3",
                "Detail estimate markers (kind=detail_estimate)",
                "FAIL",
                "No P-item headers found in Detailed Analysis",
                "template",
                "Ensure Detailed Analysis contains P-item sections",
            )
        ]

    rows = []
    for i, match in enumerate(matches):
        pnum = int(match.group(1))
        start = match.start()
        end = (
            matches[i + 1].start() if i + 1 < len(matches) else len(compute_subsection)
        )
        block = compute_subsection[start:end]

        errors = []
        detail_markers = []
        for bm in _MV_BEGIN_RE.finditer(block):
            inner = bm.group(1)
            km = _MV_KIND_ATTR_RE.search(inner)
            if km and km.group(1) == "detail_estimate":
                detail_markers.append(inner)

        has_not_quantifiable = bool(_NOT_QUANTIFIABLE_RE.search(block))

        if not detail_markers and not has_not_quantifiable:
            errors.append(
                "No <!-- impact-begin kind=detail_estimate ... --> marker "
                "and no 'not quantifiable from trace data' sentinel found"
            )
        elif detail_markers:
            for inner in detail_markers:
                attrs = _marker_attrs_from_inner(inner)
                required = ("low", "high")
                missing = [a for a in required if a not in attrs]
                if missing:
                    errors.append(
                        f"kind=detail_estimate marker missing attributes: "
                        f"{', '.join(missing)}"
                    )

            total_begin = len(_MV_BEGIN_RE.findall(block))
            end_count = len(_MV_END_RE.findall(block))
            if total_begin > end_count:
                errors.append("Unpaired impact-begin (missing impact-end)")

        result = "PASS" if not errors else "FAIL"
        details = "; ".join(errors) if errors else ""
        rows.append(
            _make_marker_row(
                f"marker_eval_3_P{pnum}",
                f"Detailed Analysis P{pnum} estimate marker (kind=detail_estimate)",
                result,
                details,
                "template" if result == "FAIL" else "",
                (
                    f"Fix kind=detail_estimate marker for P{pnum} in Detailed Analysis"
                    if result == "FAIL"
                    else ""
                ),
            )
        )
    return rows


def _extract_detailed_analysis_subsection(content, subsection_header):
    """Extract a ### subsection from ## Detailed Analysis."""
    da_match = re.search(
        r"^## Detailed Analysis[^\n]*\n(.*?)(?=^## |\Z)",
        content,
        re.MULTILINE | re.DOTALL,
    )
    if not da_match:
        return None
    da_text = da_match.group(0)
    sub_start = da_text.find(subsection_header)
    if sub_start < 0:
        return None
    sub_text = da_text[sub_start + len(subsection_header) :]
    next_h3 = re.search(r"^### ", sub_text, re.MULTILINE)
    next_h2 = re.search(r"^## ", sub_text, re.MULTILINE)
    ends = [pos.start() for pos in [next_h3, next_h2] if pos is not None]
    end = min(ends) if ends else len(sub_text)
    return sub_text[:end]


def _check_marker_reasoning_candidates(output_dir, comparison_scope=None):
    """Marker eval 4: reasoning-candidate markers match P-item headings per tier."""
    content = _read_report(output_dir)
    if content is None:
        return [
            _make_marker_row(
                "marker_eval_4",
                "Reasoning-candidate markers",
                "FAIL",
                "analysis.md not found",
                "pipeline",
                "Re-run report generation",
            )
        ]

    rows = []
    checks = [
        ("compute", "### Compute Kernel Insights"),
        ("fusion", "### Kernel Fusion Insights"),
        ("system", "### System-Level Insights"),
    ]
    for tier, header in checks:
        subsection = _extract_detailed_analysis_subsection(content, header)
        if subsection is None:
            continue
        n_headings = len(_DETAIL_P_HEADER_RE.findall(subsection))
        n_markers = sum(
            1
            for m in _REASONING_CANDIDATE_RE.finditer(subsection)
            if m.group(1).lower() == tier
        )
        if n_headings == 0:
            continue
        result = "PASS" if n_markers == n_headings else "FAIL"
        detail = (
            ""
            if result == "PASS"
            else (
                f"{n_headings} #### P<N>: headings but "
                f"{n_markers} reasoning-candidate tier={tier} markers"
            )
        )
        rows.append(
            _make_marker_row(
                f"marker_eval_4_{tier}",
                f"reasoning-candidate markers ({tier})",
                result,
                detail,
                "template" if result == "FAIL" else "",
                (
                    f"Add <!-- reasoning-candidate tier={tier} rank=N --> before each "
                    f"#### P<N>: heading in {header}"
                    if result == "FAIL"
                    else ""
                ),
            )
        )

    if not rows:
        rows.append(
            _make_marker_row(
                "marker_eval_4",
                "Reasoning-candidate markers",
                "PASS",
                "No Detailed Analysis subsections with P-items found",
                "",
                "",
            )
        )
    return rows


_MULTI_EVAL_CHECKS = [
    _check_report_template,
    _check_exec_summary,
    _check_issue_template,
    _check_model_id,
    _check_marker_top_ops,
    _check_marker_p_items,
    _check_marker_detail_estimates,
    _check_marker_reasoning_candidates,
]

_GATE_FAIL_NEW_EVALS = [
    ("workflow_eval_9", "Report Template Rendering"),
    ("workflow_eval_10", "Executive Summary has metrics table"),
    ("workflow_eval_11", "Issue Template rendering"),
    ("workflow_eval_13", "Model identification in report"),
    ("marker_eval_1", "Top Operations markers (kind=top_ops)"),
    ("marker_eval_2", "P-item markers (kind=p_item)"),
    ("marker_eval_3", "Detail estimate markers (kind=detail_estimate)"),
    ("marker_eval_4", "Reasoning-candidate markers"),
]


def run(
    output_dir: str, results_path: str, comparison_scope: str = "standalone"
) -> list[dict]:
    rows = []

    gate_fail = _pre_check_gates(output_dir)
    if gate_fail is not None:
        for i, (summary, _func, rc, fix, _mode_aware) in enumerate(
            EVAL_REGISTRY, start=1
        ):
            rows.append(
                {
                    "index": f"workflow_eval_{i}",
                    "category": "Workflow",
                    "issue_summary": summary,
                    "result": "FAIL",
                    "details": f"Pre-check gate: {gate_fail}",
                    "root_cause": "pipeline",
                    "recommended_fix": "Fix pre-check gate failure before running evals",
                }
            )
        for idx, summary in _GATE_FAIL_NEW_EVALS:
            rows.append(
                {
                    "index": idx,
                    "category": "Workflow",
                    "issue_summary": summary,
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

    for i, (summary, func, rc, fix, mode_aware) in enumerate(EVAL_REGISTRY, start=1):
        result, details = (
            func(output_dir, comparison_scope) if mode_aware else func(output_dir)
        )
        rows.append(
            {
                "index": f"workflow_eval_{i}",
                "category": "Workflow",
                "issue_summary": summary,
                "result": result,
                "details": details,
                "root_cause": rc if result == "FAIL" else "",
                "recommended_fix": fix if result == "FAIL" else "",
            }
        )

    for check_fn in _MULTI_EVAL_CHECKS:
        rows.extend(check_fn(output_dir, comparison_scope))

    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Pythonic workflow evals")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument(
        "--comparison-scope",
        choices=["standalone", "comparative"],
        default="standalone",
        help="Analysis comparison scope (default: standalone)",
    )
    args = parser.parse_args()

    rows = run(args.output_dir, args.results, args.comparison_scope)
    passed = sum(1 for r in rows if r["result"] == "PASS")
    sys.exit(0 if passed == len(rows) else 1)


if __name__ == "__main__":
    main()
