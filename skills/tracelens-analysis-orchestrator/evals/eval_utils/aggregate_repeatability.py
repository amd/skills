#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Aggregate repeatability test results from eval_summary.csv and analysis_stream.ndjson."""

import csv
import json
import os
import re
from collections import defaultdict

from TraceLens.PerfModel.utils import optional_int

RESULTS_ROOT = os.environ.get(
    "RESULTS_ROOT",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "repeatability_results",
    ),
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
)


def find_runs(results_root):
    runs = []
    for trace_id in sorted(os.listdir(results_root)):
        trace_dir = os.path.join(results_root, trace_id)
        if not os.path.isdir(trace_dir):
            continue
        for entry in sorted(os.listdir(trace_dir)):
            m = re.match(r"run_(\d+)$", entry)
            if m:
                runs.append((trace_id, int(m.group(1)), os.path.join(trace_dir, entry)))
    return runs


def aggregate_eval_summaries(runs):
    rows = []
    for trace_id, run_num, run_dir in runs:
        csv_path = os.path.join(run_dir, "eval_summary.csv")
        if not os.path.isfile(csv_path):
            rows.append(
                {
                    "trace_id": trace_id,
                    "run_id": run_num,
                    "eval_index": "MISSING",
                    "eval_category": "",
                    "issue_summary": "eval_summary.csv not found",
                    "result": "MISSING",
                    "details": "",
                }
            )
            continue
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "trace_id": trace_id,
                        "run_id": run_num,
                        "eval_index": row.get("index", ""),
                        "eval_category": row.get("category", ""),
                        "issue_summary": row.get("issue_summary", ""),
                        "result": row.get("result", ""),
                        "details": row.get("details", ""),
                        "root_cause": row.get("root_cause", ""),
                        "recommended_fix": row.get("recommended_fix", ""),
                    }
                )
    return rows


def _classify_stability(pass_count: int, total: int) -> str:
    """Classify eval stability across repeated runs.

    Returns one of:
      STABLE_PASS  - all runs passed
      FLAKY_PASS   - >50% passed but not all
      FLAKY_FAIL   - >0% passed but <=50%
      STABLE_FAIL  - all runs failed
    """
    if total == 0:
        return "N/A"
    rate = pass_count / total
    if rate == 1.0:
        return "STABLE_PASS"
    elif rate > 0.5:
        return "FLAKY_PASS"
    elif rate > 0.0:
        return "FLAKY_FAIL"
    else:
        return "STABLE_FAIL"


def build_pass_rate_summary(rows):
    counts = defaultdict(lambda: {"pass": 0, "fail": 0, "total": 0})
    trace_totals = defaultdict(lambda: {"pass": 0, "fail": 0, "total": 0})
    for r in rows:
        if r["result"] == "MISSING":
            continue
        key = (r["trace_id"], r["eval_index"])
        counts[key]["total"] += 1
        trace_totals[r["trace_id"]]["total"] += 1
        if r["result"] == "PASS":
            counts[key]["pass"] += 1
            trace_totals[r["trace_id"]]["pass"] += 1
        else:
            counts[key]["fail"] += 1
            trace_totals[r["trace_id"]]["fail"] += 1
    all_evals = sorted({k[1] for k in counts})
    all_traces = sorted({k[0] for k in counts})
    summary_rows = []
    for trace_id in all_traces:
        row = {"trace_id": trace_id}
        for ev in all_evals:
            c = counts.get((trace_id, ev))
            row[ev] = (
                "{}/{}".format(c["pass"], c["total"]) if c and c["total"] > 0 else "N/A"
            )
        tt = trace_totals[trace_id]
        row["overall_pass_rate"] = (
            "{}/{} ({:.0f}%)".format(
                tt["pass"], tt["total"], 100 * tt["pass"] / tt["total"]
            )
            if tt["total"] > 0
            else "N/A"
        )
        summary_rows.append(row)
    return summary_rows, ["trace_id"] + all_evals + ["overall_pass_rate"]


def build_stability_summary(rows):
    """Build a per-(trace, eval) stability classification table."""
    counts = defaultdict(lambda: {"pass": 0, "total": 0})
    for r in rows:
        if r["result"] == "MISSING":
            continue
        key = (r["trace_id"], r["eval_index"])
        counts[key]["total"] += 1
        if r["result"] == "PASS":
            counts[key]["pass"] += 1

    all_evals = sorted({k[1] for k in counts})
    all_traces = sorted({k[0] for k in counts})
    stability_rows = []
    class_totals = defaultdict(int)
    for trace_id in all_traces:
        row = {"trace_id": trace_id}
        for ev in all_evals:
            c = counts.get((trace_id, ev))
            if c and c["total"] > 0:
                label = _classify_stability(c["pass"], c["total"])
                row[ev] = label
                class_totals[label] += 1
            else:
                row[ev] = "N/A"
        stability_rows.append(row)
    return stability_rows, ["trace_id"] + all_evals, dict(class_totals)


def _extract_tool_stdout(result):
    """Pull stdout text from pi-style tool_execution_end result payloads."""
    if not result:
        return ""
    parts = []
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "".join(parts)


def _accumulate_usage(usage, totals):
    """Sum token usage fields across pi turn records."""
    if not usage:
        return
    totals["input"] += (
        usage.get("inputTokens") or usage.get("input_tokens") or usage.get("input") or 0
    )
    totals["output"] += (
        usage.get("outputTokens")
        or usage.get("output_tokens")
        or usage.get("output")
        or 0
    )
    totals["cache_read"] += (
        usage.get("cacheReadTokens")
        or usage.get("cache_read_input_tokens")
        or usage.get("cacheRead")
        or usage.get("cache_read")
        or 0
    )


def _detect_steps_from_shell(cmd, stdout, steps):
    if "generate_perf_report" in cmd:
        steps.add("step1_perf_report")
    if "orchestrator_prepare" in cmd:
        steps.add("step2_5_prepare")
    if "category_manifest" in stdout or "category_manifest" in cmd:
        steps.add("step2_5_prepare")
    if "_findings.md" in cmd or "_findings.md" in stdout:
        steps.add("step7_subagent_findings")
    if "multi_kernel_findings" in stdout or "cpu_idle" in cmd:
        steps.add("step6_system_analysis")
    if "analysis.md" in cmd:
        steps.add("step11_report")


def parse_ndjson_stream(ndjson_path):
    diag = {
        "outcome": "unknown",
        "duration_ms": "",
        "input_tokens": "",
        "output_tokens": "",
        "cache_read_tokens": "",
        "turns": 0,
        "tool_calls": 0,
        "report_written": False,
        "report_headers": "",
        "last_step_reached": "none",
    }
    if not os.path.isfile(ndjson_path):
        diag["outcome"] = "missing_file"
        return diag
    if os.path.getsize(ndjson_path) < 100:
        with open(ndjson_path) as f:
            raw = f.read().strip()
        if "unavailable" in raw.lower() or "service unavailable" in raw.lower():
            diag["outcome"] = "agent_cli_unavailable"
        elif "error" in raw.lower():
            diag["outcome"] = "agent_cli_error"
        else:
            diag["outcome"] = "empty_or_minimal"
        return diag

    turn_ids, tool_call_count, result_record, steps_reached, report_content = (
        set(),
        0,
        None,
        set(),
        None,
    )
    agent_end_record = None
    pi_turn_count = 0
    pi_tool_calls = 0
    first_ts = None
    last_ts = None
    token_totals = {"input": 0, "output": 0, "cache_read": 0}

    with open(ndjson_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                if (
                    "unavailable" in line.lower()
                    or "service unavailable" in line.lower()
                ):
                    diag["outcome"] = "agent_cli_unavailable"
                    return diag
                continue
            rec_type = rec.get("type", "")
            if rec_type == "result":
                result_record = rec
            elif rec_type == "agent_end":
                agent_end_record = rec
            elif rec_type == "turn_end":
                pi_turn_count += 1
                msg = rec.get("message") or {}
                ts = msg.get("timestamp")
                if ts is not None:
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts
                _accumulate_usage(msg.get("usage") or {}, token_totals)
            elif rec_type == "tool_execution_start":
                pi_tool_calls += 1
                cmd = (rec.get("args") or {}).get("command", "")
                if cmd:
                    _detect_steps_from_shell(cmd, "", steps_reached)
                    c = _detect_report_write_from_command(
                        {"shellToolCall": {"args": {"command": cmd}}}
                    )
                    if c is not None:
                        report_content = c
            elif rec_type == "tool_execution_end":
                args = rec.get("args") or {}
                cmd = args.get("command", "")
                stdout = _extract_tool_stdout(rec.get("result"))
                if cmd or stdout:
                    _detect_steps_from_shell(cmd, stdout, steps_reached)
                    tc = {
                        "shellToolCall": {
                            "args": {"command": cmd},
                            "result": {"success": {"stdout": stdout}},
                        }
                    }
                    c = _detect_report_write(tc)
                    if c is not None:
                        report_content = c
            elif rec_type == "tool_call":
                mcid = rec.get("model_call_id", "")
                if mcid:
                    parts = mcid.rsplit("-", 2)
                    if len(parts) >= 3:
                        turn_id = optional_int(parts[-2])
                        if turn_id is not None:
                            turn_ids.add(turn_id)
                if rec.get("subtype") == "started":
                    tool_call_count += 1
                tc = rec.get("tool_call", {})
                if rec.get("subtype") == "completed":
                    _detect_steps(tc, steps_reached)
                    c = _detect_report_write(tc)
                    if c is not None:
                        report_content = c
                elif rec.get("subtype") == "started":
                    c = _detect_report_write_from_command(tc)
                    if c is not None:
                        report_content = c
            elif rec_type == "assistant":
                mcid = rec.get("model_call_id", "")
                if mcid:
                    parts = mcid.rsplit("-", 2)
                    if len(parts) >= 3:
                        turn_id = optional_int(parts[-2])
                        if turn_id is not None:
                            turn_ids.add(turn_id)

    diag["turns"] = pi_turn_count if pi_turn_count else len(turn_ids)
    diag["tool_calls"] = pi_tool_calls if pi_tool_calls else tool_call_count

    completion = result_record or agent_end_record
    if completion:
        diag["outcome"] = "error" if completion.get("is_error") else "success"
        duration_ms = completion.get("duration_ms")
        if duration_ms:
            diag["duration_ms"] = duration_ms
        usage = completion.get("usage") or {}
        if usage:
            diag["input_tokens"] = (
                usage.get("inputTokens")
                or usage.get("input_tokens")
                or usage.get("input")
                or ""
            )
            diag["output_tokens"] = (
                usage.get("outputTokens")
                or usage.get("output_tokens")
                or usage.get("output")
                or ""
            )
            diag["cache_read_tokens"] = (
                usage.get("cacheReadTokens")
                or usage.get("cacheReadInputTokens")
                or usage.get("cache_read_input_tokens")
                or usage.get("cacheRead")
                or usage.get("cache_read")
                or ""
            )
    elif diag["outcome"] == "unknown":
        diag["outcome"] = "no_result_record"

    if not diag["duration_ms"] and first_ts is not None and last_ts is not None:
        diag["duration_ms"] = last_ts - first_ts

    if not diag["input_tokens"] and token_totals["input"]:
        diag["input_tokens"] = token_totals["input"]
    if not diag["output_tokens"] and token_totals["output"]:
        diag["output_tokens"] = token_totals["output"]
    if not diag["cache_read_tokens"] and token_totals["cache_read"]:
        diag["cache_read_tokens"] = token_totals["cache_read"]

    if report_content is not None:
        diag["report_written"] = True
        headers = re.findall(r"^##\s+(.+)$", report_content, re.MULTILINE)
        diag["report_headers"] = " | ".join(headers) if headers else "(no headers)"
    diag["last_step_reached"] = _compute_last_step(steps_reached)
    return diag


def _detect_steps(tc, steps):
    shell = tc.get("shellToolCall", {})
    cmd = shell.get("args", {}).get("command", "")
    res = shell.get("result", {}).get("success", {})
    stdout = res.get("stdout", "")
    _detect_steps_from_shell(cmd, stdout, steps)
    rp = tc.get("readToolCall", {}).get("args", {}).get("path", "")
    if "_metrics.json" in rp:
        steps.add("step7_subagent_findings")
    if "category_manifest" in rp:
        steps.add("step2_5_prepare")


def _detect_report_write(tc):
    """Detect report write from completed tool_call (check stdout and interleavedOutput)."""
    shell = tc.get("shellToolCall", {})
    cmd = shell.get("args", {}).get("command", "")
    res = shell.get("result", {}).get("success", {})
    if "analysis.md" in cmd:
        for key in ("stdout", "interleavedOutput"):
            content = res.get(key, "")
            if content and len(content) > 100:
                return content
    wc = tc.get("writeToolCall", {})
    if wc and "analysis.md" in wc.get("args", {}).get("path", ""):
        content = wc.get("args", {}).get("content", "")
        if content:
            return content
    return None


def _detect_report_write_from_command(tc):
    """Extract report content from heredoc in the command string (started tool_call)."""
    shell = tc.get("shellToolCall", {})
    cmd = shell.get("args", {}).get("command", "")
    if "analysis.md" not in cmd:
        return None
    if "tee " not in cmd and "cat >" not in cmd:
        return None
    for marker in ("REPORT_EOF", "EOF", "HEREDOC", "MD_EOF", "ANALYSIS_EOF"):
        pattern = r"<<\s*'?" + re.escape(marker) + r"'?\n(.*?)\n" + re.escape(marker)
        m = re.search(pattern, cmd, re.DOTALL)
        if m:
            return m.group(1)
    eof_match = re.search(r"<<\s*'?(\w+)'?\n(.*)", cmd, re.DOTALL)
    if eof_match:
        delimiter = eof_match.group(1)
        rest = eof_match.group(2)
        end_pos = rest.rfind("\n" + delimiter)
        if end_pos > 0:
            return rest[:end_pos]
    return None


def _compute_last_step(steps):
    for key, label in [
        ("step11_report", "Step 11: Report"),
        ("step7_subagent_findings", "Step 7: Subagent findings"),
        ("step6_system_analysis", "Step 6: System analysis"),
        ("step2_5_prepare", "Steps 2-5: Prepare"),
        ("step1_perf_report", "Step 1: Perf report"),
    ]:
        if key in steps:
            return label
    return "none"


def build_run_level_summary(rows):
    """Build per-(trace, run) failure counts and flag catastrophic runs.

    A run is catastrophic when >50% of its checks fail, which typically
    indicates the agent crashed or produced no analysis.md rather than
    individual eval issues.
    """
    counts = defaultdict(lambda: {"pass": 0, "fail": 0, "total": 0})
    for r in rows:
        if r["result"] in ("MISSING", ""):
            continue
        key = (r["trace_id"], r["run_id"])
        counts[key]["total"] += 1
        if r["result"] == "PASS":
            counts[key]["pass"] += 1
        else:
            counts[key]["fail"] += 1

    run_rows = []
    for (trace_id, run_id), c in sorted(counts.items()):
        is_catastrophic = c["fail"] > c["total"] * 0.5 if c["total"] > 0 else False
        run_rows.append(
            {
                "trace_id": trace_id,
                "run_id": run_id,
                "pass": c["pass"],
                "fail": c["fail"],
                "total": c["total"],
                "is_catastrophic": is_catastrophic,
            }
        )
    return run_rows


def build_failure_nature_summary(rows, run_level_rows):
    """Classify total failures into stable vs flaky vs catastrophic.

    - catastrophic_pipeline: failures from runs where >50% of checks failed
    - stable: failures for evals that fail in every run for a given trace
    - flaky: everything else
    """
    catastrophic_runs = {
        (r["trace_id"], r["run_id"]) for r in run_level_rows if r["is_catastrophic"]
    }

    per_trace_runs = defaultdict(set)
    per_eval_fail_runs = defaultdict(set)
    for r in rows:
        if r["result"] in ("MISSING", ""):
            continue
        per_trace_runs[r["trace_id"]].add(r["run_id"])
        if r["result"] == "FAIL":
            per_eval_fail_runs[(r["trace_id"], r["issue_summary"])].add(r["run_id"])

    catastrophic_count = 0
    stable_count = 0
    flaky_count = 0

    for r in rows:
        if r["result"] != "FAIL":
            continue
        key = (r["trace_id"], r["run_id"])
        if key in catastrophic_runs:
            catastrophic_count += 1
        elif (
            per_eval_fail_runs[(r["trace_id"], r["issue_summary"])]
            == per_trace_runs[r["trace_id"]]
        ):
            stable_count += 1
        else:
            flaky_count += 1

    return {
        "catastrophic_pipeline": catastrophic_count,
        "stable": stable_count,
        "flaky": flaky_count,
        "total": catastrophic_count + stable_count + flaky_count,
    }


RUN_LEVEL_COLS = ["trace_id", "run_id", "pass", "fail", "total", "is_catastrophic"]


def aggregate_stream_diagnostics(runs):
    rows = []
    for trace_id, run_num, run_dir in runs:
        diag = parse_ndjson_stream(os.path.join(run_dir, "analysis_stream.ndjson"))
        diag["trace_id"] = trace_id
        diag["run_id"] = run_num
        rows.append(diag)
    return rows


STREAM_COLS = [
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
]
EVAL_COLS = [
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


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("  Written: {} ({} rows)".format(path, len(rows)))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Scanning: {}".format(RESULTS_ROOT))
    runs = find_runs(RESULTS_ROOT)
    print("Found {} run directories\n".format(len(runs)))

    print("Part A: Aggregating eval_summary.csv files...")
    eval_rows = aggregate_eval_summaries(runs)
    write_csv(os.path.join(OUTPUT_DIR, "aggregated_results.csv"), eval_rows, EVAL_COLS)
    summary_rows, summary_cols = build_pass_rate_summary(eval_rows)
    write_csv(
        os.path.join(OUTPUT_DIR, "pass_rate_summary.csv"), summary_rows, summary_cols
    )

    print("\nPart A2: Building run-level summary...")
    run_level_rows = build_run_level_summary(eval_rows)
    write_csv(
        os.path.join(OUTPUT_DIR, "run_level_summary.csv"),
        run_level_rows,
        RUN_LEVEL_COLS,
    )
    catastrophic = [r for r in run_level_rows if r["is_catastrophic"]]
    if catastrophic:
        print(
            "  Catastrophic runs (>50% fail): {}".format(
                ", ".join(
                    "{}/run_{}".format(r["trace_id"], r["run_id"]) for r in catastrophic
                )
            )
        )

    print("\nPart A3: Building failure nature summary...")
    failure_nature = build_failure_nature_summary(eval_rows, run_level_rows)
    write_csv(
        os.path.join(OUTPUT_DIR, "failure_nature_summary.csv"),
        [failure_nature],
        ["catastrophic_pipeline", "stable", "flaky", "total"],
    )
    print("  Failure nature: {}".format(failure_nature))

    print("\nPart A4: Building stability classification...")
    stability_rows, stability_cols, class_totals = build_stability_summary(eval_rows)
    write_csv(
        os.path.join(OUTPUT_DIR, "stability_summary.csv"),
        stability_rows,
        stability_cols,
    )
    print("  Stability breakdown: {}".format(class_totals))

    print("\nPart B: Parsing analysis_stream.ndjson files...")
    stream_rows = aggregate_stream_diagnostics(runs)
    write_csv(
        os.path.join(OUTPUT_DIR, "stream_diagnostics.csv"), stream_rows, STREAM_COLS
    )

    print("\n=== Summary ===")
    total_evals = sum(1 for r in eval_rows if r["result"] not in ("MISSING", ""))
    passed = sum(1 for r in eval_rows if r["result"] == "PASS")
    failed = sum(1 for r in eval_rows if r["result"] == "FAIL")
    missing = sum(1 for r in eval_rows if r["result"] == "MISSING")
    print(
        "Eval results: {} PASS, {} FAIL, {} MISSING (total rows: {})".format(
            passed, failed, missing, len(eval_rows)
        )
    )
    if total_evals > 0:
        print("Overall pass rate: {:.1f}%".format(100 * passed / total_evals))

    if class_totals:
        stable_pass = class_totals.get("STABLE_PASS", 0)
        flaky_pass = class_totals.get("FLAKY_PASS", 0)
        flaky_fail = class_totals.get("FLAKY_FAIL", 0)
        stable_fail = class_totals.get("STABLE_FAIL", 0)
        ct = stable_pass + flaky_pass + flaky_fail + stable_fail
        print(
            "Stability: {} STABLE_PASS, {} FLAKY_PASS, {} FLAKY_FAIL, {} STABLE_FAIL "
            "(of {} trace-eval pairs)".format(
                stable_pass, flaky_pass, flaky_fail, stable_fail, ct
            )
        )

    if failure_nature["total"] > 0:
        fn = failure_nature
        print(
            "Failure nature: {} catastrophic ({:.0f}%), {} stable ({:.0f}%), "
            "{} flaky ({:.0f}%) of {} total failures".format(
                fn["catastrophic_pipeline"],
                100 * fn["catastrophic_pipeline"] / fn["total"],
                fn["stable"],
                100 * fn["stable"] / fn["total"],
                fn["flaky"],
                100 * fn["flaky"] / fn["total"],
                fn["total"],
            )
        )
    if catastrophic:
        print(
            "Catastrophic runs: {} (agent produced no analysis.md, "
            "causing all evals to fail)".format(len(catastrophic))
        )

    outcomes = defaultdict(int)
    for r in stream_rows:
        outcomes[r["outcome"]] += 1
    print("\nStream outcomes: {}".format(dict(outcomes)))

    report_written = sum(1 for r in stream_rows if r["report_written"])
    print(
        "Runs with report written (detected in stream): {}/{}".format(
            report_written, len(stream_rows)
        )
    )

    step_counts = defaultdict(int)
    for r in stream_rows:
        step_counts[r["last_step_reached"]] += 1
    print("Last step reached: {}".format(dict(step_counts)))

    print("\nAll outputs in: {}".format(OUTPUT_DIR))


if __name__ == "__main__":
    main()
