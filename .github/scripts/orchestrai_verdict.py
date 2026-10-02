#!/usr/bin/env python3
"""Publish one GitHub verdict from a normalized OrchestrAI result manifest."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from orchestrai_logs import (
    behavioral_summary_markdown,
    public_behavioral_summary,
    public_manifest_log,
)

SAFE_STATUSES = {
    "passed",
    "failed",
    "error",
    "skipped",
    "cancelled",
    "aborted",
    "missing",
    "mock",
    "unknown",
}
SAFE_BEHAVIORAL_DIAGNOSTICS = {
    "Skillscope setup artifacts were missing on the test machine.",
    "The checked-out skills commit did not match the requested commit.",
    "LLM gateway configuration was unavailable on the test machine.",
    "The skills test dependencies could not be installed.",
    "An unprivileged Linux test account was unavailable.",
    "The Linux skills workspace could not be prepared.",
    "The Claude API preflight could not authenticate through the LLM gateway.",
    "The Claude API preflight timed out.",
    "The Claude API preflight could not reach the LLM gateway.",
    "The behavioral test timed out.",
    "Claude Code rejected the behavioral invocation.",
    "The Claude agent could not authenticate through the LLM gateway.",
    "The Claude agent was rate limited by the LLM gateway.",
    "The Claude service was temporarily unavailable.",
    "The requested Claude model was unavailable through the LLM gateway.",
    "The Claude agent exceeded the gateway context limit.",
    "The Claude agent lost connectivity to the LLM gateway.",
    "The Claude Code process crashed before returning a result.",
    "Claude Code could not run under the Linux test account.",
    "The Claude agent did not return a usable result.",
    "One or more behavioral expectations were not met.",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--skill", required=True)
    parser.add_argument("--os", choices=("Linux", "Windows"), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("test-results"))
    return parser


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            "the OrchestrAI controller did not publish a result manifest; "
            "inspect the matching operating-system controller job"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            "the OrchestrAI controller published an invalid result manifest"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError("OrchestrAI result manifest must be a JSON object")
    return value


def _safe_status(value: object) -> str:
    status = str(value or "unknown").strip().lower()
    return status if status in SAFE_STATUSES else "unknown"


def _safe_duration(value: object) -> str:
    duration = str(value or "").strip().lower()
    if re.fullmatch(
        r"\d+(?:\.\d+)?\s*(?:ms|s|sec|secs|seconds?|m|min|mins|minutes?)",
        duration,
    ):
        return duration
    return ""


def _safe_error(value: object) -> str:
    """Classify an item error without copying controller-owned text."""
    raw = str(value or "").strip()
    if raw in SAFE_BEHAVIORAL_DIAGNOSTICS:
        return raw
    lowered = raw.lower()
    if not lowered:
        return ""
    if "did not publish a result manifest" in lowered:
        return (
            "The OrchestrAI controller did not publish a result manifest; "
            "inspect the matching operating-system controller job."
        )
    if "invalid result manifest" in lowered or "invalid run object" in lowered:
        return "The OrchestrAI controller published an invalid result manifest."
    if ("portal" in lowered or "controller" in lowered) and (
        "timed out" in lowered
        or "timeout" in lowered
        or "unreachable" in lowered
        or "http" in lowered
    ):
        return "The OrchestrAI control plane was unreachable."
    if (
        "acquisition" in lowered or "machine" in lowered or "provision" in lowered
    ) and ("timed out" in lowered or "timeout" in lowered):
        return "Machine acquisition timed out."
    if "timed out" in lowered or "timeout" in lowered:
        return "The behavioral test timed out."
    if "offline" in lowered:
        return "The allocated actor went offline."
    if "launcher" in lowered and "unreachable" in lowered:
        return "The behavioral test launcher was unreachable."
    if "acquisition" in lowered or "no machine" in lowered:
        return "Machine acquisition failed."
    if (
        "before any requested test session" in lowered
        or "before the requested test session" in lowered
        or "before the requested test completed" in lowered
        or ("exactly one live session" in lowered and "infrastructure:" in lowered)
    ):
        if "before the requested test completed" in lowered:
            return (
                "OrchestrAI infrastructure ended before the requested test completed."
            )
        return "OrchestrAI infrastructure failed before the requested test session started."
    if "setup" in lowered or "install" in lowered or "dependency" in lowered:
        return "Behavioral test setup failed."
    if "cancel" in lowered or "abort" in lowered:
        return "The behavioral test was cancelled."
    if "did not pass" in lowered or "test failed" in lowered:
        return "The behavioral test failed."
    if "portal" in lowered or "controller" in lowered:
        return "The OrchestrAI controller failed; inspect the controller job."
    if "duplicate" in lowered:
        return "Duplicate OrchestrAI results were published for this skill and OS."
    if "exactly one" in lowered or "no per-skill result" in lowered:
        return "Exactly one OrchestrAI result was expected for this skill and OS."
    return "The behavioral result could not be verified."


def _safe_report_url(value: object) -> str:
    raw = str(value or "").strip()
    if (
        not raw
        or len(raw) > 2048
        or any(character.isspace() or character in '<>"' for character in raw)
    ):
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ""
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return ""
    return raw


def result_category(item: dict) -> str:
    status = _safe_status(item.get("status"))
    if status == "error" and (
        "result manifest" in _safe_error(item.get("error"))
        or "OrchestrAI result" in _safe_error(item.get("error"))
        or "result artifacts" in str(item.get("error") or "")
        or item.get("error") == "no per-skill result artifact was published"
    ):
        return "Reporting / result error"
    if status == "failed":
        if (
            _safe_error(item.get("error"))
            == "One or more behavioral expectations were not met."
        ):
            return "Behavioral failure"
        return "Test failure"
    return {
        "passed": "Passed",
        "mock": "Plan-only; no hardware test",
        "error": "Execution / infrastructure error",
        "skipped": "Skipped",
        "cancelled": "Cancelled",
        "aborted": "Aborted",
    }.get(status, "Missing / unverified result")


def log_coverage(item: dict) -> str:
    if _safe_status(item.get("status")) == "mock":
        return "Not run (plan-only)"
    if summary := public_behavioral_summary(item):
        return "Complete case counts" if summary["complete"] else "Partial case counts"
    if public_manifest_log(item):
        return "Grader lines; totals unavailable"
    if item.get("public_log_status") == "no_recognized_lines":
        return "Output received; grader lines filtered"
    return "Test output unavailable"


def public_controller_state(run: dict) -> dict:
    """Carry only fixed state enums, never controller identity or raw errors."""
    if not isinstance(run, dict):
        run = {}
    states = (
        "passed",
        "failed",
        "unstable",
        "error",
        "cancelled",
        "aborted",
        "running",
        "pending",
        "queued",
        "completed",
        "finished",
        "mock",
        "unknown",
    )
    return {
        "pipeline_status": run.get("pipeline_status")
        if run.get("pipeline_status") in states
        else "unknown",
        "tests_status": run.get("tests_status")
        if run.get("tests_status") in states
        else "unknown",
        "cleanup_status": run.get("cleanup_status")
        if run.get("cleanup_status") in ("confirmed", "unconfirmed", "not_needed")
        else "unknown",
    }


def _write_step_summary(item: dict, run: dict, *, ok: bool, mock: bool) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
    if not path:
        return
    status = _safe_status(item.get("status"))
    icon = "🧪" if mock else ("✅" if ok else "❌")
    skill = html.escape(str(item.get("skill") or ""))
    os_name = html.escape(str(item.get("os") or ""))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"### {icon} `{skill}` on {os_name}\n\n")
        handle.write("| | |\n|---|---|\n")
        handle.write(f"| Result | `{html.escape(status)}` |\n")
        handle.write(f"| Result type | {result_category(item)} |\n")
        handle.write(f"| Public log coverage | {log_coverage(item)} |\n")
        controller = public_controller_state(run)
        if not mock and controller["pipeline_status"] != "unknown":
            handle.write(f"| Pipeline snapshot | `{controller['pipeline_status']}` |\n")
        if not mock and controller["cleanup_status"] != "unknown":
            handle.write(f"| Parent termination | `{controller['cleanup_status']}` |\n")
        if duration := _safe_duration(item.get("duration")):
            handle.write(f"| Duration | `{html.escape(duration)}` |\n")
        if report_url := _safe_report_url(item.get("report_url")):
            handle.write(
                f"| ReportPortal | [View Results](<{html.escape(report_url)}>) |\n"
            )
        if error := _safe_error(item.get("error")):
            error = html.escape(error)
            handle.write(f"\n> {error}\n")
        if mock:
            handle.write("\nPlan validation only; no hardware test was executed.\n")
        elif report := behavioral_summary_markdown(item):
            handle.write("\n" + report)
        if lines := public_manifest_log(item):
            handle.write(
                "\n<details><summary>Sanitized grader output</summary>\n\n```text\n"
            )
            handle.write("\n".join(lines) + "\n```\n\n</details>\n")
        elif not mock:
            handle.write(
                f"\nSanitized grader output: {_log_unavailable_reason(item)}.\n"
            )
        if not mock:
            handle.write(
                "\nReportPortal requires AMD access. Pipeline status is a snapshot; confirmed parent termination is not a machine-release verification.\n"
            )


def _log_unavailable_reason(item: dict) -> str:
    if item.get("public_log_status") == "no_recognized_lines":
        return "test output was received, but no publishable grader lines were recognized; see ReportPortal"
    return "test output was unavailable; see full logs in ReportPortal"


def _write_job_log(item: dict, *, ok: bool, mock: bool) -> None:
    """Print normalized fields and revalidated, sanitized grader lines."""
    status = _safe_status(item.get("status"))
    skill = str(item.get("skill") or "")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", skill):
        skill = "unknown"
    os_name = str(item.get("os") or "")
    if os_name not in {"Linux", "Windows"}:
        os_name = "unknown"

    print("Behavioral result")
    print(f"  Skill: {skill}")
    print(f"  OS: {os_name}")
    print(f"  Result: {status}")
    if duration := _safe_duration(item.get("duration")):
        print(f"  Duration: {duration}")
    if error := _safe_error(item.get("error")):
        print(f"  Diagnostic: {error}")
    if report_url := _safe_report_url(item.get("report_url")):
        print(f"  Full logs (ReportPortal): {report_url}")
    elif not ok and not mock:
        print("  Full logs: unavailable; inspect the matching controller job")
    if not mock and (report := behavioral_summary_markdown(item)):
        # Put the overview and failed expectations outside the folded full
        # grader output so a failure is understandable without expanding it.
        print("\n" + report)
    if lines := public_manifest_log(item):
        print("::group::Sanitized grader output (full logs in ReportPortal)")
        for line in lines:
            print(line)
        print("::endgroup::")
    elif not mock:
        print(f"  Sanitized grader output: {_log_unavailable_reason(item)}")


def evaluate(manifest: dict, *, skill: str, os_name: str) -> tuple[dict, dict, bool]:
    items = manifest.get("items") or []
    matches = [
        item
        for item in items
        if isinstance(item, dict)
        and item.get("skill") == skill
        and item.get("os") == os_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one verdict for {skill} on {os_name}, found {len(matches)}"
        )
    item = matches[0]
    run = manifest.get("run") or {}
    if not isinstance(run, dict):
        raise ValueError("OrchestrAI result manifest has an invalid run object")
    status = _safe_status(item.get("status"))
    item = {**item, "status": status}
    return item, run, status in {"passed", "mock"}


def _summary_document(
    *, skill: str, os_name: str, item: dict[str, Any], run: dict, ok: bool
) -> dict:
    status = _safe_status(item.get("status"))
    mock = status == "mock"
    duration = _safe_duration(item.get("duration"))
    error = _safe_error(item.get("error"))
    report_url = _safe_report_url(item.get("report_url"))
    compact_item = {
        "skill": skill,
        "os": os_name,
        "status": status,
        "terminal": bool(item.get("terminal")),
    }
    document = {
        "skill": skill,
        "os": os_name,
        "status": status,
        "total_tests": 0 if mock else 1,
        "passed": 1 if ok and not mock else 0,
        "failed": 0 if ok else 1,
        "skipped": 0,
        "duration": duration,
        "error": error,
        "results": [compact_item],
        "controller": public_controller_state(run),
    }
    if report_url:
        document["report_url"] = report_url
    if lines := public_manifest_log(item):
        document["public_log"] = lines
        document["public_log_status"] = "available"
        if status != "mock" and (behavioral := public_behavioral_summary(item)):
            document["behavioral"] = behavioral
    elif item.get("public_log_status") in ("unavailable", "no_recognized_lines"):
        document["public_log_status"] = item["public_log_status"]
    return document


def main() -> int:
    args = _parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "summary.json"
    try:
        manifest = _load(args.results)
        item, run, ok = evaluate(manifest, skill=args.skill, os_name=args.os)
    except ValueError as exc:
        item = {
            "skill": args.skill,
            "os": args.os,
            "status": "error",
            "error": str(exc),
        }
        run = {}
        ok = False

    summary = _summary_document(
        skill=args.skill, os_name=args.os, item=item, run=run, ok=ok
    )
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if lines := public_manifest_log(item):
        (args.output_dir / "sanitized.log").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    _write_job_log(
        item,
        ok=ok,
        mock=str(item.get("status") or "").lower() == "mock",
    )
    _write_step_summary(
        item,
        run,
        ok=ok,
        mock=str(item.get("status") or "").lower() == "mock",
    )

    label = f"{args.skill} ({args.os})"
    if ok:
        print(f"{label}: {str(item.get('status')).upper()}")
        return 0
    print(
        f"::error::{label}: {_safe_status(item.get('status'))} - "
        f"{_safe_error(item.get('error'))}"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
