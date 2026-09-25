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
        or ("exactly one live session" in lowered and "infrastructure:" in lowered)
    ):
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
        if duration := _safe_duration(item.get("duration")):
            handle.write(f"| Duration | `{html.escape(duration)}` |\n")
        if error := _safe_error(item.get("error")):
            error = html.escape(error)
            handle.write(f"\n> {error}\n")
        if mock:
            handle.write("\nPlan validation only; no hardware test was executed.\n")


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
    compact_item = {
        "skill": skill,
        "os": os_name,
        "status": status,
        "terminal": bool(item.get("terminal")),
    }
    return {
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
    }


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
