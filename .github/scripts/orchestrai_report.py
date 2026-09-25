#!/usr/bin/env python3
"""Render one trustworthy GitHub summary from per-skill OrchestrAI artifacts."""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--expected-json", default="[]")
    parser.add_argument("--output", type=Path, default=Path("orchestrai-report.md"))
    return parser


def _expected(raw: str) -> list[tuple[str, str]]:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"expected matrix is invalid JSON: {exc}") from exc
    if not isinstance(value, list):
        raise SystemExit("expected matrix must be a JSON list")
    identities = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        identity = (str(entry.get("skill") or ""), str(entry.get("os") or ""))
        if all(identity) and identity not in identities:
            identities.append(identity)
    return identities


def _load_rows(root: Path) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    pattern = str(root / "**" / "summary.json")
    for raw_path in glob.glob(pattern, recursive=True):
        path = Path(raw_path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        identity = (str(value.get("skill") or ""), str(value.get("os") or ""))
        if not all(identity):
            continue
        if identity in rows:
            rows[identity] = {
                "skill": identity[0],
                "os": identity[1],
                "status": "error",
                "error": "duplicate result artifacts were downloaded",
            }
        else:
            rows[identity] = value
    return rows


def render(
    *, expected: list[tuple[str, str]], rows: dict[tuple[str, str], dict]
) -> str:
    ordered = expected or sorted(rows)
    rendered_rows = []
    for identity in ordered:
        value = rows.get(identity)
        if value is None:
            value = {
                "skill": identity[0],
                "os": identity[1],
                "status": "error",
                "error": "no per-skill result artifact was published",
            }
        rendered_rows.append(value)

    passed = sum(
        str(row.get("status") or "").lower() == "passed" for row in rendered_rows
    )
    mocked = sum(
        str(row.get("status") or "").lower() == "mock" for row in rendered_rows
    )
    failed = len(rendered_rows) - passed - mocked
    if rendered_rows and failed == 0 and mocked == 0:
        headline = f"✅ {passed} passed"
    elif rendered_rows and failed == 0:
        headline = f"🧪 {mocked} plan-only, {passed} passed"
    elif rendered_rows:
        headline = f"❌ {failed} failed, {passed} passed"
    else:
        headline = "ℹ️ no Strix behavioral tests selected"

    lines = [f"## OrchestrAI behavioral results — {headline}", ""]
    controller = os.environ.get("ORCHESTRAI_CONTROLLER_RESULT", "").strip()
    verdicts = os.environ.get("ORCHESTRAI_VERDICT_RESULT", "").strip()
    if controller or verdicts:
        lines.extend(
            [
                "| Layer | Result |",
                "|---|---|",
                f"| OrchestrAI controller | `{html.escape(controller or 'not requested')}` |",
                f"| Per-skill verdicts | `{html.escape(verdicts or 'not requested')}` |",
                "",
            ]
        )
    if rendered_rows:
        lines.extend(["| Skill | OS | Result | Detail |", "|---|---|---|---|"])
        for row in rendered_rows:
            status = str(row.get("status") or "unknown").lower()
            icon = "✅" if status == "passed" else ("🧪" if status == "mock" else "❌")
            detail = html.escape(str(row.get("error") or "")[:240]) or "—"
            lines.append(
                f"| `{html.escape(str(row.get('skill') or ''))}` | "
                f"{html.escape(str(row.get('os') or ''))} | {icon} `{html.escape(status)}` | "
                f"{detail} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parser().parse_args()
    document = render(
        expected=_expected(args.expected_json), rows=_load_rows(args.artifacts)
    )
    args.output.write_text(document, encoding="utf-8")
    print(document, end="")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY", "").strip():
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
