#!/usr/bin/env python3
"""Render one trustworthy GitHub summary from per-skill OrchestrAI artifacts."""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from orchestrai_logs import (
    behavioral_summary_markdown,
    expectation_breakdown_markdown,
    observed_rate,
    public_behavioral_summary,
)
from orchestrai_verdict import (
    _safe_error,
    _safe_status,
    log_coverage,
    public_controller_state,
    result_category,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--expected-json", default="[]")
    parser.add_argument("--output", type=Path, default=Path("orchestrai-report.md"))
    return parser


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
        skill, os_name = identity
        skill = skill if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", skill) else "unknown"
        os_name = os_name if os_name in {"Linux", "Windows"} else "unknown"
        rendered_rows.append(
            {
                **value,
                "skill": skill,
                "os": os_name,
                "status": _safe_status(value.get("status")),
            }
        )

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
    statuses = {
        key: sum(row["status"] == key for row in rendered_rows)
        for key in ("passed", "failed", "error", "mock")
    }
    other = len(rendered_rows) - sum(statuses.values())
    summaries = [
        public_behavioral_summary(row) if row["status"] != "mock" else {}
        for row in rendered_rows
    ]
    known = [summary for summary in summaries if summary]
    complete = sum(bool(summary.get("complete")) for summary in known)
    if rendered_rows:
        skills = sorted({row["skill"] for row in rendered_rows})
        lines.extend(
            [
                "Evaluated separately: "
                + ", ".join(f"`{skill}`" for skill in skills)
                + ". Each behavioral case loads its own skill and grades the resulting work; skills are not installed as a competing routing room.",
                "",
                "### Verdicts",
                "",
                "Counts below are **skill/OS evaluations**, not individual prompts or expectations.",
                "",
                "| Verdict | Count | Meaning |",
                "|---|---|---|",
            ]
        )
        verdicts = (
            ("passed", ("passed",), "The skill/OS test reported a passing result."),
            (
                "failed",
                ("failed",),
                "The test failed; see behavioral expectations or the test diagnostic below.",
            ),
            (
                "error",
                ("error",),
                "Execution, infrastructure, or reporting prevented a verified result.",
            ),
            (
                "cancelled / aborted",
                ("cancelled", "aborted"),
                "The requested evaluation did not finish.",
            ),
            (
                "skipped",
                ("skipped",),
                "The selected test was skipped; it is not a pass.",
            ),
            (
                "missing / unknown",
                ("missing", "unknown"),
                "The requested result could not be verified.",
            ),
            ("mock", ("mock",), "Plan validation only; no hardware evaluation ran."),
        )
        for label, accepted, meaning in verdicts:
            count = sum(row["status"] in accepted for row in rendered_rows)
            lines.append(f"| `{label}` | {count} | {meaning} |")
        lines.extend(
            [
                "",
                expectation_breakdown_markdown(
                    known, heading="### By expectation type"
                ).rstrip(),
                "",
            ]
        )
        lines.extend(
            [
                "### Per skill",
                "",
                "Linux and Windows legs are combined here; platform-specific verdicts and full-log links are listed below. Rates cover recorded cases only and are **not routing recall or precision**.",
                "",
                "| Skill | OS evaluations | Passed evaluations | Cases passed | Expectations met | Case pass rate (observed) | Expectation met rate (observed) | Case-count coverage |",
                "|---|---|---|---|---|---|---|---|",
            ]
        )
        for skill in skills:
            skill_rows = [row for row in rendered_rows if row["skill"] == skill]
            skill_summaries = [
                summary
                for row, summary in zip(rendered_rows, summaries)
                if row["skill"] == skill and summary
            ]
            if skill_summaries:
                total_cases = sum(summary["cases"] for summary in skill_summaries)
                passed_cases = sum(summary["passed"] for summary in skill_summaries)
                total_checks = sum(
                    summary["expectations"] for summary in skill_summaries
                )
                met_checks = sum(summary["met"] for summary in skill_summaries)
                cases = f"{passed_cases}/{total_cases}"
                checks = f"{met_checks}/{total_checks}"
                case_rate = observed_rate(passed_cases, total_cases)
                check_rate = observed_rate(met_checks, total_checks)
            else:
                cases = checks = case_rate = check_rate = "Not reported"
            complete_skill = sum(
                bool(summary["complete"]) for summary in skill_summaries
            )
            coverage = f"{complete_skill}/{len(skill_rows)} complete"
            if any(row["status"] == "mock" for row in skill_rows):
                coverage += "; includes plan-only"
            lines.append(
                f"| `{skill}` | {len(skill_rows)} | {sum(row['status'] == 'passed' for row in skill_rows)} | {cases} | {checks} | {case_rate} | {check_rate} | {coverage} |"
            )
        lines.append("")
        lines.extend(
            [
                f"**{len(rendered_rows)} skill/OS evaluations requested.** Passed: {statuses['passed']} · test failures: {statuses['failed']} · execution errors: {statuses['error']} · plan-only: {statuses['mock']} · other/unverified: {other}.",
                "",
            ]
        )
    if known:
        lines.extend(
            [
                f"**Observed graded cases:** {sum(s['passed'] for s in known)}/{sum(s['cases'] for s in known)} passed · **expectations:** {sum(s['met'] for s in known)}/{sum(s['expectations'] for s in known)} met.",
                "",
            ]
        )
    if rendered_rows:
        lines.extend(
            [
                f"**Case-count coverage:** {complete} complete · {len(known) - complete} partial · {len(rendered_rows) - len(known) - statuses['mock']} unavailable · {statuses['mock']} not run.",
                "Case/expectation totals cover recorded grader output only, not missing tests. Controller status and per-skill verdicts are separate; this report does not change the final CI gate.",
                "",
            ]
        )
    controller = os.environ.get("ORCHESTRAI_CONTROLLER_RESULT", "").strip()
    verdicts = os.environ.get("ORCHESTRAI_VERDICT_RESULT", "").strip()
    if controller or verdicts:
        lines.extend(
            [
                "| Layer | Result |",
                "|---|---|",
                f"| OrchestrAI controller | `{controller if controller in ('success', 'failure', 'cancelled', 'skipped') else 'not reported'}` |",
                f"| Per-skill verdicts | `{verdicts if verdicts in ('success', 'failure', 'cancelled', 'skipped') else 'not reported'}` |",
                "",
            ]
        )
    if rendered_rows:
        lines.extend(
            [
                "### Platform overview",
                "",
                "| OS | Requested | Passed | Test failures | Execution errors | Other / plan-only |",
                "|---|---|---|---|---|---|",
            ]
        )
        for os_name in sorted({row["os"] for row in rendered_rows}):
            platform_rows = [row for row in rendered_rows if row["os"] == os_name]
            counts = [
                sum(row["status"] == status for row in platform_rows)
                for status in ("passed", "failed", "error")
            ]
            lines.append(
                f"| {os_name} | {len(platform_rows)} | {counts[0]} | {counts[1]} | {counts[2]} | {len(platform_rows) - sum(counts)} |"
            )

        def report_link(row: dict) -> str:
            url = _safe_report_url(row.get("report_url"))
            return f"[View Results](<{html.escape(url)}>)" if url else "Not published"

        def diagnostic(row: dict) -> str:
            raw = row.get("error")
            if raw in (
                "no per-skill result artifact was published",
                "duplicate result artifacts were downloaded",
            ):
                return str(raw)
            return _safe_error(raw) or "No verified diagnostic was published."

        attention = [
            row for row in rendered_rows if row["status"] not in {"passed", "mock"}
        ]
        if attention:
            lines.extend(
                [
                    "",
                    "### Needs attention",
                    "",
                    "| Skill | OS | Failure type | Diagnostic | Full logs |",
                    "|---|---|---|---|---|",
                ]
            )
            for row in attention:
                lines.append(
                    f"| `{row['skill']}` | {row['os']} | {result_category(row)} | {html.escape(diagnostic(row))} | {report_link(row)} |"
                )
            # Failed case explanations are visible without expanding every
            # passed skill. Raw infrastructure errors remain in ReportPortal.
            for row in attention:
                if behavioral := behavioral_summary_markdown(row):
                    lines.extend(
                        [
                            "",
                            behavioral.replace(
                                "#### Skill behavioral",
                                f"#### ❌ {row['skill']} on {row['os']} — case results",
                                1,
                            ).rstrip(),
                        ]
                    )

        lines.extend(
            [
                "",
                "### All skill results",
                "",
                "| Skill | OS | Result | Cases passed | Expectations met | Model / effort | Case time | Public coverage | ReportPortal |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for row in rendered_rows:
            status = str(row.get("status") or "unknown").lower()
            icon = "✅" if status == "passed" else ("🧪" if status == "mock" else "❌")
            summary = public_behavioral_summary(row) if status != "mock" else {}
            cases = (
                f"{summary['passed']}/{summary['cases']}" if summary else "Not reported"
            )
            checks = (
                f"{summary['met']}/{summary['expectations']}"
                if summary
                else "Not reported"
            )
            if summary and not summary["complete"]:
                cases += " (partial)"
                checks += " (partial)"
            model = (
                f"{summary['model']} / {summary['effort']}"
                if summary.get("model")
                else "Not reported"
            )
            case_time = f"{summary['elapsed_s']:g}s" if summary else "Not reported"
            lines.append(
                f"| `{row['skill']}` | {row['os']} | {icon} `{status}` | {cases} | {checks} | {model} | {case_time} | {log_coverage(row)} | {report_link(row)} |"
            )
        for row in rendered_rows:
            if row["status"] != "passed":
                continue
            if behavioral := behavioral_summary_markdown(row):
                label = html.escape(f"{row.get('skill', '')} on {row.get('os', '')}")
                lines.extend(
                    [
                        "",
                        f"<details><summary>{label} — case results</summary>",
                        "",
                        behavioral.rstrip(),
                        "",
                        "</details>",
                    ]
                )
        controller_rows = []
        for os_name in sorted({row["os"] for row in rendered_rows}):
            states = [
                public_controller_state(row.get("controller"))
                for row in rendered_rows
                if row["os"] == os_name
            ]
            if any(
                any(value != "unknown" for value in state.values()) for state in states
            ):
                fields = []
                for field in ("pipeline_status", "tests_status", "cleanup_status"):
                    values = {
                        state[field] for state in states if state[field] != "unknown"
                    }
                    fields.append(
                        next(iter(values))
                        if len(values) == 1
                        else ("mixed snapshots" if values else "not reported")
                    )
                controller_rows.append(
                    f"| {os_name} | `{fields[0]}` | `{fields[1]}` | `{fields[2]}` |"
                )
        if controller_rows:
            lines.extend(
                [
                    "",
                    "### Controller snapshots",
                    "",
                    "| OS | Pipeline snapshot | Tests snapshot | Parent termination |",
                    "|---|---|---|---|",
                    *controller_rows,
                ]
            )
        lines.extend(
            [
                "",
                "### How to investigate",
                "",
                "- **Behavioral failures:** read the unmet expectations and per-case breakdowns above for context.",
                "- **Execution errors or missing output:** inspect the matching Linux/Windows controller job and follow **View Results** to ReportPortal (AMD access required).",
                "- **Partial counts:** do not treat missing cases as passed. Public output is bounded and sanitized; dependency logs and raw agent output remain in ReportPortal.",
                "- **Timing and cleanup:** case time excludes acquisition and adapter setup, includes in-case work, and is not total wall time. Parent termination is a controller observation; machine release remains managed by OrchestrAI.",
            ]
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
