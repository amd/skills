#!/usr/bin/env python3
"""Extract bounded grader evidence for structured GitHub summaries.

These recognized lines reconstruct informational counts only. Complete redacted
stdout/stderr is published independently by orchestrai_stdout, so changes to the
grader format cannot suppress test logs.
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from collections.abc import Iterable

MAX_PUBLIC_LOG_LINES = 300
MAX_PUBLIC_LOG_BYTES = 48_000
MAX_PUBLIC_LINE_CHARS = 1200
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)")
# Actor/transport logs can prefix each line with a clock or ISO timestamp.
# Remove only these exact wrappers, never arbitrary bracketed actor metadata.
TIMESTAMP_PREFIX_RE = re.compile(
    r"^(?:\[(?:\d{4}-\d{2}-\d{2}[T ])?(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d"
    r"(?:[.,]\d{1,9})?(?:Z| UTC|[+-]\d{2}:\d{2})?\]\s*){1,2}"
)
GRADE_RE = re.compile(
    r"^\[(PASS|FAIL|ERROR)\] \((files_exist|files_not_exist|logs_contain|"
    r"logs_not_contain|expected_behavior|unexpected_behavior)\) (.+)$"
)
CASE_RE = re.compile(
    r"^(?P<result>\[(?P<status>PASS|FAIL)\] (?P<case>[a-z0-9][a-z0-9_-]{0,127}): "
    r"(?P<met>\d{1,6})/(?P<checks>\d{1,6}) checks in (?P<elapsed>\d{1,8}(?:\.\d{1,6})?)s)"
    r"(?: -- (?P<error>.+))?$"
)
# Accept only the harness's numeric footer and public Claude model aliases.
# Never forward arbitrary report tables, prompt headers or model endpoints.
TOTALS_RE = re.compile(
    r"^(?:\*\*)?(?P<passed>\d{1,6})/(?P<cases>\d{1,6}) cases passed(?:\*\*)? "
    r"\((?P<met>\d{1,6})/(?P<checks>\d{1,6}) individual expectations\) "
    r"on `?(?P<model>opus|sonnet|haiku|claude-(?:opus|sonnet|haiku)-\d{1,2}(?:-\d{1,8}){0,2})`? "
    r"\(effort `?(?P<effort>low|medium|high|max)`?\)\.$"
)
CREDENTIAL_RE = re.compile(
    r"(?i)\b(?:[a-z0-9_-]*(?:api[_-]?key|access[_-]?key|private[_-]?key|password|passwd|secret|token)|"
    r"key|subscription[_-]?key|authorization|ocp-apim-subscription-key|anthropic_custom_headers|"
    r"llm_gateway_(?:url|user))\b[\"']?\s*[:=]"
)
URL_RE = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*:(?:\\?/){2}[^\s<>\"']+")
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
ADDRESS_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HOST_RE = re.compile(
    r"(?i)\b(?:[a-z0-9_-]+\.)+(?:com|net|org|io|ai|dev|edu|gov|"
    r"internal|local|corp|lan)\b(?::\d+)?"
)
IDENTITY_RE = re.compile(
    r"(?i)\b(?:host(?:name)?|machine|actor|executionnode|user(?:name)?|"
    r"session[_-]?id|job[_-]?id|run[_-]?id)\b\s*[:=]\s*[^\s,;]+"
)
MACHINE_NAME_RE = re.compile(
    r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+){2,}\b|"
    r"\b[a-zA-Z][a-zA-Z0-9]*(?:-[a-zA-Z0-9]+)+-[a-zA-Z0-9]*\d[a-zA-Z0-9]*\b"
)
PATH_RE = re.compile(
    r"(?i)(?:[a-z]:[\\/]|\\\\)[^\s<>\"']+|"
    r"/(?:home|root|tmp|var|opt|etc|mnt|srv|Users)(?:/[^\s<>\"']*)?"
)
OPAQUE_RE = re.compile(
    r"\b[a-zA-Z0-9_+/=]{28,}\b|"
    r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b"
)
PUBLIC_NOTICES = {
    "[REDACTED: credential-related grader output; inspect the redacted test log]",
    "[REDACTED: address-related grader output; inspect the redacted test log]",
    "[Grader summary truncated; inspect the redacted test log.]",
}
GRADER_ERROR_DIAGNOSTICS = {
    "The behavioral grader was unavailable.",
    "The behavioral grader timed out.",
    "The behavioral grader exceeded the gateway context limit.",
    "The behavioral grader lost connectivity to the LLM gateway.",
    "The behavioral grader did not return a usable JSON verdict.",
    "One or more behavioral expectations could not be graded.",
}
UNVERIFIED_BEHAVIORAL_DIAGNOSTIC = "The behavioral result could not be verified."


def grader_error_diagnostic(detail: str) -> str:
    """Recognize Skillscope-owned failure details, never agent prose.

    v0.1.3 emits these instead of a verdict but labels the check FAIL. Match
    only the detail prefix, so a real judge reason quoting an API error stays
    a behavioral verdict. Keep published explanations fixed and private-free.
    """
    detail = detail.strip()
    if detail in GRADER_ERROR_DIAGNOSTICS or detail == UNVERIFIED_BEHAVIORAL_DIAGNOSTIC:
        return detail
    lowered = detail.lower()
    if lowered.startswith("llm_judge skipped:"):
        return "The behavioral grader was unavailable."
    if re.match(r"llm_judge timed out after \d+(?:\.\d+)?s\b", lowered):
        return "The behavioral grader timed out."
    if lowered.startswith("llm_judge gave no json verdict:"):
        if any(
            marker in lowered
            for marker in (
                "context_length_exceeded",
                "prompt is too long",
                "request too large",
                "maximum context length",
            )
        ):
            return "The behavioral grader exceeded the gateway context limit."
        if any(
            marker in lowered
            for marker in (
                "connection refused",
                "connection reset",
                "connection dropped",
                "network error",
                "socket hang up",
                "fetch failed",
                "econnrefused",
                "econnreset",
                "enotfound",
            )
        ):
            return "The behavioral grader lost connectivity to the LLM gateway."
        return "The behavioral grader did not return a usable JSON verdict."
    if lowered.startswith("behavioral exceeded --timeout"):
        return "The behavioral grader timed out."
    return ""


def grader_lines(test: dict) -> Iterable[str]:
    """Strip only known transport wrappers before matching harness lines."""
    for stream in ("stdout", "stderr"):
        raw = test.get(stream)
        if not isinstance(raw, str):
            continue
        for raw_line in ANSI_RE.sub("", raw[-2_000_000:]).splitlines():
            line = "".join(
                char
                for char in raw_line
                if not unicodedata.category(char).startswith("C")
            ).strip()
            yield TIMESTAMP_PREFIX_RE.sub("", line).strip()


def grade_detail(grade: re.Match) -> tuple[str, str]:
    # Judge reasons can themselves contain " -- ". Use the harness's judge
    # detail marker first so those separators stay inside the verdict reason.
    if grade[2] in {"expected_behavior", "unexpected_behavior"}:
        expectation, separator, detail = grade[3].partition(" -- llm_judge")
        if separator:
            return expectation, "llm_judge" + detail
    expectation, separator, detail = grade[3].rpartition(" -- ")
    return (expectation, detail) if separator else (grade[3], "")


def grade_error(grade: re.Match) -> str:
    if grade[1] == "PASS":
        return ""
    _, detail = grade_detail(grade)
    if grade[1] == "ERROR":
        return grader_error_diagnostic(detail) or UNVERIFIED_BEHAVIORAL_DIAGNOSTIC
    if grade[2] in {"expected_behavior", "unexpected_behavior"}:
        return grader_error_diagnostic(detail) or (
            UNVERIFIED_BEHAVIORAL_DIAGNOSTIC
            if not detail.startswith("llm_judge: ") or not detail[11:].strip()
            else ""
        )
    return ""


def behavioral_grading_error(test: dict) -> str:
    """Find an attributable per-expectation grader outage in private output."""
    for line in grader_lines(test):
        if (grade := GRADE_RE.fullmatch(line)) and (error := grade_error(grade)):
            return error
    return ""


def observed_rate(met: int, graded: int) -> str:
    """Never report a percentage for absent or contradictory denominators."""
    if graded <= 0 or not 0 <= met <= graded:
        return "Not reported"
    return f"{100 * met / graded:.0f}%"


def expectation_breakdown_markdown(summaries: Iterable[dict], *, heading: str) -> str:
    kinds: dict[str, dict[str, int]] = {}
    for summary in summaries:
        if not isinstance(summary, dict) or not isinstance(
            summary.get("expectation_kinds"), dict
        ):
            continue
        for kind, counts in summary["expectation_kinds"].items():
            if kind not in (
                "files_exist",
                "files_not_exist",
                "logs_contain",
                "logs_not_contain",
                "expected_behavior",
                "unexpected_behavior",
            ):
                continue
            if (
                not isinstance(counts, dict)
                or type(counts.get("graded")) is not int
                or type(counts.get("met")) is not int
            ):
                continue
            if not 0 <= counts["met"] <= counts["graded"] <= MAX_PUBLIC_LOG_LINES:
                continue
            ungraded = counts.get("ungraded", 0)
            if type(ungraded) is not int or not 0 <= ungraded <= MAX_PUBLIC_LOG_LINES:
                continue
            total = kinds.setdefault(kind, {"graded": 0, "met": 0, "ungraded": 0})
            total["graded"] += counts["graded"]
            total["met"] += counts["met"]
            total["ungraded"] += ungraded
    lines = [
        heading,
        "",
        "Only retained, case-associated grader lines are counted below. Missing or unassigned checks are excluded; redacted checks retain their observed verdict. Ungraded checks are excluded from met/unmet rates.",
        "",
    ]
    if not kinds:
        lines.append("No publishable expectation-level results were reported.")
    else:
        has_ungraded = any(counts["ungraded"] for counts in kinds.values())
        lines.extend(
            [
                "| Expectation type | Graded | Met | Unmet | "
                + ("Ungraded | " if has_ungraded else "")
                + "Met rate (observed) |",
                "|---|---|---|---|---|" + ("---|" if has_ungraded else ""),
            ]
        )
        for kind, counts in sorted(kinds.items()):
            lines.append(
                f"| `{kind}` | {counts['graded']} | {counts['met']} | {counts['graded'] - counts['met']} | "
                + (f"{counts['ungraded']} | " if has_ungraded else "")
                + f"{observed_rate(counts['met'], counts['graded'])} |"
            )
        if "unexpected_behavior" in kinds:
            lines.extend(
                [
                    "",
                    "For `unexpected_behavior`, met means the prohibited behavior did **not** occur.",
                ]
            )
    return "\n".join(lines) + "\n"


def private_log_values(plan: dict, live: dict, environ: dict) -> list[str]:
    """Collect known private values without exporting their names or values."""
    values = [
        str(value)
        for key, value in environ.items()
        if re.search(
            r"(?i)password|passwd|secret|token|api.?key|llm_gateway|"
            r"orchestrai_(?:portal_url|user|space|device_tags|linux_driver_sources_json|windows_driver_source)$",
            key,
        )
        and value
    ]
    for session in plan.get("sessions", []):
        if isinstance(session, dict):
            values.extend(
                tag for tag in session.get("machine_tags", []) if isinstance(tag, str)
            )

    def walk(value: object, context: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(child, str) and (
                    re.fullmatch(
                        r"(?i)(?:hostname|ip|ip_address|address|machine_name|actor_name|"
                        r"executionnode|username|launcher_url|jenkins_url|serial|"
                        r"llm_gateway_(?:key|url|user)|.*(?:password|secret|token|api_key))",
                        str(key),
                    )
                    or (
                        context in {"actor", "actors", "machine", "machines"}
                        and key in {"name", "id", "user"}
                    )
                ):
                    values.append(child)
                elif isinstance(child, (dict, list)):
                    walk(child, str(key))
        elif isinstance(value, list):
            for child in value:
                walk(child, context)

    walk(live)
    walk(plan)

    # Driver maps and fleet settings are JSON secrets: replacing the complete
    # JSON string would miss an individual URL/tag printed by dependency setup.
    def strings(value: object) -> None:
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                strings(child)
        elif isinstance(value, list):
            for child in value:
                strings(child)

    for key, value in environ.items():
        if key.startswith("ORCHESTRAI_") and isinstance(value, str):
            try:
                strings(json.loads(value))
            except (ValueError, TypeError):
                pass
    driver_vars = (plan.get("builds_json") or {}).get("vars") or {}
    if isinstance(driver_vars, dict):
        if isinstance(source := driver_vars.get("driver_source"), str) and source:
            values.append(source)
        raw = driver_vars.get("driver_sources_json")
        if isinstance(raw, str):
            try:
                strings(json.loads(raw))
            except ValueError:
                pass
    return values


def redact_private_values(value: str, private_values: Iterable[str] = ()) -> str:
    """Mask known values in raw and JSON-serialized logger representations."""
    private_values = {
        private for private in private_values
        if isinstance(private, str) and len(private) >= 4
    }
    for private in sorted(private_values, key=len, reverse=True):
        encoded = {
            private,
            json.dumps(private)[1:-1],
            json.dumps(private, ensure_ascii=False)[1:-1],
        }
        variants = encoded | {part.replace("/", "\\/") for part in encoded}
        for part in sorted(variants, key=len, reverse=True):
            value = value.replace(part, "[REDACTED]")
    return value


def redact_grader_text(value: str, private_values: Iterable[str] = ()) -> str:
    """Redact network, machine and credential data from one grader line."""
    # Credential assignments and auth headers can have arbitrary value shapes.
    # Suppress the whole line instead of guessing where those values end.
    if CREDENTIAL_RE.search(value) or re.search(
        r"(?i)\b(?:bearer|basic)\s+\S+|-----BEGIN .*PRIVATE KEY-----", value
    ):
        return "[REDACTED: credential-related grader output; inspect the redacted test log]"
    value = redact_private_values(value, private_values)
    value = URL_RE.sub("[URL REDACTED]", value)
    value = EMAIL_RE.sub("[EMAIL REDACTED]", value)
    value = ADDRESS_RE.sub("[ADDRESS REDACTED]", value)
    # IPv6 addresses and double-colon control sequences suppress the line.
    if re.search(r"(?i)(?:[0-9a-f]{0,4}:){2,}[0-9a-f:.%_-]*", value):
        return (
            "[REDACTED: address-related grader output; inspect the redacted test log]"
        )
    value = HOST_RE.sub("[HOST REDACTED]", value)
    value = IDENTITY_RE.sub("[IDENTITY REDACTED]", value)
    value = MACHINE_NAME_RE.sub("[IDENTITY REDACTED]", value)
    value = PATH_RE.sub("[PATH REDACTED]", value)
    value = OPAQUE_RE.sub("[VALUE REDACTED]", value)
    # Never allow an actor line to create Actions annotations, groups or masks,
    # or to break out of the fenced block in a GitHub step summary.
    value = value.replace("::", ": :").replace("`", "'")
    return value[:MAX_PUBLIC_LINE_CHARS]


def public_test_log(
    test: dict, *, skill: str, private_values: Iterable[str] = ()
) -> list[str]:
    """Return only recognized grader lines, with a fixed output budget."""
    lines: list[str] = []
    used_bytes = 0
    for line in grader_lines(test):
        header = re.fullmatch(
            rf"\[behavioral\] {re.escape(skill)}: \d+ case\(s\)", line
        )
        case = CASE_RE.fullmatch(line)
        totals = TOTALS_RE.fullmatch(line)
        if not (
            header
            or GRADE_RE.fullmatch(line)
            or case
            or totals
            or line in PUBLIC_NOTICES
        ):
            continue
        if grade := GRADE_RE.fullmatch(line):
            if error := grade_error(grade):
                expectation, _ = grade_detail(grade)
                line = f"[ERROR] ({grade[2]}) {expectation} -- {error}"
        if totals:
            # Canonicalize Markdown and plaintext variants before the
            # second allowlist pass in the verdict job.
            line = (
                f"{totals['passed']}/{totals['cases']} cases passed "
                f"({totals['met']}/{totals['checks']} individual expectations) "
                f"on {totals['model']} (effort {totals['effort']})."
            )
        if case and case.group("error"):
            # Appended runtime errors are not judge explanations: keep
            # their arbitrary server-owned content in the full report.
            line = (
                case.group("result")
                + " -- [Error details withheld; inspect the redacted test log]"
            )
        clean = redact_grader_text(line, private_values)
        if clean in PUBLIC_NOTICES and (grade := GRADE_RE.fullmatch(line)):
            # The verdict's structure is known before redaction. Preserve
            # that evidence without publishing the private statement/reason;
            # losing the line would turn a real negative into an apparent
            # judge outage in the next normalization pass.
            detail = (
                grade_error(grade)
                if grade[1] == "ERROR"
                else "llm_judge: Behavioral expectation was not met; details redacted."
                if grade[1] == "FAIL"
                and grade[2] in {"expected_behavior", "unexpected_behavior"}
                else "Grader details redacted; inspect the redacted test log."
            )
            clean = f"[{grade[1]}] ({grade[2]}) [details redacted] -- {detail}"
        line_bytes = len(clean.encode("utf-8")) + 1
        if (
            len(lines) >= MAX_PUBLIC_LOG_LINES
            or used_bytes + line_bytes > MAX_PUBLIC_LOG_BYTES
        ):
            lines.append("[Grader summary truncated; inspect the redacted test log.]")
            return lines
        lines.append(clean)
        used_bytes += line_bytes
    return lines


def public_log_status(test: dict, lines: list[str]) -> str:
    """Distinguish a transport gap from output the public filter rejected."""
    if lines:
        return "available"
    if any(
        isinstance(test.get(stream), str) and test[stream].strip()
        for stream in ("stdout", "stderr")
    ):
        return "no_recognized_lines"
    return "unavailable"


def public_manifest_log(item: dict) -> list[str]:
    """Revalidate transported log lines before writing public files or logs."""
    values = item.get("public_log")
    if not isinstance(values, list):
        return []
    # Apply the same allowlist again: a fabricated manifest must not bypass the
    # controller-side filter. Values are bounded before joining them.
    text = "\n".join(
        line
        for line in values[: MAX_PUBLIC_LOG_LINES + 1]
        if isinstance(line, str) and len(line) <= MAX_PUBLIC_LINE_CHARS
    )
    return public_test_log({"stdout": text}, skill=str(item.get("skill") or ""))


def public_behavioral_summary(item: dict) -> dict:
    """Reconstruct case totals and unmet expectations from sanitized lines.

    Skillscope prints checks before each case's result. Only a matching case
    footer assigns those checks to a case; a partial log is never described as
    a complete successful evaluation. These counts are informational and do
    not replace the test verdict.
    """
    cases: dict[str, dict] = {}
    pending: list[dict] = []
    pending_errors: list[dict] = []
    pending_kinds: dict[str, dict[str, int]] = {}
    declared: set[int] = set()
    footers: set[tuple] = set()
    consistent = True
    for line in public_manifest_log(item):
        if header := re.fullmatch(
            r"\[behavioral\] [a-z0-9-]+: (\d{1,6}) case\(s\)", line
        ):
            declared.add(int(header[1]))
        elif totals := TOTALS_RE.fullmatch(line):
            footers.add(
                tuple(int(totals[key]) for key in ("passed", "cases", "met", "checks"))
                + (totals["model"], totals["effort"])
            )
        elif grade := GRADE_RE.fullmatch(line):
            expectation, detail = grade_detail(grade)
            if grade[1] == "ERROR":
                counts = pending_kinds.setdefault(grade[2], {"graded": 0, "met": 0})
                counts["ungraded"] = counts.get("ungraded", 0) + 1
                pending_errors.append(
                    {"kind": grade[2], "expectation": expectation, "detail": detail}
                )
            else:
                counts = pending_kinds.setdefault(grade[2], {"graded": 0, "met": 0})
                counts["graded"] += 1
                counts["met"] += grade[1] == "PASS"
                if grade[1] == "FAIL":
                    pending.append(
                        {"kind": grade[2], "expectation": expectation, "detail": detail}
                    )
        elif case := CASE_RE.fullmatch(line):
            met, checks = int(case["met"]), int(case["checks"])
            passed = case["status"] == "PASS"
            invalid_counts = met > checks or (
                passed and (met != checks or not checks or pending or pending_errors)
            )
            if invalid_counts:
                consistent = False
            unmet = pending
            pending = []
            errors = pending_errors
            pending_errors = []
            kinds = pending_kinds
            pending_kinds = {}
            visible_checks = sum(counts["graded"] for counts in kinds.values())
            visible_met = sum(counts["met"] for counts in kinds.values())
            if (
                visible_checks + len(errors) > checks
                or visible_met > met
                or visible_checks - visible_met > checks - met
            ):
                # A missing footer can merge two cases' lines. Do not assign
                # that ambiguous block to a category or manufacture a rate.
                kinds = {}
                consistent = False
                invalid_counts = True
                unmet = []
                errors = []
                visible_met = 0
            if invalid_counts:
                met = visible_met
                errors = errors + [
                    {
                        "kind": "error",
                        "expectation": "(case result unverified)",
                        "detail": "The grader case counts contradicted the recorded expectations.",
                    }
                ]
            if case["error"]:
                errors = errors + [
                    {
                        "kind": "error",
                        "expectation": "(run failed)",
                        "detail": "Error details withheld; inspect the redacted test log.",
                    }
                ]
            # The old harness's failed count conflates judge errors with
            # negative verdicts. Missing or filtered details cannot establish
            # a violation either, especially for unexpected_behavior.
            ungraded = max(0, checks - met - len(unmet))
            if ungraded > sum(error["kind"] != "error" for error in errors):
                errors = errors + [
                    {
                        "kind": "withheld",
                        "expectation": "Expectations without a usable verdict",
                        "detail": "Some grader details were unavailable or redacted; no behavioral outcome can be inferred.",
                    }
                ]
            case_status = (
                "passed"
                if passed and not errors and not unmet
                else ("error" if errors or ungraded or not unmet else "failed")
            )
            value = {
                "passed": case_status == "passed",
                "met": met,
                "checks": checks,
                "unmet": unmet,
                "grading_errors": errors,
                "ungraded": ungraded,
                "graded": max(0, checks - ungraded),
                "status": case_status,
                "elapsed_s": float(case["elapsed"]),
                "expectation_kinds": kinds,
            }
            if case["case"] in cases and cases[case["case"]] != value:
                consistent = False
            else:
                cases[case["case"]] = value

    if not cases:
        return {}
    summary = {
        "cases": len(cases),
        "passed": sum(case["passed"] for case in cases.values()),
        "expectations": sum(case["checks"] for case in cases.values()),
        "met": sum(case["met"] for case in cases.values()),
        "graded": sum(case["graded"] for case in cases.values()),
        "ungraded": sum(case["ungraded"] for case in cases.values()),
        "failed": sum(case["status"] == "failed" for case in cases.values()),
        "errors": sum(case["status"] == "error" for case in cases.values()),
        "unmet": [
            {"case": case_id, **check}
            for case_id, case in cases.items()
            for check in case["unmet"]
        ],
        "grading_errors": [
            {"case": case_id, **check}
            for case_id, case in cases.items()
            for check in case["grading_errors"]
        ],
        "complete": False,
        "elapsed_s": round(sum(case["elapsed_s"] for case in cases.values()), 2),
        "case_results": [
            {
                "case": case_id,
                "passed": case["passed"],
                "met": case["met"],
                "expectations": case["checks"],
                "graded": case["graded"],
                "ungraded": case["ungraded"],
                "status": case["status"],
                "elapsed_s": case["elapsed_s"],
            }
            for case_id, case in cases.items()
        ],
        "expectation_kinds": {},
    }
    for case in cases.values():
        for kind, counts in case["expectation_kinds"].items():
            total = summary["expectation_kinds"].setdefault(
                kind, {"graded": 0, "met": 0}
            )
            total["graded"] += counts["graded"]
            total["met"] += counts["met"]
            if counts.get("ungraded"):
                total["ungraded"] = total.get("ungraded", 0) + counts["ungraded"]
    counts = (
        summary["passed"],
        summary["cases"],
        summary["met"],
        summary["expectations"],
    )
    if len(footers) == 1:
        footer = next(iter(footers))
        summary["complete"] = (
            consistent
            and not pending
            and not pending_errors
            and not pending_kinds
            and counts == footer[:4]
            and (not declared or declared == {summary["cases"]})
        )
        if summary["complete"]:
            summary.update(model=footer[4], effort=footer[5])
    elif not footers:
        summary["complete"] = (
            consistent
            and not pending
            and not pending_errors
            and not pending_kinds
            and declared == {summary["cases"]}
        )
    return summary


def behavioral_summary_markdown(item: dict) -> str:
    """Render only reconstructed, revalidated fields, not actor Markdown."""
    summary = public_behavioral_summary(item)
    if not summary:
        return ""
    skill = str(item.get("skill") or "")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", skill):
        skill = "unknown"

    def cell(value: str) -> str:
        # No HTML, links, fences, or table delimiters from judge text.
        return (
            html.escape(value)
            .replace("|", "&#124;")
            .replace("[", "&#91;")
            .replace("]", "&#93;")
        )

    headline = f"**{summary['passed']}/{summary['cases']} cases passed** ({summary['met']}/{summary['expectations']} individual expectations)"
    if summary["ungraded"]:
        headline = f"**{summary['passed']}/{summary['cases']} cases passed** ({summary['met']}/{summary['graded']} graded expectations met; {summary['ungraded']} ungraded of {summary['expectations']})"
    if summary.get("model"):
        headline += f" on `{summary['model']}` (effort `{summary['effort']}`)"
    lines = ["#### Skill behavioral", "", headline + ".", ""]
    if summary["grading_errors"]:
        lines.extend(
            [
                f"Cases with errors: {summary['errors']}. Ungraded expectations are excluded from met/unmet rates; a missing verdict establishes no behavioral outcome.",
                "",
                "##### Ungraded expectations / run errors",
                "",
                "| Case | Kind | Expectation | Detail |",
                "|---|---|---|---|",
            ]
        )
        for check in summary["grading_errors"]:
            lines.append(
                f"| `{cell(check['case'])}` | {cell(check['kind'])} | {cell(check['expectation'])} | {cell(check['detail']) or '—'} |"
            )
        lines.append("")
    if not summary["complete"]:
        lines.extend(
            [
                "Partial grader output: counts cover recorded cases only; inspect the redacted test log for context.",
                "",
            ]
        )
    lines.extend(
        [
            "| Skill | Cases | Passed | Expectations | Met | Case pass rate (observed) | Expectation met rate (observed) |",
            "|---|---|---|---|---|---|---|",
            f"| `{skill}` | {summary['cases']} | {summary['passed']} | {summary['expectations']} | {summary['met']} | {observed_rate(summary['passed'], summary['cases'] - summary['errors'])} | {observed_rate(summary['met'], summary['graded'])} |",
            "",
            "##### Unmet expectations",
            "",
        ]
    )
    if summary["unmet"]:
        lines.extend(["| Case | Kind | Expectation | Detail |", "|---|---|---|---|"])
        for check in summary["unmet"]:
            lines.append(
                f"| `{cell(check['case'])}` | {cell(check['kind'])} | {cell(check['expectation'])} | {cell(check['detail']) or '—'} |"
            )
    elif summary["grading_errors"]:
        lines.append(
            "No verified unmet expectations were reported. Some outcomes could not be graded."
        )
    elif summary["complete"] and summary["passed"] == summary["cases"]:
        lines.append("None. Every behavioral case met every expectation.")
    elif summary["passed"] == summary["cases"]:
        lines.append(
            "No unmet expectations in the recorded cases. Coverage is partial; additional cases may be missing from the public output."
        )
    else:
        lines.append(
            "Unmet expectation details were unavailable in the grader summary; inspect the redacted test log."
        )
    lines.extend(
        [
            "",
            expectation_breakdown_markdown(
                [summary], heading="##### By expectation type"
            ).rstrip(),
            "",
            "##### Case breakdown",
            "",
            "| Case | Result | Expectations met | Harness case time |",
            "|---|---|---|---|",
        ]
    )
    for case in sorted(summary["case_results"], key=lambda value: value["passed"]):
        result = {"passed": "✅ passed", "failed": "❌ failed", "error": "⚠️ error"}[
            case["status"]
        ]
        checks = f"{case['met']}/{case['expectations']}"
        if case["ungraded"]:
            checks = (
                f"{case['met']}/{case['graded']} graded; {case['ungraded']} ungraded"
            )
        lines.append(
            f"| `{cell(case['case'])}` | {result} | {checks} | {case['elapsed_s']:g}s |"
        )
    lines.extend(
        [
            "",
            "Case time excludes machine acquisition and adapter dependency setup; in-case setup and agent work are included. Expand Test stdout/stderr (redacted) in the job log or download stdout-stderr.log from the per-skill artifact.",
        ]
    )
    return "\n".join(lines) + "\n"
