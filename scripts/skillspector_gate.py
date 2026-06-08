#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0"]
# ///
"""Apply the SkillSpector allowlist to a scan report and gate CI on it.

SkillSpector has no native suppression: every scan emits all of its findings,
and the CLI's own exit code is driven by the *aggregate* risk score, which a
pile of MEDIUM findings can push to HIGH even when nothing individually is
HIGH/CRITICAL. This script is the gate the `skillspector` workflow actually
relies on. Given SkillSpector's JSON report it:

  1. Reads every finding from the report.
  2. Drops findings matched by the allowlist (.github/skillspector-allow.yml)
     entries for this skill -- these are documented false positives / accepted
     risks. Allowlisted findings are *never printed*, so they don't show up in
     the CI console at all (the raw report is still kept as an artifact for
     audit).
  3. Prints the remaining findings.
  4. Exits non-zero when any *non-allowlisted* finding is HIGH or CRITICAL.

Because the allowlist is applied before anything is printed, the previous
pattern of dumping the raw report to the console (`cat report.md`) and only
then filtering is gone: suppressed HIGH/CRITICAL findings never reach the
screen in the first place.

Usage:

    uv run scripts/skillspector_gate.py \
        --report reports/rocm-doctor.json \
        --skill rocm-doctor \
        --allowlist .github/skillspector-allow.yml

Exit codes:
    0  No non-allowlisted HIGH/CRITICAL findings.
    1  At least one non-allowlisted HIGH/CRITICAL finding.
    2  Usage / parse error (bad report, bad allowlist, etc.).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

# Severities that fail the gate. SkillSpector uses LOW/MEDIUM/HIGH/CRITICAL.
BLOCKING_SEVERITIES = {"HIGH", "CRITICAL"}


class GateError(Exception):
    """Raised for usage/parse problems that should exit with code 2."""


@dataclass
class Finding:
    """The subset of a SkillSpector JSON finding the gate cares about."""

    rule_id: str
    severity: str
    file: str
    start_line: int | None
    explanation: str
    remediation: str | None
    confidence: float | None
    raw: dict

    @property
    def is_blocking(self) -> bool:
        return self.severity in BLOCKING_SEVERITIES


@dataclass
class AllowEntry:
    """A single allowlist rule, already scoped to the current skill."""

    rule: str
    file: str | None
    start_line: int | None
    severity: str | None
    reason: str

    def matches(self, finding: Finding) -> bool:
        if self.rule != "*" and self.rule.upper() != finding.rule_id.upper():
            return False
        if self.severity and self.severity.upper() != finding.severity.upper():
            return False
        if self.start_line is not None and self.start_line != finding.start_line:
            return False
        if self.file is not None and not _paths_match(self.file, finding.file):
            return False
        return True


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./").strip()


def _paths_match(allow_file: str, finding_file: str) -> bool:
    """Lenient path comparison tolerant of separators and skill-relative roots."""
    a = _normalize_path(allow_file)
    b = _normalize_path(finding_file)
    if a == b:
        return True
    # Allow either side to be a suffix of the other so an allowlist entry like
    # "scripts/apply_fix.py" matches a finding reported as
    # "rocm-doctor/scripts/apply_fix.py" (or vice versa).
    return a.endswith("/" + b) or b.endswith("/" + a)


def load_report(report_path: Path) -> list[Finding]:
    """Parse a SkillSpector JSON report into Finding objects."""
    if not report_path.exists():
        raise GateError(f"Report not found: {report_path}")

    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateError(f"Report is not valid JSON ({report_path}): {exc}") from exc

    if not isinstance(data, dict):
        raise GateError(
            f"Report {report_path} must be a SkillSpector JSON object "
            "(scan with `--format json`)."
        )

    issues = data.get("issues")
    if issues is None:
        raise GateError(
            f"Report {report_path} has no `issues` key; was it scanned with "
            "`--format json`?"
        )
    if not isinstance(issues, list):
        raise GateError(f"Report {report_path}: `issues` must be a list.")

    findings: list[Finding] = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        location = issue.get("location") or {}
        findings.append(
            Finding(
                rule_id=str(issue.get("id") or "UNKNOWN"),
                severity=str(issue.get("severity") or "LOW").upper(),
                file=str(location.get("file") or ""),
                start_line=location.get("start_line"),
                explanation=str(issue.get("explanation") or "").strip(),
                remediation=(issue.get("remediation") or None),
                confidence=issue.get("confidence"),
                raw=issue,
            )
        )
    return findings


def load_allowlist(allowlist_path: Path, skill: str) -> list[AllowEntry]:
    """Load allowlist entries that apply to `skill`."""
    if not allowlist_path.exists():
        # No allowlist is a valid state: nothing is suppressed.
        return []

    try:
        data = yaml.safe_load(allowlist_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise GateError(f"Allowlist is not valid YAML ({allowlist_path}): {exc}") from exc

    if data is None:
        return []
    if not isinstance(data, dict) or "allow" not in data:
        raise GateError(
            f"Allowlist {allowlist_path} must be a mapping with a top-level "
            "`allow:` list."
        )

    raw_entries = data.get("allow") or []
    if not isinstance(raw_entries, list):
        raise GateError(f"Allowlist {allowlist_path}: `allow` must be a list.")

    entries: list[AllowEntry] = []
    for idx, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise GateError(f"Allowlist {allowlist_path}: allow[{idx}] must be a mapping.")
        entry_skill = raw.get("skill")
        rule = raw.get("rule")
        reason = raw.get("reason")
        if not entry_skill:
            raise GateError(f"Allowlist {allowlist_path}: allow[{idx}] is missing `skill`.")
        if not rule:
            raise GateError(f"Allowlist {allowlist_path}: allow[{idx}] is missing `rule`.")
        if not reason or not str(reason).strip():
            raise GateError(
                f"Allowlist {allowlist_path}: allow[{idx}] ({entry_skill}/{rule}) "
                "is missing a non-empty `reason`."
            )
        if entry_skill != skill:
            continue
        start_line = raw.get("start_line")
        entries.append(
            AllowEntry(
                rule=str(rule),
                file=(str(raw["file"]) if raw.get("file") else None),
                start_line=(int(start_line) if start_line is not None else None),
                severity=(str(raw["severity"]) if raw.get("severity") else None),
                reason=str(reason).strip(),
            )
        )
    return entries


def partition(
    findings: list[Finding], allow_entries: list[AllowEntry]
) -> tuple[list[Finding], list[tuple[Finding, AllowEntry]], set[int]]:
    """Split findings into (kept, suppressed) and report which entries were used."""
    kept: list[Finding] = []
    suppressed: list[tuple[Finding, AllowEntry]] = []
    used_entry_ids: set[int] = set()

    for finding in findings:
        match = next((e for e in allow_entries if e.matches(finding)), None)
        if match is None:
            kept.append(finding)
        else:
            suppressed.append((finding, match))
            used_entry_ids.add(id(match))
    return kept, suppressed, used_entry_ids


def _print_finding(finding: Finding) -> None:
    line = f":{finding.start_line}" if finding.start_line else ""
    print(f"  [{finding.severity}] {finding.rule_id} - {finding.file}{line}")
    if finding.confidence is not None:
        try:
            print(f"    Confidence: {float(finding.confidence):.0%}")
        except (TypeError, ValueError):
            pass
    if finding.explanation:
        print(f"    {finding.explanation}")
    if finding.remediation:
        print(f"    Remediation: {finding.remediation}")


def report_results(
    skill: str,
    kept: list[Finding],
    suppressed: list[tuple[Finding, AllowEntry]],
    allow_entries: list[AllowEntry],
    used_entry_ids: set[int],
) -> int:
    """Print the filtered results and return the process exit code."""
    print(f"===== SkillSpector gate: {skill} =====")

    # Suppressed findings are deliberately summarized, not detailed: the whole
    # point is that allowlisted HIGH/CRITICAL issues do not show up on screen.
    if suppressed:
        ids = ", ".join(sorted({f.rule_id for f, _ in suppressed}))
        print(
            f"Suppressed {len(suppressed)} allowlisted finding(s) [{ids}] "
            "(see .github/skillspector-allow.yml)."
        )

    # Warn about allowlist entries that matched nothing so stale suppressions
    # get cleaned up. This never fails the build on its own.
    unused = [e for e in allow_entries if id(e) not in used_entry_ids]
    for entry in unused:
        target = entry.file or "any file"
        print(
            f"WARNING: allowlist entry {skill}/{entry.rule} ({target}) matched "
            "no finding; consider removing it."
        )

    blocking = [f for f in kept if f.is_blocking]
    non_blocking = [f for f in kept if not f.is_blocking]

    if non_blocking:
        print(f"\nOther findings ({len(non_blocking)}):")
        for finding in non_blocking:
            _print_finding(finding)

    if blocking:
        print(f"\nHIGH/CRITICAL findings ({len(blocking)}):")
        for finding in blocking:
            _print_finding(finding)
        print(
            f"\nFAIL: {len(blocking)} non-allowlisted HIGH/CRITICAL finding(s) "
            f"in `{skill}`. Fix the skill, or, if this is a documented false "
            "positive, add an entry to .github/skillspector-allow.yml."
        )
        return 1

    print(f"\nPASS: no non-allowlisted HIGH/CRITICAL findings in `{skill}`.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to the SkillSpector JSON report (scan with `--format json`).",
    )
    parser.add_argument(
        "--skill",
        required=True,
        help="Skill directory name being gated (selects allowlist entries).",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path(".github/skillspector-allow.yml"),
        help="Path to the allowlist YAML (default: .github/skillspector-allow.yml).",
    )
    args = parser.parse_args(argv)

    try:
        findings = load_report(args.report)
        allow_entries = load_allowlist(args.allowlist, args.skill)
    except GateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    kept, suppressed, used_entry_ids = partition(findings, allow_entries)
    return report_results(args.skill, kept, suppressed, allow_entries, used_entry_ids)


if __name__ == "__main__":
    raise SystemExit(main())
