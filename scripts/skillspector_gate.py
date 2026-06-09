#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0"]
# ///
"""Gate a SkillSpector markdown report against a documented allowlist.

SkillSpector's static scan is high-recall / moderate-precision, so it emits
the occasional false positive that no amount of legitimate code can avoid
(for example, a YARA `backdoor_persistence` hit on the standard
`echo 'export PATH=...' >> ~/.bashrc` install step). SkillSpector has no
native per-finding suppression, so this script provides one at the CI gate:

  * Parse the markdown report SkillSpector produced for a single skill.
  * Drop any HIGH/CRITICAL finding that matches an entry in the allowlist
    (keyed by skill + rule id + file, with an optional message substring so a
    suppression stays narrowly scoped).
  * Fail (exit 1) only if a HIGH/CRITICAL finding remains after that filter.

This keeps the "fail on any HIGH/CRITICAL" policy intact for everything that
isn't an explicitly justified exception, and it never requires editing the
scanned skill just to dodge a regex.

Usage:

    uv run scripts/skillspector_gate.py \
        --report reports/rocm-doctor.md \
        --skill rocm-doctor \
        --allowlist .github/skillspector-allow.yml

Exits 0 when no un-allowlisted HIGH/CRITICAL findings remain, 1 otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

BLOCKING_SEVERITIES = {"HIGH", "CRITICAL"}

# Matches a finding header such as "### 🔴 HIGH: YR1". The severity and rule id
# are the only stable parts; the emoji between "###" and the severity varies.
_HEADER_RE = re.compile(r"^###\s+.*?\b(LOW|MEDIUM|HIGH|CRITICAL):\s*(\S+)\s*$")
# Matches "**Location:** `scripts/apply_fix.py:673`" (line/range suffix optional).
_LOCATION_RE = re.compile(r"^\*\*Location:\*\*\s*`(?P<loc>[^`]+)`")
# Matches "**Message:** ...".
_MESSAGE_RE = re.compile(r"^\*\*Message:\*\*\s*(?P<msg>.*)$")


@dataclass
class Finding:
    severity: str
    rule: str
    file: str
    message: str


@dataclass
class Suppression:
    skill: str
    rule: str
    file: str
    reason: str
    match: str | None = None

    def covers(self, finding: Finding, skill: str) -> bool:
        if self.skill != skill or self.rule != finding.rule:
            return False
        if _normalize(self.file) != _normalize(finding.file):
            return False
        if self.match and self.match.lower() not in finding.message.lower():
            return False
        return True


def _normalize(path: str) -> str:
    """Normalize a report path for comparison (slash direction, surrounding space)."""
    return path.strip().replace("\\", "/")


def _strip_line_suffix(location: str) -> str:
    """Turn 'scripts/apply_fix.py:673' or '...:88-90' into just the file path."""
    return re.sub(r":\d+(?:[\u2013-]\d+)?\s*$", "", location.strip())


def parse_report(text: str) -> list[Finding]:
    """Extract findings from a SkillSpector markdown report."""
    findings: list[Finding] = []
    severity = rule = file = message = None

    def flush() -> None:
        nonlocal severity, rule, file, message
        if severity and rule:
            findings.append(
                Finding(
                    severity=severity,
                    rule=rule,
                    file=_strip_line_suffix(file or ""),
                    message=message or "",
                )
            )
        severity = rule = file = message = None

    for line in text.splitlines():
        header = _HEADER_RE.match(line)
        if header:
            flush()
            severity, rule = header.group(1), header.group(2)
            continue
        if severity is None:
            continue
        loc = _LOCATION_RE.match(line)
        if loc:
            file = loc.group("loc")
            continue
        msg = _MESSAGE_RE.match(line)
        if msg:
            message = msg.group("msg")
    flush()
    return findings


def load_suppressions(path: Path) -> list[Suppression]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("suppressions") or []
    suppressions: list[Suppression] = []
    for i, entry in enumerate(raw):
        try:
            suppressions.append(
                Suppression(
                    skill=entry["skill"],
                    rule=entry["rule"],
                    file=entry["file"],
                    reason=entry["reason"],
                    match=entry.get("match"),
                )
            )
        except (KeyError, TypeError) as exc:
            raise SystemExit(
                f"{path}: suppression #{i} is missing a required field ({exc})."
            )
    return suppressions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="Markdown report path.")
    parser.add_argument("--skill", required=True, help="Skill name the report is for.")
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=Path(__file__).resolve().parent.parent / ".github" / "skillspector-allow.yml",
        help="Path to the suppression allowlist (YAML).",
    )
    args = parser.parse_args()

    if not args.report.exists():
        print(f"Report not found: {args.report}", file=sys.stderr)
        return 1

    findings = parse_report(args.report.read_text(encoding="utf-8"))
    suppressions = [s for s in load_suppressions(args.allowlist) if s.skill == args.skill]

    blocking: list[Finding] = []
    suppressed: list[tuple[Finding, Suppression]] = []
    for finding in findings:
        if finding.severity.upper() not in BLOCKING_SEVERITIES:
            continue
        match = next((s for s in suppressions if s.covers(finding, args.skill)), None)
        if match:
            suppressed.append((finding, match))
        else:
            blocking.append(finding)

    for finding, supp in suppressed:
        print(
            f"ALLOWLISTED {finding.severity} {finding.rule} {finding.file}: {supp.reason}"
        )

    if blocking:
        print(
            f"\n{len(blocking)} un-allowlisted HIGH/CRITICAL finding(s) for "
            f"'{args.skill}'; failing.",
            file=sys.stderr,
        )
        for finding in blocking:
            print(
                f"  {finding.severity} {finding.rule} {finding.file}: {finding.message}",
                file=sys.stderr,
            )
        return 1

    print(
        f"No un-allowlisted HIGH/CRITICAL findings for '{args.skill}' "
        f"({len(suppressed)} allowlisted); passing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
