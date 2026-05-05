#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0"]
# ///
"""Validate AMD skills against the standardized Agent Skills format.

Enforces the rules documented in CONTRIBUTING.md:

  - SKILL.md exists at the skill root
  - YAML frontmatter is parseable
  - `name` is lowercase-with-hyphens, <=64 chars, no `anthropic`/`claude`
    substrings, and matches the directory name
  - `description` is a non-empty string <=1024 chars
  - SKILL.md body is <=500 lines

Also validates that `.claude-plugin/marketplace.json` is in sync with the
skills on disk: every skill must have a marketplace entry, and every
marketplace entry must point at an existing skill.

Run from the repo root:

    ./scripts/check.sh                          # used by CI; thin wrapper
    uv run scripts/validate_skills.py           # ad-hoc
    uv run scripts/validate_skills.py --skills-dir skills

Exits non-zero if any skill fails validation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SKILLS_DIR = REPO_ROOT / "skills"
CLAUDE_MARKETPLACE = REPO_ROOT / ".claude-plugin" / "marketplace.json"

# Limits from CONTRIBUTING.md and the standardized Agent Skills format.
MAX_NAME_LEN = 64
MAX_DESCRIPTION_LEN = 1024
MAX_BODY_LINES = 500

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
FRONTMATTER_RE = re.compile(
    r"\A---\r?\n(?P<frontmatter>.*?)\r?\n---\r?\n?(?P<body>.*)\Z",
    re.DOTALL,
)
RESERVED_NAME_SUBSTRINGS = ("anthropic", "claude")


@dataclass
class SkillReport:
    skill: str
    errors: list[str] = field(default_factory=list)


def validate_skill(skill_dir: Path) -> SkillReport:
    """Run every validation rule against `skill_dir` and return a report."""
    report = SkillReport(skill=skill_dir.name)
    skill_md = skill_dir / "SKILL.md"

    if not skill_md.exists():
        report.errors.append("Missing SKILL.md.")
        return report

    text = skill_md.read_text(encoding="utf-8")
    match = FRONTMATTER_RE.match(text)
    if match is None:
        report.errors.append(
            "SKILL.md must start with a `---` YAML frontmatter block "
            "followed by `---` on its own line."
        )
        return report

    try:
        frontmatter = yaml.safe_load(match.group("frontmatter"))
    except yaml.YAMLError as exc:
        report.errors.append(f"YAML frontmatter is invalid: {exc}")
        return report

    if not isinstance(frontmatter, dict):
        report.errors.append(
            "YAML frontmatter must be a mapping with at least `name` "
            "and `description`."
        )
        return report

    _validate_name(frontmatter.get("name"), skill_dir.name, report)
    _validate_description(frontmatter.get("description"), report)
    _validate_body(match.group("body"), report)
    return report


def _validate_name(name: object, dir_name: str, report: SkillReport) -> None:
    if not isinstance(name, str) or not name:
        report.errors.append("Frontmatter `name` is missing or not a non-empty string.")
        return

    if len(name) > MAX_NAME_LEN:
        report.errors.append(
            f"`name` length {len(name)} exceeds {MAX_NAME_LEN} characters."
        )
    if not NAME_RE.match(name):
        report.errors.append(
            f"`name` `{name}` must be lowercase-with-hyphens "
            "(letters, digits, single hyphens between segments)."
        )
    for sub in RESERVED_NAME_SUBSTRINGS:
        if sub in name.lower():
            report.errors.append(f"`name` may not contain `{sub}`.")
    if name != dir_name:
        report.errors.append(
            f"`name` `{name}` must match the skill directory name `{dir_name}`."
        )


def _validate_description(description: object, report: SkillReport) -> None:
    if not isinstance(description, str) or not description:
        report.errors.append(
            "Frontmatter `description` is missing or not a non-empty string."
        )
        return
    if len(description) > MAX_DESCRIPTION_LEN:
        report.errors.append(
            f"`description` length {len(description)} exceeds "
            f"{MAX_DESCRIPTION_LEN} characters."
        )


def _validate_body(body: str, report: SkillReport) -> None:
    # Skip surrounding blank lines so the blank line after `---` doesn't
    # inflate the count.
    lines = body.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) > MAX_BODY_LINES:
        report.errors.append(
            f"SKILL.md body is {len(lines)} lines; max is {MAX_BODY_LINES}. "
            "Move reference material into sibling files (reference.md, "
            "examples.md, ...) and link to them from SKILL.md."
        )


def discover_skills(root: Path) -> list[Path]:
    """List skill directories under `root`, ignoring dotfiles."""
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


def validate_claude_marketplace(skill_dirs: list[Path]) -> list[str]:
    """Return error strings if marketplace entries don't match skills/ on disk.

    The marketplace's human-readable `description` is intentionally allowed
    to differ from the SKILL.md description (per CONTRIBUTING.md), so this only
    enforces that names and source paths line up.
    """
    if not CLAUDE_MARKETPLACE.exists():
        return [
            f"Missing {CLAUDE_MARKETPLACE.relative_to(REPO_ROOT)}; expected one "
            "entry per skill."
        ]

    try:
        data = json.loads(CLAUDE_MARKETPLACE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{CLAUDE_MARKETPLACE.relative_to(REPO_ROOT)}: invalid JSON: {exc}"]

    plugins = data.get("plugins") if isinstance(data, dict) else None
    if not isinstance(plugins, list):
        return [
            f"{CLAUDE_MARKETPLACE.relative_to(REPO_ROOT)}: top-level `plugins` "
            "array is missing."
        ]

    errors: list[str] = []
    skill_names = {p.name for p in skill_dirs}
    listed_names: set[str] = set()

    for idx, entry in enumerate(plugins):
        if not isinstance(entry, dict):
            errors.append(f"plugins[{idx}] must be an object.")
            continue
        name = entry.get("name")
        source = entry.get("source")
        description = entry.get("description")

        if not isinstance(name, str) or not name:
            errors.append(f"plugins[{idx}] is missing a non-empty `name`.")
            continue
        listed_names.add(name)

        if name not in skill_names:
            errors.append(
                f"plugins[{idx}] (`{name}`) has no matching directory under skills/."
            )
            continue

        expected_source = f"./skills/{name}"
        if source != expected_source:
            errors.append(
                f"plugins[{idx}] (`{name}`): `source` must be `{expected_source}`, "
                f"got `{source}`."
            )
        if not isinstance(description, str) or not description.strip():
            errors.append(
                f"plugins[{idx}] (`{name}`) is missing a non-empty `description`."
            )

    for missing in sorted(skill_names - listed_names):
        errors.append(
            f"skills/{missing} has no entry in "
            f"{CLAUDE_MARKETPLACE.relative_to(REPO_ROOT)}."
        )

    return errors


def run(skills_dir: Path) -> int:
    skills = discover_skills(skills_dir)
    if not skills:
        print(f"No skills found under {skills_dir}", file=sys.stderr)
        return 1

    print(f"Validating {len(skills)} skill(s) in {skills_dir}\n")
    total_errors = 0
    for skill_dir in skills:
        report = validate_skill(skill_dir)
        status = "OK  " if not report.errors else "FAIL"
        print(f"[{status}] {report.skill}")
        for err in report.errors:
            print(f"        {err}")
        total_errors += len(report.errors)

    marketplace_errors = validate_claude_marketplace(skills)
    marketplace_status = "OK  " if not marketplace_errors else "FAIL"
    print(f"\n[{marketplace_status}] .claude-plugin/marketplace.json")
    for err in marketplace_errors:
        print(f"        {err}")
    total_errors += len(marketplace_errors)

    print(f"\nSummary: {total_errors} error(s) across {len(skills)} skill(s)")
    return 0 if total_errors == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=DEFAULT_SKILLS_DIR,
        help=f"Directory containing skill folders (default: {DEFAULT_SKILLS_DIR}).",
    )
    args = parser.parse_args(argv)
    return run(args.skills_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
