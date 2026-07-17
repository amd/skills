#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Select which behavioral tests to run, by skill name.

Behavioral tests live with each skill (see CONTRIBUTING.md) at:

    skills/<skill>/evals/evals.py

Skill names are lowercase-with-hyphens. Keeping each test beside the skill
makes the behavioral coverage part of the skill's own source tree.

This script maps a set of changed files (read from stdin, one path per line)
to the skills whose behavioral test should run, and is also used to enumerate
every testable skill for manual / full runs.

Output is always a JSON array of skill names on stdout, suitable for a GitHub
Actions matrix:

    uv run .github/scripts/select_behavioral.py --all
    uv run .github/scripts/select_behavioral.py --names "local-ai-use,rocm-doctor"
    git diff --name-only BASE HEAD | uv run .github/scripts/select_behavioral.py --changed

A skill is "testable" only when both its test file and its skill folder exist;
that keeps the matrix honest if a test is added before its skill (or vice
versa).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SKILLS_DIR = REPO_ROOT / "skills"

EVAL_PATH = Path("evals") / "evals.py"

# Touching any of these means the shared harness (not one skill) changed, so we
# re-run every behavioral test rather than trying to guess the blast radius.
# Paths are repo-root-relative and use forward slashes to match `git diff`.
INFRA_FILES = {
    "eval/behavioral/harness.py",
    "eval/behavioral/conftest.py",
    "eval/behavioral/pytest.ini",
    "eval/behavioral/requirements.txt",
    "eval/claude_eval.py",
    ".github/scripts/select_behavioral.py",
    ".github/workflows/behavioral.yml",
}


def is_testable(skill: str) -> bool:
    """A skill is testable when both its test file and skill folder exist."""
    skill_dir = SKILLS_DIR / skill
    has_test = (skill_dir / EVAL_PATH).is_file()
    has_skill = (skill_dir / "SKILL.md").is_file()
    return has_test and has_skill


def all_testable_skills() -> list[str]:
    """Every skill that currently has a behavioral test and a skill folder."""
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(
        path.name
        for path in SKILLS_DIR.iterdir()
        if path.is_dir() and is_testable(path.name)
    )


def select_from_changes(changed: list[str]) -> list[str]:
    """Map changed file paths to the testable skills they affect."""
    normalized = {p.strip().replace("\\", "/") for p in changed if p.strip()}

    # Shared-harness change: run the whole suite.
    if normalized & INFRA_FILES:
        return all_testable_skills()

    selected = set()
    for path in normalized:
        # A change inside skills/<name>/...
        if path.startswith("skills/"):
            parts = path.split("/")
            if len(parts) >= 2 and is_testable(parts[1]):
                selected.add(parts[1])
    return sorted(selected)


def select_from_names(names: str) -> list[str]:
    """Filter an explicit, comma-separated skill list down to testable ones."""
    requested = [n.strip() for n in names.split(",") if n.strip()]
    return sorted({n for n in requested if is_testable(n)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--all",
        action="store_true",
        help="Print every skill that has a behavioral test.",
    )
    mode.add_argument(
        "--changed",
        action="store_true",
        help="Read changed file paths from stdin and print the affected skills.",
    )
    mode.add_argument(
        "--names",
        metavar="A,B,C",
        help="Print the testable subset of this comma-separated skill list.",
    )
    args = parser.parse_args(argv)

    if args.all:
        skills = all_testable_skills()
    elif args.names is not None:
        skills = select_from_names(args.names)
    else:
        skills = select_from_changes(sys.stdin.read().splitlines())

    print(json.dumps(skills))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
