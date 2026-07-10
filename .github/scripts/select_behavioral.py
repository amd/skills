#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Build the behavioral-test CI matrix from JSON eval files.

Behavioral tests are declared as data at:

    skills/<skill>/evals/evals.json

(see eval/behavioral/evals_common.py for the schema). Each test may request
specific runner labels via its optional ``runners`` field; a skill therefore
fans out into one CI job per (skill, runner-target) pair, where a target is a
concrete set of self-hosted runner labels (e.g. Strix-Halo + Windows).

This script maps a set of changed files (or an explicit skill list, or "all")
to those matrix entries. Output is always a JSON array on stdout, suitable for
a GitHub Actions matrix:

    uv run .github/scripts/select_behavioral.py --all
    uv run .github/scripts/select_behavioral.py --names "local-ai-use,rocm-doctor"
    git diff --name-only BASE HEAD | uv run .github/scripts/select_behavioral.py --changed

Each entry is an object::

    {"skill": "local-ai-use", "target": "strix_halo+Windows",
     "labels": ["self-hosted", "strix_halo", "Windows"]}

`labels` feeds the job's ``runs-on``; `target` is passed to
``run_evals.py --runner-target`` so the job runs only the tests that asked for
that runner. A skill with no eval file (or removed) contributes no entries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Reuse the exact schema / runner-resolution logic the executor uses, so the
# matrix and the runner never disagree on target ids or label sets.
sys.path.insert(0, str(REPO_ROOT / "eval" / "behavioral"))

import evals_common as ec  # noqa: E402

# Touching any of these means the shared harness (not one skill) changed, so we
# rebuild the matrix for every testable skill rather than guessing blast radius.
INFRA_FILES = {
    "eval/behavioral/harness.py",
    "eval/behavioral/run_evals.py",
    "eval/behavioral/evals_common.py",
    "eval/behavioral/conftest.py",
    "eval/behavioral/pytest.ini",
    "eval/behavioral/requirements.txt",
    "eval/claude_eval.py",
    ".github/scripts/select_behavioral.py",
    ".github/workflows/behavioral.yml",
}


def entries_for_skill(skill: str) -> list[dict]:
    """Expand one skill into its (skill, target, labels) matrix entries."""
    data = ec.load_evals(skill)
    out = []
    for tid, labels in sorted(ec.skill_targets(data).items()):
        out.append({"skill": skill, "target": tid, "labels": labels})
    return out


def entries_for_skills(skills: list[str]) -> list[dict]:
    out: list[dict] = []
    for skill in sorted(set(skills)):
        if ec.has_evals(skill):
            out.extend(entries_for_skill(skill))
    return out


def select_from_changes(changed: list[str]) -> list[str]:
    """Map changed file paths to the testable skills they affect."""
    normalized = {p.strip().replace("\\", "/") for p in changed if p.strip()}

    # Shared-harness change: rebuild the whole matrix.
    if normalized & INFRA_FILES:
        return ec.all_testable_skills()

    selected = set()
    for path in normalized:
        # A change anywhere inside skills/<name>/ (skill content OR its evals).
        if path.startswith("skills/"):
            parts = path.split("/")
            if len(parts) >= 2 and ec.has_evals(parts[1]):
                selected.add(parts[1])
    return sorted(selected)


def select_from_names(names: str) -> list[str]:
    """Filter an explicit, comma-separated skill list down to testable ones."""
    requested = [n.strip() for n in names.split(",") if n.strip()]
    return sorted({n for n in requested if ec.has_evals(n)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="Every skill with an eval file.")
    mode.add_argument(
        "--changed",
        action="store_true",
        help="Read changed file paths from stdin and print the affected entries.",
    )
    mode.add_argument("--names", metavar="A,B,C", help="Testable subset of this list.")
    args = parser.parse_args(argv)

    if args.all:
        skills = ec.all_testable_skills()
    elif args.names is not None:
        skills = select_from_names(args.names)
    else:
        skills = select_from_changes(sys.stdin.read().splitlines())

    print(json.dumps(entries_for_skills(skills)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
