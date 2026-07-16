#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Select which skills SkillSpector should scan based on a git diff.

The SkillSpector per-skill matrix used to fan out over *every* skill on every
run, so a one-line edit to a single skill triggered a full re-scan of the whole
catalog. This script narrows the matrix to just the skills that actually
changed.

Behaviour:

  * Diff the working tree against ``--base`` and collect the changed paths.
  * If any *infra* path changed (this script, the SkillSpector workflow, the
    gate, or the allowlist), scan EVERY skill -- a change to the scanning
    machinery can affect the result for all skills.
  * Otherwise, scan only the skills with changes under ``skills/<name>/``.
  * Print the selected skill names as a compact JSON array (for a CI matrix).
    The array may be empty, in which case there is nothing to scan.

If the base ref is missing or unresolvable (a manual ``workflow_dispatch`` run,
a brand-new branch with no parent, a shallow checkout, ...), fall back to
scanning every skill so we never silently skip a scan.

Usage:

    uv run .github/scripts/changed_skills.py --base "$BASE_SHA"
    uv run .github/scripts/changed_skills.py --base origin/main --skills-dir skills
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SKILLS_DIR = REPO_ROOT / "skills"

# Paths that, when touched, force a full re-scan of every skill because they
# change the scanning machinery itself rather than a single skill's content.
INFRA_PATHS = (
    ".github/workflows/skillspector.yml",
    ".github/scripts/skillspector_gate.py",
    ".github/scripts/changed_skills.py",
    ".github/skillspector-allow.yml",
)


def discover_skills(root: Path) -> list[str]:
    """List skill directory names under `root`, ignoring dotfiles."""
    if not root.exists():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


def _ref_exists(ref: str) -> bool:
    """Return True if `ref` resolves to a commit in the local repo."""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def changed_paths(base: str) -> list[str]:
    """Return paths changed between `base` and the working tree's HEAD.

    Uses a three-dot diff so the comparison is against the merge base of
    `base` and HEAD -- the set of changes introduced on this branch/PR.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def select_skills(base: str | None, skills_dir: Path) -> list[str]:
    """Pick the skills to scan for the given diff base.

    Returns every skill when the base is unusable or when infra changed, and
    only the changed skills otherwise.
    """
    all_skills = discover_skills(skills_dir)

    if not base or not _ref_exists(base):
        print(
            f"Base ref {base!r} is missing or unresolvable; scanning all skills.",
            file=sys.stderr,
        )
        return all_skills

    try:
        paths = changed_paths(base)
    except subprocess.CalledProcessError as exc:
        print(
            f"git diff against {base!r} failed ({exc}); scanning all skills.",
            file=sys.stderr,
        )
        return all_skills

    if any(p in INFRA_PATHS for p in paths):
        print("SkillSpector infra changed; scanning all skills.", file=sys.stderr)
        return all_skills

    prefix = f"{skills_dir.relative_to(REPO_ROOT).as_posix()}/"
    changed: set[str] = set()
    known = set(all_skills)
    for path in paths:
        if not path.startswith(prefix):
            continue
        name = path[len(prefix) :].split("/", 1)[0]
        # Only scan skills that still exist on disk (skip pure deletions).
        if name in known:
            changed.add(name)

    return sorted(changed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base",
        default="",
        help="Git ref/SHA to diff against. Empty or unresolvable means "
        "scan every skill.",
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=DEFAULT_SKILLS_DIR,
        help=f"Directory containing skill folders (default: {DEFAULT_SKILLS_DIR}).",
    )
    args = parser.parse_args(argv)

    skills = select_skills(args.base, args.skills_dir.resolve())
    print(json.dumps(skills, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
