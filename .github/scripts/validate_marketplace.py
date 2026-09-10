#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# ///
"""Validate this catalog's published plugin bundle against the skills on disk.

Whether a skill is *well formed* -- frontmatter, `name` rules, description
length, body length, the governance card, the eval dataset, the references its
markdown makes -- is checked by skillscope, which grades every repo that keeps
skills in a tree the same way. See .github/workflows/evals.yml for how this
repo configures it, and docs/skill-requirements.md for the rules.

What skillscope deliberately does not read is a repo-wide manifest: which
skills a catalog publishes, and where it lists them, is that catalog's
business, and a harness with an opinion about it would be a second place to
update every time a skill ships. So that check stays here.

It validates the single-bundle plugin model: `.claude-plugin/marketplace.json`
lists exactly one plugin whose `source` is the repo root (`./`) with
`strict: false`, and whose `skills` array names the published skill folders as
`./skills/<name>` paths. Each listed path must resolve to a real skill under
`skills/`. Skills that are not listed are allowed -- they are simply
unpublished (the "canonical catalog, curated publish" model), so a skill can
live under `skills/` without shipping. No files are duplicated: the bundle
ships the skill folders in place, so there is no generated `plugins/` tree to
keep in sync.

Run from the repo root:

    ./.github/scripts/check.sh                              # thin wrapper; what CI runs
    uv run .github/scripts/validate_marketplace.py
    uv run .github/scripts/validate_marketplace.py --skills-dir skills

Exits non-zero if the manifest and `skills/` disagree.

One thing this cannot see: the routing room in .github/workflows/evals.yml is
listed by hand, so publishing a skill means adding it there as well. That is
deliberate -- who a skill competes against is what its routing score means, so
it is a visible diff rather than something derived from this file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SKILLS_DIR = REPO_ROOT / "skills"
CLAUDE_MARKETPLACE = REPO_ROOT / ".claude-plugin" / "marketplace.json"

# Every published skill is referenced from the bundle's `skills` array as a
# path of this form, relative to the plugin `source` (the repo root).
SKILLS_PATH_PREFIX = "./skills/"


def discover_skills(root: Path) -> list[Path]:
    """List skill directories under `root`, ignoring dotfiles."""
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


def validate_claude_marketplace(skill_dirs: list[Path]) -> list[str]:
    """Return error strings if the bundle plugin doesn't match skills/ on disk.

    AMD ships a single curated plugin whose `source` is the repo root (`./`)
    with `strict: false` (so no `plugin.json` is needed). Its `skills` array
    lists the published skills as `./skills/<name>` paths; each must resolve to
    a real skill under `skills/`. Skills that are not listed are allowed -- they
    are simply unpublished. The plugin's human-readable `description` is
    intentionally allowed to differ from the SKILL.md descriptions (per
    docs/skill-requirements.md), so this only enforces that names and paths line up.
    """
    if not CLAUDE_MARKETPLACE.exists():
        return [
            f"Missing {CLAUDE_MARKETPLACE.relative_to(REPO_ROOT)}; expected the "
            "AMD bundle plugin entry."
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
    if len(plugins) != 1:
        return [
            f"{CLAUDE_MARKETPLACE.relative_to(REPO_ROOT)}: expected exactly one "
            f"plugin (the AMD bundle), found {len(plugins)}."
        ]

    entry = plugins[0]
    if not isinstance(entry, dict):
        return ["plugins[0] must be an object."]

    errors: list[str] = []
    name = entry.get("name")
    source = entry.get("source")
    description = entry.get("description")

    if not isinstance(name, str) or not name:
        errors.append("plugins[0] is missing a non-empty `name`.")
        return errors

    if source != "./":
        errors.append(
            f"plugins[0] (`{name}`): `source` must be `./` (the repo root is the "
            f"bundle), got `{source}`."
        )
    # With a repo-root source the plugin ships no `plugin.json`, so the entry
    # must declare `strict: false` or Claude Code will look for one and fail.
    if entry.get("strict") is not False:
        errors.append(
            f"plugins[0] (`{name}`): must set `strict` to `false` (no plugin.json "
            "ships with a repo-root source)."
        )
    if not isinstance(description, str) or not description.strip():
        errors.append(f"plugins[0] (`{name}`) is missing a non-empty `description`.")

    errors.extend(_validate_bundle_skills(entry, {p.name for p in skill_dirs}))
    return errors


def _validate_bundle_skills(entry: dict, skill_names: set[str]) -> list[str]:
    """Check the bundle's `skills` paths resolve to real skills under skills/."""
    skills = entry.get("skills")
    if not isinstance(skills, list) or not skills:
        return [
            "plugins[0] `skills` must be a non-empty list of "
            f"`{SKILLS_PATH_PREFIX}<name>` paths naming the published skills."
        ]

    errors: list[str] = []
    seen: set[str] = set()
    for path in skills:
        if not isinstance(path, str) or not path.startswith(SKILLS_PATH_PREFIX):
            errors.append(
                f"`skills` entry {path!r} must be a `{SKILLS_PATH_PREFIX}<name>` path."
            )
            continue
        skill = path[len(SKILLS_PATH_PREFIX) :].strip("/")
        if not skill or "/" in skill:
            errors.append(f"`skills` entry {path!r} must point at a single skill folder.")
            continue
        if skill in seen:
            errors.append(f"`skills` lists `{path}` more than once.")
            continue
        seen.add(skill)
        if skill not in skill_names:
            errors.append(
                f"`skills` names `{path}`, which has no directory under skills/."
            )

    # Skills present under skills/ but absent from `skills` are intentionally
    # unpublished, so there is no error for that difference here.
    return errors


def run(skills_dir: Path) -> int:
    skills = discover_skills(skills_dir)
    if not skills:
        print(f"No skills found under {skills_dir}", file=sys.stderr)
        return 1

    errors = validate_claude_marketplace(skills)
    status = "OK  " if not errors else "FAIL"
    print(f"[{status}] .claude-plugin/marketplace.json")
    for err in errors:
        print(f"        {err}")
    print(
        f"\nSummary: {len(errors)} error(s) in the manifest, "
        f"checked against {len(skills)} skill(s) in {skills_dir}"
    )
    return 0 if not errors else 1


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
