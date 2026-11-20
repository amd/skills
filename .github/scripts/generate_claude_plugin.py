#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate the plugin manifest under `skills/` from the Claude marketplace.

The Claude plugin directory validates the folder the AMD listing points at
(`skills/`) and expects a plugin manifest there. Claude Code itself never reads
it -- the marketplace bundle's `source` is the repo root with `strict: false`
-- so this manifest only restates the bundle entry. To avoid drift it is
generated, not hand-maintained.

Source of truth:
- `.claude-plugin/marketplace.json`: the bundle entry's `name`, `description`,
  and `skills` array.

Output:
- `skills/.claude-plugin/plugin.json`: the same name, description, and skills,
  with each skill path rebased from `./skills/<name>` to `./<name>` because
  paths resolve relative to `skills/`. `version` and `displayName` are left
  out on purpose.

Usage:
    uv run .github/scripts/generate_claude_plugin.py            # write
    uv run .github/scripts/generate_claude_plugin.py --check    # validate only

`--check` fails if the generated file is stale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CLAUDE_MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"
SKILLS_PLUGIN = ROOT / "skills" / ".claude-plugin" / "plugin.json"
SKILL_PREFIX = "./skills/"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def bundle_entry(claude: dict) -> dict:
    plugins = claude.get("plugins")
    if not isinstance(plugins, list) or not plugins:
        raise ValueError(".claude-plugin/marketplace.json is missing its `plugins` array.")
    entry = plugins[0]
    if not isinstance(entry, dict):
        raise ValueError(".claude-plugin/marketplace.json plugins[0] must be an object.")
    return entry


def rebase_skill_path(path: str) -> str:
    if not path.startswith(SKILL_PREFIX):
        raise ValueError(
            f".claude-plugin/marketplace.json skill path {path!r} must start "
            f"with {SKILL_PREFIX!r}."
        )
    return "./" + path[len(SKILL_PREFIX):]


def build_skills_plugin(claude: dict) -> dict:
    entry = bundle_entry(claude)
    return {
        "name": entry["name"],
        "description": entry["description"],
        "skills": [rebase_skill_path(p) for p in entry.get("skills", [])],
    }


def render_json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def write_or_check(path: Path, content: str, check: bool) -> bool:
    """Return True when the file is already up to date."""
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == content:
        return True
    if check:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate skills/.claude-plugin/plugin.json from the "
        "canonical Claude marketplace."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the generated manifest is up to date without writing.",
    )
    args = parser.parse_args(argv)

    try:
        content = render_json(build_skills_plugin(load_json(CLAUDE_MARKETPLACE)))
    except (KeyError, ValueError) as exc:
        print(f"Cannot build {SKILLS_PLUGIN.relative_to(ROOT)}: {exc}", file=sys.stderr)
        return 1

    up_to_date = write_or_check(SKILLS_PLUGIN, content, check=args.check)

    if args.check:
        if not up_to_date:
            print(
                f"{SKILLS_PLUGIN.relative_to(ROOT)} is out of date.",
                file=sys.stderr,
            )
            print(
                "Run: uv run .github/scripts/generate_claude_plugin.py",
                file=sys.stderr,
            )
            return 1
        print("Claude skills plugin manifest is up to date.")
        return 0

    print(f"Wrote {SKILLS_PLUGIN.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
