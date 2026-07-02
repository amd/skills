#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate the Cursor marketplace manifest from the canonical sources.

Both ecosystems use the same marketplace model: a single curated plugin
(`amd-skills`) whose `source` is the repo root bundles the published skills
listed in its `skills` array (the skill folders ship in place under `skills/`,
so there is no separate plugin tree). The Cursor manifest mirrors the Claude
marketplace one-for-one; to avoid drift it is generated, not hand-maintained.

Sources of truth:
- `plugin-metadata.json` (repo root): shared identity and discovery metadata
  (name, description, version, author, homepage, repository, license,
  keywords). This is the vendor-neutral metadata file, reused by every
  marketplace/manifest target. It is NOT a plugin manifest.
- `.claude-plugin/marketplace.json`: the per-skill plugin entries and their
  human-readable descriptions (hand-maintained, since the catalog blurbs
  intentionally differ from the SKILL.md routing descriptions).

Output:
- `.cursor-plugin/marketplace.json`: a mirror of the Claude marketplace so
  Cursor exposes exactly the same skills as Claude.

Usage:
    uv run .github/scripts/generate_cursor_marketplace.py            # write
    uv run .github/scripts/generate_cursor_marketplace.py --check    # validate only

`--check` fails if the generated file is stale or if the Claude marketplace
top-level identity has drifted from `plugin-metadata.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PLUGIN_METADATA = ROOT / "plugin-metadata.json"
CLAUDE_MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"
CURSOR_MARKETPLACE = ROOT / ".cursor-plugin" / "marketplace.json"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_identity_consistency(metadata: dict, claude: dict) -> list[str]:
    """Return error strings if the Claude marketplace top-level identity has
    drifted from the canonical `plugin-metadata.json`."""
    errors: list[str] = []

    name = metadata.get("name")
    description = metadata.get("description")
    version = metadata.get("version")

    if claude.get("name") != name:
        errors.append(
            f".claude-plugin/marketplace.json `name` ({claude.get('name')!r}) "
            f"must match plugin-metadata.json `name` ({name!r})."
        )
    if claude.get("description") != description:
        errors.append(
            ".claude-plugin/marketplace.json `description` must match "
            "plugin-metadata.json `description`."
        )
    claude_version = (claude.get("metadata") or {}).get("version")
    if claude_version != version:
        errors.append(
            f".claude-plugin/marketplace.json metadata.version "
            f"({claude_version!r}) must match plugin-metadata.json `version` "
            f"({version!r})."
        )
    return errors


def build_cursor_marketplace(metadata: dict, claude: dict) -> dict:
    author = metadata.get("author") or {}
    owner_name = author.get("name") if isinstance(author, dict) else None

    return {
        "name": metadata["name"],
        "owner": {"name": owner_name} if owner_name else {},
        "description": metadata["description"],
        "metadata": {
            "description": metadata["description"],
            "version": metadata["version"],
        },
        "plugins": claude.get("plugins", []),
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
        description="Generate .cursor-plugin/marketplace.json from the "
        "canonical Claude marketplace and plugin-metadata.json."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the generated manifest is up to date without writing.",
    )
    args = parser.parse_args(argv)

    metadata = load_json(PLUGIN_METADATA)
    claude = load_json(CLAUDE_MARKETPLACE)

    identity_errors = check_identity_consistency(metadata, claude)
    if identity_errors:
        print("Marketplace identity is inconsistent:", file=sys.stderr)
        for err in identity_errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    content = render_json(build_cursor_marketplace(metadata, claude))
    up_to_date = write_or_check(CURSOR_MARKETPLACE, content, check=args.check)

    if args.check:
        if not up_to_date:
            print(
                f"{CURSOR_MARKETPLACE.relative_to(ROOT)} is out of date.",
                file=sys.stderr,
            )
            print(
                "Run: uv run .github/scripts/generate_cursor_marketplace.py",
                file=sys.stderr,
            )
            return 1
        print("Cursor marketplace manifest is up to date.")
        return 0

    print(f"Wrote {CURSOR_MARKETPLACE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
