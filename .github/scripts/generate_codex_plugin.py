#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate the Codex plugin manifest and marketplace from canonical sources.

Codex packages a plugin with a manifest at `.codex-plugin/plugin.json` and
distributes it through a repo marketplace catalog at
`.agents/plugins/marketplace.json`
(https://developers.openai.com/codex/plugins/build). AMD ships the same single
curated `amd-skills` bundle it exposes to Claude and Cursor, so both Codex files
are generated (never hand-maintained) from the same sources the Cursor mirror
uses, keeping every ecosystem in lockstep.

Sources of truth:
- `plugin-metadata.json` (repo root): shared, vendor-neutral identity and
  discovery metadata (name, displayName, description, version, brandColor,
  author, homepage, repository, license, keywords).
- `.claude-plugin/marketplace.json`: the curated bundle entry -- its `skills`
  array (which skills ship) and its human-readable catalog `description` (reused
  as the Codex `longDescription`).

Outputs:
- `.codex-plugin/plugin.json`: the Codex plugin manifest (identity, the curated
  `skills` list, and the `interface` install-surface metadata).
- `.agents/plugins/marketplace.json`: the Codex repo marketplace that points at
  the plugin (`source.path` `./`, the repo root) with its install policy.

Usage:
    uv run .github/scripts/generate_codex_plugin.py            # write
    uv run .github/scripts/generate_codex_plugin.py --check    # validate only

`--check` fails if either generated file is stale or if the Claude marketplace
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
CODEX_PLUGIN = ROOT / ".codex-plugin" / "plugin.json"
CODEX_MARKETPLACE = ROOT / ".agents" / "plugins" / "marketplace.json"

# Codex-specific catalog taxonomy and install policy. These describe how the
# bundle presents in Codex-facing catalogs rather than vendor-neutral identity,
# so they live here instead of plugin-metadata.json.
CATEGORY = "Developer Tools"
# The bundled skills read files/config and can write files, install packages,
# and launch local services; declare both surfaces so install prompts are
# accurate.
CAPABILITIES = ["Read", "Write"]
# `AVAILABLE` (installable, opt-in) + `ON_INSTALL` are the values Codex's own
# build docs use for a local marketplace entry. The bundle needs no auth of its
# own, but `policy.authentication` is required, so use the documented default.
POLICY = {"installation": "AVAILABLE", "authentication": "ON_INSTALL"}
# Brand image used for both the catalog logo and the composer icon. Path is
# relative to the plugin root (the repo root, since source.path is "./"), per
# the guide's path rules.
LOGO = "./assets/amd.png"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_identity_consistency(metadata: dict, claude: dict) -> list[str]:
    """Return error strings if the Claude marketplace top-level identity has
    drifted from the canonical `plugin-metadata.json`.

    Mirrors generate_cursor_marketplace.py: both generators read the same
    canonical files, so both refuse to emit anything while the sources
    disagree.
    """
    errors: list[str] = []

    name = metadata.get("name")
    description = metadata.get("description")
    version = metadata.get("version")

    if claude.get("name") != name:
        errors.append(
            f".claude-plugin/marketplace.json `name` ({claude.get('name')!r}) "
            f"must match plugin-metadata.json `name` ({name!r})."
        )
    claude_description = (claude.get("metadata") or {}).get("description")
    if claude_description != description:
        errors.append(
            ".claude-plugin/marketplace.json metadata.description must match "
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


def get_bundle_entry(claude: dict) -> dict:
    """Return the single curated bundle plugin entry from the Claude manifest."""
    plugins = claude.get("plugins")
    if not isinstance(plugins, list) or not plugins:
        raise ValueError(
            ".claude-plugin/marketplace.json is missing its `plugins` array."
        )
    entry = plugins[0]
    if not isinstance(entry, dict):
        raise ValueError(".claude-plugin/marketplace.json plugins[0] must be an object.")
    return entry


def build_codex_plugin(metadata: dict, bundle: dict) -> dict:
    """Build the `.codex-plugin/plugin.json` manifest.

    Identity and discovery fields come from plugin-metadata.json; the curated
    `skills` list and the human-readable long description come from the Claude
    bundle entry so Codex publishes exactly the same skills as the other
    ecosystems.
    """
    author = metadata.get("author") or {}
    developer_name = author.get("name") if isinstance(author, dict) else None

    manifest: dict = {
        "name": metadata["name"],
        "version": metadata["version"],
        "description": metadata["description"],
        "author": author,
        "homepage": metadata.get("homepage"),
        "repository": metadata.get("repository"),
        "license": metadata.get("license"),
        "keywords": metadata.get("keywords", []),
        # Curate the same folders the Claude bundle publishes. Codex documents
        # `skills` as a directory pointer, but an explicit list keeps the
        # unpublished skills under skills/ out of the shipped bundle.
        "skills": list(bundle.get("skills", [])),
        "interface": {
            "displayName": metadata.get("displayName", metadata["name"]),
            "shortDescription": metadata["description"],
            "longDescription": bundle.get("description", metadata["description"]),
            "developerName": developer_name or metadata["name"],
            "category": CATEGORY,
            "capabilities": list(CAPABILITIES),
            "websiteURL": metadata.get("homepage"),
            "brandColor": metadata.get("brandColor"),
            "logo": LOGO,
            "composerIcon": LOGO,
        },
    }
    # Drop keys that resolved to None so the manifest stays clean when an
    # optional source field is absent.
    return _prune_none(manifest)


def build_codex_marketplace(metadata: dict) -> dict:
    """Build the `.agents/plugins/marketplace.json` repo catalog.

    A single local entry points at the repo root (`./`), where the
    `.codex-plugin/plugin.json` manifest lives, so `codex plugin marketplace
    add amd/skills` exposes the bundle.
    """
    return {
        "name": metadata["name"],
        "interface": {
            "displayName": metadata.get("displayName", metadata["name"]),
        },
        "plugins": [
            {
                "name": metadata["name"],
                "source": {"source": "local", "path": "./"},
                "policy": dict(POLICY),
                "category": CATEGORY,
            }
        ],
    }


def _prune_none(value):
    """Recursively drop dict keys whose value is None."""
    if isinstance(value, dict):
        return {k: _prune_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_prune_none(v) for v in value]
    return value


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
        description="Generate .codex-plugin/plugin.json and "
        ".agents/plugins/marketplace.json from the canonical Claude marketplace "
        "and plugin-metadata.json."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the generated files are up to date without writing.",
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

    bundle = get_bundle_entry(claude)

    targets = [
        (CODEX_PLUGIN, render_json(build_codex_plugin(metadata, bundle))),
        (CODEX_MARKETPLACE, render_json(build_codex_marketplace(metadata))),
    ]

    stale = [
        path for path, content in targets
        if not write_or_check(path, content, check=args.check)
    ]

    if args.check:
        if stale:
            for path in stale:
                print(f"{path.relative_to(ROOT).as_posix()} is out of date.", file=sys.stderr)
            print(
                "Run: uv run .github/scripts/generate_codex_plugin.py",
                file=sys.stderr,
            )
            return 1
        print("Codex plugin and marketplace manifests are up to date.")
        return 0

    for path, _ in targets:
        print(f"Wrote {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
