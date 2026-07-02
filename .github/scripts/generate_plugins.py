#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Generate the committed plugin tree from the canonical `skills/` catalog.

A repository becomes a Claude Code plugin marketplace when it has a
`.claude-plugin/marketplace.json` at its root and one or more *plugins* inside
it. Each plugin is a self-contained folder with its own
`.claude-plugin/plugin.json` and a `skills/` directory holding the skills it
ships. This script materializes that structure for every *published* plugin:

    plugins/<name>/
      .claude-plugin/
        plugin.json          # generated: name, description, version, author, ...
      skills/
        <name>/              # a copy of skills/<name>/ (minus .federated.json)
          SKILL.md
          ...

`skills/` stays the single source of truth (the "canonical catalog, curated
publish" pattern): authors and the federation importer only ever touch
`skills/`, and this script packages the published subset into installable
plugins. A skill is "published" when it has an entry in
`.claude-plugin/marketplace.json` whose `source` is `./plugins/<name>`; a skill
with no such entry simply stays unpublished and gets no plugin folder.

Because Claude Code copies each plugin folder on install, a plugin's skills
must physically live inside it -- relative `../` paths that escape the plugin
do not survive installation. The skill content under `plugins/` is therefore a
duplicate of `skills/` by design, kept in sync automatically and verified in CI
with `--check`.

Sources of truth:
- `plugin-metadata.json` (repo root): shared identity (version, author,
  homepage, repository, license) stamped into every generated `plugin.json`.
- `.claude-plugin/marketplace.json`: the published plugin entries (name +
  human-readable description). Hand-maintained.
- `skills/<name>/`: the actual skill content that gets copied in.

Usage:
    uv run .github/scripts/generate_plugins.py            # write the plugin tree
    uv run .github/scripts/generate_plugins.py --check    # verify it is up to date

`--check` fails if any generated file is missing, stale, or orphaned (present
under `plugins/` but no longer produced by the generator).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PLUGIN_METADATA = ROOT / "plugin-metadata.json"
CLAUDE_MARKETPLACE = ROOT / ".claude-plugin" / "marketplace.json"
SKILLS_DIR = ROOT / "skills"
PLUGINS_DIR = ROOT / "plugins"

# Every plugin `source` in the marketplace points at a folder under this prefix.
PLUGINS_SOURCE_PREFIX = "./plugins/"

# Internal bookkeeping that should not ship inside a published plugin. The
# federation marker records where a vendored skill came from; it is useful in
# the canonical `skills/` tree but noise in the installable copy.
EXCLUDED_SKILL_FILES = {".federated.json"}

# `plugin.json` fields copied verbatim from plugin-metadata.json when present.
INHERITED_METADATA_FIELDS = ("homepage", "repository", "license")


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def render_json(data: dict) -> bytes:
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def plugin_source_name(entry: dict) -> str | None:
    """Return the plugin folder name for a marketplace entry, or None.

    Only entries whose `source` points under `./plugins/` are materialized by
    this script. The folder name is taken from the `source` path so the
    generated tree always matches what the marketplace advertises.
    """
    source = entry.get("source")
    if not isinstance(source, str) or not source.startswith(PLUGINS_SOURCE_PREFIX):
        return None
    return source[len(PLUGINS_SOURCE_PREFIX) :].strip("/")


def build_plugin_manifest(entry: dict, metadata: dict) -> dict:
    """Assemble a plugin.json from a marketplace entry + shared metadata."""
    manifest: dict = {
        "name": entry["name"],
        "description": entry.get("description", ""),
        "version": metadata.get("version"),
        "author": metadata.get("author"),
    }
    for field in INHERITED_METADATA_FIELDS:
        value = metadata.get(field)
        if value:
            manifest[field] = value
    # Drop keys that ended up empty so the manifest stays clean.
    return {k: v for k, v in manifest.items() if v not in (None, "")}


def collect_desired_files(metadata: dict, marketplace: dict) -> dict[Path, bytes]:
    """Compute the full set of files the `plugins/` tree should contain.

    Returns a mapping of absolute path -> file contents (bytes). Skill files are
    copied byte-for-byte so the published copy is identical to the canonical
    skill (line endings included).
    """
    plugins = marketplace.get("plugins")
    if not isinstance(plugins, list):
        raise ValueError(
            f"{CLAUDE_MARKETPLACE.relative_to(ROOT)}: top-level `plugins` array "
            "is missing or not a list."
        )

    desired: dict[Path, bytes] = {}
    for idx, entry in enumerate(plugins):
        if not isinstance(entry, dict):
            raise ValueError(f"plugins[{idx}] must be an object.")
        name = plugin_source_name(entry)
        if name is None:
            raise ValueError(
                f"plugins[{idx}] (`{entry.get('name')}`) must set "
                f"`source` to `{PLUGINS_SOURCE_PREFIX}<name>`."
            )
        if entry.get("name") != name:
            raise ValueError(
                f"plugins[{idx}]: `name` ({entry.get('name')!r}) must match the "
                f"plugin folder in `source` ({name!r})."
            )

        skill_dir = SKILLS_DIR / name
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            raise FileNotFoundError(
                f"Plugin `{name}` has no skill at skills/{name}/SKILL.md."
            )

        plugin_root = PLUGINS_DIR / name
        manifest = build_plugin_manifest(entry, metadata)
        desired[plugin_root / ".claude-plugin" / "plugin.json"] = render_json(manifest)

        skill_dest_root = plugin_root / "skills" / name
        for src in sorted(skill_dir.rglob("*")):
            if not src.is_file():
                continue
            if src.name in EXCLUDED_SKILL_FILES:
                continue
            rel = src.relative_to(skill_dir)
            desired[skill_dest_root / rel] = src.read_bytes()

    return desired


def existing_files() -> set[Path]:
    if not PLUGINS_DIR.exists():
        return set()
    return {p for p in PLUGINS_DIR.rglob("*") if p.is_file()}


def prune_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def check(desired: dict[Path, bytes]) -> list[str]:
    problems: list[str] = []
    for path, content in sorted(desired.items()):
        if not path.exists():
            problems.append(f"missing: {path.relative_to(ROOT).as_posix()}")
        elif path.read_bytes() != content:
            problems.append(f"stale:   {path.relative_to(ROOT).as_posix()}")
    for path in sorted(existing_files() - set(desired)):
        problems.append(f"orphan:  {path.relative_to(ROOT).as_posix()}")
    return problems


def write(desired: dict[Path, bytes]) -> int:
    written = 0
    for path, content in sorted(desired.items()):
        if path.exists() and path.read_bytes() == content:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        written += 1

    removed = 0
    for path in sorted(existing_files() - set(desired)):
        path.unlink()
        removed += 1

    prune_empty_dirs(PLUGINS_DIR)
    return written + removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the plugins/ tree is up to date without writing.",
    )
    args = parser.parse_args(argv)

    metadata = load_json(PLUGIN_METADATA)
    marketplace = load_json(CLAUDE_MARKETPLACE)
    desired = collect_desired_files(metadata, marketplace)

    if args.check:
        problems = check(desired)
        if problems:
            print("plugins/ tree is out of date:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            print(
                "\nRun: uv run .github/scripts/generate_plugins.py",
                file=sys.stderr,
            )
            return 1
        print(f"plugins/ tree is up to date ({len(desired)} file(s)).")
        return 0

    changed = write(desired)
    plugin_count = sum(
        1 for p in desired if p.name == "plugin.json" and p.parent.name == ".claude-plugin"
    )
    if changed:
        print(f"Wrote plugins/ tree: {plugin_count} plugin(s), {changed} file(s) updated.")
    else:
        print(f"plugins/ tree already up to date ({plugin_count} plugin(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
