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
ships. AMD ships a single curated bundle: one plugin that contains every
published skill, materialized under `plugins/<name>/`:

    plugins/amd-skills/
      .claude-plugin/
        plugin.json          # generated: name, description, version, author, ...
      skills/
        local-ai-use/        # a copy of skills/local-ai-use/ (minus .federated.json)
          SKILL.md ...
        serving-llms-on-instinct/
          SKILL.md ...
        ...                  # one folder per skill listed in plugin-metadata.json

`skills/` stays the single source of truth (the "canonical catalog, curated
publish" pattern): authors and the federation importer only ever touch
`skills/`, and this script packages the curated subset into the installable
plugin. The subset is the `skills` list in `plugin-metadata.json`; a skill that
is not listed simply stays unpublished and is not copied into the bundle.

Because Claude Code copies the plugin folder on install, the plugin's skills
must physically live inside it -- relative `../` paths that escape the plugin
do not survive installation. The skill content under `plugins/` is therefore a
duplicate of `skills/` by design, kept in sync automatically and verified in CI
with `--check`.

Sources of truth:
- `plugin-metadata.json` (repo root): shared identity (version, author,
  homepage, repository, license) stamped into the generated `plugin.json`, plus
  the `skills` list naming which skill folders the bundle ships.
- `.claude-plugin/marketplace.json`: the single plugin entry (name + source +
  the human-readable catalog description). Hand-maintained.
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


def single_plugin_entry(marketplace: dict) -> dict:
    """Return the sole plugin entry from the marketplace, enforcing the bundle.

    AMD ships one curated plugin, so `marketplace.json` must list exactly one
    plugin. Supporting several plugins would need a per-plugin skill list (this
    generator draws the bundle's skills from `plugin-metadata.json`), so we fail
    loudly rather than silently give every plugin the same skills.
    """
    plugins = marketplace.get("plugins")
    if not isinstance(plugins, list):
        raise ValueError(
            f"{CLAUDE_MARKETPLACE.relative_to(ROOT)}: top-level `plugins` array "
            "is missing or not a list."
        )
    if len(plugins) != 1:
        raise ValueError(
            f"{CLAUDE_MARKETPLACE.relative_to(ROOT)}: expected exactly one plugin "
            f"(the AMD bundle), found {len(plugins)}. To ship more than one "
            "plugin, extend the generator with a per-plugin skill list."
        )
    entry = plugins[0]
    if not isinstance(entry, dict):
        raise ValueError("plugins[0] must be an object.")
    return entry


def plugin_source_name(entry: dict) -> str:
    """Return the plugin folder name from a marketplace entry's `source`."""
    source = entry.get("source")
    if not isinstance(source, str) or not source.startswith(PLUGINS_SOURCE_PREFIX):
        raise ValueError(
            f"plugin `{entry.get('name')}` must set `source` to "
            f"`{PLUGINS_SOURCE_PREFIX}<name>`, got {source!r}."
        )
    name = source[len(PLUGINS_SOURCE_PREFIX) :].strip("/")
    if entry.get("name") != name:
        raise ValueError(
            f"plugin `name` ({entry.get('name')!r}) must match the plugin folder "
            f"in `source` ({name!r})."
        )
    return name


def bundled_skills(metadata: dict) -> list[str]:
    """Return the curated list of skill folder names the bundle ships."""
    skills = metadata.get("skills")
    if not isinstance(skills, list) or not skills:
        raise ValueError(
            f"{PLUGIN_METADATA.relative_to(ROOT)} must list the bundled skill "
            "folder names under a non-empty `skills` array."
        )
    seen: set[str] = set()
    for name in skills:
        if not isinstance(name, str) or not name:
            raise ValueError(f"{PLUGIN_METADATA.relative_to(ROOT)}: `skills` entries must be non-empty strings.")
        if name in seen:
            raise ValueError(
                f"{PLUGIN_METADATA.relative_to(ROOT)}: duplicate skill {name!r} in `skills`."
            )
        seen.add(name)
    return skills


def build_plugin_manifest(entry: dict, metadata: dict) -> dict:
    """Assemble a plugin.json from the marketplace entry + shared metadata."""
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
    entry = single_plugin_entry(marketplace)
    name = plugin_source_name(entry)
    plugin_root = PLUGINS_DIR / name

    desired: dict[Path, bytes] = {}
    manifest = build_plugin_manifest(entry, metadata)
    desired[plugin_root / ".claude-plugin" / "plugin.json"] = render_json(manifest)

    for skill_name in bundled_skills(metadata):
        skill_dir = SKILLS_DIR / skill_name
        if not (skill_dir / "SKILL.md").is_file():
            raise FileNotFoundError(
                f"Bundled skill `{skill_name}` has no skills/{skill_name}/SKILL.md."
            )
        skill_dest_root = plugin_root / "skills" / skill_name
        for src in sorted(skill_dir.rglob("*")):
            if not src.is_file() or src.name in EXCLUDED_SKILL_FILES:
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
    skill_count = len(bundled_skills(metadata))

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
        print(f"plugins/ tree is up to date ({len(desired)} file(s), {skill_count} skill(s)).")
        return 0

    changed = write(desired)
    if changed:
        print(
            f"Wrote plugins/ tree: 1 plugin, {skill_count} skill(s), "
            f"{changed} file(s) updated."
        )
    else:
        print(f"plugins/ tree already up to date ({skill_count} skill(s)).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
