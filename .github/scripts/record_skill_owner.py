#!/usr/bin/env python3
"""Record an approved product repo in `.github/skill_owners.json`.

Called by the `product-repo-approval` workflow once both the engineering owner
and the product release owner named on an approval issue have approved. The workflow
puts the result up for review as a pull request, so this script only edits the
registry; it never decides whether an approval is valid.

Registry schema:

    {
      "repos": [
        {
          "repo": "AMD-AGI/TraceLens",          # owner/repo
          "engineering_owner": "octocat",       # GitHub handle, no leading @
          "product_release_owner": "octocat",   # GitHub handle, no leading @
        },
        {
          "repo": "ROCm/rocm-systems",
          "path": "projects/rocprofiler-sdk",   # optional, see below
          "engineering_owner": "octocat",
          "product_release_owner": "octocat",
        }
      ]
    }

`path` scopes an approval to one project inside a super-repo that holds many
unrelated ones. Without it the approval covers the whole repo; with it, only
skills under that directory. Each project in a super-repo is approved on its
own, by its own owners.

Entries are keyed on (`repo`, `path`) and kept sorted by it. Approving a scope
that is already listed replaces the existing entry rather than adding a second
one, so a re-approval after an ownership change reads as the current truth.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
PATH_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "skill_owners.json"


def handle(value: str) -> str:
    """Normalize a GitHub handle for storage: no leading @, no stray spaces."""
    return value.strip().lstrip("@").strip()


def scope_path(value: str) -> str:
    """Normalize a subdirectory for storage: POSIX, no leading or trailing `/`.

    Paths are matched against `federation.json` skill paths segment by segment,
    so `.`/`..` segments, which would make that comparison lie, are refused.
    """
    path = value.strip().replace("\\", "/").strip("/")
    if not path:
        return ""
    for segment in path.split("/"):
        if segment in ("", ".", "..") or not PATH_SEGMENT.match(segment):
            raise SystemExit(f"'{value}' is not a plain subdirectory path.")
    return path


def entry_key(entry: dict) -> tuple[str, str]:
    return (entry.get("repo", "").lower(), entry.get("path", ""))


def load_registry(path: Path) -> dict:
    if not path.exists():
        return {"repos": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise SystemExit(f"{path} is not valid JSON: {err}") from err
    if not isinstance(data, dict) or not isinstance(data.get("repos", []), list):
        raise SystemExit(f"{path} must be an object with a 'repos' array.")
    data.setdefault("repos", [])
    return data


def write_registry(path: Path, data: dict) -> None:
    data["repos"].sort(key=entry_key)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="Product repo as owner/repo.")
    parser.add_argument(
        "--path",
        default="",
        help="Subdirectory of the repo to approve. Omit to approve the whole repo.",
    )
    parser.add_argument(
        "--engineering-owner", required=True, help="GitHub handle of the engineering owner."
    )
    parser.add_argument(
        "--product-release-owner",
        required=True,
        help="GitHub handle of the product release owner.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"Registry file to edit (default: {DEFAULT_REGISTRY}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    repo = args.repo.strip()
    if not REPO_PATTERN.match(repo):
        raise SystemExit(f"'{repo}' is not an owner/repo slug.")
    subdir = scope_path(args.path)

    entry = {"repo": repo}
    if subdir:
        entry["path"] = subdir
    entry["engineering_owner"] = handle(args.engineering_owner)
    entry["product_release_owner"] = handle(args.product_release_owner)
    for field in ("engineering_owner", "product_release_owner"):
        if not entry[field]:
            raise SystemExit(f"--{field.replace('_', '-')} cannot be empty.")

    scope = f"{repo}/{subdir}" if subdir else repo
    registry = load_registry(args.registry)
    existing = next((e for e in registry["repos"] if entry_key(e) == entry_key(entry)), None)
    if existing is None:
        registry["repos"].append(entry)
        print(f"Added {scope} to {args.registry.name}.")
    else:
        existing.clear()
        existing.update(entry)
        print(f"Updated the existing {scope} entry in {args.registry.name}.")

    write_registry(args.registry, registry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
