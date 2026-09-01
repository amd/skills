#!/usr/bin/env python3
"""Record an approved product repo in `.github/skill_owners.json`.

Called by the `product-repo-approval` workflow once both the engineering owner
and the product manager named on an approval issue have approved. The workflow
puts the result up for review as a pull request, so this script only edits the
registry; it never decides whether an approval is valid.

Registry schema:

    {
      "repos": [
        {
          "repo": "AMD-AGI/TraceLens",       # owner/repo, unique per entry
          "product": "TraceLens",            # customer-facing product name
          "status": "tech-preview" | "ga",   # product status at approval time
          "engineering_owner": "octocat",    # GitHub handle, no leading @
          "product_manager": "octocat",      # GitHub handle, no leading @
          "approved_in": "<approval issue URL>",
          "approved_at": "2026-09-01T17:04:22Z"
        }
      ]
    }

Entries are keyed on `repo` and kept sorted by it. Approving a repo that is
already listed updates the existing entry in place rather than adding a second
one, so a re-approval after an ownership or status change reads as the current
truth and `approved_at` moves to the later approval.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
STATUSES = ("tech-preview", "ga")
DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "skill_owners.json"


def handle(value: str) -> str:
    """Normalize a GitHub handle for storage: no leading @, no stray spaces."""
    return value.strip().lstrip("@").strip()


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
    data["repos"].sort(key=lambda entry: entry.get("repo", "").lower())
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="Product repo as owner/repo.")
    parser.add_argument("--product", required=True, help="Customer-facing product name.")
    parser.add_argument(
        "--status", required=True, choices=STATUSES, help="Product status at approval time."
    )
    parser.add_argument(
        "--engineering-owner", required=True, help="GitHub handle of the engineering owner."
    )
    parser.add_argument(
        "--product-manager", required=True, help="GitHub handle of the product manager."
    )
    parser.add_argument(
        "--issue", required=True, help="URL of the approval issue the sign-off happened on."
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

    entry = {
        "repo": repo,
        "product": args.product.strip(),
        "status": args.status,
        "engineering_owner": handle(args.engineering_owner),
        "product_manager": handle(args.product_manager),
        "approved_in": args.issue.strip(),
        "approved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for field in ("product", "engineering_owner", "product_manager", "approved_in"):
        if not entry[field]:
            raise SystemExit(f"--{field.replace('_', '-')} cannot be empty.")

    registry = load_registry(args.registry)
    existing = next(
        (e for e in registry["repos"] if e.get("repo", "").lower() == repo.lower()), None
    )
    if existing is None:
        registry["repos"].append(entry)
        print(f"Added {repo} to {args.registry.name} ({entry['status']}).")
    else:
        existing.update(entry)
        print(f"Updated the existing {repo} entry in {args.registry.name} ({entry['status']}).")

    write_registry(args.registry, registry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
