#!/usr/bin/env python3
"""
Refresh the bundled route-policy schema from the Lemonade repository.

scripts/validate.py reads its whole vocabulary - allowed keys at every level,
condition names, classifier types and their per-type required/forbidden
fields, value ranges - from scripts/route_policy.schema.json. That file is a
verbatim copy of the server's own schema
(src/cpp/resources/schemas/route_policy.schema.json in lemonade-sdk/lemonade),
so keeping the skill in step with the router is a matter of re-running this
script when Lemonade ships router changes.

Pin --ref to the release your server runs (e.g. v11.8.1, v2026.43.3) so the
validator accepts exactly what that server accepts; `main` may carry features
no release has yet.

Usage:
    python3 scripts/sync_schema.py                 # fetch main, write if changed
    python3 scripts/sync_schema.py --ref v11.8.1   # fetch a release tag
    python3 scripts/sync_schema.py --check         # exit 1 if bundled copy is stale

Needs network access to raw.githubusercontent.com. No extra packages.
"""

import argparse
import datetime
import json
import os
import sys
import urllib.request

REPO = "lemonade-sdk/lemonade"
SCHEMA_REPO_PATH = "src/cpp/resources/schemas/route_policy.schema.json"
HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(HERE, "route_policy.schema.json")
SOURCE_PATH = os.path.join(HERE, "route_policy.schema.source.json")

sys.path.insert(0, HERE)
from validate import load_vocab  # noqa: E402  (sibling module)


def fetch(ref):
    url = f"https://raw.githubusercontent.com/{REPO}/{ref}/{SCHEMA_REPO_PATH}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    schema = json.loads(raw)
    if schema.get("properties", {}).get("recipe", {}).get("const") != "collection.router":
        raise ValueError(f"{url} does not look like the collection.router schema")
    return url, raw, schema


def vocab_diff(old_schema, new_schema):
    """Human-readable lines for every vocabulary set that changed."""
    if old_schema is None:
        return ["(no bundled schema yet)"]
    old, new = load_vocab(old_schema), load_vocab(new_schema)
    lines = []
    for name in sorted(new):
        a, b = old.get(name), new[name]
        if not isinstance(b, (set, frozenset)) or a == b:
            continue
        added, removed = sorted(b - (a or set())), sorted((a or set()) - b)
        if added:
            lines.append(f"{name}: + {', '.join(added)}")
        if removed:
            lines.append(f"{name}: - {', '.join(removed)}")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", default="main", help="branch or tag of lemonade-sdk/lemonade (default: main)")
    p.add_argument("--check", action="store_true", help="report drift and exit 1 if stale; write nothing")
    args = p.parse_args()

    try:
        url, raw, schema = fetch(args.ref)
    except Exception as e:  # network, 404 on a bad ref, bad JSON
        print(f"error: could not fetch schema at ref '{args.ref}': {e}", file=sys.stderr)
        sys.exit(2)

    old_schema = None
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, encoding="utf-8") as f:
            old_schema = json.load(f)

    if old_schema == schema:
        print(f"Bundled schema is up to date with {REPO}@{args.ref}.")
        sys.exit(0)

    changes = vocab_diff(old_schema, schema)
    print(f"Bundled schema differs from {REPO}@{args.ref}:")
    for line in changes or ["(descriptions/constraints only - no vocabulary change)"]:
        print(f"  {line}")

    if args.check:
        sys.exit(1)

    with open(SCHEMA_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(raw)
    with open(SOURCE_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump({
            "repo": REPO,
            "ref": args.ref,
            "url": url,
            "synced_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }, f, indent=2)
        f.write("\n")
    print(f"Wrote {os.path.relpath(SCHEMA_PATH)}. Update SKILL.md/reference.md for any "
          "new conditions or classifier types listed above.")


if __name__ == "__main__":
    main()
