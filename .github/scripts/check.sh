#!/usr/bin/env bash
# Validate every SKILL.md and that generated plugin manifests are up to date.
#
# Usage:
#   ./.github/scripts/check.sh              Validate every skill and check manifests.
#   ./.github/scripts/check.sh -h|--help    Print this help.
#
# Requires `uv` (https://github.com/astral-sh/uv).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

usage() {
  sed -n 's/^# \{0,1\}//p' "${BASH_SOURCE[0]}" | sed -n '/^Usage:/,/^Requires/p'
}

case "${1:-}" in
  "")
    uv run .github/scripts/validate_skills.py
    uv run .github/scripts/generate_plugins.py --check
    uv run .github/scripts/generate_cursor_marketplace.py --check
    ;;
  -h|--help)
    usage
    ;;
  *)
    echo "Unknown option: $1" >&2
    echo "Run with --help for usage." >&2
    exit 2
    ;;
esac
