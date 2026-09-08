#!/usr/bin/env bash
# Every check that runs no agent, clones nothing, and spends no tokens: each
# skill's structure and eval dataset, the references our markdown makes, the
# federation file, and the generated plugin manifests.
#
# The skill checks are skillscope (https://github.com/amd/skillscope), pinned
# to the same version .github/workflows/evals.yml grades this repo with, so a
# green run here means the same thing CI's `results` check does. Bump both
# together. The rest is this repo's own.
#
# Usage:
#   ./.github/scripts/check.sh              Validate every skill, dataset, and manifest.
#   ./.github/scripts/check.sh --external   Also fetch every external URL our markdown links to.
#   ./.github/scripts/check.sh -h|--help    Print this help.
#
# Requires `uv` (https://github.com/astral-sh/uv).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Keep in step with `version:` in .github/workflows/evals.yml.
SKILLSCOPE_VERSION="v0.1.0"
SKILLSCOPE=(uv tool run --from "git+https://github.com/amd/skillscope@${SKILLSCOPE_VERSION}" skillscope)

# Keep in step with the structural inputs in .github/workflows/evals.yml.
SKILLSCOPE_ARGS=(
  --skills-dir 'skills/*'
  --docs '*.md,docs/**/*.md,walkthroughs/*.md,staging/**/*.md'
  --skill-files skill-card.md
  --skill-sections Description,Owner,License
)

usage() {
  sed -n 's/^# \{0,1\}//p' "${BASH_SOURCE[0]}" | sed -n '/^Usage:/,/^Requires/p'
}

case "${1:-}" in
  "")
    ;;
  --external)
    # Other people's servers, so this fails for reasons that have nothing to do
    # with your change. CI keeps it out of the merge gate for that reason.
    SKILLSCOPE_ARGS+=(--external)
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown option: $1" >&2
    echo "Run with --help for usage." >&2
    exit 2
    ;;
esac

"${SKILLSCOPE[@]}" structural "${SKILLSCOPE_ARGS[@]}"
uv run .github/scripts/federate_skills.py --check-catalog
uv run .github/scripts/validate_marketplace.py
uv run .github/scripts/generate_cursor_marketplace.py --check
uv run .github/scripts/generate_codex_plugin.py --check
