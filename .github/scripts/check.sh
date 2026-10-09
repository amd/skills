#!/usr/bin/env bash
# Local checks that run no agent and spend no inference tokens: OrchestrAI unit
# tests, each skill's structure and eval dataset, markdown references, the
# federation file, and the generated plugin manifests.
#
# The skill checks are skillscope (https://github.com/amd/skillscope), pinned
# to the same version .github/workflows/evals.yml grades this repo with. These
# checks cover CI's local validation; they do not establish that routing or
# behavioral evaluations passed on hardware.
#
# Usage:
#   ./.github/scripts/check.sh              Validate every skill, dataset, and manifest.
#   ./.github/scripts/check.sh --external   Also fetch every external URL our markdown links to.
#   ./.github/scripts/check.sh -h|--help    Print this help.
#
# Requires Python 3.10+ and `uv` (https://github.com/astral-sh/uv).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Keep in step with SKILLSCOPE_VERSION in .github/workflows/evals.yml, which is
# the released version CI grades with.
SKILLSCOPE_VERSION="v0.1.3"
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

python3 -m unittest discover -s .github/scripts -p 'test_orchestrai*.py'
"${SKILLSCOPE[@]}" structural "${SKILLSCOPE_ARGS[@]}"
uv run .github/scripts/federate_skills.py --check-catalog
uv run .github/scripts/validate_marketplace.py
uv run .github/scripts/generate_cursor_marketplace.py --check
uv run .github/scripts/generate_codex_plugin.py --check
