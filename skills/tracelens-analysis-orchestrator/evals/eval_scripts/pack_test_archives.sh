#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -euo pipefail

# ---------------------------------------------------------------------------
# Pack analysis_tests/{unit,e2e}_tests_{standalone,comparative} into tar.gz
# archives that generate_ref.sh and run_repeatability_parallel.sh extract with:
#   tar xzf <archive> -C "$EVALS_DIR" --strip-components=2
#
# Usage (from repo root):
#   bash TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/eval_scripts/pack_test_archives.sh
#   bash TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/eval_scripts/pack_test_archives.sh standalone
#   bash TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/eval_scripts/pack_test_archives.sh unit comparative
#   bash TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/eval_scripts/pack_test_archives.sh unit_tests_standalone
# ---------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: bash pack_test_archives.sh [suite|scope|archive_name ...]

With no arguments, packs all four archives:
  unit_tests_standalone.tar.gz
  unit_tests_comparative.tar.gz
  e2e_tests_standalone.tar.gz
  e2e_tests_comparative.tar.gz

Arguments can be mixed:
  standalone | comparative     both unit and e2e for that scope
  unit | e2e                   both scopes for that suite
  unit_tests_standalone        that archive only (same for the other three names)

Archives are written next to the source directories under
TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/analysis_tests/. Intermediate analysis_output/ trees and
analysis_stream.ndjson files are excluded; analysis_output_ref/ is kept.
EOF
}

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
EVALS_DIR="$REPO_ROOT/TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals"
TESTS_DIR="$EVALS_DIR/analysis_tests"

ALL_ARCHIVES=(
    unit_tests_standalone
    unit_tests_comparative
    e2e_tests_standalone
    e2e_tests_comparative
)

if [[ ! -d "$TESTS_DIR" ]]; then
    echo "ERROR: test directory not found: $TESTS_DIR" >&2
    echo "Run this script from the TraceLens repo root (or set REPO_ROOT)." >&2
    exit 1
fi

resolve_names() {
    local arg suite scope name
    local -a selected=()

    if [[ $# -eq 0 ]]; then
        printf '%s\n' "${ALL_ARCHIVES[@]}"
        return 0
    fi

    for arg in "$@"; do
        case "$arg" in
            -h|--help)
                usage
                exit 0
                ;;
            standalone|comparative)
                selected+=("unit_tests_${arg}" "e2e_tests_${arg}")
                ;;
            unit|e2e)
                selected+=("${arg}_tests_standalone" "${arg}_tests_comparative")
                ;;
            unit_tests_standalone|unit_tests_comparative|e2e_tests_standalone|e2e_tests_comparative)
                selected+=("$arg")
                ;;
            *.tar.gz)
                name="${arg##*/}"
                name="${name%.tar.gz}"
                selected+=("$name")
                ;;
            *)
                echo "ERROR: Unknown argument '$arg'." >&2
                usage >&2
                exit 1
                ;;
        esac
    done

    # Deduplicate while preserving order
    local -A seen=()
    for name in "${selected[@]}"; do
        [[ -n "${seen[$name]:-}" ]] && continue
        seen[$name]=1
        printf '%s\n' "$name"
    done
}

pack_one() {
    local name="$1"
    local src="$TESTS_DIR/$name"
    local archive="$TESTS_DIR/${name}.tar.gz"
    local rel="analysis_tests/$name"

    if [[ ! -d "$src" ]]; then
        echo "ERROR: source directory not found: $src" >&2
        return 1
    fi

    echo "Packing $rel -> $archive"

    # Archive members keep the legacy agent_evals/Analysis/ prefix, which the
    # extractors strip and the skill's evals.json prompt refers to.
    tar -czf "$archive" \
        --exclude='analysis_output' \
        --exclude='analysis_stream.ndjson' \
        --exclude='__pycache__' \
        --transform='s,^,agent_evals/Analysis/,' \
        -C "$EVALS_DIR" \
        "$rel"

    ls -lh "$archive"
}

mapfile -t NAMES < <(resolve_names "$@")

if [[ ${#NAMES[@]} -eq 0 ]]; then
    echo "ERROR: no archives selected." >&2
    exit 1
fi

echo "Repo root: $REPO_ROOT"
echo ""

failed=0
for name in "${NAMES[@]}"; do
    pack_one "$name" || failed=$((failed + 1))
done

echo ""
if [[ "$failed" -gt 0 ]]; then
    echo "Finished with $failed failure(s)." >&2
    exit 1
fi
echo "Packed ${#NAMES[@]} archive(s)."
