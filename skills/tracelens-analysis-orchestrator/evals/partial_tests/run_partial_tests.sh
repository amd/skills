#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -uo pipefail

# ---------------------------------------------------------------------------
# Partial-workflow test runner.
#
# Unlike run_repeatability_parallel.sh (which runs the FULL 11-step analysis
# orchestrator per case), this harness runs only PART of a workflow -- just
# far enough to produce the artifact a scripted eval scores against a
# pre-baked reference. The first such test is semantic_purity, which runs the
# semantic-comparison workflow through its "Generate TraceDiff Output" step
# (== main orchestrator through Step 1.S) to produce
#   <OUTPUT_DIR>/tracediff_output/diff_stats.csv
# and scores its LCA-partition purity against the committed gold.
#
# New partial-workflow tests can be added by giving them a distinct value in
# the `workflow` column of partial_test_cases.csv and a matching branch in
# run_single_job below.
#
# Usage: bash run_partial_tests.sh
#   NUM_REPEATS   repeats per case (default 1; raise for a variance study)
#   TEST_IDS      space-separated id whitelist (default: all)
#   MAX_PARALLEL  concurrent jobs (default 3)
#   CONTAINER     optional docker container to exec python in
# ---------------------------------------------------------------------------

MAX_PARALLEL="${MAX_PARALLEL:-3}"
NUM_REPEATS="${NUM_REPEATS:-1}"
TEST_IDS="${TEST_IDS:-}"
CONTAINER="${CONTAINER:-}"
AGENT_MODEL="${AGENT_MODEL:-claude-opus-4-8-thinking-medium}"

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
ANALYSIS_DIR="$REPO_ROOT/TraceLens/Agent/Analysis"
EVALS_DIR="$REPO_ROOT/TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals"
PARTIAL_DIR="$EVALS_DIR/partial_tests"
TEST_CASES_CSV="${TEST_CASES_CSV:-$PARTIAL_DIR/partial_test_cases.csv}"
RESULTS_ROOT="${RESULTS_ROOT:-$PARTIAL_DIR/results}"

if [[ -n "$CONTAINER" ]]; then
    DEXEC=(docker exec -w "$REPO_ROOT" "$CONTAINER")
else
    DEXEC=()
fi

ts() { date "+%H:%M:%S"; }
log_status() { flock 1 echo "$@"; }

# Expand a fixture tarball (members carry a legacy agent_evals/ prefix, stripped on extract).
expand_fixture() {
    local id="$1"
    local archive="$PARTIAL_DIR/fixtures/${id}.tar.gz"
    local target="$PARTIAL_DIR/$id"
    if [[ -f "$archive" ]] && [[ ! -d "$target" ]]; then
        echo "Expanding fixtures/${id}.tar.gz..."
        tar xzf "$archive" -C "$PARTIAL_DIR" --strip-components=3
    fi
}

# ---------------------------------------------------------------------------
# One (test_case, repeat) iteration.
# Args: id workflow repeat trace_a trace_b reference_dir platform_a platform_b
# ---------------------------------------------------------------------------
run_single_job() {
    local id="$1" workflow="$2" repeat="$3" trace_a="$4" trace_b="$5" reference_dir="$6" platform_a="$7" platform_b="$8"
    local tag="[$id|run_$repeat]"

    local CASE_RESULTS="$RESULTS_ROOT/$id/run_${repeat}"
    local OUTPUT_DIR="$CASE_RESULTS/analysis_output"
    rm -rf "$CASE_RESULTS" 2>/dev/null || true
    mkdir -p "$OUTPUT_DIR"

    local abs_a="$REPO_ROOT/$trace_a"
    local abs_b="$REPO_ROOT/$trace_b"
    local abs_ref="$REPO_ROOT/$reference_dir"

    log_status "  $tag [$(ts)] workflow=$workflow starting"

    case "$workflow" in
        semantic_comparison)
            local agent_success=false
            local attempts=0
            while [ "$agent_success" = false ] && [ "$attempts" -lt 3 ]; do
                attempts=$((attempts + 1))
                (
                    cd "$ANALYSIS_DIR" || exit
                    timeout 1800 agent --model "$AGENT_MODEL" --print --force --trust --output-format stream-json \
                        "Follow the semantic-comparison-agent workflow (TraceLens/Agent/Analysis/skills/analysis-orchestrator/agents/semantic-comparison-agent.md). Run it on trace A=$abs_a (platform $platform_a) and trace B=$abs_b (platform $platform_b), output to $OUTPUT_DIR. Run only through Step 3 'Generate TraceDiff Output' so that $OUTPUT_DIR/tracediff_output/diff_stats.csv is produced; do NOT run the full analysis orchestrator or any downstream report/category steps."
                ) < /dev/null > "$CASE_RESULTS/semantic_stream.ndjson" 2>&1

                if head -c 2048 "$CASE_RESULTS/semantic_stream.ndjson" | grep -qiE 'Error:.*unavailable|Service Unavailable'; then
                    log_status "  $tag Attempt $attempts/3 failed (agent unavailable). Backing off 30s..."
                    sleep 30
                else
                    agent_success=true
                fi
            done
            if [ "$agent_success" = false ]; then
                log_status "  $tag FAILED after 3 attempts (agent unavailable). Skipping eval."
                return 1
            fi

            "${DEXEC[@]}" python3 "$PARTIAL_DIR/eval_utils/semantic_partition_scripted_evals.py" \
                --output-dir "$OUTPUT_DIR" --reference-dir "$abs_ref" \
                --results "$CASE_RESULTS/semantic_purity_results.csv" \
                > "$CASE_RESULTS/semantic_purity_eval.log" 2>&1 || true
            ;;
        *)
            log_status "  $tag ERROR: unknown workflow '$workflow'"
            return 1
            ;;
    esac

    log_status "  $tag [$(ts)] done -> $CASE_RESULTS/semantic_purity_results.csv"
}

# ---------------------------------------------------------------------------
# FIFO semaphore for concurrency control
# ---------------------------------------------------------------------------
FIFO="$RESULTS_ROOT/.job_fifo"
cleanup() { rm -f "$FIFO"; }
setup_semaphore() {
    rm -f "$FIFO"; mkfifo "$FIFO"; exec 4<>"$FIFO"
    for ((t = 0; t < MAX_PARALLEL; t++)); do echo >&4; done
    trap cleanup EXIT
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
mkdir -p "$RESULTS_ROOT"

# Expand every fixture referenced by the manifest.
while IFS=, read -r id workflow trace_a trace_b reference_dir platform_a platform_b <&3; do
    [[ -z "$id" ]] && continue
    expand_fixture "$id"
done 3< <(tail -n +2 "$TEST_CASES_CSV"; echo)

echo "========================================="
echo "  Partial-workflow tests"
echo "  Repeats:      $NUM_REPEATS"
echo "  Max parallel: $MAX_PARALLEL"
echo "  CSV:          $TEST_CASES_CSV"
if [[ -n "$TEST_IDS" ]]; then echo "  Test filter:  $TEST_IDS"; fi
echo "========================================="
echo ""

_spawn_jobs() {
    local id="$1" workflow="$2" trace_a="$3" trace_b="$4" reference_dir="$5" platform_a="$6" platform_b="$7"
    if [[ -n "$TEST_IDS" ]]; then
        case " $TEST_IDS " in
            *" $id "*) ;;
            *) return ;;
        esac
    fi
    for ((i = 0; i < NUM_REPEATS; i++)); do
        read -r -u4
        (
            run_single_job "$id" "$workflow" "$i" "$trace_a" "$trace_b" "$reference_dir" "$platform_a" "$platform_b" || true
            echo >&4
            sleep 2  # stagger agent startup to avoid ~/.cursor/cli-config.json rename race
        ) &
        sleep 2
    done
}

setup_semaphore

# manifest: id,workflow,trace_a,trace_b,reference_dir,platform_a,platform_b
while IFS=, read -r id workflow trace_a trace_b reference_dir platform_a platform_b <&3; do
    [[ -z "$id" ]] && continue
    _spawn_jobs "$id" "$workflow" "$trace_a" "$trace_b" "$reference_dir" "$platform_a" "$platform_b"
done 3< <(tail -n +2 "$TEST_CASES_CSV"; echo)

wait

echo ""
echo "========================================="
echo "  Runs finished. Applying regression gate..."
echo "========================================="
"${DEXEC[@]}" python3 "$PARTIAL_DIR/eval_utils/semantic_purity_aggregate.py" \
    --results-root "$RESULTS_ROOT" || true

echo ""
echo "  Results in: $RESULTS_ROOT"
