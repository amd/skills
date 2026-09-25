#!/usr/bin/env bash
set -uo pipefail

# Tags this process and all descendants with TRACELENS_EVAL_JOB=1.
if [[ "${TRACELENS_EVAL_JOB:-}" != 1 ]]; then
    export TRACELENS_EVAL_JOB=1
    exec bash "$0" "$@"
fi

# ---------------------------------------------------------------------------
# Usage: bash generate_ref.sh [standalone|comparative]
#
# COMPARISON_SCOPE can also be set via the COMPARISON_SCOPE environment variable.
# Defaults to standalone.
#
# USE_PI=1 (or --pi on run_repeatability_parallel.sh) uses `pi` instead of the
# Cursor `agent` CLI, matching the repeatability harness agent backend.
#
# NUM_REPEATS retries each failed reference generation (default: 3). When invoked
# from run_repeatability_parallel.sh, NUM_REPEATS is passed through from the harness.
# ---------------------------------------------------------------------------
COMPARISON_SCOPE="${1:-${COMPARISON_SCOPE:-standalone}}"

if [[ "$COMPARISON_SCOPE" != "standalone" && "$COMPARISON_SCOPE" != "comparative" ]]; then
    echo "ERROR: Unknown comparison scope '$COMPARISON_SCOPE'. Use 'standalone' or 'comparative'." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Configuration (run from repo root on the node)
# ---------------------------------------------------------------------------
CONTAINER="${CONTAINER:-}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-30}"
LAUNCH_STAGGER="${LAUNCH_STAGGER:-8}"
TEST_IDS="${TEST_IDS:-}"
NUM_REPEATS="${NUM_REPEATS:-3}"
USE_PI="${USE_PI:-false}"
AGENT_MODEL="${AGENT_MODEL:-claude-opus-4-8-thinking-medium}"
PI_VENV_PREFIX="use venv_tracelens for all commands and tool calls. "

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
ANALYSIS_DIR="$REPO_ROOT/TraceLens/Agent/Analysis"
EVALS_DIR="$REPO_ROOT/TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals"
TEST_TRACES_CSV="${TEST_TRACES_CSV:-$EVALS_DIR/analysis_tests/combined_traces_${COMPARISON_SCOPE}.csv}"
STATUS_FILE="$(mktemp)"

if [[ -n "$CONTAINER" ]]; then
    DEXEC=(docker exec -w "$REPO_ROOT" "$CONTAINER")
    RUNTIME_LABEL="container $CONTAINER"
    NODE_LABEL="node $(hostname)"
else
    DEXEC=()
    RUNTIME_LABEL="host (no container)"
    NODE_LABEL="local"
fi

if [[ "$USE_PI" == true || "$USE_PI" == 1 || "$USE_PI" == "1" ]]; then
    USE_PI=true
    AGENT_BACKEND="pi"
else
    USE_PI=false
    AGENT_BACKEND="cursor agent"
fi

# ---------------------------------------------------------------------------
# Auto-extract test archives if trace CSV references them
# ---------------------------------------------------------------------------
for archive in "$EVALS_DIR"/analysis_tests/e2e_tests_${COMPARISON_SCOPE}.tar.gz "$EVALS_DIR"/analysis_tests/unit_tests_${COMPARISON_SCOPE}.tar.gz; do
    [ -f "$archive" ] || continue
    target_dir="${archive%.tar.gz}"
    if [ ! -d "$target_dir" ]; then
        echo "Extracting $(basename "$archive")..."
        # Archives store paths as agent_evals/Analysis/analysis_tests/...
        tar -xzf "$archive" -C "$EVALS_DIR" --strip-components=2
    fi
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ts() { date "+%H:%M:%S"; }

log_status() {
    flock 1 echo "$@"
}

repo_abs_path() {
    local p="$1"
    if [[ "$p" = /* ]]; then
        echo "$p"
    else
        echo "$REPO_ROOT/$p"
    fi
}

# Run an LLM agent step. Optional second arg: non-empty enables a 1800s timeout (cursor only).
run_llm_agent() {
    local prompt="$1"
    local with_timeout="${2:-}"

    if [[ "$USE_PI" == true ]]; then
        pi --mode json "${PI_VENV_PREFIX}${prompt}"
    elif [[ -n "$with_timeout" ]]; then
        timeout 1800 agent --model "$AGENT_MODEL" --print --force --trust --output-format stream-json \
            "$prompt"
    else
        agent --model "$AGENT_MODEL" --print --force --trust --output-format stream-json \
            "$prompt"
    fi
}

# ---------------------------------------------------------------------------
# Single job: generate one golden reference
#
# Args: id trace1_path trace2_path reference_dir platform
# For standalone, trace2_path is empty string "".
# ---------------------------------------------------------------------------

generate_single_ref() {
    local id="$1" trace1_path="$2" trace2_path="$3" reference_dir="$4" platform="$5" platform2="$6" capture_folder1="${7:-}" capture_folder2="${8:-}"
    local tag="[$id]"

    local REF_DIR="$REPO_ROOT/$reference_dir"
    local CASE_DIR
    CASE_DIR="$(dirname "$REF_DIR")"
    local OUTPUT_DIR="$CASE_DIR/analysis_output"

    trace1_path="$(repo_abs_path "$trace1_path")"
    if [[ -n "$trace2_path" ]]; then
        trace2_path="$(repo_abs_path "$trace2_path")"
    fi
    if [[ -n "$capture_folder1" ]]; then
        capture_folder1="$(repo_abs_path "$capture_folder1")"
    fi
    if [[ -n "$capture_folder2" ]]; then
        capture_folder2="$(repo_abs_path "$capture_folder2")"
    fi

    # Verify trace file(s) exist
    if [ ! -f "$trace1_path" ]; then
        log_status "  $tag ERROR: Trace file not found: $trace1_path — skipping."
        flock "$STATUS_FILE" bash -c "echo 'failed' >> '$STATUS_FILE'"
        return 1
    fi
    if [[ "$COMPARISON_SCOPE" == "comparative" ]] && [ ! -f "$trace2_path" ]; then
        log_status "  $tag ERROR: Trace2 file not found: $trace2_path — skipping."
        flock "$STATUS_FILE" bash -c "echo 'failed' >> '$STATUS_FILE'"
        return 1
    fi

    local ref_attempt=0
    while (( ref_attempt < NUM_REPEATS )); do
        ref_attempt=$((ref_attempt + 1))
        if (( ref_attempt > 1 )); then
            log_status "  $tag [$(ts)] Retry $ref_attempt/$NUM_REPEATS..."
            "${DEXEC[@]}" rm -rf "$OUTPUT_DIR" 2>/dev/null || rm -rf "$OUTPUT_DIR"
            rm -f "$CASE_DIR/analysis_stream.ndjson"
            sleep "$SLEEP_BETWEEN"
        else
            log_status "  $tag [$(ts)] Generating golden reference..."
        fi
        "${DEXEC[@]}" bash -c "mkdir -p $OUTPUT_DIR && chmod -R 777 $OUTPUT_DIR"

        # Run analysis with retry + backoff on transient agent errors
        local agent_success=false
        local agent_attempts=0
        while [ "$agent_success" = false ] && [ "$agent_attempts" -lt 3 ]; do
            agent_attempts=$((agent_attempts + 1))
            (
                cd "$ANALYSIS_DIR" || exit
                if [[ "$COMPARISON_SCOPE" == "comparative" ]]; then
                    local capture_suffix=""
                    [[ -n "$capture_folder1" ]] && capture_suffix+=" capture folder for trace1 $capture_folder1"
                    [[ -n "$capture_folder2" ]] && capture_suffix+=" capture folder for trace2 $capture_folder2"
                    local analysis_mode="default"
                    [[ -n "$capture_folder1" || -n "$capture_folder2" ]] && analysis_mode="inference"
                    run_llm_agent \
                        "Follow the analysis orchestrator installed with the TraceLens pip package (look under TraceLens/Agent/Analysis/skills/analysis-orchestrator/ in the package installation directory) and run the full agentic analysis workflow on $trace1_path and $trace2_path${capture_suffix} with platform $platform (trace1) and $platform2 (trace2), analysis mode $analysis_mode, $NODE_LABEL, $RUNTIME_LABEL, output to $OUTPUT_DIR" \
                        1
                else
                    run_llm_agent \
                        "Follow the analysis orchestrator installed with the TraceLens pip package (look under TraceLens/Agent/Analysis/skills/analysis-orchestrator/ in the package installation directory) and run the full agentic analysis workflow on $trace1_path with platform $platform, analysis mode default, $NODE_LABEL, $RUNTIME_LABEL, output to $OUTPUT_DIR" \
                        1
                fi
            ) < /dev/null > "$CASE_DIR/analysis_stream.ndjson" 2>&1

            if [[ "$USE_PI" == true ]]; then
                if head -c 2048 "$CASE_DIR/analysis_stream.ndjson" | grep -qiE 'Error:.*unavailable|Service Unavailable'; then
                    log_status "  $tag Agent attempt $agent_attempts/3 failed (agent unavailable). Backing off 30s..."
                    sleep 30
                else
                    agent_success=true
                fi
            elif grep -qiE 'Error:.*unavailable|Service Unavailable|usage limit|out of usage|You'\''ve reached your' "$CASE_DIR/analysis_stream.ndjson" 2>/dev/null; then
                log_status "  $tag Agent attempt $agent_attempts/3 failed (agent unavailable or usage limit). Backing off 30s..."
                sleep 30
            else
                agent_success=true
            fi
        done

        if [ "$agent_success" = false ]; then
            log_status "  $tag Reference attempt $ref_attempt/$NUM_REPEATS failed (agent unavailable or usage limit)."
            "${DEXEC[@]}" rm -rf "$OUTPUT_DIR" 2>/dev/null || rm -rf "$OUTPUT_DIR"
            rm -f "$CASE_DIR/analysis_stream.ndjson"
            continue
        fi

        if [ ! -f "$OUTPUT_DIR/analysis.md" ]; then
            log_status "  $tag Reference attempt $ref_attempt/$NUM_REPEATS failed (analysis.md not found)."
            "${DEXEC[@]}" rm -rf "$OUTPUT_DIR" 2>/dev/null || rm -rf "$OUTPUT_DIR"
            rm -f "$CASE_DIR/analysis_stream.ndjson"
            continue
        fi

        # Copy output as reference (remove old ref first, then copy contents directly)
        rm -rf "$REF_DIR"
        cp -r "$OUTPUT_DIR" "$REF_DIR"

        # Remove unwanted files from reference dir (keep only analysis.md + perf_report_csvs/)
        rm -rf "$REF_DIR/category_data" \
               "$REF_DIR/category_findings" \
               "$REF_DIR/system_findings" \
               "$REF_DIR/metadata" \
               "$REF_DIR/cache" \
               "$REF_DIR/perf_improvement.png" \
               "$REF_DIR/perf_improvement_base64.txt" \
               "$REF_DIR/plot_data.json" \
               "$REF_DIR/perf_report.xlsx" \
               "$REF_DIR/priority_data.json"

        # Remove intermediate analysis output (docker-owned files need container cleanup)
        "${DEXEC[@]}" rm -rf "$OUTPUT_DIR"
        rm -f "$CASE_DIR/analysis_stream.ndjson"

        log_status "  $tag [$(ts)] Reference saved to $reference_dir (cleaned)"
        flock "$STATUS_FILE" bash -c "echo 'generated' >> '$STATUS_FILE'"
        return 0
    done

    log_status "  $tag FAILED after $NUM_REPEATS reference generation attempt(s)."
    flock "$STATUS_FILE" bash -c "echo 'failed' >> '$STATUS_FILE'"
    return 1
}

# ---------------------------------------------------------------------------
# FIFO semaphore for concurrency control
# ---------------------------------------------------------------------------

FIFO="$(mktemp -u)"
cleanup() {
    rm -f "$FIFO" "$STATUS_FILE"
}

setup_semaphore() {
    mkfifo "$FIFO"
    exec 4<>"$FIFO"
    for ((t = 0; t < MAX_PARALLEL; t++)); do echo >&4; done
    trap cleanup EXIT
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "========================================="
echo "  Golden Reference Generation"
echo "  Mode:         $COMPARISON_SCOPE"
echo "  Agent:        $AGENT_BACKEND"
echo "  Node:         $NODE_LABEL"
echo "  Runtime:      $RUNTIME_LABEL"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Retries:      $NUM_REPEATS"
echo "  CSV:          $TEST_TRACES_CSV"
if [[ -n "$TEST_IDS" ]]; then
    echo "  Test filter:  $TEST_IDS"
fi
echo "========================================="
echo ""

# Return 0 if $id should run given TEST_IDS (empty = all).
# Supports exact match and underscore-delimited prefix (e.g. gemm_01 -> gemm_01_compute_few_tiles).
should_run_id() {
    local id="$1"
    [[ -z "$TEST_IDS" ]] && return 0
    local token
    for token in $TEST_IDS; do
        if [[ "$id" == "$token" || "$id" == "${token}_"* ]]; then
            return 0
        fi
    done
    return 1
}

setup_semaphore

if [[ "$COMPARISON_SCOPE" == "comparative" ]]; then
    # comparative CSV: id,sub_category,trace1_path,trace2_path,reference_dir,platform,platform2,capture_folder1,capture_folder2
    while IFS=, read -r id sub_category trace1_path trace2_path reference_dir platform platform2 capture_folder1 capture_folder2 <&3; do
        [[ -z "$id" ]] && continue
        should_run_id "$id" || continue

        read -u4  # acquire semaphore slot
        (
            generate_single_ref "$id" "$trace1_path" "$trace2_path" "$reference_dir" "$platform" "$platform2" "${capture_folder1:-}" "${capture_folder2:-}" || true
            sleep "$SLEEP_BETWEEN"
            echo >&4  # release semaphore slot
        ) &
        sleep "$LAUNCH_STAGGER"  # stagger agent startup to avoid ~/.cursor/cli-config.json rename race
    done 3< <(tail -n +2 "$TEST_TRACES_CSV"; echo)
else
    # standalone CSV: id,sub_category,trace_path,reference_dir,platform
    while IFS=, read -r id sub_category trace_path reference_dir platform <&3; do
        [[ -z "$id" ]] && continue
        should_run_id "$id" || continue

        read -u4  # acquire semaphore slot
        (
            generate_single_ref "$id" "$trace_path" "" "$reference_dir" "$platform" "" || true
            sleep "$SLEEP_BETWEEN"
            echo >&4  # release semaphore slot
        ) &
        sleep "$LAUNCH_STAGGER"  # stagger agent startup to avoid ~/.cursor/cli-config.json rename race
    done 3< <(tail -n +2 "$TEST_TRACES_CSV"; echo)
fi

wait

# Tally results from status file
generated="$(grep -c '^generated$' "$STATUS_FILE" 2>/dev/null || true)"
failed="$(grep -c '^failed$' "$STATUS_FILE" 2>/dev/null || true)"
generated="${generated:-0}"
failed="${failed:-0}"
total=$(( generated + failed ))

if [[ -n "$TEST_IDS" && "$total" -eq 0 ]]; then
    echo ""
    echo "WARNING: TEST_IDS='$TEST_IDS' matched no trace ids in $TEST_TRACES_CSV." >&2
    echo "  Use exact ids or underscore-delimited prefixes (e.g. gemm_01 -> gemm_01_compute_few_tiles)." >&2
fi

echo ""
echo "========================================="
echo "  Golden Reference Generation Complete"
echo "  Total: $total | Generated: $generated | Failed: $failed"
echo "========================================="
