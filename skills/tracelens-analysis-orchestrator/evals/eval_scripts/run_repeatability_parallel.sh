#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -uo pipefail

# Tags this process and all descendants with TRACELENS_EVAL_JOB=1.
if [[ "${TRACELENS_EVAL_JOB:-}" != 1 ]]; then
    export TRACELENS_EVAL_JOB=1
    exec bash "$0" "$@"
fi

# ---------------------------------------------------------------------------
# Usage: bash run_repeatability_parallel.sh [both|standalone|comparative] [--pi]
#    or: bash run_repeatability_parallel.sh --pi [both|standalone|comparative]
#
# Default scope is 'both'. With 'both' (or no scope argument) the script runs
# the full pipeline sequentially:
#   1. Generate golden references for standalone  (via generate_ref.sh)
#   2. Generate golden references for comparative  (via generate_ref.sh)
#   3. Repeatability eval for standalone
#   4. Repeatability eval for comparative
#   5. A single combined pr_report.md + fix_ticket_report.md
#
# Passing 'standalone' or 'comparative' (or COMPARISON_SCOPE=standalone|comparative)
# restricts the run to that one scope only (gen-ref + repeatability + report
# over just that scope). Golden references are always regenerated from scratch
# as local directories under TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/analysis_tests/ before the
# repeatability stage, so the repeatability evals compare against the freshly
# generated references.
#
# --pi  Use `pi` instead of the Cursor `agent` CLI. Also settable via USE_PI=1.
#       Applies to golden-reference generation (via generate_ref.sh), repeatability
#       evals, and post-processing.
#
# CONTAINER is optional. If set, python/setup commands run via
# docker exec -w $REPO_ROOT $CONTAINER ... ; if unset, they run on the host.
# ---------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: bash run_repeatability_parallel.sh [both|standalone|comparative] [--pi]

  both|standalone|comparative   Comparison scope (default: both)
  --pi                          Use pi instead of the Cursor agent CLI (or USE_PI=1)

With 'both' (or no scope argument) the script runs, sequentially:
  1. generate golden references (standalone)  via generate_ref.sh
  2. generate golden references (comparative)  via generate_ref.sh
  3. repeatability eval (standalone)
  4. repeatability eval (comparative)
  5. a single combined pr_report.md + fix_ticket_report.md

To kill a running pipeline (and any orphaned jobs it spawned), use:
  bash eval_scripts/kill_eval_jobs.sh
EOF
}

USE_PI="${USE_PI:-false}"
# Comparison scope: both (default), standalone, or comparative.
SCOPE_FILTER="${COMPARISON_SCOPE:-both}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pi)
            USE_PI=true
            shift
            ;;
        both|standalone|comparative)
            SCOPE_FILTER="$1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument '$1'." >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "$USE_PI" == true || "$USE_PI" == 1 || "$USE_PI" == "1" ]]; then
    USE_PI=true
else
    USE_PI=false
fi

if [[ "$SCOPE_FILTER" != "both" && "$SCOPE_FILTER" != "standalone" && "$SCOPE_FILTER" != "comparative" ]]; then
    echo "ERROR: Unknown comparison scope '$SCOPE_FILTER'. Use 'both', 'standalone', or 'comparative'." >&2
    exit 1
fi

if [[ "$SCOPE_FILTER" == "both" ]]; then
    SCOPES=(standalone comparative)
else
    SCOPES=("$SCOPE_FILTER")
fi

# ---------------------------------------------------------------------------
# Configuration (scope-independent)
# ---------------------------------------------------------------------------
MAX_PARALLEL="${MAX_PARALLEL:-5}"
NUM_REPEATS="${NUM_REPEATS:-3}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-30}"
CONTAINER="${CONTAINER:-}"
TEST_IDS="${TEST_IDS:-}"
SUITE_NAME="${SUITE_NAME:-eval}"
SKIP_GENERATE_REF="${SKIP_GENERATE_REF:-}"
SKIP_POST_PROCESSING="${SKIP_POST_PROCESSING:-}"

AGENT_MODEL="${AGENT_MODEL:-claude-opus-4-8-thinking-medium}"
PI_VENV_PREFIX="use venv_tracelens for all commands and tool calls. "

# Paths (REPO_ROOT may differ from the shell cwd)
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
ANALYSIS_DIR="$REPO_ROOT/TraceLens/Agent/Analysis"
EVALS_DIR="$REPO_ROOT/TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATE_REF_SCRIPT="$SCRIPT_DIR/generate_ref.sh"

# Single combined report + intermediate combined results tree for post-processing.
REPORT_DIR="${REPORT_DIR:-$EVALS_DIR/reports}"
COMBINED_RESULTS_ROOT="${COMBINED_RESULTS_ROOT:-$EVALS_DIR/repeatability_results_combined}"

# Scope-specific globals, (re)set per scope by run_repeatability_for_scope().
COMPARISON_SCOPE=""
RESULTS_ROOT=""
TEST_TRACES_CSV=""

if [[ -n "$CONTAINER" ]]; then
    DEXEC=(docker exec -w "$REPO_ROOT" "$CONTAINER")
    RUNTIME_LABEL="container $CONTAINER"
    NODE_LABEL="node $(hostname)"
else
    DEXEC=()
    RUNTIME_LABEL="host (no container)"
    NODE_LABEL="local"
fi

if [[ "$USE_PI" == true ]]; then
    AGENT_BACKEND="pi"
else
    AGENT_BACKEND="cursor agent"
fi

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

expand_archive() {
    local name="$1"
    local archive="$EVALS_DIR/analysis_tests/${name}.tar.gz"
    local target="$EVALS_DIR/analysis_tests/$name"
    if [[ -f "$archive" ]] && [[ ! -d "$target" ]]; then
        echo "Expanding ${name}.tar.gz..."
        # Archives store paths as agent_evals/Analysis/analysis_tests/...
        tar xzf "$archive" -C "$EVALS_DIR" --strip-components=2
        echo "Done."
    fi
}

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

print_scheduled_tests() {
    local -a scheduled_ids=()
    local id sub_category trace1_path trace2_path trace_path reference_dir platform platform2 capture_folder1 capture_folder2

    if [[ "$COMPARISON_SCOPE" == "comparative" ]]; then
        while IFS=, read -r id sub_category trace1_path trace2_path reference_dir platform platform2 capture_folder1 capture_folder2; do
            [[ -z "$id" ]] && continue
            should_run_id "$id" && scheduled_ids+=("$id")
        done < <(tail -n +2 "$TEST_TRACES_CSV"; echo)
    else
        while IFS=, read -r id sub_category trace_path reference_dir platform; do
            [[ -z "$id" ]] && continue
            should_run_id "$id" && scheduled_ids+=("$id")
        done < <(tail -n +2 "$TEST_TRACES_CSV"; echo)
    fi

    if [[ ${#scheduled_ids[@]} -eq 0 ]]; then
        return
    fi

    echo "Tests to run (${#scheduled_ids[@]}):"
    local scheduled_id
    for scheduled_id in "${scheduled_ids[@]}"; do
        echo "  - $scheduled_id ($NUM_REPEATS repeat(s))"
    done
    echo ""
}

# ---------------------------------------------------------------------------
# Single job: one (test_case, repeat) iteration
#
# Args: id repeat trace1_path trace2_path reference_dir platform platform2
#       capture_folder1 capture_folder2  (comparative mode; empty string if absent)
# ---------------------------------------------------------------------------

run_single_job() {
    local id="$1" repeat="$2" trace1_path="$3" trace2_path="$4" reference_dir="$5" platform="$6" platform2="$7"
    local capture_folder1="${8:-}" capture_folder2="${9:-}"
    local tag="[$id|run_$repeat]"

    log_status "  $tag [$(ts)] Running"

    trace1_path="$(repo_abs_path "$trace1_path")"
    if [[ -n "$trace2_path" ]]; then
        trace2_path="$(repo_abs_path "$trace2_path")"
    fi
    reference_dir="$(repo_abs_path "$reference_dir")"
    if [[ -n "$capture_folder1" ]]; then
        capture_folder1="$(repo_abs_path "$capture_folder1")"
    fi
    if [[ -n "$capture_folder2" ]]; then
        capture_folder2="$(repo_abs_path "$capture_folder2")"
    fi

    local capture_suffix=""
    local analysis_mode="default"
    if [[ -n "$capture_folder1" ]]; then
        capture_suffix+=" capture folder for trace1 $capture_folder1"
    fi
    if [[ -n "$capture_folder2" ]]; then
        capture_suffix+=" capture folder for trace2 $capture_folder2"
    fi
    if [[ -n "$capture_folder1" || -n "$capture_folder2" ]]; then
        analysis_mode="inference"
    fi

    local CASE_RESULTS="$RESULTS_ROOT/$id/run_${repeat}"
    local OUTPUT_DIR="$CASE_RESULTS/analysis_output"
    "${DEXEC[@]}" rm -rf "$CASE_RESULTS" 2>/dev/null || true
    rm -rf "$CASE_RESULTS" 2>/dev/null || true
    mkdir -p "$OUTPUT_DIR"
    "${DEXEC[@]}" bash -c "mkdir -p $OUTPUT_DIR && chmod -R 777 $OUTPUT_DIR"
    "${DEXEC[@]}" bash -c "mkdir -p $CASE_RESULTS && chmod -R 777 $CASE_RESULTS"

    # -- Phase 1: Agent analysis with retry + backoff -----------------------
    log_status "  $tag [$(ts)] Phase 1: analysis starting"

    local agent_success=false
    local agent_attempts=0
    while [ "$agent_success" = false ] && [ "$agent_attempts" -lt 3 ]; do
        agent_attempts=$((agent_attempts + 1))
        (
            cd "$ANALYSIS_DIR" || exit
            if [[ "$COMPARISON_SCOPE" == "comparative" ]]; then
                run_llm_agent \
                    "Follow the analysis orchestrator installed with the TraceLens pip package (look under TraceLens/Agent/Analysis/skills/analysis-orchestrator/ in the package installation directory) and run the full agentic analysis workflow on $trace1_path and $trace2_path${capture_suffix} with platform $platform (trace1) and $platform2 (trace2), analysis mode $analysis_mode, $NODE_LABEL, $RUNTIME_LABEL, output to $OUTPUT_DIR" \
                    1
            else
                run_llm_agent \
                    "Follow the analysis orchestrator installed with the TraceLens pip package (look under TraceLens/Agent/Analysis/skills/analysis-orchestrator/ in the package installation directory) and run the full agentic analysis workflow on $trace1_path with platform $platform, $NODE_LABEL, $RUNTIME_LABEL, output to $OUTPUT_DIR" \
                    1
            fi
        ) < /dev/null > "$CASE_RESULTS/analysis_stream.ndjson" 2>&1

        if head -c 2048 "$CASE_RESULTS/analysis_stream.ndjson" | grep -qiE 'Error:.*unavailable|Service Unavailable'; then
            log_status "  $tag Attempt $agent_attempts/3 failed (agent unavailable). Backing off 30s..."
            sleep 30
        else
            agent_success=true
        fi
    done

    if [ "$agent_success" = false ]; then
        log_status "  $tag FAILED after 3 attempts (agent unavailable). Skipping evals."
        return 1
    fi

    log_status "  $tag [$(ts)] Phase 1 complete."
    sleep "$SLEEP_BETWEEN"

    # -- Phase 2: 4 parallel evals ------------------------------------------
    log_status "  $tag [$(ts)] Phase 2: evals starting"
    local eval_pids=()

    "${DEXEC[@]}" python3 "$EVALS_DIR/eval_utils/workflow_scripted_evals.py" \
        --output-dir "$OUTPUT_DIR" \
        --results "$CASE_RESULTS/workflow_scripted_results.csv" \
        --comparison-scope "$COMPARISON_SCOPE" \
        > "$CASE_RESULTS/workflow_scripted_eval.log" 2>&1 &
    eval_pids+=($!)

    (
        cd "$EVALS_DIR" || exit
        run_llm_agent \
            "Run workflow LLM eval skill on $OUTPUT_DIR for test case $id mode=$COMPARISON_SCOPE. Write results to $CASE_RESULTS/workflow_llm_results.csv"
    ) < /dev/null > "$CASE_RESULTS/workflow_llm_eval.ndjson" 2>&1 &
    eval_pids+=($!)

    "${DEXEC[@]}" python3 "$EVALS_DIR/eval_utils/quality_scripted_evals.py" \
        --output-dir "$OUTPUT_DIR" --reference-dir "$reference_dir" \
        --results "$CASE_RESULTS/quality_scripted_results.csv" \
        --comparison-scope "$COMPARISON_SCOPE" \
        > "$CASE_RESULTS/quality_scripted_eval.log" 2>&1 &
    eval_pids+=($!)

    (
        cd "$EVALS_DIR" || exit
        run_llm_agent \
            "Run quality LLM eval skill on $OUTPUT_DIR with reference $reference_dir for test case $id mode=$COMPARISON_SCOPE. Write results to $CASE_RESULTS/quality_llm_results.csv"
    ) < /dev/null > "$CASE_RESULTS/quality_llm_eval.ndjson" 2>&1 &
    eval_pids+=($!)

    for pid in "${eval_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    log_status "  $tag [$(ts)] Phase 2 complete."

    # -- Merge results -------------------------------------------------------
    "${DEXEC[@]}" python3 "$EVALS_DIR/eval_utils/merge_results.py" \
        --results-dir "$CASE_RESULTS" \
        --output "$CASE_RESULTS/eval_summary.csv" || true
    log_status "  $tag Summary -> $CASE_RESULTS/eval_summary.csv"
    log_status "  $tag [$(ts)] Finished"
}

# ---------------------------------------------------------------------------
# FIFO semaphore for concurrency control
# ---------------------------------------------------------------------------

FIFO=""
cleanup() {
    [[ -n "$FIFO" ]] && rm -f "$FIFO"
}

setup_semaphore() {
    FIFO="$RESULTS_ROOT/.job_fifo"
    rm -f "$FIFO"
    mkfifo "$FIFO"
    exec 4<>"$FIFO"
    for ((t = 0; t < MAX_PARALLEL; t++)); do echo >&4; done
    trap cleanup EXIT
}

_spawn_jobs() {
    local id="$1" trace1_path="$2" trace2_path="$3" reference_dir="$4" platform="$5" platform2="$6"
    local capture_folder1="${7:-}" capture_folder2="${8:-}"

    should_run_id "$id" || return
    JOBS_SPAWNED=$((JOBS_SPAWNED + 1))

    for ((i = 0; i < NUM_REPEATS; i++)); do
        read -r -u4  # acquire semaphore slot
        (
            run_single_job "$id" "$i" "$trace1_path" "$trace2_path" "$reference_dir" "$platform" "$platform2" "$capture_folder1" "$capture_folder2" || true
            echo >&4  # release semaphore slot
            sleep 2  # stagger agent startup to avoid ~/.cursor/cli-config.json rename race
        ) &
        sleep 2  # stagger agent startup to avoid ~/.cursor/cli-config.json rename race
    done
}

# ---------------------------------------------------------------------------
# Stage 1: golden reference generation for one scope (from scratch)
#
# Delegates to generate_ref.sh so the golden references are produced with the
# exact same flow, written as local analysis_output_ref/ directories under
# TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/analysis_tests/. generate_ref.sh already removes each
# case's existing reference before regenerating it.
# ---------------------------------------------------------------------------

generate_refs_for_scope() {
    local scope="$1"

    echo "========================================="
    echo "  Stage 1: Golden Reference Generation"
    echo "  Scope:        $scope"
    echo "  Agent:        $AGENT_BACKEND"
    echo "  Node:         $NODE_LABEL"
    echo "  Runtime:      $RUNTIME_LABEL"
    echo "========================================="
    echo ""

    if [[ ! -f "$GENERATE_REF_SCRIPT" ]]; then
        echo "ERROR: generate_ref.sh not found at $GENERATE_REF_SCRIPT" >&2
        return 1
    fi

    REPO_ROOT="$REPO_ROOT" \
    CONTAINER="$CONTAINER" \
    MAX_PARALLEL="$MAX_PARALLEL" \
    SLEEP_BETWEEN="$SLEEP_BETWEEN" \
    TEST_IDS="$TEST_IDS" \
    NUM_REPEATS="$NUM_REPEATS" \
    USE_PI="$USE_PI" \
    AGENT_MODEL="$AGENT_MODEL" \
        bash "$GENERATE_REF_SCRIPT" "$scope"
}

# ---------------------------------------------------------------------------
# Stage 2: repeatability evals for one scope
# ---------------------------------------------------------------------------

run_repeatability_for_scope() {
    COMPARISON_SCOPE="$1"
    RESULTS_ROOT="$EVALS_DIR/repeatability_results_${COMPARISON_SCOPE}"
    TEST_TRACES_CSV="$EVALS_DIR/analysis_tests/combined_traces_${COMPARISON_SCOPE}.csv"

    if [[ ! -f "$TEST_TRACES_CSV" ]]; then
        echo "ERROR: trace CSV not found: $TEST_TRACES_CSV" >&2
        return 1
    fi

    mkdir -p "$RESULTS_ROOT"
    "${DEXEC[@]}" bash -c "mkdir -p $RESULTS_ROOT && chmod -R 777 $RESULTS_ROOT"

    if [[ "$COMPARISON_SCOPE" == "comparative" ]]; then
        expand_archive unit_tests_comparative
        expand_archive e2e_tests_comparative
    else
        expand_archive unit_tests_standalone
        expand_archive e2e_tests_standalone
    fi

    echo "========================================="
    echo "  Stage 2: Analysis Repeatability Test"
    echo "  Mode:         $COMPARISON_SCOPE"
    echo "  Agent:        $AGENT_BACKEND"
    echo "  Node:         $NODE_LABEL"
    echo "  Runtime:      $RUNTIME_LABEL"
    echo "  Repeats:      $NUM_REPEATS"
    echo "  Max parallel: $MAX_PARALLEL"
    echo "  CSV:          $TEST_TRACES_CSV"
    if [[ -n "$TEST_IDS" ]]; then
        echo "  Test filter:  $TEST_IDS"
    fi
    echo "========================================="
    echo ""

    print_scheduled_tests

    setup_semaphore

    JOBS_SPAWNED=0

    if [[ "$COMPARISON_SCOPE" == "comparative" ]]; then
        # comparative CSV: id,sub_category,trace1_path,trace2_path,reference_dir,platform,platform2,capture_folder1,capture_folder2
        while IFS=, read -r id sub_category trace1_path trace2_path reference_dir platform platform2 capture_folder1 capture_folder2 <&3; do
            [[ -z "$id" ]] && continue
            _spawn_jobs "$id" "$trace1_path" "$trace2_path" "$reference_dir" "$platform" "$platform2" "${capture_folder1:-}" "${capture_folder2:-}"
        done 3< <(tail -n +2 "$TEST_TRACES_CSV"; echo)
    else
        # standalone CSV: id,sub_category,trace_path,reference_dir,platform
        while IFS=, read -r id sub_category trace_path reference_dir platform <&3; do
            [[ -z "$id" ]] && continue
            _spawn_jobs "$id" "$trace_path" "" "$reference_dir" "$platform" ""
        done 3< <(tail -n +2 "$TEST_TRACES_CSV"; echo)
    fi

    wait
    rm -f "$FIFO"

    if [[ -n "$TEST_IDS" && "$JOBS_SPAWNED" -eq 0 ]]; then
        echo ""
        echo "WARNING: TEST_IDS='$TEST_IDS' matched no trace ids in $TEST_TRACES_CSV." >&2
        echo "  Use exact ids or underscore-delimited prefixes (e.g. gemm_01 -> gemm_01_compute_few_tiles)." >&2
    fi

    echo ""
    echo "  Repeatability ($COMPARISON_SCOPE) finished. Results in: $RESULTS_ROOT"
    echo ""
}

# ---------------------------------------------------------------------------
# Build a single unified trace CSV covering every scope that ran, using the
# standalone 5-column schema (id,sub_category,trace_path,reference_dir,platform)
# so the post-processing skill can consume both scopes in one pass. Comparative
# rows are down-projected: trace1_path -> trace_path, platform (trace1) kept.
# ---------------------------------------------------------------------------

build_combined_csv() {
    local out="$1"
    local scope csv id sub t1 t2 ref plat plat2 cf1 cf2

    echo "id,sub_category,trace_path,reference_dir,platform" > "$out"

    for scope in "${SCOPES[@]}"; do
        csv="$EVALS_DIR/analysis_tests/combined_traces_${scope}.csv"
        [[ -f "$csv" ]] || continue
        if [[ "$scope" == "comparative" ]]; then
            while IFS=, read -r id sub t1 t2 ref plat plat2 cf1 cf2; do
                [[ -z "$id" ]] && continue
                echo "$id,$sub,$t1,$ref,$plat" >> "$out"
            done < <(tail -n +2 "$csv"; echo)
        else
            while IFS= read -r line; do
                [[ -z "$line" ]] && continue
                echo "$line" >> "$out"
            done < <(tail -n +2 "$csv"; echo)
        fi
    done
}

# ---------------------------------------------------------------------------
# Stage 3: single combined post-processing across all scopes that ran.
# Merges the per-scope results trees (via symlinks) and CSVs into one, then
# invokes the post-processing skill once to emit a single pr_report.md and a
# single fix_ticket_report.md in $REPORT_DIR.
# ---------------------------------------------------------------------------

run_post_processing() {
    if [[ "$SKIP_POST_PROCESSING" == "1" ]]; then
        echo ""
        echo "  Post-processing skipped -- SKIP_POST_PROCESSING=1."
        echo "  To run later, first rebuild the combined results tree + CSV, then:"
        if [[ "$USE_PI" == true ]]; then
            echo "    pi --mode json '${PI_VENV_PREFIX}Run eval post processing on results_root=$COMBINED_RESULTS_ROOT suite=$SUITE_NAME test_traces_csv=$REPORT_DIR/combined_traces.csv report_dir=$REPORT_DIR container=${CONTAINER:-} $NODE_LABEL $RUNTIME_LABEL'"
        else
            echo "    agent 'Run eval post processing on results_root=$COMBINED_RESULTS_ROOT suite=$SUITE_NAME test_traces_csv=$REPORT_DIR/combined_traces.csv report_dir=$REPORT_DIR container=${CONTAINER:-} $NODE_LABEL $RUNTIME_LABEL'"
        fi
        return 0
    fi

    mkdir -p "$REPORT_DIR"

    # Merge per-scope results trees into one combined tree via symlinks so a
    # single aggregate/report pass sees every trace. Trace ids are unique
    # across scopes, so there are no name collisions.
    rm -rf "$COMBINED_RESULTS_ROOT"
    mkdir -p "$COMBINED_RESULTS_ROOT"

    local scope rr child name src
    for scope in "${SCOPES[@]}"; do
        rr="$EVALS_DIR/repeatability_results_${scope}"
        [[ -d "$rr" ]] || continue
        for child in "$rr"/*/; do
            [[ -d "$child" ]] || continue
            src="${child%/}"
            name="$(basename "$src")"
            ln -sfn "$src" "$COMBINED_RESULTS_ROOT/$name"
        done
    done

    local combined_csv="$REPORT_DIR/combined_traces.csv"
    build_combined_csv "$combined_csv"

    echo ""
    echo "========================================="
    echo "  Stage 3: Combined eval post-processing"
    echo "  Scopes:       ${SCOPES[*]}"
    echo "  Results tree: $COMBINED_RESULTS_ROOT"
    echo "  CSV:          $combined_csv"
    echo "  Report dir:   $REPORT_DIR"
    echo "========================================="

    (
        cd "$EVALS_DIR" || exit
        run_llm_agent \
            "Run eval post processing on results_root=$COMBINED_RESULTS_ROOT suite=$SUITE_NAME test_traces_csv=$combined_csv report_dir=$REPORT_DIR container=${CONTAINER:-} $NODE_LABEL $RUNTIME_LABEL"
    ) < /dev/null > "$REPORT_DIR/post_processing.ndjson" 2>&1

    echo "  Post-processing complete."
    echo "  PR report:         $REPORT_DIR/pr_report.md"
    echo "  Fix-ticket report: $REPORT_DIR/fix_ticket_report.md"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "========================================="
echo "  Analysis Eval Pipeline"
echo "  Scopes:       ${SCOPES[*]}"
echo "  Agent:        $AGENT_BACKEND"
echo "  Node:         $NODE_LABEL"
echo "  Runtime:      $RUNTIME_LABEL"
echo "  Repeats:      $NUM_REPEATS"
echo "  Max parallel: $MAX_PARALLEL"
if [[ -n "$TEST_IDS" ]]; then
    echo "  Test filter:  $TEST_IDS"
fi
if [[ "$SKIP_GENERATE_REF" == "1" ]]; then
    echo "  Skip reference generation: TRUE"
fi
echo "  Report dir:   $REPORT_DIR"
echo "========================================="
echo ""

# Stage 1: regenerate golden references from scratch for each scope.
if [[ "$SKIP_GENERATE_REF" == "1" ]]; then
    echo "  Stage 1 skipped -- SKIP_GENERATE_REF=1."
    echo ""
else
    for scope in "${SCOPES[@]}"; do
        generate_refs_for_scope "$scope"
    done
fi

# Stage 2: run repeatability evals for each scope against the fresh references.
for scope in "${SCOPES[@]}"; do
    run_repeatability_for_scope "$scope"
done

# Stage 3: one combined post-processing pass -> single reports.
run_post_processing

echo ""
echo "========================================="
echo "  Pipeline finished."
echo "  Reports in: $REPORT_DIR"
echo "========================================="
