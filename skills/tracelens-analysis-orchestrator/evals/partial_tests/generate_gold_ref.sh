#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -uo pipefail

# ---------------------------------------------------------------------------
# OPTIONAL, RARELY RUN. The gold reference ships pre-baked inside each
# partial-test fixture tarball -- normal test runs never call this script.
#
# Run it only to regenerate the gold reference (e.g. after an intentional,
# reviewed change to the with-capture TraceDiff path). It reproduces gold the
# same way it was originally produced: the with-capture perf-report / TraceDiff
# path on the single-execution DECODE trace pair (batch-16 decode), which is
# what makes gold's gpu_op_uid range line up with the no-capture candidate.
#
# It sources the DECODE traces AND their capture_traces folders from the
# original full-fidelity trace location (NOT from the slimmed fixture, which
# has no capture data), writes the refreshed CSV into each fixture's
# analysis_output_ref/, and rebuilds the fixture tarball so gold stays shipped.
#
# Usage: bash generate_gold_ref.sh [test_id ...]   (default: all)
# ---------------------------------------------------------------------------

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
PARTIAL_DIR="$REPO_ROOT/TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/partial_tests"
GOLD_FILENAME="semantic_purity_gold_diff_stats.csv"
ARCH_JSON="${ARCH_JSON:-$REPO_ROOT/TraceLens/Agent/Analysis/utils/arch/MI300X.json}"
PERF_CLI="${PERF_CLI:-TraceLens_generate_perf_report_pytorch_inference}"

# Original full-capture source locations (repo-relative). Override via env if
# the traces move. Format: "<decode_a> <cap_a> <decode_b> <cap_b>".
SEM_ROOT="tests/traces/semantic"
declare -A SOURCES=(
    [semantic_purity_deepseek_r1]="\
$SEM_ROOT/deepseek_R1/Deepseek-R1-Distill-LLama-8B/MI300/torch_trace/1782841132.2905657-TP-0-DECODE.trace.json.gz \
$SEM_ROOT/deepseek_R1/Deepseek-R1-Distill-LLama-8B/MI300/torch_trace/capture_traces \
$SEM_ROOT/deepseek_R1/Deepseek-R1-Distill-LLama-8B/B300/torch_trace/1782843274.7730265-TP-0-DECODE.trace.json.gz \
$SEM_ROOT/deepseek_R1/Deepseek-R1-Distill-LLama-8B/B300/torch_trace/capture_traces"
    [semantic_purity_qwen3_30b_a3b]="\
$SEM_ROOT/qwen3_30b_a3b/Qwen3-30B-A3B/MI300/torch_trace/1782859605.7198358-TP-0-DECODE.trace.json.gz \
$SEM_ROOT/qwen3_30b_a3b/Qwen3-30B-A3B/MI300/torch_trace/capture_traces \
$SEM_ROOT/qwen3_30b_a3b/Qwen3-30B-A3B/B300/torch_trace/1782799835.6885648-TP-0-DECODE.trace.json.gz \
$SEM_ROOT/qwen3_30b_a3b/Qwen3-30B-A3B/B300/torch_trace/capture_traces"
)

generate_one() {
    local id="$1"
    local spec="${SOURCES[$id]:-}"
    if [[ -z "$spec" ]]; then
        echo "ERROR: unknown test id '$id' (no source mapping)" >&2
        return 1
    fi
    # shellcheck disable=SC2086
    set -- $spec
    local decode_a="$REPO_ROOT/$1" cap_a="$REPO_ROOT/$2" decode_b="$REPO_ROOT/$3" cap_b="$REPO_ROOT/$4"

    for p in "$decode_a" "$cap_a" "$decode_b" "$cap_b" "$ARCH_JSON"; do
        [[ -e "$p" ]] || { echo "ERROR [$id]: missing source $p" >&2; return 1; }
    done

    local scratch="$PARTIAL_DIR/_gold_scratch_$id"
    local out_csvs="$scratch/perf_report_trace1_csvs"
    rm -rf "$scratch"; mkdir -p "$scratch"

    echo "[$id] running with-capture TraceDiff on the DECODE pair..."
    # trace1 (MI300) with capture + comparison against trace2 (B300) with its
    # capture triggers TraceDiff internally and writes diff_stats.csv into
    # perf_report_trace1_csvs/. Only platform1's arch JSON is needed.
    "$PERF_CLI" \
        --profile_json_path "$decode_a" --capture_folder "$cap_a" \
        --gpu_arch_json_path "$ARCH_JSON" \
        --group_by_parent_module --enable_pseudo_ops --group_by_num_kernels --include_call_stack \
        --comparison_json_path "$decode_b" --comparison_capture_folder "$cap_b" \
        --output_xlsx_path "$scratch/perf_report_trace1.xlsx" \
        --output_csvs_dir "$out_csvs" \
        || { echo "ERROR [$id]: perf-report CLI failed" >&2; return 1; }

    local gold_src="$out_csvs/diff_stats.csv"
    [[ -f "$gold_src" ]] || { echo "ERROR [$id]: $gold_src not produced" >&2; return 1; }

    local ref_dir="$PARTIAL_DIR/$id/analysis_output_ref"
    mkdir -p "$ref_dir"
    cp "$gold_src" "$ref_dir/$GOLD_FILENAME"
    echo "[$id] gold -> $ref_dir/$GOLD_FILENAME ($(wc -l < "$ref_dir/$GOLD_FILENAME") lines)"

    # Re-pack the fixture tarball so the refreshed gold ships with it.
    tar czf "$PARTIAL_DIR/fixtures/${id}.tar.gz" -C "$PARTIAL_DIR" --transform='s,^,agent_evals/Analysis/partial_tests/,' "$id"
    echo "[$id] fixture repacked -> fixtures/${id}.tar.gz"
    rm -rf "$scratch"
}

ids=("$@")
if [[ ${#ids[@]} -eq 0 ]]; then
    ids=("${!SOURCES[@]}")
fi

rc=0
for id in "${ids[@]}"; do
    generate_one "$id" || rc=1
done
exit "$rc"
