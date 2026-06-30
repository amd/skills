#!/bin/bash
# Drive ONE benchmark run end-to-end, fully configured by environment variables
# (nothing model/image/PR-specific is hardcoded):
#   - start the memory poller
#   - run run_sweep.sh (starts the stack, runs guidellm, tears down)
#   - stop the poller, print the peak
#
# Usage:
#   LABEL=<name> VLLM_IMAGE=<image> MODEL="<repo-or-path> | <tag>" \
#       [knobs...] bash run_combo.sh
#   (LABEL may also be passed as the first positional arg.)
#
# Required:
#   LABEL              Short name for this run; used in output filenames.
#   VLLM_IMAGE         Container image to benchmark.
#   MODEL              vLLM model spec "left | tag" (left = repo id or local path,
#                      tag = short [A-Za-z0-9-] name used in test_name).
#
# Common knobs (all optional, passed through only when set):
#   NATIVE=1                       Add --native (bypass zentorch).
#   NUM_INSTANCES (default 3)      vLLM instances.
#   CORES_PER_INSTANCE (default 32)
#   GUIDELLM_RATES (default [32,64])  Concurrency rate list.
#   RUN_TAG                        Suffix for output files (e.g. _c96) so
#                                  different rate sweeps don't clobber each other.
#   HF_CACHE_DIR (default ~/.cache/hf-shared)  Shared HF cache (pre-warmed).
#   MODELS_DIR                     Host dir of local models (bind-mounted).
#   BENCH_ROOT                     Where results/ + tmp/ go (default: $PWD).
#   HARNESS                        Override the vendored harness dir if needed.
#   EXTRA_SWEEP_ARGS               Extra args appended verbatim to run_sweep.sh.
#
# Example (one run):
#   LABEL=run1 VLLM_IMAGE=amdih/zendnn_zentorch:vllm_v0.22.0_zentorch_v2.11.0.1_ubuntu22.04_2026_ww23 \
#     MODEL="Qwen/Qwen3-0.6B | qwen3-0.6b" \
#     bash run_combo.sh
set -uo pipefail

# LABEL from $1 or env.
LABEL="${LABEL:-${1:-}}"
: "${LABEL:?set LABEL (a short name for this run, used in output filenames)}"
: "${VLLM_IMAGE:?set VLLM_IMAGE (the container image to benchmark)}"
: "${MODEL:?set MODEL (vLLM model spec: \"repo-or-path | tag\")}"
export VLLM_IMAGE

# Resolve paths relative to this script's location. The benchmark harness is
# vendored alongside this skill (../harness) — no dependency on any other repo.
#   scripts/ -> vllm-multiinstance/{harness,scripts}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# Vendored harness (run_sweep.sh / start.sh / generate-config.sh / stop.sh).
HARNESS="${HARNESS:-$SKILL_DIR/harness}"
# Where results/ and tmp/ land. Defaults to the current directory so the caller
# controls output location; override via env.
BENCH_ROOT="${BENCH_ROOT:-$PWD}"
RESULTS_DIR="$BENCH_ROOT/results"
mkdir -p "$RESULTS_DIR"

# start.sh pre-warms the HF cache via `huggingface-cli`, often not on the default
# login PATH. Use the active env; otherwise fall back to conda or a venv. The
# model is normally already cached, so the pre-warm call is a fast no-op.
ensure_hf_cli() {
    command -v huggingface-cli >/dev/null 2>&1 && return 0
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
            && conda activate "${CONDA_ENV:-base}" 2>/dev/null
        command -v huggingface-cli >/dev/null 2>&1 && return 0
    fi
    for v in "${VENV_PATH:-}" "$BENCH_ROOT/.venv" "$SKILL_DIR/.venv" "$HOME/.venv"; do
        if [ -n "$v" ] && [ -f "$v/bin/activate" ]; then
            # shellcheck disable=SC1091
            source "$v/bin/activate"
            command -v huggingface-cli >/dev/null 2>&1 && return 0
        fi
    done
    echo "WARN: huggingface-cli not found via PATH/conda/venv; HF pre-warm may fail" >&2
    return 1
}
ensure_hf_cli

# Build the sweep args. Knobs are exported only when the caller set them, so
# generate-config.sh emits exactly the env the run needs (and nothing else).
SWEEP_ARGS=(-m "$MODEL")
[ "${NO_MEM_LIMIT:-1}" = "1" ] && SWEEP_ARGS=(--no-mem-limit "${SWEEP_ARGS[@]}")
[ "${NATIVE:-0}" = "1" ]       && SWEEP_ARGS=(--native "${SWEEP_ARGS[@]}")
[ -n "${MODELS_DIR:-}" ]       && SWEEP_ARGS+=(--models-dir "$MODELS_DIR")
# shellcheck disable=SC2206
[ -n "${EXTRA_SWEEP_ARGS:-}" ] && SWEEP_ARGS+=(${EXTRA_SWEEP_ARGS})

# Name our stack distinctly so it never collides with another vLLM stack on this
# host (the poller and stop logic key off this prefix too).
export VLLM_NAME_PREFIX="${VLLM_NAME_PREFIX:-bench-vllm-instance}"
export VLLM_NGINX_NAME="${VLLM_NGINX_NAME:-bench-vllm-nginx-lb}"

# Keep all temp off a possibly-full root fs. The patched ansible playbook reads
# BENCH_TMPDIR for its metrics script + vllm-logs.
export TMPDIR="${BENCH_TMPDIR:-$BENCH_ROOT/tmp}"
mkdir -p "$TMPDIR"
export BENCH_TMPDIR="$TMPDIR"
export ANSIBLE_LOCAL_TEMP="$TMPDIR/ansible"
export ANSIBLE_REMOTE_TEMP="$TMPDIR/ansible"
mkdir -p "$ANSIBLE_LOCAL_TEMP"

# Stack layout + cache (all overridable).
export NUM_INSTANCES="${NUM_INSTANCES:-3}"
export CORES_PER_INSTANCE="${CORES_PER_INSTANCE:-32}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/hf-shared}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_TOKEN="${HF_TOKEN:-offline}"   # avoid run_sweep.sh's interactive prompt
export GUIDELLM_RATES="${GUIDELLM_RATES:-[32,64]}"
RUN_TAG="${RUN_TAG:-}"

MEM_CSV="$RESULTS_DIR/mem_${LABEL}${RUN_TAG}.csv"
SWEEP_LOG="$RESULTS_DIR/sweep_${LABEL}${RUN_TAG}.log"

echo "=================================================================="
echo " LABEL=$LABEL"
echo "   VLLM_IMAGE=$VLLM_IMAGE"
echo "   MODEL=$MODEL"
echo "   NATIVE=${NATIVE:-0}  NUM_INSTANCES=$NUM_INSTANCES  CORES=$CORES_PER_INSTANCE"
echo "   rates=$GUIDELLM_RATES   sweep args: ${SWEEP_ARGS[*]}"
echo "   mem csv:   $MEM_CSV"
echo "   sweep log: $SWEEP_LOG"
echo "=================================================================="

# Start memory poller in background.
bash "$SCRIPT_DIR/mem_poll.sh" "$LABEL" "$MEM_CSV" 2 &
POLL_PID=$!

# Run the sweep.
( cd "$HARNESS" && ./run_sweep.sh "${SWEEP_ARGS[@]}" ) 2>&1 | tee "$SWEEP_LOG"
SWEEP_RC=${PIPESTATUS[0]}

# Stop poller.
rm -f "${MEM_CSV}.run"
wait "$POLL_PID" 2>/dev/null

echo ""
echo "--- $LABEL done (sweep rc=$SWEEP_RC) ---"
grep "^PEAK" "$MEM_CSV" || echo "WARN: no PEAK recorded (stack may not have started)"
