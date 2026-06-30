#!/bin/bash
# ============================================================================
# Multi-model, multi-variant (native + zentorch) benchmark sweep
#
# For each (variant, model):
#   1. Stop any running stack
#   2. Start the stack with the chosen image (zentorch or native) and model
#   3. Run guidellm benchmark via ansible at rates [32, 64, 96]
#   4. Stop the stack
#
# Results are tagged: test_name=<PREFIX>_<variant>_<sanitized-model>
# ============================================================================

set -euo pipefail

# ----------------------------------------------------------------------------
# 0. ARG PARSING
# ----------------------------------------------------------------------------
DRY_RUN=false
CLI_MODELS=()
CLI_MODELS_DIR=""
CLI_NUM_INSTANCES=""
CLI_QUANT=false
CLI_TORCHAO=false
CLI_NO_MEM_LIMIT=false
CLI_VARIANT=""
CLI_BECOME=""   # "" auto | "off" rootless | "on" force become
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n)
            DRY_RUN=true; shift ;;
        --native)
            CLI_VARIANT="native"; shift ;;
        --zentorch)
            CLI_VARIANT="zentorch"; shift ;;
        -m|--model)
            CLI_MODELS+=("$2"); shift 2 ;;
        --models-dir)
            CLI_MODELS_DIR="$2"; shift 2 ;;
        -N|--num-instances)
            CLI_NUM_INSTANCES="$2"; shift 2 ;;
        --quant)
            CLI_QUANT=true; shift ;;
        --torchao)
            CLI_TORCHAO=true; shift ;;
        --no-mem-limit)
            CLI_NO_MEM_LIMIT=true; shift ;;
        --no-become)
            CLI_BECOME="off"; shift ;;
        --become)
            CLI_BECOME="on"; shift ;;
        -h|--help)
            cat <<EOF
Usage: $0 [options]

Sweeps every (variant, model) pair and runs the guidellm benchmark.

Options:
  -m, --model ENTRY      Model to sweep, as "path-or-repo | short-tag" (the
                         tag is optional). Repeat to add several. Overrides
                         the built-in MODELS list when given at least once.
  --models-dir DIR       Host dir holding local model folders. Bind-mounted
                         into each container at the same path; model ENTRY
                         left-sides are resolved relative to it.
  -N, --num-instances N  Number of vLLM instances (also seeds the default
                         test_name prefix EPYC<N>).
  --native               Sweep only the native (non-zentorch) variant.
                         Overrides the VARIANTS array.
  --zentorch             Sweep only the zentorch variant (default).
                         Overrides the VARIANTS array.
  --quant                Quantized-model run (default: llm-compressor format).
                         Uses the base image as-is, NO torchao install. Models
                         may be HF repo ids (pulled from the hub, needs
                         HF_TOKEN) or local dirs via --models-dir (or MODELS_DIR
                         env).
  --torchao              Only meaningful with --quant. Builds the derived
                         torchao-enabled image (passes --torchao to start.sh).
                         Use for models with quant_method=torchao. Without this,
                         --quant does NOT change the image.
  --no-mem-limit         Drop the per-instance mem_limit while keeping cpuset
                         pinning (forwarded to start.sh). Use when the memory
                         cgroup OOM-kills containers despite ample free host RAM.
  --no-become            Run ansible (incl. the guidellm load generator)
                         rootless as the invoking user (-e ansible_become=false).
                         guidellm doesn't need root (network:host, user 0:0).
                         Use on hosts without passwordless sudo. Default is
                         auto: become is used only if 'sudo -n true' succeeds.
                         Env equivalent: ANSIBLE_NO_BECOME=1.
  --become               Force ansible become on (override auto-detection).
  --dry-run, -n          Validate paths/config and print every command that
                         would run, without starting containers, calling
                         ansible, or touching sudo. Safe with HF_TOKEN unset.
  -h, --help             Show this help.
EOF
            exit 0 ;;
        *)
            echo "Unknown option: $1 (try --help)" >&2
            exit 1 ;;
    esac
done

# --quant marks a quantized run (llm-compressor by default, no image change).
# --torchao (only with --quant) opts into the derived torchao image build.
QUANT="$CLI_QUANT"
TORCHAO="$CLI_TORCHAO"
if [[ "$TORCHAO" == "true" && "$QUANT" != "true" ]]; then
    echo "ERROR: --torchao requires --quant." >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# 1. MODELS TO SWEEP  --  fill in your 20 models here
# ----------------------------------------------------------------------------
# Format per entry:  "hf-path-or-relative-dir | short-tag"
#   - Left side  = passed to vLLM as --model (HF repo id, or path under MODELS_DIR)
#   - Right side = used in test_name (must be [A-Za-z0-9-] and short enough that
#                  "${PREFIX}-${variant}-${tag}" stays <= 30 chars).
# The ansible playbook enforces: test_name matches ^[A-Za-z0-9-]{1,30}$.
MODELS=(
    "Llama-3.1-8B-Instruct                | llama31-8b"
    "gpt-oss-20b-BF16                     | gptoss-20b"
)

# CLI -m/--model entries override the built-in list above.
if (( ${#CLI_MODELS[@]} > 0 )); then
    MODELS=("${CLI_MODELS[@]}")
fi

# ----------------------------------------------------------------------------
# 2. VARIANTS  --  comment out one if you only want native or only zentorch
# ----------------------------------------------------------------------------
VARIANTS=(
    "zentorch"
#    "native"
)

# CLI --native/--zentorch overrides the VARIANTS array above.
if [[ -n "$CLI_VARIANT" ]]; then
    VARIANTS=("$CLI_VARIANT")
fi

# ----------------------------------------------------------------------------
# 3. BENCHMARK CONFIG
# ----------------------------------------------------------------------------
GUIDELLM_RATES="${GUIDELLM_RATES:-[32,64]}"
# GUIDELLM_RATES="${GUIDELLM_RATES:-[64,96,128,256,364,512,1024]}"
GUIDELLM_MAX_SECONDS="${GUIDELLM_MAX_SECONDS:-300}"
# Max concurrent in-flight requests. Must be >= the largest GUIDELLM_RATES
# value or the concurrent profile clamps high rates to this cap. Forwarded to
# ansible as guidellm_max_concurrency (the playbook defaults to 128).
GUIDELLM_MAX_CONCURRENCY="${GUIDELLM_MAX_CONCURRENCY:-1024}"
BASE_WORKLOAD="${BASE_WORKLOAD:-chat}"

# --no-mem-limit: CLI flag wins over the NO_MEM_LIMIT env var. Forwarded to
# start.sh.
if [[ "$CLI_NO_MEM_LIMIT" == "true" ]]; then
    NO_MEM_LIMIT=true
fi
export NO_MEM_LIMIT="${NO_MEM_LIMIT:-false}"
# Resolve instance count early: CLI -N wins, then env, then default 1.
# It seeds both the stack layout and the default test_name prefix below.
NUM_INSTANCES="${CLI_NUM_INSTANCES:-${NUM_INSTANCES:-1}}"

# test_name must be 1-30 chars, [A-Za-z0-9-] only.
# Format used below: "${PREFIX}-${variant}-${tag}"  (variant = "zentorch"|"native")
# Budget: "EPYC7" (5) + "-zentorch-" (10) + tag = 15 + len(tag), so tag <= 15.
TEST_NAME_PREFIX="${TEST_NAME_PREFIX:-EPYC${NUM_INSTANCES}}"

# If MODELS_DIR is set, each model name in the MODELS array is expected to be
# a relative path under MODELS_DIR (e.g. "meta-llama/Llama-3.1-8B-Instruct"
# resolves to "$MODELS_DIR/meta-llama/Llama-3.1-8B-Instruct"). The directory
# is bind-mounted at the same path inside the container, so the absolute path
# is passed straight to "vllm serve".
# Leave MODELS_DIR unset to pull models from the HuggingFace hub as before.
# CLI --models-dir wins over the MODELS_DIR env var.
MODELS_DIR="${CLI_MODELS_DIR:-${MODELS_DIR:-}}"

# ----------------------------------------------------------------------------
# 4. STACK / CPU LAYOUT  (passed to start.sh --regenerate)
# ----------------------------------------------------------------------------
export NUM_INSTANCES
export CORES_PER_INSTANCE="${CORES_PER_INSTANCE:-32}"
export VLLM_START_CORE="${VLLM_START_CORE:-32}"
export NGINX_CORES="${NGINX_CORES:-1-15}"
export NGINX_PORT="${NGINX_PORT:-8080}"
export MEM_LIMIT="${MEM_LIMIT:-200g}"
export VLLM_KV_CACHE_SPACE="${VLLM_KV_CACHE_SPACE:-90}"

# Zentorch image (base). The native variant is built from this image by
# uninstalling zentorch -- start.sh --native handles this automatically.
export VLLM_IMAGE="${VLLM_IMAGE:-amdih/zendnn_zentorch:vllm_v0.22.0_zentorch_v2.11.0.1_ubuntu22.04_2026_ww23}"

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"

# ----------------------------------------------------------------------------
# 5. ANSIBLE / ENDPOINT CONFIG
# ----------------------------------------------------------------------------
# GuideLLM load-generator CPU / NUMA pinning (optional passthrough to ansible).
# The load generator is otherwise strictly memory-bound to NUMA node 0
# (cpuset_mems default '0'); with many vLLM instances node 0 can run out of
# memory and the client's worker processes get OOM-killed at startup. Set
# GUIDELLM_NUMA_NODE to a range spanning all nodes (e.g. "0-7") or to a node
# with free RAM, and optionally GUIDELLM_CPUS to matching cores.
GUIDELLM_CPUS="${GUIDELLM_CPUS:-}"
GUIDELLM_NUMA_NODE="${GUIDELLM_NUMA_NODE:-}"

export VLLM_ENDPOINT_MODE="${VLLM_ENDPOINT_MODE:-external}"
export VLLM_ENDPOINT_URL="${VLLM_ENDPOINT_URL:-http://localhost:${NGINX_PORT}}"
export LOADGEN_HOSTNAME="${LOADGEN_HOSTNAME:-localhost}"
export DUT_HOSTNAME="${DUT_HOSTNAME:-localhost}"
export ANSIBLE_SSH_USER="${ANSIBLE_SSH_USER:-$(whoami)}"
export ANSIBLE_SSH_KEY="${ANSIBLE_SSH_KEY:-$HOME/.ssh/id_rsa}"

# --- Rootless / ansible become handling -------------------------------------
# The playbook runs several tasks (incl. the guidellm load generator) under
# ansible `become: true`, which needs passwordless sudo. On hosts without it
# the run hard-fails -- and worse, only after ~10 min of health-check retries
# ("sudo: a password is required"). guidellm doesn't actually need root (its
# container is network:host + user 0:0), so we can run the whole playbook as
# the invoking user by passing `-e ansible_become=false`.
#
# Decision (CLI --no-become/--become > env ANSIBLE_NO_BECOME > auto-detect):
#   off / 1     -> rootless (ansible_become=false)
#   on  / 0     -> force become (default ansible behavior)
#   auto        -> become only if `sudo -n true` works.
NO_BECOME=false
case "$CLI_BECOME" in
    off) NO_BECOME=true ;;
    on)  NO_BECOME=false ;;
    *)
        if [[ -n "${ANSIBLE_NO_BECOME:-}" ]]; then
            [[ "$ANSIBLE_NO_BECOME" == "1" ]] && NO_BECOME=true || NO_BECOME=false
        elif ! $DRY_RUN && ! sudo -n true 2>/dev/null; then
            NO_BECOME=true
            echo "[INFO] no passwordless sudo detected -> running ansible rootless" \
                 "(ansible_become=false). Override with --become." >&2
        fi
        ;;
esac
BECOME_ARGS=()
if $NO_BECOME; then
    BECOME_ARGS=(-e "ansible_become=false")
fi

# HF_TOKEN is required for gated models. Don't use ${VAR:?} -- it hard-exits
# the shell (which can take down the tmux pane that launched the script).
# Instead, prompt interactively so the session stays alive.
# In --dry-run we skip the check entirely.
if ! $DRY_RUN; then
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo ""
        echo "HF_TOKEN is not set. Gated models (e.g. Llama) will fail without it."
        if [[ -t 0 ]]; then
            read -r -s -p "Paste your HuggingFace token (or press Enter to abort): " HF_TOKEN
            echo ""
        fi
        if [[ -z "${HF_TOKEN:-}" ]]; then
            echo "No token provided. Aborting (tmux session preserved)." >&2
            exit 1
        fi
    fi
fi
export HF_TOKEN="${HF_TOKEN:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ANSIBLE_DIR="${ANSIBLE_DIR:-$SCRIPT_DIR/vllm-cpu-perf-eval/automation/test-execution/ansible}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/sweep-logs/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOG_DIR"

if [[ -n "$MODELS_DIR" ]]; then
    if [[ ! -d "$MODELS_DIR" ]]; then
        echo "ERROR: MODELS_DIR='$MODELS_DIR' does not exist or is not a directory." >&2
        exit 1
    fi
    MODELS_DIR="$(cd "$MODELS_DIR" && pwd)"
fi

# Resolve a model entry to the actual path/repo passed to vLLM.
# - With MODELS_DIR set: returns "$MODELS_DIR/$model" (absolute local path)
# - Without MODELS_DIR : returns "$model" unchanged (HF repo id)
resolve_model_path() {
    local m="$1"
    if [[ -n "$MODELS_DIR" ]]; then
        echo "$MODELS_DIR/$m"
    else
        echo "$m"
    fi
}

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
sanitize_model() {
    # meta-llama/Llama-3.1-8B-Instruct  ->  meta-llama__Llama-3.1-8B-Instruct
    # Used only for filesystem-safe log file names, NOT for test_name.
    echo "$1" | sed 's|/|__|g'
}

# Trim leading/trailing whitespace.
trim() {
    local s="$1"
    s="${s#"${s%%[![:space:]]*}"}"
    s="${s%"${s##*[![:space:]]}"}"
    echo "$s"
}

# Split a MODELS entry "hf/path | tag" -> sets global MODEL_PATH and MODEL_TAG.
parse_model_entry() {
    local entry="$1"
    local left right
    left="${entry%%|*}"
    right="${entry##*|}"
    if [[ "$left" == "$entry" ]]; then
        # No '|' separator -- fall back to using the path as the tag too.
        MODEL_PATH="$(trim "$entry")"
        MODEL_TAG="$(trim "$entry")"
    else
        MODEL_PATH="$(trim "$left")"
        MODEL_TAG="$(trim "$right")"
    fi
}

# Map a model tag to the [A-Za-z0-9-] charset the ansible test_name validator
# requires: replace disallowed chars with '-', collapse runs, trim edges. Lets
# common ids like "qwen3-0.6b" work verbatim instead of failing preflight.
sanitize_tag() {
    local t="$1"
    t="${t//[^A-Za-z0-9-]/-}"
    while [[ "$t" == *--* ]]; do t="${t//--/-}"; done
    t="${t#-}"; t="${t%-}"
    printf '%s' "$t"
}

# Validate a test_name against the ansible playbook's rules.
# Returns 0 if OK, 1 otherwise and prints a reason on stderr.
validate_test_name() {
    local name="$1"
    local len=${#name}
    if (( len < 1 || len > 30 )); then
        echo "  test_name '$name' is $len chars (must be 1-30)" >&2
        return 1
    fi
    if [[ ! "$name" =~ ^[A-Za-z0-9-]+$ ]]; then
        echo "  test_name '$name' contains non-[A-Za-z0-9-] characters" >&2
        return 1
    fi
    return 0
}

banner() {
    echo ""
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
}

run_one() {
    local variant="$1"
    local entry="$2"

    parse_model_entry "$entry"
    local model="$MODEL_PATH"
    local tag="$MODEL_TAG"

    # test_name uses ONLY [A-Za-z0-9-] and is <= 30 chars (ansible validates).
    local safe_tag
    safe_tag=$(sanitize_tag "$tag")
    if [[ "$safe_tag" != "$tag" ]]; then
        echo "  Note: tag '$tag' sanitized to '$safe_tag' for test_name." >&2
    fi
    local test_name="${TEST_NAME_PREFIX}-${variant}-${safe_tag}"

    # Log file name can use the sanitized HF path -- not subject to test_name rules.
    local log_tag
    log_tag=$(sanitize_model "$model")
    local run_log="$LOG_DIR/${variant}_${log_tag}.log"

    # Resolve the actual path/repo to load (local absolute path or HF repo id).
    local model_path
    model_path=$(resolve_model_path "$model")

    banner "[$variant] $model"
    echo "  test_name  = $test_name (${#test_name} chars)"
    echo "  model_path = $model_path"
    echo "  rates      = $GUIDELLM_RATES"
    echo "  max_conc   = $GUIDELLM_MAX_CONCURRENCY"
    echo "  log        = $run_log"

    # Build start.sh args (same for dry-run and real run).
    # --torchao is only added when both --quant and --torchao were passed.
    local start_args=(--regenerate --timeout "$HEALTH_TIMEOUT" -m "$model_path")
    if [[ "$TORCHAO" == "true" ]]; then
        start_args=(--torchao "${start_args[@]}")
    fi
    if [[ -n "$MODELS_DIR" ]]; then
        start_args+=(--models-dir "$MODELS_DIR")
    fi
    if [[ "$variant" == "native" ]]; then
        start_args=(--native "${start_args[@]}")
    fi
    if [[ "${NO_MEM_LIMIT:-false}" == "true" ]]; then
        start_args+=(--no-mem-limit)
    fi

    if $DRY_RUN; then
        echo ""
        echo "  [dry-run] would call: $SCRIPT_DIR/stop.sh --clean"
        echo "  [dry-run] would call: $SCRIPT_DIR/start.sh ${start_args[*]}"
        echo "  [dry-run] would cd:   $ANSIBLE_DIR"
        echo "  [dry-run] would run ansible-playbook:"
        cat <<EOF
    ansible-playbook -i inventory/hosts.yml \\
        llm-benchmark-concurrent-load.yml \\
        --connection=local \\
        -e "ansible_python_interpreter=/usr/bin/python3" \\
        -e "test_model=$model_path" \\
        -e "base_workload=$BASE_WORKLOAD" \\
        -e "skip_phase_2=true" \\
        -e "skip_phase_3=true" \\
        -e "guidellm_rate=$GUIDELLM_RATES" \\
        -e "guidellm_max_seconds=$GUIDELLM_MAX_SECONDS" \\
        -e "guidellm_max_concurrency=$GUIDELLM_MAX_CONCURRENCY" \\
        -e "test_name=$test_name" \\
        ${GUIDELLM_CPUS:+-e \"guidellm_cpus=$GUIDELLM_CPUS\" }${GUIDELLM_NUMA_NODE:+-e \"guidellm_numa_node=$GUIDELLM_NUMA_NODE\" }$($NO_BECOME && printf -- '-e "ansible_become=false" ')\\
        -e '{"health_check":{"timeout":600,"interval":5}}'
EOF
        echo "  [dry-run] would call: $SCRIPT_DIR/stop.sh --clean"
        echo "  [dry-run] would run:  sync && echo 3 | sudo -n tee /proc/sys/vm/drop_caches"
        echo "  [dry-run] Done: $test_name"
        return 0
    fi

    # --- 1. Stop any previous stack -----------------------------------------
    echo ""
    echo "--- Stopping any previous stack ---"
    "$SCRIPT_DIR/stop.sh" --clean >> "$run_log" 2>&1 || true

    # --- 2. Start fresh stack with the right image + model ------------------
    echo "--- Starting stack ($variant, $model) ---"
    if ! "$SCRIPT_DIR/start.sh" "${start_args[@]}" >> "$run_log" 2>&1; then
        echo "  ERROR: start.sh failed for $variant / $model_path -- skipping."
        echo "  See $run_log for details."
        echo "  Containers left running for inspection. To inspect:"
        echo "    podman ps -a --filter 'name=vllm-'"
        echo "    podman logs vllm-instance-1"
        echo "  When done, manually clean with: $SCRIPT_DIR/stop.sh --clean"
        return 1
    fi

    # --- 3. Run ansible benchmark sweep -------------------------------------
    echo "--- Running ansible benchmark ---"
    pushd "$ANSIBLE_DIR" > /dev/null

    # Optional GuideLLM load-generator pinning overrides.
    local guidellm_extra_args=()
    if [[ -n "$GUIDELLM_CPUS" ]]; then
        guidellm_extra_args+=(-e "guidellm_cpus=$GUIDELLM_CPUS")
    fi
    if [[ -n "$GUIDELLM_NUMA_NODE" ]]; then
        guidellm_extra_args+=(-e "guidellm_numa_node=$GUIDELLM_NUMA_NODE")
    fi

    local run_rc=0
    if ! ansible-playbook -i inventory/hosts.yml \
            llm-benchmark-concurrent-load.yml \
            --connection=local \
            -e "ansible_python_interpreter=/usr/bin/python3" \
            -e "test_model=$model_path" \
            -e "base_workload=$BASE_WORKLOAD" \
            -e "skip_phase_2=true" \
            -e "skip_phase_3=true" \
            -e "guidellm_rate=$GUIDELLM_RATES" \
            -e "guidellm_max_seconds=$GUIDELLM_MAX_SECONDS" \
            -e "guidellm_max_concurrency=$GUIDELLM_MAX_CONCURRENCY" \
            -e "test_name=$test_name" \
            "${guidellm_extra_args[@]}" \
            "${BECOME_ARGS[@]}" \
            -e '{"health_check":{"timeout":600,"interval":5}}' \
            >> "$run_log" 2>&1; then
        run_rc=1
        echo "  WARNING: ansible benchmark failed for $variant / $model."
        echo "  See $run_log for details."
        if ! $NO_BECOME && grep -qiE "sudo: a password is required|Missing sudo password" "$run_log"; then
            echo "  CAUSE: ansible 'become' needs passwordless sudo, which this host lacks." >&2
            echo "         Re-run with --no-become (or ANSIBLE_NO_BECOME=1) to run rootless." >&2
        fi
    fi

    popd > /dev/null

    # --- 4. Tear down -------------------------------------------------------
    echo "--- Stopping stack ---"
    "$SCRIPT_DIR/stop.sh" --clean >> "$run_log" 2>&1 || true

    # --- 5. Drop page cache / dentries / inodes -----------------------------
    echo "--- Dropping page cache ---"
    sync
    if echo 3 | sudo -n tee /proc/sys/vm/drop_caches >> "$run_log" 2>&1; then
        echo "  Page cache dropped."
    else
        echo "  WARNING: failed to drop page cache (need passwordless sudo for 'tee /proc/sys/vm/drop_caches')."
    fi

    if [[ "$run_rc" -ne 0 ]]; then
        echo "  Done (WITH FAILURES): $test_name"
    else
        echo "  Done: $test_name"
    fi
    return "$run_rc"
}

# ----------------------------------------------------------------------------
# Pre-flight validation
# ----------------------------------------------------------------------------
banner "Pre-flight checks"

preflight_ok=true
check_path() {
    local label="$1" path="$2"
    if [[ -e "$path" ]]; then
        echo "  [OK]   $label : $path"
    else
        echo "  [MISS] $label : $path"
        preflight_ok=false
    fi
}

check_path "SCRIPT_DIR    " "$SCRIPT_DIR"
check_path "start.sh      " "$SCRIPT_DIR/start.sh"
check_path "stop.sh       " "$SCRIPT_DIR/stop.sh"
check_path "LOG_DIR       " "$LOG_DIR"
check_path "ANSIBLE_DIR   " "$ANSIBLE_DIR"
check_path "playbook      " "$ANSIBLE_DIR/llm-benchmark-concurrent-load.yml"
check_path "inventory     " "$ANSIBLE_DIR/inventory/hosts.yml"

if (( ${#MODELS[@]} == 0 )); then
    echo "  [MISS] MODELS array is empty"
    preflight_ok=false
else
    echo "  [OK]   MODELS array: ${#MODELS[@]} entries"
fi
if (( ${#VARIANTS[@]} == 0 )); then
    echo "  [MISS] VARIANTS array is empty"
    preflight_ok=false
else
    echo "  [OK]   VARIANTS array: ${VARIANTS[*]}"
fi

if command -v ansible-playbook >/dev/null 2>&1; then
    echo "  [OK]   ansible-playbook found at: $(command -v ansible-playbook)"
else
    echo "  [MISS] ansible-playbook not found in PATH"
    preflight_ok=false
fi
if command -v podman >/dev/null 2>&1; then
    echo "  [OK]   podman found at: $(command -v podman)"
else
    echo "  [WARN] podman not found in PATH (start.sh will fail)"
fi

# Host-level blockers (image short-name / rootless cpuset / CNI version) that
# detect.py and the path checks above don't cover. Hard-fails the preflight on
# a real run; informational only under --dry-run (the image may not be pulled
# yet). Set SKIP_HOST_CHECK=1 to skip.
echo ""
echo "  --- host environment (image / cpuset / CNI) ---"
if VLLM_IMAGE="$VLLM_IMAGE" bash "$SCRIPT_DIR/check-host.sh"; then
    :
elif $DRY_RUN; then
    echo "  [INFO] host check reported a blocker (dry-run: not failing preflight)."
else
    preflight_ok=false
fi

if $DRY_RUN; then
    echo "  [INFO] HF_TOKEN check skipped (dry-run)"
elif [[ -n "${HF_TOKEN:-}" ]]; then
    echo "  [OK]   HF_TOKEN is set (${#HF_TOKEN} chars)"
else
    echo "  [WARN] HF_TOKEN is not set"
fi

# ansible become mode. Fail fast here if become is required but passwordless
# sudo is unavailable -- otherwise the playbook only fails ~10 min into the
# health-check retries with "sudo: a password is required".
if $NO_BECOME; then
    echo "  [OK]   ansible become : OFF (running rootless as $(whoami))"
elif $DRY_RUN; then
    echo "  [INFO] ansible become : ON (sudo check skipped in dry-run)"
elif sudo -n true 2>/dev/null; then
    echo "  [OK]   ansible become : ON (passwordless sudo available)"
else
    echo "  [BAD]  ansible become : ON but no passwordless sudo on this host."
    echo "         The guidellm load generator would fail ~10 min into the run."
    echo "         Re-run with --no-become (or ANSIBLE_NO_BECOME=1) to run rootless."
    preflight_ok=false
fi

if [[ -n "$MODELS_DIR" ]]; then
    echo "  [OK]   MODELS_DIR    : $MODELS_DIR (will be bind-mounted into containers)"
    for entry in "${MODELS[@]}"; do
        parse_model_entry "$entry"
        if [[ -d "$MODELS_DIR/$MODEL_PATH" ]]; then
            echo "  [OK]   model dir    : $MODELS_DIR/$MODEL_PATH"
        else
            echo "  [MISS] model dir    : $MODELS_DIR/$MODEL_PATH"
            preflight_ok=false
        fi
    done
else
    echo "  [INFO] MODELS_DIR not set -- models will be downloaded from HuggingFace hub"
fi

if [[ "$QUANT" == "true" ]]; then
    if [[ "$TORCHAO" == "true" ]]; then
        echo "  [OK]   --quant ON     : torchao image (start.sh --torchao)"
    else
        echo "  [OK]   --quant ON     : llm-compressor (base image, no torchao)"
    fi
else
    echo "  [INFO] --quant OFF    : base image, no torchao"
fi

# Validate every (variant, model) -> test_name fits ansible's rules.
echo ""
echo "  --- test_name validation ---"
for variant in "${VARIANTS[@]}"; do
    for entry in "${MODELS[@]}"; do
        parse_model_entry "$entry"
        candidate="${TEST_NAME_PREFIX}-${variant}-$(sanitize_tag "$MODEL_TAG")"
        if validate_test_name "$candidate" 2>/dev/null; then
            printf "  [OK]   %-30s  (%d chars)\n" "$candidate" "${#candidate}"
        else
            printf "  [BAD]  %-30s  (%d chars, [A-Za-z0-9-] only, 1-30 chars)\n" \
                "$candidate" "${#candidate}"
            preflight_ok=false
        fi
    done
done

if ! $preflight_ok; then
    echo ""
    echo "Pre-flight failed. Fix the [MISS] items above before running." >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
mode_label="online (guidellm)"
if $DRY_RUN; then
    banner "DRY RUN -- ${#MODELS[@]} models x ${#VARIANTS[@]} variants -- $mode_label (no commands will execute)"
else
    banner "Benchmark sweep -- ${#MODELS[@]} models x ${#VARIANTS[@]} variants -- $mode_label"
fi
echo "  Variants : ${VARIANTS[*]}"
echo "  Models   :"
for entry in "${MODELS[@]}"; do
    parse_model_entry "$entry"
    printf "    - %-40s -> %s\n" "$MODEL_PATH" "$MODEL_TAG"
done
echo "  Rates    : $GUIDELLM_RATES"
echo "  Log dir  : $LOG_DIR"

total=$(( ${#MODELS[@]} * ${#VARIANTS[@]} ))
i=0
failed=()

for variant in "${VARIANTS[@]}"; do
    for entry in "${MODELS[@]}"; do
        parse_model_entry "$entry"
        i=$((i + 1))
        echo ""
        echo ">>> [$i/$total] variant=$variant model=$MODEL_PATH"
        if ! run_one "$variant" "$entry"; then
            failed+=("$variant / $MODEL_PATH")
        fi
    done
done

banner "Sweep complete"
echo "  Total runs : $total"
echo "  Failed     : ${#failed[@]}"
for f in "${failed[@]}"; do echo "    - $f"; done
echo "  Logs       : $LOG_DIR"
echo ""
echo "Results from each run are in:"
echo "  $ANSIBLE_DIR/../../../results/llm/<sanitized-model>/<workload>-<timestamp>-<test_name>/external-endpoint/"
