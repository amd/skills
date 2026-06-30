#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GENERATED_DIR="$SCRIPT_DIR/generated"
COMPOSE_FILE="$GENERATED_DIR/docker-compose.yml"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"
HEALTH_INTERVAL=10

# Container health lives at a different inspect path across podman major
# versions: podman 4.x exposes .State.Health.Status, podman 3.x exposes
# .State.Healthcheck.Status. Reading the 4.x path on podman 3.x yields an
# empty string (the field is absent), which made the wait loop below treat a
# perfectly healthy instance as never-ready and hang the full timeout. Try the
# 4.x path first, fall back to the 3.x path, so both versions report correctly.
container_health() {
    local c="$1" h
    h="$(podman inspect --format '{{.State.Health.Status}}' "$c" 2>/dev/null || true)"
    if [[ -z "$h" || "$h" == "<no value>" ]]; then
        h="$(podman inspect --format '{{.State.Healthcheck.Status}}' "$c" 2>/dev/null || true)"
    fi
    [[ -z "$h" || "$h" == "<no value>" ]] && h="missing"
    printf '%s' "$h"
}

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Start the multi-instance vLLM + NGINX stack.

Options:
  --regenerate    Force regeneration of config files (passes all extra args to generate-config.sh)
  --native        Use the native backend: build an image without zentorch from
                  the base VLLM_IMAGE and use it (same as BACKEND=native)
  --zentorch      Use the zentorch backend (default; same as BACKEND=zentorch)
  --torchao       Build a derived image with torchao>=0.10.0 pre-installed
                  (required for models with quant_method=torchao)
  --no-limits     Skip cpuset/mem_limit/shm_size (for rootless/LSF environments)
  --no-mem-limit  Drop mem_limit only; keep cpuset/shm_size/cap_add/security_opt
  --no-wait       Start containers but don't wait for health checks
  --timeout N     Health check timeout in seconds (default: $HEALTH_TIMEOUT)
  -h, --help      Show this help

If generated/ directory doesn't exist, generate-config.sh runs automatically with defaults.
Pass configuration flags after --regenerate to customize (they forward to generate-config.sh).

Backend defaults to zentorch. Set BACKEND=native (or pass --native) to use the
native (non-zentorch) backend.

Examples:
  $0                                    # start with existing or default config
  $0 --regenerate -n 3 -c 24           # regenerate for 3 instances, 24 cores each
  $0 --regenerate --model /path/model  # regenerate with a local model
  $0 --native --regenerate             # build native image and regenerate config to use it
EOF
    exit 0
}

REGENERATE=false
NO_WAIT=false
# Backend selection: "zentorch" (default) or "native". Override via the
# BACKEND env var or the --native/--zentorch flags below.
BACKEND="${BACKEND:-zentorch}"
USE_TORCHAO=false
GENERATE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --regenerate) REGENERATE=true; shift ;;
        --native) BACKEND=native; shift ;;
        --zentorch) BACKEND=zentorch; shift ;;
        --torchao|--with-torchao) USE_TORCHAO=true; shift ;;
        --no-limits) GENERATE_ARGS+=("--no-limits"); shift ;;
        --no-mem-limit) GENERATE_ARGS+=("--no-mem-limit"); shift ;;
        --no-wait) NO_WAIT=true; shift ;;
        --timeout) HEALTH_TIMEOUT="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) GENERATE_ARGS+=("$1"); shift ;;
    esac
done

case "$BACKEND" in
    zentorch|native) ;;
    *) echo "ERROR: invalid BACKEND='$BACKEND' (expected 'zentorch' or 'native')." >&2; exit 1 ;;
esac

# --- Host / environment preflight ------------------------------------------
# Fail fast (with remediation) on the host-level blockers that otherwise only
# surface as a cryptic deep failure or a 20-min health-wait hang: unresolvable
# image short-names, missing rootless cpuset delegation, and CNI version skew.
# Runs against the BASE image (native/torchao builds pull from it first).
if [[ "${SKIP_HOST_CHECK:-0}" != "1" ]]; then
    LIMITS_ON=1
    for a in "${GENERATE_ARGS[@]:-}"; do
        [[ "$a" == "--no-limits" ]] && LIMITS_ON=0
    done
    echo "--- Host preflight (image / cpuset / CNI) ---"
    if ! LIMITS_ON="$LIMITS_ON" bash "$SCRIPT_DIR/check-host.sh"; then
        echo "Host preflight failed. Fix the [BLOCK] items above, or set" >&2
        echo "SKIP_HOST_CHECK=1 to bypass at your own risk." >&2
        exit 1
    fi
    echo ""
fi

if [[ "$BACKEND" == "native" ]]; then
    BASE_IMAGE="${VLLM_IMAGE:-amdih/zendnn_zentorch:vllm_v0.22.0_zentorch_v2.11.0.1_ubuntu22.04_2026_ww23}"
    NATIVE_IMAGE="${BASE_IMAGE}_native"
    TEMP_CONTAINER="vllm-native-build-$$"

    if podman image exists "$NATIVE_IMAGE" 2>/dev/null; then
        echo "--- Native image already exists: $NATIVE_IMAGE ---"
    else
        echo "--- Building native image (removing zentorch) ---"
        echo "  Base:   $BASE_IMAGE"
        echo "  Target: $NATIVE_IMAGE"

        podman run --entrypoint bash --name "$TEMP_CONTAINER" "$BASE_IMAGE" \
            -c "pip uninstall -y zentorch zentorch-weekly 2>/dev/null; echo 'zentorch packages removed'"

        podman commit --change 'ENTRYPOINT ["vllm", "serve"]' "$TEMP_CONTAINER" "$NATIVE_IMAGE"
        podman rm "$TEMP_CONTAINER"

        echo "  Native image built successfully."
    fi

    export VLLM_IMAGE="$NATIVE_IMAGE"
    GENERATE_ARGS+=("--image" "$NATIVE_IMAGE" "--native")
    REGENERATE=true
    echo ""
fi

if $USE_TORCHAO; then
    BASE_IMAGE="${VLLM_IMAGE:-amdih/zendnn_zentorch:vllm_v0.22.0_zentorch_v2.11.0.1_ubuntu22.04_2026_ww23}"
    TORCHAO_SPEC="${TORCHAO_VERSION:+torchao==${TORCHAO_VERSION}}"
    TORCHAO_SPEC="${TORCHAO_SPEC:-torchao>=0.10.0}"
    TORCHAO_TAG="${TORCHAO_VERSION:-latest}"
    TORCHAO_IMAGE="${BASE_IMAGE}_torchao-${TORCHAO_TAG}"
    TEMP_CONTAINER="vllm-torchao-build-$$"

    if podman image exists "$TORCHAO_IMAGE" 2>/dev/null; then
        echo "--- torchao image already exists: $TORCHAO_IMAGE ---"
    else
        echo "--- Building torchao image (installing ${TORCHAO_SPEC}) ---"
        echo "  Base:   $BASE_IMAGE"
        echo "  Target: $TORCHAO_IMAGE"

        podman run --entrypoint bash --name "$TEMP_CONTAINER" "$BASE_IMAGE" \
            -c "pip install --no-cache-dir '${TORCHAO_SPEC}' && python -c 'import torchao; print(\"torchao\", torchao.__version__)'"

        podman commit --change 'ENTRYPOINT ["vllm", "serve"]' "$TEMP_CONTAINER" "$TORCHAO_IMAGE"
        podman rm "$TEMP_CONTAINER"

        echo "  torchao image built successfully."
    fi

    export VLLM_IMAGE="$TORCHAO_IMAGE"
    GENERATE_ARGS+=("--image" "$TORCHAO_IMAGE")
    REGENERATE=true
    echo ""
fi

if $REGENERATE || [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "--- Generating configuration ---"
    bash "$SCRIPT_DIR/generate-config.sh" "${GENERATE_ARGS[@]}"
    echo ""
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "ERROR: $COMPOSE_FILE not found. Run generate-config.sh first."
    exit 1
fi

# --- HuggingFace pre-warm ---------------------------------------------------
# Pull the model once on the host before any vLLM container starts, so the
# shared HF cache (bind-mounted into every instance) is fully populated and
# all N workers load from disk in parallel without racing on hf_hub lockfiles.
# Hard-fails on download error: skipping the pre-warm and letting N containers
# race on the same download is exactly the bug this step exists to prevent.
ENV_FILE="$GENERATED_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

if [[ -z "${MODEL_NAME:-}" || -z "${HF_CACHE_DIR:-}" ]]; then
    echo "ERROR: MODEL_NAME or HF_CACHE_DIR not set after sourcing $ENV_FILE."
    exit 1
fi

if [[ "$MODEL_NAME" == /* ]]; then
    echo "--- Pre-warm skipped: model is a local path ($MODEL_NAME) ---"
else
    echo "--- Pre-warming HF cache for $MODEL_NAME -> $HF_CACHE_DIR ---"
    # huggingface_hub 1.0+ replaced `huggingface-cli` with `hf`. Prefer `hf`
    # when present; fall back to `huggingface-cli` for older installs.
    if command -v hf >/dev/null 2>&1; then
        HF_CLI=(hf download)
    elif command -v huggingface-cli >/dev/null 2>&1; then
        HF_CLI=(huggingface-cli download)
    else
        echo "ERROR: neither 'hf' nor 'huggingface-cli' found on host. Install with:" >&2
        echo "  pip install huggingface_hub" >&2
        exit 1
    fi
    if ! HF_HOME="$HF_CACHE_DIR/huggingface" \
         HF_TOKEN="${HF_TOKEN:-}" \
         "${HF_CLI[@]}" "$MODEL_NAME"; then
        echo "ERROR: ${HF_CLI[*]} failed for '$MODEL_NAME'." >&2
        echo "  Check HF_TOKEN, model id, and network access." >&2
        exit 1
    fi
    echo "  Pre-warm complete."
fi
# ---------------------------------------------------------------------------

num_instances=$(grep -c "container_name: ${VLLM_NAME_PREFIX:-vllm-instance}-" "$COMPOSE_FILE")
echo "--- Starting $num_instances vLLM instances + NGINX ---"

# Pre-create the compose network to avoid a known podman-compose race condition
# (multiple services in the same compose file all call "podman network exists"
# concurrently, all see "no", all try to create, only the first succeeds and
# the rest fail with exit 125 "already exists"). Pre-creating it once here,
# single-threaded, before `podman-compose up` means compose always finds it
# already present and never races.
#
# podman < 4.1 has no `network create --ignore` flag, so we must not rely on it.
# Instead we guard creation with an explicit `network exists` check, which works
# on every podman version (the stale-subnet block above already ensures any
# leftover network has the right subnet, or has been removed).
COMPOSE_PROJECT="$(basename "$GENERATED_DIR")"
COMPOSE_NETWORK="${COMPOSE_PROJECT}_vllm-network"
# Pull the subnet straight out of the generated compose so the pre-created
# network matches the static IPs assigned to each service. Without a matching
# subnet, podman-compose's ipv4_address assignments would fall outside the
# network and fail. Static IPs are required because rootless aardvark-dns is
# unreachable here, so NGINX must reach instances by IP, not hostname.
COMPOSE_SUBNET="$(grep -E '^[[:space:]]+- subnet:' "$COMPOSE_FILE" | head -1 | awk '{print $NF}')"
SUBNET_ARG=()
if [[ -n "$COMPOSE_SUBNET" ]]; then
    SUBNET_ARG=(--subnet "$COMPOSE_SUBNET")
fi
# A network left over from a crashed/killed run may carry a DIFFERENT subnet than
# the one the current compose assigns its static IPs from. A reused stale network
# would make every container fail with "requested static ip ... not in any
# subnet". Guard against it: if a network already exists with a mismatched
# subnet, remove it so we recreate it correctly.
#
# The subnet lives in different places depending on podman's network backend:
#   - netavark (podman >= 4): top-level `.Subnets` (.Subnet field)
#   - CNI (podman 3.x):       `.plugins[].ipam.ranges[][].subnet`
# Try the netavark template first, then fall back to the CNI path.
network_subnet() {
    local net="$1" out
    out="$(podman network inspect "$net" \
        --format '{{range .Subnets}}{{.Subnet}}{{end}}' 2>/dev/null)"
    if [[ -z "$out" ]]; then
        out="$(podman network inspect "$net" \
            --format '{{range .plugins}}{{range .ipam.ranges}}{{range .}}{{.subnet}}{{end}}{{end}}{{end}}' 2>/dev/null)"
    fi
    printf '%s' "$out"
}

if podman network exists "$COMPOSE_NETWORK" 2>/dev/null; then
    existing_subnet="$(network_subnet "$COMPOSE_NETWORK")"
    if [[ -n "$COMPOSE_SUBNET" && "$existing_subnet" != "$COMPOSE_SUBNET" ]]; then
        echo "  Stale network $COMPOSE_NETWORK has subnet '${existing_subnet:-none}'," \
             "expected '$COMPOSE_SUBNET' -- removing it."
        podman network rm -f "$COMPOSE_NETWORK" >/dev/null 2>&1 || true
    fi
fi

# Create only if absent. `network exists` + create is portable across all podman
# versions (podman < 4.1 lacks `network create --ignore`). The pre-create is
# single-threaded here, so there is no race with the check.
if ! podman network exists "$COMPOSE_NETWORK" 2>/dev/null; then
    podman network create \
        --label "io.podman.compose.project=${COMPOSE_PROJECT}" \
        --label "com.docker.compose.project=${COMPOSE_PROJECT}" \
        --driver bridge \
        "${SUBNET_ARG[@]}" \
        "$COMPOSE_NETWORK" >/dev/null
fi

# On the CNI backend (podman 3.x) the conflist podman just wrote may declare a
# cniVersion newer than the host's containernetworking-plugins support. The
# bridge/portmap plugins then reject it and every container silently falls back
# to the default podman net (10.88.x), losing the static IPs nginx.conf routes
# to -> every LB request 504s and the health wait below would hang the full
# timeout. Downgrade the conflist to a broadly-supported cniVersion so the
# static IPs hold. (0.4.0 is accepted by both pre-1.0 and 1.x plugins.)
downgrade_cni_version() {
    local net="$1" backend dir f
    backend="$(podman info --format '{{.Host.NetworkBackend}}' 2>/dev/null || echo "")"
    # .Host.NetworkBackend only exists on podman >= 4 (empty on podman 3.x).
    # Guarding on `== "cni"` therefore skipped the downgrade on exactly the
    # podman-3.x hosts that need it. Bail only on an explicit netavark backend;
    # for cni / empty / unknown we proceed, and the `-f "$f"` conflist check
    # below makes this a safe no-op when there's no CNI conflist to rewrite.
    [[ "$backend" == "netavark" ]] && return 0
    if [[ "$(id -u)" -eq 0 ]]; then
        dir="/etc/cni/net.d"
    else
        dir="${XDG_CONFIG_HOME:-$HOME/.config}/cni/net.d"
    fi
    f="$dir/$net.conflist"
    [[ -f "$f" ]] || return 0
    if grep -q '"cniVersion"[[:space:]]*:[[:space:]]*"1\.' "$f"; then
        echo "  CNI backend: downgrading $net cniVersion -> 0.4.0 (host plugins are pre-1.0)."
        sed -i 's/"cniVersion"[[:space:]]*:[[:space:]]*"1\.[0-9.]*"/"cniVersion": "0.4.0"/' "$f" || true
    fi
}
downgrade_cni_version "$COMPOSE_NETWORK"

cd "$GENERATED_DIR"
podman-compose up -d
cd "$SCRIPT_DIR"

echo ""
echo "--- Container status ---"
podman ps --filter "name=vllm-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

if $NO_WAIT; then
    echo ""
    echo "Started without waiting for health checks (--no-wait)."
    echo "Run ./test-setup.sh to verify the setup once instances are ready."
    exit 0
fi

echo ""
echo "--- Waiting for all instances to become healthy (timeout: ${HEALTH_TIMEOUT}s) ---"
echo "    vLLM model loading can take several minutes..."

# Required containers gate the wait loop. nginx-lb is shown for visibility
# only -- its self-healthcheck (wget through the upstream) can be flaky
# while vLLM workers are still warming up, even when the proxy is working.
required_containers=()
for i in $(seq 1 "$num_instances"); do
    required_containers+=("${VLLM_NAME_PREFIX:-vllm-instance}-$i")
done
optional_containers=("${VLLM_NGINX_NAME:-vllm-nginx-lb}")

# Guard against the CNI fallback: if a vLLM container didn't get its assigned
# static IP from our subnet, it landed on the default podman net and NGINX
# (which routes by static IP) will 504 every request -> the health wait below
# would hang the whole timeout. Detect it immediately and fail with guidance.
if [[ -n "$COMPOSE_SUBNET" ]]; then
    subnet_net="${COMPOSE_SUBNET%/*}"      # 10.201.0.0/24 -> 10.201.0.0
    expected_prefix="${subnet_net%.*}."     # -> 10.201.0.
    bad_ip=false
    for c in "${required_containers[@]}"; do
        cip="$(podman inspect \
            --format '{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}' \
            "$c" 2>/dev/null || true)"
        if [[ "$cip" != *"$expected_prefix"* ]]; then
            echo "ERROR: $c is not on the expected subnet $COMPOSE_SUBNET (IP: '${cip:-none}')." >&2
            bad_ip=true
        fi
    done
    if $bad_ip; then
        echo "" >&2
        echo "Containers fell back off their static IPs. NGINX routes by static IP, so" >&2
        echo "every load-balanced request would 504 and the health wait would hang." >&2
        echo "Most common cause on podman 3.x: a CNI cniVersion the host plugins reject" >&2
        echo "(start.sh tries to auto-downgrade it; if you still see this, the plugins" >&2
        echo "are too old). Remedies:" >&2
        echo "  - upgrade 'containernetworking-plugins' on the host, or" >&2
        echo "  - inspect: podman network inspect $COMPOSE_NETWORK" >&2
        echo "Aborting now instead of waiting ${HEALTH_TIMEOUT}s." >&2
        exit 1
    fi
fi

start_time=$(date +%s)

while true; do
    elapsed=$(( $(date +%s) - start_time ))
    if [[ $elapsed -ge $HEALTH_TIMEOUT ]]; then
        echo ""
        echo "ERROR: Timed out after ${HEALTH_TIMEOUT}s waiting for vLLM instances to become healthy."
        echo "Check logs with: podman logs <container-name>"
        for c in "${required_containers[@]}" "${optional_containers[@]}"; do
            status=$(container_health "$c")
            echo "  $c: $status"
        done
        exit 1
    fi

    # Fast-fail: if a required container has exited/died (e.g. cpuset error,
    # OOM, bad model/image), abort now instead of waiting out the full timeout.
    for c in "${required_containers[@]}"; do
        state=$(podman inspect --format '{{.State.Status}}' "$c" 2>/dev/null || echo "missing")
        if [[ "$state" == "exited" || "$state" == "dead" ]]; then
            ec=$(podman inspect --format '{{.State.ExitCode}}' "$c" 2>/dev/null || echo "?")
            echo ""
            echo "ERROR: $c is '$state' (exit code $ec) -- it will never become healthy." >&2
            echo "Last 25 log lines:" >&2
            podman logs --tail 25 "$c" 2>&1 | sed 's/^/    /' >&2 || true
            echo "Aborting now instead of waiting ${HEALTH_TIMEOUT}s." >&2
            exit 1
        fi
    done

    all_healthy=true
    status_line=""
    for c in "${required_containers[@]}"; do
        health=$(container_health "$c")
        short_name="${c#vllm-}"
        if [[ "$health" == "healthy" ]]; then
            status_line="$status_line [$short_name:OK]"
        else
            status_line="$status_line [$short_name:$health]"
            all_healthy=false
        fi
    done
    for c in "${optional_containers[@]}"; do
        health=$(container_health "$c")
        short_name="${c#vllm-}"
        # Suffix with '*' so it's visually obvious this one is informational.
        status_line="$status_line [$short_name:${health}*]"
    done

    printf "\r  [%3ds]%s" "$elapsed" "$status_line"

    if $all_healthy; then
        echo ""
        echo ""
        echo "All vLLM instances are healthy. (nginx-lb status is informational only.)"
        break
    fi

    sleep "$HEALTH_INTERVAL"
done

echo ""
echo "--- Final status ---"
podman ps --filter "name=vllm-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo ""
echo "NGINX endpoint: http://localhost:${NGINX_PORT:-8080}"
echo "Run ./test-setup.sh to validate the full setup."
