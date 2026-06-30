#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GENERATED_DIR="$SCRIPT_DIR/generated"
COMPOSE_FILE="$GENERATED_DIR/docker-compose.yml"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Stop the multi-instance vLLM + NGINX stack.

Options:
  --clean         Remove volumes (model caches) after stopping
  --purge         Remove volumes AND delete generated config files
  --log-dir DIR   Directory to dump container logs into before teardown.
                  Defaults to \$LOG_DIR if set, else
                  sweep-logs/<timestamp>/container-logs/.
                  Pass --no-logs to skip the dump entirely.
  --no-logs       Skip dumping container logs before teardown.
  -h, --help      Show this help
EOF
    exit 0
}

CLEAN=false
PURGE=false
CLI_LOG_DIR=""
DUMP_LOGS=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean) CLEAN=true; shift ;;
        --purge) PURGE=true; CLEAN=true; shift ;;
        --log-dir) CLI_LOG_DIR="$2"; shift 2 ;;
        --no-logs) DUMP_LOGS=false; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ----------------------------------------------------------------------------
# Dump podman logs for every stack container BEFORE we stop/remove them.
# Matches vllm-* and nginx* containers (the names used by generate-config.sh).
# Includes stopped containers so we still capture exit logs.
# ----------------------------------------------------------------------------
dump_container_logs() {
    if ! $DUMP_LOGS; then
        return 0
    fi

    local containers
    containers=$(podman ps -a \
        --filter "name=vllm-" \
        --filter "name=nginx" \
        --format "{{.Names}}" | sort -u)

    if [[ -z "$containers" ]]; then
        echo "  No vllm-* or nginx* containers found -- skipping log dump."
        return 0
    fi

    local out_dir
    if [[ -n "$CLI_LOG_DIR" ]]; then
        out_dir="$CLI_LOG_DIR"
    elif [[ -n "${LOG_DIR:-}" ]]; then
        out_dir="$LOG_DIR/container-logs"
    else
        out_dir="$SCRIPT_DIR/sweep-logs/$(date +%Y%m%d-%H%M%S)/container-logs"
    fi
    mkdir -p "$out_dir"

    echo "--- Dumping container logs to $out_dir ---"
    local c
    for c in $containers; do
        local log_file="$out_dir/${c}.log"
        if podman logs "$c" > "$log_file" 2>&1; then
            local size
            size=$(wc -c < "$log_file" | tr -d ' ')
            echo "  saved $log_file ($size bytes)"
        else
            echo "  WARNING: failed to dump logs for $c (see $log_file)"
        fi
    done
}

dump_container_logs

if [[ ! -f "$COMPOSE_FILE" ]]; then
    echo "No compose file found at $COMPOSE_FILE."
    echo "Attempting to stop containers by name..."
    for c in $(podman ps -a --filter "name=vllm-" --format "{{.Names}}"); do
        echo "  Stopping $c..."
        podman stop "$c" --timeout 10 2>/dev/null || true
        podman rm "$c" 2>/dev/null || true
    done
    echo "Done."
    exit 0
fi

echo "--- Stopping multi-instance vLLM stack ---"

cd "$GENERATED_DIR"

if $CLEAN; then
    echo "  Stopping containers and removing volumes..."
    podman-compose down -v
else
    echo "  Stopping containers (volumes preserved)..."
    podman-compose down
fi

cd "$SCRIPT_DIR"

# Explicitly remove the compose network. `podman-compose down` leaves it behind
# when a run was killed mid-flight, and a stale network with a mismatched subnet
# breaks the next start.sh ("requested static ip ... not in any subnet").
COMPOSE_NETWORK="$(basename "$GENERATED_DIR")_vllm-network"
if podman network exists "$COMPOSE_NETWORK" 2>/dev/null; then
    echo "  Removing network $COMPOSE_NETWORK..."
    podman network rm -f "$COMPOSE_NETWORK" >/dev/null 2>&1 || true
fi

echo ""
echo "--- Remaining vLLM containers ---"
remaining=$(podman ps -a --filter "name=vllm-" --format "{{.Names}}" | wc -l)
if [[ "$remaining" -eq 0 ]]; then
    echo "  None (clean shutdown)."
else
    podman ps -a --filter "name=vllm-" --format "table {{.Names}}\t{{.Status}}"
fi

if $PURGE; then
    echo ""
    echo "--- Removing generated config ---"
    rm -rf "$GENERATED_DIR"
    echo "  Deleted $GENERATED_DIR"
fi

echo ""
echo "Stack stopped."
