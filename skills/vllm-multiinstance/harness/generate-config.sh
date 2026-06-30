#!/bin/bash
set -euo pipefail

# Configurable parameters (override via env vars or flags)
NUM_INSTANCES="${NUM_INSTANCES:-5}"
CORES_PER_INSTANCE="${CORES_PER_INSTANCE:-32}"
NGINX_CORES="${NGINX_CORES:-1-15}"
VLLM_START_CORE="${VLLM_START_CORE:-32}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
HF_TOKEN="${HF_TOKEN:-}"
MEM_LIMIT="${MEM_LIMIT:-100g}"
NGINX_MEM_LIMIT="${NGINX_MEM_LIMIT:-5g}"
SHM_SIZE="${SHM_SIZE:-16g}"
# Extra args appended verbatim to the vLLM command in each instance
# (e.g. EXTRA_VLLM_ARGS="--trust-remote-code --max-model-len 8192").
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
VLLM_IMAGE="${VLLM_IMAGE:-amdih/zendnn_zentorch:vllm_v0.22.0_zentorch_v2.11.0.1_ubuntu22.04_2026_ww23}"
VLLM_KV_CACHE_SPACE="${VLLM_KV_CACHE_SPACE:-63}"
NGINX_PORT="${NGINX_PORT:-8080}"
# Dedicated /24 subnet for the compose bridge network. Each vLLM instance gets a
# static IP on this subnet so NGINX can reach instances by IP instead of by
# hostname. This is required because rootless podman's DNS (aardvark-dns) is not
# reachable on the bridge gateway in this environment -- container name/alias
# resolution fails with "connection refused", which made NGINX crash-loop with
# "host not found in upstream". Static IPs sidestep DNS entirely. Use a subnet
# outside netavark's default 10.89.0.0/16 auto-allocation pool to avoid clashes.
VLLM_SUBNET="${VLLM_SUBNET:-10.201.0.0/24}"
# Derive the /24 prefix (e.g. 10.201.0.0/24 -> 10.201.0). Instance i is assigned
# <prefix>.IP_BASE+i; NGINX takes <prefix>.IP_BASE.
_subnet_base="${VLLM_SUBNET%/*}"
IP_PREFIX="${_subnet_base%.*}"
IP_BASE="${IP_BASE:-10}"
# Run each vLLM instance with HF_HUB_OFFLINE so it loads the model straight from
# the shared cache instead of contacting huggingface.co. Defaults to 1 (offline)
# because start.sh pre-warms the cache on the host before any container starts,
# so the model is always present on disk -- and the compose bridge network has no
# reliable DNS to huggingface.co, which otherwise makes instances crash with
# "Temporary failure in name resolution". Set to 0 (or pass --hf-online) only
# when instances must download/refresh the model themselves.
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
MODELS_DIR="${MODELS_DIR:-}"
# Shared HuggingFace cache dir bind-mounted into every vLLM instance at
# /opt/app-root/src/.cache. Lets all instances reuse a single set of model
# downloads instead of each one filling its own named volume. Persists across
# `podman compose down` / `podman system prune` because it's a host path.
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/hf-shared}"

# Backend selection: "zentorch" (default) or "native". Override via the
# BACKEND env var or the --native/--zentorch flags. Native mode skips the
# zentorch-specific env vars emitted into the compose file.
BACKEND="${BACKEND:-zentorch}"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Generate docker-compose.yml and nginx.conf for multi-instance vLLM.

Options:
  -n, --num-instances N       Number of vLLM instances (default: $NUM_INSTANCES)
  -c, --cores-per-instance N  Cores per instance (default: $CORES_PER_INSTANCE)
  --nginx-cores RANGE         CPU range for nginx (default: $NGINX_CORES)
  --start-core N              First core for vLLM instances (default: $VLLM_START_CORE)
  -m, --model NAME            Model name or path (default: $MODEL_NAME)
  --image IMAGE               vLLM container image (default: $VLLM_IMAGE)
  --mem-limit SIZE            Memory limit per instance (default: $MEM_LIMIT)
  --kv-cache-space N          KV cache space in GB (default: $VLLM_KV_CACHE_SPACE)
  --nginx-port PORT           Host port for nginx (default: $NGINX_PORT)
  --hf-offline                Run instances with HF_HUB_OFFLINE=1 (default; load
                              from the pre-warmed shared cache, no network)
  --hf-online                 Run instances with HF_HUB_OFFLINE=0 (let each
                              instance contact huggingface.co)
  --models-dir DIR            Host dir bind-mounted into each container at
                              the same absolute path (for local model loading)
  --hf-cache-dir DIR          Host dir bind-mounted as the shared HuggingFace
                              cache in every instance (default: $HF_CACHE_DIR).
                              All instances reuse one cache so each model is
                              downloaded exactly once per sweep.
  --native                    Use native backend: skip zentorch-specific env
                              vars (same as BACKEND=native)
  --zentorch                  Use zentorch backend (default; BACKEND=zentorch)
  --no-limits                 Skip cpuset/mem_limit/shm_size (for rootless/LSF environments)
  --no-mem-limit              Drop mem_limit only; keep cpuset/shm_size/cap_add/security_opt
  -o, --output-dir DIR        Output directory (default: generated/)
  --dry-run                   Print config summary without writing files
  -h, --help                  Show this help

Environment variables:
  NUM_INSTANCES, CORES_PER_INSTANCE, NGINX_CORES, VLLM_START_CORE,
  MODEL_NAME, HF_TOKEN, MEM_LIMIT, SHM_SIZE, VLLM_IMAGE,
  VLLM_KV_CACHE_SPACE, NGINX_PORT, HF_HUB_OFFLINE (0|1, default 1),
  BACKEND (zentorch|native, default zentorch)

Example:
  # 3 instances on a 96-core CPU, starting at core 16
  $0 -n 3 -c 24 --start-core 16 --nginx-cores 0-15

  # Use a local model path
  $0 -m /models/Llama-3.1-8B-Instruct
EOF
    exit 0
}

OUTPUT_DIR="generated"
DRY_RUN=false
NO_LIMITS=false
NO_MEM_LIMIT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-instances) NUM_INSTANCES="$2"; shift 2 ;;
        -c|--cores-per-instance) CORES_PER_INSTANCE="$2"; shift 2 ;;
        --nginx-cores) NGINX_CORES="$2"; shift 2 ;;
        --start-core) VLLM_START_CORE="$2"; shift 2 ;;
        -m|--model) MODEL_NAME="$2"; shift 2 ;;
        --image) VLLM_IMAGE="$2"; shift 2 ;;
        --mem-limit) MEM_LIMIT="$2"; shift 2 ;;
        --kv-cache-space) VLLM_KV_CACHE_SPACE="$2"; shift 2 ;;
        --nginx-port) NGINX_PORT="$2"; shift 2 ;;
        --hf-offline) HF_HUB_OFFLINE=1; shift ;;
        --hf-online) HF_HUB_OFFLINE=0; shift ;;
        --models-dir) MODELS_DIR="$2"; shift 2 ;;
        --hf-cache-dir) HF_CACHE_DIR="$2"; shift 2 ;;
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --native) BACKEND=native; shift ;;
        --zentorch) BACKEND=zentorch; shift ;;
        --no-limits) NO_LIMITS=true; shift ;;
        --no-mem-limit) NO_MEM_LIMIT=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

case "$BACKEND" in
    zentorch) NATIVE_MODE=false ;;
    native)   NATIVE_MODE=true ;;
    *) echo "ERROR: invalid BACKEND='$BACKEND' (expected 'zentorch' or 'native')." >&2; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR/$OUTPUT_DIR"

# Resolve HF_CACHE_DIR to an absolute path and ensure it exists so podman
# doesn't auto-create it as root-owned on first mount.
mkdir -p "$HF_CACHE_DIR"
HF_CACHE_DIR="$(cd "$HF_CACHE_DIR" && pwd)"

# Append each token in EXTRA_VLLM_ARGS as a YAML list item to the compose
# command block (word-split on whitespace; no quoting/escaping needed for
# typical flags like --trust-remote-code or --max-model-len 8192).
emit_extra_vllm_args() {
    [[ -z "$EXTRA_VLLM_ARGS" ]] && return 0
    local arg
    for arg in $EXTRA_VLLM_ARGS; do
        printf '      - %s\n' "$arg" >> "$COMPOSE"
    done
}

last_core=$(( VLLM_START_CORE + NUM_INSTANCES * CORES_PER_INSTANCE - 1 ))

echo "=== Multi-Instance vLLM Configuration ==="
echo "  Instances:       $NUM_INSTANCES"
echo "  Cores/instance:  $CORES_PER_INSTANCE"
echo "  NGINX cores:     $NGINX_CORES"
echo "  vLLM cores:      $VLLM_START_CORE-$last_core"
echo "  Model:           $MODEL_NAME"
echo "  Backend:         $BACKEND"
echo "  Image:           $VLLM_IMAGE"
echo "  Memory limit:    $MEM_LIMIT per instance"
echo "  KV cache space:  ${VLLM_KV_CACHE_SPACE}GB"
echo "  NGINX port:      $NGINX_PORT"
echo "  Network subnet:  $VLLM_SUBNET (nginx=${IP_PREFIX}.${IP_BASE}, vllm-i=${IP_PREFIX}.$((IP_BASE+1))..)"
echo "  HF hub offline:  $HF_HUB_OFFLINE"
echo "  Models dir:      ${MODELS_DIR:-<unset>}"
echo "  HF cache dir:    $HF_CACHE_DIR (shared across all instances)"
echo "  Output dir:      $OUTPUT_DIR"
echo "=========================================="

if $DRY_RUN; then
    echo ""
    echo "Core allocation:"
    for i in $(seq 1 "$NUM_INSTANCES"); do
        start=$(( VLLM_START_CORE + (i - 1) * CORES_PER_INSTANCE ))
        end=$(( start + CORES_PER_INSTANCE - 1 ))
        echo "  vllm-$i: cores $start-$end"
    done
    echo "  nginx:  cores $NGINX_CORES"
    exit 0
fi

mkdir -p "$OUTPUT_DIR"

# --- Generate .env ---
cat > "$OUTPUT_DIR/.env" <<EOF
MODEL_NAME=$MODEL_NAME
HF_TOKEN=$HF_TOKEN
MEM_LIMIT=$MEM_LIMIT
HF_CACHE_DIR=$HF_CACHE_DIR
EOF
echo "  Written: $OUTPUT_DIR/.env"

# --- Generate nginx.conf ---
cat > "$OUTPUT_DIR/nginx.conf" <<'NGINX_HEADER'
user root;
pid /tmp/nginx.pid;

events {
    worker_connections 4096;
}

http {
    upstream vllm_backend {
NGINX_HEADER

for i in $(seq 1 "$NUM_INSTANCES"); do
    echo "        server ${IP_PREFIX}.$(( IP_BASE + i )):8000 max_fails=10 fail_timeout=10s;" >> "$OUTPUT_DIR/nginx.conf"
done

cat >> "$OUTPUT_DIR/nginx.conf" <<'NGINX_BODY'
    }

    log_format upstream_log '$remote_addr - [$time_local] "$request" $status '
                            'upstream=$upstream_addr response_time=$upstream_response_time';

    access_log /var/log/nginx/access.log upstream_log;
    error_log /var/log/nginx/error.log warn;

    server {
        listen 80;
        server_name _;

        proxy_connect_timeout 2s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        send_timeout 300s;

        proxy_buffer_size 128k;
        proxy_buffers 8 256k;
        proxy_busy_buffers_size 512k;

        location /health {
            proxy_pass http://vllm_backend/health;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_next_upstream error timeout http_502 http_503 http_504;
            proxy_next_upstream_tries 0;
        }

        location / {
            proxy_pass http://vllm_backend;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_cache off;
            proxy_next_upstream error timeout http_502 http_503 http_504;
            proxy_next_upstream_tries 0;
            client_max_body_size 50M;
        }

        location /v1/completions {
            proxy_pass http://vllm_backend/v1/completions;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_cache off;
            proxy_next_upstream error timeout http_502 http_503 http_504;
            proxy_next_upstream_tries 0;
            client_max_body_size 50M;
        }

        location /v1/chat/completions {
            proxy_pass http://vllm_backend/v1/chat/completions;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Connection "";
            proxy_buffering off;
            proxy_cache off;
            proxy_next_upstream error timeout http_502 http_503 http_504;
            proxy_next_upstream_tries 0;
            client_max_body_size 50M;
        }

        location /v1/models {
            proxy_pass http://vllm_backend/v1/models;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
        }

        location /nginx_status {
            stub_status on;
            access_log off;
        }
    }
}
NGINX_BODY
echo "  Written: $OUTPUT_DIR/nginx.conf"

# --- Generate docker-compose.yml ---
COMPOSE="$OUTPUT_DIR/docker-compose.yml"

cat > "$COMPOSE" <<EOF
version: '3.8'

services:
EOF

DEPENDS_LIST=""
for i in $(seq 1 "$NUM_INSTANCES"); do
    start=$(( VLLM_START_CORE + (i - 1) * CORES_PER_INSTANCE ))
    end=$(( start + CORES_PER_INSTANCE - 1 ))
    DEPENDS_LIST="${DEPENDS_LIST}      - vllm-$i
"

    cat >> "$COMPOSE" <<EOF
  vllm-$i:
    image: $VLLM_IMAGE
    container_name: ${VLLM_NAME_PREFIX:-vllm-instance}-$i
EOF

    if ! $NO_LIMITS; then
        cat >> "$COMPOSE" <<EOF
    cpuset: "$start-$end"
EOF
        if ! $NO_MEM_LIMIT; then
            cat >> "$COMPOSE" <<EOF
    mem_limit: \${MEM_LIMIT:-$MEM_LIMIT}
EOF
        fi
        cat >> "$COMPOSE" <<EOF
    shm_size: $SHM_SIZE
    cap_add:
      - SYS_NICE
    security_opt:
      - seccomp=unconfined
EOF
    fi

    cat >> "$COMPOSE" <<EOF
    environment:
      - HF_HUB_OFFLINE=$HF_HUB_OFFLINE
      - HF_HOME=/opt/app-root/src/.cache/huggingface
      - HF_TOKEN=\${HF_TOKEN}
      - VLLM_CPU_KVCACHE_SPACE=$VLLM_KV_CACHE_SPACE
EOF

    if $NATIVE_MODE; then
        cat >> "$COMPOSE" <<EOF
      - TORCHINDUCTOR_FREEZING=0
EOF
    else
        cat >> "$COMPOSE" <<EOF
      - TORCHINDUCTOR_FREEZING=1
      - VLLM_USE_AOT_COMPILE=0
      - TORCHINDUCTOR_AUTOGRAD_CACHE=0
      - ZENDNNL_MATMUL_ALGO=1
EOF
    fi

    cat >> "$COMPOSE" <<EOF
    volumes:
      - $HF_CACHE_DIR:/opt/app-root/src/.cache:z
EOF

    if [[ -n "$MODELS_DIR" ]]; then
        cat >> "$COMPOSE" <<EOF
      - $MODELS_DIR:$MODELS_DIR:ro,z
EOF
    fi

    cat >> "$COMPOSE" <<EOF
    entrypoint:
      - bash
      - -c
      - 'export VLLM_CPU_OMP_THREADS_BIND=$start-$end && echo "=== VLLM_CPU_OMP_THREADS_BIND=\$\$VLLM_CPU_OMP_THREADS_BIND ===" && echo "=== pip list (\$\$(hostname)) ===" && pip list && echo "=== launching: vllm serve \$\$@ ===" && exec vllm serve "\$\$@"'
      - --
    command:
      - --model
      - \${MODEL_NAME:-$MODEL_NAME}
      - --port
      - "8000"
      - --host
      - "0.0.0.0"
      - --no-enable-prefix-caching
      - --max-model-len
      - "4096"
EOF

    emit_extra_vllm_args

    cat >> "$COMPOSE" <<EOF
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 30s
      retries: 10
      start_period: 600s
    networks:
      vllm-network:
        ipv4_address: ${IP_PREFIX}.$(( IP_BASE + i ))

EOF
done

cat >> "$COMPOSE" <<EOF
  nginx:
    image: docker.io/library/nginx:alpine
    container_name: ${VLLM_NGINX_NAME:-vllm-nginx-lb}
EOF

    if ! $NO_LIMITS; then
        cat >> "$COMPOSE" <<EOF
    cpuset: "$NGINX_CORES"
    mem_limit: $NGINX_MEM_LIMIT
EOF
    fi

    cat >> "$COMPOSE" <<EOF
    ports:
      - "$NGINX_PORT:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro,Z
    depends_on:
${DEPENDS_LIST}    networks:
      vllm-network:
        ipv4_address: ${IP_PREFIX}.${IP_BASE}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost/health"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 360s

networks:
  vllm-network:
    driver: bridge
    ipam:
      config:
        - subnet: $VLLM_SUBNET
EOF

echo "  Written: $OUTPUT_DIR/docker-compose.yml"
echo ""
echo "Configuration generated successfully in $OUTPUT_DIR/"
echo "Core allocation:"
for i in $(seq 1 "$NUM_INSTANCES"); do
    start=$(( VLLM_START_CORE + (i - 1) * CORES_PER_INSTANCE ))
    end=$(( start + CORES_PER_INSTANCE - 1 ))
    echo "  vllm-$i: cores $start-$end"
done
echo "  nginx:  cores $NGINX_CORES"
