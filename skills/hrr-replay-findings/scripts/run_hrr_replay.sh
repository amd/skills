#!/usr/bin/env bash
# Run HRR replay against an archive and optionally analyze the log.
# Portable wrapper: native hrr-playback or Docker (ROCm/vLLM image + HIP inject).
#
# Usage:
#   GPU=1 ./run_hrr_replay.sh --archive capture.hrr/pid-123 --log replay.log
#   ./run_hrr_replay.sh --archive capture.hrr/pid-123 --info
#   ./run_hrr_replay.sh --archive capture.hrr/pid-123 --analyze
#
# Env:
#   HRR_PLAYBACK     path to hrr-playback (required unless on PATH)
#   HIP_SO, HSA_SO   for Docker inject mode (libamdhip64 + libhsa)
#   GPU              device index (default 0)
#   IMAGE            Docker image (default rocm/vllm rocm7.13 gfx950)
#   HRR_REPLAY_MODE  auto|native|docker (default auto)
#   HRR_REPO_ROOT    if set and contains scripts/maf-hrr-docker-playback.sh, use it
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZER="$SCRIPT_DIR/analyze_replay_finding.py"

ARCHIVE=""
LOG=""
DO_ANALYZE=0
DO_INFO=0
MODE="${HRR_REPLAY_MODE:-auto}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive) ARCHIVE="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    --analyze) DO_ANALYZE=1; shift ;;
    --info) DO_INFO=1; shift ;;
    --native) MODE=native; shift ;;
    --docker) MODE=docker; shift ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

[[ -n "$ARCHIVE" ]] || { echo "error: --archive required" >&2; exit 1; }
ARCHIVE="$(readlink -f "$ARCHIVE" 2>/dev/null || realpath "$ARCHIVE" 2>/dev/null || echo "$ARCHIVE")"
[[ -d "$ARCHIVE" ]] || { echo "error: archive not found: $ARCHIVE" >&2; exit 1; }

resolve_playback() {
  if [[ -n "${HRR_PLAYBACK:-}" && -x "$HRR_PLAYBACK" ]]; then
    echo "$HRR_PLAYBACK"
    return
  fi
  if command -v hrr-playback >/dev/null 2>&1; then
    command -v hrr-playback
    return
  fi
  echo ""
}

HRR_PLAY="$(resolve_playback)"
GPU="${GPU:-0}"

if [[ "$DO_INFO" == "1" ]]; then
  [[ -n "$HRR_PLAY" ]] || { echo "error: hrr-playback not found; set HRR_PLAYBACK" >&2; exit 1; }
  exec "$HRR_PLAY" "$ARCHIVE" --info "${EXTRA_ARGS[@]}"
fi

if [[ -z "$LOG" ]]; then
  LOG="hrr-replay-$(basename "$ARCHIVE")-gpu${GPU}-$(date -u +%Y%m%dT%H%M%SZ).log"
fi
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true

# Prefer workspace docker helper when available (full inject recipe).
if [[ -n "${HRR_REPO_ROOT:-}" && -f "$HRR_REPO_ROOT/scripts/maf-hrr-docker-playback.sh" ]]; then
  echo "[run_hrr_replay] using $HRR_REPO_ROOT/scripts/maf-hrr-docker-playback.sh"
  sudo docker rm -f "${HRR_NAME:-maf-hrr-playback}" 2>/dev/null || true
  set +e
  sudo -E GPU="$GPU" bash "$HRR_REPO_ROOT/scripts/maf-hrr-docker-playback.sh" \
    "$ARCHIVE" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  set -e
elif [[ "$MODE" == "docker" || ( "$MODE" == "auto" && -n "${HIP_SO:-}" ) ]]; then
  [[ -n "$HRR_PLAY" ]] || { echo "error: hrr-playback required for docker mode; set HRR_PLAYBACK" >&2; exit 1; }
  IMAGE="${IMAGE:-rocm/vllm:rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1}"
  HIP_DST=/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7
  HSA_DST=/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1
  ROC_LIB=/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib
  CORES_HOST="${CORES_HOST:-/tmp/hrr-cores}"
  mkdir -p "$CORES_HOST"
  docker_env=( -e ROCR_VISIBLE_DEVICES="$GPU" -e HRR_INJECT=1 -e LD_LIBRARY_PATH="$ROC_LIB:/opt/rocm/lib" )
  vols=( -v "$ARCHIVE":/data/hrr-archive:ro -v "$HRR_PLAY":/opt/hrr-tools/hrr-playback:ro -v "$CORES_HOST":/data/cores )
  [[ -f "${HIP_SO:-}" ]] && vols+=( -v "$HIP_SO":"$HIP_DST":ro )
  [[ -f "${HSA_SO:-}" ]] && vols+=( -v "$HSA_SO":"$HSA_DST":ro )
  echo "[run_hrr_replay] docker image=$IMAGE GPU=$GPU archive=$ARCHIVE"
  sudo docker rm -f "${HRR_NAME:-maf-hrr-playback}" 2>/dev/null || true
  set +e
  sudo -E docker run --rm --privileged --init \
    --name "${HRR_NAME:-maf-hrr-playback}" \
    --device /dev/kfd -v /dev/dri:/dev/dri --shm-size=4g \
    --security-opt seccomp=unconfined --ulimit core=-1:-1 \
    "${docker_env[@]}" "${vols[@]}" \
    "$IMAGE" /opt/hrr-tools/hrr-playback /data/hrr-archive "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  set -e
else
  [[ -n "$HRR_PLAY" ]] || { echo "error: hrr-playback not found; set HRR_PLAYBACK or use --docker with HIP_SO" >&2; exit 1; }
  echo "[run_hrr_replay] native GPU=$GPU archive=$ARCHIVE"
  set +e
  ROCR_VISIBLE_DEVICES="$GPU" "$HRR_PLAY" "$ARCHIVE" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  set -e
fi

echo "[run_hrr_replay] log=$LOG exit=$RC"

if [[ "$DO_ANALYZE" == "1" ]]; then
  FINDING="${LOG%.log}.finding.md"
  python3 "$ANALYZER" --log "$LOG" --archive "$ARCHIVE" --hrr-playback "$HRR_PLAY" \
    --format markdown -o "$FINDING"
  echo "[run_hrr_replay] finding=$FINDING"
fi

exit "$RC"
