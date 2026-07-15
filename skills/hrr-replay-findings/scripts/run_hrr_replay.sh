#!/usr/bin/env bash
# Run HRR replay against an archive and optionally analyze the log.
#
# Customer-facing: needs only an archive directory and hrr-playback (on PATH or
# HRR_PLAYBACK). No source tree, no GPU index, no HIP library paths required.
#
# Usage:
#   ./run_hrr_replay.sh --archive capture.hrr/pid-123 --analyze
#   ./run_hrr_replay.sh --archive capture.hrr/pid-123 --info
#
# Optional env (agent may set; user does not need to know):
#   HRR_PLAYBACK   path to hrr-playback if not on PATH
#   GPU            force device index; otherwise auto-pick most free VRAM
#   IMAGE          container for docker replay (default: ROCm/vLLM image)
#   HRR_REPLAY_MODE  auto|native|docker
#   HIP_SO, HSA_SO   advanced: override container HIP/HSA libs (support bundles only)
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

pick_gpu() {
  if [[ -n "${GPU:-}" ]]; then
    echo "$GPU"
    return
  fi
  if command -v rocm-smi >/dev/null 2>&1; then
    local best="" best_free=-1
    while read -r idx free; do
      [[ -n "$idx" ]] || continue
      if (( free > best_free )); then
        best_free=$free
        best=$idx
      fi
    done < <(rocm-smi --showmeminfo vram 2>/dev/null | awk '
      /GPU\[/ { gsub(/[^0-9]/,"",$1); idx=$1 }
      /Used Memory/ { used=$NF }
      /Total Memory/ { total=$NF; if (idx!="") { print idx, total-used; idx="" } }
    ')
    if [[ -n "$best" ]]; then
      echo "[run_hrr_replay] auto-selected GPU $best (most free VRAM)" >&2
      echo "$best"
      return
    fi
  fi
  echo "[run_hrr_replay] default GPU 0 (set GPU=N to override)" >&2
  echo "0"
}

HRR_PLAY="$(resolve_playback)"
GPU="$(pick_gpu)"

if [[ "$DO_INFO" == "1" ]]; then
  [[ -n "$HRR_PLAY" ]] || {
    echo "error: hrr-playback not found. Install it on PATH or set HRR_PLAYBACK=/path/to/hrr-playback" >&2
    exit 1
  }
  exec "$HRR_PLAY" "$ARCHIVE" --info "${EXTRA_ARGS[@]}"
fi

if [[ -z "$LOG" ]]; then
  LOG="hrr-replay-$(basename "$ARCHIVE")-$(date -u +%Y%m%dT%H%M%SZ).log"
fi
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true

use_docker=0
if [[ "$MODE" == "docker" ]]; then
  use_docker=1
elif [[ "$MODE" == "native" ]]; then
  use_docker=0
elif command -v docker >/dev/null 2>&1 && [[ ! -r /dev/kfd ]]; then
  use_docker=1
else
  use_docker=0
fi

run_native() {
  [[ -n "$HRR_PLAY" ]] || {
    echo "error: hrr-playback not found. Install on PATH or set HRR_PLAYBACK." >&2
    exit 1
  }
  echo "[run_hrr_replay] native playback GPU=$GPU archive=$ARCHIVE"
  set +e
  ROCR_VISIBLE_DEVICES="$GPU" "$HRR_PLAY" "$ARCHIVE" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
  return "${PIPESTATUS[0]}"
}

run_docker() {
  [[ -n "$HRR_PLAY" ]] || {
    echo "error: hrr-playback not found. Set HRR_PLAYBACK to the shipped playback binary." >&2
    exit 1
  }
  command -v docker >/dev/null 2>&1 || { echo "error: docker not found" >&2; exit 1; }
  IMAGE="${IMAGE:-rocm/vllm:rocm7.13.0_gfx950-dcgpu_ubuntu24.04_py3.13_pytorch_2.10.0_vllm_0.19.1}"
  ROC_LIB=/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib
  docker_env=( -e ROCR_VISIBLE_DEVICES="$GPU" -e LD_LIBRARY_PATH="$ROC_LIB:/opt/rocm/lib" )
  vols=( -v "$ARCHIVE":/data/hrr-archive:ro -v "$HRR_PLAY":/opt/hrr-tools/hrr-playback:ro )
  # Stock container HIP/HSA by default — no source build or library paths required.
  if [[ -f "${HIP_SO:-}" ]]; then
    docker_env+=( -e HRR_INJECT=1 )
    vols+=( -v "$HIP_SO":/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7:ro )
    [[ -f "${HSA_SO:-}" ]] && vols+=( -v "$HSA_SO":/opt/python/lib/python3.13/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1:ro )
    echo "[run_hrr_replay] docker with custom HIP/HSA overlay (support bundle)" >&2
  else
    echo "[run_hrr_replay] docker with stock container ROCm stack" >&2
  fi
  echo "[run_hrr_replay] image=$IMAGE GPU=$GPU archive=$ARCHIVE"
  sudo docker rm -f "${HRR_NAME:-hrr-replay}" 2>/dev/null || true
  set +e
  sudo -E docker run --rm --privileged --init \
    --name "${HRR_NAME:-hrr-replay}" \
    --device /dev/kfd -v /dev/dri:/dev/dri --shm-size=4g \
    --security-opt seccomp=unconfined --ulimit core=-1:-1 \
    "${docker_env[@]}" "${vols[@]}" \
    "$IMAGE" /opt/hrr-tools/hrr-playback /data/hrr-archive "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG"
  return "${PIPESTATUS[0]}"
}

if [[ "$use_docker" == "1" ]]; then
  run_docker
  RC=$?
else
  run_native
  RC=$?
fi

echo "[run_hrr_replay] log=$LOG exit=$RC"

if [[ "$DO_ANALYZE" == "1" ]]; then
  FINDING="${LOG%.log}.finding.md"
  python3 "$ANALYZER" --log "$LOG" --archive "$ARCHIVE" \
    ${HRR_PLAY:+--hrr-playback "$HRR_PLAY"} \
    --format markdown -o "$FINDING"
  echo "[run_hrr_replay] finding=$FINDING"
fi

exit "$RC"
