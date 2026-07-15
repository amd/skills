#!/usr/bin/env bash
# Run HRR replay on the host GPU and optionally analyze the log.
#
# Needs: archive directory + hrr-playback (on PATH or HRR_PLAYBACK).
# No Docker, no source tree, no GPU index (auto-picked by free VRAM).
#
# Usage:
#   ./run_hrr_replay.sh --archive capture.hrr/pid-123 --analyze
#   ./run_hrr_replay.sh --archive capture.hrr/pid-123 --info
#
# Optional env:
#   HRR_PLAYBACK   path to hrr-playback if not on PATH
#   GPU            force device index; otherwise auto-pick most free VRAM
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZER="$SCRIPT_DIR/analyze_replay_finding.py"

ARCHIVE=""
LOG=""
DO_ANALYZE=0
DO_INFO=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive) ARCHIVE="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    --analyze) DO_ANALYZE=1; shift ;;
    --info) DO_INFO=1; shift ;;
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
    echo "error: hrr-playback not found. Set HRR_PLAYBACK=/path/to/hrr-playback" >&2
    exit 1
  }
  exec "$HRR_PLAY" "$ARCHIVE" --info "${EXTRA_ARGS[@]}"
fi

[[ -n "$HRR_PLAY" ]] || {
  echo "error: hrr-playback not found. Set HRR_PLAYBACK=/path/to/hrr-playback" >&2
  exit 1
}

[[ -r /dev/kfd ]] || {
  echo "error: /dev/kfd not accessible — AMD GPU driver required for native replay" >&2
  exit 1
}

if [[ -z "$LOG" ]]; then
  LOG="hrr-replay-$(basename "$ARCHIVE")-$(date -u +%Y%m%dT%H%M%SZ).log"
fi
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true

echo "[run_hrr_replay] native GPU=$GPU archive=$ARCHIVE"
set +e
ROCR_VISIBLE_DEVICES="$GPU" "$HRR_PLAY" "$ARCHIVE" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "[run_hrr_replay] log=$LOG exit=$RC"

if [[ "$DO_ANALYZE" == "1" ]]; then
  FINDING="${LOG%.log}.finding.md"
  python3 "$ANALYZER" --log "$LOG" --archive "$ARCHIVE" \
    ${HRR_PLAY:+--hrr-playback "$HRR_PLAY"} \
    --format markdown -o "$FINDING"
  echo "[run_hrr_replay] finding=$FINDING"
fi

exit "$RC"
