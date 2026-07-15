#!/usr/bin/env bash
# Run HRR replay on the host GPU and optionally analyze the log.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYZER="$SCRIPT_DIR/analyze_replay_finding.py"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

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
  local c candidates=()
  [[ -n "${HRR_PLAYBACK:-}" ]] && candidates+=("$HRR_PLAYBACK")
  if command -v hrr-playback >/dev/null 2>&1; then
    candidates+=("$(command -v hrr-playback)")
  fi
  candidates+=(
    "$ROCM_PATH/bin/hrr-playback"
    "/opt/rocm/bin/hrr-playback"
  )
  local p
  for p in "${candidates[@]}"; do
    [[ -n "$p" && -x "$p" ]] || continue
    echo "$p"
    return
  done
  echo ""
}

setup_library_path() {
  local play="$1"
  local bin_dir lib_dir paths=()
  bin_dir="$(cd "$(dirname "$play")" && pwd)"
  lib_dir="$(cd "$bin_dir/.." && pwd)/lib"
  paths+=("$ROCM_PATH/lib" "/opt/rocm/lib")
  [[ -d "$lib_dir" ]] && paths+=("$lib_dir")
  [[ -d "$bin_dir/../lib" ]] && paths+=("$(cd "$bin_dir/../lib" && pwd)")
  local seen="" p
  for p in "${paths[@]}"; do
    [[ -d "$p" ]] || continue
    [[ ":$seen:" == *":$p:"* ]] && continue
    seen="${seen:+$seen:}$p"
    LD_LIBRARY_PATH="${p}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  done
  export LD_LIBRARY_PATH
}

pick_gpu() {
  if [[ -n "${GPU:-}" ]]; then
    echo "$GPU"
    return
  fi
  if command -v rocm-smi >/dev/null 2>&1; then
    local best="" best_free=-1 idx free
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
  echo "[run_hrr_replay] default GPU 0" >&2
  echo "0"
}

HRR_PLAY="$(resolve_playback)"
GPU="$(pick_gpu)"

if [[ -z "$HRR_PLAY" ]]; then
  echo "error: hrr-playback not found. Checked PATH, \$ROCM_PATH/bin ($ROCM_PATH), /opt/rocm/bin." >&2
  echo "error: Ask the user where hrr-playback is installed, then set HRR_PLAYBACK for this run." >&2
  exit 1
fi

setup_library_path "$HRR_PLAY"
echo "[run_hrr_replay] playback=$HRR_PLAY" >&2

if [[ "$DO_INFO" == "1" ]]; then
  exec "$HRR_PLAY" "$ARCHIVE" --info "${EXTRA_ARGS[@]}"
fi

[[ -r /dev/kfd ]] || {
  echo "error: /dev/kfd not accessible — AMD GPU driver required" >&2
  exit 1
}

if [[ -z "$LOG" ]]; then
  LOG="hrr-replay-$(basename "$ARCHIVE")-$(date -u +%Y%m%dT%H%M%SZ).log"
fi
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true

echo "[run_hrr_replay] GPU=$GPU archive=$ARCHIVE" >&2
set +e
ROCR_VISIBLE_DEVICES="$GPU" "$HRR_PLAY" "$ARCHIVE" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "[run_hrr_replay] log=$LOG exit=$RC"

if [[ "$DO_ANALYZE" == "1" ]]; then
  FINDING="${LOG%.log}.finding.md"
  python3 "$ANALYZER" --log "$LOG" --archive "$ARCHIVE" \
    --hrr-playback "$HRR_PLAY" \
    --format markdown -o "$FINDING"
  echo "[run_hrr_replay] finding=$FINDING"
fi

exit "$RC"
