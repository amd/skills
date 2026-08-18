#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Resume the session recorded at launch. Re-run IR-2 (install.sh) and IR-1
# (preflight.py) first, exactly as for a fresh launch; this script does not
# re-check the gates.
#
# --resume-from is always passed explicitly: a bare --resume auto-picks the
# newest session and can target the wrong run. Resume writes its own log so the
# original run log is preserved.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./_env.sh
. "${SCRIPT_DIR}/_env.sh"

if [ -z "${SESSION_DIR:-}" ] && [ -f "$LAST_LAUNCH_ENV" ]; then
  . "$LAST_LAUNCH_ENV"
fi
if [ -z "${SESSION_DIR:-}" ]; then
  echo "ERROR: SESSION_DIR unknown -- read .session_dir from the launch-info JSON" >&2
  exit 1
fi
if [ ! -f "${SESSION_DIR}/state.json" ]; then
  echo "ERROR: ${SESSION_DIR} has no state.json; the CLI refuses to resume it" >&2
  exit 1
fi

RESUME_TAG="resume-$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${RUN_DIR}/run_${RESUME_TAG}.log"
PID_FILE="${RUN_DIR}/run_${RESUME_TAG}.pid"
LAUNCH_INFO_FILE="${RUN_DIR}/launch_${RESUME_TAG}.json"

# shellcheck disable=SC2086
setsid nohup "$PYTHON" -m hyperloom.inference_optimizer.cli --verbose optimize \
  --resume --resume-from "$SESSION_DIR" \
  --tick-interval-sec 30 \
  --launch-info-file "$LAUNCH_INFO_FILE" \
  ${OPT_FLAGS:-} \
  > "$RUN_LOG" 2>&1 < /dev/null &

echo $! > "$PID_FILE"

cat > "$LAST_LAUNCH_ENV" <<EOF
export RUN_TAG="${RESUME_TAG}"
export RUN_LOG="${RUN_LOG}"
export PID_FILE="${PID_FILE}"
export LAUNCH_INFO_FILE="${LAUNCH_INFO_FILE}"
export SESSION_DIR="${SESSION_DIR}"
EOF

echo "resume_tag=${RESUME_TAG}"
echo "run_log=${RUN_LOG}"
echo "session_dir=${SESSION_DIR}"
echo "next=run scripts/launch_health.sh to capture the real optimizer pid"
