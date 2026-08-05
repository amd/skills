#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Verify a run actually started and record its real handles. The pid stored by
# launch.sh is the setsid wrapper, which exits immediately, so the optimizer pid
# and session_dir are read from the launch-info JSON. session_dir is never
# guessed from a timestamp: concurrent sessions share USER_DATA_PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./_env.sh
. "${SCRIPT_DIR}/_env.sh"

if [ ! -f "$LAST_LAUNCH_ENV" ]; then
  echo "ERROR: $LAST_LAUNCH_ENV missing -- launch.sh has not run" >&2
  exit 1
fi
. "$LAST_LAUNCH_ENV"

sleep "${LAUNCH_HEALTH_DELAY_SEC:-30}"

if [ ! -f "$LAUNCH_INFO_FILE" ]; then
  echo "ERROR: $LAUNCH_INFO_FILE not written; inspect $RUN_LOG" >&2
  exit 1
fi

read_json() {
  "$PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],""))' \
    "$1" "$2" 2>/dev/null || true
}

REAL_PID="$(read_json "$LAUNCH_INFO_FILE" pid)"
if [ -z "$REAL_PID" ]; then
  REAL_PID="$(pgrep -f 'hyperloom.inference_optimizer.cli .*optimize' | head -1 || true)"
fi
if [ -z "$REAL_PID" ]; then
  echo "ERROR: optimizer pid not found; inspect $RUN_LOG" >&2
  exit 1
fi
echo "$REAL_PID" > "$PID_FILE"

if [ -d "/proc/${REAL_PID}" ]; then
  echo "optimizer_alive=true pid=${REAL_PID}"
else
  echo "ERROR: pid ${REAL_PID} is already gone; inspect $RUN_LOG" >&2
  exit 1
fi

SESSION_DIR="$(read_json "$LAUNCH_INFO_FILE" session_dir)"
if [ -z "$SESSION_DIR" ]; then
  echo "ERROR: no session_dir yet in $LAUNCH_INFO_FILE; inspect $RUN_LOG" >&2
  exit 1
fi

cat > "$LAST_LAUNCH_ENV" <<EOF
export RUN_TAG="${RUN_TAG}"
export RUN_LOG="${RUN_LOG}"
export PID_FILE="${PID_FILE}"
export LAUNCH_INFO_FILE="${LAUNCH_INFO_FILE}"
export SESSION_DIR="${SESSION_DIR}"
EOF

echo "session_dir=${SESSION_DIR}"
[ -f "${SESSION_DIR}/manifest.json" ] && echo "manifest_present=true"
[ -f "${SESSION_DIR}/state.json" ] && echo "state_exists=true"
exit 0
