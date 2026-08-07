#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Start a fresh optimize run in the background. Run IR-2 (install.sh) and IR-1
# (preflight.py) first; this script does not re-check the gates.
#
# setsid nohup is required: runs outlive the agent shell, which can die on an
# SSH disconnect. Every workload value comes from workload.env.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./_env.sh
. "${SCRIPT_DIR}/_env.sh"

RUN_TAG="$(basename "$MODEL_PATH")-$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${RUN_DIR}/run_${RUN_TAG}.log"
PID_FILE="${RUN_DIR}/run_${RUN_TAG}.pid"
LAUNCH_INFO_FILE="${RUN_DIR}/launch_${RUN_TAG}.json"

# OPT_FLAGS holds the optional Phase 2 flags and is left unquoted on purpose so
# it word-splits into separate arguments.
# shellcheck disable=SC2086
setsid nohup "$PYTHON" -m hyperloom.inference_optimizer.cli --verbose optimize \
  --model "$MODEL_PATH" \
  --framework "$FRAMEWORK" \
  --tp "$TP" \
  --ep "$EP" \
  --conc "$CONC" \
  --isl "$ISL" \
  --osl "$OSL" \
  --precision "$PRECISION" \
  --max-hours "$MAX_HOURS" \
  --target-gain "$TARGET_GAIN" \
  --tick-interval-sec 30 \
  --launch-info-file "$LAUNCH_INFO_FILE" \
  ${OPT_FLAGS:-} \
  > "$RUN_LOG" 2>&1 < /dev/null &

# This is the setsid wrapper pid, which exits immediately; launch_health.sh
# replaces it with the real optimizer pid from the launch-info JSON.
echo $! > "$PID_FILE"

cat > "$LAST_LAUNCH_ENV" <<EOF
export RUN_TAG="${RUN_TAG}"
export RUN_LOG="${RUN_LOG}"
export PID_FILE="${PID_FILE}"
export LAUNCH_INFO_FILE="${LAUNCH_INFO_FILE}"
EOF

echo "run_tag=${RUN_TAG}"
echo "run_log=${RUN_LOG}"
echo "launch_info_file=${LAUNCH_INFO_FILE}"
echo "next=run scripts/launch_health.sh to capture the real optimizer pid"
