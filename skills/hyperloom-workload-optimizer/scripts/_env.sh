# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

# Shared Phase 3 preamble, sourced by launch.sh, launch_health.sh and resume.sh.
# Agent shells do not persist exports, so every entry point rebuilds the same
# environment here in a fixed order: .env, then workload.env, then the
# kernel-agent env written by install.sh.
#
# Sourced, not executed: a failure here exits the calling script.

: "${INSTALL_DIR:="$(pwd -P)"}"
cd "$INSTALL_DIR" || exit 1

# Defaulting to the current directory is only right when the caller is already
# in the install directory, so say which directory was wrong rather than failing
# later on a path built from it.
if [ ! -d "$INSTALL_DIR/hyperloom" ] && [ ! -d "$INSTALL_DIR/src/hyperloom" ]; then
  echo "ERROR: $INSTALL_DIR holds no hyperloom/ or src/hyperloom/ -- set INSTALL_DIR to the directory the wheel was installed into" >&2
  exit 1
fi

if [ -f "$INSTALL_DIR/.env" ]; then
  # Values containing spaces must be double-quoted in .env, otherwise this
  # source fails with exit 127. hyperloom-setup writes them quoted.
  set -a
  . "$INSTALL_DIR/.env"
  set +a
fi

: "${USER_DATA_PATH:?USER_DATA_PATH missing -- run the Hyperloom setup skill first}"
export INSTALL_DIR USER_DATA_PATH
export RUN_DIR="${USER_DATA_PATH}/optimizer_runs"
mkdir -p "$RUN_DIR"

WORKLOAD_ENV="${RUN_DIR}/workload.env"
if [ ! -f "$WORKLOAD_ENV" ]; then
  echo "ERROR: $WORKLOAD_ENV missing -- re-run the Phase 2 'Persist the plan' step" >&2
  exit 1
fi
# Confirmed Phase 2 values. Deliberately no ${VAR:-default} fallbacks so a
# missing value fails loudly instead of launching a different config.
. "$WORKLOAD_ENV"
: "${MODEL_PATH:?MODEL_PATH empty -- re-run the Phase 2 'Persist the plan' step}"

KERNEL_AGENT_ENV="${KERNEL_AGENT_ENV:-${USER_DATA_PATH}/runtime/kernel-agent.env.sh}"
if [ ! -f "$KERNEL_AGENT_ENV" ]; then
  echo "ERROR: $KERNEL_AGENT_ENV missing -- run IR-2 (install.sh) first" >&2
  exit 1
fi
export KERNEL_AGENT_ENV
. "$KERNEL_AGENT_ENV"

export PYTHON="${PYTHON:-$(command -v python3)}"
export PATH="$(dirname "$PYTHON"):/usr/local/bin:$PATH"
export PYTHONPATH="${INSTALL_DIR}:${PYTHONPATH:-}"

# Run handles are recorded here because the health check, monitor and resume
# steps each run in a fresh shell.
LAST_LAUNCH_ENV="${RUN_DIR}/last_launch.env"
