#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
#
# Smoke test for launch.sh, launch_health.sh and resume.sh against a stub
# optimizer. No GPU, no Hyperloom wheel and no network are required.
#
# Usage: bash scripts/tests/test_launch_flow.sh

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
REAL_PYTHON="$(command -v python3)"
WORK="$(mktemp -d)"
STARTED_PIDS=()

cleanup() {
  for pid in ${STARTED_PIDS[@]+"${STARTED_PIDS[@]}"}; do
    kill "$pid" 2>/dev/null || true
  done
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
  echo "[FAIL] $1" >&2
  exit 1
}

# launch.sh and resume.sh background the optimizer with setsid nohup, so the stub
# writes its argv dump after they return. Poll for it rather than racing.
wait_for_argv() {
  local path="$1" deadline=$((SECONDS + 10))
  while [ ! -e "$path" ]; do
    [ "$SECONDS" -lt "$deadline" ] || fail "timed out waiting for $path"
    sleep 0.1
  done
}

# --- stub optimizer -------------------------------------------------------
# Only "-m hyperloom..." is simulated; every other invocation (the launch-info
# JSON reader inside launch_health.sh) goes to the real interpreter.
cat > "$WORK/stub_python" <<STUB
#!/usr/bin/env bash
set -euo pipefail

if [ "\${1:-}" != "-m" ]; then
  exec "$REAL_PYTHON" "\$@"
fi

info=""
args=("\$@")
for i in "\${!args[@]}"; do
  if [ "\${args[\$i]}" = "--launch-info-file" ]; then
    info="\${args[\$((i + 1))]}"
  fi
done

# Renamed into place so a poll on the path cannot observe a partial dump.
printf '%s\n' "\$@" > "\${STUB_ARGV_OUT}.tmp"
mv "\${STUB_ARGV_OUT}.tmp" "\${STUB_ARGV_OUT}"

mkdir -p "\${STUB_SESSION_DIR}"
echo '{}' > "\${STUB_SESSION_DIR}/manifest.json"
echo '{"phase": "PRELUDE"}' > "\${STUB_SESSION_DIR}/state.json"

# Report a forked child as the optimizer pid. The launcher's \$! is the wrapper,
# which is a different process -- that gap is what the health check must close.
sleep 30 &
child=\$!

if [ -n "\$info" ]; then
  printf '{"pid": %s, "session_dir": "%s"}\n' "\$child" "\${STUB_SESSION_DIR}" > "\$info"
fi

wait "\$child"
STUB
chmod +x "$WORK/stub_python"

# --- fixture --------------------------------------------------------------
# Each fixture is self-contained: .env and workload.env hold absolute paths, so a
# variant must be generated fresh rather than copied from another root.
make_fixture() {
  local root="$1"
  # hyperloom/ is what `pip install --target` leaves behind, and what marks this
  # root as an install directory rather than whatever the caller happened to cd to.
  mkdir -p "$root/model" "$root/data/runtime" "$root/data/optimizer_runs" \
           "$root/hyperloom/inference_optimizer"

  # A value with a space and a colon, double-quoted the way hyperloom-setup
  # writes it. Unquoted, the .env source in _env.sh would fail with exit 127.
  cat > "$root/.env" <<EOF
USER_DATA_PATH=$root/data
ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: placeholder"
EOF

  echo '{}' > "$root/model/config.json"
  echo 'export KERNEL_AGENT_MARKER=1' > "$root/data/runtime/kernel-agent.env.sh"

  cat > "$root/data/optimizer_runs/workload.env" <<EOF
export MODEL_PATH=$root/model
export FRAMEWORK=vllm
export TP=1
export EP=1
export CONC=64
export ISL=1024
export OSL=1024
export PRECISION=fp8
export MAX_HOURS=3
export TARGET_GAIN=20
export OPT_FLAGS="--no-kernel"
EOF
}

ROOT="$WORK/repo"
make_fixture "$ROOT"

export INSTALL_DIR="$ROOT"
export PYTHON="$WORK/stub_python"
export STUB_SESSION_DIR="$ROOT/data/model/20260101_000000"
export STUB_ARGV_OUT="$WORK/argv_launch.txt"
export LAUNCH_HEALTH_DELAY_SEC=1
cd "$ROOT"

# --- launch ---------------------------------------------------------------
bash "$SCRIPTS_DIR/launch.sh" > "$WORK/launch.out"

LAST_LAUNCH="$ROOT/data/optimizer_runs/last_launch.env"
[ -f "$LAST_LAUNCH" ] || fail "launch.sh did not write last_launch.env"
# shellcheck disable=SC1090
. "$LAST_LAUNCH"
WRAPPER_PID="$(cat "$PID_FILE")"
STARTED_PIDS+=("$WRAPPER_PID")

grep -q "^export LAUNCH_INFO_FILE=" "$LAST_LAUNCH" \
  || fail "last_launch.env is missing LAUNCH_INFO_FILE"
echo "[ok] launch.sh recorded the run handles on disk"

wait_for_argv "$STUB_ARGV_OUT"
for expected in --model --framework --tp --ep --conc --isl --osl --precision \
  --max-hours --target-gain; do
  grep -qx -- "$expected" "$STUB_ARGV_OUT" || fail "$expected not passed to the CLI"
done
grep -qx -- "--no-kernel" "$STUB_ARGV_OUT" \
  || fail "OPT_FLAGS did not word-split into a separate argument"
grep -qx -- "$ROOT/model" "$STUB_ARGV_OUT" || fail "MODEL_PATH not passed"
echo "[ok] every workload.env value reached the CLI, OPT_FLAGS word-split"

# --- health check ---------------------------------------------------------
bash "$SCRIPTS_DIR/launch_health.sh" > "$WORK/health.out"
grep -q "optimizer_alive=true" "$WORK/health.out" || fail "health check reported no live optimizer"

REPORTED_PID="$(cat "$PID_FILE")"
STARTED_PIDS+=("$REPORTED_PID")
JSON_PID="$("$REAL_PYTHON" -c 'import json,sys;print(json.load(open(sys.argv[1]))["pid"])' "$LAUNCH_INFO_FILE")"
[ "$WRAPPER_PID" != "$JSON_PID" ] \
  || fail "fixture is not exercising the gap: wrapper pid equals the optimizer pid"
[ "$REPORTED_PID" = "$JSON_PID" ] \
  || fail "pid file holds $REPORTED_PID, expected the optimizer pid $JSON_PID"
echo "[ok] pid file holds the optimizer pid, not the setsid wrapper"

grep -q "^export SESSION_DIR=" "$LAST_LAUNCH" || fail "SESSION_DIR was not persisted"
echo "[ok] session_dir persisted for the monitor and resume steps"

# --- health check failure path -------------------------------------------
# A run whose launch-info JSON never appeared: the handles exist but the file
# does not, which is what a crash during startup looks like.
BROKEN="$WORK/broken"
make_fixture "$BROKEN"
cat > "$BROKEN/data/optimizer_runs/last_launch.env" <<EOF
export RUN_TAG="broken"
export RUN_LOG="$BROKEN/data/optimizer_runs/run_broken.log"
export PID_FILE="$BROKEN/data/optimizer_runs/run_broken.pid"
export LAUNCH_INFO_FILE="$BROKEN/data/optimizer_runs/launch_broken.json"
EOF
if (cd "$BROKEN" && INSTALL_DIR="$BROKEN" bash "$SCRIPTS_DIR/launch_health.sh") \
  > "$WORK/health_fail.out" 2>&1; then
  fail "health check passed even though the launch-info JSON was missing"
fi
grep -q "launch_broken.json not written" "$WORK/health_fail.out" \
  || fail "health check did not name the missing launch-info JSON"
echo "[ok] health check blocks when the launch-info JSON never appeared"

# --- resume ---------------------------------------------------------------
export STUB_ARGV_OUT="$WORK/argv_resume.txt"
bash "$SCRIPTS_DIR/resume.sh" > "$WORK/resume.out"
# shellcheck disable=SC1090
. "$LAST_LAUNCH"
STARTED_PIDS+=("$(cat "$PID_FILE")")

wait_for_argv "$WORK/argv_resume.txt"
grep -qx -- "--resume-from" "$WORK/argv_resume.txt" \
  || fail "resume.sh did not pass --resume-from explicitly"
if grep -qx -- "--model" "$WORK/argv_resume.txt"; then
  fail "resume.sh passed --model, which resume must omit"
fi
case "$RUN_LOG" in
  *run_resume-*) ;;
  *) fail "resume.sh reused the original run log: $RUN_LOG" ;;
esac
echo "[ok] resume targets the recorded session and writes its own log"

# --- missing gate inputs --------------------------------------------------
NOPLAN="$WORK/noplan"
make_fixture "$NOPLAN"
rm -f "$NOPLAN/data/optimizer_runs/workload.env"
if (cd "$NOPLAN" && INSTALL_DIR="$NOPLAN" bash "$SCRIPTS_DIR/launch.sh") \
  > "$WORK/noplan.out" 2>&1; then
  fail "launch.sh started without workload.env"
fi
grep -q "workload.env missing" "$WORK/noplan.out" \
  || fail "launch.sh did not name the missing workload.env"
echo "[ok] launch.sh refuses to start without the confirmed plan"

# A directory that is not the install directory -- what the caller gets when the
# shell is left somewhere else and INSTALL_DIR falls back to the current one.
WRONGDIR="$WORK/wrongdir"
make_fixture "$WRONGDIR"
rm -rf "$WRONGDIR/hyperloom"
# INSTALL_DIR is unset so the script falls back to the current directory, which
# is the case the check exists for; the earlier export would otherwise mask it.
if (cd "$WRONGDIR" && env -u INSTALL_DIR bash "$SCRIPTS_DIR/launch.sh") \
  > "$WORK/wrongdir.out" 2>&1; then
  fail "launch.sh started from a directory holding no hyperloom/"
fi
grep -q "holds no hyperloom/" "$WORK/wrongdir.out" \
  || fail "launch.sh did not name the directory that is missing hyperloom/"
echo "[ok] launch.sh refuses to start outside the install directory"

NOINSTALL="$WORK/noinstall"
make_fixture "$NOINSTALL"
rm -f "$NOINSTALL/data/runtime/kernel-agent.env.sh"
if (cd "$NOINSTALL" && INSTALL_DIR="$NOINSTALL" bash "$SCRIPTS_DIR/launch.sh") \
  > "$WORK/noinstall.out" 2>&1; then
  fail "launch.sh started without the kernel-agent env from install.sh"
fi
grep -q "run IR-2" "$WORK/noinstall.out" \
  || fail "launch.sh did not point at IR-2 when the kernel-agent env was missing"
echo "[ok] launch.sh refuses to start before IR-2 has run"

echo
echo "all launch-flow checks passed"
