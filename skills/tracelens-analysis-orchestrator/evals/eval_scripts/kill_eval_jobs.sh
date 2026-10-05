#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -uo pipefail

# ---------------------------------------------------------------------------
# Kill every process the eval pipeline may have spawned.
# Covers jobs started by generate_ref.sh and run_repeatability_parallel.sh.
#
# Usage:
#   bash kill_eval_jobs.sh          # SIGTERM, then SIGKILL, rescanning until clear
#   bash kill_eval_jobs.sh --list   # show matching processes, kill nothing
#   bash kill_eval_jobs.sh -9       # SIGKILL from the first pass (no grace period)
# ---------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: bash kill_eval_jobs.sh [--list|-9|-h]

  (no args)   Terminate matching jobs: SIGTERM, then SIGKILL, rescanning until
              no marked processes remain (bounded number of passes).
  --list      List matching processes without killing anything.
  -9          SIGKILL from the first pass (skip the graceful SIGTERM).
  -h,--help   Show this help.

Kills processes spawned by generate_ref.sh and run_repeatability_parallel.sh:
the scripts themselves, their subshells, agent/pi analysis + eval calls (and the
node/python workers underneath), and the scripted eval + merge python. Scoped to
the current user; unrelated agent/claude/python sessions are not touched.
EOF
}

MODE="kill"   # kill | list | force
case "${1:-}" in
    ""              ) MODE="kill"  ;;
    --list          ) MODE="list"  ;;
    -9              ) MODE="force" ;;
    -h|--help       ) usage; exit 0 ;;
    *) echo "ERROR: Unknown argument '$1'." >&2; usage >&2; exit 1 ;;
esac

MAX_PASSES=6

# Print the PIDs of this user's processes carrying the TRACELENS_EVAL_JOB=1
# marker, one per line, unique + numerically sorted. environ entries are
# NUL-separated, so grep -z matches across them. Guard on ownership (-O): only
# this user's /proc entries are readable, but be explicit.
scan_marked() {
    local envf pid
    for envf in /proc/[0-9]*/environ; do
        pid="${envf#/proc/}"; pid="${pid%/environ}"
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        [[ -O "$envf" ]] || continue
        if grep -qzs 'TRACELENS_EVAL_JOB=1' "$envf" 2>/dev/null; then
            printf '%s\n' "$pid"
        fi
    done | sort -un
}

# --- list mode: show matches and exit, killing nothing ----------------------
if [[ "$MODE" == "list" ]]; then
    mapfile -t UNIQ < <(scan_marked)
    if [[ ${#UNIQ[@]} -eq 0 ]]; then
        echo "No eval jobs found."
        exit 0
    fi
    echo "Matching eval jobs (${#UNIQ[@]}):"
    for pid in "${UNIQ[@]}"; do
        printf '  %s  %s\n' "$pid" "$(ps -o args= -p "$pid" 2>/dev/null | cut -c1-100)"
    done
    exit 0
fi

# --- kill mode: scan + signal, rescanning until clear or MAX_PASSES ----------
killed_any=false
for ((pass = 1; pass <= MAX_PASSES; pass++)); do
    mapfile -t UNIQ < <(scan_marked)
    if [[ ${#UNIQ[@]} -eq 0 ]]; then
        if [[ "$killed_any" == true ]]; then
            echo "All eval jobs cleared."
        else
            echo "No eval jobs found."
        fi
        exit 0
    fi

    # SIGTERM on the first pass (unless -9) to allow graceful shutdown; SIGKILL
    # on every pass thereafter (and always in force mode).
    if [[ "$MODE" == "force" || "$pass" -gt 1 ]]; then
        echo "Pass $pass: SIGKILL ${#UNIQ[@]} job(s): ${UNIQ[*]}"
        kill -9 "${UNIQ[@]}" 2>/dev/null
    else
        echo "Pass $pass: SIGTERM ${#UNIQ[@]} job(s): ${UNIQ[*]}"
        kill "${UNIQ[@]}" 2>/dev/null
    fi
    killed_any=true
    sleep 2
done

# Final check after the last pass.
mapfile -t UNIQ < <(scan_marked)
if [[ ${#UNIQ[@]} -gt 0 ]]; then
    echo "WARNING: ${#UNIQ[@]} job(s) still alive after $MAX_PASSES passes: ${UNIQ[*]}" >&2
    echo "  (possibly stuck in uninterruptible I/O -- inspect with: ps -o pid,stat,args -p ${UNIQ[*]// /,})" >&2
    exit 1
fi
echo "All eval jobs cleared."
