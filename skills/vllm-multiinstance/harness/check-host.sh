#!/bin/bash
# ============================================================================
# Host / environment preflight for the multi-instance vLLM stack.
#
# Catches the host-level blockers that otherwise surface only as a cryptic deep
# failure or a 20-minute health-wait hang (none of these are caught by
# detect.py or the run_sweep preflight):
#
#   3. image short-name that won't resolve on a host without unqualified-search
#      registries (e.g. "amdih/..." instead of "docker.io/amdih/...").
#   4. rootless cgroup v2 cpuset NOT delegated to the user slice -> core pinning
#      dies with "cpuset.cpus: no such file or directory".
#   5. CNI backend (podman 3.x) whose plugins are older than the cniVersion
#      podman writes -> containers silently drop their static IPs -> every
#      load-balanced request 504s and the health wait never passes.
#
# Exit 0 if it's safe to proceed, 1 on a hard blocker. CNI is a WARN only
# (start.sh auto-downgrades the conflist and the static-IP guard catches any
# residual failure fast).
#
# Reads (all optional):
#   VLLM_IMAGE   image to validate; skipped if unset.
#   LIMITS_ON    1 (default) if cpuset pinning is in use; 0 disables the cpuset
#                check (matches start.sh --no-limits).
#
# Escape hatches (set =1 to downgrade a hard failure to a warning):
#   SKIP_HOST_CHECK    skip this script entirely (callers honor it).
#   ALLOW_SHORT_NAME   allow an unresolved short-name image.
#   ALLOW_NO_CPUSET    proceed without cpuset delegation.
# ============================================================================
set -uo pipefail

if [[ "${SKIP_HOST_CHECK:-0}" == "1" ]]; then
    echo "  [skip] host check (SKIP_HOST_CHECK=1)"
    exit 0
fi

if ! command -v podman >/dev/null 2>&1; then
    echo "  [BLOCK] podman not found in PATH." >&2
    exit 1
fi

# ----------------------------------------------------------------------------
# 3. Image resolvability (short-name).
# ----------------------------------------------------------------------------
check_image_resolvable() {
    local img="${VLLM_IMAGE:-}"
    [[ -z "$img" ]] && { echo "  [skip] image check (VLLM_IMAGE unset)"; return 0; }

    if podman image exists "$img" 2>/dev/null; then
        echo "  [OK]   image present locally: $img"
        return 0
    fi

    # Fully qualified = first path component looks like a registry host
    # (contains a '.' or ':port', or is literally "localhost").
    local first="${img%%/*}"
    if [[ "$img" == */* && ( "$first" == *.* || "$first" == *:* || "$first" == "localhost" ) ]]; then
        echo "  [WARN] image $img not present locally; podman will try to pull it."
        return 0
    fi

    if [[ "${ALLOW_SHORT_NAME:-0}" == "1" ]]; then
        echo "  [WARN] short-name image '$img' not present (ALLOW_SHORT_NAME=1; continuing)."
        return 0
    fi

    echo "  [BLOCK] image short-name '$img' is not present locally and is not fully qualified." >&2
    echo "          Hosts without unqualified-search registries can't resolve bare names," >&2
    echo "          so podman-compose would fail deep into the run." >&2
    echo "    Fix:    use a fully-qualified name, e.g.  VLLM_IMAGE=docker.io/$img" >&2
    echo "    Or:     pre-pull it once,            e.g.  podman pull docker.io/$img" >&2
    echo "    Bypass: ALLOW_SHORT_NAME=1" >&2
    return 1
}

# ----------------------------------------------------------------------------
# 4. Rootless cgroup v2 cpuset delegation.
# ----------------------------------------------------------------------------
check_cpuset_delegation() {
    if [[ "${LIMITS_ON:-1}" != "1" ]]; then
        echo "  [skip] cpuset delegation check (limits disabled)"
        return 0
    fi
    if [[ "$(id -u)" -eq 0 ]]; then
        echo "  [OK]   running rootful; cpuset always available"
        return 0
    fi
    if [[ ! -f /sys/fs/cgroup/cgroup.controllers ]]; then
        echo "  [skip] not cgroup v2 unified; cpuset delegation N/A"
        return 0
    fi

    local uid; uid="$(id -u)"
    local candidates=(
        "/sys/fs/cgroup/user.slice/user-${uid}.slice/user@${uid}.service/cgroup.controllers"
    )
    local cg
    cg="$(awk -F: '/^0::/{print $3}' /proc/self/cgroup 2>/dev/null)"
    [[ -n "$cg" ]] && candidates+=("/sys/fs/cgroup${cg}/cgroup.controllers")

    local found_file=false f
    for f in "${candidates[@]}"; do
        [[ -r "$f" ]] || continue
        found_file=true
        if grep -qw cpuset "$f"; then
            echo "  [OK]   rootless cpuset delegated ($f)"
            return 0
        fi
    done

    if ! $found_file; then
        echo "  [WARN] could not read cgroup controllers; cannot verify cpuset delegation."
        echo "         If pinning fails with 'cpuset.cpus: no such file or directory', see below."
        return 0
    fi

    if [[ "${ALLOW_NO_CPUSET:-0}" == "1" ]]; then
        echo "  [WARN] cpuset NOT delegated to rootless user slice (ALLOW_NO_CPUSET=1; continuing)."
        return 0
    fi

    echo "  [BLOCK] rootless cgroup v2 cpuset is NOT delegated to your user slice." >&2
    echo "          Core pinning will fail with 'cpuset.cpus: no such file or directory'." >&2
    echo "    Fix (needs root, one-time -- no session restart required):" >&2
    echo "      sudo mkdir -p /etc/systemd/system/user@.service.d" >&2
    echo "      printf '[Service]\\nDelegate=cpu cpuset io memory pids\\n' | \\" >&2
    echo "        sudo tee /etc/systemd/system/user@.service.d/delegate.conf" >&2
    echo "      sudo systemctl daemon-reload && systemctl --user daemon-reload" >&2
    echo "      sudo systemctl restart user@${uid}.service   # or re-login" >&2
    echo "    Alternative (no pinning): re-run with --no-limits." >&2
    echo "    Bypass: ALLOW_NO_CPUSET=1" >&2
    return 1
}

# ----------------------------------------------------------------------------
# 5. CNI cniVersion vs installed plugins (WARN only; start.sh auto-downgrades).
# ----------------------------------------------------------------------------
check_cni_version() {
    local backend
    backend="$(podman info --format '{{.Host.NetworkBackend}}' 2>/dev/null || echo "")"
    # .Host.NetworkBackend exists only on podman >= 4. On podman 3.x the field
    # is absent and the template renders empty -- so we must NOT treat "not
    # literally cni" as "no risk", or the podman-3.x CNI path (the one that
    # actually has the cniVersion problem) would be skipped entirely. Bail only
    # on an EXPLICIT netavark backend (genuinely no CNI risk). For "cni" or an
    # empty/unknown backend (i.e. podman 3.x), fall through; the plugin probe
    # below is a safe no-op when no CNI bridge plugin is installed.
    if [[ "$backend" == "netavark" ]]; then
        echo "  [OK]   network backend: netavark (no CNI cniVersion risk)"
        return 0
    fi

    echo "  [WARN] network backend is CNI / podman 3.x style (backend='${backend:-unknown}')."
    local plugin="" d
    for d in /opt/cni/bin /usr/lib/cni /usr/libexec/cni; do
        [[ -x "$d/bridge" ]] && { plugin="$d/bridge"; break; }
    done

    if [[ -n "$plugin" ]]; then
        local out
        out="$(printf '{"cniVersion":"1.0.0","name":"x","type":"bridge"}' \
            | CNI_COMMAND=VERSION "$plugin" 2>/dev/null || true)"
        if [[ -n "$out" ]] && ! echo "$out" | grep -q '"1.0.0"'; then
            echo "         $plugin does not advertise cniVersion 1.0.0 support."
            echo "         start.sh will downgrade the network conflist to 0.4.0 automatically."
            echo "         If static IPs still fail, upgrade 'containernetworking-plugins'."
        fi
    fi
    return 0
}

rc=0
check_image_resolvable   || rc=1
check_cpuset_delegation  || rc=1
check_cni_version        || true
exit "$rc"
