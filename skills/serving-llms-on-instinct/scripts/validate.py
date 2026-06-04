#!/usr/bin/env python3
"""
Validate the environment on an AMD GPU machine before launching vLLM.

Usage:
    python scripts/validate.py
    python scripts/validate.py --host root@10.0.0.5
    python scripts/validate.py --auto-fix        # apply safe fixes (NUMA, hipBLASLt)

Checks: /dev/kfd, /dev/dri, Docker, NUMA balancing, hipBLASLt, HF_TOKEN.
Each issue is classified as: error (blocks launch), warning (degrades perf), advisory (info).

Exits 0 if no error-severity issues remain, 1 otherwise.

Env vars:
    ROCM_SSH_HOST, ROCM_SSH_USER, ROCM_SSH_PORT
"""

import argparse
import json
import os
import subprocess
import sys


def _is_local(host):
    return not host or host in ("local", "localhost", "127.0.0.1")


def _run(cmd, host, user, port, timeout=20):
    try:
        if _is_local(host):
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        else:
            ssh_target = f"{user}@{host}" if user else host
            ssh = [
                "ssh",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=15",
                "-o", "BatchMode=yes",
                "-o", "LogLevel=ERROR",
                "-p", str(port),
                ssh_target, cmd,
            ]
            r = subprocess.run(ssh, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        target = f"{user}@{host}" if user else host
        return 1, "", f"Command timed out after {timeout}s on {target}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="", help="[user@]host (default: local or ROCM_SSH_HOST)")
    parser.add_argument("--user", default="", help="SSH user (default: root)")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--auto-fix", action="store_true", help="Apply safe fixes without prompting")
    args = parser.parse_args()

    host = args.host
    user = args.user
    if "@" in host:
        user, host = host.split("@", 1)

    host = host or os.environ.get("ROCM_SSH_HOST", "")
    user = user or os.environ.get("ROCM_SSH_USER", "")
    port = args.port or int(os.environ.get("ROCM_SSH_PORT", "22"))

    issues = []
    fixes_applied = []

    # /dev/kfd
    rc, out, _ = _run("test -e /dev/kfd && echo exists || echo missing", host, user, port)
    if "missing" in out:
        issues.append({
            "check": "dev_kfd",
            "severity": "error",
            "message": "/dev/kfd not found. The amdgpu kernel module is not loaded or the driver is not installed.",
            "fix": "sudo modprobe amdgpu  # or install ROCm driver",
        })
    else:
        rc2, out2, _ = _run("test -r /dev/kfd && echo ok || echo denied", host, user, port)
        if "denied" in out2:
            # Docker passes --device /dev/kfd directly, so host user permissions
            # don't block containerized workloads. Downgrade to warning.
            issues.append({
                "check": "dev_kfd",
                "severity": "warning",
                "message": "/dev/kfd exists but current user is not in video/render group. Docker containers will still work.",
                "fix": "sudo usermod -aG video,render $USER  # then re-login (only needed for non-Docker use)",
            })

    # /dev/dri
    rc, out, _ = _run("ls /dev/dri/renderD* 2>/dev/null | wc -l", host, user, port)
    try:
        render_count = int(out)
    except ValueError:
        render_count = 0
    if render_count == 0:
        issues.append({
            "check": "dev_dri",
            "severity": "error",
            "message": "No /dev/dri/renderD* nodes found. GPU render nodes not present.",
            "fix": "Check that the amdgpu driver is loaded: lsmod | grep amdgpu",
        })

    # Docker
    rc, out, err = _run("docker ps -q 2>&1 | head -1", host, user, port)
    if rc != 0 or "permission denied" in err.lower() or "cannot connect" in err.lower():
        issues.append({
            "check": "docker",
            "severity": "error",
            "message": f"Docker not accessible: {err or 'docker ps failed'}",
            "fix": "Start Docker: sudo systemctl start docker  |  Or add user to docker group: sudo usermod -aG docker $USER",
        })

    # NUMA balancing
    rc, out, _ = _run("cat /proc/sys/kernel/numa_balancing 2>/dev/null || echo 0", host, user, port)
    numa_val = out.strip()
    if numa_val == "1":
        if args.auto_fix:
            rc2, _, _ = _run("echo 0 | sudo tee /proc/sys/kernel/numa_balancing > /dev/null", host, user, port)
            if rc2 == 0:
                fixes_applied.append("NUMA balancing disabled (non-persistent, resets on reboot)")
            else:
                issues.append({
                    "check": "numa_balancing",
                    "severity": "warning",
                    "message": "NUMA balancing is enabled. Causes latency spikes during GPU inference.",
                    "fix": "echo 0 | sudo tee /proc/sys/kernel/numa_balancing",
                })
        else:
            issues.append({
                "check": "numa_balancing",
                "severity": "warning",
                "message": "NUMA balancing is enabled. Causes latency spikes during GPU inference.",
                "fix": "echo 0 | sudo tee /proc/sys/kernel/numa_balancing  (or run with --auto-fix)",
            })

    # hipBLASLt
    rc, out, _ = _run("ls /opt/rocm/lib/libhipblaslt* 2>/dev/null | head -1", host, user, port)
    if not out.strip():
        issues.append({
            "check": "hipblaslt",
            "severity": "warning",
            "message": "hipBLASLt not found at /opt/rocm/lib/. GEMM performance may be reduced.",
            "fix": "Ensure ROCm is fully installed: sudo apt install hipblaslt  or reinstall ROCm",
        })

    # HF_TOKEN
    rc, out, _ = _run("printenv HF_TOKEN | head -c 4", host, user, port)
    if not out.strip():
        issues.append({
            "check": "hf_token",
            "severity": "advisory",
            "message": "HF_TOKEN not set. Required for gated models (Llama, Gemma). Not needed for Qwen3.",
            "fix": "export HF_TOKEN=hf_...",
        })

    # vLLM Docker image
    rc, out, _ = _run("docker images vllm/vllm-openai-rocm --format '{{.Tag}}' 2>/dev/null | head -1", host, user, port)
    if not out.strip():
        issues.append({
            "check": "vllm_image",
            "severity": "advisory",
            "message": "vllm/vllm-openai-rocm image not pulled yet. First launch will download ~20GB.",
            "fix": "docker pull vllm/vllm-openai-rocm:latest",
        })

    # CUDA_VISIBLE_DEVICES footgun
    rc, out, _ = _run("printenv CUDA_VISIBLE_DEVICES 2>/dev/null || echo ''", host, user, port)
    if out.strip():
        issues.append({
            "check": "cuda_visible_devices",
            "severity": "error",
            "message": f"CUDA_VISIBLE_DEVICES is set to '{out.strip()}'. This hides AMD GPUs from the ROCm runtime.",
            "fix": "unset CUDA_VISIBLE_DEVICES",
        })

    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]
    advisories = [i for i in issues if i["severity"] == "advisory"]

    result = {
        "ready": len(errors) == 0,
        "target": "local" if _is_local(host) else host,
        "errors": errors,
        "warnings": warnings,
        "advisories": advisories,
        "fixes_applied": fixes_applied,
    }
    print(json.dumps(result, indent=2))
    sys.exit(0 if len(errors) == 0 else 1)


if __name__ == "__main__":
    main()
