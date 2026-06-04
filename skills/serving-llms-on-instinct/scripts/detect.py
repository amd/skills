#!/usr/bin/env python3
"""
Detect AMD GPU hardware via amd-smi.

Usage:
    python scripts/detect.py
    python scripts/detect.py --host root@10.0.0.5

Output: JSON with gpu_count, gfx_version (first GPU), rocm_version, full GPU list.
Exits 0 on success, 1 on failure.

Env vars (used when --host is not given):
    ROCM_SSH_HOST  -- remote host
    ROCM_SSH_USER  -- SSH user (default: root)
    ROCM_SSH_PORT  -- SSH port (default: 22)
"""

import argparse
import json
import os
import subprocess
import sys


def _is_local(host):
    return not host or host in ("local", "localhost", "127.0.0.1")


def _run(cmd, host, user, port, timeout=30):
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
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        target = f"{user}@{host}" if user else host
        return 1, "", f"Command timed out after {timeout}s on {target}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="", help="[user@]host (default: local or ROCM_SSH_HOST)")
    parser.add_argument("--user", default="", help="SSH user (default: root)")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()

    host = args.host
    user = args.user
    if "@" in host:
        user, host = host.split("@", 1)

    host = host or os.environ.get("ROCM_SSH_HOST", "")
    user = user or os.environ.get("ROCM_SSH_USER", "")
    port = args.port or int(os.environ.get("ROCM_SSH_PORT", "22"))

    rc, out, err = _run("amd-smi static --asic --vram --json", host, user, port)
    if rc != 0 and "required groups" in err:
        # User not in video/render group -- retry with sudo
        rc, out, err = _run("sudo amd-smi static --asic --vram --json", host, user, port)
    if rc != 0:
        print(json.dumps({
            "error": "amd-smi failed",
            "detail": err.strip() or f"exit code {rc}",
            "hint": "Is amd-smi installed? Is amdgpu kernel module loaded? Try: lsmod | grep amdgpu",
        }))
        sys.exit(1)

    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"amd-smi JSON parse failed: {e}", "raw": out[:200]}))
        sys.exit(1)

    gpu_list = data.get("gpu_data", data if isinstance(data, list) else [data])
    gpus = []
    for entry in gpu_list:
        asic = entry.get("asic", {})
        vram = entry.get("vram", {})
        vram_size = vram.get("size", {})
        vram_mb = vram_size.get("value") if isinstance(vram_size, dict) else vram_size
        gpus.append({
            "index": entry.get("gpu", len(gpus)),
            "market_name": asic.get("market_name", "Unknown"),
            "gfx_version": asic.get("target_graphics_version", "unknown").lower(),
            "vram_gb": round(vram_mb / 1024, 1) if vram_mb else None,
            "vram_type": vram.get("type"),
            "compute_units": asic.get("num_compute_units"),
        })

    rocm_version = "unknown"
    rc2, out2, err2 = _run("amd-smi version --json", host, user, port, timeout=10)
    if rc2 != 0 and "required groups" in err2:
        rc2, out2, _ = _run("sudo amd-smi version --json", host, user, port, timeout=10)
    if rc2 == 0:
        try:
            vdata = json.loads(out2)
            if isinstance(vdata, list) and vdata:
                rocm_version = vdata[0].get("rocm_version", "unknown")
            elif isinstance(vdata, dict):
                rocm_version = vdata.get("rocm_version", "unknown")
        except json.JSONDecodeError:
            pass

    print(json.dumps({
        "gpu_count": len(gpus),
        "gfx_version": gpus[0]["gfx_version"] if gpus else "unknown",
        "rocm_version": rocm_version,
        "target": "local" if _is_local(host) else host,
        "gpus": gpus,
    }, indent=2))


if __name__ == "__main__":
    main()
