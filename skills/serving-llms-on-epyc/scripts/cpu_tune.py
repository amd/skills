#!/usr/bin/env python3
"""
Derive vLLM-on-CPU runtime knobs from the host, for a single instance pinned to
ONE socket (with its memory). Read-only.

Socket choice (dual-socket hosts): vLLM scales poorly across sockets, so we run on
one. We sample per-socket CPU load (~0.5s via /proc/stat) and prefer a free socket:
  - both sockets below --busy-threshold  -> socket 0 (deterministic; both free)
  - exactly one below the threshold       -> that socket
  - both at/above the threshold           -> WARN and proceed on the least-busy one
  - --socket N                            -> force a socket, skip the load check
A single-socket host just uses socket 0. (NPS2/NPS4 -> a socket spans multiple
NUMA nodes; we bind memory to all of the chosen socket's nodes.)

Emits two env vars:
  - VLLM_CPU_OMP_THREADS_BIND : physical cores of the chosen socket (SMT siblings
    dropped). vLLM sets OMP_NUM_THREADS itself (= len(cores)), so we don't.
  - VLLM_CPU_KVCACHE_SPACE    : KV-cache RAM (GB), sized from the chosen socket's
    LOCAL RAM (not whole-system) so the pool stays on-socket.

And a memory-bound pin for the chosen socket:
  - container : --cpuset-cpus=<phys cores> --cpuset-mems=<socket nodes>
  - conda     : numactl --cpunodebind=<nodes> --membind=<nodes>  (preferred)
                falls back to  taskset -c <phys cores>  (CPU-only, no mem bind)
                if neither exists, reported -- launch proceeds unpinned.

Not set: OMP_NUM_THREADS (vLLM derives it) and VLLM_CPU_NUM_OF_RESERVED_CPU
(vLLM has its own default when unset).

Usage:
    python3 scripts/cpu_tune.py                       # export lines for `eval`
    python3 scripts/cpu_tune.py --format json         # machine-readable
    python3 scripts/cpu_tune.py --socket 1            # force socket 1
    python3 scripts/cpu_tune.py --busy-threshold 70   # "free" means < 70% busy
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time

OS_HEADROOM_GB = 16


def _sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, timeout=15)
        return r.stdout
    except Exception:
        return ""


def _lscpu_int(out, label, default):
    m = re.search(rf"^{re.escape(label)}:\s*(\d+)", out, re.MULTILINE)
    return int(m.group(1)) if m else default


def _ranges(items):
    """Compress a sorted int list to a range string: [0,1,2,5] -> '0-2,5'."""
    items = sorted(items)
    if not items:
        return ""
    out, start, prev = [], items[0], items[0]
    for c in items[1:]:
        if c == prev + 1:
            prev = c
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = c
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(out)


def topology():
    """Per-socket layout from `lscpu -p`. Returns {sid: {phys, all, nodes}} where
    phys = one CPU per core (SMT dropped), all = every logical CPU, nodes = set of
    NUMA node ids on that socket. Also returns cpu->socket."""
    socks, cpu_socket = {}, {}
    for line in _sh("lscpu -p=CPU,CORE,SOCKET,NODE").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        cpu, core, sid = int(parts[0]), parts[1], int(parts[2])
        node = parts[3] if len(parts) > 3 and parts[3] != "" else str(sid)
        s = socks.setdefault(sid, {"phys": [], "all": [], "nodes": set(), "_cores": set()})
        s["all"].append(cpu)
        s["nodes"].add(int(node))
        cpu_socket[cpu] = sid
        if core not in s["_cores"]:
            s["_cores"].add(core)
            s["phys"].append(cpu)
    return socks, cpu_socket


def node_ram_gb(node):
    out = _sh(f"grep MemTotal /sys/devices/system/node/node{node}/meminfo")
    m = re.search(r"(\d+)", out)
    return (int(m.group(1)) // (1024 * 1024)) if m else 0


def socket_busy_pct(cpus, interval=0.5):
    """Mean CPU-busy% across `cpus` over `interval` seconds, from /proc/stat."""
    def snap():
        d = {}
        for ln in open("/proc/stat"):
            if ln.startswith("cpu") and len(ln) > 3 and ln[3].isdigit():
                p = ln.split()
                vals = list(map(int, p[1:]))
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
                d[int(p[0][3:])] = (idle, sum(vals))
        return d
    a = snap(); time.sleep(interval); b = snap()
    di = sum(b[c][0] - a[c][0] for c in cpus if c in a and c in b)
    dt = sum(b[c][1] - a[c][1] for c in cpus if c in a and c in b)
    return round(100 * (1 - di / dt), 1) if dt else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kv-frac", type=float, default=0.4, help="fraction of the chosen socket's RAM for KV cache")
    p.add_argument("--socket", type=int, default=None, help="force a socket id (skips the load check)")
    p.add_argument("--busy-threshold", type=float, default=15.0,
                   help="a socket is 'free' if its CPU-busy%% is below this (default 15)")
    p.add_argument("--format", choices=["env", "json"], default="env")
    args = p.parse_args()

    socks, _ = topology()
    if not socks:  # lscpu -p unavailable; degrade to a no-pin single instance
        print('export VLLM_CPU_KVCACHE_SPACE=4' if args.format == "env"
              else json.dumps({"error": "no topology from lscpu -p"}))
        return

    sids = sorted(socks)
    busy = {s: socket_busy_pct(socks[s]["all"]) for s in sids}

    warn = ""
    if args.socket is not None and args.socket in socks:
        chosen, reason = args.socket, "forced via --socket"
    elif len(sids) == 1:
        chosen, reason = sids[0], "single socket"
    else:
        free = [s for s in sids if busy[s] < args.busy_threshold]
        if len(free) >= 2:
            chosen, reason = sids[0], f"both sockets free (<{args.busy_threshold}% busy) -> socket 0"
        elif len(free) == 1:
            chosen, reason = free[0], f"only free socket (<{args.busy_threshold}% busy)"
        else:
            chosen = min(sids, key=lambda s: busy[s])
            reason = f"all sockets busy (>={args.busy_threshold}%) -> least-busy"
            warn = (f"all {len(sids)} sockets are busy (>= {args.busy_threshold}%): "
                    f"{ {s: busy[s] for s in sids} }. Proceeding on the least-busy socket "
                    f"{chosen}; performance may suffer. Pass --socket N to override.")

    sock = socks[chosen]
    bind = _ranges(sock["phys"])
    nodes = sorted(sock["nodes"])
    nodes_str = _ranges(nodes)

    sock_ram = sum(node_ram_gb(n) for n in nodes)
    if sock_ram <= 0:  # sysfs unavailable: fall back to total/sockets
        m = re.search(r"MemTotal:\s*(\d+)", _sh("grep MemTotal /proc/meminfo"))
        total = int(m.group(1)) // (1024 * 1024) if m else 0
        sock_ram = total // max(1, len(sids))
    if sock_ram <= 2 * OS_HEADROOM_GB:
        kv = max(1, int(sock_ram * 0.5))
    else:
        kv = max(1, min(int(sock_ram * args.kv_frac), sock_ram - OS_HEADROOM_GB))

    container_cpuset = f"--cpuset-cpus={bind} --cpuset-mems={nodes_str}"
    if shutil.which("numactl"):
        conda_prefix = f"numactl --cpunodebind={nodes_str} --membind={nodes_str}"
        conda_pin = "numactl (cpu + memory bound to the socket's nodes)"
    elif shutil.which("taskset"):
        conda_prefix = f"taskset -c {bind}"
        conda_pin = "taskset (CPU-only; memory NOT node-bound -- numactl not found)"
    else:
        conda_prefix = ""
        conda_pin = "none (no numactl/taskset; launching unpinned -- install numactl for memory binding)"

    nps_note = ""
    if len(nodes) > 1:
        nps_note = (f"socket {chosen} spans {len(nodes)} NUMA nodes (NPS{len(nodes)}); memory is "
                    f"bound across nodes {nodes_str}. Finer per-node binding could add performance.")

    result = {
        "chosen_socket": chosen,
        "socket_choice_reason": reason,
        "sockets": len(sids),
        "socket_busy_pct": busy,
        "busy_threshold": args.busy_threshold,
        "vllm_cpu_omp_threads_bind": bind,
        "vllm_cpu_kvcache_space_gb": kv,
        "socket_ram_gb": sock_ram,
        "numa_nodes_on_socket": nodes,
        "container_cpuset": container_cpuset,
        "conda_launch_prefix": conda_prefix,
        "conda_pin_tool": conda_pin,
        "warning": warn,
        "nps_note": nps_note,
    }

    if args.format == "json":
        print(json.dumps(result, indent=2))
        return

    print(f'export VLLM_CPU_OMP_THREADS_BIND="{bind}"')
    print(f"export VLLM_CPU_KVCACHE_SPACE={kv}")
    print(f"# socket {chosen} ({reason}); per-socket busy%: {busy}")
    print(f"#   container: {container_cpuset}")
    print(f"#   conda:     {conda_prefix or '(unpinned)'} vllm serve ...   [{conda_pin}]")
    if warn:
        print(f"# WARNING: {warn}")
    if nps_note:
        print(f"# NOTE: {nps_note}")


if __name__ == "__main__":
    main()
