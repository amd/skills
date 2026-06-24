#!/usr/bin/env python3
"""
Derive vLLM-on-CPU runtime knobs from the host. Deterministic and read-only.

Default policy (kept deliberately simple): a single instance uses **socket 0's
entire CPU**, with **no memory binding** -- regardless of NPS mode (NPS1/2/4).

Emits two env vars:
  - VLLM_CPU_OMP_THREADS_BIND : physical cores of socket 0. vLLM binds its OMP
    threads to these and sets OMP_NUM_THREADS itself (= len(cores)), so we don't.
  - VLLM_CPU_KVCACHE_SPACE    : KV-cache RAM (GB).

And, for the container path on a multi-socket host, a CPU-only cpuset:
  - container_cpuset : --cpuset-cpus=<socket 0 physical cores>   (no --cpuset-mems)
The conda path needs nothing extra -- VLLM_CPU_OMP_THREADS_BIND binds the threads.

We do NOT set OMP_NUM_THREADS (vLLM derives it) or VLLM_CPU_NUM_OF_RESERVED_CPU
(vLLM has its own default when unset).

If socket 0 spans multiple NUMA nodes (NPS2/NPS4), `perf_note` flags that optimal
per-node binding could give more performance -- the default does not do it.

Usage:
    python3 scripts/cpu_tune.py                 # export lines for `eval`
    python3 scripts/cpu_tune.py --format json   # machine-readable
    python3 scripts/cpu_tune.py --kv-frac 0.5
"""

import argparse
import json
import re
import subprocess
import sys

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


def _ranges(cpus):
    """Compress a sorted int list to a range string: [0,1,2,5] -> '0-2,5'."""
    if not cpus:
        return ""
    out, start, prev = [], cpus[0], cpus[0]
    for c in cpus[1:]:
        if c == prev + 1:
            prev = c
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = c
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(out)


def socket0_physical_cpus():
    """Physical cores of socket 0 from `lscpu -p`: one CPU per core, SMT siblings
    dropped. vLLM CPU gains nothing from SMT, so we run on physical cores only."""
    phys, seen = [], set()
    for line in _sh("lscpu -p=CPU,CORE,SOCKET").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 3 or parts[2] != "0":
            continue
        cpu, core = int(parts[0]), parts[1]
        if core not in seen:
            seen.add(core)
            phys.append(cpu)
    return sorted(phys)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kv-frac", type=float, default=0.4, help="fraction of RAM for KV cache")
    p.add_argument("--format", choices=["env", "json"], default="env")
    args = p.parse_args()

    lscpu = _sh("lscpu")
    numa = _lscpu_int(lscpu, "NUMA node(s)", 1)
    sockets = _lscpu_int(lscpu, "Socket(s)", 1)
    nodes_per_socket = max(1, numa // max(1, sockets))

    mem = 0
    m = re.search(r"MemTotal:\s*(\d+)", _sh("grep MemTotal /proc/meminfo"))
    if m:
        mem = int(m.group(1)) // (1024 * 1024)
    if mem <= 2 * OS_HEADROOM_GB:
        kv = max(1, int(mem * 0.5))
    else:
        kv = max(1, min(int(mem * args.kv_frac), mem - OS_HEADROOM_GB))

    bind = _ranges(socket0_physical_cpus())
    # CPU-only cpuset for the container, physical cores only (same list as the
    # bind). Only meaningful when there is more than one socket to exclude. No
    # --cpuset-mems (no memory binding by default).
    container_cpuset = f"--cpuset-cpus={bind}" if sockets > 1 else ""

    perf_note = ""
    if nodes_per_socket > 1:
        perf_note = (f"socket 0 spans {nodes_per_socket} NUMA nodes (NPS{nodes_per_socket}); "
                     "the default uses the whole socket with no memory binding. Optimal "
                     "per-NUMA-node binding (memory bound to each node) could give more "
                     "performance -- not done by default.")

    result = {
        "vllm_cpu_omp_threads_bind": bind,
        "vllm_cpu_kvcache_space_gb": kv,
        "sockets": sockets,
        "numa_nodes": numa,
        "nodes_per_socket": nodes_per_socket,
        "container_cpuset": container_cpuset,
        "memory_gb": mem,
        "perf_note": perf_note,
    }

    if args.format == "json":
        print(json.dumps(result, indent=2))
        return

    print(f'export VLLM_CPU_OMP_THREADS_BIND="{bind}"')
    print(f"export VLLM_CPU_KVCACHE_SPACE={kv}")
    print(f"# default: socket 0's CPUs, no memory binding ({sockets} socket(s), {numa} NUMA node(s))")
    if container_cpuset:
        print(f"#   container: {container_cpuset}")
    print("#   conda: VLLM_CPU_OMP_THREADS_BIND binds the threads; nothing else needed")
    if perf_note:
        print(f"# NOTE: {perf_note}")


if __name__ == "__main__":
    main()
