#!/usr/bin/env python3
"""
Detect the local CPU hardware for the multi-instance vLLM benchmark.

Usage:
    python3 scripts/detect.py

Output: JSON with cpu_model, is_amd_epyc, epyc_generation
(Naples/Rome/Milan/Genoa/Bergamo/Siena/Turin), zen_arch, avx512, physical_cores,
logical_cores, sockets, threads_per_core, numa_nodes, memory_gb. Exits 0 on
success, 1 if no CPU info could be read.

Use physical_cores to size the sweep: with CORES_PER_INSTANCE=32 fixed, run
NUM_INSTANCES = floor((physical_cores - 16) / 32) on a single socket.
"""

import json
import re
import subprocess
import sys


def _run(cmd, timeout=20):
    r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def _lscpu_field(lscpu_out, label):
    m = re.search(rf"^{re.escape(label)}:\s*(.+)$", lscpu_out, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _epyc_generation(model):
    """Map an AMD EPYC model name to (generation, zen_arch).

    EPYC numbering encodes the generation: 7xx1=Naples (Zen1), 7xx2=Rome (Zen2),
    7xx3=Milan (Zen3), 8xx4=Siena (Zen4c), 97x4=Bergamo (Zen4c), 9xx4=Genoa (Zen4),
    9xx5=Turin (Zen5)."""
    m = re.search(r"EPYC\s+(\d{4})", model.upper())
    if not m:
        return "unknown", "unknown"
    num = m.group(1)
    first, last = num[0], num[3]
    if first == "7":
        return {"1": ("Naples", "Zen1"), "2": ("Rome", "Zen2"),
                "3": ("Milan", "Zen3")}.get(last, ("unknown", "unknown"))
    if first == "8" and last == "4":
        return "Siena", "Zen4c"
    if first == "9":
        if num.startswith("97") and last == "4":
            return "Bergamo", "Zen4c"
        if last == "4":
            return "Genoa", "Zen4"
        if last == "5":
            return "Turin", "Zen5"
    return "unknown", "unknown"


def main():
    rc, lscpu_out, err = _run("lscpu")
    if rc != 0 or not lscpu_out:
        print(json.dumps({"error": "lscpu failed",
                          "detail": err.strip() or f"exit {rc}"}))
        sys.exit(1)

    model = _lscpu_field(lscpu_out, "Model name") or "unknown"
    vendor = _lscpu_field(lscpu_out, "Vendor ID")

    def _int(label, default=0):
        v = _lscpu_field(lscpu_out, label)
        try:
            return int(v)
        except ValueError:
            return default

    sockets = _int("Socket(s)", 1)
    cores_per_socket = _int("Core(s) per socket", 0)
    threads_per_core = _int("Thread(s) per core", 1) or 1
    numa_nodes = _int("NUMA node(s)", 1)

    rc, nproc_out, _ = _run("nproc --all")
    try:
        logical = int(nproc_out.strip())
    except (ValueError, AttributeError):
        logical = sockets * cores_per_socket * threads_per_core

    physical = sockets * cores_per_socket if cores_per_socket else logical // threads_per_core

    rc, mem_out, _ = _run("grep MemTotal /proc/meminfo")
    mem_kb = 0
    m = re.search(r"(\d+)", mem_out or "")
    if m:
        mem_kb = int(m.group(1))
    memory_gb = mem_kb // (1024 * 1024)

    is_epyc = vendor == "AuthenticAMD" and "EPYC" in model.upper()
    generation, zen_arch = _epyc_generation(model)
    avx512 = "avx512f" in _lscpu_field(lscpu_out, "Flags").split()

    print(json.dumps({
        "cpu_model": model,
        "vendor": vendor,
        "is_amd_epyc": is_epyc,
        "epyc_generation": generation,
        "zen_arch": zen_arch,
        "avx512": avx512,
        "logical_cores": logical,
        "physical_cores": physical,
        "sockets": sockets,
        "threads_per_core": threads_per_core,
        "numa_nodes": numa_nodes,
        "memory_gb": memory_gb,
    }, indent=2))


if __name__ == "__main__":
    main()
