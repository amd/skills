---
name: rocprof-compute
description: This skill should be used when profiling AMD GPU kernels with rocprof-compute to collect metrics, roofline data, and analyze bottlenecks for HIP kernels.
---

# rocprof-compute Profiling

## Purpose
- Capture AMD GPU kernel metrics, roofline data, and traces with rocprof-compute.
- Analyze collected workloads to identify bottlenecks (compute vs memory, cache/TCP/TCC/SQ utilization).

## When to Use
- Need kernel-level performance diagnostics on AMD GPUs (MI200/MI300 family).
- Comparing different kernel implementations or launch configs.
- Triaging stalls/low occupancy indicated by runtime benchmarks.

## How to Use
- Activate project venv and install rocprof-compute Python deps (once per environment):
  - `source .venv/bin/activate`
  - `python -m pip install -r /opt/rocm-7.0.0/libexec/rocprofiler-compute/requirements.txt`
- Profile a workload:
  - `source .venv/bin/activate && rocprof-compute profile -n <name> --path <out_dir> --join-type kernel -b SQ -b TCP -b TCC -- <cmd> <args>`
  - Example (paged attention ragged test):  
    `rocprof-compute profile -n kernelgen --path rocprof_compute_profile --no-roof --join-type kernel -b SQ -b TCP -b TCC -- .venv/bin/python -O op_tests/test_pa_ragged.py -p Shomy -q none -c 128`
  - Prefer `--join-type kernel` for comparing same kernel across grids; switch to `grid` if grid-sensitive.
  - Add/adjust `-b` blocks to target specific hardware units; use `--list-metrics <arch>` if unsure.
  - Use `--no-roof` to skip roofline if only counters are needed; remove it to gather roofline data.
- Analyze collected data:
  - `rocprof-compute analyze --path <out_dir> -b <metric_ids>` or `--list-stats` / `--list-metrics <arch>` to discover ids.
  - Example: `rocprof-compute analyze --path rocprof_compute_profile -b 2`
  - For interactive review, use `--gui` (default port 8050 or `--random-port`) or `--tui`.
- Typical workflow checklist:
  - Pick a short, reproducible workload and seed; pin `--name` + `--path` per experiment.
  - Collect counters (SQ/TCP/TCC) and optionally roofline in one run; avoid mixing many kernels in a single profile when isolating a hotspot.
  - After analyze, inspect top stats, occupancy, LDS/HBM bandwidth, and hotspot kernels; rerun with filtered `--kernel` or `--dispatch` if needed.

## References
- Load `references/rocprof_compute_profile_help.txt` for full `rocprof-compute profile --help`.
- Load `references/rocprof_compute_analyze_help.txt` for full `rocprof-compute analyze --help`.
