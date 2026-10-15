---
name: flydsl-kernel-trace-analysis
description: >-
  Profiles FlyDSL and other AMD GPU kernels with rocprofv3 ATT traces, maps
  instruction stalls to source lines, and produces a measured optimization plan.
  Applies to existing ui_output_agent dispatch directories, code.json stall
  hotspots, VMEM or LDS waits, MFMA stalls, and rocprofv3 PMC cache/HBM counters
  on gfx942 or gfx950. Compile-only resource comparisons and application-level
  PyTorch timeline analysis are outside this skill's scope.
---

# Kernel Trace Analysis

Analyze instruction-level traces with the bundled Python analyzers. Start from
existing artifacts when supplied; collect a new trace only when the task needs it.

## Prerequisites

- Offline analysis needs Python 3.10+ and the standard library; no GPU, ROCm
  installation, container, or FlyDSL checkout is needed.
- Collection defaults to Linux with ROCm 7.2, `rocprofv3`, matching aqlprofile and
  rocprof-trace-decoder, and an AMD Instinct MI300X (`gfx942`) or MI350 (`gfx950`).
  Confirm `/opt/rocm/.info/version`, `rocprofv3 --version`, and the actual GPU.
  The resource estimator is specific to these CDNA targets, not RDNA/wave32.
- Use the existing compatible host or container. No container image is required
  or pinned here; record its tag/digest if one is used. GPU collection needs a
  compatible host driver and access to `/dev/kfd` and `/dev/dri`.
- FlyDSL collection also needs an importable FlyDSL installation and its workload
  dependencies. Set `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` before compilation for
  source mapping. Check `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, and `ARCH`;
  do not use `HSA_OVERRIDE_GFX_VERSION` to disguise an unsupported architecture.

## Inputs

| Argument | Behavior |
| --- | --- |
| `<CMD>` | Profile a command, for example `python bench_pa.py --batch 32` |
| `--dir <path>` | Analyze an existing dispatch directory and skip collection |
| `--topk N` | Show the top N hotspots; default 15 |

Use the supplied command or dispatch directory. These arguments select the
workflow; pass `--topk` through to the hotspot analyzer.

## Bundled analyzers

Resolve scripts relative to this SKILL.md, independently of the workload's
working directory. In the commands below, `SKILL_DIR` is that absolute directory.

- Run [scripts/hotspot_analyzer.py](scripts/hotspot_analyzer.py) for ATT stalls,
  source context, and resource estimates.
- Run [scripts/pmc_l2_analyzer.py](scripts/pmc_l2_analyzer.py) for PMC L2 hit rate,
  partial read fraction, and estimated HBM over-fetch.

## Analyze an existing trace

1. Confirm the dispatch directory contains `code.json` with nonzero total cycles.
   Preserve `snapshots.json` and embedded source files for line context, and keep
   the matching `out_kernel_trace.csv` in the directory or its parent.
2. Select the kernel name from that CSV. Pass `--kernel` for standard
   `ui_output_agent_*_dispatch_<id>` names: the script uses both the kernel
   substring and dispatch ID to select resource metadata.
3. Run the reports:

```bash
python "$SKILL_DIR/scripts/hotspot_analyzer.py" <dispatch_dir> --kernel <kernel_name> --topk 15 --mode both
python "$SKILL_DIR/scripts/hotspot_analyzer.py" <dispatch_dir> --kernel <kernel_name> --topk 5 --mode src --detail --context 4
```

4. Rank source lines by stall cycles, then inspect the associated instructions.
   Read [trace-format-and-patterns.md](trace-format-and-patterns.md) for field
   definitions, source mapping, and scheduling patterns. Aggregated instruction
   cycles across waves are not elapsed kernel time.
5. Read [resource-and-memory-analysis.md](resource-and-memory-analysis.md) when
   register pressure, occupancy, or memory efficiency matters. Occupancy is a
   heuristic estimate: verify the architecture and matching resource metadata.
   Missing CSV metadata, ISA-only register scans, and rounded LDS allocations
   limit precision. ATT alone cannot establish cache hit rate or HBM over-fetch.

For PMC CSVs, supply the intended kernel, ideal read bytes **per dispatch** in
decimal GB, and the EA channel scaling appropriate to the measured counters:

```bash
python "$SKILL_DIR/scripts/pmc_l2_analyzer.py" <l2.csv> <ea.csv> --kernel <kernel_name> --ideal-gb <GB_per_dispatch> --ea-channels 2
```

The analyzer combines dispatch IDs across files. Only combine complementary
counter groups from equivalent dispatch sets; do not mix repeated captures that
reuse IDs or overlap counters. The default two-channel scaling is an assumption
to verify. Streaming-decode interpretations in the reference are workload-specific.

## Collect a trace

When given a command to profile, use [collection.md](collection.md): discover
kernels, select a dispatch after warmup, configure ATT, enable source debug info,
and retain the decoded output and resource CSV. It also describes separate PMC
collection; cap counter groups to avoid accidental multi-pass collection.

If required hardware, decoder, permissions, or workload dependencies are missing,
report the missing prerequisite and analyze available artifacts. Do not invent
trace results. For missing source locations, report instruction-level hotspots;
for empty or truncated data, fix collection before drawing performance conclusions.

## Deliver an optimization plan

Report the analyzed kernel/dispatch and architecture, total attributed cycles,
top stall types, and the largest source hotspots with their instructions. For
each proposed change, connect observed evidence to a testable scheduling or
layout hypothesis and its register/LDS tradeoff. Prioritize by measured stall
share and effort. Treat predicted gains as hypotheses, not measured speedups.

Examples include moving loads earlier to overlap independent MFMA work, batching
LDS reads before dependent matrix instructions, or reducing live values when
register pressure limits occupancy. Validate changes with numerical tests and
comparable unprofiled timings before claiming an improvement.
