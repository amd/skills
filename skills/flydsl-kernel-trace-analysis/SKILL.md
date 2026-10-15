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

# FlyDSL Kernel Trace Analysis

Use FlyDSL's [existing kernel trace analysis skill](https://github.com/ROCm/FlyDSL/blob/82d59dcde9a943747a58f714977324aab359d0b1/.claude/skills/kernel-trace-analysis/SKILL.md)
and its analyzers to interpret ATT and PMC data. This entry locates that skill;
trace collection instructions, analysis code, and tuning guidance stay in FlyDSL.

## Prerequisites

- Offline analysis needs Python 3.10+ and the standard library. It needs no GPU,
  ROCm installation, container, or installed FlyDSL package.
- Git and network access are needed only if a source checkout must be fetched.
- Collection follows the upstream skill's ROCm, `rocprofv3`, aqlprofile, and
  trace-decoder requirements. The supported targets are `gfx942` and `gfx950`;
  use the existing compatible Linux host or container, with no image pinned here.
- For FlyDSL collection, enable `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` before
  compilation. Check `ARCH`, `HIP_VISIBLE_DEVICES`, and `ROCR_VISIBLE_DEVICES`;
  do not change `HSA_OVERRIDE_GFX_VERSION` to disguise unsupported hardware.

## Locate the source skill

1. Prefer the FlyDSL checkout supplied by the user or containing the workload.
   Check `FLYDSL_ROOT` if it is already set. Resolve it to an absolute path.
2. If no checkout is available, reuse or create a source checkout in the user's
   cache. On Linux:

   ```bash
   FLYDSL_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/flydsl-skills/FlyDSL"
   # Run clone only if this directory does not already exist.
   git clone --depth 1 https://github.com/ROCm/FlyDSL.git "$FLYDSL_ROOT"
   ```

   On Windows, use a directory under `%LOCALAPPDATA%/flydsl-skills` instead.
   Fetching the repository does not install FlyDSL. Preserve existing checkouts
   and their revision; use a user-requested revision when supplied.
3. Confirm that `.claude/skills/kernel-trace-analysis/SKILL.md` and its two
   scripts, `scripts/hotspot_analyzer.py` and `scripts/pmc_l2_analyzer.py`, exist
   under that checkout. If the source is unavailable, report what is missing.

## Load and use the existing skill

Read `$FLYDSL_ROOT/.claude/skills/kernel-trace-analysis/SKILL.md` as a file and
follow it. Resolve its scripts and companion skills against the same checkout;
run the existing analyzers by their absolute paths. Keep trace inputs and
generated reports in the user's workload directory.

Preserve the user's command, `--dir`, `--topk`, kernel filter, and output paths.
For an existing dispatch directory, perform offline analysis and skip collection
or GPU setup. For a new collection, use the upstream collection instructions.

Record the source checkout and revision used (`git -C "$FLYDSL_ROOT" rev-parse HEAD`).
Report measured hotspots or counters and their limits. ATT data alone cannot
provide L2 hit rate; optimization gains require correctness and timing validation.
