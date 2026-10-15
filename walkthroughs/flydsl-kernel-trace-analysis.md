# FlyDSL kernel trace analysis

Install the catalog entry and analyze a small synthetic trace using the existing
analyzers in FlyDSL. This walkthrough needs Python 3.10+ but no GPU, ROCm runtime,
or installed FlyDSL package.

## Install

From your project directory:

```bash
npx skills add amd/skills --skill flydsl-kernel-trace-analysis --agent claude-code
```

The [entry point](../skills/flydsl-kernel-trace-analysis/SKILL.md) reads the native
`.claude/skills/kernel-trace-analysis/SKILL.md` and runs the scripts beside it in
your FlyDSL checkout. If you have no checkout, it fetches the official repository
into your cache. Git and network access are needed only for that fetch.

## Analyze the offline ATT fixture

The installed skill includes [synthetic evaluation data](../skills/flydsl-kernel-trace-analysis/evals/fixtures/offline/out_kernel_trace.csv).
With a project installation for Claude Code, ask:

```text
Use flydsl-kernel-trace-analysis to analyze sample_kernel on gfx942.
The synthetic dispatch directory is:
.claude/skills/flydsl-kernel-trace-analysis/evals/fixtures/offline/ui_output_agent_0_dispatch_3

Use the matching resource CSV in its parent and include source context.
Write analysis.md. Do not collect a new GPU trace.
```

Expected observations are `kernel.py:4` as the largest source hotspot with 800
stall cycles, and dispatch 3's combined 280-register allocation. Estimated
occupancy is one wave per SIMD. These are fixture results, not a performance
measurement on your GPU.

## Analyze the PMC fixture

```text
Use flydsl-kernel-trace-analysis on l2.csv and ea.csv under the same fixture
directory, filtering sample_kernel. These are complementary counter groups
for one dispatch. Ideal reads are 0.000128 GB per dispatch; EA0 scaling is 2.
Write memory.md with the metrics and what they do and do not establish.
```

The expected values are 2% L2 hit rate, 0% partial reads, and 1.00x estimated
over-fetch. ATT alone cannot supply these cache metrics. Neither report proves
an optimization speedup.

For your own data, supply the dispatch directory or PMC files, kernel name, and
GPU architecture. For new collection, supply the workload command and follow the
native [FlyDSL trace skill](https://github.com/ROCm/FlyDSL/blob/main/.claude/skills/kernel-trace-analysis/SKILL.md).
