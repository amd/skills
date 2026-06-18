---
name: analysis-orchestrator
description: >-
  Orchestrates modular PyTorch profiler trace analysis with TraceLens: generates perf
  reports, prepares category data, runs system-level and compute-kernel subagents in
  parallel, validates outputs, and writes a prioritized stakeholder report (analysis.md).
  Use when the user asks to follow the analysis orchestrator, run the agentic analysis
  workflow, analyze a trace, compare two traces, or mentions standalone or comparative
  TraceLens analysis.
---

<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Analysis orchestrator

Coordinate **system-level** analysis (CPU/idle, kernel fusion, multi-kernel / comm / memcpy) and **compute-kernel** analysis (GEMM, SDPA, elementwise, etc.): one trace load, shared prep, parallel subagents, then aggregation into `analysis.md`.

## Full procedure

Follow **[reference.md](reference.md)** for every step (user prompts, `<prefix>` / `{CMD}` usage, CLI commands, subagent launch text, validation, report `tee` order, plot embedding, and trace diagnostics).

## Workflow index

```
0. Query User Inputs (Platform, Trace Path(s), Analysis Mode, Environment Setup)
1. Generate Performance Report (branches on analysis mode: training vs inference then, comparison scope)
2-5. Prepare Category Data (GPU Util, Top Ops, Tree Data, Multi-Kernel Data, Category Filtering)
6. System-Level Analysis (PARALLEL) → system_findings/
7. Compute Kernel Subagents (PARALLEL) → category_findings/
   7.5. Aggregate → priority_data.json::findings[]
8. Validate Subagent Outputs
9. load_findings + Model Identification (subagent) → metadata/model_info.json
10. Render performance PNG if agent_extension.py is absent
11. Generate analysis.md (orchestrator writes via <prefix> tee), optional extension, embed PNG
```

## Rules

- **Subagents:** Use the Task tool **only** where reference.md says “subagent” (Steps **6**, **7**, **9**). The orchestrator runs everything else, including Step 7.5, using the command prefix from `<output_dir>/cache/cmd_prefix.txt` (`{CMD}` substitution).
- **Language:** Prefer vendor-agnostic terms (GPU kernels, collective communication, vendor GEMM library, DNN primitives, GPU graph). When quoting trace data, real kernel names are fine.
- **Subagent prompts:** Point each subagent at the checked-in agent file under `TraceLens/Agent/Analysis/skills/analysis-orchestrator/agents/<name>.md` (see reference.md for exact paths and prompt shells).

## Primary outputs

- **Deliverable:** `<output_dir>/analysis.md`
- **Internals:** `system_findings/`, `category_findings/`, `category_data/`, `metadata/`, `perf_report*.xlsx`, CSV folders — see package README for layout.

## Agent layout

Project subagents ship with this skill: `TraceLens/Agent/Analysis/skills/analysis-orchestrator/agents/*.md`.
