<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: workflow-llm-eval
description: >-
  Runs LLM-based workflow eval 12 only: checks the analysis.md Appendix for plausible
  hardware reference values (platform, HBM BW, MAF) and writes a one-row results CSV.
  Use when the user asks for a workflow LLM eval, eval 12, or Appendix hardware
  reference scoring. Scripted evals 9–11, 13–14 live in workflow_scripted_evals.py.
---

# Workflow LLM eval

Semantic check of **`## Appendix`** hardware facts in `analysis.md` (Gaia-style weighted **correctness** + **completeness**). Produces a single CSV row for `workflow_eval_12`.

## Full procedure

See **[reference.md](reference.md)** for inputs, files to read, scoring table, pass/fail thresholds, and CSV column contract.

## Skill location

`TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/skills/workflow-llm-eval/`
