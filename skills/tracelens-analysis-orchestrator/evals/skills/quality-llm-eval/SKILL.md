<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: quality-llm-eval
description: >-
  Runs LLM-based quality evals 2 and 3: semantic alignment of compute P-item titles and
  content against a reference analysis tree, and writes a two-row results CSV.
  Use when the user asks for a quality LLM eval or to compare generated vs reference
  analysis.md and category_findings. Skips system-level P-items for content alignment.
---

# Quality LLM eval

Compare generated **`analysis.md`** and **`category_findings/*`** to reference artifacts; score **correctness**, **completeness**, and **precision** (Gaia-style). Output CSV rows for **`quality_eval_2`** and **`quality_eval_3`**.

## Full procedure

See **[reference.md](reference.md)** for inputs, file list, per-eval scoring guides, comparative-scope notes, and CSV format.

## Skill location

`TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/skills/quality-llm-eval/`
