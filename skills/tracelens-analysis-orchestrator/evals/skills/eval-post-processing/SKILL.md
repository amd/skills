<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: eval-post-processing
description: >-
  Aggregates TraceLens analysis repeatability CSVs, classifies failures using
  report_section_rules.yaml, writes PR and fix-ticket markdown reports, and builds
  reproducer tarballs. Use when the user runs eval post processing, aggregates
  repeatability results after the harness, or asks for automated eval reports from
  results_root / report_dir parameters.
---

# Eval post processing

Run **after** `run_repeatability_parallel.sh` (or manually on an existing results tree): aggregate per-run eval data, split **unit** vs **e2e** metrics, classify failures, emit `pr_report.md` and `fix_ticket_report.md`, package reproducers, then copy the report tree to `TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/eval_reports/latest/`.

## Full procedure

Follow **[reference.md](reference.md)** for key=value inputs, exact report templates, reproducer layout, path sanitization, and the final summary block.

## Step index

```
4. Aggregate — aggregate_repeatability.py → report_dir/aggregates/*.csv
5. Read and classify — YAML rules + per-split metrics
6. Write reports — pr_report.md, fix_ticket_report.md
7. Build reproducer packages — per issue_summary, tar.gz
8. Save and summarize — copy to eval_reports/latest/, print paths
```

## Skill location

Bundled at `TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/skills/eval-post-processing/` (`SKILL.md`, `reference.md`). For Cursor auto-discovery, symlink or copy this folder under `.cursor/skills/` in a workspace that includes this repo.
