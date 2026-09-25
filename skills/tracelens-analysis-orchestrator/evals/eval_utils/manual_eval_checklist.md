<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Manual Evaluation Checklist

Checks **not** covered by `validate_report()`.
Run these against every generated `analysis.md` after the programmatic validator passes.

## Link / Anchor Integrity

- [ ] Every P-item card contains a link to its Detailed Analysis anchor (e.g., `[Detailed Analysis: Compute kernel insights > P1](#detailed-analysis-compute-p1)`)
- [ ] Every Detailed Analysis P-block has a matching HTML anchor (`<a id="detailed-analysis-compute-pN">` or `<a id="detailed-analysis-system-pN">`)

## Label Formatting

- [ ] Each of the five required labels (`**Identification:**`, `**Data:**`, `**Reasoning for Slowdown:**`, `**Resolution:**`, `**Impact estimate:**`) starts on its own line with a newline between consecutive labels

## Card–Detailed Analysis Consistency

- [ ] Every operation mentioned in the card **Insight** appears in the Detailed Analysis **Data** table
- [ ] Efficiency figures cited in the card match the Detailed Analysis **Data** table values (allowing for rounding)
- [ ] Time values and percentages in the card match the Detailed Analysis **Data** table (allowing for rounding)
- [ ] Card **Impact** range is consistent with Detailed Analysis **Impact estimate** bullets
- [ ] No numbers or claims appear in the card that are absent from the Detailed Analysis block, and vice versa

## Resolution Quality

- [ ] No tautological roofline restatements beyond what the regex validator catches (e.g., novel phrasings of "getting closer to roofline = faster")
- [ ] Resolution explains the **mechanism** by which the optimization raises utilization when inferable from the trace
- [ ] If the mechanism is not inferable, Resolution states only the action without a roofline truism

## Field Name Placement

- [ ] JSON keys, dotted field paths, internal CSV column names, and internal variable names (e.g. `matrix_bf16`, `peak_hbm_bw`) appear **only** in the **Identification** `(source: ...)` parenthetical
- [ ] **Data**, **Reasoning for Slowdown**, **Resolution**, and **Impact estimate** use plain-language display headers and narrative only

## Metadata Integrity

- [ ] `impact_estimates` in each `metadata/*.json` is an **array** (not a bare object) with exactly one entry per `reasoning-candidate` block, ordered by ascending `rank`. Each entry sums only the operations that its candidate covers — not the entire category
- [ ] Reasoning candidate blocks (`<!-- reasoning-candidate tier=... rank=... -->`) are present in the corresponding findings files under `category_findings/` or `system_findings/`

## Sentence Quality

- [ ] No run-on sentences used to artificially compress content into the template's suggested sentence count (e.g., cramming 3–4 ideas into 2 sentences with semicolons or em-dashes)
- [ ] Sentence counts in the template are treated as **guidance for thoroughness**, not hard limits — if the content naturally requires a few more sentences, that is preferred over run-on compression

## Cross-Section Consistency

- [ ] No claim or calculation in the Detailed Analysis section contradicts the corresponding Optimization Insight card
- [ ] Top Operations table totals and percentages are consistent with per-P-item breakdowns
- [ ] System-Level Optimizations section content is consistent with `### System-level insights` under Detailed Analysis
