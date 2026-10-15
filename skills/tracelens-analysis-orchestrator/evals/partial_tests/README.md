<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Partial-workflow tests

This folder holds tests that run only **part** of an analysis workflow — just
far enough to produce an artifact that a scripted eval scores against a
pre-baked reference. This is deliberately lighter than
`../eval_scripts/run_repeatability_parallel.sh`, which runs the full 11-step
analysis orchestrator per case.

The first (and currently only) partial-workflow test is **`semantic_purity`**.
The folder is named generically so other partial-workflow tests can be added
beside it (give them a new value in the `workflow` column of
`partial_test_cases.csv` and a matching branch in `run_partial_tests.sh`).

## Layout

```
partial_tests/
├── run_partial_tests.sh              # the runner
├── generate_gold_ref.sh # OPTIONAL, rare: regenerate gold
├── partial_test_cases.csv            # manifest (id,workflow,trace_a,trace_b,reference_dir,platform_a,platform_b)
├── eval_utils/
│   ├── semantic_partition_scripted_evals.py   # per-run purity metrics (informational)
│   └── semantic_purity_aggregate.py           # the regression gate
├── fixtures/*.tar.gz                 # DECODE trace pair + pre-baked gold (small)
└── <test_id>/                        # expanded fixture (MI300/, B300/, analysis_output_ref/)
```

`compare_lca_partitions.py` — the shared purity/consistency math — lives in
`../eval_utils/` (it is also imported by `tests/test_compare_lca_partitions.py`).

## What `semantic_purity` does

For each model (DeepSeek-R1, Qwen3-30B-A3B) the fixture ships:
- the **DECODE** trace for MI300 and B300 (single batch-16 decode execution — the
  only trace kind these tests use), and
- a pre-baked **gold** `analysis_output_ref/semantic_purity_gold_diff_stats.csv`.

The runner invokes the **semantic-comparison workflow** (the no-capture,
name-first + coherence bucketing method) on the DECODE pair, running only
through its "Generate TraceDiff Output" step — equivalent to the main
orchestrator through Step 1.S. That produces
`analysis_output/tracediff_output/diff_stats.csv`, whose LCA partition is then
compared to gold.

Gold itself is the *with-capture* TraceDiff partition on the same DECODE pair.
Because both sides derive from the same single-execution DECODE trace, their
`gpu_op_uid` numbering lines up, so the `(source, gpu_op_uid)` join used by
`compare_lca_partitions.py` is meaningful.

## Running

```bash
# from the repo root
bash TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/partial_tests/run_partial_tests.sh

# one model, single run (quick check)
TEST_IDS=semantic_purity_deepseek_r1 NUM_REPEATS=1 \
  bash TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/partial_tests/run_partial_tests.sh

# variance study
NUM_REPEATS=5 bash TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/partial_tests/run_partial_tests.sh
```

Env knobs: `NUM_REPEATS` (default 1), `TEST_IDS` (space-separated whitelist),
`MAX_PARALLEL` (default 3), `CONTAINER` (docker container to exec python in),
`AGENT_MODEL`. Results land under `partial_tests/results/<id>/run_*/`.

## Semantic-purity quality gate

The semantic bucketing method uses an LLM for its unification/coherence steps,
so it has **real run-to-run variance**. Therefore:

- `semantic_partition_scripted_evals.py` writes each run's metrics
  (`forward_purity`, `strict_forward`, etc.) but is **informational only** — its
  per-run `result` is PASS unless the pipeline itself failed (missing candidate,
  no matched keys).
- `semantic_purity_aggregate.py` is the **actual gate**. It averages
  `strict_forward` across the runs found and compares to a per-model floor.
  The floors — and the observed-run data they're derived from — live in that
  script's `MIN_STRICT_FORWARD_AVG` and its module docstring, which are the
  single source of truth (not duplicated here). The floors are set below the
  worst single observed run with margin, so a quick `NUM_REPEATS=1` check does
  not false-alarm, while staying well above the random-shuffle baseline. Raise
  them if the method is intentionally improved; do not lower them without a
  documented reason.

## Regenerating gold (optional, rare — maintainer only)

Gold ships **pre-baked** inside the fixture tarballs, so normal test runs never
need this. Regenerate only after an intentional, reviewed change to the
with-capture TraceDiff path.

**Precondition — not runnable from a clean clone.** Regeneration needs the
original **full-capture** traces (the DECODE traces *and* their
`capture_traces/` folders). Those are large and are **not committed to the
repo** — only the slimmed DECODE-only traces ship in the fixtures. You must
already have the full-capture traces available locally (on the machine where
they were captured). The script's source paths default to
`tests/traces/semantic/...` and are set in the `SEM_ROOT` / `SOURCES` entries at
the top of the script; edit them to point at wherever you keep the traces. The
script fails with a clear "missing source" error if they aren't present.

```bash
bash TraceLens/Agent/Analysis/skills/analysis-orchestrator/evals/partial_tests/generate_gold_ref.sh
```

It runs the with-capture perf-report / TraceDiff path on the DECODE pair, writes
the refreshed CSV into each `analysis_output_ref/`, and repacks the fixture
tarball.
