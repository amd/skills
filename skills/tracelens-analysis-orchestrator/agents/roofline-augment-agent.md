<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: roofline-augment-agent
description: Derive the checkpoint, parallelism, and per-phase batch/sequence arguments for the analytical roofline augmentation from orchestrator artifacts. Invoked by the orchestrator only when analytical roofline augmentation is enabled.
model: claude-opus-4-7-high
---

# Roofline Augment Subagent

Derive the arguments for `roofline_augment` from artifacts already on disk.

Use a value only when one of the sources below states it. When a required field cannot be derived, write `null` and set `skipped` to `"Cannot be inferred from trace"`. Do not guess, and do not use a CLI default (batch 1, sequence 2048, tp 1 on a multi-rank trace) as if it had been measured.

---

## Context Passing

**Required context provided by orchestrator:**
- `output_dir`: base analysis output directory
- `platform`: platform name from Step 0 (for example `MI300X`)
- `prefix`: command prefix from `<output_dir>/cache/cmd_prefix.txt` — substitute the command for `{CMD}`

**Inputs:**
- `<output_dir>/cache/model_info.json` — `model`, `architecture`, `scale`, `precision`
- `<output_dir>/cache/condensed_op_info.csv` — columns `name`, `Input type`, `Input Dims`. If the file is missing, produce it and then read it:

```bash
<prefix> python3 -c "
import sys
from TraceLens.Agent.Analysis.utils.report_utils import prepare_model_identification_data
if not prepare_model_identification_data('<output_dir>'):
    sys.exit(1)
"
```

- `<output_dir>/cache/nn_modules.txt` — nn.Module class names, when present
- `<output_dir>/cache/run_config.yaml` — `analysis_mode`, and each job's `name` and `phase`
- `<output_dir>/category_data/category_manifest.json` — `focus_csv_dir`
- The perf CSVs in `<output_dir>/<focus_csv_dir>`. Read headers and the rows you need; do not load a trace JSON.

**Output file you must write:**
- `<output_dir>/cache/roofline_augment_args.json`

---

## Output Schema

```json
{
  "source": "<hf-id-or-checkpoint-dir or null>",
  "code_path": null,
  "engine": "vllm",
  "tp": 8,
  "pp": 1,
  "ep": 1,
  "gpu_arch_name": "<platform>",
  "phases": [
    {"phase": "prefill", "batch_size": 1, "seq_len": 8192, "dims": {}},
    {"phase": "decode", "batch_size": 32, "seq_len": 1, "dims": {"past_len": 65536}}
  ],
  "skipped": null
}
```

`engine` is `vllm`, `sglang`, or `null`. `phases` is a list; each `phase` is `prefill` or `decode`. `dims` is an object of integer shape overrides (`past_len` for decode). `skipped` is `null` when `source` and `phases` are both resolved, and `"Cannot be inferred from trace"` otherwise. On the unresolved path set `source` to `null` and `phases` to `[]`.

---

## Derivation

### source

Read `model` from `cache/model_info.json`.

- Use it as `source` when it is a Hugging Face id: exactly one `/`, no spaces, both sides non-empty (`org/name`).
- Use it as `source` when it is an absolute path to a directory that contains `config.json`.
- Otherwise `source` is `null`. A family name (`LLM`, `Inkling`, `Workload`) is not a checkpoint. Do not invent an id from `nn_modules.txt`.

`code_path` stays `null` unless that same evidence is an existing modeling `.py` file.

### engine

Read `analysis_mode` from `cache/run_config.yaml`. Set `engine` only when `analysis_mode` is `inference`:

- `vllm` when `nn_modules.txt` or `condensed_op_info.csv` `name` cells contain `vllm`
- `sglang` when they contain `sglang`

Leave `engine` `null` otherwise. Do not set it from the platform.

### gpu_arch_name

Copy `<platform>` from the orchestrator context. Do not search for an arch file.

### tp, pp, ep

Look for an explicit degree, in this order:

1. A key `tp` / `tensor_parallel_size`, `pp` / `pipeline_parallel_size`, or `ep` / `expert_parallel_size` in `cache/run_config.yaml` or in a perf-CSV column of that name. Use the integer you find, and `1` for an axis that is absent.
2. Otherwise, if every perf CSV that has a `rank` column contains a single distinct rank, use `tp=1`, `pp=1`, `ep=1`.
3. Otherwise the split of a multi-rank trace is not in the file. Set `tp`, `pp`, and `ep` to `null`, `phases` to `[]`, and `skipped` to `"Cannot be inferred from trace"`. Do not divide a world size into tp, pp, and ep.

### phases

Collect candidate phase names from `cache/run_config.yaml` `jobs[].phase`. Keep `prefill` and `decode` only (case-insensitive; write them in lowercase). A job phase with any other name is not a roofline phase.

For each kept phase, read `batch_size` and `seq_len` from `condensed_op_info.csv` `Input Dims` on attention-like ops (`attention`, `sdpa`, `bmm` with a 4D shape):

- The leading dimension of a `[B, ...]`, `[B, S, ...]`, or `[B, H, S, D]` shape is `batch_size`.
- **prefill:** `seq_len` is the sequence axis (the axis that is not a head count and is greater than 1). Leave `dims` empty.
- **decode:** `seq_len` is `1` when the query length is 1. Put the cached context length in `dims.past_len`.

When a trace has no job phase, infer the phase from those shapes: query length `1` with a cached context is `decode`; a query length equal to the context is `prefill`. When both patterns occur, emit both phases.

When `Input Dims` is empty for every op that could supply `batch_size` and `seq_len` for a phase, drop that phase. When no phase remains, set `phases` to `[]` and `skipped` to `"Cannot be inferred from trace"`.

A `multi_trace` run projects each resolved phase once. Do not collapse prefill and decode into one phase.

---

## Error Handling

- If `model_info.json` is missing, `source` is unresolved.
- Always leave `cache/roofline_augment_args.json` in place before returning. Do not write `priority_data.json` or `roofline_augmentation.json`.
