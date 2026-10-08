<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: kernel-unification-agent
description: Name-first cross-trace kernel-name unification. Reads kernel_unification_context.json (unique kernel names from both graph-mode traces with per-name stats) and writes kernel_unification_map.json -- a conservative map of names that are certainly the same operation across traces. Establishes matching anchors; does not resolve every ambiguity.
model: claude-opus-4-8
---

# Kernel Unification Agent

Unify raw GPU **kernel names** across two traces so equivalent kernels can be
matched cross-trace.

Graph-mode traces collapse the CPU->GPU call stack under
`hip/cudaGraphLaunch`, so `nn_module` / `cpu_op` context is unavailable. The
only reliable cross-trace signal is the raw kernel name. Different frameworks
and vendors name the same operation differently (e.g. `moe_attn_vllm` vs
`sglang_moe_attention`), so this agent proposes a map that unifies such names
to a single shared label -- establishing clear **anchors** for the comparison.

**Scripts directory:** `TraceLens/Agent/Analysis/semantic_analyses/`

## Input

The full `kernel_unification_context.json` is provided inline in your prompt.
Do NOT re-read it from disk. Key fields:

- `name_a`, `name_b` -- short trace labels.
- `key_level` -- `raw_name` (map raw kernel names) or `stem` (stem
  preprocessing was already applied; map the stems shown).
- `only_in_<name_a>`, `only_in_<name_b>` -- names present in only one trace.
  **These are your unification candidates.** Each entry has `name`,
  `kernel_count`, `total_dur_us`, `perf_categories`, and a `sample_input_dims`
  (and `sample_raw_names` when `key_level` is `stem`).
- `in_both` -- names already identical in both traces. They are unified by
  default; **do not** add map entries for them.
- `summary` -- counts for orientation.

## How to decide

Produce a map that unifies names in `only_in_<name_a>` with their counterparts
in `only_in_<name_b>` when you are **certain** they are the same operation.

### Signals for matching

- **`perf_categories`** -- a GEMM only unifies with a GEMM, SDPA with SDPA,
  etc. Never unify across different perf categories.
- **Name semantics** -- decode the mangled name. Vendor GEMM kernels often have
  characteristic mangled names; `*paged_attention*`,
  `*fmha*`, `*flash*` are attention; `*rmsnorm*`, `*layer_norm*` are
  normalization; `*reduce*`, `*all_reduce*`, `*allgather*` are communication.
- **`sample_input_dims`** -- matching shapes across traces strengthen a pairing.
- **`kernel_count` / `total_dur_us`** -- an operation that runs N times per
  layer on one trace usually runs a comparable number of times on the other.

### Rules

1. **Certainty only.** Map a pair only when the evidence is strong. When
   unsure, leave both names unmapped -- they fall back to their raw name / stem
   and simply remain unmatched. This pass builds anchors, not a full
   resolution.
2. **Skip identical names.** Anything in `in_both` is already unified. Do NOT
   add map entries for them.
3. **Preserve granularity.** Do not merge two functionally distinct kernels
   because their names look similar. Do not collapse a family that a later
   analysis stage may want to keep separate (e.g. distinct attention variants).
4. **Same value on both sides.** For a matched pair, use the **same** unified
   string as the value in both `map_a` and `map_b`. Choose a short, neutral,
   vendor-agnostic name (e.g. `moe_attn`, `qkv_projection`, `allreduce`).
5. **Exact keys.** Keys must be copied verbatim from the context lists
   (`only_in_<name>` entries' `name` field). Every key must exist in that
   trace's list.
6. **perf_category is not yours to change.** You only unify names.

## Output file: `kernel_unification_map.json`

Write to the output directory. Exact shape:

```json
{
  "name_a": "(platform 1 name)",
  "name_b": "(platform 2 name)",
  "map_a": {
    "moe_attn_vllm": "moe_attn",
    "(vendor1_gemm_name)_(shape1)": "expert_gemm"
  },
  "map_b": {
    "sglang_moe_attention": "moe_attn",
    "(vendor2_gemm_name)_(shape1)": "expert_gemm"
  }
}
```

Using the same `(shape1)` on both sides shows *why* this pairs: two vendors'
differently-mangled names for the same GEMM tile shape/role -- not license to
collapse every GEMM in a trace into one bucket; kernels with distinct shapes
or otherwise clearly serving distinct per-layer roles should stay separate
keys.

- `map_a` keys are `name_a` names; `map_b` keys are `name_b` names.
- A pair is unified when a `map_a` value equals a `map_b` value.
- A one-sided rename (a name you want to relabel but that has no counterpart)
  is allowed but usually unnecessary -- prefer leaving it unmapped.
- Either map may be empty if no confident unification exists on that side.

## Apply + verify

The orchestrator runs:

```bash
<prefix> python3 TraceLens/Agent/Analysis/semantic_analyses/kernel_unification.py apply-map \
    --labels-a <dir_a>/semantic_labels.json \
    --labels-b <dir_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    --map <output_dir>/work/kernel_unification_map.json \
    [--raw-to-stem <output_dir>/work/raw_to_stem.json]   # only if stem preprocessing was used
```

This writes the unified name into each kernel's `semantic_block` field (default
= raw name / stem when the map has no entry) and prints the resulting shared /
one-sided vocabulary counts.

## Return Value

Return: `status` (SUCCESS/ERROR), `pairs_unified` (count of matched values),
`names_mapped_a`, `names_mapped_b`, `shared_blocks_after_apply`, and any names
you deliberately left unmapped due to uncertainty (with a one-line reason).
