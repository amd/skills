<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: kernel-unification-agent
description: Name-first cross-trace kernel-name unification. Reads kernel_unification_context.json (unique kernel names from both graph-mode traces with per-name stats) and writes kernel_unification_decisions.json -- a conservative, id-referenced set of candidate pairs that are certainly the same operation across traces. Establishes matching anchors; does not resolve every ambiguity.
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
  **These are your unification candidates.** Each entry has a short `id` (`a0`,
  `b1`, ...), plus `name`, `kernel_count`, `total_dur_us`, `perf_categories`,
  and a `sample_input_dims` (and `sample_raw_names` when `key_level` is `stem`).
  **Reference candidates by their `id` in your output -- never retype the
  `name`.**
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
4. **Shared label per pair.** Give each matched pair a single `label` -- a
   short, neutral, vendor-agnostic name (e.g. `moe_attn`, `qkv_projection`,
   `allreduce`). The shared label is what unifies the two kernels.
5. **Reference ids, never names.** Identify each candidate by its `id` from the
   context (`a3`, `b7`); every id must exist in that trace's `only_in_<name>`
   list. Do not retype kernel names.
6. **perf_category is not yours to change.** You only unify names.

## Output file: `kernel_unification_decisions.json`

Write to the output directory. Reference candidates by `id`:

```json
{
  "pairs": [
    {"a": "a0", "b": "b3", "label": "moe_attn"},
    {"a": "a5", "b": "b8", "label": "expert_gemm"}
  ],
  "renames_a": [],
  "renames_b": []
}
```

Pairing `a5`/`b8` under `expert_gemm` shows *why* this unifies: two vendors'
differently-mangled names for the same GEMM tile shape/role -- not license to
collapse every GEMM in a trace into one bucket; kernels with distinct shapes or
otherwise clearly serving distinct per-layer roles should stay separate pairs.

- `pairs` -- each pairs one `only_in_<name_a>` candidate (`a` id) with one
  `only_in_<name_b>` candidate (`b` id) under a shared `label`. This is the
  normal case and establishes the anchor.
- `renames_a` / `renames_b` -- optional one-sided relabels
  (`{"id": "a4", "label": "..."}`) for a candidate with no counterpart. Usually
  unnecessary -- prefer leaving such a candidate unmapped.
- Each id is used at most once. Omit uncertain candidates entirely; they fall
  back to their raw name / stem and remain unmatched.
- `pairs` may be empty if no confident unification exists.

## Apply + verify

The orchestrator validates your decisions, expands the ids back to names, then
applies the resulting map:

```bash
# 1. Validate (retry gate) -- validate_kernel_unification_map(decisions, context).
#    On failure the orchestrator returns the errors for you to fix and re-emit.
# 2. Expand ids -> names:
<prefix> python3 TraceLens/Agent/Analysis/semantic_analyses/kernel_unification.py expand-map \
    --decisions <output_dir>/work/kernel_unification_decisions.json \
    --context <output_dir>/work/kernel_unification_context.json \
    -o <output_dir>/work/kernel_unification_map.json
# 3. Apply:
<prefix> python3 TraceLens/Agent/Analysis/semantic_analyses/kernel_unification.py apply-map \
    --labels-a <dir_a>/semantic_labels.json \
    --labels-b <dir_b>/semantic_labels.json \
    --name-a <name_a> --name-b <name_b> \
    --map <output_dir>/work/kernel_unification_map.json \
    [--raw-to-stem <output_dir>/work/raw_to_stem.json]   # only if stem preprocessing was used
```

Apply writes the unified name into each kernel's `semantic_block` field (default
= raw name / stem when no pair covers it) and prints the resulting shared /
one-sided vocabulary counts.

## Return Value

Return: `status` (SUCCESS/ERROR), `pairs_unified` (count of pairs), `renames_a`,
`renames_b`, and any candidates you deliberately left unmapped due to
uncertainty (with a one-line reason).
