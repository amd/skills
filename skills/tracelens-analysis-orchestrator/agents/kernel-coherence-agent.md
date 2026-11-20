<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: kernel-coherence-agent
description: Second-pass cross-trace refinement. Consumes kernel_coherence_context.json (one-sided kernel buckets with their shared-neighbor context) and writes kernel_coherence_decisions.json, pairing one-sided buckets across traces by position and splitting names that occur in different contexts, so the comparison has no one-sided condensed symbols.
model: claude-opus-4-8
---

# Kernel Coherence Agent

Second pass over the name-first unification. The first pass leaves **one-sided**
buckets: kernels whose unified name appears in only one trace -- most importantly
vendor GEMM families that could not be paired by name ((platform 1 name) `(likely gemm name platform 1)_*` vs
(platform 2 name) `(likely gemm name platform 2)_*`). This pass uses the first-pass **shared** buckets as cross-trace
positional anchors and re-labels the one-sided buckets by their **neighbor
context**.

Two capabilities:

1. **Cross-trace pairing by position.** A one-sided bucket in trace A and a
   one-sided bucket in trace B that sit in the *same* shared-neighbor context are
   the same operation -- give them the **same new shared name**. (The GEMM
   between `add_rmsnorm` and `rotary_embedding` is the QKV projection on both
   traces, even though the raw kernels are `(likely gemm name platform 1)_...` vs `(likely gemm name platform 2)_...`.)
2. **Context-dependent splitting.** One name that appears in *different* contexts
   should become *different* buckets (a GEMM between attention and norm vs a GEMM
   between embedding and MoE).

**Scripts directory:** `TraceLens/Agent/Analysis/semantic_analyses/`

## Input

The full `kernel_coherence_context.json` is provided inline. Key fields:

- `condensed_sequence_a` / `_b` -- kernel-order semantic_block values with
  consecutive duplicates collapsed (one symbol per run).
- `shared_blocks` -- symbols in both condensed sequences (your stable anchors).
- `one_sided_in_a` / `one_sided_in_b` -- the symbols you must resolve.
- `one_sided_details_a` / `_b` -- per one-sided symbol, a list of `contexts`,
  each with `id`, `left_window` / `right_window` (nearest shared symbols),
  `kernels_in_run`, `perf_categories`, `sample_input_dims`, and
  `top_kernel_names_by_dur`.
- `context_catalog` -- flat list of every context with its `id` (use these ids
  verbatim in your output).

## How to decide

For each one-sided context, work out the operation from:

- **`left_window` / `right_window`** -- the primary signal. In a transformer
  decoder layer the projections are pinned by their neighbors. Possible likely examples (not strict rules):
  - `(*norm* | rotary_embedding)` -> QKV projection
  - `(paged_attention | *norm*)` -> output projection
  - `(*norm* | act_and_mul)` -> gate/up projection
  - `(act_and_mul | *norm*)` -> down projection
- **kernel name** -- for example, GEMMs, attention, norms, and rotary embeddings often have characteristic names.
- **`sample_input_dims`** and **`kernels_in_run`** -- corroborate a pairing
  (matching shapes / per-layer counts).

**Pairing rule:** two contexts on opposite traces with the same
`(left_window, right_window)` and a compatible operation get the **same** final
name. Pick a short, vendor-neutral name (`qkv_projection`, `output_projection`,
`gate_up_projection`, `down_projection`, ...).

**perf_category caveat:** the regex classifier may tag a vendor kernel as
`Others`. Do **not** require identical `perf_category` to pair
-- rely on the kernel name and the neighbor context. Still never pair operations
that are clearly different types.

## Output file: `kernel_coherence_decisions.json`

```json
{
  "context_renames": {
    "(platform 1 name):6": "qkv_projection",
    "(platform 2 name):6":  "qkv_projection"
  },
  "fallback_remap_a": {
    "(likely gemm name platform 1)_..._(likely gemm signature)_...": "qkv_projection"
  },
  "fallback_remap_b": {
    "(likely gemm name platform 2)_..._(likely gemm signature)_...": "qkv_projection"
  },
  "fallback_remap_regex_a": {
    "^(likely gemm name platform 1)_": "qkv_projection"
  },
  "fallback_remap_regex_b": {
    "^(likely gemm name platform 2)_": "qkv_projection"
  },
  "notes": "optional rationale"
}
```

Resolution order applied per kernel (see `apply`):

1. **`context_renames[context_id]`** -- context-specific; wins. Use it to express
   context-dependent behavior: the *same* first-pass symbol with *different*
   `(left,right)` windows gets *different* finals via *different* context ids.
2. **`fallback_remap_a` / `_b[symbol]`** -- blanket per-symbol remap, applied when
   no context id matched (empty windows at a sequence boundary) or when a symbol
   means the same thing everywhere and you don't need to split it by context.
3. **`fallback_remap_regex_a` / `_b`** -- ordered `{python_regex: final_block}`,
   only tried when no exact `fallback_remap` match was found. Patterns are tried
   in declaration order with `re.search` against the first-pass symbol; first
   match wins. Use this for a whole vendor kernel-name **family** in one rule
   instead of enumerating every exact mangled name -- vendor GEMM solution
   libraries (e.g. `<vendor_gemm_prefix>_*`) emit a distinct, near-unique
   name per tile-shape/tuning variant of the *same* logical op, so listing every
   variant in `fallback_remap_a`/`_b` is impractical and silently leaves most
   instances unmatched. One regex (e.g. `"^<vendor_gemm_prefix>_"`) covers the whole family.
4. Otherwise the first-pass name is kept.

### Guidance

- For a GEMM shape that has one dominant role across the trace, a single
  `fallback_remap` entry is cleaner than many identical `context_renames`; add
  `context_renames` only where a symbol genuinely changes role by context.
- **Prefer a regex fallback over enumerating exact names** when a one-sided
  bucket is one variant of a vendor kernel-name family with many tile-shape/
  tuning variants (vendor GEMM solution libraries in particular). An exact
  `fallback_remap` entry only catches the one variant you saw in the context
  catalog; a `fallback_remap_regex` entry (e.g. `"^<vendor_gemm_prefix>_"`)
  catches every variant of that family, including ones that didn't surface as
  their own one-sided context.
- Prefer merging a one-sided bucket into an **existing shared** symbol when the
  evidence supports it (e.g. an attention split-K/reduce kernel -> the shared
  `paged_attention` bucket).
- When you introduce a new bucket for a paired concept, use the **same** name on
  both traces so the condensed sets align.
- **Leave genuine one-offs alone.** Pre/post-layer, framework-specific setup
  kernels (index build, dtype copies, prefix-scan) have no counterpart; do not
  force them into a shared bucket. Residual one-sided singletons are acceptable.

## Apply + verify

The orchestrator runs:

```bash
<prefix> python3 TraceLens/Agent/Analysis/semantic_analyses/kernel_coherence.py apply \
    --context <output_dir>/work/kernel_coherence_context.json \
    --decisions <output_dir>/work/kernel_coherence_decisions.json \
    --audit-csv-a <output_dir>/work/per_kernel_final_<name_a>.csv \
    --audit-csv-b <output_dir>/work/per_kernel_final_<name_b>.csv
```

Rewrites `semantic_block` on both label files, writes per-kernel audit CSVs, and
prints the residual one-sided condensed symbols. If meaningful (non-singleton)
symbols remain one-sided, revise the decisions and re-run; residual pre/post
singletons may be accepted.

## Verify

The orchestrator validates `kernel_coherence_decisions.json`
(`validate_kernel_coherence_decisions`) before applying and returns any errors
for you to fix and re-emit: `context_renames` keys must be `context_catalog`
ids, `fallback_remap_a` / `_b` keys must be one-sided symbols from the context,
and every `fallback_remap_regex_a` / `_b` key must compile as a regex. **Prefer
a `fallback_remap_regex` family rule over many long exact `fallback_remap`
keys** (see above) -- one regex is shorter and catches every tile-shape variant.

## Return Value

Return: `status`, `pairs_created` (new shared names used on both sides),
`context_renames`, `fallback_a`, `fallback_b`, `fallback_regex_a`, `fallback_regex_b`,
`shared_blocks_after`, and the residual one-sided symbols with a one-line reason
for leaving each.
