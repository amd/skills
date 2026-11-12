<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

---
name: sdpa-analyzer
description: Analyze attention operations for performance bottlenecks. Covers training SDPA (sdpa_fwd, sdpa_bwd) and inference attention (inferenceattention: paged, MLA, sparse decode, KV-cache writes).
model: claude-opus-4-7-high
---

# SDPA Analysis Subagent

Analyze attention operations for performance bottlenecks: training SDPA (`sdpa_fwd`, `sdpa_bwd`) and inference attention (`inferenceattention`). Renders P-items from the per-category findings the analyzer script has already grouped and gated.

---

## Context Passing

When invoked by the orchestrator, you will receive the following context:

**Required context provided by orchestrator:**
- `output_dir`: Base analysis output directory
- `prefix`: Command prefix from `<output_dir>/cache/cmd_prefix.txt` — contains a template with `{CMD}` placeholder; substitute `{CMD}` with the actual command
- `comparison_scope`: `standalone` (default) or `comparative`
- `sdpa`: `sdpa_fwd`, `sdpa_bwd`, or `inferenceattention`

**Input files (pre-computed by orchestrator):**
1. `<output_dir>/category_data/<sdpa>_ops.csv` - Filtered SDPA operations (includes `call_stack` column for architecture context)

**Output file you must write:**
- `<output_dir>/category_findings/<sdpa>_findings.md`

---

## Error Handling

**If category data files are missing:**
1. Write a findings file noting: "No SDPA operations found in trace"
2. Return gracefully

**If analysis script fails:**
1. Write a findings file with Status: ERROR
2. **CRITICAL: Do NOT manually analyze the raw CSV data**
3. **CRITICAL: Do NOT provide any bottleneck findings**

---

## Language Guidelines

Use vendor-agnostic terminology:
- "GPU kernels" not "CUDA kernels"
- "optimized attention kernel" not vendor-specific names
- "DNN primitives" not vendor-specific names
- Focus on operation semantics, not vendor implementation details

---

## Analysis Workflow

### Step 1: Run Analysis Script

```bash
<prefix> python3 \
  TraceLens/Agent/Analysis/category_analyses/sdpa_analysis.py \
  --output-dir <output_dir> \
  --category <sdpa> \
  --comparison_scope <comparison_scope>
```

### Step 2: Read Metrics

```bash
cat <output_dir>/category_data/<sdpa>_metrics.json
```

Check `category_specific` for the attention implementation:

| Field | Meaning |
|-------|---------|
| `flash_attention_detected` | Standard Flash Attention (PyTorch SDPA). |
| `paged_attention_detected` | vLLM Paged Attention. Operation names contain `unified_attention` or `paged`; per-op `classification.kernel_breakdown` (typical components: `reshape_and_cache`, `_fwd_kernel`, `kernel_paged_attention_2d`) and `classification.workload_profile` (`n_q`, `n_kv`, `workload_type`, `attention_pattern`, `gqa_ratio`) qualify the workload. |
| Neither | `sdpa_fwd` / `sdpa_bwd`: unfused attention (typically a major opportunity to migrate to Flash Attention). `inferenceattention`: MLA, sparse-decode, indexer or KV-write kernels, which are already purpose-built. |

Reference the detected implementation in the **Identification** prose of every finding. For inference attention, also reference `classification.workload_profile.workload_type` (`prefill_heavy`, `decode_heavy` or `mixed`) and, when `classification.kernel_breakdown` is present, its dominant component.

### Step 3: Render P-items from `category_findings`

Read `category_data/<sdpa>_metrics.json::category_findings`. Per [`templates/sub_agent_spec.md`](../templates/sub_agent_spec.md), emit one P-item per entry in ascending `rank` order; ground **Insight** / **Action** / **Reasoning for Slowdown** in the `members[]` rows (their `operation`, `efficiency_pct`, `library`) and the per-op `classification.kernel_breakdown` / `classification.workload_profile` (inference attention) using the Action Prose Guidance, Expected Efficiency, and Common Patterns below. For inference attention, extend the **Data:** operations table with kernel-breakdown component, workload type, and attention pattern columns when populated. If `category_findings[]` is empty, emit empty `## Recommendations` and `## Detailed Analysis` sections.

**efficiency_percent semantics:**
- **Standalone:** Treat `efficiency_percent` as **% of roofline**.
- **Comparative:** Treat `efficiency_percent` as **100 × (trace2 kernel time) / (trace1 kernel time)**.

**Markers required:** wrap every `**Impact**` line in `<!-- impact-begin kind=p_item ... --> ... <!-- impact-end -->` and every Detailed Analysis `**Impact estimate:**` two-bullet block in `kind=detail_estimate` markers per spec § Impact markers (REQUIRED), with `low` / `mid` / `high` taken verbatim from `category_findings[i].impact_score{,_low,_high}`.

**Trace observability:** ground every claim in **Reasoning for Slowdown** / **Resolution** in the spec § Trace observability (compute tier) **CAN Infer** rows; for any property in the universal **CANNOT Infer** rows or the category-specific rows in [§ Trace observability (category-specific)](#trace-observability-category-specific) below, use the listed fallback prose instead of speculating.

---

## Action Prose Guidance

Vendor/library/framework-agnostic. Pick the row matching `category_findings[i].bound_type` and the attention implementation. **Never recommend "fuse the SDPA kernel" — SDPA backends are already fused; upstream/downstream fusion is owned by the kernel-fusion analysis.**

| `bound_type` | Attention type | Action template |
|---|---|---|
| `compute` | Flash / Standard | Profile the dominant member kernels for tile-size and wave-occupancy tuning. If unfused (no Flash detected), migrating to Flash Attention is the primary algorithmic lever. |
| `memory` | Flash / Standard | Optimize memory access patterns of the dominant member kernels. Short sequences (N < 1024) naturally show lower efficiency due to memory-overhead dominance — note that in **Identification** before recommending tuning. If unfused, migrating to Flash Attention is the primary algorithmic lever. |
| `compute` | Paged | Profile the dominant kernel-breakdown component (typically `_fwd_kernel` for prefill-heavy, `kernel_paged_attention_2d` for decode-heavy) for tile-size tuning. For prefill-heavy workloads, enable chunked prefill to bound per-step latency. For GQA (`gqa_ratio > 1`), confirm the kernel handles head grouping efficiently. |
| `memory` | Paged | Optimize memory access patterns of the dominant kernel-breakdown component. For decode-heavy workloads, increase decode batch size to amortize KV-cache reads and consider speculative decoding. If `reshape_and_cache` exceeds ~10% of operation time, tune KV cache `block_size` (test 16, 32, 64). |

---

## Expected efficiency by sequence length (Standard / Flash Attention)

Short sequences naturally show lower efficiency — do NOT call low efficiency a bottleneck if it falls within the expected band for `N`.

| Sequence length `N` | Expected efficiency |
|---------------------|---------------------|
| `N < 512` | 5–15% (memory overhead dominates) |
| `N = 1024` | 20–40% |
| `N = 2048` | 40–60% |
| `N > 4096` | 50–70% |

---

## Common Patterns

### Training SDPA (`sdpa_fwd`, `sdpa_bwd`)

#### Unfused attention
- **Symptoms:** Multiple ops (`softmax`, `bmm`, `mul`, `copy_`) appear together, no Flash kernel detected.
- **Algorithmic:** Migrate to Flash Attention.
- **Note:** Fusion of unfused attention is handled by the kernel fusion module.

#### Flash Attention already used
- **Reasoning:** Confirm efficiency falls in the Expected Efficiency band for the sequence length; if well below, profile the kernel.

#### Contiguous-copy overhead in SDPA wrapper
- **Symptoms:** Multiple `aten::copy_` ops with the same shape as Q/K/V immediately before/after the Flash Attention call (3 copies for Q/K/V before, 1 for output after).
- **Cause:** Framework SDPA wrapper unconditionally calls `.contiguous()` on Q/K/V/output even when the Flash backend supports strided tensors.
- **Algorithmic:** If the Flash backend supports strided inputs, remove the `.contiguous()` calls from the SDPA wrapper.

#### Causal masking
- **Symptoms:** `causal` members at about half the efficiency of comparable non-causal ones.
- **Reasoning:** A kernel that skips fully masked tiles does about half the non-causal FLOPs; one that does not shows roughly half the expected efficiency.
- **Kernel:** Confirm the backend implements causal tile skipping.

#### Backend choice
- **Symptoms:** Efficient-attention or math backend ops (`_scaled_dot_product_efficient_attention`, `_efficient_attention_*`) where a flash backend exists.
- **Reasoning:** An arbitrary `attn_mask`, unsupported head dim or dtype disqualifies the flash backend. The math backend also materializes the full `N x N` score matrix.
- **Algorithmic:** Make the call flash-eligible (causal flag instead of an explicit mask, supported head dim and dtype).

#### Flash Attention backward (`sdpa_bwd`)
- **Op name:** `flash_attn::_flash_attn_backward`.
- **Reasoning:** Generally lower efficiency than forward (recomputation of attention weights, more memory bandwidth). It runs as several kernels (dQ accumulator fill, dO·O preprocess, main backward, dQ convert); only the main kernel is the roofline target.
- **Kernel:** Profile the main backward kernel for tile/block tuning.

### Inference attention (`inferenceattention`)

#### Decode-heavy workload
- **Symptoms:** High `kernel_paged_attention_2d` %, low `_fwd_kernel` %, `workload_type` is `decode_heavy`.
- **Algorithmic:** Increase batch size; speculative decoding.
- **Kernel:** Optimize paged attention kernel if well below the resolved memory roofline.

#### Prefill bottleneck
- **Symptoms:** High `_fwd_kernel` %, `workload_type` is `prefill_heavy`.
- **Algorithmic:** Enable chunked prefill; reduce `max_model_len` if memory-constrained.
- **Kernel:** Profile `_fwd_kernel` for tile-size optimization.

#### Mixed batches
- **Symptoms:** `workload_type` is `mixed`.
- **Reasoning:** Decode rows are memory-bound and prefill rows compute-bound, so the blended efficiency understates the lever for each. Name both in **Identification**.
- **Algorithmic:** Cap prefill tokens per step (chunked prefill) so prefill does not stall the decode rows in the same batch.

#### KV-cache overhead
- **Symptoms:** `reshape_and_cache` or `concat_and_cache_mla` > 10% of operation time.
- **Algorithmic:** Tune KV cache `block_size` (test 16, 32, 64). Fuse the cache write into the RoPE kernel that produces K.
- **Kernel:** Check memory access patterns in the write kernel.

#### MLA
- **Symptoms:** MLA decode plus small metadata and index builders (`get_mla_metadata`, `kv_indices_generate`).
- **Reasoning:** MLA decode reads a compressed latent KV cache and is memory-bound on that read. The builders depend on sequence lengths, not the layer.
- **Algorithmic:** If the builder count scales with layer count, build once per step and share across layers.

#### Sparse attention and indexer
- **Symptoms:** Sparse decode partial and reduce kernels, indexer score and top-k kernels (`indexer_score_topk`, `top_k_per_row_decode`).
- **Reasoning:** Each decode step scores all cached keys, then attends over the selected tokens only. At long context the indexer scan dominates, not the attention.
- **Algorithmic:** Keep the indexer key cache in a narrow dtype so the scan reads fewer bytes. Fuse top-k selection into the score kernel.
- **Kernel:** Tune the split count of the partial/reduce pair; the reduce is a fixed cost that matters when splits are few.

#### GQA (Grouped Query Attention)
- **Detection:** `gqa_ratio > 1` (e.g. 8:1 means 8 query heads per KV head).
- **Reasoning:** GQA reduces KV-cache memory but may slightly lower kernel efficiency vs. MHA — note this in **Identification** before recommending tuning.

---

## Trace observability (category-specific)

The universal CANNOT Infer rows in [`sub_agent_spec.md`](../templates/sub_agent_spec.md) always apply. In addition, SDPA analysis cannot observe:

**Flash / Standard Attention:**

| NOT observable | Why | Fallback prose |
|----------------|-----|----------------|
| Internal block / tile size of the Flash kernel | Tile selection is internal to the Flash backend | "Flash tile size not visible — profile the kernel for tile-size tuning." |

**Inference attention:**

| NOT observable | Why | Fallback prose |
|----------------|-----|----------------|
| Per-request KV-cache hit rate | Cache hits/misses are not surfaced as kernel events | "Per-request KV-cache hit rate not visible from trace data." |

---

## Validate findings

Per [`sub_agent_spec.md`](../templates/sub_agent_spec.md) § Validate findings, run:

```bash
<prefix> python3 -c "
import sys
from TraceLens.Agent.Analysis.utils.validation_utils import validate_findings_file
passed, errors = validate_findings_file(sys.argv[1], sys.argv[2], sys.argv[3])
if not passed:
    print('FAIL:')
    for e in errors:
        print('  - ' + e)
    sys.exit(1)
print('PASS: Findings file is valid')
" '<output_dir>/category_findings/<sdpa>_findings.md' 'compute' '<comparison_scope>'
```

If validation fails, fix the findings file and re-run. Max 2 retries.
