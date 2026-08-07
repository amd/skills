# AMD Skill Finder Routing and Source Policy

Use this reference for ambiguous discovery, source-registry maintenance, or
deciding whether source material is ready to become an installable skill.

## Contents

- [Search tiers](#search-tiers)
- [Stable routing lanes](#stable-routing-lanes)
- [Matching rules](#matching-rules)
- [Paired upstream and AMD repositories](#paired-upstream-and-amd-repositories)
- [Result provenance](#result-provenance)
- [Promoting source material into a skill](#promoting-source-material-into-a-skill)
- [Registry maintenance](#registry-maintenance)

## Search tiers

Search in this order:

0. Skills installed in the current agent environment.
1. Published skills in the live `amd/skills` catalog.
2. Official repositories owned by AMD, ROCm, or AMD-AGI.
3. Reviewed upstream repositories in `data/sources.json`.
4. General GitHub results only when explicitly enabled.

Higher trust does not erase relevance: an exact upstream API answer can be more
useful than an unrelated AMD document. When two sources answer the same AMD
question equally well, rank the AMD source first.

## Stable routing lanes

- System platform: ROCm installation, compatibility, releases, runtime, driver,
  HIP, and system-management questions.
- GPU development: HIP porting, Triton and TileLang kernels, AITER, ATOM,
  compilation, profiling, tracing, and debugging.
- Training: PyTorch, JAX, TorchTitan, RCCL, FSDP, tensor/pipeline parallelism,
  distributed checkpointing, and low-precision training.
- Inference: vLLM, vLLM recipes, SGLang, model serving, attention, MoE, prefix
  caching, speculative decoding, and quantization with Quark.
- KV-cache and networking: MORI, LMCache, Mooncake, NIXL, RDMA, UCX,
  prefill/decode disaggregation, expert parallelism, and cache transfer.
- Post-training: Miles, VERL, VIME, RLHF, GRPO, PPO, reward models, rollout
  generation, and vLLM-backed RL scaling.
- Performance optimization: Hyperloom, TraceLens, Magpie, ROCprofiler SDK,
  kernel bottlenecks, roofline analysis, and validated optimization campaigns.
- Local AI: Lemonade Server, Ryzen AI, local chat, image generation,
  transcription, embeddings, reranking, and cloud-to-local application ports.

Route by the user's task verb and artifact, not merely a product token. For
example, "profile a Triton kernel" should include the profiling lane as well as
the Triton project.

## Matching rules

For very large repositories such as PyTorch, JAX, and Triton, require an AMD
signal for implicit broad searches: AMD, ROCm, HIP, an AMD product, an Instinct
model, or a `gfx` architecture. Relax that requirement when the user explicitly
names the project and asks an upstream API question.

Treat ordinary uses of these words as negative signals unless AMD or accelerated
computing is also present:

- `route`: HTTP or file routing.
- `optimize`: CSS, SQL, bundle size, or generic refactoring.
- `deploy`: generic web hosting, Kubernetes, or CI/CD.
- `AI`, `data`, `training`, or `infrastructure`: generic uses without an AMD,
  GPU, ROCm, or distinctive project signal.

Do not send a full user prompt to an external search service when a short,
non-sensitive capability phrase is sufficient.

## Paired upstream and AMD repositories

Represent a fork pair with one project identifier and ordered repositories:

```json
{
  "id": "triton",
  "repositories": [
    {"repo": "ROCm/triton", "role": "amd-fork", "tier": "amd-official"},
    {"repo": "triton-lang/triton", "role": "upstream", "tier": "reviewed-upstream"}
  ]
}
```

Use AMD forks for ROCm patches, CI, architecture support, and AMD-specific
performance behavior. Use upstream for public interfaces and cross-platform
design. Cite the exact repository and file used; do not silently blend them.

## Result provenance

Only a validated catalog folder is `installable_skill`. A source repository can
contain a `SKILL.md` without being installable from `amd/skills`; label that
`embedded_skill`. Hyperloom currently provides a useful example of embedded
product-owned agent guidance.

Issues and pull requests are current but less stable than versioned docs or
code. Label them discussions and verify whether the change merged before using
them as instructions. Prefer commit-pinned live-search URLs when available.

The repository `ai-dynamo/dynamo` is explicitly excluded. The separately
requested `ai-dynamo/nixl` repository remains a reviewed NIXL source; exclusion
is repository-specific, not organization-wide.

## Promoting source material into a skill

Recommend creating or porting a skill only when the workflow has:

1. A single clear outcome.
2. Predictable inputs and outputs.
3. Repeated procedural value beyond ordinary documentation lookup.
4. Safe, deterministic scripts for fragile operations.
5. Maintainer ownership and an acceptable redistribution license.
6. AMD-specific validation, including supported hardware and ROCm versions.
7. Positive and negative routing tests plus an end-to-end behavioral test.

Review embedded instructions for assumptions about repository layout, internal
tools, credentials, model-specific paths, or destructive actions before porting.
Do not copy third-party text or code unless its license permits redistribution.

## Registry maintenance

Edit `data/sources.json`, not `.github/scripts/sources.yml`, to add discovery
sources. The latter is reserved for repositories that actually federate
installable skill folders into the AMD catalog.

For every registry change:

1. Verify the canonical GitHub owner and repository.
2. Record whether the source is AMD official or reviewed upstream.
3. Group forks under one logical project.
4. Add task-oriented keywords and narrow path hints.
5. Confirm excluded repositories cannot appear.
6. Add or update deterministic routing tests.
7. Run the finder offline and live on representative queries.
