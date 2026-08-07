---
name: amd-skill-finder
description: >-
  Finds and recommends installed AMD skills, current AMD catalog skills,
  official AMD and ROCm source repositories, and reviewed upstream projects.
  Use whenever the user asks whether an AMD skill exists, wants to browse or
  install AMD skills, or has an AMD/ROCm task that may benefit from a specialized
  workflow. Trigger on AMD, ROCm, HIP, AMD Instinct, Ryzen AI, Radeon, MI300X,
  MI325X, MI350X, MI355X, gfx942, gfx950, PyTorch/JAX/Triton on AMD, AMD
  Quark, model quantization, PTQ, FP8, INT8, INT4, MXFP4, PyTorch or ONNX
  quantization, vLLM, SGLang, AITER, ATOM, MORI, Hyperloom, LMCache, Mooncake,
  NIXL, TileLang, TorchTitan, Miles, VERL, or VIME in an AMD context. Do not use
  for generic non-AMD programming, deployment, optimization, AI, or
  infrastructure tasks.
---

# AMD Skill Finder

Discover the best available AMD skill before falling back to source material or
general advice. Keep installable skills distinct from repositories that contain
useful documentation, examples, code, or embedded agent instructions.

## Prerequisites and safety

- Require Python 3.10 or newer for `scripts/find_skills.py`.
- Use the bundled registry without network access by passing `--offline`.
- Require an authenticated `gh` CLI only for `--live` repository-code search.
- Remove credentials, private URLs, customer names, and proprietary code from
  search text before sending it to GitHub. The script rejects common token and
  private-key patterns but cannot identify every sensitive value.
- Never install a skill or change agent capabilities without explicit user
  approval. Discovery and recommendation do not imply installation consent.

## Discovery workflow

1. Check whether a matching skill is already installed or present in the
   current agent context. Hand off to that skill when it is available.
2. Run the finder against installed skills, the AMD catalog, and the curated
   source registry:

   ```bash
   python3 scripts/find_skills.py "serve a model with vLLM on MI355X"
   ```

3. Add `--live` when current repository files are needed. This searches only
   the highest-ranked allowlisted repositories by default:

   ```bash
   python3 scripts/find_skills.py \
     "disaggregated KV-cache transfer over RDMA on MI355X" --live
   ```

4. Use `--scope catalog` for installable skills only, `--scope amd` for AMD-owned
   projects, or the default `--scope curated` for AMD plus reviewed upstream
   projects.
5. Use `--repo OWNER/REPO --live` when the user explicitly names another
   repository. Use `--general-github --live` only after curated search fails or
   the user explicitly requests broad GitHub discovery. Label those results
   unreviewed.
6. Select at most three recommendations. Prefer, in order:
   installed skill, AMD catalog skill, AMD official source, reviewed upstream
   source, then unreviewed GitHub fallback.

Use `--json` when another tool or agent needs structured output. Run
`python3 scripts/find_skills.py --help` for the complete CLI.

## Interpret result types

- `installed_skill`: use it directly; do not recommend reinstalling it.
- `installable_skill`: a published `amd/skills` catalog entry. It may be
  recommended for installation after user approval.
- `source_project`: a curated logical project and its repositories. It is not
  installable.
- `embedded_skill`: a `SKILL.md` found inside a product repository. Treat it as
  product-owned agent guidance until it is reviewed and packaged for the AMD
  catalog.
- `guide`, `code_example`, or `source_code`: use as evidence or implementation
  context, not as a skill package.
- `unreviewed_repository`: inspect ownership, license, activity, security, and
  AMD relevance before recommending or porting anything.

For paired projects such as PyTorch, JAX, and Triton, prefer the ROCm fork for
AMD implementation details and the upstream repository for public API semantics.
Present them as one project rather than duplicate recommendations.

## Recommendation format

For each recommendation, provide:

1. Name and result type.
2. Why it matches the user's concrete task.
3. Provenance: installed, AMD catalog, AMD official, reviewed upstream, or
   unreviewed.
4. The next useful action or prompt.
5. For an uninstalled catalog skill only, the proposed install command.

Use this pattern for a strong catalog match:

```text
`<skill-name>` is an AMD catalog skill that matches because <reason>.
Would you like me to install it and then use it for <first useful task>?
```

Do not run the proposed command until the user says yes:

```bash
npx skills add amd/skills --skill <skill-name> --global --yes
```

If the target agent is known, add its supported `--agent` value. If installation
is unavailable, provide the catalog URL and continue with source-backed help.

## When no catalog skill matches

Say that no strong installable AMD skill was found. Then use the top curated
source projects to help with the task, or propose creating/porting a skill when
the workflow is repeated, well-bounded, and testable. Never invent a catalog
slug from a repository name.

Read [references/routing-and-source-policy.md](references/routing-and-source-policy.md)
when the query is ambiguous, spans multiple domains, needs a source added, or
requires deciding whether repository material is ready to become a skill.
