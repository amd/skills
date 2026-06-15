# Contributing to AMD Skills

This guide covers everything you need to ship a skill: how to choose a contribution path, what to validate locally, and the writing conventions every AMD skill should follow.

For repository structure and the broader catalog model, see the [README](README.md). For the upstream reference, see Anthropic's [Skill authoring best practices](https://docs.anthropic.com/en/docs/agents-and-tools/agent-skills/best-practices).

We welcome contributions from AMD engineers and selected partners.

## Contribution paths

There are two contribution paths, matching how the catalog is organized.

### Path A: Skills authored in this repository

Best for cross-cutting skills that do not have a natural product home.

1. Copy an existing skill folder under `skills/` as a starting point and rename it.
2. Update the `SKILL.md` frontmatter so the `name` and `description` clearly explain *what* the skill does and *when* an agent should reach for it.
3. Add the supporting scripts, templates, and reference docs your instructions point to. Keep skills focused: one well-scoped task per skill is better than one mega-skill.
4. Add a `skill-card.md` at the skill root with `## Description`, `## Owner`, and `## License` sections. This is the skill's governance card; see [Skill cards](#skill-cards) and [docs/skill-cards.md](docs/skill-cards.md).
5. Register the skill in `.claude-plugin/marketplace.json` with a human-readable description (the marketplace description is for humans browsing the catalog; the `SKILL.md` description is what the agent uses for routing).
6. Regenerate the Cursor manifest so it tracks the new skill:
   ```bash
   ./.github/scripts/publish.sh   # writes .cursor-plugin/marketplace.json
   ```
7. Validate the skill locally before pushing:
   ```bash
   ./.github/scripts/check.sh   # validates every SKILL.md and that manifests are in sync
   ```
8. Open a pull request. The `validate` GitHub Actions workflow runs `./.github/scripts/check.sh` and must pass before merge. See [Validating locally](#validating-locally) for the full set of enforced rules.

### Path B: Skills authored in a product repository

Best for skills that should ship and version with a product (HIP, MIGraphX, Ryzen AI, Lemonade, etc.).

1. Add the skill folder to your product repository; a common location is `.agents/skills/<skill-name>/`.
2. Open a pull request here that adds (or extends) an entry in [`.github/scripts/sources.yml`](.github/scripts/sources.yml) — the master list — naming your repo, a pinned ref, the sub-path that holds skill folders, and your skill's folder name.
3. Once the catalog change merges, dispatch the **Import external skills** workflow from the Actions tab. It shallow-clones your repo at the pinned ref, vendors the skill into `skills/<name>/`, updates `.claude-plugin/marketplace.json`, and opens a follow-up pull request. Validation then runs against the same rules as in-repo skills before merge.

## Is this task a good fit for a skill?

Skills earn their keep on repeated, opinionated workflows. Before writing one, check that the task has these properties:

- **Single clear outcome.** One job, one measurable success condition. If you can't state success in one sentence, split the skill.
- **Well-defined inputs and outputs.** Predictable shape, minimal ambiguity. The agent should know what to ask for and what to produce.
- **Tool-bounded.** Uses only the tools and data it truly needs. Fewer moving parts means fewer ways to fail.
- **Deterministic where possible.** Same input should produce a similar output across runs. Lean on scripts for the deterministic parts.
- **Short execution path.** Few steps, low latency, low token cost. Long workflows belong in a checklist or split skills.
- **Recoverable failures.** Detects errors and either retries or exits cleanly with a useful message, and never leaves the user mid-state.
- **Context-light.** Works from the user's prompt and the skill body. Doesn't require long conversation history or hidden setup.

If the task fails several of these, it is probably documentation, a runbook, or a one-off prompt, not a skill.

## Write the description for the goal, not the mechanics

The `description` is the only part of the skill always loaded into context. The agent uses it to decide *whether* to load the rest. Treat it as a routing signal, not marketing copy.

### Describe the user's goal, not how the skill works

The agent matches descriptions against what the user is trying to *achieve*. Internal mechanics (which library, which container, which API) belong in the body of `SKILL.md`.

```yaml
# Good: names the goal and the trigger surface
description: >-
  Port a CUDA kernel to HIP and flag anything that needs manual review.
  Use when the user wants to run CUDA code on AMD GPUs, mentions hipify,
  HIP, ROCm porting, or asks how to convert a .cu file.

# Bad: describes how the skill works internally
description: >-
  Runs hipify-perl on .cu files, parses the output, and post-processes
  the result with regex rules.
```

### Description checklist

- **Third person.** The description is injected into the system prompt. Use *"Ports CUDA kernels..."*, not *"I help you port..."* or *"You can use this to..."*.
- **State WHAT and WHEN.** What the skill produces, and the situations in which the agent should reach for it.
- **Include the trigger surface.** List the words and phrases a user is likely to say, including product names, file extensions, API names, and error messages. Missing triggers cause under-triggering.
- **Add negative triggers when boundaries are easily crossed.** *"Do not use for system-wide installs; see X instead."*
- **Be pushy when the use case is ambiguous.** It is better to err toward being invoked than to be silently skipped.
- **Stay under ~1024 characters** (the hard cap on Anthropic-compatible runtimes).

### Naming

Use `lowercase-with-hyphens`, max 64 characters, no `anthropic` or `claude` substrings. Prefer gerund or action-oriented names tied to the outcome:

- Good: `porting-cuda-to-hip`, `tuning-mi300x`, `picking-rocm-container`
- Avoid: `helper`, `utils`, `gpu-stuff`

## SKILL.md body

The body loads only when the description matches. Once loaded, every token competes with conversation history and other context.

### Be concise

Assume the agent already knows general programming, common libraries, and standard CLI tools. Only add what it would *otherwise guess wrong*. Challenge each paragraph: *"does this justify its tokens?"*

Keep the body under ~500 lines. Push reference material into sibling files (`reference.md`, `examples.md`, etc.) and link to them from `SKILL.md`.

### Match degrees of freedom to the task

| Freedom | Use when | Form |
| --- | --- | --- |
| **Low** | Operation is fragile, exact sequence matters | Specific scripts, exact commands |
| **Medium** | Preferred pattern with acceptable variation | Pseudocode, parameterized templates |
| **High** | Multiple valid approaches, context-dependent | Text instructions, heuristics |

Database migrations want low freedom. Code review wants high freedom. Mismatched freedom is a top cause of skills that frustrate users.

### Use progressive disclosure, one level deep

Link from `SKILL.md` directly to reference files. Do not chain references through intermediate files because agents may only partially read deeply nested content.

```
skill-name/
  SKILL.md          # overview, quick start, links
  skill-card.md     # governance card (Description, Owner, License)
  reference.md      # full API / flag reference
  examples.md       # worked examples
  scripts/          # executable utilities
```

For reference files longer than ~100 lines, put a table of contents at the top so the agent can see the full scope even when it previews with `head`.

### Provide a default, not a menu

```
Bad:  "You can use pdfplumber, pypdf, PyMuPDF, or pdf2image..."
Good: "Use pdfplumber for text extraction. For scanned PDFs that need OCR,
       use pdf2image with pytesseract instead."
```

One opinionated path with a single named escape hatch beats a buffet.

### Be consistent with terminology

Pick one term per concept and stick with it. Mixing *"endpoint"*, *"URL"*, *"route"*, and *"path"* in the same skill makes instructions harder for the agent to follow.

### Avoid time-sensitive content

`"Before August 2025, use the old API"` becomes wrong on its own schedule. Instead, write a `## Current method` section and tuck legacy guidance into a collapsed `## Old patterns` block.

### Use forward slashes everywhere

`scripts/helper.py`, never `scripts\helper.py`. Forward slashes work on every platform; backslashes break on Linux and macOS.

## Scripts and tools

Pre-made scripts beat generated code: more reliable, fewer tokens, consistent across runs.

- **Solve, don't punt.** Handle expected error cases inside the script. Don't return a stack trace and hope the agent figures it out.
- **No voodoo constants.** If a timeout is 47 seconds, say *why* in a comment. If you don't know the right value, neither will the agent.
- **State dependencies.** List required packages and target versions in `SKILL.md`. Don't assume `rocm-smi`, `hipify-perl`, or any framework is on the path.
- **Make execution intent explicit.** Write *"Run `analyze.py`..."* (execute) or *"See `analyze.py` for the algorithm"* (read), never both.
- **Use fully qualified MCP tool names.** `ServerName:tool_name`, e.g. `BigQuery:bigquery_schema`. Bare names fail when multiple servers are registered.

## Skill cards

Every skill ships a `skill-card.md` at its root: a short, human-facing governance record that tells a reviewer what the skill does, who owns it, and under what license it ships, without reading the source. It is not loaded by the agent.

The AMD card is intentionally minimal. Three required sections, each a `##` heading with non-empty body text:

```markdown
# Skill Card

## Description

<one sentence: what the skill does, for whom>

## Owner

<team or org accountable for maintenance, e.g. AMD>

## License

<SPDX identifier or link, e.g. MIT>
```

The validator fails any skill whose card is missing or whose required sections are absent or empty. For the full guide, examples, and how federated skills get cards, see [docs/skill-cards.md](docs/skill-cards.md).

## AMD-specific guidance

- **State prerequisites up front.** ROCm version, kernel version, GPU architecture (`gfx942`, `gfx90a`, `gfx1100`, ...), container image, driver branch.
- **Pin to a known-good container when one exists.** Don't make the agent guess between `rocm/pytorch`, `rocm/dev-ubuntu-22.04`, etc.
- **Call out silent footguns.** Environment variables that change behavior without warning (`HSA_OVERRIDE_GFX_VERSION`, `PYTORCH_HIP_ALLOC_CONF`, `HIP_VISIBLE_DEVICES`) deserve their own section.
- **Note unsupported architectures explicitly.** A skill that only works on CDNA should say so, not fail mysteriously on RDNA.

## Iterate against real usage

Test the skill the way users will hit it:

1. Run a fresh agent against ~10 prompts that *should* trigger the skill and ~10 that *shouldn't*. The description should route both sets correctly.
2. Run the skill end-to-end on a real machine. Watch where the agent hesitates, asks unnecessary questions, or goes off-script.
3. Bring those observations back into the skill, usually as a sharper description, a clearer default, or a missing prerequisite, rather than adding more prose.

## Pre-publish checklist

- [ ] Description states the user's goal and includes likely trigger phrases
- [ ] Description is third person and under 1024 characters
- [ ] `skill-card.md` exists with non-empty Description, Owner, and License sections
- [ ] Skill name is lowercase-with-hyphens and ties to the outcome
- [ ] `SKILL.md` body is under 500 lines
- [ ] Reference files are linked one level deep from `SKILL.md`
- [ ] One opinionated default per decision, with named escape hatches
- [ ] Consistent terminology throughout
- [ ] No time-sensitive statements outside an `Old patterns` section
- [ ] All paths use forward slashes
- [ ] Scripts handle expected errors and document their constants and dependencies
- [ ] Prerequisites (ROCm version, GPU arch, container, env vars) are stated explicitly
- [ ] Tested end-to-end on the target hardware against real prompts
- [ ] `./.github/scripts/check.sh` passes (CI runs this on every PR)

## Validating locally

The structural rules from this guide (frontmatter shape, name format, description length, and `SKILL.md` body size) are enforced by `.github/scripts/validate_skills.py` and run on every pull request. Run them locally before pushing:

```bash
./.github/scripts/check.sh   # validates every skill and plugin manifests (same command CI runs)
```

The validator checks every skill under `skills/` for:

- a `SKILL.md` file with a valid YAML frontmatter block
- `name`: lowercase-with-hyphens, ≤ 64 characters, no `anthropic` / `claude` substrings, matches the directory name
- `description`: non-empty, ≤ 1024 characters
- `SKILL.md` body: ≤ 500 lines
- `skill-card.md`: present at the skill root with non-empty `## Description`, `## Owner`, and `## License` sections

It also checks the plugin manifests:

- every skill under `skills/` has a matching entry in `.claude-plugin/marketplace.json` (and vice versa), with `source` set to `./skills/<name>` and a non-empty human-readable `description`
- `.cursor-plugin/marketplace.json` is up to date — it mirrors `.claude-plugin/marketplace.json` and pulls shared identity (name, description, version, author) from `plugin-metadata.json` (regenerate with `./.github/scripts/publish.sh`)
