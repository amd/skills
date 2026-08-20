# Skill Best Practices

How to write a skill an agent actually reaches for and follows. None of this is
enforced by CI, but a reviewer will raise it. For the rules that block a merge,
see [skill-requirements.md](skill-requirements.md).

Upstream references: Anthropic's [Skill authoring best practices](https://docs.anthropic.com/en/docs/agents-and-tools/agent-skills/best-practices)
and [The Complete Guide to Building Skills for Claude](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf).

## Is this a good fit for a skill?

Skills earn their keep on repeated, opinionated workflows. Check the task has
these properties before writing one:

- **Single clear outcome.** One job, one measurable success condition. If you can't state success in one sentence, split the skill.
- **Well-defined inputs and outputs.** The agent should know what to ask for and what to produce.
- **Tool-bounded.** Uses only the tools and data it truly needs.
- **Deterministic where possible.** Lean on scripts for the deterministic parts.
- **Short execution path.** Long workflows belong in a checklist or split skills.
- **Recoverable failures.** Exits cleanly with a useful message, never mid-state.
- **Context-light.** Works from the user's prompt and the skill body alone.
- **Composable.** Plays well with other skills loaded at the same time.

If the task fails several of these, it is probably documentation, a runbook, or
a one-off prompt, not a skill.

## Write the description for the goal, not the mechanics

The `description` is the only part of the skill always loaded into context. The
agent uses it to decide *whether* to load the rest. Treat it as a routing
signal, not marketing copy. The agent matches against what the user is trying
to *achieve*; which library or container you use belongs in the body.

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

- **Third person.** The description is injected into the system prompt. Use *"Ports CUDA kernels..."*, not *"I help you port..."*.
- **State WHAT and WHEN.** What the skill produces, and when the agent should reach for it.
- **Include the trigger surface.** Product names, file extensions, API names, error messages. Missing triggers cause under-triggering.
- **Add negative triggers** when boundaries are easily crossed: *"Do not use for system-wide installs; see X instead."*
- **Be pushy when the use case is ambiguous.** Better to err toward being invoked than to be silently skipped.

### Naming

Prefer gerund or action-oriented names tied to the outcome.

- Good: `porting-cuda-to-hip`, `tuning-mi300x`, `picking-rocm-container`
- Avoid: `helper`, `utils`, `gpu-stuff`

## The SKILL.md body

The body loads only when the description matches. Once loaded, every token
competes with conversation history and other context.

**Be concise.** Assume the agent already knows general programming, common
libraries, and standard CLI tools. Only add what it would *otherwise guess
wrong*. Challenge each paragraph: *"does this justify its tokens?"*

**Match degrees of freedom to the task.** Mismatched freedom is a top cause of
skills that frustrate users — database migrations want low freedom, code review
wants high.

| Freedom | Use when | Form |
| --- | --- | --- |
| **Low** | Operation is fragile, exact sequence matters | Specific scripts, exact commands |
| **Medium** | Preferred pattern with acceptable variation | Pseudocode, parameterized templates |
| **High** | Multiple valid approaches, context-dependent | Text instructions, heuristics |

**Use progressive disclosure, one level deep.** Link from `SKILL.md` directly to
reference files. Do not chain references through intermediate files; agents may
only partially read deeply nested content.

```
skill-name/
  SKILL.md          # overview, quick start, links
  skill-card.md     # governance card
  reference.md      # full API / flag reference
  examples.md       # worked examples
  scripts/          # executable utilities
```

For reference files longer than ~100 lines, put a table of contents at the top
so the agent sees the full scope even when it previews with `head`.

**Provide a default, not a menu.** One opinionated path with a single named
escape hatch beats a buffet.

```
Bad:  "You can use pdfplumber, pypdf, PyMuPDF, or pdf2image..."
Good: "Use pdfplumber for text extraction. For scanned PDFs that need OCR,
       use pdf2image with pytesseract instead."
```

**Be consistent with terminology.** Pick one term per concept. Mixing
*"endpoint"*, *"URL"*, *"route"*, and *"path"* makes instructions harder to follow.

**Avoid time-sensitive content.** *"Before August 2025, use the old API"* becomes
wrong on its own schedule. Write a `## Current method` section and tuck legacy
guidance into a collapsed `## Old patterns` block.

**Use forward slashes everywhere.** `scripts/helper.py`, never
`scripts\helper.py`. Backslashes break on Linux and macOS.

## Scripts and tools

Pre-made scripts beat generated code: more reliable, fewer tokens, consistent
across runs.

- **Solve, don't punt.** Handle expected error cases inside the script. Don't return a stack trace and hope the agent figures it out.
- **No voodoo constants.** If a timeout is 47 seconds, say *why* in a comment. If you don't know the right value, neither will the agent.
- **State dependencies.** List required packages and versions in `SKILL.md`. Don't assume `rocm-smi` or `hipify-perl` is on the path.
- **Make execution intent explicit.** Write *"Run `analyze.py`"* (execute) or *"See `analyze.py` for the algorithm"* (read), never both.
- **Use fully qualified MCP tool names.** `ServerName:tool_name`. Bare names fail when multiple servers are registered.

## AMD-specific guidance

- **State prerequisites up front.** ROCm version, kernel version, GPU architecture (`gfx942`, `gfx90a`, `gfx1100`, ...), container image, driver branch.
- **Pin to a known-good container when one exists.** Don't make the agent guess between `rocm/pytorch`, `rocm/dev-ubuntu-22.04`, etc.
- **Call out silent footguns.** Environment variables that change behavior without warning (`HSA_OVERRIDE_GFX_VERSION`, `PYTORCH_HIP_ALLOC_CONF`, `HIP_VISIBLE_DEVICES`) deserve their own section.
- **Note unsupported architectures explicitly.** A skill that only works on CDNA should say so, not fail mysteriously on RDNA.

## Iterate against real usage

0. **Prototype first.** Get the agent through one hard, real instance of the task *before* writing the skill, then extract the winning approach.
1. **Test routing.** Run a fresh agent against prompts that should and shouldn't trigger the skill. Write the ones that taught you something into `evals/evals.json` so they keep being checked; see [evals.md](evals.md).
2. **Run it end-to-end on a real machine.** Watch where the agent hesitates, asks unnecessary questions, or goes off-script.
3. **Fold the observations back in**, usually as a sharper description, a clearer default, or a missing prerequisite, rather than more prose.
