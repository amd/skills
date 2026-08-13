# Contributing to AMD Skills

This guide covers everything you need to ship a skill: how to choose a contribution path, what to validate locally, and the writing conventions every AMD skill should follow.

For repository structure and the broader catalog model, see the [README](README.md). For the upstream reference, see Anthropic's [Skill authoring best practices](https://docs.anthropic.com/en/docs/agents-and-tools/agent-skills/best-practices) and [The Complete Guide to Building Skills for Claude](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf).

We welcome contributions from AMD engineers and selected partners.

## Contribution paths

There are two contribution paths, matching how the catalog is organized.

### Path A: Skills authored in this repository

Best for cross-cutting skills that do not have a natural product home.

1. Copy an existing skill folder under `skills/` as a starting point and rename it.
2. Update the `SKILL.md` frontmatter so the `name` and `description` clearly explain *what* the skill does and *when* an agent should reach for it.
3. Add the supporting scripts, templates, and reference docs your instructions point to. Keep skills focused: one well-scoped task per skill is better than one mega-skill.
4. Add a `skill-card.md` at the skill root with `## Description`, `## Owner`, and `## License` sections. This is the skill's governance card; see [Skill cards](#skill-cards) and [docs/skill-cards.md](docs/skill-cards.md).
5. Add an eval dataset at `skills/<name>/evals/evals.json` — copy [`eval/TEMPLATE.json`](eval/TEMPLATE.json). Every skill needs at least three prompts that should trigger it and two that should not; CI rejects a skill without them. See [Evals](#evals).
6. Publish the skill by adding a `./skills/<name>` entry to the `skills` array of the single `amd-skills` plugin in [`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json). All published skills ship together in that one plugin; a skill left out of the array stays unpublished. (The `SKILL.md` description is what the agent uses for routing; the plugin's catalog description is a bundle-level blurb for humans.)
7. Regenerate the derived manifests so they track the marketplace:
   ```bash
   ./.github/scripts/publish.sh   # writes .cursor-plugin/ and .codex-plugin/ + .agents/plugins/
   ```
8. Validate the skill locally before pushing:
   ```bash
   ./.github/scripts/check.sh   # validates every SKILL.md and dataset, and that manifests are in sync
   ```
9. Open a pull request. The `validate` workflow runs `./.github/scripts/check.sh` and must pass before merge; the `evals` workflow runs your prompts against a real agent. See [Validating locally](#validating-locally) for the full set of enforced rules.

### Path B: Skills authored in a product repository (federation)

Best for skills that should ship and version with a product (HIP, MIGraphX, Ryzen AI, Lemonade, etc.). Your repo stays the source of truth and the catalog vendors a pinned copy. This is called **federation**, and the full walkthrough lives in [docs/federating-your-repo.md](docs/federating-your-repo.md).

The short version:

1. Keep each skill as a folder with a valid `SKILL.md` and `skill-card.md` in your AMD-owned repo (a common location is `skills/` or `.agents/skills/<skill-name>/`).
2. Add (or extend) an entry in [`.github/scripts/sources.yml`](.github/scripts/sources.yml) — the master list — naming your repo, a pinned ref, the sub-path that holds skill folders, and each skill's folder name.
3. Vendor the skills and refresh the manifests locally, then open a pull request here for review. Validation runs against the same rules as in-repo skills before merge.

See [docs/federating-your-repo.md](docs/federating-your-repo.md) for the exact `sources.yml` schema, the import commands, and how to run the catalog's checks in your own repo.

## Is this task a good fit for a skill?

Skills earn their keep on repeated, opinionated workflows. Before writing one, check that the task has these properties:

- **Single clear outcome.** One job, one measurable success condition. If you can't state success in one sentence, split the skill.
- **Well-defined inputs and outputs.** Predictable shape, minimal ambiguity. The agent should know what to ask for and what to produce.
- **Tool-bounded.** Uses only the tools and data it truly needs. Fewer moving parts means fewer ways to fail.
- **Deterministic where possible.** Same input should produce a similar output across runs. Lean on scripts for the deterministic parts.
- **Short execution path.** Few steps, low latency, low token cost. Long workflows belong in a checklist or split skills.
- **Recoverable failures.** Detects errors and either retries or exits cleanly with a useful message, and never leaves the user mid-state.
- **Context-light.** Works from the user's prompt and the skill body. Doesn't require long conversation history or hidden setup.
- **Composable.** Plays well with other skills loaded at the same time. Don't assume yours is the only capability available.

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

0. Prototype first. Get the agent through one hard, real instance of the task *before* writing the skill, then extract the winning approach. This gives faster signal than authoring against an untested idea.
1. Run a fresh agent against prompts that *should* trigger the skill and prompts that *shouldn't*. The description should route both sets correctly. Write the ones that taught you something into `evals/evals.json` so they keep being checked; see [Evals](#evals).
2. Run the skill end-to-end on a real machine. Watch where the agent hesitates, asks unnecessary questions, or goes off-script.
3. Bring those observations back into the skill, usually as a sharper description, a clearer default, or a missing prerequisite, rather than adding more prose.

## Evals

Structural validation proves a skill is *well-formed*; evals prove it *works*.
There are two questions to answer, and they come in order:

1. **Routing** — with the published bundle installed side by side, does the
   agent pick yours? This is where most skills actually fail, and you cannot
   test it alone: a skill tested by itself will happily answer prompts that
   belong to its neighbour.
2. **Behavior** — once yours is picked, does it do the job?

You write **one file** and both questions are graded from it:
`skills/<your-skill>/evals/evals.json`. Copy
[`eval/TEMPLATE.json`](eval/TEMPLATE.json) and start there.

### The file

One `evaluations` array. Each entry is a prompt plus `skill_should_trigger`:
yes if your skill should wake up for it, no if nothing should.

```json
{
  "evaluations": [
    {
      "id": "epyc-vllm-zentorch",
      "skill_should_trigger": true,
      "prompt": "Serve Llama 3.1 8B on this box with vLLM and zentorch."
    },
    {
      "id": "vllm-on-nvidia",
      "skill_should_trigger": false,
      "prompt": "Serve Llama 3.1 70B with vLLM on my NVIDIA H100 cluster."
    }
  ]
}
```

No evaluation names a skill: the folder says which skill this is, and
`skill_should_trigger` refers to it. The flag is required, because a default
would put the routing expectation back into a field nobody wrote.

A `false` evaluation is exactly what you see above: an id, a prompt, and the
flag, plus an optional `note`. It answers one question — did anything fire? —
and no skill is ever loaded for it, so there is no behavior phase to assert
anything about. Writing `unexpected_behavior` there is an error, not a no-op.

If a prompt should trigger *someone else's* skill, put it in that skill's
dataset as `true` rather than in yours as `false`. Routing installs the bundle
all at once, so it is the same assertion either way, and `false` keeps meaning
exactly what it says.

Only a `true` evaluation can grade behavior. All four fields below are
optional, and any one of them promotes it from routing-only to behavior-graded:

| Field | Graded by | Use it for |
| --- | --- | --- |
| `logs_contain` | exact substring | a literal that must appear: a script name, a flag, a pinned image tag |
| `files_exist` | the filesystem | an artifact the run must produce |
| `expected_behavior` | an LLM judge | a step the agent must take, in plain language |
| `unexpected_behavior` | an LLM judge | the mistake this skill exists to prevent |

Prefer the deterministic two where you can: they are instant and free, where a
judged expectation costs a second agent call.

Two things you never write. **Do not assert your own skill's name** in
`logs_contain` — routing mode grades that properly, and a substring match only
proved the skill was staged. **Do not label a prompt** as positive, near-miss,
or unrelated: that is derived from the flag and the file it lives in.

The full field reference is
[`eval/schema/evals.schema.json`](eval/schema/evals.schema.json). Datasets do
not link to it; `python eval/run_evals.py --validate` is what enforces it.

### How much is enough

| Tier | Required when | What it costs you |
| --- | --- | --- |
| **0** | always | 3 evaluations with `skill_should_trigger: true`, 2 with `false`. CI rejects a skill without them. |
| **1** | your skill can be exercised on a generic runner | one evaluation with an `expected_behavior` / `unexpected_behavior` / `logs_contain` / `files_exist` expectation |
| **2** | your skill needs hardware or a live service | an `evals/machine.yml` naming the kind of machine |

Tier 0 is deliberately cheap, and it buys more than it looks like. Routing
pools the published skills' cases into one run, so your five prompts become
negative cases for every other published skill, and theirs become negatives for
yours. Twelve owners writing five prompts each produce a routing matrix none
of them had to coordinate.

Spend your effort on the near misses. Positive prompts mostly pass; the ones
that find real problems are the prompts that sit just outside your scope — the
wrong vendor's hardware, the adjacent skill's job, your vocabulary used to mean
something else.

**Before your skill is published**, routing has nowhere to put it: it is not in
[`.claude-plugin/marketplace.json`](.claude-plugin/marketplace.json), so it is
not installed alongside the others and cannot win a prompt. Your dataset is
still validated and your behavior cases still run, and your `false` prompts
still run as near misses against the published skills. Your positives start
being scored the day the skill joins the bundle, with no edit to them. Routing
is measured against the set of skills a user actually installs, so an
unpublished skill would otherwise change everyone else's score while shipping
to nobody.

### When JSON isn't enough

Two optional files sit beside the dataset.

`evals/machine.yml` declares where the behavior cases run. Skip it unless you
need something other than the default runners on Linux and Windows. It replaces
the hardcoded hardware list CI used to carry, and the `skipif` guards each test
file used to repeat. It holds two keys, both optional:

```yaml
runner_type: instinct    # `default` (assumed) or `instinct`
os: [Linux]              # defaults to every platform that runner type has
```

Most skills that need this file need only `os: [Linux]`, to drop a Windows leg
that would just exercise the failure path of Linux-only tooling.

Name the kind of machine, not its consequences. `runner_type: instinct` is the
whole declaration for an Instinct skill: the runner labels, the fact that the
job is Linux-only, the `enable_mi_ci` pull-request label that rations that
scarce pool, and the scoped credentials it uses all follow automatically from
the runner type. Missing the gate label is a warning rather than a failure, so
a PR is never blocked by a test it deliberately did not request.

Adding a new class of hardware means adding an entry to `RUNNER_TYPES` in
[`eval/datasets.py`](eval/datasets.py) once, not teaching every skill that
needs it the same set of labels.

`evals/hooks.py` is the escape hatch for setup a JSON file cannot express —
cloning a repo, tearing down a container, running an external scoring script.
Every function is optional:

```python
def setup_session(cache_dir): ...   # once per run; returns {name: value} for {placeholders} in prompts
def setup(workspace, case, ctx): ...    # before each case; may return more placeholders
def teardown(workspace, case, ctx): ...
def check(run, case, ctx): ...      # after each case; raise AssertionError to fail it
```

Keep prompts and expectations in the dataset even when you use hooks, so what
is being asserted stays readable without opening Python. See
[`skills/tracelens-analysis-orchestrator/evals/hooks.py`](skills/tracelens-analysis-orchestrator/evals/hooks.py)
for the involved case and
[`skills/serving-llms-on-instinct/evals/hooks.py`](skills/serving-llms-on-instinct/evals/hooks.py)
for the simple one.

### Running them

```bash
python eval/run_evals.py --validate              # structure only: no agent, no tokens, instant
python eval/run_evals.py --skill <your-skill>    # routing and behavior for your skill
python eval/run_evals.py --mode routing          # the published bundle
python eval/run_evals.py --only <case-id> --keep-logs logs   # one case, keeping the transcript
```

Everything but `--validate` needs the `claude` CLI authenticated, plus whatever
your own cases need. No `pip install`: the runner is standard library only.

In CI, the `evals` workflow runs routing when a change can move a routing
decision (a published description, any dataset, or the bundle itself), and runs
behavior for the skills a change touches.

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
- [ ] `evals/evals.json` has at least 3 evaluations with `skill_should_trigger: true` and 2 with `false` (Tier 0)
- [ ] At least one evaluation grades behavior, or the skill genuinely needs hardware CI cannot reach
- [ ] `./.github/scripts/check.sh` passes (CI runs this on every PR)
- [ ] `python eval/run_evals.py --skill <name>` passes

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

It also checks every eval dataset (`python eval/run_evals.py --validate`, no agent and no tokens):

- `evals/evals.json`: present, parseable, and using only known fields — a typo'd key is an error rather than a silently dropped expectation
- Tier 0 coverage: ≥ 3 evaluations with `skill_should_trigger: true`, ≥ 2 with `false`
- `skill_should_trigger`: present and a real boolean on every evaluation
- case ids: unique across the whole repo, because routing pools every skill's evaluations into one run
- a `false` evaluation: an id and a prompt only, since no skill loads for those prompts
- `workspace`: points at a directory that exists

It also checks the plugin manifests:

- `.claude-plugin/marketplace.json` lists exactly one plugin (the `amd-skills` bundle) with `source` set to `./`, `strict: false`, and a non-empty human-readable `description`
- `.cursor-plugin/marketplace.json` is up to date — it mirrors `.claude-plugin/marketplace.json` and pulls shared identity (name, description, version, author) from `plugin-metadata.json` (regenerate with `./.github/scripts/publish.sh`)
- `.codex-plugin/plugin.json` and `.agents/plugins/marketplace.json` are up to date. The Codex plugin manifest and repo marketplace catalog, generated from the same canonical sources (curated `skills` list from `.claude-plugin/marketplace.json`, identity from `plugin-metadata.json`); You can regenerate with `./.github/scripts/publish.sh`.
