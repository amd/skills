# Skill Requirements

What must be true for a skill to merge. Every rule here is checked by
`./.github/scripts/check.sh`, which is the same command CI runs on every pull
request. These apply to in-repo and federated skills alike: vendored copies go
through the identical validator.

For advice on writing a *good* skill rather than a valid one, see
[best-practices.md](best-practices.md).

## Required files

```
skills/<skill-name>/
  SKILL.md          # required: frontmatter + instructions
  skill-card.md     # required: governance card
  evals/evals.json  # required: routing and behavior dataset
```

## SKILL.md

The frontmatter must be a valid YAML block containing `name` and `description`.

| Field | Rule |
| --- | --- |
| `name` | lowercase-with-hyphens, ≤ 64 characters, no `anthropic` or `claude` substring, matches the directory name |
| `description` | non-empty, ≤ 1024 characters |
| body | ≤ 500 lines |

## skill-card.md

A short, human-facing governance record at the skill root. It tells a reviewer
what the skill is, who owns it, and under what license it ships, without making
them read the source. `SKILL.md` is written for the agent; the card is written
for the people deciding whether to trust, install, or maintain the skill. It is
never loaded into agent context.

Three sections are required, each a `##` heading with non-empty body text:

| Section | Question it answers |
| --- | --- |
| Description | What does this skill do, in one sentence? |
| Owner | Who is accountable for maintaining it? |
| License | What license governs its use and redistribution? |

Copy this into `skills/<your-skill>/skill-card.md`:

```markdown
# Skill Card

## Description

<one sentence: what the skill does, for whom>

## Owner

<team or org accountable for maintenance, e.g. AMD>

## License

<SPDX identifier or link, e.g. MIT>
```

Keep the Description to one sentence stating the outcome, matching the
marketplace blurb. Internal mechanics belong in `SKILL.md`.

```
Good: Diagnose why ROCm, PyTorch, or llama.cpp isn't working on an AMD GPU
      and propose the next step.
Bad:  Runs a series of Python scripts that parse logs and apply regex rules.
```

The card stays at these three sections. Evaluation results, benchmark data,
risk statements, and signing identifiers are not part of the AMD card today;
sections can be added later without breaking the validation gate.

**Federated skills** are vendored wholesale, so a card authored here would be
overwritten on the next import. If upstream ships its own `skill-card.md` it is
kept as-is; otherwise the importer synthesizes a minimal card from the source
metadata. To customize a federated skill's card, add it to the skill folder in
the upstream repository.

## evals/evals.json

Every skill needs a dataset. The floor is **Tier 0**: at least 3 evaluations
with `skill_should_trigger: true` and 2 with `false`. CI rejects a skill
without them.

The validator also enforces that the file is parseable and uses only known
fields (a typo'd key is an error, not a silently dropped expectation), that
`skill_should_trigger` is present and a real boolean on every evaluation, that
case ids are unique across the whole repo, that a `false` evaluation carries
only an id and a prompt, and that any `workspace` points at a directory that
exists.

See [evals.md](evals.md) for what to put in the file and how it is graded.

## Publishing

A skill is published by adding a `./skills/<name>` entry to the `skills` array
of the single `amd-skills` plugin in
[`.claude-plugin/marketplace.json`](../.claude-plugin/marketplace.json). All
published skills ship together in that one plugin; a skill left out of the
array stays unpublished.

The other manifests are derived and must be in sync, which the validator
checks. Regenerate them rather than hand-editing:

```bash
./.github/scripts/publish.sh   # writes .cursor-plugin/, .codex-plugin/, .agents/plugins/
```

`.claude-plugin/marketplace.json` must list exactly one plugin with `source`
set to `./`, `strict: false`, and a non-empty human-readable description.
Shared identity (name, description, version, author) comes from
`plugin-metadata.json`.

## Before you open the PR

CI will reject the skill if any of these are false:

- [ ] `SKILL.md` has valid frontmatter; `name` is lowercase-with-hyphens, ≤ 64 chars, and matches the directory
- [ ] `description` is non-empty and ≤ 1024 characters
- [ ] `SKILL.md` body is ≤ 500 lines
- [ ] `skill-card.md` exists with non-empty Description, Owner, and License
- [ ] `evals/evals.json` has ≥ 3 `true` and ≥ 2 `false` evaluations, with unique ids
- [ ] Manifests are regenerated and in sync
- [ ] `./.github/scripts/check.sh` passes

A reviewer will push back if any of these are false:

- [ ] The description states the user's goal and includes likely trigger phrases
- [ ] At least one evaluation grades behavior, or the skill genuinely needs hardware CI cannot reach
- [ ] Prerequisites (ROCm version, GPU arch, container, env vars) are stated explicitly
- [ ] Scripts handle expected errors and document their constants and dependencies
- [ ] The skill was tested end-to-end on the target hardware against real prompts

```bash
./.github/scripts/check.sh                    # structural validation, no tokens
python eval/run_evals.py --skill <your-skill> # routing and behavior
```
