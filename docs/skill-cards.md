# Skill cards

Every skill in this catalog ships a `skill-card.md` next to its `SKILL.md`. The card is a short, human-facing governance record: it tells a reviewer *what* the skill is, *who* owns it, and *under what license* it ships, without making them read the source first.

A `SKILL.md` is written for the agent (routing and instructions). A skill card is written for the people deciding whether to trust, install, or maintain the skill.

## Required sections

The AMD card is intentionally minimal. Three sections are required, each a top-level `##` heading with non-empty body text:

| Section | Question it answers |
| --- | --- |
| Description | What does this skill do, in one sentence? |
| Owner | Who is accountable for maintaining it? |
| License | What license governs its use and redistribution? |

The validator (`.github/scripts/validate_skills.py`) fails any skill whose card is missing or whose required sections are absent or empty.

## Template

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

## Writing a good Description

Keep it to one sentence that states the outcome, matching the marketplace blurb. Avoid restating internal mechanics (that belongs in `SKILL.md`).

```
Good: Diagnose why ROCm, PyTorch, or llama.cpp isn't working on an AMD GPU
      and propose the next step.
Bad:  Runs a series of Python scripts that parse logs and apply regex rules.
```

## Federated skills

Skills imported from a product repository (see [`.github/scripts/sources.yml`](../.github/scripts/sources.yml)) are vendored wholesale, so a card authored here would be overwritten on the next import. If upstream ships its own `skill-card.md`, it is kept as-is; otherwise the importer synthesizes a minimal card from the source metadata (description, owner repo, and license). To customize a federated skill's card, add a `skill-card.md` to the skill folder in the upstream repository.

## Out of scope (for now)

The card stays at Description, Owner, and License. Evaluation results, benchmark data, risk statements, and signing identifiers are not part of the AMD card today; sections can be added later without breaking the validation gate.
