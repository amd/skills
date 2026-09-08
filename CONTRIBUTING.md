# Contributing to AMD Skills

We welcome contributions from AMD engineers and selected partners.

> **Only federated submissions are accepted.** Your skill lives in an AMD-owned
> product repo, which stays the source of truth, and this catalog vendors a
> pinned copy. We no longer accept skills authored directly under `skills/` in
> this repository. Skills already there continue to ship.

Federation keeps each skill owned and versioned by the team that owns the product it describes, on that product's release cadence, while users still get everything from one install.

Three companion guides hold the detail:

| Guide | Read it for |
| --- | --- |
| [docs/skill-requirements.md](docs/skill-requirements.md) | The rules CI enforces: required files, frontmatter limits, skill cards, the pre-PR checklist |
| [docs/best-practices.md](docs/best-practices.md) | How to write a skill agents actually reach for: fit, descriptions, body structure, scripts, AMD specifics |
| [docs/evals.md](docs/evals.md) | How structure, routing, and behavior are graded, and what to put in `evals/evals.json` |
| [amd/skillscope](https://github.com/amd/skillscope) | The harness that runs those graders, here and in your own repo |

For repository structure and the broader catalog model, see the
[README](README.md).

## Eligibility

The source repo must be under an AMD GitHub org (e.g. `AMD-AGI/...`). Non-AMD
repos are not accepted at this time.

## 1. Author the skill in your repo

Each skill is a folder holding a valid `SKILL.md`, a `skill-card.md`, and an
`evals/evals.json` dataset. Put the folders anywhere in your repo, commonly
`skills/` or `.agents/skills/`.

The catalog always tracks your **`main`** branch. That is deliberate: the
catalog cannot be pointed at a side branch, so what reaches users is what your
own review process has already merged. Land skill changes on `main` and the
catalog follows.

Everything in the folder ships, so the requirements are yours to maintain
upstream alongside the skill. See
[docs/skill-requirements.md](docs/skill-requirements.md) for what a valid skill
must contain and [docs/best-practices.md](docs/best-practices.md) for how to
make it good.

## 2. Register your source

Add your repo to [`.github/federation.json`](.github/federation.json). Every
skill is declared individually, by the full path of its folder in your repo, so
one repo can federate as many skills as it likes from wherever they live:

```json
{
  "sources": [
    {
      "repo": "AMD-Org/MyProject",
      "license": "MIT",
      "skills": [
        {
          "path": "skills/my-skill",
          "as": "myproject-my-skill"
        },
        {
          "path": "tools/agents/skills/other-skill",
          "as": "myproject-other-skill"
        }
      ]
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `repo` | GitHub `<owner>/<repo>`, must be AMD-owned. Always tracked at `main` |
| `license` | SPDX id, carried into each vendored copy's marker file |
| `skills[].path` | Path of the skill folder inside your repo, from the repo root |
| `skills[].as` | Optional local catalog name; use it to namespace as `<project>-<skill>` so names stay unique |

There is no ref, branch, or commit field. Federation is `main`-only by design.

## 3. Vendor and validate locally

The scripts read `.github/federation.json` from your working tree.

```bash
uv run .github/scripts/federate_skills.py --check-catalog  # schema only, no clone
uv run .github/scripts/federate_skills.py                  # vendor into skills/<name>/
./.github/scripts/publish.sh                               # regenerate the manifests
./.github/scripts/check.sh                                 # validate (same command CI runs)
```

The importer also adds your skill to the published bundle, so there is no
manifest to edit by hand.

## 4. Open a pull request

Commit `.github/federation.json`, `skills/**`, and the regenerated manifests. A
maintainer reviews and merges once CI passes. The `validate` workflow checks the
manifests; the `evals` workflow runs [skillscope](https://github.com/amd/skillscope)
— the structural checks, then your prompts against a real agent.

Never hand-edit vendored skills under `skills/`. Changes must come from your
repo via re-import, or they will be overwritten.

## Catch failures before nightly

The catalog runs checks against your skills. Run the **same** checks in your own
repo by calling them as reusable workflows, so you catch breakage during normal
development instead of in the catalog's nightly run. You never copy or maintain
a check yourself, and green in your repo means green in the catalog.

Every check the catalog runs is [skillscope](https://github.com/amd/skillscope),
which grades a skill wherever it lives — structure, the references your
markdown makes, routing, and behavior. Point it at your skill folder and it
reads the same `evals/evals.json` this catalog does:

```yaml
name: skills-checks
on:
  pull_request:
  workflow_dispatch:
jobs:
  evals:
    uses: amd/skillscope/.github/workflows/reusable.yml@v0.1.0
    secrets:
      api_key: ${{ secrets.ANTHROPIC_API_KEY }}
    with:
      skills: path/to/your/skill
```

Skillscope's [usage guide](https://github.com/amd/skillscope/blob/main/docs/usage.md)
covers the rest: holding a check to `optional` while you get a bar green, GPU
runners, pooling several skills into one routing run, and fetching external
URLs on a schedule to catch link rot.

## Update or remove

Merge the change to `main` in your repo and the catalog picks it up on its own.
The `federate-skills` workflow runs nightly (and on demand), re-vendors any
skill whose upstream folder contents changed, and opens a pull request titled
`Bump <skill> to <short commit>`. A night with no upstream change produces no
pull request, so the only ones you see are real bumps.

To remove a skill, open a pull request that drops its entry from
`.github/federation.json`, deletes the vendored `skills/<name>/` folder, and
regenerates the manifests. Federation never deletes anything on its own: a
nightly run only reports a vendored copy that no source declares any more.
