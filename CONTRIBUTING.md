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

It must also be **approved once**, before its first skill lands here. File a
[Product repo approval](https://github.com/amd/skills/issues/new?template=product-repo-approval.yml)
issue and the workflow walks the two named accounts — the engineering owner and
the product release owner — through commenting `/approve`. That opens a pull
request adding the repo to
[`.github/skill_owners.json`](.github/skill_owners.json); once it merges, the
repo can federate as many skills as it likes and never needs approving again.

The issue also asks which **branch** to federate from. Leave it blank for
`main`.

## 1. Author the skill in your repo

Each skill is a folder holding a valid `SKILL.md`, a `skill-card.md`, and an
`evals/evals.json` dataset. Put the folders anywhere in your repo, commonly
`skills/` or `.agents/skills/`.

The catalog tracks one branch of your repo, **`main`** unless your approval
named another. It can be:

- a branch name, such as `main` or `develop`, followed as it moves;
- a release pattern, such as `release/*`, which follows your newest release
  branch. The `*` stands for a version number, so `release/0.12` wins over
  `release/0.9`. Branches like `release/next` never match. The nightly
  federation run moves the catalog to a new release branch once you push it.
  Use this when your skills
  ship with your releases (Quark does this).

Only the approved branch can be tracked, and tags and commits can't be tracked
at all. Whatever lands on that branch reaches users, so it should be a branch
your own review process protects. Land skill changes there and the catalog
follows.

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
    },
    {
      "repo": "amd/Quark",
      "license": "MIT",
      "branch": "release/*",
      "skills": [
        {
          "path": ".claude/skills/quark-install",
          "as": "quark-install"
        }
      ]
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `repo` | GitHub `<owner>/<repo>`, must be AMD-owned |
| `license` | SPDX id, carried into each vendored copy's marker file |
| `branch` | Optional. The branch to track, or a release pattern such as `release/*`. Defaults to `main`. Must match the branch in your approval, or the federation guard fails the pull request |
| `skills[].path` | Path of the skill folder inside your repo, from the repo root |
| `skills[].as` | Optional local catalog name; use it to namespace as `<project>-<skill>` so names stay unique |

There is no tag or commit field. A source always follows a branch.

## 3. Vendor and validate locally

The scripts read `.github/federation.json` from your working tree.

```bash
uv run .github/scripts/federate_skills.py --check-catalog  # schema only, no clone
uv run .github/scripts/federate_skills.py --only <skill>   # vendor into skills/<skill>/ (repeat --only per skill)
./.github/scripts/publish.sh                               # regenerate the manifests
./.github/scripts/check.sh                                 # validate (same command CI runs)
```

Pass `--only` once for each skill you added, using its local catalog name (the
`as:` value). Always pass it: without `--only`, the importer also bumps every
existing federated skill whose upstream has moved, and the federation guard
closes a pull request that edits those. For the same reason, do not name a
skill that is already federated on `main`, and make sure your `as:` name does
not match a skill already under `skills/`, since the import replaces that
folder.

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
    uses: amd/skillscope/.github/workflows/reusable.yml@v0.1.2
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

Merge the change to your tracked branch (or push a new release branch, if you
track `release/*`) and the catalog picks it up on its own.
The `federate-skills` workflow runs nightly (and on demand), re-vendors any
skill whose upstream folder contents changed, and opens a pull request titled
`Bump <skill> to <short commit>`. A night with no upstream change produces no
pull request, so the only ones you see are real bumps.
