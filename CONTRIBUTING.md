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

For repository structure and the broader catalog model, see the
[README](README.md).

## Eligibility

The source repo must be under an AMD GitHub org (e.g. `AMD-AGI/...`). Non-AMD
repos are not accepted at this time.

## 1. Author the skill in your repo

Each skill is a folder holding a valid `SKILL.md`, a `skill-card.md`, and an
`evals/evals.json` dataset. Put the folders in a known directory in your repo,
commonly `skills/` or `.agents/skills/`, and pick a branch for the catalog to
track.

Everything ships with the folder, so the requirements and the eval dataset are
yours to maintain upstream alongside the skill. See
[docs/skill-requirements.md](docs/skill-requirements.md) for what a valid skill
must contain and [docs/best-practices.md](docs/best-practices.md) for how to
make it good.

## 2. Register your source

Edit [`.github/scripts/sources.yml`](.github/scripts/sources.yml) and append an
entry:

```yaml
sources:
  - name: amd-myproject          # kebab-case source id
    repo: AMD-Org/MyProject      # must be AMD-owned
    ref: main                    # branch to track (e.g. main or a release branch)
    path: skills                 # dir in your repo holding the skill folders
    license: MIT                 # SPDX id, carried into the marker file
    skills:
      - name: my-skill           # folder name in your repo
        as: myproject-my-skill   # local catalog name: <project>-<skill>
```

Use `as:` to namespace skills as `<project>-<skill>` so catalog names stay
unique.

## 3. Import and validate locally

The scripts read `sources.yml` from your working tree.

```bash
uv run .github/scripts/import_external_skills.py   # vendor into skills/<name>/
./.github/scripts/publish.sh                       # regenerate the manifests
./.github/scripts/check.sh                         # validate (same command CI runs)
```

The importer also adds your skill to the published bundle, so there is no
manifest to edit by hand.

## 4. Open a pull request

Commit `skills/**`, `.github/scripts/sources.yml`, and the regenerated
manifests. A maintainer reviews and merges once CI passes. The `validate`
workflow runs `check.sh`; the `evals` workflow runs your prompts against a real
agent.

Never hand-edit vendored skills under `skills/`. Changes must come from your
repo via re-import, or they will be overwritten.

## Catch failures before nightly

The catalog runs checks against your skills. Run the **same** checks in your own
repo by calling them as reusable workflows, so you catch breakage during normal
development instead of in the catalog's nightly run. The logic and config live
in `amd/skills`, so green in your repo means green in the catalog, and you never
copy or maintain the check yourself.

Add a caller workflow to your repo (e.g. `.github/workflows/skills-checks.yml`):

```yaml
name: skills-checks
on:
  pull_request:
  workflow_dispatch:
jobs:
  external-references:
    uses: amd/skills/.github/workflows/external-reference-check.yml@main
    permissions:
      contents: read
      issues: write
```

## Update or remove

Change the skill in your repo and the catalog picks it up on the next import.
To remove a skill, drop it from `sources.yml`; the importer prunes the vendored
copy and its bundle entry. Automatic refresh and pruning through nightly
workflows will be enabled soon.
