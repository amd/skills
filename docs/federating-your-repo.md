# Federate Your Repo Into the Catalog

How to list skills that live in **your own AMD repo** in this catalog. Your repo
stays the source of truth; the catalog vendors a pinned copy.

This is the detailed version of **Path B** in
[CONTRIBUTING.md](../CONTRIBUTING.md#path-b-skills-authored-in-a-product-repository-federation).
Start there for an overview of how it compares to authoring skills directly in
this repo.

> **Eligibility: AMD-owned repositories only.** The source `repo` must be under
> an AMD GitHub org (e.g. `AMD-AGI/...`). Non-AMD repos are not accepted.

## Prerequisites

- Each skill is a folder with a valid `SKILL.md` and `skill-card.md`.
  See [CONTRIBUTING.md](../CONTRIBUTING.md) and [skill-cards.md](skill-cards.md).
- Skills live in a known directory in your repo (e.g. `skills/`).
- Pick a branch to track (e.g. `main` or a release branch).

Product repositories may also keep lightweight public wrapper folders that
point to implementations elsewhere in the same repository. See
[Wrapper-based product skills](#wrapper-based-product-skills) for the stricter
format and standalone packaging behavior.

## Add your source

Edit [`.github/scripts/sources.yml`](../.github/scripts/sources.yml) and append an entry:

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

Use `as:` to namespace skills as `<project>-<skill>` so catalog names stay unique.

## Wrapper-based product skills

Set `resolve_wrappers: true` when the public skill folders contain only routing
frontmatter and an exact implementation pointer:

```markdown
---
name: my-product-build
description: Build a project with My Product. Use when ...
---

Read and follow the instructions in `.agents/skills-impl/my-product-build/SKILL.md`.
```

Configure `path` at a common ancestor of both the public wrappers and their
implementations, then address each wrapper relative to that path:

```yaml
  - name: amd-myproduct
    repo: AMD-Org/MyProduct
    ref: v1.0
    path: .agents
    license: MIT
    resolve_wrappers: true
    skills:
      - name: skills/my-product-build
        as: myproduct-build
```

The importer uses the wrapper's `name` and `description` for routing, vendors
the implementation and its adjacent resources, rewrites escaping Markdown
links to the pinned upstream commit, and bundles referenced non-public helper
skills under `references/`. The generated catalog folder is therefore usable
when installed by itself; the wrapper is never shipped as a broken pointer.

Wrapper resolution is deliberately strict: the body must contain exactly the
single `Read and follow .../SKILL.md` sentence, the target must be a tracked
file inside the same repository, and implementation skill names must be
unique beneath the configured source path.

If a released implementation contains stale internal skill names, the source
or an individual skill entry may declare a narrow `aliases:` mapping from each
upstream term to its catalog wording. Skill-entry aliases override source-wide
values. Keep these corrections exceptional and remove them when the next
product release fixes the source.

## Import

Run the import scripts locally (they read `sources.yml` from your working tree),
then open a PR for review.

1. Vendor the skills and refresh the manifests:

   ```bash
   uv run .github/scripts/import_external_skills.py    # vendor into skills/<name>/
   ./.github/scripts/publish.sh                        # regenerate the Cursor manifest
   ./.github/scripts/check.sh                          # validate
   ```

2. Commit `skills/**`, `.github/scripts/sources.yml`, and the manifests.
3. Open a PR; a maintainer reviews and merges once CI passes.

## Catch failures before nightly

The catalog runs checks against your skills. Run the **same** checks in your own
repo by calling them as reusable workflows, so you catch breakage during normal
development instead of in the catalog's nightly run. The logic and config live in
`amd/skills`, so green in your repo means green in the catalog — and you never copy
or maintain the check yourself.

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

Automatic refresh and pruning will soon be enabled through nightly workflows.

Never hand-edit vendored skills under `skills/`; changes must come from your repo
via re-import, or they'll be overwritten.
