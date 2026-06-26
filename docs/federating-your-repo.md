# Federate Your Repo Into the Catalog

How to list skills that live in **your own AMD repo** in this catalog. Your repo
stays the source of truth; the catalog vendors a pinned copy.

> **Eligibility: AMD-owned repositories only.** The source `repo` must be under
> an AMD GitHub org (e.g. `AMD-AGI/...`). Non-AMD repos are not accepted.

## Prerequisites

- Each skill is a folder with a valid `SKILL.md` and `skill-card.md`.
  See [CONTRIBUTING.md](../CONTRIBUTING.md) and [skill-cards.md](skill-cards.md).
- Skills live in a known directory in your repo (e.g. `skills/`).
- Pick a branch to track (e.g. `main` or a release branch).

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

## Import

Run the import scripts locally (they read `sources.yml` from your working tree),
then open a PR for review.

1. Vendor the skills and refresh the manifests:

   ```bash
   uv run .github/scripts/import_external_skills.py    # vendor into skills/<name>/
   uv run .github/scripts/generate_cursor_marketplace.py
   ./.github/scripts/check.sh                          # validate
   ```

2. Commit `skills/**`, `.github/scripts/sources.yml`, and the manifests.
3. Open a PR; a maintainer reviews and merges once CI passes.

## Update or remove

Automatic refresh and pruning will soon be enabled through nightly workflows.

Never hand-edit vendored skills under `skills/`; changes must come from your repo
via re-import, or they'll be overwritten.
