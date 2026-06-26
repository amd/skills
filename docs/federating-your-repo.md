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

1. Open a PR with your `sources.yml` change.
2. Run the **Import external skills** workflow (manual dispatch). It shallow-clones
   your repo at `ref`, vendors the folders into `skills/<name>/`, writes a
   `.federated.json` marker (repo, ref, resolved commit, license, timestamp), and
   opens a PR with the result.
3. A maintainer reviews and merges. CI validation must pass.

## Update or remove

- **Update:** bump `ref` in `sources.yml`, re-run the workflow. The marker records
  the new commit.
- **Remove:** delete the entry and re-run; the importer prunes the vendored copy.

Never hand-edit vendored skills under `skills/`; changes must come from your repo
via re-import, or they'll be overwritten.
