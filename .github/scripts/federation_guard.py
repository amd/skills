#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0"]
# ///
"""Report the two federation rules a pull request can break.

Pure reporting: this script decides nothing and touches nothing. It reads the
base branch's tree, the pull request's changed paths, and the
`.github/federation.json` the pull request proposes, then writes a JSON report.
The `federation-guard` workflow is what closes a pull request or fails a check
from that report, because those need a token and this needs to be runnable by
hand.

Rule 1 -- vendored skills are edited upstream, not here.
    A skill folder carrying a `.federated.json` marker is a mirror of a folder
    in a product repo. `federate_skills.py` re-imports it with rmtree +
    copytree, so anything committed here is deleted by the next nightly run:
    an edit that looks merged is really just pending its own reversal. The one
    exception is the skill's top-level `evals/` folder, which federation never
    carries -- the catalog owns those datasets, so editing them here is the
    only way to edit them at all.

    Whether a skill is federated is read from the *base* branch, so the pull
    request that first vendors a skill (and therefore adds the marker) does not
    trip over itself.

Rule 2 -- a new federated skill needs its product repo approved.
    `.github/skill_owners.json` records the repos whose engineering owner and
    product release owner have both signed off, via the `product-repo-approval`
    workflow. A pull request that declares a skill from a repo missing from
    that registry is asking the catalog to vendor code nobody has vouched for.

    Only skills the pull request *adds* are held to this. A skill already
    declared on the base branch predates the registry and stays as it is, so
    the rule cannot retroactively break the catalog. "Already declared" is
    keyed on the (repo, local skill name) pair: moving a skill's upstream path
    is maintenance, but pointing an existing name at a different repo is a new
    approval question.

    The registry is read from the base branch too. Reading the pull request's
    copy would let the same pull request that federates a skill also grant it
    approval, which is the gate approving itself.

Usage:
    uv run .github/scripts/federation_guard.py \
        --changed-files changed.txt \
        --head-federation head-federation.json \
        --report report.json

`--changed-files` is a file holding one repo-relative path per line (what
`gh api .../pulls/N/files --jq '.[].filename'` prints). `--head-federation` is
the pull request's copy of `.github/federation.json`; leave it out when the
pull request does not change that file and rule 2 is skipped.

The exit status is 0 whenever the evaluation itself succeeded, violations or
not. What to do about a violation is the workflow's call.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import federate_skills as fed  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parent.parent
SKILLS_PREFIX = "skills/"


def read_changed_files(path: Path) -> list[str]:
    """Read one repo-relative path per line, normalized to POSIX separators.

    `utf-8-sig` because a list written by a PowerShell one-liner on a
    maintainer's machine starts with a byte-order mark, which would otherwise
    hide the first path in it.
    """
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    return [line.strip().replace("\\", "/") for line in lines if line.strip()]


def load_approved_repos(registry: Path) -> set[str]:
    """Return the approved repos from `.github/skill_owners.json`, lowercased.

    A missing or malformed registry reads as "nothing is approved" rather than
    as an error: the strict reading is the safe one, and `record_skill_owner.py`
    is what validates the file when it writes it.
    """
    if not registry.is_file():
        return set()
    try:
        data = json.loads(registry.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    repos = data.get("repos") if isinstance(data, dict) else None
    if not isinstance(repos, list):
        return set()
    return {
        entry["repo"].strip().lower()
        for entry in repos
        if isinstance(entry, dict) and isinstance(entry.get("repo"), str)
    }


def federated_skills(base_dir: Path) -> dict[str, dict]:
    """Map each federated skill on the base branch to where it comes from.

    Two things say a skill is federated and they are deliberately unioned,
    because on `main` today they disagree in both directions:

      * a `.federated.json` marker, written by the importer. `magpie-kernel-
        evaluator` has one but is no longer declared, so nothing refreshes it;
        it is still a mirror of AMD-AGI/Magpie and still the wrong place to
        change it.
      * a declaration in `.github/federation.json`. `hyperloom-workload-
        optimizer` is declared but shipped without a marker, and a skill with
        no marker is exactly the case the importer treats as stale: the next
        nightly run rmtree's the folder and copies upstream over it. Protecting
        only marked skills would leave the one most certain to be overwritten
        unprotected.

    Marker values win where both exist, since the marker records what was
    actually imported. A malformed federation file reads as "declares nothing"
    rather than raising; rule 2 is where that file's health is reported.
    """
    federated: dict[str, dict] = {}
    try:
        sources = fed.parse_federation(base_dir / ".github" / "federation.json")
    except (ValueError, FileNotFoundError):
        sources = []
    for source in sources:
        for spec in source.skills:
            federated[spec.dest_name] = {
                "repo": source.repo,
                "source_path": spec.path,
            }

    skills_dir = base_dir / "skills"
    if skills_dir.is_dir():
        for skill_dir in sorted(p for p in skills_dir.iterdir() if p.is_dir()):
            if not (skill_dir / fed.MARKER_FILENAME).is_file():
                continue
            marker = fed.read_marker(skill_dir)
            declared = federated.get(skill_dir.name, {})
            federated[skill_dir.name] = {
                "repo": marker.get("repo") or declared.get("repo", ""),
                "source_path": marker.get("path") or declared.get("source_path", ""),
            }
    return federated


def vendored_edits(changed: list[str], federated: dict[str, dict]) -> list[dict]:
    """Group the changed paths that edit a vendored skill, by skill.

    The `evals/` exemption mirrors `federate_skills.UNFEDERATED_DIR_NAMES` and
    the way that module's `content_hash` applies it, so what counts as "not
    federated" is the same on both sides rather than a second opinion.
    """
    hits: dict[str, dict] = {}
    for path in changed:
        if not path.startswith(SKILLS_PREFIX):
            continue
        name, _, tail = path[len(SKILLS_PREFIX) :].partition("/")
        if not tail:
            # Something directly under `skills/`, not inside a skill.
            continue
        parts = tail.split("/")
        if len(parts) > 1 and parts[0] in fed.UNFEDERATED_DIR_NAMES:
            continue
        source = federated.get(name)
        if source is None:
            continue
        hit = hits.setdefault(
            name,
            {
                "skill": name,
                "repo": source["repo"],
                "source_path": source["source_path"],
                "paths": [],
            },
        )
        hit["paths"].append(path)
    for hit in hits.values():
        hit["paths"].sort()
    return [hits[name] for name in sorted(hits)]


def declared_skills(sources: list[fed.Source]) -> dict[tuple[str, str], dict]:
    """Map every declared skill to its (repo, local name) identity."""
    declared: dict[tuple[str, str], dict] = {}
    for source in sources:
        for spec in source.skills:
            key = (source.repo.lower(), spec.dest_name)
            declared[key] = {
                "skill": spec.dest_name,
                "repo": source.repo,
                "path": spec.path,
            }
    return declared


def new_skills_needing_approval(
    base: dict[tuple[str, str], dict],
    head: dict[tuple[str, str], dict],
    approved: set[str],
) -> list[dict]:
    return [
        entry
        for key, entry in sorted(head.items())
        if key not in base and key[0] not in approved
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--changed-files",
        type=Path,
        required=True,
        metavar="PATH",
        help="File listing the pull request's changed paths, one per line.",
    )
    parser.add_argument(
        "--head-federation",
        type=Path,
        metavar="PATH",
        help=(
            "The pull request's copy of .github/federation.json. Omit to skip "
            "the product repo approval rule."
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=REPO_ROOT,
        help=f"Checkout of the base branch (default: {REPO_ROOT}).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        metavar="PATH",
        help="Write the JSON report here (default: stdout only).",
    )
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict:
    base_dir = args.base_dir
    report: dict = {
        "vendored_edits": vendored_edits(
            read_changed_files(args.changed_files), federated_skills(base_dir)
        ),
        "new_skills_needing_approval": [],
        "federation_error": "",
    }

    if args.head_federation is None:
        return report

    base_federation = base_dir / ".github" / "federation.json"
    try:
        head = declared_skills(fed.parse_federation(args.head_federation))
    except (ValueError, FileNotFoundError) as exc:
        # The `validate` workflow is what explains a malformed federation file
        # in detail; recording it here keeps this rule from passing a pull
        # request whose declarations could not be read at all.
        report["federation_error"] = str(exc)
        return report
    try:
        base = declared_skills(fed.parse_federation(base_federation))
    except (ValueError, FileNotFoundError) as exc:
        report["federation_error"] = f"base branch copy: {exc}"
        return report

    report["new_skills_needing_approval"] = new_skills_needing_approval(
        base,
        head,
        load_approved_repos(base_dir / ".github" / "skill_owners.json"),
    )
    return report


def print_summary(report: dict) -> None:
    edits = report["vendored_edits"]
    if edits:
        print("Vendored (federated) skills edited by this pull request:")
        for hit in edits:
            print(f"  skills/{hit['skill']} -- vendored from {hit['repo']}")
            for path in hit["paths"]:
                print(f"    {path}")

    if report["federation_error"]:
        print(f".github/federation.json could not be read: {report['federation_error']}")

    pending = report["new_skills_needing_approval"]
    if pending:
        print("New federated skills whose product repo is not approved:")
        for entry in pending:
            print(f"  {entry['skill']} from {entry['repo']} ({entry['path']})")

    if not edits and not pending and not report["federation_error"]:
        print("Nothing to flag: no vendored skill was edited and no new skill "
              "comes from an unapproved product repo.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    print_summary(report)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
