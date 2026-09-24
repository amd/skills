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
    A federated skill folder is a mirror of a folder in a product repo.
    `federate_skills.py` re-imports it with rmtree + copytree, so anything
    committed here is deleted by the next nightly run: an edit that looks
    merged is really just pending its own reversal. That includes the skill's
    `evals/` folder, which is imported like everything else.

    Federated means declared in `.github/federation.json`, and nothing else.
    The vendored copies carry a `.federated.json` marker too, but the marker is
    the importer's bookkeeping and the two disagree on `main` today; the
    declaration is the catalog's statement of intent, and it is the file a
    reviewer reads.

    The declarations are read from the *base* branch, so the pull request that
    federates a skill for the first time does not trip over its own new entry,
    and deleting an entry in the same commit that edits the skill does not slip
    past the rule.

Rule 2 -- a new federated skill needs its product repo approved.
    `.github/skill_owners.json` records the repos whose engineering owner and
    product release owner have both signed off, via the `product-repo-approval`
    workflow. An entry with a `path` approves only that subdirectory, for
    super-repos whose projects each have their own owners. A pull request that
    declares a skill no entry covers is asking the catalog to vendor code
    nobody has vouched for.

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


def load_approvals(registry: Path) -> set[tuple[str, str]]:
    """Return the approved scopes from `.github/skill_owners.json`.

    Each scope is (repo lowercased, subdirectory). An empty subdirectory is the
    whole repo; otherwise the approval covers only skills under that directory,
    which is how one project in a super-repo is approved without the rest.

    A missing or malformed registry reads as "nothing is approved" rather than
    as an error: the strict reading is the safe one, and `record_skill_owner.py`
    is what validates the file when it writes it. For the same reason an entry
    whose `path` is not a string is dropped, not widened to the whole repo.
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
    approvals = set()
    for entry in repos:
        if not isinstance(entry, dict) or not isinstance(entry.get("repo"), str):
            continue
        path = entry.get("path", "")
        if not isinstance(path, str):
            continue
        approvals.add((entry["repo"].strip().lower(), path.strip().strip("/")))
    return approvals


def is_approved(repo: str, skill_path: str, approvals: set[tuple[str, str]]) -> bool:
    """Whether some approval covers the skill at `skill_path` in `repo`.

    Compared segment by segment, so `projects/rocprofiler` does not cover
    `projects/rocprofiler-sdk`.
    """
    parts = skill_path.strip("/").split("/")
    for approved_repo, subdir in approvals:
        if approved_repo != repo.lower():
            continue
        if not subdir:
            return True
        scope = subdir.split("/")
        if parts[: len(scope)] == scope:
            return True
    return False


def source_ref(skill_dir: Path, branch: str) -> str:
    """A ref that github.com can browse for a skill tracked at `branch`.

    A pattern is not a ref, so it is swapped for the release branch the last
    import resolved it to, or for the repo's default branch before any import.
    """
    if not fed.branches.is_pattern(branch):
        return branch
    return fed.read_marker(skill_dir).get("resolved_ref") or "HEAD"


def vendored_edits(changed: list[str], declared: dict[str, dict]) -> list[dict]:
    """Group the changed paths that edit a vendored skill, by skill."""
    hits: dict[str, dict] = {}
    for path in changed:
        if not path.startswith(SKILLS_PREFIX):
            continue
        name, _, tail = path[len(SKILLS_PREFIX) :].partition("/")
        if not tail:
            # Something directly under `skills/`, not inside a skill.
            continue
        entry = declared.get(name)
        if entry is None:
            continue
        branch = entry.get("branch", fed.branches.DEFAULT_BRANCH)
        hit = hits.setdefault(
            name,
            {
                "skill": name,
                "repo": entry["repo"],
                "source_path": entry["path"],
                "branch": branch,
                "source_ref": entry.get("source_ref", branch),
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
    approvals: set[tuple[str, str]],
) -> list[dict]:
    return [
        entry
        for key, entry in sorted(head.items())
        if key not in base and not is_approved(entry["repo"], entry["path"], approvals)
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
        "vendored_edits": [],
        "new_skills_needing_approval": [],
        "federation_error": "",
    }

    # `.github/federation.json` on the base branch is the only thing that says
    # which skills are federated, so an unreadable one leaves both rules with
    # nothing to go on. Report that instead of passing every pull request.
    try:
        base_sources = fed.parse_federation(base_dir / ".github" / "federation.json")
    except (ValueError, FileNotFoundError) as exc:
        report["federation_error"] = f"base branch copy: {exc}"
        return report
    base = declared_skills(base_sources)
    branch_of = {source.repo: source.branch for source in base_sources}

    report["vendored_edits"] = vendored_edits(
        read_changed_files(args.changed_files),
        {
            entry["skill"]: {
                **entry,
                "branch": branch_of[entry["repo"]],
                "source_ref": source_ref(
                    base_dir / SKILLS_PREFIX / entry["skill"], branch_of[entry["repo"]]
                ),
            }
            for entry in base.values()
        },
    )

    if args.head_federation is None:
        return report

    try:
        head = declared_skills(fed.parse_federation(args.head_federation))
    except (ValueError, FileNotFoundError) as exc:
        # The `validate` workflow is what explains a malformed federation file
        # in detail; recording it here keeps this rule from passing a pull
        # request whose declarations could not be read at all.
        report["federation_error"] = str(exc)
        return report

    report["new_skills_needing_approval"] = new_skills_needing_approval(
        base,
        head,
        load_approvals(base_dir / ".github" / "skill_owners.json"),
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
