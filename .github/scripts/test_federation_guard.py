#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0"]
# ///
"""Tests for the federation guard's two rules. No network, no tokens.

    uv run .github/scripts/test_federation_guard.py

Both rules are enforced by closing a pull request or failing a required check,
so a wrong answer is expensive in either direction: a false positive turns away
a contributor, and a false negative lets an edit land that the next nightly run
silently deletes, or vendors a skill nobody approved. The cases that decide
which way the guard reads are pinned here.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import federation_guard as guard  # noqa: E402

TRACELENS = "AMD-AGI/TraceLens"


def source(repo: str, *skills: dict, branch: str | None = None) -> dict:
    entry = {"repo": repo, "license": "MIT", "skills": list(skills)}
    if branch is not None:
        entry["branch"] = branch
    return entry


def build_base(
    tmp: Path,
    *,
    vendored: dict[str, str] | None = None,
    local: tuple[str, ...] = (),
    sources: list[dict] | None = None,
    approved: tuple[str, ...] | dict[str, str | None] = (),
) -> Path:
    """Lay out a base-branch checkout for the guard to read.

    `vendored` maps a skill folder to the repo that federates it, and produces
    both the folder (marker included, as the importer writes it) and the
    `federation.json` entry that declares it. `local` names skills authored in
    the catalog, which nothing declares. Pass `sources` to set the declarations
    independently of what is on disk, which is how the two get to disagree.
    `approved` is a tuple of repos approved for `main`, or a mapping of repo to
    the branch its registry entry records (None for an entry without one).
    """
    if not isinstance(approved, dict):
        approved = dict.fromkeys(approved)
    base = tmp / "base"
    (base / ".github").mkdir(parents=True)
    vendored = vendored or {}
    for name, repo in vendored.items():
        skill = base / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: x\n---\n", encoding="utf-8")
        (skill / guard.fed.MARKER_FILENAME).write_text(
            json.dumps({"repo": repo, "path": f"skills/{name}", "ref": "main"}),
            encoding="utf-8",
        )
    for name in local:
        skill = base / "skills" / name
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: x\n---\n", encoding="utf-8")
    if sources is None:
        by_repo: dict[str, list[dict]] = {}
        for name, repo in vendored.items():
            by_repo.setdefault(repo, []).append({"path": f"skills/{name}", "as": name})
        sources = [source(repo, *skills) for repo, skills in by_repo.items()] or [
            source(TRACELENS, {"path": "agent/orchestrator"})
        ]
    (base / ".github" / "federation.json").write_text(
        json.dumps({"sources": sources}),
        encoding="utf-8",
    )
    (base / ".github" / "skill_owners.json").write_text(
        json.dumps(
            {
                "repos": [
                    {
                        "repo": repo,
                        "engineering_owner": "octocat",
                        "product_release_owner": "octocat",
                        **({"branch": branch} if branch else {}),
                    }
                    for repo, branch in approved.items()
                ]
            }
        ),
        encoding="utf-8",
    )
    return base


def report(
    tmp: Path,
    base: Path,
    changed: tuple[str, ...] = (),
    head_sources: list[dict] | None | str = None,
) -> dict:
    """Run the guard the way the workflow does, on files in `tmp`."""
    changed_file = tmp / "changed.txt"
    changed_file.write_text("\n".join(changed) + "\n", encoding="utf-8")
    argv = ["--changed-files", str(changed_file), "--base-dir", str(base)]
    if head_sources is not None:
        head = tmp / "head-federation.json"
        head.write_text(
            head_sources
            if isinstance(head_sources, str)
            else json.dumps({"sources": head_sources}),
            encoding="utf-8",
        )
        argv += ["--head-federation", str(head)]
    return guard.build_report(guard.parse_args(argv))


class TestVendoredEdits(unittest.TestCase):
    def test_an_edit_to_a_vendored_skill_is_reported_with_its_source(self):
        # The message this feeds has to tell the contributor where to go
        # instead, so the source repo and path come along with the hit.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp, vendored={"tracelens-orchestrator": TRACELENS})
            edits = report(
                tmp,
                base,
                changed=(
                    "skills/tracelens-orchestrator/SKILL.md",
                    "skills/tracelens-orchestrator/agents/gemm.md",
                ),
            )["vendored_edits"]
            self.assertEqual(len(edits), 1)
            self.assertEqual(edits[0]["repo"], TRACELENS)
            self.assertEqual(edits[0]["source_path"], "skills/tracelens-orchestrator")
            self.assertEqual(
                edits[0]["paths"],
                [
                    "skills/tracelens-orchestrator/SKILL.md",
                    "skills/tracelens-orchestrator/agents/gemm.md",
                ],
            )

    def test_the_skills_own_evals_folder_is_the_catalogs_to_edit(self):
        # Federation neither imports nor overwrites a skill's top-level
        # `evals/`, so editing it here is the only way to edit it. Closing
        # those pull requests would leave the datasets unmaintainable.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp, vendored={"tracelens-orchestrator": TRACELENS})
            self.assertEqual(
                report(
                    tmp,
                    base,
                    changed=(
                        "skills/tracelens-orchestrator/evals/evals.json",
                        "skills/tracelens-orchestrator/evals/machine.yml",
                    ),
                )["vendored_edits"],
                [],
            )
            # A folder of the same name deeper in the tree is upstream's.
            self.assertEqual(
                len(
                    report(
                        tmp,
                        base,
                        changed=("skills/tracelens-orchestrator/agents/evals/notes.md",),
                    )["vendored_edits"]
                ),
                1,
            )

    def test_a_declaration_alone_makes_a_skill_federated(self):
        # `hyperloom-workload-optimizer` on main: declared in federation.json,
        # shipped without a marker. It is the case the importer treats as
        # stale, so it is the skill most certain to be overwritten on the next
        # nightly run -- and the one a marker-based rule would have missed.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(
                tmp,
                local=("hyperloom-optimizer",),
                sources=[
                    source(
                        "AMD-AGI/Hyperloom",
                        {
                            "path": "examples/skills/optimizer",
                            "as": "hyperloom-optimizer",
                        },
                    )
                ],
            )
            edits = report(
                tmp, base, changed=("skills/hyperloom-optimizer/SKILL.md",)
            )["vendored_edits"]
            self.assertEqual(len(edits), 1)
            self.assertEqual(edits[0]["repo"], "AMD-AGI/Hyperloom")
            self.assertEqual(edits[0]["source_path"], "examples/skills/optimizer")

    def test_a_marker_no_source_declares_does_not_make_a_skill_federated(self):
        # `magpie-kernel-evaluator` on main: a vendored copy left behind by a
        # source nobody declares any more. Federation no longer touches it, and
        # the declarations are the only thing this rule reads, so it is editable
        # here like any other in-repo skill. Re-declaring it protects it again.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(
                tmp,
                vendored={"magpie-evaluator": "AMD-AGI/Magpie"},
                sources=[source(TRACELENS, {"path": "agent/orchestrator"})],
            )
            self.assertEqual(
                report(tmp, base, changed=("skills/magpie-evaluator/SKILL.md",))[
                    "vendored_edits"
                ],
                [],
            )

    def test_skills_authored_in_this_catalog_are_still_edited_here(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp, local=("local-ai-use",))
            self.assertEqual(
                report(tmp, base, changed=("skills/local-ai-use/SKILL.md",))[
                    "vendored_edits"
                ],
                [],
            )

    def test_the_pull_request_that_first_vendors_a_skill_is_not_closed(self):
        # The declaration arrives with that pull request, so the rule reads the
        # base branch. Reading the pull request's own tree would make federating
        # a new skill impossible: it would close itself over its own new entry.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp, vendored={"tracelens-orchestrator": TRACELENS})
            self.assertEqual(
                report(
                    tmp,
                    base,
                    changed=(
                        "skills/hyperloom-optimizer/SKILL.md",
                        "skills/hyperloom-optimizer/.federated.json",
                    ),
                    head_sources=[
                        source(TRACELENS, {"path": "skills/tracelens-orchestrator"}),
                        source(
                            "AMD-AGI/Hyperloom",
                            {"path": "examples/optimizer", "as": "hyperloom-optimizer"},
                        ),
                    ],
                )["vendored_edits"],
                [],
            )

    def test_dropping_the_declaration_in_the_same_pull_request_does_not_help(self):
        # The other half of reading the base branch: if the pull request's own
        # federation.json counted, an author could delete the entry in the
        # commit that edits the skill and the rule would find nothing.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp, vendored={"tracelens-orchestrator": TRACELENS})
            edits = report(
                tmp,
                base,
                changed=(
                    "skills/tracelens-orchestrator/SKILL.md",
                    ".github/federation.json",
                ),
                head_sources=[source(TRACELENS, {"path": "agent/other", "as": "other"})],
            )["vendored_edits"]
            self.assertEqual([e["skill"] for e in edits], ["tracelens-orchestrator"])

    def test_changes_outside_a_skill_folder_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp, vendored={"tracelens-orchestrator": TRACELENS})
            self.assertEqual(
                report(
                    tmp,
                    base,
                    changed=("README.md", "skills/README.md", "docs/evals.md"),
                )["vendored_edits"],
                [],
            )


class TestProductRepoApproval(unittest.TestCase):
    def test_a_new_skill_from_an_unapproved_repo_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp)
            pending = report(
                tmp,
                base,
                head_sources=[
                    source(TRACELENS, {"path": "agent/orchestrator"}),
                    source("AMD-Org/MyProject", {"path": "skills/mine", "as": "mp-mine"}),
                ],
            )["new_skills_needing_approval"]
            self.assertEqual(
                pending,
                [{"skill": "mp-mine", "repo": "AMD-Org/MyProject", "path": "skills/mine"}],
            )

    def test_an_approved_repo_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            # The registry is matched case-insensitively: a handle typed with
            # different capitalization is the same repo on GitHub.
            base = build_base(tmp, approved=("amd-org/myproject",))
            self.assertEqual(
                report(
                    tmp,
                    base,
                    head_sources=[
                        source(TRACELENS, {"path": "agent/orchestrator"}),
                        source("AMD-Org/MyProject", {"path": "skills/mine"}),
                    ],
                )["new_skills_needing_approval"],
                [],
            )

    def test_skills_already_declared_are_grandfathered(self):
        # TraceLens predates the approval process and is not in the registry.
        # Its existing skill must keep merging, or the gate breaks the catalog
        # it was added to protect.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp)
            self.assertEqual(
                report(
                    tmp,
                    base,
                    changed=(".github/federation.json",),
                    head_sources=[source(TRACELENS, {"path": "agent/orchestrator"})],
                )["new_skills_needing_approval"],
                [],
            )

    def test_moving_an_existing_skills_upstream_path_is_not_a_new_skill(self):
        # Upstream reorganizing its tree is maintenance on a skill the catalog
        # already ships, not a new one.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(
                tmp,
                sources=[
                    source(TRACELENS, {"path": "agent/orchestrator", "as": "tl-orch"})
                ],
            )
            self.assertEqual(
                report(
                    tmp,
                    base,
                    head_sources=[
                        source(
                            TRACELENS,
                            {"path": "tools/agents/orchestrator", "as": "tl-orch"},
                        )
                    ],
                )["new_skills_needing_approval"],
                [],
            )

    def test_pointing_an_existing_name_at_another_repo_needs_approval(self):
        # Same local name, different product repo: the question of who vouches
        # for the code is open again, so grandfathering the name is not enough.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(
                tmp,
                sources=[
                    source(TRACELENS, {"path": "agent/orchestrator", "as": "tl-orch"})
                ],
            )
            pending = report(
                tmp,
                base,
                head_sources=[
                    source("AMD-Org/Fork", {"path": "agent/orchestrator", "as": "tl-orch"})
                ],
            )["new_skills_needing_approval"]
            self.assertEqual([p["repo"] for p in pending], ["AMD-Org/Fork"])

    def test_a_second_skill_from_an_unapproved_repo_still_needs_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp)
            pending = report(
                tmp,
                base,
                head_sources=[
                    source(
                        TRACELENS,
                        {"path": "agent/orchestrator"},
                        {"path": "agent/second", "as": "tl-second"},
                    )
                ],
            )["new_skills_needing_approval"]
            self.assertEqual([p["skill"] for p in pending], ["tl-second"])

    def test_the_pull_requests_own_registry_entry_does_not_count(self):
        # The registry is read from the base branch. Otherwise a pull request
        # could add its repo to `skill_owners.json` alongside its skill and
        # approve itself, which is the whole gate.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp)
            pending = report(
                tmp,
                base,
                changed=(".github/federation.json", ".github/skill_owners.json"),
                head_sources=[source("AMD-Org/MyProject", {"path": "skills/mine"})],
            )["new_skills_needing_approval"]
            self.assertEqual([p["repo"] for p in pending], ["AMD-Org/MyProject"])

    def test_a_pull_request_that_leaves_the_federation_file_alone_is_not_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp, local=("local-ai-use",))
            result = report(tmp, base, changed=("skills/local-ai-use/SKILL.md",))
            self.assertEqual(result["new_skills_needing_approval"], [])
            self.assertEqual(result["federation_error"], "")

    def test_an_unreadable_federation_file_is_reported_rather_than_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp)
            result = report(tmp, base, head_sources="{not json")
            self.assertTrue(result["federation_error"])
            self.assertEqual(result["new_skills_needing_approval"], [])

    def test_an_unreadable_file_on_the_base_branch_is_reported_too(self):
        # The declarations are the only thing either rule reads, so a base
        # branch whose federation.json cannot be parsed has to fail the check
        # rather than quietly find every skill unfederated and every skill old.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp, vendored={"tracelens-orchestrator": TRACELENS})
            (base / ".github" / "federation.json").write_text(
                "{not json", encoding="utf-8"
            )
            result = report(
                tmp, base, changed=("skills/tracelens-orchestrator/SKILL.md",)
            )
            self.assertTrue(result["federation_error"])
            self.assertEqual(result["vendored_edits"], [])


QUARK = "amd/Quark"
QUARK_SKILL = {"path": ".claude/skills/quark-install", "as": "quark-install"}


class TestBranchApproval(unittest.TestCase):
    def branches(self, tmp: Path, base: Path, head_sources: list[dict]) -> list[dict]:
        return report(tmp, base, head_sources=head_sources)["branches_needing_approval"]

    def test_a_source_on_its_approved_release_pattern_passes(self):
        # Quark ships skills with its releases and has no `main` at all, so
        # its owners approve `release/*` and the entry tracks exactly that.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp, approved={QUARK: "release/*"})
            result = report(
                tmp,
                base,
                head_sources=[
                    source(TRACELENS, {"path": "agent/orchestrator"}),
                    source(QUARK, QUARK_SKILL, branch="release/*"),
                ],
            )
            self.assertEqual(result["new_skills_needing_approval"], [])
            self.assertEqual(result["branches_needing_approval"], [])

    def test_leaving_the_branch_out_means_main_which_was_not_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp, approved={QUARK: "release/*"})
            self.assertEqual(
                self.branches(
                    tmp,
                    base,
                    [
                        source(TRACELENS, {"path": "agent/orchestrator"}),
                        source(QUARK, QUARK_SKILL),
                    ],
                ),
                [{"repo": QUARK, "branch": "main", "approved_branch": "release/*"}],
            )

    def test_an_entry_without_a_branch_approves_main_only(self):
        # Every registry entry written before branches existed approves `main`,
        # and must not read as approval for any branch at all.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(
                tmp,
                sources=[source("AMD-Org/MyProject", {"path": "skills/mine"})],
                approved=("AMD-Org/MyProject",),
            )
            flagged = self.branches(
                tmp,
                base,
                [source("AMD-Org/MyProject", {"path": "skills/mine"}, branch="develop")],
            )
            self.assertEqual(
                flagged,
                [{"repo": "AMD-Org/MyProject", "branch": "develop", "approved_branch": "main"}],
            )
            self.assertEqual(
                self.branches(
                    tmp, base, [source("AMD-Org/MyProject", {"path": "skills/mine"}, branch="main")]
                ),
                [],
            )

    def test_a_grandfathered_source_cannot_switch_branch_without_approval(self):
        # TraceLens predates the registry. Keeping it on `main` needs nothing;
        # moving it is a change nobody has signed off on.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp)
            self.assertEqual(
                self.branches(
                    tmp, base, [source(TRACELENS, {"path": "agent/orchestrator"}, branch="dev")]
                ),
                [{"repo": TRACELENS, "branch": "dev", "approved_branch": None}],
            )

    def test_a_branch_the_pull_request_leaves_alone_is_not_its_problem(self):
        # The registry moved to `release/*` after the source was declared on
        # `main`. An unrelated pull request should not be the one that fails.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(
                tmp,
                sources=[source(QUARK, QUARK_SKILL)],
                approved={QUARK: "release/*"},
            )
            self.assertEqual(self.branches(tmp, base, [source(QUARK, QUARK_SKILL)]), [])

    def test_an_unapproved_new_repo_is_reported_once_by_the_repo_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(tmp)
            result = report(
                tmp,
                base,
                head_sources=[
                    source(TRACELENS, {"path": "agent/orchestrator"}),
                    source(QUARK, QUARK_SKILL, branch="release/*"),
                ],
            )
            self.assertEqual(
                [p["repo"] for p in result["new_skills_needing_approval"]], [QUARK]
            )
            self.assertEqual(result["branches_needing_approval"], [])

    def test_the_pull_requests_own_registry_branch_does_not_count(self):
        # Same as for repos: only the base branch's registry approves.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(
                tmp, sources=[source(QUARK, QUARK_SKILL)], approved=(QUARK,)
            )
            self.assertEqual(
                [
                    b["branch"]
                    for b in self.branches(
                        tmp, base, [source(QUARK, QUARK_SKILL, branch="release/*")]
                    )
                ],
                ["release/*"],
            )

    def test_an_edit_to_a_release_tracked_skill_links_to_its_release(self):
        # `release/*` is not something github.com can browse, so the redirect
        # points at the release branch the last import actually came from.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = build_base(
                tmp,
                local=("quark-install",),
                sources=[source(QUARK, QUARK_SKILL, branch="release/*")],
            )
            (base / "skills" / "quark-install" / guard.fed.MARKER_FILENAME).write_text(
                json.dumps({"ref": "release/*", "resolved_ref": "release/0.12"}),
                encoding="utf-8",
            )
            edits = report(tmp, base, changed=("skills/quark-install/SKILL.md",))[
                "vendored_edits"
            ]
            self.assertEqual(edits[0]["branch"], "release/*")
            self.assertEqual(edits[0]["source_ref"], "release/0.12")


if __name__ == "__main__":
    unittest.main(verbosity=2)
