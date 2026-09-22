#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0"]
# ///
"""Tests for the federation machinery. No network, no clones, no tokens.

    uv run .github/scripts/test_federate_skills.py

Two behaviors carry the whole mechanism and both fail silently when broken,
so they are guarded here:

  - Change detection. The nightly workflow opens a pull request whenever the
    work tree is dirty, so a content hash that moves for the wrong reason
    turns every quiet night into a no-op pull request. That includes moving
    between platforms: the hash has to agree between a maintainer's machine
    and the Linux runner.
  - Branch tracking. A source follows `main` unless it names a branch, and a
    release pattern such as `release/*` follows the newest release by version,
    not alphabetically. A tag or commit must fail the run, not get silently
    coerced to `main`.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import federate_skills as fed  # noqa: E402


def parse(payload: dict) -> list[fed.Source]:
    with tempfile.TemporaryDirectory() as tmp:
        catalog = Path(tmp) / "federation.json"
        catalog.write_text(json.dumps(payload), encoding="utf-8")
        return fed.parse_federation(catalog)


def one_source(**overrides) -> dict:
    source = {
        "repo": "AMD-Org/MyProject",
        "license": "MIT",
        "skills": [{"path": "skills/my-skill", "as": "myproject-my-skill"}],
    }
    source.update(overrides)
    return {"sources": [source]}


def write_skill(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


class TestFederationFile(unittest.TestCase):
    def test_declares_each_skill_by_its_full_path(self):
        sources = parse(
            one_source(
                skills=[
                    {"path": "skills/first"},
                    {"path": "tools/agents/skills/second", "as": "myproject-second"},
                ]
            )
        )
        self.assertEqual(len(sources), 1)
        first, second = sources[0].skills
        self.assertEqual(first.path, "skills/first")
        self.assertEqual(first.folder, "first")
        self.assertEqual(first.dest_name, "first")
        self.assertEqual(second.folder, "second")
        self.assertEqual(second.dest_name, "myproject-second")

    def test_source_slug_derives_from_repo(self):
        self.assertEqual(parse(one_source())[0].name, "amd-org-myproject")

    def test_tracks_main_unless_a_branch_is_set(self):
        self.assertEqual(parse(one_source())[0].branch, "main")
        for branch in ("develop", "release/1.0", "release/*", "releases/v*"):
            with self.subTest(branch=branch):
                self.assertEqual(parse(one_source(branch=branch))[0].branch, branch)

    def test_rejects_tags_commits_and_other_pins(self):
        # Only a branch can be tracked. Naming a tag or commit must fail
        # loudly rather than quietly track `main` instead.
        for key in ("ref", "tag", "commit", "rev"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as ctx:
                    parse(one_source(**{key: "v1.0"}))
                self.assertIn("branch", str(ctx.exception))

    def test_rejects_branches_git_would_misread(self):
        for branch in (
            "",
            " ",
            "-rf",
            "main..evil",
            "release/",
            "release//1",
            "a b",
            "release/*/*",
            "main.lock",
            "$(whoami)",
            7,
        ):
            with self.subTest(branch=branch):
                with self.assertRaises(ValueError):
                    parse(one_source(branch=branch))


    def test_rejects_the_previous_yaml_schema(self):
        # A source-level `path` plus skills named by folder is how the old
        # catalog addressed skills. Reject it so a stale entry can't resolve
        # to the wrong folder.
        with self.assertRaises(ValueError):
            parse(one_source(name="amd-myproject", path="skills"))
        with self.assertRaises(ValueError):
            parse(one_source(skills=[{"name": "my-skill"}]))
        with self.assertRaises(ValueError):
            parse(one_source(skills=["my-skill"]))

    def test_rejects_malformed_entries(self):
        for payload in (
            {},
            {"sources": []},
            one_source(repo="not-a-repo"),
            one_source(repo=None),
            one_source(skills=[]),
            one_source(skills=[{"path": ""}]),
            one_source(skills=[{"path": "skills/../../etc"}]),
            one_source(skills=[{"path": "skills/x", "as": "nested/name"}]),
            one_source(skills=[{"path": "skills/x", "typo": 1}]),
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    parse(payload)

    def test_accepts_windows_style_separators_in_paths(self):
        skills = parse(
            one_source(skills=[{"path": r"TraceLens\Agent\skills\orchestrator"}])
        )[0].skills
        self.assertEqual(skills[0].path, "TraceLens/Agent/skills/orchestrator")


class TestReleaseBranches(unittest.TestCase):
    QUARK = ["main", "release/0.8", "release/0.9", "release/0.10", "release/0.12", "release/0.11"]

    def test_the_newest_release_wins_by_version_not_alphabet(self):
        # Alphabetically `release/0.9` sorts last; by version it is old.
        self.assertEqual(
            fed.branches.latest_matching("release/*", self.QUARK), "release/0.12"
        )

    def test_only_version_numbers_match_the_wildcard(self):
        names = self.QUARK + ["release/next", "release/0.13-wip", "release/0.13/x"]
        self.assertEqual(fed.branches.latest_matching("release/*", names), "release/0.12")

    def test_multi_part_and_v_prefixed_versions(self):
        names = ["rel-v1.2.9", "rel-v1.10", "rel-v1.2.10", "rel-2"]
        self.assertEqual(fed.branches.latest_matching("rel-*", names), "rel-2")
        self.assertEqual(
            fed.branches.latest_matching("rel-*", names[:3]), "rel-v1.10"
        )

    def test_no_match_is_none(self):
        self.assertIsNone(fed.branches.latest_matching("release/*", ["main", "dev"]))

    def test_a_plain_branch_is_used_as_is(self):
        source = parse(one_source(branch="develop"))[0]
        self.assertEqual(fed.resolve_branch(source), "develop")

    def test_the_marker_records_what_a_pattern_resolved_to(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill = Path(tmp)
            source = parse(one_source(branch="release/*"))[0]
            fed.write_marker(skill, source, "release/0.12", "abc", "skills/x", "h")
            marker = fed.read_marker(skill)
            self.assertEqual(marker["ref"], "release/*")
            self.assertEqual(marker["resolved_ref"], "release/0.12")

            fed.write_marker(skill, parse(one_source())[0], "main", "abc", "skills/x", "h")
            self.assertNotIn("resolved_ref", fed.read_marker(skill))


class TestVendoredCopy(unittest.TestCase):
    def test_files_absent_upstream_are_deleted_from_the_vendored_copy(self):
        # The vendored folder mirrors upstream, so a re-import is also how a
        # deletion propagates.
        with tempfile.TemporaryDirectory() as tmp:
            src = write_skill(
                Path(tmp) / "src", {"SKILL.md": "new", "agents/one.md": "x"}
            )
            dest = write_skill(
                Path(tmp) / "dest", {"SKILL.md": "old", "stale.md": "gone"}
            )
            fed.copy_skill(src, dest)
            self.assertEqual(
                sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*")),
                ["SKILL.md", "agents", "agents/one.md"],
            )
            self.assertEqual((dest / "SKILL.md").read_text(encoding="utf-8"), "new")

    def test_evals_are_outside_federation_in_both_directions(self):
        # The catalog owns and runs the datasets, so a re-import must neither
        # delete the local `evals/` nor import upstream's over it. Both halves
        # fail silently: a wiped dataset only shows up as an eval that stopped
        # running, and an imported one as expectations nobody here wrote.
        with tempfile.TemporaryDirectory() as tmp:
            src = write_skill(
                Path(tmp) / "src",
                {"SKILL.md": "new", "evals/evals.json": '{"upstream": true}'},
            )
            dest = write_skill(
                Path(tmp) / "dest",
                {"SKILL.md": "old", "evals/evals.json": '{"local": true}'},
            )
            fed.copy_skill(src, dest)
            self.assertEqual(
                (dest / "evals" / "evals.json").read_text(encoding="utf-8"),
                '{"local": true}',
            )

    def test_a_nested_evals_folder_is_ordinary_content(self):
        # Only the skill's own top-level `evals/` is the catalog's; a folder
        # of the same name deeper in the tree is upstream's to ship.
        with tempfile.TemporaryDirectory() as tmp:
            src = write_skill(
                Path(tmp) / "src", {"SKILL.md": "new", "agents/evals/notes.md": "x"}
            )
            dest = write_skill(Path(tmp) / "dest", {"SKILL.md": "old"})
            fed.copy_skill(src, dest)
            self.assertTrue((dest / "agents" / "evals" / "notes.md").is_file())


class TestChangeDetection(unittest.TestCase):
    def test_hash_order_does_not_depend_on_the_platform(self):
        # `sorted()` over Path objects is case-insensitive on Windows, so it
        # feeds `SKILL.md` and `agents/one.md` to the digest in one order
        # there and the opposite order on Linux. Pin the ordering to the
        # POSIX relative path so both platforms agree; otherwise a runner
        # re-hashes identical files to a different digest and re-vendors.
        with tempfile.TemporaryDirectory() as tmp:
            root = write_skill(
                Path(tmp) / "skill", {"SKILL.md": "body", "agents/one.md": "x"}
            )
            expected = hashlib.sha256()
            for rel in ("SKILL.md", "agents/one.md"):
                expected.update(rel.encode("utf-8"))
                expected.update(b"\0")
                expected.update(hashlib.sha256((root / rel).read_bytes()).digest())
            self.assertEqual(fed.content_hash(root), expected.hexdigest())

    def test_hash_skips_excluded_paths_and_build_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_skill(Path(tmp) / "skill", {"SKILL.md": "body"})
            digest = fed.content_hash(root, exclude=[fed.MARKER_FILENAME])
            write_skill(
                root,
                {
                    fed.MARKER_FILENAME: '{"commit": "abc"}',
                    "evals/__pycache__/evals.pyc": "junk",
                },
            )
            self.assertEqual(
                fed.content_hash(root, exclude=[fed.MARKER_FILENAME]), digest
            )

    def test_hash_ignores_the_skills_own_evals_folder(self):
        # Federation does not move `evals/` any more, so an upstream edit
        # there must not re-vendor the skill, and the local dataset must not
        # make the vendored copy look stale on every run.
        with tempfile.TemporaryDirectory() as tmp:
            root = write_skill(Path(tmp) / "skill", {"SKILL.md": "body"})
            digest = fed.content_hash(root)
            write_skill(root, {"evals/evals.json": '{"evaluations": []}'})
            self.assertEqual(fed.content_hash(root), digest)
            # Same name, but nested: that is upstream's content and counts.
            write_skill(root, {"agents/evals/notes.md": "x"})
            self.assertNotEqual(fed.content_hash(root), digest)

    def test_hash_covers_contents_and_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = write_skill(
                Path(tmp) / "a", {"SKILL.md": "body", "agents/one.md": "x"}
            )
            same = write_skill(
                Path(tmp) / "b", {"SKILL.md": "body", "agents/one.md": "x"}
            )
            edited = write_skill(
                Path(tmp) / "c", {"SKILL.md": "body!", "agents/one.md": "x"}
            )
            renamed = write_skill(
                Path(tmp) / "d", {"SKILL.md": "body", "agents/two.md": "x"}
            )
            added = write_skill(
                Path(tmp) / "e",
                {"SKILL.md": "body", "agents/one.md": "x", "extra.md": ""},
            )
            digest = fed.content_hash(base)
            self.assertEqual(digest, fed.content_hash(same))
            for other in (edited, renamed, added):
                self.assertNotEqual(digest, fed.content_hash(other))

    def test_up_to_date_only_when_hash_repo_path_and_ref_all_match(self):
        source = parse(one_source())[0]
        spec = source.skills[0]
        marker = {
            "repo": "AMD-Org/MyProject",
            "ref": "main",
            "path": "skills/my-skill",
            "content_hash": "abc",
        }
        self.assertTrue(fed.is_up_to_date(marker, source, spec, "abc"))
        # A marker written before content hashing existed re-vendors once.
        legacy = {k: v for k, v in marker.items() if k != "content_hash"}
        self.assertFalse(fed.is_up_to_date(legacy, source, spec, "abc"))
        for key, value in (
            ("content_hash", "def"),
            ("repo", "AMD-Org/Other"),
            ("path", "skills/elsewhere"),
            ("ref", "release/1.0"),
        ):
            with self.subTest(key=key):
                self.assertFalse(
                    fed.is_up_to_date({**marker, key: value}, source, spec, "abc")
                )

    def test_a_new_release_branch_with_the_same_contents_is_not_a_bump(self):
        # The marker is compared on the declared pattern, not on the release
        # it resolved to, so `release/0.13` arriving with an identical skill
        # folder leaves the vendored copy alone.
        source = parse(one_source(branch="release/*"))[0]
        marker = {
            "repo": "AMD-Org/MyProject",
            "ref": "release/*",
            "resolved_ref": "release/0.12",
            "path": "skills/my-skill",
            "content_hash": "abc",
        }
        self.assertTrue(fed.is_up_to_date(marker, source, source.skills[0], "abc"))
        self.assertFalse(
            fed.is_up_to_date({**marker, "ref": "main"}, source, source.skills[0], "abc")
        )


class TestPullRequestSummary(unittest.TestCase):
    def result(
        self,
        folder: str,
        commit: str,
        updated: bool = True,
        branch: str = "main",
        resolved: str | None = None,
    ):
        source = fed.Source(repo="AMD-Org/MyProject", license="MIT", branch=branch)
        return fed.ImportResult(
            source=source,
            folder=folder,
            path=f"skills/{folder}",
            commit=commit,
            updated=updated,
            branch=resolved or branch,
        )

    def test_the_body_names_the_branch_each_bump_came_from(self):
        summary = fed.build_summary(
            [
                self.result("a", "1111111"),
                self.result("b", "2222222", branch="release/*", resolved="release/0.12"),
            ]
        )
        self.assertIn("on `main`", summary["body"])
        self.assertIn("on `release/0.12` (newest `release/*`)", summary["body"])
        self.assertEqual(
            [u["branch"] for u in summary["updated"]], ["main", "release/0.12"]
        )

    def test_single_bump_names_the_skill_and_short_commit(self):
        summary = fed.build_summary([self.result("my-skill", "f4af496dbe9c")])
        self.assertTrue(summary["changed"])
        self.assertEqual(summary["title"], "Bump `my-skill` to `f4af496`")
        self.assertIn("f4af496", summary["body"])

    def test_nothing_changed_means_no_pull_request(self):
        summary = fed.build_summary(
            [self.result("my-skill", "f4af496dbe9c", updated=False)]
        )
        self.assertFalse(summary["changed"])
        self.assertEqual(summary["title"], "")
        self.assertEqual(summary["unchanged"], ["my-skill"])

    def test_several_bumps_share_one_title(self):
        two = [self.result("a", "1111111"), self.result("b", "2222222")]
        self.assertEqual(fed.build_summary(two)["title"], "Bump 2 federated skills")


if __name__ == "__main__":
    unittest.main(verbosity=2)
