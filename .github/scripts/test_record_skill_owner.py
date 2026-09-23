#!/usr/bin/env python3
"""Tests for how approvals are keyed in the skill owners registry.

    uv run .github/scripts/test_record_skill_owner.py

The federation guard trusts whatever this writes, so the cases pinned here are
the ones that decide what an approval clears: one project of a super-repo must
not overwrite, or be overwritten by, another.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import record_skill_owner as rec  # noqa: E402


def record(registry: Path, repo: str, path: str = "", owner: str = "octocat") -> None:
    argv = [
        "--registry", str(registry),
        "--repo", repo,
        "--engineering-owner", f"@{owner}",
        "--product-release-owner", owner,
    ]
    if path:
        argv += ["--path", path]
    rec.main(argv)


def entries(registry: Path) -> list[dict]:
    return json.loads(registry.read_text(encoding="utf-8"))["repos"]


class TestRecordSkillOwner(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.registry = Path(self._tmp.name) / "skill_owners.json"

    def tearDown(self):
        self._tmp.cleanup()

    def test_a_whole_repo_approval_has_no_path(self):
        record(self.registry, "AMD-AGI/TraceLens")
        self.assertEqual(
            entries(self.registry),
            [
                {
                    "repo": "AMD-AGI/TraceLens",
                    "engineering_owner": "octocat",
                    "product_release_owner": "octocat",
                }
            ],
        )

    def test_projects_of_one_super_repo_are_separate_entries(self):
        record(self.registry, "ROCm/rocm-systems", "projects/rocprofiler-sdk", "alice")
        record(self.registry, "ROCm/rocm-systems", "projects/rocm-smi", "bob")
        self.assertEqual(
            [(e["path"], e["engineering_owner"]) for e in entries(self.registry)],
            [("projects/rocm-smi", "bob"), ("projects/rocprofiler-sdk", "alice")],
        )

    def test_re_approving_a_project_replaces_its_entry(self):
        record(self.registry, "ROCm/rocm-systems", "projects/rocprofiler-sdk", "alice")
        record(self.registry, "rocm/rocm-systems", "/projects/rocprofiler-sdk/", "carol")
        self.assertEqual(
            entries(self.registry),
            [
                {
                    "repo": "rocm/rocm-systems",
                    "path": "projects/rocprofiler-sdk",
                    "engineering_owner": "carol",
                    "product_release_owner": "carol",
                }
            ],
        )

    def test_a_path_that_escapes_the_directory_is_refused(self):
        for bad in ("projects/../other", "./projects", "projects//x", "a b"):
            with self.subTest(path=bad), self.assertRaises(SystemExit):
                record(self.registry, "ROCm/rocm-systems", bad)
        self.assertFalse(self.registry.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
