#!/usr/bin/env python3
"""Tests for the eval section `skill-card-evals` writes into a skill card.

    uv run .github/scripts/test_skill_card_evals.py

The fixtures are lines lifted from real `evals` runs, so a change to how
skillscope prints a result fails here rather than as a card that quietly
reports nothing.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import skill_card_evals as sce  # noqa: E402

LINUX_LOG = """\
2026-10-29T20:12:56.8776433Z   [PASS] rocm-hip-no-binary-for-gpu: 4/4 checks in 28.96s
2026-10-29T20:13:18.2130901Z   [PASS] rocm-permission-denied-kfd: 2/2 checks in 21.34s
2026-10-29T20:14:25.1501487Z **2/2 cases passed** (6/6 individual expectations) on `opus` (effort `high`).
"""

WINDOWS_LOG = """\
2026-10-29T20:13:18.3222893Z   [PASS] rocm-hip-no-binary-for-gpu: 4/4 checks in 34.9s
2026-10-29T20:13:42.8138666Z   [FAIL] rocm-permission-denied-kfd: 1/2 checks in 24.49s
2026-10-29T20:16:21.1817959Z **1/2 cases passed** (5/6 individual expectations) on `opus` (effort `high`).
"""

ROUTING = {
    "meta": {"model": "opus", "effort": "high", "skills": ["a", "b", "c", "d", "e"]},
    "totals": {"graded": 45, "passed": 41},
    "per_skill": {
        "tracelens": {"expected": 5, "correct": 5, "fired_total": 5},
    },
    "cases": [
        {"id": "node-event-loop", "category": "near_miss", "skill": "tracelens", "passed": True},
        {"id": "cprofile", "category": "near_miss", "skill": "tracelens", "passed": True},
        {"id": "vllm-on-nvidia", "category": "near_miss", "skill": "other", "passed": True},
    ],
}

GEMM_LOG = """\
2026-10-28T23:31:40.0000000Z   [PASS] gemm-01-repeatability: 7/7 checks in 606.46s
2026-10-28T23:31:40.0000000Z **1/1 cases passed** (7/7 individual expectations) on `opus` (effort `high`).
"""


class JobLabelTest(unittest.TestCase):
    def test_plain_leg(self) -> None:
        self.assertEqual(
            sce.behavioral_job_label("evals / Behavioral (rocm-doctor on Windows)", "rocm-doctor"),
            "Windows",
        )

    def test_scoped_leg_keeps_its_environment(self) -> None:
        self.assertEqual(
            sce.behavioral_job_label(
                "evals / Behavioral (hyperloom on Linux, behavioral-instinct)", "hyperloom"
            ),
            "Linux (behavioral-instinct)",
        )

    def test_other_skills_and_jobs_are_ignored(self) -> None:
        self.assertIsNone(sce.behavioral_job_label("evals / Behavioral (rocm-doctor on Linux)", "rocm"))
        self.assertIsNone(sce.behavioral_job_label("evals / Routing", "rocm-doctor"))


class ParseLogTest(unittest.TestCase):
    def test_reads_cases_and_model(self) -> None:
        leg = sce.parse_behavioral_log(WINDOWS_LOG)
        self.assertEqual((leg["model"], leg["effort"]), ("opus", "high"))
        self.assertEqual([c["id"] for c in leg["cases"]],
                         ["rocm-hip-no-binary-for-gpu", "rocm-permission-denied-kfd"])
        self.assertEqual(leg["cases"][1], {
            "id": "rocm-permission-denied-kfd", "passed": False,
            "met": 1, "total": 2, "elapsed_s": 24.49,
        })

    def test_routing_failures_are_not_cases(self) -> None:
        line = "  [FAIL] images-cost: expected local-ai-use -> got no skill (missed_trigger, tool_budget, 16.09s)"
        self.assertEqual(sce.parse_behavioral_log(line)["cases"], [])


class RenderTest(unittest.TestCase):
    URL = "https://github.com/amd/skills/actions/runs/1"

    def test_routing_and_one_os(self) -> None:
        section = sce.render(
            "tracelens", "a2575ae", self.URL, ROUTING, {"Linux": sce.parse_behavioral_log(GEMM_LOG)}
        )
        self.assertIn("| Routing recall (triggers when it should) | 5/5 (100%) |", section)
        self.assertIn("| Near-miss prompts correctly ignored | 2/2 |", section)
        self.assertIn("| Behavioral | 1/1 cases (7/7 checks), 606s |", section)
        self.assertIn(
            "Model: opus (high) · Behavioral on Linux · "
            "Routing room: 5 published skills, room accuracy 41/45 (91%)",
            section,
        )

    def test_both_oses_share_one_row(self) -> None:
        section = sce.render("rocm-doctor", "3dec0eb", self.URL, None, {
            "Linux": sce.parse_behavioral_log(LINUX_LOG),
            "Windows": sce.parse_behavioral_log(WINDOWS_LOG),
        })
        self.assertIn(
            "| Behavioral | Linux 2/2 cases (6/6 checks), 50s · "
            "Windows 1/2 cases (5/6 checks), 59s |",
            section,
        )
        self.assertEqual(section.count("| Behavioral"), 1)
        self.assertIn("Model: opus (high) · Routing: not run for this bump", section)

    def test_job_that_failed_partway(self) -> None:
        windows = sce.parse_behavioral_log(WINDOWS_LOG.splitlines()[0]) | {"conclusion": "failure"}
        section = sce.render("rocm-doctor", "3dec0eb", self.URL, None, {
            "Linux": sce.parse_behavioral_log(LINUX_LOG),
            "Windows": windows,
        })
        self.assertIn("Windows 1/2 cases (4/4 checks), 35s, job failed |", section)

    def test_structure_row_comes_first(self) -> None:
        section = sce.render(
            "tracelens", "a2575ae", self.URL, ROUTING,
            {"Linux": sce.parse_behavioral_log(GEMM_LOG)}, "success",
        )
        rows = [line for line in section.splitlines() if line.startswith("| ") and "---" not in line]
        self.assertEqual(
            rows[1], "| Structure (SKILL.md format, skill card, eval dataset, internal links) | Passed |"
        )

    def test_nothing_graded(self) -> None:
        section = sce.render("pc-sampling", "8a45f2c", self.URL, None, {}, "success")
        self.assertIn("internal links) | Passed |", section)
        self.assertIn("\nNo graded evals ran for this bump\n", section)
        self.assertNotIn("Routing", section)


class ApplyTest(unittest.TestCase):
    CARD = "# Skill Card\n\n## Description\n\nX.\n\n## License\n\nMIT\n"

    def test_appends_then_replaces_in_place(self) -> None:
        first = sce.apply(self.CARD, f"{sce.START}\nold\n{sce.END}\n")
        self.assertTrue(first.startswith(self.CARD + "\n" + sce.START))
        second = sce.apply(first, f"{sce.START}\nnew\n{sce.END}\n")
        self.assertEqual(second, self.CARD + "\n" + f"{sce.START}\nnew\n{sce.END}\n")


if __name__ == "__main__":
    unittest.main()
