# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the eval machinery itself. No agent, no tokens, no network.

    python eval/test_evals.py            # or: python -m unittest discover eval

Two jobs. First, guard the parts that decide whether a paid run is
trustworthy: routing verdicts, activation detection, and the rules that reject
a malformed dataset. Second, keep the JSON Schema in lockstep with the parser
-- the schema exists for editor autocomplete, and a schema that has quietly
drifted from what the runner enforces is worse than no schema at all.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

import agent  # noqa: E402
import datasets  # noqa: E402
import routing  # noqa: E402
import run_evals  # noqa: E402


def parse(payload: dict, skill: str | None = "demo-skill") -> tuple[list, list[str]]:
    """Run the dataset parser over an in-memory payload."""
    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "evals.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        cases = datasets._parse_cases(payload, skill, source, errors)
    return cases, errors


class TestSchemaStaysInSyncWithParser(unittest.TestCase):
    """The schema is documentation; these tests stop it becoming fiction."""

    def setUp(self) -> None:
        self.schema = json.loads(
            (EVAL_DIR / "schema" / "evals.schema.json").read_text(encoding="utf-8")
        )

    def test_case_properties_match_parser(self) -> None:
        documented = set(self.schema["$defs"]["case"]["properties"])
        self.assertEqual(documented, datasets.CASE_KEYS)

    def test_top_level_properties_match_parser(self) -> None:
        documented = set(self.schema["properties"])
        self.assertEqual(documented, datasets.DATASET_KEYS)

    def test_only_id_and_prompt_are_required(self) -> None:
        # The whole point of the format: the folder names the skill, so a
        # positive case needs nothing else.
        self.assertEqual(self.schema["$defs"]["case"]["required"], ["id", "prompt"])

    def test_unknown_keys_are_rejected_by_both(self) -> None:
        self.assertFalse(self.schema["$defs"]["case"]["additionalProperties"])
        _, errors = parse({"cases": [{"id": "a", "prompt": "p", "expect_skil": None}]})
        self.assertTrue(any("unknown key" in e for e in errors), errors)


class TestMachineSchema(unittest.TestCase):
    def test_every_machine_yml_uses_documented_keys(self) -> None:
        schema = json.loads(
            (EVAL_DIR / "schema" / "machine.schema.json").read_text(encoding="utf-8")
        )
        documented = set(schema["properties"])
        for skill in datasets.catalog_skills():
            if not datasets.machine_path(skill).is_file():
                continue
            with self.subTest(skill=skill):
                self.assertLessEqual(set(datasets.load_machine(skill)), documented)


class TestExpectSkillDefaulting(unittest.TestCase):
    def test_omitted_expect_skill_means_this_skill(self) -> None:
        cases, errors = parse({"cases": [{"id": "a", "prompt": "p"}]})
        self.assertEqual(errors, [])
        self.assertEqual(cases[0].expect_skill, "demo-skill")
        self.assertEqual(cases[0].category, "positive")

    def test_explicit_null_is_a_near_miss_for_the_owning_skill(self) -> None:
        cases, errors = parse({"cases": [{"id": "a", "prompt": "p", "expect_skill": None}]})
        self.assertEqual(errors, [])
        self.assertIsNone(cases[0].expect_skill)
        self.assertEqual(cases[0].category, "near_miss")

    def test_shared_pool_cases_are_unrelated(self) -> None:
        cases, errors = parse({"cases": [{"id": "a", "prompt": "p"}]}, skill=None)
        self.assertEqual(errors, [])
        self.assertEqual(cases[0].category, "unrelated")

    def test_naming_another_skill_asserts_a_handoff(self) -> None:
        cases, errors = parse(
            {"cases": [{"id": "a", "prompt": "p", "expect_skill": "other-skill"}]}
        )
        self.assertEqual(errors, [])
        self.assertEqual(cases[0].expect_skill, "other-skill")

    def test_has_behavior_only_when_something_is_asserted(self) -> None:
        cases, _ = parse(
            {
                "cases": [
                    {"id": "a", "prompt": "p"},
                    {"id": "b", "prompt": "p", "should": ["do the thing"]},
                ]
            }
        )
        self.assertFalse(cases[0].has_behavior)
        self.assertTrue(cases[1].has_behavior)


class TestDatasetRejections(unittest.TestCase):
    def test_missing_id(self) -> None:
        _, errors = parse({"cases": [{"prompt": "p"}]})
        self.assertTrue(any("`id`" in e for e in errors), errors)

    def test_missing_prompt(self) -> None:
        _, errors = parse({"cases": [{"id": "a"}]})
        self.assertTrue(any("`prompt`" in e for e in errors), errors)

    def test_empty_cases_array(self) -> None:
        _, errors = parse({"cases": []})
        self.assertTrue(any("non-empty array" in e for e in errors), errors)

    def test_should_on_a_negative_case(self) -> None:
        # There is no skill loaded to grade, so this would grade the base model.
        _, errors = parse(
            {"cases": [{"id": "a", "prompt": "p", "expect_skill": None, "should": ["x"]}]}
        )
        self.assertTrue(any("cannot use" in e and "`should`" in e for e in errors), errors)

    def test_should_not_on_a_negative_case_is_allowed(self) -> None:
        cases, errors = parse(
            {"cases": [{"id": "a", "prompt": "p", "expect_skill": None, "should_not": ["x"]}]}
        )
        self.assertEqual(errors, [])
        self.assertTrue(cases[0].has_behavior)

    def test_shared_negatives_cannot_assert_behavior(self) -> None:
        # Behavior mode stages one skill and these belong to none, so the
        # assertion would be silently skipped rather than run.
        _, errors = parse(
            {"cases": [{"id": "a", "prompt": "p", "should_not": ["x"]}]}, skill=None
        )
        self.assertTrue(any("routing-only" in e for e in errors), errors)

    def test_shared_negatives_cannot_expect_a_skill(self) -> None:
        _, errors = parse(
            {"cases": [{"id": "a", "prompt": "p", "expect_skill": "x"}]}, skill=None
        )
        self.assertTrue(any("must be null" in e for e in errors), errors)

    def test_string_lists_reject_a_bare_string(self) -> None:
        _, errors = parse({"cases": [{"id": "a", "prompt": "p", "should": "do the thing"}]})
        self.assertTrue(any("array of non-empty strings" in e for e in errors), errors)

    def test_duplicate_ids_are_found_across_files(self) -> None:
        cases, _ = parse({"cases": [{"id": "a", "prompt": "p"}, {"id": "a", "prompt": "q"}]})
        self.assertEqual(datasets.duplicate_ids(cases), ["a"])


class TestTier0(unittest.TestCase):
    def test_thin_dataset_is_rejected(self) -> None:
        cases, _ = parse({"cases": [{"id": "a", "prompt": "p"}]})
        errors = datasets.tier0_errors("local-ai-use", cases)
        self.assertTrue(any("positive case" in e for e in errors), errors)
        self.assertTrue(any("near-miss case" in e for e in errors), errors)

    def test_a_handoff_counts_as_a_negative_not_a_positive(self) -> None:
        cases, _ = parse(
            {
                "cases": [
                    {"id": "a", "prompt": "p"},
                    {"id": "b", "prompt": "p"},
                    {"id": "c", "prompt": "p"},
                    {"id": "d", "prompt": "p", "expect_skill": "somewhere-else"},
                    {"id": "e", "prompt": "p", "expect_skill": None},
                ]
            },
            skill="local-ai-use",
        )
        # Rewritten to the skill under test so tier0_errors sees its own cases.
        for case in cases[:3]:
            case.expect_skill = "local-ai-use"
        self.assertEqual(datasets.tier0_errors("local-ai-use", cases), [])

    def test_a_skill_with_no_dataset_is_reported(self) -> None:
        errors = datasets.tier0_errors("no-such-skill", [])
        self.assertEqual(len(errors), 1)
        self.assertIn("no eval dataset", errors[0])


class TestRepositoryDatasets(unittest.TestCase):
    """The real datasets, as CI sees them."""

    def test_all_datasets_are_valid(self) -> None:
        self.assertEqual(datasets.validate_all(), [])

    def test_every_catalog_skill_has_a_dataset(self) -> None:
        self.assertEqual(
            sorted(datasets.catalog_skills()), sorted(datasets.skills_with_datasets())
        )

    def test_hooks_are_importable_and_expose_known_entry_points(self) -> None:
        known = {"setup_session", "setup", "teardown", "check"}
        for skill in datasets.skills_with_datasets():
            if not datasets.hooks_path(skill).is_file():
                continue
            with self.subTest(skill=skill):
                module = run_evals._load_hooks(skill)
                exported = {
                    name
                    for name in dir(module)
                    if not name.startswith("_") and callable(getattr(module, name))
                }
                self.assertTrue(exported & known, f"{skill} hooks export nothing usable")

    def test_no_case_asserts_its_own_skill_name_in_the_logs(self) -> None:
        # That was the old stand-in for a routing assertion. Routing mode grades
        # it properly now, and a substring match only proved the skill was
        # staged -- which behavior mode guarantees by construction.
        for case in datasets.load_all_cases():
            for text in case.logs_contain:
                self.assertNotEqual(
                    text.lower(),
                    (case.skill or "").lower(),
                    f"{case.id}: routing is graded by routing mode, not logs_contain",
                )

    def test_template_matches_the_schema_shape(self) -> None:
        template = json.loads((EVAL_DIR / "TEMPLATE.json").read_text(encoding="utf-8"))
        _, errors = parse(
            {k: v for k, v in template.items() if k != "$schema"}, skill="demo-skill"
        )
        self.assertEqual(errors, [])


class TestRoutingClassification(unittest.TestCase):
    def test_verdicts(self) -> None:
        cases = [
            ("skill-a", "skill-a", "correct_trigger"),
            (None, None, "true_negative"),
            ("skill-a", None, "missed_trigger"),
            ("skill-a", "skill-b", "wrong_skill"),
            (None, "skill-a", "false_trigger"),
        ]
        for expect, observed, verdict in cases:
            with self.subTest(expect=expect, observed=observed):
                self.assertEqual(routing.classify(expect, observed), verdict)

    def test_only_correct_and_true_negative_pass(self) -> None:
        self.assertEqual(
            routing.PASSING_VERDICTS, {"correct_trigger", "true_negative"}
        )


class TestActivationDetection(unittest.TestCase):
    SKILLS = ["local-ai-use", "local-ai-app-integration", "serving-llms-on-instinct"]

    def event(self, tool: str, tool_input: dict) -> dict:
        return {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "name": tool, "input": tool_input}]},
        }

    def test_skill_tool_call_is_an_activation(self) -> None:
        event = self.event("Skill", {"command": "local-ai-use"})
        self.assertEqual(routing.detect_activation(event, self.SKILLS), "local-ai-use")

    def test_longest_name_wins_when_one_is_a_prefix_of_another(self) -> None:
        event = self.event("Skill", {"command": "local-ai-app-integration"})
        self.assertEqual(
            routing.detect_activation(event, self.SKILLS), "local-ai-app-integration"
        )

    def test_a_skill_outside_the_catalog_is_flagged_not_scored(self) -> None:
        event = self.event("Skill", {"command": "somebody-elses-skill"})
        self.assertEqual(
            routing.detect_activation(event, self.SKILLS), "other:somebody-elses-skill"
        )

    def test_listing_the_catalog_is_not_an_activation(self) -> None:
        event = self.event("Bash", {"command": "ls .claude/skills"})
        self.assertIsNone(routing.detect_activation(event, self.SKILLS))

    def test_reading_a_skill_body_counts_only_without_a_skill_tool(self) -> None:
        event = self.event("Read", {"file_path": "/tmp/x/.claude/skills/local-ai-use/SKILL.md"})
        self.assertEqual(
            routing.detect_activation(event, self.SKILLS, allow_body_path=True),
            "local-ai-use",
        )
        self.assertIsNone(
            routing.detect_activation(event, self.SKILLS, allow_body_path=False)
        )

    def test_a_tool_result_listing_every_skill_is_not_an_activation(self) -> None:
        # An empty workspace answers a file hunt with a recursive listing of
        # every SKILL.md; scoring that credited whichever name sorted first.
        event = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "content": "skills/local-ai-use/SKILL.md\nskills/serving-llms-on-instinct/SKILL.md",
                    }
                ]
            },
        }
        self.assertIsNone(routing.detect_activation(event, self.SKILLS))

    def test_catalog_inspection_is_recognized(self) -> None:
        self.assertTrue(routing._is_catalog_inspection('{"path": ".claude/skills"}', self.SKILLS))
        self.assertFalse(routing._is_catalog_inspection('{"path": "src/main.py"}', self.SKILLS))


class TestPromptTemplating(unittest.TestCase):
    def test_placeholders_are_substituted(self) -> None:
        self.assertEqual(
            run_evals._expand("trace: {trace_path}", {"trace_path": "/tmp/a.json"}),
            "trace: /tmp/a.json",
        )

    def test_literal_braces_survive(self) -> None:
        # Prompts routinely contain JSON snippets and regex quantifiers, which
        # str.format would choke on.
        text = 'produce {"a": 1} and match \\d{3}'
        self.assertEqual(run_evals._expand(text, {"x": "y"}), text)


def stream(*tool_calls: tuple[str, dict], result: str = "done") -> list[dict]:
    """Synthetic stream-json events, shaped like the CLI's output."""
    events: list[dict] = [{"type": "system", "subtype": "init", "tools": ["Bash", "Skill"]}]
    for name, tool_input in tool_calls:
        events.append(
            {
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "name": name, "input": tool_input}]},
            }
        )
    events.append({"type": "result", "result": result})
    return events


class TestRunGrading(unittest.TestCase):
    """Deterministic grading only; `should` / `should_not` need a live judge."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def make_run(self, events: list[dict]) -> agent.Run:
        return agent.Run(workspace=self.workspace, events=events, judge_model=None)

    def test_transcript_and_tools_are_captured(self) -> None:
        run = self.make_run(stream(("Bash", {"command": "python detect.py"})))
        self.assertIn("Bash", run.tool_names)
        self.assertIn("detect.py", run.logs)
        self.assertEqual(run.result_text, "done")

    def test_logs_contain_is_case_insensitive(self) -> None:
        run = self.make_run(stream(("Bash", {"command": "python DETECT.py"})))
        checks = run.evaluate(logs_contain=["detect.py"])
        self.assertTrue(checks[0].passed)

    def test_logs_contain_reports_a_miss(self) -> None:
        run = self.make_run(stream(("Bash", {"command": "ls"})))
        checks = run.evaluate(logs_contain=["detect.py"])
        self.assertFalse(checks[0].passed)

    def test_files_exist(self) -> None:
        (self.workspace / "out.png").write_bytes(b"x")
        checks = self.make_run(stream()).evaluate(files_exist=["out.png", "missing.txt"])
        self.assertTrue(checks[0].passed)
        self.assertFalse(checks[1].passed)

    def test_every_expectation_is_reported_not_just_the_first(self) -> None:
        # A run that cost minutes should not have to be repeated to discover
        # the second thing wrong with it.
        checks = self.make_run(stream()).evaluate(
            logs_contain=["nope"], files_exist=["also-nope"]
        )
        self.assertEqual(len(checks), 2)
        self.assertFalse(any(c.passed for c in checks))

    def test_dot_claude_is_excluded_from_workspace_listing(self) -> None:
        staged = self.workspace / ".claude" / "skills" / "demo"
        staged.mkdir(parents=True)
        (staged / "SKILL.md").write_text("x", encoding="utf-8")
        (self.workspace / "out.png").write_bytes(b"x")
        self.assertEqual(self.make_run(stream()).files, ["out.png"])


class FakeAgent:
    """Stands in for a real agent session so the flow can be tested offline."""

    def __init__(self, events: list[dict], seed: Path | None) -> None:
        self.events = events
        self.seed = seed
        self.workspace: Path | None = None
        self.prompts: list[str] = []
        self._tmp: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> "FakeAgent":
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        if self.seed is not None:
            for path in self.seed.iterdir():
                (self.workspace / path.name).write_bytes(path.read_bytes())
        return self

    def __exit__(self, *exc) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()

    def prompt(self, text: str):
        self.prompts.append(text)
        return agent.Run(workspace=self.workspace, events=self.events, judge_model=None)


class TestBehaviorCaseFlow(unittest.TestCase):
    """The hook contract and prompt templating, without spending tokens."""

    def run_case(self, case_payload: dict, hooks=None, events=None, skill="local-ai-use"):
        cases, errors = parse({"cases": [case_payload]}, skill=skill)
        self.assertEqual(errors, [])
        made: list[FakeAgent] = []

        def fake_claude(model, *, skill, effort, seed=None):
            made.append(FakeAgent(events or stream(), seed))
            return made[-1]

        original = run_evals.claude
        run_evals.claude = fake_claude
        try:
            outcome = run_evals.run_behavior_case(cases[0], {}, hooks, "opus", "high")
        finally:
            run_evals.claude = original
        return outcome, made[0]

    def test_a_passing_case(self) -> None:
        outcome, session = self.run_case(
            {"id": "a", "prompt": "run it", "logs_contain": ["detect.py"]},
            events=stream(("Bash", {"command": "detect.py"})),
        )
        self.assertTrue(outcome.passed)
        self.assertEqual(session.prompts, ["run it"])

    def test_a_failing_expectation_fails_the_case(self) -> None:
        outcome, _ = self.run_case({"id": "a", "prompt": "run it", "logs_contain": ["nope"]})
        self.assertFalse(outcome.passed)

    def test_hooks_run_in_order_and_can_template_the_prompt(self) -> None:
        calls: list[str] = []

        class Hooks:
            @staticmethod
            def setup(workspace, case, ctx):
                calls.append("setup")
                return {"output_dir": workspace / "out"}

            @staticmethod
            def check(run, case, ctx):
                calls.append("check")

            @staticmethod
            def teardown(workspace, case, ctx):
                calls.append("teardown")

        outcome, session = self.run_case(
            {"id": "a", "prompt": "write to {output_dir}", "logs_contain": ["detect"]},
            hooks=Hooks,
            events=stream(("Bash", {"command": "detect"})),
        )
        self.assertEqual(calls, ["setup", "check", "teardown"])
        self.assertNotIn("{output_dir}", session.prompts[0])
        self.assertTrue(outcome.passed)

    def test_a_raising_hook_check_fails_the_case_without_killing_the_run(self) -> None:
        class Hooks:
            @staticmethod
            def check(run, case, ctx):
                raise AssertionError("scorer reported 3 failures")

        outcome, _ = self.run_case({"id": "a", "prompt": "p", "logs_contain": []}, hooks=Hooks)
        self.assertFalse(outcome.passed)
        self.assertTrue(any("scorer reported" in c["detail"] for c in outcome.checks))

    def test_teardown_runs_even_when_the_agent_raises(self) -> None:
        calls: list[str] = []

        class Hooks:
            @staticmethod
            def teardown(workspace, case, ctx):
                calls.append("teardown")

        class Exploding(FakeAgent):
            def prompt(self, text):
                raise RuntimeError("claude produced no output")

        cases, _ = parse({"cases": [{"id": "a", "prompt": "p", "should_not": ["x"]}]})
        original = run_evals.claude
        run_evals.claude = lambda model, *, skill, effort, seed=None: Exploding(stream(), seed)
        try:
            outcome = run_evals.run_behavior_case(cases[0], {}, Hooks, "opus", "high")
        finally:
            run_evals.claude = original
        self.assertEqual(calls, ["teardown"])
        self.assertFalse(outcome.passed)
        self.assertIn("claude produced no output", outcome.error)

    def test_workspace_fixtures_are_staged(self) -> None:
        outcome, session = self.run_case(
            {
                "id": "a",
                "prompt": "edit it",
                "workspace": "evals/files/openai-stub",
                "files_exist": ["main.py"],
            },
            skill="local-ai-app-integration",
        )
        self.assertTrue(outcome.passed, outcome.checks)


class TestBehaviorReporting(unittest.TestCase):
    def test_summary_counts_cases_and_expectations(self) -> None:
        outcomes = [
            run_evals.BehaviorOutcome(
                id="a", skill="s", prompt="p", passed=True, elapsed_s=1.0,
                checks=[{"kind": "logs_contain", "expectation": "x", "passed": True, "detail": ""}],
            ),
            run_evals.BehaviorOutcome(
                id="b", skill="s", prompt="p", passed=False, elapsed_s=1.0,
                checks=[{"kind": "should", "expectation": "y", "passed": False, "detail": "no"}],
            ),
        ]
        summary = run_evals.summarize_behavior(outcomes, {"model": "opus", "effort": "high"})
        self.assertEqual(summary["totals"], {
            "cases": 2, "passed": 1, "checks": 2, "checks_passed": 1, "errors": 0
        })
        report = run_evals.render_behavior_markdown(summary)
        self.assertIn("1/2 cases passed", report)
        self.assertIn("`b`", report)


class TestCaseFiltering(unittest.TestCase):
    def setUp(self) -> None:
        self.cases, _ = parse(
            {"cases": [{"id": "a", "prompt": "p"}, {"id": "b", "prompt": "q"}]},
            skill="local-ai-use",
        )

    def test_filter_by_id(self) -> None:
        self.assertEqual([c.id for c in datasets.filter_cases(self.cases, "a")], ["a"])

    def test_filter_by_skill(self) -> None:
        self.assertEqual(len(datasets.filter_cases(self.cases, "local-ai-use")), 2)

    def test_empty_filter_keeps_everything(self) -> None:
        self.assertEqual(len(datasets.filter_cases(self.cases, "")), 2)

    def test_no_match_is_an_error(self) -> None:
        with self.assertRaises(SystemExit):
            datasets.filter_cases(self.cases, "nope")


if __name__ == "__main__":
    unittest.main(verbosity=2)
