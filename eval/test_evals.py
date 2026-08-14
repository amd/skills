# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the eval machinery itself. No agent, no tokens, no network.

    python eval/test_evals.py            # or: python -m unittest discover eval

Two jobs. First, guard the parts that decide whether a paid run is
trustworthy: routing verdicts, activation detection, and the rules that reject
a malformed dataset. Second, keep the JSON Schema in lockstep with the parser
-- the schema is the field reference skill owners read, and one that has
quietly drifted from what the runner enforces is worse than no schema at all.
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
from datasets import EVALUATIONS_KEY, TRIGGER_KEY  # noqa: E402

TRIGGERING = "triggeringEvaluation"
NON_TRIGGERING = "nonTriggeringEvaluation"


def parse(payload: dict, skill: str | None = "demo-skill") -> tuple[list, list[str]]:
    """Run the dataset parser over an in-memory payload."""
    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "evals.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        cases = datasets._parse_cases(payload, skill, source, errors)
    return cases, errors


def triggers(**case) -> dict:
    """A dataset holding one evaluation that should fire the skill."""
    return {EVALUATIONS_KEY: [{TRIGGER_KEY: True, **case}]}


def triggers_nothing(**case) -> dict:
    """A dataset holding one evaluation where nothing should fire."""
    return {EVALUATIONS_KEY: [{TRIGGER_KEY: False, **case}]}


class TestSchemaStaysInSyncWithParser(unittest.TestCase):
    """The schema is documentation; these tests stop it becoming fiction."""

    def setUp(self) -> None:
        self.schema = json.loads(
            (EVAL_DIR / "schema" / "evals.schema.json").read_text(encoding="utf-8")
        )

    def defs(self, name: str) -> dict:
        return self.schema["$defs"][name]

    def test_top_level_properties_match_parser(self) -> None:
        self.assertEqual(set(self.schema["properties"]), datasets.DATASET_KEYS)

    def test_triggering_properties_match_parser(self) -> None:
        self.assertEqual(
            set(self.defs(TRIGGERING)["properties"]), datasets.TRIGGER_CASE_KEYS
        )

    def test_non_triggering_properties_match_parser(self) -> None:
        self.assertEqual(
            set(self.defs(NON_TRIGGERING)["properties"]), datasets.NO_TRIGGER_CASE_KEYS
        )

    def test_the_flag_is_required_and_discriminates_the_two_shapes(self) -> None:
        for name, value in ((TRIGGERING, True), (NON_TRIGGERING, False)):
            with self.subTest(name):
                self.assertEqual(self.defs(name)["required"], ["id", "prompt", TRIGGER_KEY])
                self.assertEqual(self.defs(name)["properties"][TRIGGER_KEY]["const"], value)

    def test_unknown_keys_are_rejected_by_both(self) -> None:
        for name in (TRIGGERING, NON_TRIGGERING):
            with self.subTest(name):
                self.assertFalse(self.defs(name)["additionalProperties"])
        _, errors = parse(triggers(id="a", prompt="p", expect_skill="demo-skill"))
        self.assertTrue(any("unknown key" in e for e in errors), errors)


class TestMachineSchema(unittest.TestCase):
    """A bad machine.yml means a job that never schedules, so catch it here."""

    def setUp(self) -> None:
        self.schema = json.loads(
            (EVAL_DIR / "schema" / "machine.schema.json").read_text(encoding="utf-8")
        )

    def test_documented_keys_match_the_parser(self) -> None:
        self.assertEqual(set(self.schema["properties"]), datasets.MACHINE_KEYS)

    def test_documented_runner_types_match_the_parser(self) -> None:
        self.assertEqual(
            set(self.schema["properties"]["runner_type"]["enum"]), set(datasets.RUNNER_TYPES)
        )

    def test_every_machine_yml_in_the_repo_resolves(self) -> None:
        for skill in datasets.catalog_skills():
            with self.subTest(skill=skill):
                plan = datasets.machine_plan(skill)
                self.assertIn(plan["runner_type"], datasets.RUNNER_TYPES)
                self.assertTrue(plan["os"])

    def test_a_skill_without_the_file_gets_the_default_everywhere(self) -> None:
        plan = datasets.machine_plan("local-ai-use")
        self.assertEqual(plan["runner_type"], "default")
        self.assertEqual(plan["os"], ["Linux", "Windows"])
        self.assertEqual(plan["gate"], "")
        self.assertEqual(plan["environment"], "")

    def test_instinct_carries_its_gate_and_environment_without_saying_so(self) -> None:
        # The file says `runner_type: instinct` and nothing else; the label
        # that rations the runner and the environment holding its key are
        # properties of the hardware, not of the skill.
        raw = datasets._read_machine("serving-llms-on-instinct")
        self.assertEqual(raw, {"runner_type": "instinct"})
        plan = datasets.machine_plan("serving-llms-on-instinct")
        self.assertEqual(plan["gate"], "enable_mi_ci")
        self.assertEqual(plan["environment"], "behavioral-instinct")
        self.assertEqual(plan["os"], ["Linux"])

    def test_the_platform_label_is_not_duplicated(self) -> None:
        default = datasets.machine_plan("local-ai-use")
        self.assertEqual(
            datasets.runner_labels(default, "Windows"),
            ["self-hosted", "strix_halo", "Windows"],
        )
        instinct = datasets.machine_plan("serving-llms-on-instinct")
        self.assertEqual(
            datasets.runner_labels(instinct, "Linux"), instinct["labels"]
        )


class TestMachineRejections(unittest.TestCase):
    """Failing at planning beats scheduling a job onto a pool that has no runners."""

    def plan(self, text: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "machine.yml"
            path.write_text(text, encoding="utf-8")
            original = datasets.machine_path
            datasets.machine_path = lambda skill: path
            try:
                return datasets.machine_plan("demo-skill")
            finally:
                datasets.machine_path = original

    def test_a_retired_key_is_rejected_rather_than_ignored(self) -> None:
        # `runner`, `gate`, `environment`, and `reason` used to live here.
        # Silently dropping one would leave a skill on the wrong hardware.
        for text in ("gate: enable_mi_ci\n", "reason: because\n", "runner: [a, b]\n"):
            with self.subTest(text.strip()), self.assertRaises(SystemExit) as caught:
                self.plan(text)
            self.assertIn("unknown key", str(caught.exception))

    def test_an_unknown_runner_type(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.plan("runner_type: mi500x\n")
        self.assertIn("runner_type", str(caught.exception))

    def test_a_platform_the_runner_type_does_not_have(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.plan("runner_type: instinct\nos: [Windows]\n")
        self.assertIn("`os`", str(caught.exception))

    def test_an_empty_os_list(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.plan("os: []\n")
        self.assertIn("`os`", str(caught.exception))

    def test_the_minimum_useful_files(self) -> None:
        self.assertEqual(self.plan("os: [Linux]\n")["os"], ["Linux"])
        self.assertEqual(self.plan("runner_type: instinct\n")["gate"], "enable_mi_ci")


class TestCaseExpectations(unittest.TestCase):
    """`skill_should_trigger` is the whole expectation."""

    def test_a_triggering_evaluation_targets_the_owning_skill(self) -> None:
        cases, errors = parse(triggers(id="a", prompt="p"))
        self.assertEqual(errors, [])
        self.assertEqual(cases[0].expect_skill, "demo-skill")
        self.assertEqual(cases[0].category, "positive")

    def test_a_non_triggering_evaluation_is_a_near_miss_for_the_owning_skill(self) -> None:
        cases, errors = parse(triggers_nothing(id="a", prompt="p"))
        self.assertEqual(errors, [])
        self.assertIsNone(cases[0].expect_skill)
        self.assertEqual(cases[0].category, "near_miss")

    def test_shared_pool_cases_are_unrelated(self) -> None:
        cases, errors = parse(triggers_nothing(id="a", prompt="p"), skill=None)
        self.assertEqual(errors, [])
        self.assertIsNone(cases[0].expect_skill)
        self.assertEqual(cases[0].category, "unrelated")

    def test_both_kinds_live_in_one_array(self) -> None:
        cases, errors = parse(
            {
                EVALUATIONS_KEY: [
                    {"id": "a", TRIGGER_KEY: True, "prompt": "p"},
                    {"id": "b", TRIGGER_KEY: False, "prompt": "q"},
                ]
            }
        )
        self.assertEqual(errors, [])
        self.assertEqual([c.skill_should_trigger for c in cases], [True, False])

    def test_has_behavior_only_when_something_is_asserted(self) -> None:
        cases, _ = parse(
            {
                EVALUATIONS_KEY: [
                    {"id": "a", TRIGGER_KEY: True, "prompt": "p"},
                    {
                        "id": "b",
                        TRIGGER_KEY: True,
                        "prompt": "p",
                        "expected_behavior": ["do the thing"],
                    },
                ]
            }
        )
        self.assertFalse(cases[0].has_behavior)
        self.assertTrue(cases[1].has_behavior)


class TestDatasetRejections(unittest.TestCase):
    def test_missing_id(self) -> None:
        _, errors = parse(triggers(prompt="p"))
        self.assertTrue(any("`id`" in e for e in errors), errors)

    def test_missing_prompt(self) -> None:
        _, errors = parse(triggers(id="a"))
        self.assertTrue(any("`prompt`" in e for e in errors), errors)

    def test_the_trigger_flag_is_required(self) -> None:
        # Defaulting it would recreate the hazard the flag exists to remove:
        # an omitted field silently deciding the routing expectation.
        _, errors = parse({EVALUATIONS_KEY: [{"id": "a", "prompt": "p"}]})
        self.assertTrue(any(TRIGGER_KEY in e for e in errors), errors)

    def test_the_trigger_flag_must_be_a_boolean(self) -> None:
        for value in ("yes", "true", 1, None):
            with self.subTest(value=value):
                _, errors = parse({EVALUATIONS_KEY: [{"id": "a", "prompt": "p", TRIGGER_KEY: value}]})
                self.assertTrue(any(TRIGGER_KEY in e for e in errors), errors)

    def test_an_empty_dataset(self) -> None:
        _, errors = parse({EVALUATIONS_KEY: []})
        self.assertTrue(any("non-empty array" in e for e in errors), errors)

    def test_evaluations_must_be_an_array(self) -> None:
        _, errors = parse({EVALUATIONS_KEY: {"id": "a", "prompt": "p"}})
        self.assertTrue(any("non-empty array" in e for e in errors), errors)

    def test_a_non_triggering_evaluation_takes_a_prompt_and_nothing_else(self) -> None:
        # No skill is ever loaded for these, so there is no behavior phase for
        # an assertion to be graded in or a workspace to be staged into.
        for key, value in (
            ("expected_behavior", ["x"]),
            ("unexpected_behavior", ["x"]),
            ("logs_contain", ["x"]),
            ("files_exist", ["x"]),
            ("workspace", "evals/files/thing"),
        ):
            with self.subTest(key):
                _, errors = parse(triggers_nothing(id="a", prompt="p", **{key: value}))
                self.assertTrue(
                    any(f"`{key}`" in e and TRIGGER_KEY in e for e in errors), errors
                )

    def test_a_non_triggering_evaluation_never_reaches_behavior_mode(self) -> None:
        cases, errors = parse(triggers_nothing(id="a", prompt="p", note="why"))
        self.assertEqual(errors, [])
        self.assertFalse(cases[0].has_behavior)

    def test_the_shared_pool_cannot_expect_a_trigger(self) -> None:
        _, errors = parse(triggers(id="a", prompt="p"), skill=None)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("belongs to no skill", errors[0])

    def test_string_lists_reject_a_bare_string(self) -> None:
        _, errors = parse(triggers(id="a", prompt="p", expected_behavior="do the thing"))
        self.assertTrue(any("array of non-empty strings" in e for e in errors), errors)

    def test_duplicate_ids_are_found(self) -> None:
        cases, _ = parse(
            {
                EVALUATIONS_KEY: [
                    {"id": "a", TRIGGER_KEY: True, "prompt": "p"},
                    {"id": "a", TRIGGER_KEY: False, "prompt": "q"},
                ]
            }
        )
        self.assertEqual(datasets.duplicate_ids(cases), ["a"])


class TestTier0(unittest.TestCase):
    def test_thin_dataset_is_rejected(self) -> None:
        cases, _ = parse(triggers(id="a", prompt="p"))
        errors = datasets.tier0_errors("local-ai-use", cases)
        self.assertTrue(any(f"{TRIGGER_KEY}: true" in e for e in errors), errors)
        self.assertTrue(any(f"{TRIGGER_KEY}: false" in e for e in errors), errors)

    def test_the_minimum_dataset_passes(self) -> None:
        cases, errors = parse(
            {
                EVALUATIONS_KEY: [
                    {"id": c, TRIGGER_KEY: True, "prompt": "p"} for c in "abc"
                ]
                + [{"id": c, TRIGGER_KEY: False, "prompt": "p"} for c in "de"]
            },
            skill="local-ai-use",
        )
        self.assertEqual(errors, [])
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

    def test_the_routing_catalog_is_the_published_bundle(self) -> None:
        # Not every skill on disk: routing installs what a user installs.
        catalog = datasets.routing_catalog()
        self.assertTrue(catalog, "the marketplace bundle lists no skills")
        self.assertLessEqual(set(catalog), set(datasets.catalog_skills()))
        marketplace = json.loads(
            (datasets.REPO_ROOT / ".claude-plugin" / "marketplace.json").read_text(
                encoding="utf-8"
            )
        )
        listed = [
            entry.rstrip("/").rsplit("/", 1)[-1]
            for entry in marketplace["plugins"][0]["skills"]
        ]
        self.assertEqual(sorted(catalog), sorted(listed))

    def test_every_published_skill_still_has_routing_prompts(self) -> None:
        # A published skill with no gradeable prompt would silently drop out of
        # the score rather than failing.
        catalog = datasets.routing_catalog()
        runnable = datasets.routing_cases(datasets.load_all_cases(), catalog)
        covered = {case.expect_skill for case in runnable if case.expect_skill}
        self.assertEqual(sorted(covered), sorted(catalog))

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

    def test_template_is_a_valid_dataset(self) -> None:
        # New owners copy this file, so a template the parser rejects would
        # greet every one of them with an error they did not cause.
        template = json.loads((EVAL_DIR / "TEMPLATE.json").read_text(encoding="utf-8"))
        cases, errors = parse(template, skill="demo-skill")
        self.assertEqual(errors, [])
        self.assertEqual(datasets.tier0_errors("local-ai-use", cases), [])


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
    """Deterministic grading only; the judged fields need a live judge."""

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

    def test_files_exist_finds_the_artifact_in_a_subdirectory(self) -> None:
        # Where a plan lands is the agent's call; asking for `plan.md` and
        # getting `examples/fixture/plan.md` is a pass, not a defect.
        nested = self.workspace / "examples" / "fixture"
        nested.mkdir(parents=True)
        (nested / "plan.md").write_text("x", encoding="utf-8")
        checks = self.make_run(stream()).evaluate(files_exist=["plan.md"])
        self.assertTrue(checks[0].passed)
        self.assertIn("examples/fixture/plan.md", checks[0].detail)

    def test_files_exist_matches_whole_segments_only(self) -> None:
        (self.workspace / "analyze_plan.md").write_text("x", encoding="utf-8")
        checks = self.make_run(stream()).evaluate(files_exist=["plan.md"])
        self.assertFalse(checks[0].passed)

    def test_files_exist_keeps_the_directory_context_it_was_given(self) -> None:
        deep = self.workspace / "run-1" / "analysis_output"
        deep.mkdir(parents=True)
        (deep / "analysis.md").write_text("x", encoding="utf-8")
        (self.workspace / "analysis.md").write_text("x", encoding="utf-8")
        run = self.make_run(stream())
        self.assertTrue(run.evaluate(files_exist=["analysis_output/analysis.md"])[0].passed)
        self.assertFalse(run.evaluate(files_exist=["other_output/analysis.md"])[0].passed)

    def test_files_exist_ignores_a_directory_of_the_wanted_name(self) -> None:
        (self.workspace / "out.png").mkdir()
        checks = self.make_run(stream()).evaluate(files_exist=["out.png"])
        self.assertFalse(checks[0].passed)

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
        cases, errors = parse(triggers(**case_payload), skill=skill)
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

        cases, _ = parse(triggers(id="a", prompt="p", unexpected_behavior=["x"]))
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
                checks=[
                    {
                        "kind": "expected_behavior",
                        "expectation": "y",
                        "passed": False,
                        "detail": "no",
                    }
                ],
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
            {
                EVALUATIONS_KEY: [
                    {"id": "a", TRIGGER_KEY: True, "prompt": "p"},
                    {"id": "b", TRIGGER_KEY: True, "prompt": "q"},
                ]
            },
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


class TestRoutingCaseSelection(unittest.TestCase):
    """Only prompts the installed bundle could actually satisfy get graded."""

    def cases(self, skill: str) -> list:
        parsed, _ = parse(
            {
                EVALUATIONS_KEY: [
                    {"id": f"{skill}-yes", TRIGGER_KEY: True, "prompt": "p"},
                    {"id": f"{skill}-no", TRIGGER_KEY: False, "prompt": "q"},
                ]
            },
            skill=skill,
        )
        return parsed

    def test_a_published_skill_keeps_both_kinds(self) -> None:
        kept = datasets.routing_cases(self.cases("published"), ["published"])
        self.assertEqual([c.id for c in kept], ["published-yes", "published-no"])

    def test_an_unpublished_positive_is_held_out(self) -> None:
        # It expects a skill that is not installed, so it would lose by
        # construction and read as a routing defect.
        kept = datasets.routing_cases(self.cases("unpublished"), ["published"])
        self.assertEqual([c.id for c in kept], ["unpublished-no"])

    def test_shared_negatives_survive_any_catalog(self) -> None:
        shared = datasets.load_shared_negatives()
        self.assertTrue(shared)
        self.assertEqual(len(datasets.routing_cases(shared, ["published"])), len(shared))

    def test_an_empty_catalog_keeps_only_negatives(self) -> None:
        kept = datasets.routing_cases(self.cases("published"), [])
        self.assertEqual([c.id for c in kept], ["published-no"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
