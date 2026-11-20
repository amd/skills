"""Results execution-view contract, attribution, retry and privacy regressions."""

import json
import os
import unittest
import urllib.parse
import urllib.request
from copy import deepcopy
from unittest.mock import Mock, patch

import orchestrai_control_plane as cp
import orchestrai_run as controller
import orchestrai_verdict as verdict
import test_orchestrai_control_plane as fixtures

ENV = {**fixtures.ENV, "ORCHESTRAI_V2_EXECUTION_RESULTS": "true"}


def plan():
    source = fixtures.plan()
    second = deepcopy(source["sessions"][0])
    second["name"] = "skills-another-skill-linux"
    second["tests"][0]["variables"]["SKILLS"] = "another-skill"
    source["sessions"].append(second)
    return source


def execution(index=0, **changes):
    session = plan()["sessions"][index]
    return {
        "run_id": "private-run-id",
        "execution_id": f"private-execution-{index}",
        # The current event producer records the path as name, not test_id.
        "test_id": None,
        "name": session["tests"][0]["path"],
        "level": "L4-sys",
        "session_id": f"private-session-{index}",
        "plan_session_id": session["name"],
        "plan_session_name": session["name"],
        "attempt": 1,
        "retry_of": None,
        "superseded": False,
        "outcome": "passed",
        "exit_code": 0,
        "executor_error": None,
        # The execution view also includes these private fields. They must not
        # leak into public reporting, which uses its own explicit projection.
        "resource_id": "private-machine",
        "recorded_at": "2026-11-04T12:00:00Z",
        **changes,
    }


def page(*items, cursor=""):
    return {"items": list(items), "next_cursor": cursor}


@patch.dict(os.environ, ENV, clear=True)
class ExecutionResultsTests(unittest.TestCase):
    def client(self, *pages):
        client = cp.ControlPlaneClient()
        client.oidc = Mock(return_value="private-oidc")
        client.request = Mock(side_effect=pages)
        return client

    def snapshot(self, *rows):
        return self.client(page(*rows)).snapshot("private-run-id", plan())

    def test_explicit_session_ids_survive_inline_submission(self):
        source = plan()
        cp.require_result_identity(source)
        submitted = cp.inline_request(source)["plan_document"]["document"]
        self.assertEqual(
            [s["id"] for s in submitted["sessions"]],
            [s["name"] for s in source["sessions"]],
        )
        self.assertTrue(all("id" not in s for s in source["sessions"]))
        self.assertEqual(len(submitted["sessions"]), 2)

    def test_duplicate_or_conflicting_planned_identity_is_refused(self):
        for change in ({"id": "different"}, {"name": plan()["sessions"][1]["name"]}):
            source = plan()
            source["sessions"][0].update(change)
            with self.subTest(change=change), self.assertRaises(cp.ControlPlaneError):
                cp.inline_request(source)

    def test_rollout_switch_is_explicit_and_fails_closed(self):
        for value in ("", "false", "TRUE", "1", "unexpected"):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"ORCHESTRAI_V2_EXECUTION_RESULTS": value}),
                self.assertRaises(cp.ControlPlaneError),
            ):
                cp.require_result_identity(plan())

    def test_shared_adapter_shuffled_pages_keep_different_skill_outcomes(self):
        client = self.client(
            page(execution(1, outcome="failed", exit_code=1), cursor="opaque+/=cursor"),
            page(execution(0)),
        )
        live = client.snapshot("private-run-id", plan())
        sessions = live["launcher"]["sessions"]
        self.assertEqual(
            [s["name"] for s in sessions], [s["name"] for s in plan()["sessions"]]
        )
        self.assertEqual(
            [s["tests"][0]["status"] for s in sessions], ["passed", "failed"]
        )
        calls = client.request.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs, {"contract": cp.RESULTS_CONTRACT})
        self.assertEqual(
            urllib.parse.parse_qs(urllib.parse.urlsplit(calls[1].args[1]).query),
            {"attempts": ["all"], "limit": ["100"], "cursor": ["opaque+/=cursor"]},
        )
        self.assertNotIn("private", json.dumps(live))

    def test_execution_requests_use_ci_oidc_and_validate_contract_header(self):
        client = cp.ControlPlaneClient()
        client.oidc = Mock(return_value="private-oidc")
        with patch.object(
            cp, "read_response", return_value=page(execution(0), execution(1))
        ) as read:
            client.snapshot("private-run-id", plan())
        request = read.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer private-oidc")
        self.assertEqual(request.get_header("X-orchestrai-workspace"), "fixture-space")
        self.assertEqual(
            request.full_url,
            "https://control.example/ci/v1/runs/private-run-id/executions?attempts=all&limit=100",
        )
        self.assertEqual(read.call_args.kwargs["contract"], cp.RESULTS_CONTRACT)

    def test_linux_and_windows_shared_adapters_keep_four_independent_verdicts(self):
        source = plan()
        rows = [execution(0), execution(1, outcome="failed", exit_code=1)]
        for index, linux in enumerate(list(source["sessions"])):
            windows = deepcopy(linux)
            windows["name"] = linux["name"].replace("linux", "windows")
            windows["os_image"] = "windows"
            windows["tests"][0]["path"] = windows["tests"][0]["path"].replace(
                "linux", "windows"
            )
            windows["tests"][0]["variables"]["SKILLS_OS"] = "Windows"
            source["sessions"].append(windows)
            rows.append(
                execution(
                    index,
                    execution_id=f"windows-execution-{index}",
                    session_id=f"windows-session-{index}",
                    plan_session_id=windows["name"],
                    plan_session_name=windows["name"],
                    name=windows["tests"][0]["path"],
                )
            )
        live = self.client(page(*reversed(rows))).snapshot("private-run-id", source)
        manifest = controller.build_results_manifest(
            source, live, mode="live", pipeline_status="failed"
        )
        self.assertEqual(
            {(item["skill"], item["os"]): item["status"] for item in manifest["items"]},
            {
                ("local-ai-use", "Linux"): "passed",
                ("another-skill", "Linux"): "failed",
                ("local-ai-use", "Windows"): "passed",
                ("another-skill", "Windows"): "passed",
            },
        )

    def test_missing_or_wrong_contract_header_is_refused_before_reading_body(self):
        for header in (None, "orchestrai.results/v1", cp.RESULTS_CONTRACT):
            response = Mock()
            response.headers = {"X-OrchestrAI-Results-Contract": header}
            response.read.return_value = b'{"items": [], "next_cursor": ""}'
            opener = Mock()
            opener.open.return_value.__enter__ = Mock(return_value=response)
            opener.open.return_value.__exit__ = Mock(return_value=False)
            with (
                self.subTest(header=header),
                patch.object(cp.urllib.request, "build_opener", return_value=opener),
            ):
                request = urllib.request.Request(
                    "https://control.example/ci/v1/runs/test/executions"
                )
                if header == cp.RESULTS_CONTRACT:
                    self.assertEqual(
                        cp.read_response(request, contract=cp.RESULTS_CONTRACT), page()
                    )
                else:
                    with self.assertRaisesRegex(
                        cp.ControlPlaneError, "required execution-results contract"
                    ):
                        cp.read_response(request, contract=cp.RESULTS_CONTRACT)
                    response.read.assert_not_called()

    def test_legacy_raw_rows_never_fall_back_to_test_path(self):
        raw = {
            "run_id": "private-run-id",
            "name": execution()["name"],
            "outcome": "passed",
            "document": {"token": "private-secret"},
        }
        with self.assertRaises(cp.ControlPlaneError) as error:
            self.snapshot(raw, execution(1))
        self.assertEqual(str(error.exception), cp.EXECUTION_CONTRACT_ERROR)

    def test_not_yet_projected_run_is_pending_without_compatibility_fallback(self):
        client = cp.ControlPlaneClient()
        client.oidc = Mock(return_value="private-oidc")
        with patch.object(
            cp, "read_response", side_effect=fixtures.http_error(404)
        ) as read:
            with self.assertRaises(cp.ResultsPending):
                client.snapshot("private-run-id", plan())
        self.assertEqual(read.call_count, 1)
        self.assertIn("/executions?", read.call_args.args[0].full_url)

    def test_retry_uses_authoritative_counted_attempt_in_either_order(self):
        old = execution(superseded=True, outcome="failed", exit_code=1)
        retry = execution(
            execution_id="retry",
            session_id="retry-session",
            attempt=2,
            retry_of=old["execution_id"],
        )
        for rows in ((old, retry), (retry, old)):
            with self.subTest(rows=rows):
                self.assertEqual(
                    self.snapshot(*rows, execution(1))["launcher"]["sessions"][0][
                        "tests"
                    ][0]["status"],
                    "passed",
                )

    def test_cut_short_newer_retry_does_not_replace_counting_failure(self):
        old = execution(outcome="failed", exit_code=1)
        retry = execution(
            execution_id="retry",
            session_id="retry-session",
            attempt=2,
            retry_of=old["execution_id"],
            superseded=True,
            outcome="cancelled",
        )
        live = self.snapshot(retry, execution(1), old)
        self.assertEqual(
            live["launcher"]["sessions"][0]["tests"][0]["status"], "failed"
        )

    def test_two_counting_attempts_or_cross_skill_retry_are_rejected(self):
        old = execution()
        for changes in (
            {},
            {"retry_of": execution(1)["execution_id"], "superseded": True},
        ):
            retry = execution(
                execution_id="retry",
                session_id="retry-session",
                attempt=2,
                retry_of=old["execution_id"],
            )
            retry.update(changes)
            with (
                self.subTest(changes=changes),
                self.assertRaises(cp.ControlPlaneError) as error,
            ):
                self.snapshot(old, execution(1), retry)
            self.assertEqual(str(error.exception), cp.EXECUTION_IDENTITY_ERROR)

    def test_incomplete_publication_never_returns_a_partial_success(self):
        for rows in (
            (),
            (execution(),),
            (execution(plan_session_id=None), execution(1)),
            (execution(superseded=True), execution(1)),
            (execution(attempt=2, retry_of="missing-predecessor"), execution(1)),
        ):
            with self.subTest(rows=rows), self.assertRaises(cp.ResultsPending):
                self.snapshot(*rows)

    def test_wrong_or_malformed_identity_is_not_assigned_by_elimination(self):
        for changes in (
            {"run_id": "other-run"},
            {"execution_id": None},
            {"execution_id": execution(1)["execution_id"]},
            {"session_id": execution(1)["session_id"]},
            {"plan_session_id": "unknown"},
            {"plan_session_id": []},
            {"plan_session_name": "wrong"},
            {"name": "other-adapter"},
            {"level": "L5-cat"},
            {"attempt": True},
            {"attempt": 0},
            {"superseded": "false"},
            {"retry_of": []},
            {"retry_of": execution()["execution_id"]},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaises(cp.ControlPlaneError) as error,
            ):
                self.snapshot(execution(**changes), execution(1))
            self.assertEqual(str(error.exception), cp.EXECUTION_IDENTITY_ERROR)

    def test_unknown_or_contradictory_outcome_cannot_pass(self):
        for changes in (
            {"outcome": None},
            {"outcome": "unknown"},
            {"outcome": "running"},
            {"exit_code": 1},
            {"executor_error": "private-secret"},
        ):
            with self.subTest(changes=changes):
                live = self.snapshot(execution(**changes), execution(1))
                self.assertEqual(
                    live["launcher"]["sessions"][0]["tests"][0]["status"], "error"
                )
                self.assertNotIn("private-secret", json.dumps(live))

    def test_pages_are_bounded_and_duplicate_identities_cannot_be_hidden(self):
        cases = [
            [page(execution(), cursor="loop"), page(execution(1), cursor="loop")],
            [page(execution(), cursor="next"), page(execution())],
            [page(cursor="next")],
            [{"items": []}],
            [page(execution(), cursor={"url": "https://private.example"})],
            [page(execution(), cursor="x" * 2049)],
            [page(*([execution()] * 101))],
            [{"items": "private", "next_cursor": ""}],
        ]
        for pages in cases:
            with (
                self.subTest(pages=pages),
                self.assertRaises(cp.ControlPlaneError) as error,
            ):
                self.client(*pages).snapshot("private-run-id", plan())
            self.assertIn(str(error.exception), cp.SAFE_RESULT_DIAGNOSTICS)

    def test_page_and_total_row_limits_bound_unique_retry_rows(self):
        for page_size, page_count in ((1, 21), (100, 11)):
            pages = []
            for index in range(page_count):
                rows = [
                    execution(
                        execution_id=f"attempt-{number}",
                        attempt=number + 1,
                        superseded=True,
                    )
                    for number in range(index * page_size, (index + 1) * page_size)
                ]
                pages.append(page(*rows, cursor=f"next-{index}"))
            client = self.client(*pages)
            with (
                self.subTest(page_size=page_size),
                self.assertRaises(cp.ControlPlaneError) as error,
            ):
                client.snapshot("private-run-id", plan())
            self.assertEqual(str(error.exception), cp.EXECUTION_PAGE_ERROR)
            self.assertLessEqual(client.request.call_count, 20)

    def test_raw_private_fields_never_enter_reporting_artifacts(self):
        row = execution(
            outcome="error",
            exit_code=1,
            executor_error="private-secret https://private.example/secret",
        )
        row.update(
            document={"token": "private-token"},
            attributes={"machine": "private-host"},
            stdout="private-log",
        )
        live = self.snapshot(row, execution(1))
        result = controller.build_results_manifest(
            plan(), live, mode="live", pipeline_status="failed"
        )
        safe = controller.sanitized_live_snapshot(live)
        self.assertNotIn("private", json.dumps([result, safe]))
        self.assertEqual([i["status"] for i in result["items"]], ["error", "passed"])
        self.assertTrue(all(not i["public_streams"] for i in result["items"]))

    def test_lifecycle_accepts_shared_adapter_results_and_waits_for_publication(self):
        client = self.client()
        client.start = Mock(return_value=fixtures.state("passed"))
        client.request.side_effect = [
            page(execution()),
            page(execution(1), execution()),
        ]
        evidence = Mock(snapshot=client.snapshot)
        with patch.object(controller, "validate_plan", return_value=plan()):
            code, result, output, _ = fixtures.V2LifecycleTests().execute(
                client, evidence
            )
        self.assertEqual(code, 0)
        self.assertEqual([i["status"] for i in result["items"]], ["passed", "passed"])
        self.assertIn("still being published", output)
        self.assertNotIn("private", output)
        self.assertEqual(client.start.call_count, 1)

    def test_publication_timeout_produces_error_not_parent_pass(self):
        client = self.client(page(execution()))
        client.start = Mock(return_value=fixtures.state("passed"))
        evidence = Mock(snapshot=client.snapshot)
        with patch.object(controller, "validate_plan", return_value=plan()):
            code, result, output, _ = fixtures.V2LifecycleTests().execute(
                client, evidence, RESULT_TIMEOUT_SEC="0"
            )
        self.assertEqual(code, 1)
        self.assertEqual(len(result["items"]), 2)
        for item in result["items"]:
            self.assertEqual(item["status"], "error")
            self.assertEqual(verdict._safe_error(item["error"]), cp.EXECUTION_PENDING)
            self.assertEqual(verdict.result_category(item), "Reporting / result error")
        self.assertNotIn("private", output)


if __name__ == "__main__":
    unittest.main()
