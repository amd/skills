"""Offline V2 wire-contract, lifecycle, reporting and privacy regressions."""

import http.client
import io
import json
import os
import signal
import ssl
import tempfile
import unittest
import urllib.error
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import orchestrai_control_plane as cp
import orchestrai_run as controller

ENV = {
    "ORCHESTRAI_V2_CI_API_URL": "https://control.example/ci/v1",
    "ORCHESTRAI_V2_OIDC_AUDIENCE": "https://control.example",
    "ORCHESTRAI_V2_WORKSPACE": "fixture-space",
    "ACTIONS_ID_TOKEN_REQUEST_URL": "https://oidc.example/token?api-version=2",
    "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "private-request-token",
    "GITHUB_REPOSITORY": "amd/skills",
    "GITHUB_RUN_ID": "123",
    "GITHUB_RUN_ATTEMPT": "1",
    "GITHUB_JOB": "oai",
}
BUILD = "https://pipeline.example/job/folder/job/build/42/"


def plan():
    return {
        "name": "skills-evals-123-1-linux",
        "sessions": [
            {
                "name": "skills-local-ai-use-linux",
                "os_image": "ubuntu",
                "machine_tags": ["private-fleet"],
                "tests": [
                    {
                        "path": "L4-sys/skills/linux/sys_func-skills_behavioral",
                        "variables": {
                            "SKILLS": "local-ai-use",
                            "SKILLS_OS": "Linux",
                            "SKILLS_SHA": "a" * 40,
                            "SKILLSCOPE_REF": "v0.1.3",
                        },
                    }
                ],
            }
        ],
        "run_settings": {"max_duration": 180},
        "builds_json": {
            "vars": {
                "driver_sources_json": '{"ubuntu:24.04":"https://private-driver.example"}'
            },
            "install_scripts": [{"script": "InstallationScripts/gfx/linux.sh"}],
        },
    }


def state(status="running", **extra):
    return {"id": "private-run-id", "status": status, "pipeline_url": BUILD, **extra}


def snapshot(status="passed", stdout="[PASS] (files_exist) out.png\n"):
    return {
        "launcher": {
            "sessions": [
                {
                    "name": "skills-local-ai-use-linux #42",
                    "status": "completed",
                    "tests": [
                        {
                            "path": plan()["sessions"][0]["tests"][0]["path"],
                            "status": status,
                            "stdout": stdout,
                            "stderr": "",
                        }
                    ],
                }
            ]
        }
    }


def http_error(status, code="", body=None):
    content = (
        body
        or json.dumps({"error": {"code": code, "message": "private-secret"}}).encode()
    )
    return urllib.error.HTTPError(
        "https://private.example", status, "private-message", {}, io.BytesIO(content)
    )


def ci_results(**overrides):
    return {
        "api_version": "ci/v1",
        "run_id": "private-run-id",
        "total": 1,
        "tests": [
            {
                "execution_id": "test-1",
                "test_id": None,
                "name": "adapter",
                "path": plan()["sessions"][0]["tests"][0]["path"],
                "outcome": "passed",
                "exit_code": 0,
                "error": None,
                "recorded_at": "2026-10-21T12:00:00Z",
            }
        ],
        **overrides,
    }


@patch.dict(os.environ, ENV, clear=True)
class V2ClientTests(unittest.TestCase):
    def client(self):
        client = cp.ControlPlaneClient()
        client.oidc = Mock(return_value="private-oidc")
        return client

    def test_inline_plan_preserves_candidate_release_and_provisioning(self):
        source = plan()
        before = deepcopy(source)
        result = cp.inline_request(source)
        self.assertEqual(source, before)
        self.assertEqual(
            set(result), {"require_capacity", "plan_document", "builds_json"}
        )
        self.assertEqual(result["require_capacity"], "queue")
        document = result["plan_document"]["document"]
        self.assertEqual(
            document["sessions"][0]["tests"], source["sessions"][0]["tests"]
        )
        self.assertEqual(document["sessions"][0]["level"], "L4-sys")
        self.assertNotIn("executor", document["run_settings"])
        self.assertEqual(document["run_settings"]["os_image"], "ubuntu")
        self.assertEqual(result["builds_json"], source["builds_json"])

    def test_images_are_cp_aliases_or_explicit_overrides_without_discovery(self):
        source = plan()
        source["sessions"][0]["os_image"] = "windows"
        with patch.dict(
            os.environ, {"ORCHESTRAI_V2_WINDOWS_IMAGE": "windows11-fixture"}
        ):
            request = cp.inline_request(source)
        self.assertEqual(
            request["plan_document"]["document"]["sessions"][0]["os_image"],
            "windows11-fixture",
        )

    def test_oversized_plan_is_refused_before_submission(self):
        source = plan()
        source["description"] = "x" * 65536
        with self.assertRaisesRegex(cp.ControlPlaneError, "size limit"):
            cp.inline_request(source)

    def test_key_separates_attempt_and_os_but_is_stable_for_recovery(self):
        key = cp.idempotency_key(plan())
        self.assertEqual(key, cp.idempotency_key(plan()))
        with patch.dict(os.environ, {"GITHUB_RUN_ATTEMPT": "2"}):
            self.assertNotEqual(key, cp.idempotency_key(plan()))
        other = plan()
        other["name"] += "-windows"
        self.assertNotEqual(key, cp.idempotency_key(other))

    def test_start_persists_replay_guard_with_private_permissions(self):
        client = self.client()
        client.request = Mock(return_value={"run": state()})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            client.start(plan(), path)
            client.start(plan(), path)
            calls = client.request.call_args_list
            self.assertEqual(calls[0], calls[1])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertNotIn("private", path.read_text())
            changed = plan()
            changed["sessions"][0]["tests"][0]["variables"]["SKILLS_SHA"] = "b" * 40
            with self.assertRaisesRegex(cp.ControlPlaneError, "changed run inputs"):
                client.start(changed, path)
            self.assertEqual(client.request.call_count, 2)

    def test_oidc_audience_is_encoded_and_token_is_cached(self):
        client = cp.ControlPlaneClient()
        with (
            patch.object(
                cp, "read_response", return_value={"value": "fixture.jwt.token"}
            ) as read,
            patch("sys.stdout", new_callable=io.StringIO) as out,
        ):
            client.oidc()
            client.oidc()
        self.assertEqual(read.call_count, 1)
        self.assertIn(
            "audience=https%3A%2F%2Fcontrol.example", read.call_args.args[0].full_url
        )
        self.assertIn("::add-mask::fixture.jwt.token", out.getvalue())

    def test_oidc_refreshes_expiring_token(self):
        client = cp.ControlPlaneClient()
        client.token, client.expires = "old", 1
        with (
            patch.object(cp, "read_response", return_value={"value": "new-token"}),
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            self.assertEqual(client.oidc(), "new-token")

    def test_request_sends_oidc_workspace_and_idempotency_headers(self):
        client = self.client()
        with patch.object(cp, "read_response", return_value={"run": state()}) as read:
            client.request(
                "POST", "/runs", {"require_capacity": "queue"}, key="fixed-key"
            )
        req = read.call_args.args[0]
        self.assertEqual(req.get_header("Authorization"), "Bearer private-oidc")
        self.assertEqual(req.get_header("X-orchestrai-workspace"), "fixture-space")
        self.assertEqual(req.get_header("Idempotency-key"), "fixed-key")
        self.assertEqual(req.full_url, "https://control.example/ci/v1/runs")

    def test_transient_and_non_json_retries_keep_identical_submission(self):
        for error in (
            http_error(502),
            http_error(409, "idempotency_in_progress"),
            ValueError("private body"),
            ConnectionResetError("private URL"),
            http.client.IncompleteRead(b"private data"),
            ssl.SSLError("private certificate metadata"),
        ):
            with (
                self.subTest(error=type(error).__name__),
                patch.object(cp.time, "sleep"),
                patch.object(
                    cp, "read_response", side_effect=[error, {"run": state()}]
                ) as read,
            ):
                self.client().request("POST", "/runs", {"plan": "fixture"}, key="same")
                first, second = [call.args[0] for call in read.call_args_list]
                self.assertEqual(first.data, second.data)
                self.assertEqual(first.headers, second.headers)

    def test_401_refreshes_once_and_403_does_not_retry_or_echo_payload(self):
        client = self.client()
        with (
            patch.object(cp.time, "sleep"),
            patch.object(
                cp, "read_response", side_effect=[http_error(401), {"run": state()}]
            ),
        ):
            client.request("GET", "/runs/id")
        self.assertEqual(client.token, "")
        with patch.object(cp, "read_response", side_effect=http_error(403)) as read:
            with self.assertRaisesRegex(
                cp.ControlPlaneError, "workspace CI access"
            ) as error:
                client.request("GET", "/runs/id")
        self.assertEqual(read.call_count, 1)
        self.assertNotIn("private", str(error.exception))

    def test_exact_audience_preserves_trailing_slash(self):
        with patch.dict(
            os.environ, {"ORCHESTRAI_V2_OIDC_AUDIENCE": "https://control.example/"}
        ):
            self.assertEqual(
                cp.ControlPlaneClient().audience, "https://control.example/"
            )

    def test_authorization_diagnostics_preserve_only_known_enums(self):
        for code in cp.AUTH_DIAGNOSTICS:
            with self.subTest(code=code):
                body = json.dumps(
                    {
                        "error": {
                            "code": code,
                            "scope": "plans:write",
                            "message": "private-secret https://private.example",
                            "project_id": "private-project",
                            "rule": "private-rule",
                        },
                        "request_id": "private-request-id",
                    }
                ).encode()
                with patch.object(
                    cp, "read_response", side_effect=http_error(403, body=body)
                ) as read:
                    with self.assertRaises(cp.ControlPlaneError) as error:
                        self.client().request("POST", "/runs", {}, key="same")
                message = str(error.exception)
                self.assertIn(f"({code})", message)
                self.assertNotIn("private", message)
                self.assertEqual(read.call_count, 1)
                self.assertEqual(
                    controller.classify_controller_exception(RuntimeError(message)),
                    message,
                )
                if code == "scope_missing":
                    self.assertIn("Required scope: plans:write.", message)

    def test_unknown_or_malformed_denials_never_leak_remote_values(self):
        for body in (
            b"[]",
            b"null",
            b'{"error": []}',
            b"<html>private-secret</html>",
            b'{"error": {"code": ["private-secret"]}}',
            b'{"error": {"code": "private-secret"}}',
            b'{"error": {"code": "scope_missing", "scope": "private-secret"}}',
            b'{"error": {"code": "scope_missing", "scope": ["private-secret"]}}',
        ):
            with (
                self.subTest(body=body),
                patch.object(
                    cp, "read_response", side_effect=http_error(403, body=body)
                ),
            ):
                with self.assertRaises(cp.ControlPlaneError) as error:
                    self.client().request("POST", "/runs", {}, key="same")
                self.assertIn(str(error.exception), cp.SAFE_AUTH_DIAGNOSTICS)
                self.assertNotIn("private", str(error.exception))

    def test_persistent_transport_failure_is_bounded_and_safe(self):
        with (
            patch.object(
                cp, "read_response", side_effect=ConnectionResetError("private-host")
            ) as read,
            patch.object(cp.time, "sleep"),
        ):
            with self.assertRaisesRegex(
                cp.ControlPlaneError, "bounded retries"
            ) as error:
                self.client().request("GET", "/runs/id", attempts=3)
        self.assertEqual(read.call_count, 3)
        self.assertNotIn("private-host", str(error.exception))

    def test_400_and_redirects_are_not_retried(self):
        for status in (400, 302, 409):
            with (
                self.subTest(status=status),
                patch.object(
                    cp, "read_response", side_effect=http_error(status)
                ) as read,
            ):
                with self.assertRaisesRegex(cp.ControlPlaneError, f"HTTP {status}"):
                    self.client().request("POST", "/runs", {}, key="key")
                self.assertEqual(read.call_count, 1)

    def test_transport_refuses_redirects_and_oversized_response(self):
        self.assertIsNone(
            cp.NoRedirect().redirect_request(None, None, None, None, None, None)
        )
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b"x" * 11
        with patch.object(cp.urllib.request, "build_opener") as opener:
            opener.return_value.open.return_value = response
            with self.assertRaisesRegex(cp.ControlPlaneError, "transport limit"):
                cp.read_response(Mock(), limit=10)

    def test_cancel_accepts_empty_response_and_checks_terminal_after_lost_post(self):
        with patch.object(cp, "read_response", return_value=None):
            self.assertEqual(self.client().request("POST", "/runs/id/cancel", {}), {})
        client = self.client()
        client.request = Mock(
            side_effect=[
                {"run": state()},
                cp.ControlPlaneError("unavailable"),
                {"run": state("cancelled")},
            ]
        )
        with patch.object(cp.time, "sleep"):
            self.assertTrue(client.cancel_and_wait("private-run-id"))
        self.assertEqual(
            [c.args[0] for c in client.request.call_args_list], ["GET", "POST", "GET"]
        )

    def test_cleanup_is_bounded_and_unconfirmed_without_terminal_state(self):
        client = self.client()
        client.request = Mock(return_value={"run": state()})
        with (
            patch.object(cp.time, "monotonic", side_effect=[0, 0, 61]),
            patch.object(cp.time, "sleep"),
        ):
            self.assertFalse(client.cancel_and_wait("private-run-id"))

    def test_already_terminal_run_is_not_cancelled_again(self):
        client = self.client()
        client.request = Mock(return_value={"run": state("passed")})
        self.assertTrue(client.cancel_and_wait("private-run-id"))
        self.assertEqual(client.request.call_count, 1)
        self.assertEqual(client.request.call_args.args[0], "GET")

    def test_mismatched_run_identity_is_rejected(self):
        client = self.client()
        client.request = Mock(return_value={"run": state()})
        with self.assertRaisesRegex(cp.ControlPlaneError, "mismatched"):
            client.status("another-run")

    def test_results_use_only_ci_api_and_oidc(self):
        with patch.object(cp, "read_response", return_value=ci_results()) as read:
            live = self.client().snapshot("private-run-id", plan())
        request = read.call_args.args[0]
        self.assertEqual(
            request.full_url, "https://control.example/ci/v1/runs/private-run-id/tests"
        )
        self.assertEqual(request.get_header("Authorization"), "Bearer private-oidc")
        self.assertEqual(request.get_header("X-orchestrai-workspace"), "fixture-space")
        self.assertEqual(read.call_count, 1)
        result = controller.build_results_manifest(
            plan(), live, mode="live", pipeline_status="passed"
        )
        self.assertTrue(result["ready"])
        self.assertEqual(result["items"][0]["status"], "passed")
        self.assertFalse(result["items"][0]["public_streams"])
        self.assertEqual(result["items"][0]["duration"], "")

    def test_plan_with_shared_adapter_stops_before_any_request(self):
        source = plan()
        second = deepcopy(source["sessions"][0])
        second["name"] = "skills-second-linux"
        second["tests"][0]["variables"]["SKILLS"] = "second"
        source["sessions"].append(second)
        with patch.object(cp, "read_response") as read:
            with self.assertRaisesRegex(
                cp.ControlPlaneError, "No machines were requested"
            ):
                self.client().snapshot("private-run-id", source)
        read.assert_not_called()

    def test_results_reject_malformed_partial_or_wrong_run_responses(self):
        for changes in (
            {"api_version": "unknown"},
            {"run_id": "other-run"},
            {"tests": 42},
            {"tests": ["private-secret"]},
            {"total": 2},
            {"total": True},
            {"next_cursor": "private-cursor"},
        ):
            with (
                self.subTest(changes=changes),
                patch.object(cp, "read_response", return_value=ci_results(**changes)),
            ):
                with self.assertRaisesRegex(
                    cp.ControlPlaneError, "invalid or incomplete"
                ) as error:
                    self.client().snapshot("private-run-id", plan())
                self.assertNotIn("private", str(error.exception))

    def test_result_url_cannot_be_changed_by_run_id(self):
        for run_id in ("../other", "run/tests?x=1", "https://evil.example", "x%2fy"):
            with self.subTest(run_id=run_id), patch.object(cp, "read_response") as read:
                with self.assertRaises(cp.ControlPlaneError):
                    self.client().snapshot(run_id, plan())
                read.assert_not_called()

    def test_duplicate_or_absent_execution_ids_fail_closed(self):
        for rows in (
            [{**ci_results()["tests"][0], "execution_id": None}],
            ci_results()["tests"] * 2,
        ):
            with patch.object(
                cp,
                "read_response",
                return_value=ci_results(tests=rows, total=len(rows)),
            ):
                with self.assertRaisesRegex(
                    cp.ControlPlaneError, "execution identities"
                ):
                    self.client().snapshot("private-run-id", plan())

    def test_duplicate_adapter_executions_are_not_last_row_wins(self):
        rows = ci_results()["tests"] + [
            {**ci_results()["tests"][0], "execution_id": "test-2", "outcome": "failed"}
        ]
        for ordered in (rows, list(reversed(rows))):
            with patch.object(
                cp, "read_response", return_value=ci_results(tests=ordered, total=2)
            ):
                with self.assertRaisesRegex(cp.ControlPlaneError, "ambiguous results"):
                    self.client().snapshot("private-run-id", plan())

    def test_execution_identity_is_opaque_not_used_as_a_request_path(self):
        payload = ci_results()
        payload["tests"][0]["execution_id"] = "execution:attempt-2"
        with patch.object(cp, "read_response", return_value=payload) as read:
            self.client().snapshot("private-run-id", plan())
        self.assertEqual(read.call_count, 1)
        self.assertNotIn("attempt-2", read.call_args.args[0].full_url)

    def test_status_only_results_ignore_unknown_log_and_download_fields(self):
        payload = ci_results()
        payload["tests"][0].update(
            {
                "stdout": "private-secret",
                "document": {"password": "private-secret"},
                "log_uri": "https://evil.example",
                "pipeline_url": BUILD,
            }
        )
        with patch.object(cp, "read_response", return_value=payload) as read:
            live = self.client().snapshot("private-run-id", plan())
        self.assertNotIn("private", json.dumps(live))
        self.assertEqual(read.call_count, 1)

    def test_unknown_or_conflicting_outcomes_never_pass(self):
        for fields in (
            {"outcome": "unknown"},
            {"outcome": None},
            {"outcome": {}},
            {"exit_code": 1},
            {"error": "a failure was recorded"},
        ):
            payload = ci_results()
            payload["tests"][0].update(fields)
            with patch.object(cp, "read_response", return_value=payload):
                live = self.client().snapshot("private-run-id", plan())
            result = controller.build_results_manifest(
                plan(), live, mode="live", pipeline_status="passed"
            )
            self.assertEqual(result["items"][0]["status"], "error")

    def test_malformed_error_or_exit_code_cannot_pass(self):
        for fields in ({"exit_code": False}, {"exit_code": "0"}, {"error": {}}):
            payload = ci_results()
            payload["tests"][0].update(fields)
            with patch.object(cp, "read_response", return_value=payload):
                with self.assertRaisesRegex(cp.ControlPlaneError, "invalid test"):
                    self.client().snapshot("private-run-id", plan())

    def test_failure_reason_never_bypasses_public_redaction(self):
        payload = ci_results()
        payload["tests"][0].update(
            outcome="error",
            error="dependency failed on private-machine token=private-secret",
        )
        with patch.object(cp, "read_response", return_value=payload):
            live = self.client().snapshot("private-run-id", plan())
        result = controller.build_results_manifest(
            plan(), live, mode="live", pipeline_status="failed"
        )
        self.assertEqual(result["items"][0]["status"], "error")
        self.assertNotIn("private", json.dumps(result))

    def test_results_can_be_pending_but_are_never_inferred_from_green_parent(self):
        with patch.object(
            cp, "read_response", return_value=ci_results(tests=[], total=0)
        ):
            live = self.client().snapshot("private-run-id", plan())
        result = controller.build_results_manifest(
            plan(), live, mode="live", pipeline_status="passed"
        )
        self.assertFalse(result["ready"])
        self.assertNotEqual(result["items"][0]["status"], "passed")

    def test_missing_or_duplicate_evidence_cannot_pass(self):
        for sessions in ([], snapshot()["launcher"]["sessions"] * 2):
            result = controller.build_results_manifest(
                plan(),
                {"launcher": {"sessions": sessions}},
                mode="live",
                pipeline_status="passed",
            )
            self.assertFalse(result["ready"])
            self.assertNotEqual(result["items"][0]["status"], "passed")

    def test_absent_logs_do_not_claim_full_grader_coverage(self):
        live = snapshot(stdout="")
        test = live["launcher"]["sessions"][0]["tests"][0]
        test.pop("stdout")
        test.pop("stderr")
        item = controller.build_results_manifest(
            plan(), live, mode="live", pipeline_status="passed"
        )["items"][0]
        self.assertFalse(item["public_streams"])
        self.assertFalse(item["public_log"])


@patch.dict(os.environ, ENV, clear=True)
class V2LifecycleTests(unittest.TestCase):
    def execute(self, client, evidence, **env):
        client.snapshot = evidence.snapshot
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "plan.json").write_text(json.dumps(plan()))
            with (
                patch.dict(
                    os.environ,
                    {
                        "MODE": "live",
                        "PLAN_FILE": str(path / "plan.json"),
                        "RESULTS_FILE": str(path / "results.json"),
                        "LIVE_FILE": str(path / "live.json"),
                        "CONTROL_STATE_FILE": str(path / "state.json"),
                        **env,
                    },
                ),
                patch.object(controller, "ControlPlaneClient", return_value=client),
                patch.object(controller.time, "sleep"),
                patch.object(controller.signal, "signal") as signals,
                patch("sys.stdout", new_callable=io.StringIO) as output,
            ):
                code = controller.main()
            result = json.loads((path / "results.json").read_text())
            return code, result, output.getvalue(), signals

    def test_waits_for_parent_not_test_counter_and_preserves_verdicts(self):
        client = Mock()
        client.start.return_value = state(tests_status="passed")
        client.status.side_effect = [
            state(tests_status="passed"),
            state("passed", tests_status="passed"),
        ]
        evidence = Mock()
        evidence.snapshot.return_value = snapshot()
        code, result, output, _ = self.execute(client, evidence)
        self.assertEqual(code, 0)
        self.assertEqual(client.status.call_count, 2)
        client.cancel_and_wait.assert_not_called()
        self.assertEqual(result["items"][0]["status"], "passed")
        self.assertNotIn("private", output)

    def test_behavioral_failure_is_still_gated_by_individual_verdict(self):
        client, evidence = Mock(), Mock()
        client.start.return_value = state("failed")
        evidence.snapshot.return_value = snapshot(
            "failed", "[FAIL] (files_exist) out.png\n"
        )
        code, result, _, _ = self.execute(client, evidence)
        self.assertEqual(code, 0)
        self.assertEqual(result["items"][0]["status"], "failed")

    def test_green_parent_with_no_result_fails_closed(self):
        client, evidence = Mock(), Mock()
        client.start.return_value = state("passed")
        evidence.snapshot.return_value = {"launcher": {"sessions": []}}
        code, result, _, _ = self.execute(client, evidence, RESULT_TIMEOUT_SEC="0")
        self.assertEqual(code, 1)
        self.assertFalse(result["ready"])

    def test_pipeline_url_is_ignored_even_when_supplied_by_control_plane(self):
        client, evidence = Mock(), Mock()
        client.start.return_value = state(
            "error", pipeline_url="https://pipeline.example/queue/item/42"
        )
        evidence.snapshot.return_value = {"launcher": {"sessions": []}}
        code, _, _, _ = self.execute(client, evidence, RESULT_TIMEOUT_SEC="0")
        self.assertEqual(code, 1)
        evidence.snapshot.assert_called_once_with("private-run-id", plan())

    def test_multi_skill_selection_fails_before_start_and_writes_diagnostic(self):
        from orchestrai_verdict import _safe_error, result_category

        client, evidence = Mock(), Mock()
        source = plan()
        second = deepcopy(source["sessions"][0])
        second["name"] = "skills-second-linux"
        second["tests"][0]["variables"]["SKILLS"] = "second"
        source["sessions"].append(second)
        with patch.object(controller, "validate_plan", return_value=source):
            code, result, output, _ = self.execute(client, evidence)
        self.assertEqual(code, 1)
        client.start.assert_not_called()
        client.snapshot.assert_not_called()
        client.cancel_and_wait.assert_not_called()
        self.assertEqual(len(result["items"]), 2)
        self.assertEqual(result["run"]["cleanup_status"], "not_needed")
        self.assertIn("No machines were requested", output)
        for item in result["items"]:
            self.assertEqual(_safe_error(item["error"]), cp.RESULT_IDENTITY_GAP)
            self.assertEqual(result_category(item), "Reporting / result error")

    def test_interruption_cancels_and_confirms_parent(self):
        client, evidence = Mock(), Mock()
        client.start.return_value = state()
        client.status.side_effect = KeyboardInterrupt
        for confirmed in (True, False):
            client.cancel_and_wait.return_value = confirmed
            code, result, _, signals = self.execute(client, evidence)
            self.assertEqual(code, 1)
            self.assertEqual(
                result["run"]["cleanup_status"],
                "confirmed" if confirmed else "unconfirmed",
            )
            self.assertEqual(signals.call_args_list[0].args[0], signal.SIGINT)
            self.assertEqual(signals.call_count, 4)

    def test_uncertain_post_recovers_same_submission_before_cleanup(self):
        client, evidence = Mock(), Mock()
        client.start.side_effect = [
            cp.ControlPlaneError(
                "OrchestrAI V2 CI API was unavailable after bounded retries."
            ),
            state(),
        ]
        client.cancel_and_wait.return_value = True
        code, result, _, _ = self.execute(client, evidence)
        self.assertEqual(code, 1)
        self.assertEqual(client.start.call_count, 2)
        self.assertEqual(
            client.start.call_args_list[0].args, client.start.call_args_list[1].args
        )
        self.assertEqual(result["run"]["cleanup_status"], "confirmed")

    def test_results_can_arrive_after_parent_is_terminal(self):
        client, evidence = Mock(), Mock()
        client.start.return_value = state("passed")
        client.status.return_value = state("passed")
        evidence.snapshot.side_effect = [
            {"launcher": {"sessions": []}},
            cp.ControlPlaneError(
                "OrchestrAI V2 CI API was unavailable after bounded retries."
            ),
            snapshot(),
        ]
        code, result, _, _ = self.execute(client, evidence)
        self.assertEqual(code, 0)
        self.assertTrue(result["ready"])
        self.assertEqual(evidence.snapshot.call_count, 3)

    def test_sigterm_uses_cleanup_and_returns_interrupted_exit_code(self):
        client, evidence = Mock(), Mock()
        client.start.return_value = state()
        client.cancel_and_wait.return_value = True

        def interrupted_status(_run_id):
            calls = controller.signal.signal.call_args_list
            handler = next(
                call.args[1] for call in calls if call.args[0] == signal.SIGTERM
            )
            handler(signal.SIGTERM, None)

        client.status.side_effect = interrupted_status
        code, result, _, _ = self.execute(client, evidence)
        self.assertEqual(code, 130)
        self.assertEqual(result["run"]["cleanup_status"], "confirmed")

    def test_timeout_cancels_instead_of_abandoning_hardware(self):
        client, evidence = Mock(), Mock()
        client.start.return_value = state()
        client.cancel_and_wait.return_value = True
        with patch.object(controller.time, "monotonic", side_effect=[0, 2]):
            code, result, _, _ = self.execute(client, evidence, POLL_TIMEOUT_SEC="1")
        self.assertEqual(code, 1)
        self.assertEqual(result["run"]["cleanup_status"], "confirmed")


if __name__ == "__main__":
    unittest.main()
