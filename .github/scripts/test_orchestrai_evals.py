#!/usr/bin/env python3
"""Offline tests for the OrchestrAI eval plan and trigger helper."""

from __future__ import annotations

import http.client
import io
import json
import os
import ssl
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_orchestrai_plan  # noqa: E402
import orchestrai_report  # noqa: E402
import orchestrai_logs  # noqa: E402
import orchestrai_run  # noqa: E402
import orchestrai_verdict  # noqa: E402

DEVICE_TAGS_JSON = json.dumps({"strix_halo": ["test-strix-tag"]})


def isolated_subprocess_env(**overrides: str) -> dict[str, str]:
    """Do not let helper CLI tests write fixture output to the real Actions UI."""
    child = dict(os.environ)
    child.pop("GITHUB_STEP_SUMMARY", None)
    child.pop("GITHUB_OUTPUT", None)
    child.update(overrides)
    return child


def reporting_plan() -> dict:
    return {
        "name": "reporting-test",
        "sessions": [
            {
                "name": "skills-local-ai-use-linux",
                "tests": [
                    {
                        "path": "L4-sys/skills/linux/sys_func-skills_behavioral",
                        "variables": {
                            "SKILLS": "local-ai-use",
                            "SKILLS_OS": "Linux",
                        },
                    }
                ],
            },
            {
                "name": "skills-local-ai-use-windows",
                "tests": [
                    {
                        "path": "L4-sys/skills/windows/sys_func-skills_behavioral",
                        "variables": {
                            "SKILLS": "local-ai-use",
                            "SKILLS_OS": "Windows",
                        },
                    }
                ],
            },
        ],
    }


def router_grader_output() -> str:
    """Public-only fixture matching the three-case report's transport format."""
    lines = ["[behavioral] lemonade-router-builder: 3 case(s)"]
    for case_id, checks in (
        ("keyword-router", 9),
        ("pii-regex-router", 7),
        ("llm-as-router", 5),
    ):
        for index in range(checks):
            if case_id == "keyword-router" and index == 5:
                lines.append(
                    "[FAIL] (expected_behavior) Output curl commands for the user to register and test the policy -- "
                    "llm_judge: The final message and transcript show no curl commands for registering or testing the policy."
                )
            else:
                lines.append(
                    f"[PASS] (expected_behavior) Check {index + 1} -- llm_judge: The expected behavior was observed."
                )
        failed = case_id == "keyword-router"
        lines.append(
            f"[{'FAIL' if failed else 'PASS'}] {case_id}: {checks - failed}/{checks} checks in 82.45s"
        )
    lines.extend(
        [
            "## Skill behavioral",
            "**2/3 cases passed** (20/21 individual expectations) on `opus` (effort `high`).",
            "| Skill | Cases | Passed | Expectations | Met |",
            "| `lemonade-router-builder` | 3 | 2 | 21 | 20 |",
            "### Unmet expectations",
            "| keyword-router | expected_behavior | actor-supplied table | arbitrary report text |",
        ]
    )
    return "\n".join(f"[21:02:16] {line}" for line in lines)


class PlanTests(unittest.TestCase):
    def test_groups_and_deduplicates_by_os(self) -> None:
        args = build_orchestrai_plan._parser().parse_args(
            [
                "--matrix-json",
                json.dumps(
                    [
                        {"skill": "local-ai-use", "os": "Windows", "runner": "ignored"},
                        {"skill": "local-ai-use", "os": "Linux", "runner": "ignored"},
                        {
                            "skill": "tracelens-analysis-orchestrator",
                            "os": "Linux",
                            "runner": "ignored",
                        },
                        {"skill": "local-ai-use", "os": "Linux", "runner": "ignored"},
                    ]
                ),
                "--device-tags-json",
                DEVICE_TAGS_JSON,
                "--repository",
                "amd/skills",
                "--ref",
                "feature/evals",
                "--sha",
                "a" * 40,
                "--skillscope-ref",
                "v0.1.3",
                "--extended-flag=--no-extended",
                "--run-id",
                "42",
                "--output",
                "unused.json",
            ]
        )
        plan = build_orchestrai_plan.build_plan(args)
        self.assertEqual(plan["name"], "skills-evals-42-1-linux-windows")
        self.assertEqual(
            [s["name"] for s in plan["sessions"]],
            [
                "skills-local-ai-use-linux",
                "skills-tracelens-analysis-orchestrator-linux",
                "skills-local-ai-use-windows",
            ],
        )
        self.assertEqual(
            plan["sessions"][0]["tests"][0]["variables"]["SKILLS"],
            "local-ai-use",
        )
        self.assertEqual(plan["sessions"][0]["machine_tags"], ["test-strix-tag"])
        self.assertEqual(plan["sessions"][0]["os_image"], "ubuntu")
        self.assertEqual(plan["run_settings"]["acquire_timeout"], 60)
        self.assertEqual(plan["sessions"][0]["waiting_timeout"], "60m")
        self.assertEqual(
            plan["sessions"][1]["tests"][0]["variables"]["SKILLS"],
            "tracelens-analysis-orchestrator",
        )
        self.assertEqual(plan["sessions"][2]["os_image"], "windows")
        self.assertEqual(
            plan["sessions"][2]["tests"][0]["variables"]["SKILLS"],
            "local-ai-use",
        )
        self.assertEqual(plan["run_settings"]["machines_per_hw_group"], 1)
        variables = plan["sessions"][0]["tests"][0]["variables"]
        config = json.loads(build_orchestrai_plan.DEFAULT_CONFIG.read_text())
        self.assertEqual(
            variables["SKILLS_REPO"], config["skills_source"]["repository"]
        )
        self.assertEqual(variables["SKILLS_REF"], "feature/evals")
        self.assertEqual(variables["SKILLSCOPE_REF"], "v0.1.3")
        self.assertEqual(variables["SKILLS_SHA"], "a" * 40)
        self.assertEqual(variables["SOURCE_REPOSITORY"], "amd/skills")
        self.assertEqual(variables["SOURCE_REF"], "feature/evals")
        self.assertEqual(variables["SOURCE_SHA"], "a" * 40)

    def test_single_os_plan_has_a_unique_name(self) -> None:
        args = build_orchestrai_plan._parser().parse_args(
            [
                "--matrix-json",
                json.dumps([{"skill": "local-ai-use", "os": "Windows"}]),
                "--device-tags-json",
                DEVICE_TAGS_JSON,
                "--repository",
                "amd/skills",
                "--ref",
                "main",
                "--sha",
                "b" * 40,
                "--skillscope-ref",
                "v0.1.3",
                "--extended-flag=--no-extended",
                "--run-id",
                "43",
                "--run-attempt",
                "2",
                "--output",
                "unused.json",
            ]
        )
        plan = build_orchestrai_plan.build_plan(args)
        self.assertEqual(plan["name"], "skills-evals-43-2-windows")
        self.assertEqual({s["os_image"] for s in plan["sessions"]}, {"windows"})

    def test_linux_plan_injects_playbooks_style_driver_map(self) -> None:
        args = build_orchestrai_plan._parser().parse_args(
            [
                "--matrix-json",
                json.dumps([{"skill": "local-ai-use", "os": "Linux"}]),
                "--device-tags-json",
                DEVICE_TAGS_JSON,
                "--repository",
                "amd/skills",
                "--ref",
                "main",
                "--sha",
                "c" * 40,
                "--skillscope-ref",
                "v0.1.3",
                "--extended-flag=--no-extended",
                "--run-id",
                "44",
                "--output",
                "unused.json",
            ]
        )
        wrapped_map = (
            '{"ubuntu:24.04":"https://drivers.example/noble.deb",'
            '"ubuntu:26.04":"https://drivers.example/resolute\n  .deb"}'
        )
        with mock.patch.dict(
            os.environ,
            {"ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON": wrapped_map},
        ):
            plan = build_orchestrai_plan.build_plan(args)

        builds = plan["builds_json"]
        self.assertEqual(
            json.loads(builds["vars"]["driver_sources_json"]),
            {
                "ubuntu:24.04": "https://drivers.example/noble.deb",
                "ubuntu:26.04": "https://drivers.example/resolute.deb",
            },
        )
        self.assertEqual(
            builds["install_scripts"],
            [
                {
                    "script": "InstallationScripts/common/install-kernel-headers.sh",
                    "reboot_after": False,
                },
                {
                    "script": "InstallationScripts/gfx/linux.sh",
                    "reboot_after": True,
                },
            ],
        )

    def test_linux_driver_map_errors_do_not_echo_private_value(self) -> None:
        args = build_orchestrai_plan._parser().parse_args(
            [
                "--matrix-json",
                json.dumps([{"skill": "local-ai-use", "os": "Linux"}]),
                "--device-tags-json",
                DEVICE_TAGS_JSON,
                "--repository",
                "amd/skills",
                "--ref",
                "main",
                "--sha",
                "d" * 40,
                "--skillscope-ref",
                "v0.1.3",
                "--extended-flag=--no-extended",
                "--run-id",
                "45",
                "--output",
                "unused.json",
            ]
        )
        private_value = "not-json private-driver.example"
        with mock.patch.dict(
            os.environ,
            {"ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON": private_value},
        ):
            with self.assertRaises(SystemExit) as raised:
                build_orchestrai_plan.build_plan(args)
        self.assertNotIn(private_value, str(raised.exception))
        self.assertNotIn("private-driver", str(raised.exception))

    def test_rejects_an_untrusted_repository(self) -> None:
        args = build_orchestrai_plan._parser().parse_args(
            [
                "--matrix-json",
                "[]",
                "--device-tags-json",
                DEVICE_TAGS_JSON,
                "--repository",
                "attacker/repo",
                "--ref",
                "main",
                "--sha",
                "a" * 40,
                "--skillscope-ref",
                "v0.1.3",
                "--extended-flag=--no-extended",
                "--run-id",
                "42",
                "--output",
                "unused.json",
            ]
        )
        with self.assertRaises(SystemExit):
            build_orchestrai_plan.build_plan(args)

    def test_rejects_missing_fleet_configuration(self) -> None:
        args = build_orchestrai_plan._parser().parse_args(
            [
                "--matrix-json",
                json.dumps([{"skill": "local-ai-use", "os": "Linux"}]),
                "--repository",
                "amd/skills",
                "--ref",
                "main",
                "--sha",
                "a" * 40,
                "--skillscope-ref",
                "v0.1.3",
                "--extended-flag=--no-extended",
                "--run-id",
                "42",
                "--output",
                "unused.json",
            ]
        )
        with self.assertRaisesRegex(SystemExit, "ORCHESTRAI_DEVICE_TAGS"):
            build_orchestrai_plan.build_plan(args)


class TriggerTests(unittest.TestCase):
    def test_live_linux_requires_driver_map_before_allocation(self) -> None:
        with self.assertRaisesRegex(SystemExit, "ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON"):
            orchestrai_run.require_linux_provisioning(
                {"sessions": [{"os_image": "ubuntu/noble"}]}
            )

    def test_trigger_body_carries_private_builds_without_mutating_plan(self) -> None:
        builds = {
            "vars": {"driver_sources_json": '{"ubuntu:24.04":"https://x"}'},
            "install_scripts": [],
        }
        plan = {"builds_json": builds}
        request = orchestrai_run.build_run_request(plan, "plan-42")
        self.assertEqual(request, {"plan_id": "plan-42", "builds_json": builds})
        self.assertEqual(plan, {"builds_json": builds})

    def test_resolves_generic_os_to_allowlisted_images(self) -> None:
        class Client:
            @staticmethod
            def request(method: str, path: str) -> list[dict[str, str]]:
                self.assertEqual(
                    (method, path), ("GET", "/api/playground/boot-resources")
                )
                return [
                    {"os": "ubuntu/noble", "type": "ratio"},
                    {
                        "os": "windows/azure-host-2025sp1-hyperv",
                        "type": "on-demand",
                    },
                    {
                        "os": "windows/vhlk-controller-25h2",
                        "type": "on-demand",
                    },
                    {
                        "os": "windows11-25h2-26200.7840-disabled-telemetry",
                        "type": "on-demand",
                    },
                    {"os": "custom/windows11-experimental", "type": "ratio"},
                    {"os": "custom/experimental", "type": "on-demand"},
                ]

        plan = {"sessions": [{"os_image": "ubuntu"}, {"os_image": "windows"}]}
        orchestrai_run.resolve_stock_os_images(plan, Client())
        self.assertEqual(
            [session["os_image"] for session in plan["sessions"]],
            ["ubuntu/noble", "windows11-25h2-26200.7840-disabled-telemetry"],
        )

    @mock.patch.object(orchestrai_run.time, "sleep")
    def test_retries_an_empty_image_catalog(self, sleep: mock.Mock) -> None:
        class Client:
            calls = 0

            @classmethod
            def request(cls, _method: str, _path: str) -> list[dict[str, str]]:
                cls.calls += 1
                if cls.calls == 1:
                    return []
                return [{"os": "ubuntu/noble", "type": "ratio"}]

        plan = {"sessions": [{"os_image": "ubuntu"}]}
        orchestrai_run.resolve_stock_os_images(plan, Client())
        self.assertEqual(plan["sessions"][0]["os_image"], "ubuntu/noble")
        self.assertEqual(Client.calls, 2)
        sleep.assert_called_once_with(5)

    def test_does_not_select_a_specialized_windows_image(self) -> None:
        class Client:
            @staticmethod
            def request(_method: str, _path: str) -> list[dict[str, str]]:
                return [
                    {
                        "os": "windows/azure-host-2025sp1-hyperv",
                        "type": "on-demand",
                    },
                    {
                        "os": "windows/vhlk-controller-25h2",
                        "type": "on-demand",
                    },
                ]

        plan = {"sessions": [{"os_image": "windows"}]}
        with self.assertRaisesRegex(
            RuntimeError, "could not resolve generic 'windows'"
        ):
            orchestrai_run.resolve_stock_os_images(plan, Client())

    @mock.patch.object(orchestrai_run.time, "sleep")
    def test_fails_before_submission_when_image_discovery_fails(
        self, sleep: mock.Mock
    ) -> None:
        class Client:
            calls = 0

            @staticmethod
            def request(_method: str, _path: str) -> list[dict[str, str]]:
                Client.calls += 1
                raise RuntimeError("portal unavailable")

        plan = {"sessions": [{"os_image": "ubuntu"}]}
        with self.assertRaisesRegex(
            RuntimeError, "could not resolve exact stock OS images"
        ):
            orchestrai_run.resolve_stock_os_images(plan, Client())
        self.assertEqual(Client.calls, 5)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list], [5, 10, 15, 20]
        )

    def test_skips_discovery_for_an_exact_image(self) -> None:
        class Client:
            @staticmethod
            def request(_method: str, _path: str) -> list[dict[str, str]]:
                raise AssertionError("discovery should not be called")

        plan = {"sessions": [{"os_image": "ubuntu/noble"}]}
        orchestrai_run.resolve_stock_os_images(plan, Client())
        self.assertEqual(plan["sessions"][0]["os_image"], "ubuntu/noble")

    def test_builds_one_verdict_per_skill_and_os_from_live_snapshot(self) -> None:
        live = {
            "rp_url": "https://reports.example/launch/42",
            "launcher_url": "https://launcher.example",
            "launcher": {
                "sessions": [
                    {
                        "name": "skills-local-ai-use-linux #101",
                        "status": "completed",
                        "tests": [
                            {
                                "path": "L4-sys/skills/linux/sys_func-skills_behavioral",
                                "status": "passed",
                                "duration": "12s",
                                "job_id": "job-linux",
                                "stdout": "linux output\n",
                            }
                        ],
                    },
                    {
                        "name": "skills-local-ai-use-windows #101",
                        "status": "failed",
                        "error": "behavioral evaluation failed",
                        "tests": [
                            {
                                "path": "L4-sys/skills/windows/sys_func-skills_behavioral",
                                "status": "failed",
                                "duration": "14s",
                                "job_id": "job-windows",
                            }
                        ],
                    },
                ]
            },
        }
        manifest = orchestrai_run.build_results_manifest(
            reporting_plan(),
            live,
            mode="live",
            pipeline_status="unstable",
            tests_status="failed",
        )
        self.assertTrue(manifest["ready"])
        self.assertEqual(
            [item["status"] for item in manifest["items"]], ["passed", "failed"]
        )
        self.assertNotIn("job_id", manifest["items"][0])
        self.assertNotIn("stdout", manifest["items"][0])
        self.assertEqual(
            manifest["items"][1]["error"], "The behavioral test did not pass."
        )
        self.assertTrue(all("report_url" not in item for item in manifest["items"]))
        self.assertNotIn("run-42", json.dumps(manifest))
        self.assertNotIn("reportportal_url", manifest["run"])

    def test_classifies_private_test_output_with_fixed_safe_messages(self) -> None:
        cases = (
            (
                "FAIL: Skillscope setup artifacts are missing",
                "Skillscope setup artifacts were missing on the test machine.",
            ),
            (
                "FAIL: checked-out skills commit does not match SKILLS_SHA",
                "The checked-out skills commit did not match the requested commit.",
            ),
            (
                "FAIL: secure node variable LLM_GATEWAY_KEY is unavailable",
                "LLM gateway configuration was unavailable on the test machine.",
            ),
            (
                "Install steps failed for 'lib.amd.skills-runner' after 3 attempts",
                "The skills test dependencies could not be installed.",
            ),
            (
                "FAIL: unprivileged skills runner is unavailable",
                "An unprivileged Linux test account was unavailable.",
            ),
            (
                "FAIL: could not prepare the skills artifact directory",
                "The Linux skills workspace could not be prepared.",
            ),
            (
                "error: claude API not reachable -- HTTP 401 Unauthorized",
                "The Claude API preflight could not authenticate through the LLM gateway.",
            ),
            (
                "error: claude API not reachable -- API preflight timed out after 60s",
                "The Claude API preflight timed out.",
            ),
            (
                "error: claude API not reachable -- network route unavailable",
                "The Claude API preflight could not reach the LLM gateway.",
            ),
            (
                "RuntimeError: agent timed out after 7200s",
                "The behavioral test timed out.",
            ),
            (
                "RuntimeError: claude exited with code 1 and produced no parseable stream-json output",
                "The Claude agent did not return a usable result.",
            ),
            (
                "claude produced no parseable stream-json output; stderr: "
                "--dangerously-skip-permissions cannot be used with root/sudo privileges",
                "Claude Code could not run under the Linux test account.",
            ),
            (
                "claude exited with code 1 and produced no parseable stream-json output; "
                "stderr: error: unknown option --effort token=secret private-host.example",
                "Claude Code rejected the behavioral invocation.",
            ),
            (
                "claude produced no parseable stream-json output; stderr: HTTP 401 Unauthorized",
                "The Claude agent could not authenticate through the LLM gateway.",
            ),
            (
                "claude produced no parseable stream-json output; stderr: rate_limit_error HTTP 429",
                "The Claude agent was rate limited by the LLM gateway.",
            ),
            (
                "claude produced no parseable stream-json output; stderr: overloaded_error HTTP 529",
                "The Claude service was temporarily unavailable.",
            ),
            (
                "claude produced no parseable stream-json output; stderr: model_not_found",
                "The requested Claude model was unavailable through the LLM gateway.",
            ),
            (
                "claude produced no parseable stream-json output; stderr: context_length_exceeded",
                "The Claude agent exceeded the gateway context limit.",
            ),
            (
                "claude produced no parseable stream-json output; stderr: ECONNRESET",
                "The Claude agent lost connectivity to the LLM gateway.",
            ),
            (
                "claude produced no parseable stream-json output; stderr: Illegal instruction",
                "The Claude Code process crashed before returning a result.",
            ),
            (
                "[FAIL] generate-cat-image: 5/7 checks in 120.0s",
                "One or more behavioral expectations were not met.",
            ),
        )
        for output, expected in cases:
            with self.subTest(output=output):
                self.assertEqual(
                    orchestrai_run.classify_behavioral_test_output({"stdout": output}),
                    expected,
                )
                self.assertEqual(orchestrai_verdict._safe_error(expected), expected)

    def test_manifest_uses_diagnostic_without_exporting_private_test_output(
        self,
    ) -> None:
        private_output = (
            "token=super-secret private-host.example /private/workspace\\n"
            "[FAIL] generate-cat-image: 5/7 checks in 120.0s"
        )
        live = {
            "launcher": {
                "sessions": [
                    {
                        "name": "skills-local-ai-use-linux #private-job-id",
                        "status": "failed",
                        "error": "behavioral evaluation failed",
                        "tests": [
                            {
                                "path": "L4-sys/skills/linux/sys_func-skills_behavioral",
                                "status": "failed",
                                "stdout": private_output,
                                "stderr": "password=hunter2",
                                "job_id": "private-job-id",
                            }
                        ],
                    }
                ]
            }
        }
        manifest = orchestrai_run.build_results_manifest(
            {"sessions": reporting_plan()["sessions"][:1]},
            live,
            mode="live",
            pipeline_status="unstable",
            tests_status="failed",
        )
        self.assertEqual(
            manifest["items"][0]["error"],
            "One or more behavioral expectations were not met.",
        )
        encoded = json.dumps(manifest)
        for private_value in (
            "super-secret",
            "private-host",
            "/private/workspace",
            "hunter2",
            "private-job-id",
        ):
            self.assertNotIn(private_value, encoded)

    def test_unknown_error_keeps_fixed_diagnostic_and_full_context(self) -> None:
        live = {
            "launcher": {
                "sessions": [
                    {
                        "name": "skills-local-ai-use-linux",
                        "status": "failed",
                        "tests": [
                            {
                                "path": "L4-sys/skills/linux/sys_func-skills_behavioral",
                                "status": "failed",
                                "stdout": "unknown failure at a private endpoint",
                            }
                        ],
                    }
                ]
            }
        }
        manifest = orchestrai_run.build_results_manifest(
            {"sessions": reporting_plan()["sessions"][:1]}, live, mode="live"
        )
        self.assertEqual(
            manifest["items"][0]["error"], "The behavioral test did not pass."
        )
        self.assertIn(
            "private endpoint", manifest["items"][0]["public_streams"]["stdout"]
        )

    def test_portal_http_error_does_not_export_response_body(self) -> None:
        error = urllib.error.HTTPError(
            "https://portal.example/api/runs",
            400,
            "bad request",
            {},
            io.BytesIO(b"machine=private-tag token=do-not-export"),
        )
        client = orchestrai_run.PortalClient("https://portal.example")
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(RuntimeError) as raised:
                client.request("POST", "/api/runs", {})
        rendered = str(raised.exception)
        self.assertIn("HTTP 400", rendered)
        self.assertNotIn("private-tag", rendered)
        self.assertNotIn("do-not-export", rendered)

    def test_portal_timeout_is_retryable_and_does_not_export_request(self) -> None:
        client = orchestrai_run.PortalClient("https://private-portal.example")
        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError):
            with self.assertRaises(RuntimeError) as raised:
                client.request("GET", "/api/runs/private-run-id")
        rendered = str(raised.exception)
        self.assertEqual(rendered, "OrchestrAI Portal request timed out")
        self.assertNotIn("private-portal", rendered)
        self.assertNotIn("private-run-id", rendered)

    def test_portal_invalid_json_is_wrapped_for_retry(self) -> None:
        client = orchestrai_run.PortalClient("https://private-portal.example")
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b"not-json private-payload"
        with mock.patch("urllib.request.urlopen", return_value=response):
            with self.assertRaises(RuntimeError) as raised:
                client.request("GET", "/api/runs/private-run-id")
        rendered = str(raised.exception)
        self.assertEqual(
            rendered, "The OrchestrAI Portal returned an invalid response."
        )
        self.assertNotIn("private-payload", rendered)

    def test_portal_read_errors_are_wrapped_for_retry(self) -> None:
        client = orchestrai_run.PortalClient("https://private-portal.example")
        for error in (
            http.client.IncompleteRead(b"private-partial", 100),
            ConnectionResetError("private-reset"),
            ssl.SSLError("private-tls"),
        ):
            with self.subTest(error=type(error).__name__):
                response = mock.MagicMock()
                response.__enter__.return_value = response
                response.read.side_effect = error
                with mock.patch("urllib.request.urlopen", return_value=response):
                    with self.assertRaises(RuntimeError) as raised:
                        client.request("GET", "/api/runs/private-run-id")
                rendered = str(raised.exception)
                self.assertEqual(
                    rendered, "OrchestrAI Portal response could not be read"
                )
                self.assertNotIn("private", rendered)

    def test_cancellation_uses_a_short_request_timeout(self) -> None:
        client = orchestrai_run.PortalClient("https://portal.example")
        with (
            mock.patch.object(client, "request") as request,
            mock.patch.object(orchestrai_run, "log"),
        ):
            client.cancel("run-id")
        request.assert_called_once_with(
            "POST", "/api/runs/run-id/cancel", {}, timeout_seconds=5
        )

    def test_portal_reads_bounded_plain_text_without_json_decoding(self) -> None:
        client = orchestrai_run.PortalClient("https://portal.example")
        client.token = "private-test-token"
        client.space_id = "private-space"
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b"[FAIL] (expected_behavior) missing output\n\xff"
        with mock.patch("urllib.request.urlopen", return_value=response) as open_url:
            text = client.request(
                "GET", "/api/runs/run-id/logs/job-id?stream=stdout", text_response=True
            )
        self.assertIn("[FAIL] (expected_behavior)", text)
        response.read.assert_called_once_with(orchestrai_run.MAX_TEST_LOG_BYTES + 1)
        headers = dict(open_url.call_args.args[0].header_items())
        self.assertEqual(headers["Accept"], "text/plain")
        self.assertEqual(headers["Authorization"], "Bearer private-test-token")
        self.assertEqual(headers["X-space-id"], "private-space")

    def test_oversized_test_log_is_rejected_without_exposing_content(self) -> None:
        client = orchestrai_run.PortalClient("https://portal.example")
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b"private" * (
            orchestrai_run.MAX_TEST_LOG_BYTES // 7 + 1
        )
        with mock.patch("urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(
                RuntimeError, "test log exceeded its size limit"
            ) as raised:
                client.request(
                    "GET",
                    "/api/runs/run-id/logs/job-id?stream=stdout",
                    text_response=True,
                )
        self.assertNotIn("private", str(raised.exception))

    def test_cleanup_confirms_terminal_state_after_a_timed_out_post(self) -> None:
        client = orchestrai_run.PortalClient("https://portal.example")
        calls = []
        states = iter(
            [{"status": "running"}, {"status": "cancelling"}, {"status": "cancelled"}]
        )

        def request(method, path, *args, **kwargs):
            calls.append((method, path))
            if method == "POST":
                raise RuntimeError("timeout with private server payload")
            return next(states)

        with (
            mock.patch.object(client, "request", side_effect=request),
            mock.patch.object(orchestrai_run.time, "sleep"),
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            self.assertTrue(client.cancel_and_wait("run-id"))
        self.assertEqual([method for method, _ in calls], ["GET", "POST", "GET", "GET"])
        self.assertIn("cleanup confirmed", output.getvalue())
        self.assertNotIn("private", output.getvalue())

    def test_cleanup_retries_and_fails_closed_if_still_cancelling(self) -> None:
        client = orchestrai_run.PortalClient("https://portal.example")
        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        with (
            mock.patch.object(
                client, "request", return_value={"status": "cancelling"}
            ) as request,
            mock.patch.object(
                orchestrai_run.time, "monotonic", side_effect=lambda: clock[0]
            ),
            mock.patch.object(orchestrai_run.time, "sleep", side_effect=sleep),
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            self.assertFalse(client.cancel_and_wait("run-id", timeout_seconds=20))
        self.assertEqual(clock[0], 20)
        self.assertEqual(
            sum(call.args[0] == "POST" for call in request.call_args_list), 2
        )
        self.assertIn("cleanup could not be confirmed", output.getvalue())
        self.assertNotIn("cleanup confirmed", output.getvalue())

    def test_cleanup_accepts_a_terminal_race_without_recancelling(self) -> None:
        client = orchestrai_run.PortalClient("https://portal.example")
        with (
            mock.patch.object(
                client,
                "request",
                side_effect=[
                    {"status": "running"},
                    RuntimeError("HTTP 409"),
                    {"pipeline_status": "passed"},
                ],
            ) as request,
            mock.patch.object(orchestrai_run.time, "sleep"),
            mock.patch.object(orchestrai_run, "log"),
        ):
            self.assertTrue(client.cancel_and_wait("run-id"))
        self.assertEqual(
            [call.args[0] for call in request.call_args_list], ["GET", "POST", "GET"]
        )

    def test_cleanup_of_an_already_terminal_run_does_not_send_a_post(self) -> None:
        client = orchestrai_run.PortalClient("https://portal.example")
        with (
            mock.patch.object(
                client, "request", return_value={"status": "failed"}
            ) as request,
            mock.patch.object(orchestrai_run, "log"),
        ):
            self.assertTrue(client.cancel_and_wait("run-id"))
        self.assertEqual([call.args[0] for call in request.call_args_list], ["GET"])

    def test_log_collection_retries_missing_streams_and_keeps_raw_data_private(
        self,
    ) -> None:
        plan = {"sessions": reporting_plan()["sessions"][:1]}
        live = {
            "launcher": {
                "sessions": [
                    {
                        "name": "skills-local-ai-use-linux",
                        "status": "failed",
                        "tests": [
                            {
                                "path": "L4-sys/skills/linux/sys_func-skills_behavioral",
                                "status": "failed",
                                "job_id": "private-job-id",
                            }
                        ],
                    }
                ]
            }
        }
        client = mock.Mock()
        client.request.side_effect = [
            RuntimeError("HTTP 404 private payload"),
            "",
            "dependency token=secret\n[FAIL] (expected_behavior) cat image was not produced\n[FAIL] generate-cat-image: 5/7 checks in 20s\n",
        ]
        with (
            mock.patch.object(orchestrai_run.time, "sleep") as sleep,
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            enriched = orchestrai_run.collect_test_logs(
                client, plan, live, run_id="run-id"
            )
        sleep.assert_called_once_with(5)
        self.assertEqual(
            [call.args[1] for call in client.request.call_args_list],
            [
                "/api/runs/run-id/logs/private-job-id?stream=stdout",
                "/api/runs/run-id/logs/private-job-id?stream=stderr",
                "/api/runs/run-id/logs/private-job-id?stream=stdout",
            ],
        )
        self.assertNotIn("stdout", live["launcher"]["sessions"][0]["tests"][0])
        manifest = orchestrai_run.build_results_manifest(
            plan, enriched, mode="live", pipeline_status="failed"
        )
        self.assertEqual(
            manifest["items"][0]["error"],
            "One or more behavioral expectations were not met.",
        )
        self.assertIn(
            "cat image was not produced", manifest["items"][0]["public_log"][0]
        )
        for private in (
            "private-job-id",
            "token=secret",
        ):
            self.assertNotIn(private, json.dumps(manifest))
            self.assertNotIn(
                private, json.dumps(orchestrai_run.sanitized_live_snapshot(enriched))
            )
            self.assertNotIn(private, output.getvalue())

    def test_log_collection_is_bounded_and_does_not_change_a_verdict(self) -> None:
        plan = {"sessions": reporting_plan()["sessions"][:1]}
        live = {
            "launcher": {
                "sessions": [
                    {
                        "name": "skills-local-ai-use-linux",
                        "status": "completed",
                        "tests": [
                            {
                                "path": "L4-sys/skills/linux/sys_func-skills_behavioral",
                                "status": "passed",
                                "job_id": "job-id",
                            }
                        ],
                    }
                ]
            }
        }
        client = mock.Mock()
        client.request.side_effect = RuntimeError("HTTP 404 private payload")
        clock = [0.0]
        with (
            mock.patch.object(
                orchestrai_run.time, "monotonic", side_effect=lambda: clock[0]
            ),
            mock.patch.object(
                orchestrai_run.time,
                "sleep",
                side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            ),
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            enriched = orchestrai_run.collect_test_logs(
                client, plan, live, run_id="run-id", timeout_seconds=10
            )
        self.assertEqual(clock[0], 10)
        self.assertEqual(client.request.call_count, 4)
        self.assertIn("some test streams were unavailable", output.getvalue())
        self.assertNotIn("private", output.getvalue())
        manifest = orchestrai_run.build_results_manifest(plan, enriched, mode="live")
        self.assertEqual(manifest["items"][0]["status"], "passed")

    def test_log_collection_ignores_foreign_tests_and_unsafe_job_ids(self) -> None:
        plan = {"sessions": reporting_plan()["sessions"][:1]}
        for job_id, status in (
            ("../../../secrets", "passed"),
            ("https://private-host", "passed"),
            ("job-id", "running"),
        ):
            with self.subTest(job_id=job_id, status=status):
                live = {
                    "launcher": {
                        "sessions": [
                            {
                                "name": "skills-local-ai-use-linux",
                                "status": "completed",
                                "tests": [
                                    {
                                        "path": "L4-sys/skills/linux/sys_func-skills_behavioral",
                                        "status": status,
                                        "job_id": job_id,
                                    }
                                ],
                            },
                            {
                                "name": "unrequested-session",
                                "tests": [{"job_id": "other-id", "status": "passed"}],
                            },
                        ]
                    }
                }
                client = mock.Mock()
                orchestrai_run.collect_test_logs(client, plan, live, run_id="run-id")
                client.request.assert_not_called()

    def test_incomplete_tests_explain_machine_acquisition_failure_without_claiming_a_behavioral_failure(
        self,
    ) -> None:
        live = {
            "jenkins": {
                "safe_failure_summary": [
                    "Machine acquisition timed out after 180s.",
                    "The OrchestrAI pipeline reported an infrastructure failure.",
                ]
            },
            "launcher": {
                "sessions": [
                    {
                        "name": "skills-local-ai-use-linux",
                        "status": "unknown",
                        "tests": [
                            {
                                "path": "L4-sys/skills/linux/sys_func-skills_behavioral",
                                "status": "unknown",
                            }
                        ],
                    }
                ]
            },
        }
        manifest = orchestrai_run.build_results_manifest(
            {"sessions": reporting_plan()["sessions"][:1]},
            live,
            mode="live",
            pipeline_status="failed",
        )
        item = manifest["items"][0]
        self.assertEqual(item["status"], "error")
        self.assertFalse(item["terminal"])
        self.assertIn("Machine acquisition timed out after 180s.", item["error"])
        self.assertEqual(
            orchestrai_verdict._safe_error(item["error"]),
            "Machine acquisition timed out.",
        )

    def test_controller_exception_classifier_drops_unknown_payloads(self) -> None:
        rendered = orchestrai_run.classify_controller_exception(
            RuntimeError("server said password=hunter2 at https://internal-host")
        )
        self.assertNotIn("hunter2", rendered)
        self.assertNotIn("internal-host", rendered)
        self.assertIn("inspect the Portal logs", rendered)

    def test_poll_outage_budget_uses_elapsed_time(self) -> None:
        self.assertFalse(
            orchestrai_run.poll_outage_exhausted(100.0, now=399.9, timeout_seconds=300)
        )
        self.assertTrue(
            orchestrai_run.poll_outage_exhausted(100.0, now=400.0, timeout_seconds=300)
        )

    def test_missing_live_session_fails_closed(self) -> None:
        manifest = orchestrai_run.build_results_manifest(
            reporting_plan(),
            {"launcher": {"sessions": []}},
            mode="live",
        )
        self.assertFalse(manifest["ready"])
        self.assertEqual(
            [item["status"] for item in manifest["items"]],
            ["missing", "missing"],
        )

    def test_terminal_pipeline_failure_explains_missing_sessions(self) -> None:
        manifest = orchestrai_run.build_results_manifest(
            reporting_plan(),
            {"launcher": {"reachable": False, "sessions": []}},
            mode="live",
            pipeline_status="failed",
        )
        explanation = manifest["run"]["failure_summary"][0]
        self.assertIn("failed before any requested test session", explanation)
        self.assertIn("launcher was unreachable", explanation)
        self.assertEqual(manifest["items"][0]["status"], "error")
        self.assertIn("infrastructure:", manifest["items"][0]["error"])

    def test_retains_only_bounded_redacted_jenkins_failure_lines(self) -> None:
        live = {
            "jenkins": {
                "log_text": "\n".join(
                    [
                        "normal setup output",
                        "[2026-09-24T06:39:08.715Z] [provision] ERROR: "
                        "Timed out waiting for machines after 2400s",
                        "[report] request token=super-secret failed with ERROR",
                        "[launcher] Authorization: Bearer should-not-leak ERROR",
                    ]
                )
            },
            "launcher": {"sessions": []},
        }
        failure_lines = orchestrai_run.jenkins_failure_summary(live, limit=2)
        self.assertEqual(len(failure_lines), 2)
        self.assertNotIn("super-secret", "\n".join(failure_lines))
        self.assertNotIn("should-not-leak", "\n".join(failure_lines))
        self.assertNotIn("normal setup", "\n".join(failure_lines))
        self.assertEqual(
            failure_lines,
            ["Report generation failed.", "The launcher failed."],
        )

        sanitized = orchestrai_run.sanitized_live_snapshot(live)
        self.assertNotIn("log_text", sanitized["jenkins"])
        self.assertEqual(len(sanitized["jenkins"]["failure_summary"]), 3)

        manifest = orchestrai_run.build_results_manifest(
            reporting_plan(), live, mode="live", pipeline_status="failed"
        )
        self.assertIn("timed out", "\n".join(manifest["run"]["failure_summary"]))
        self.assertIn("infrastructure:", manifest["items"][0]["error"])

    def test_live_poll_keeps_earlier_failure_context_without_raw_log(self) -> None:
        earlier = {
            "jenkins": {"log_text": "[provision] ERROR: Timed out waiting for machines"}
        }
        later = {"jenkins": {"stages": []}, "launcher": {"sessions": []}}
        merged = orchestrai_run.merge_live_snapshot(earlier, later)
        self.assertNotIn("log_text", merged["jenkins"])
        self.assertEqual(
            merged["jenkins"]["safe_failure_summary"],
            ["Machine acquisition timed out."],
        )

    def test_live_poll_retains_report_link_when_later_snapshot_omits_it(self) -> None:
        earlier = {
            "rp_url": "https://reports.example/launch/42",
            "secret": "private-value",
        }
        later = {"rp_url": "", "launcher": {"sessions": []}}
        merged = orchestrai_run.merge_live_snapshot(earlier, later)
        self.assertEqual(merged["rp_url"], earlier["rp_url"])
        self.assertNotIn("secret", merged)
        self.assertEqual(later["rp_url"], "")
        merged = orchestrai_run.merge_live_snapshot(
            earlier, {"rp_url": "http://unsafe"}
        )
        self.assertEqual(merged["rp_url"], earlier["rp_url"])
        self.assertEqual(
            orchestrai_run.snapshot_report_url(
                {
                    "rp_url": "http://unsafe",
                    "rp_launch_url": earlier["rp_url"],
                }
            ),
            earlier["rp_url"],
        )

    def test_final_metadata_does_not_wait_for_reportportal(self) -> None:
        client = mock.Mock()
        client.request.side_effect = [
            {"pipeline_status": "cancelled", "password": "private-secret"},
            {"rp_url": "", "launcher": {"sessions": []}},
            {"pipeline_status": "cancelled"},
            {
                "rp_launch_url": "https://reports.example/launch/42",
                "raw": "private-log",
            },
        ]
        with mock.patch.object(orchestrai_run.time, "sleep") as sleep:
            metadata = orchestrai_run.collect_report_metadata(
                client,
                run_id="private-id",
                initial_live={},
            )
        self.assertEqual(
            metadata,
            {
                "final_pipeline_status": "cancelled",
            },
        )
        self.assertEqual(client.request.call_count, 1)
        self.assertEqual(sleep.call_count, 0)
        self.assertNotIn("/live", client.request.call_args.args[1])
        self.assertNotIn("private", json.dumps(metadata))

    def test_report_metadata_timeout_is_bounded_and_advisory(self) -> None:
        client = mock.Mock()
        client.request.side_effect = RuntimeError("password=private-secret")
        with (
            mock.patch.object(
                orchestrai_run.time, "monotonic", side_effect=iter(range(30))
            ),
            mock.patch.object(orchestrai_run.time, "sleep"),
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            metadata = orchestrai_run.collect_report_metadata(
                client,
                run_id="private-id",
                initial_live={},
                timeout_seconds=4,
            )
        self.assertEqual(metadata, {"final_pipeline_status": "unknown"})
        self.assertLessEqual(client.request.call_count, 2)
        self.assertIn("final parent status was unavailable", output.getvalue())
        self.assertNotIn("private", output.getvalue())

    def test_metadata_refresh_never_uses_cached_live_state_as_final_parent(
        self,
    ) -> None:
        client = mock.Mock()
        client.request.side_effect = [
            {"pipeline_status": "cancelled"},
            {
                "pipeline_status": "running",
                "rp_url": "https://reports.example/launch/42",
            },
        ]
        metadata = orchestrai_run.collect_report_metadata(
            client, run_id="id", initial_live={}
        )
        self.assertEqual(metadata["final_pipeline_status"], "cancelled")

    def test_metadata_refresh_rejects_unsafe_report_links(self) -> None:
        client = mock.Mock()
        client.request.side_effect = [
            {"pipeline_status": "cancelled", "rp_url": "http://unsafe"},
            {"rp_url": "https://user:password@example.com/report"},
        ]
        with (
            mock.patch.object(
                orchestrai_run.time, "monotonic", side_effect=[0, 0, 0, 0, 1, 2]
            ),
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ):
            metadata = orchestrai_run.collect_report_metadata(
                client,
                run_id="id",
                initial_live={},
                timeout_seconds=1,
            )
        self.assertNotIn("report_url", metadata)

    def test_summary_labels_pre_cleanup_and_final_parent_states(self) -> None:
        results = orchestrai_run.mock_results_manifest(reporting_plan())
        results["run"].update(
            cleanup_status="confirmed", final_pipeline_status="cancelled"
        )
        for item in results["items"]:
            item["status"] = "passed"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "summary.md"
            with mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(path)}):
                orchestrai_run.summary(
                    mode="live",
                    plan_name="test-plan",
                    pipeline_status="running",
                    tests_status="running",
                    results=results,
                )
            text = path.read_text()
        self.assertIn("Pipeline snapshot before cleanup: `running`", text)
        self.assertIn("Final parent state: `cancelled`", text)
        self.assertIn("| Skill | OS | Result |\n|---|---|---|", text)
        self.assertNotIn("Test streams", text)
        self.assertNotIn("- Pipeline: `running`", text)

    def test_summary_table_omits_stream_coverage_for_live_and_mock(self) -> None:
        for mode in ("live", "mock"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                results = orchestrai_run.mock_results_manifest(reporting_plan())
                for item in results["items"]:
                    item["public_streams"] = {"stdout": "output", "stderr": ""}
                path = Path(temp) / "summary.md"
                with mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(path)}):
                    orchestrai_run.summary(
                        mode=mode, plan_name="test-plan", results=results
                    )
                text = path.read_text()
                self.assertIn("| Skill | OS | Result |\n|---|---|---|", text)
                self.assertNotIn("Test streams", text)
                self.assertNotIn("stdout + stderr", text)
                rows = [line for line in text.splitlines() if line.startswith("| `")]
                self.assertEqual(len(rows), len(results["items"]))
                self.assertTrue(all(row.count("|") == 4 for row in rows))

    def test_sanitized_snapshot_omits_plan_urls_and_test_output(self) -> None:
        clean = orchestrai_run.sanitized_live_snapshot(
            {
                "run_id": "private-run-id",
                "plan_sessions": [{"variables": {"SECRET": "do-not-keep"}}],
                "launcher_url": "http://internal-launcher",
                "rp_url": "https://internal-report",
                "jenkins_build": 4242,
                "jenkins": {
                    "log_text": "ordinary output",
                    "stages": [
                        {
                            "id": "private-stage-id",
                            "name": "05-L4-sys",
                            "status": "failed",
                            "duration_ms": 1000,
                            "url": "https://internal-stage",
                        }
                    ],
                },
                "launcher": {
                    "sessions": [
                        {
                            "name": "skills-local-ai-use-linux #private-job-id",
                            "status": "failed",
                            "error": "connection to private-host failed token=secret",
                            "tests": [
                                {
                                    "path": "L4-sys/skills/linux/sys_func-skills_behavioral",
                                    "status": "failed",
                                    "job_id": "private-job-id",
                                    "error": "password=hunter2",
                                    "stdout": "do-not-upload",
                                    "stderr": "do-not-upload",
                                }
                            ],
                        }
                    ]
                },
            }
        )
        encoded = json.dumps(clean)
        self.assertNotIn("do-not-keep", encoded)
        self.assertNotIn("do-not-upload", encoded)
        self.assertNotIn("internal-launcher", encoded)
        self.assertNotIn("internal-report", encoded)
        self.assertNotIn("private-run-id", encoded)
        self.assertNotIn("private-stage-id", encoded)
        self.assertNotIn("private-job-id", encoded)
        self.assertNotIn("internal-stage", encoded)
        self.assertNotIn("private-host", encoded)
        self.assertNotIn("hunter2", encoded)
        self.assertIn("behavioral test", encoded.lower())

    def test_failure_artifacts_replace_partial_output_and_sanitize_live_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            results_path = Path(temp) / "results.json"
            live_path = Path(temp) / "live.json"
            results_path.write_text('{"partial": true}\n', encoding="utf-8")
            results = orchestrai_run.write_failure_artifacts(
                reporting_plan(),
                mode="live",
                error="OrchestrAI Portal request timed out",
                results_path=results_path,
                live_path=live_path,
                live={
                    "run_id": "private-run-id",
                    "launcher_url": "http://private-launcher",
                    "jenkins": {"log_text": "token=secret"},
                },
            )

            published_results = json.loads(results_path.read_text(encoding="utf-8"))
            published_live = json.loads(live_path.read_text(encoding="utf-8"))
            encoded = json.dumps([published_results, published_live])
            self.assertEqual(published_results, results)
            self.assertFalse(published_results["ready"])
            self.assertEqual(
                published_results["run"]["failure_summary"],
                ["OrchestrAI Portal request timed out"],
            )
            self.assertNotIn("partial", published_results)
            self.assertNotIn("private-run-id", encoded)
            self.assertNotIn("private-launcher", encoded)
            self.assertNotIn("secret", encoded)

    def test_public_outputs_include_only_safe_reportportal_links(self) -> None:
        live = {
            "rp_url": "https://reports.example/launch/42",
            "launcher_url": "https://launcher.example",
            "launcher": {"sessions": []},
        }
        manifest = orchestrai_run.build_results_manifest(
            reporting_plan(), live, mode="live"
        )
        sanitized = orchestrai_run.sanitized_live_snapshot(live)
        self.assertNotIn("report_url", manifest["items"][0])
        self.assertNotIn("reports.example", json.dumps(manifest))
        self.assertNotIn("reports.example", json.dumps(sanitized))
        self.assertNotIn("launcher.example", json.dumps([manifest, sanitized]))

        for unsafe in (
            "http://reports.example/launch/42",
            "https://user:secret@reports.example/launch/42",
            "javascript:alert(1)",
            "https://reports.example/<script>",
        ):
            with self.subTest(unsafe=unsafe):
                rejected = orchestrai_run.build_results_manifest(
                    reporting_plan(),
                    {**live, "rp_url": unsafe},
                    mode="live",
                )
                self.assertNotIn("report_url", rejected["items"][0])

    def test_summary_prints_infrastructure_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            summary_path = Path(temp) / "summary.md"
            previous = os.environ.get("GITHUB_STEP_SUMMARY")
            os.environ["GITHUB_STEP_SUMMARY"] = str(summary_path)
            try:
                orchestrai_run.summary(
                    mode="live",
                    plan_name="skills-evals-1-1-linux",
                    results={
                        "run": {
                            "failure_summary": [
                                "[provision] ERROR: Timed out waiting for machines"
                            ]
                        },
                        "items": [],
                    },
                )
            finally:
                if previous is None:
                    os.environ.pop("GITHUB_STEP_SUMMARY", None)
                else:
                    os.environ["GITHUB_STEP_SUMMARY"] = previous
            rendered = summary_path.read_text(encoding="utf-8")
            self.assertIn("Infrastructure failure", rendered)
            self.assertIn("Machine acquisition timed out", rendered)
            self.assertNotIn("waiting for machines", rendered)

    def test_rejects_grouped_skills_in_reporting_plan(self) -> None:
        plan = reporting_plan()
        plan["sessions"][0]["tests"][0]["variables"]["SKILLS"] = "one,two"
        with self.assertRaisesRegex(ValueError, "invalid reporting metadata"):
            orchestrai_run.expected_items(plan)

    def test_separates_test_failures_from_infrastructure_failures(self) -> None:
        passed = [{"status": "passed", "terminal": True}] * 2
        one_failed = [passed[0], {"status": "failed", "terminal": True}]
        self.assertTrue(orchestrai_run.pipeline_reporting_completed("passed", passed))
        self.assertTrue(
            orchestrai_run.pipeline_reporting_completed("unstable", one_failed)
        )
        self.assertTrue(
            orchestrai_run.pipeline_reporting_completed("failed", one_failed)
        )
        self.assertFalse(
            orchestrai_run.pipeline_reporting_completed("unstable", passed)
        )
        self.assertFalse(
            orchestrai_run.pipeline_reporting_completed("cancelled", one_failed)
        )
        self.assertTrue(
            orchestrai_run.pipeline_reporting_completed("running", one_failed)
        )
        self.assertFalse(
            orchestrai_run.pipeline_reporting_completed(
                "running", [{"status": "passed", "terminal": False}]
            )
        )

    def test_test_completion_stops_parent_pipeline_polling(self) -> None:
        self.assertTrue(
            orchestrai_run.run_outcome_is_terminal("passed", {"ready": False})
        )
        self.assertTrue(
            orchestrai_run.run_outcome_is_terminal("running", {"ready": True})
        )
        self.assertFalse(
            orchestrai_run.run_outcome_is_terminal("running", {"ready": False})
        )

    def test_controller_waits_for_every_session_then_fetches_logs_after_cleanup(
        self,
    ) -> None:
        self._exercise_completed_controller(cleanup_confirmed=True)

    def test_controller_fails_closed_when_cleanup_is_unconfirmed(self) -> None:
        self._exercise_completed_controller(cleanup_confirmed=False)

    def test_controller_recovers_late_link_without_changing_completed_verdicts(
        self,
    ) -> None:
        self._exercise_completed_controller(cleanup_confirmed=True, late_report=True)

    def _exercise_completed_controller(
        self, *, cleanup_confirmed: bool, late_report: bool = False
    ) -> None:
        plan = reporting_plan()
        snapshots = []
        for windows_session, windows_test in (
            ("queued", "queued"),
            ("running", "passed"),
            ("failed", "failed"),
        ):
            snapshots.append(
                {
                    "rp_url": ""
                    if late_report
                    else "https://reports.example/launch/42",
                    "launcher": {
                        "sessions": [
                            {
                                "name": expected["session"],
                                "status": session_status,
                                "tests": [
                                    {
                                        "path": expected["path"],
                                        "status": test_status,
                                        "job_id": f"private-job-{expected['os']}",
                                    }
                                ],
                            }
                            for expected, session_status, test_status in (
                                (
                                    orchestrai_run.expected_items(plan)[0],
                                    "completed",
                                    "passed",
                                ),
                                (
                                    orchestrai_run.expected_items(plan)[1],
                                    windows_session,
                                    windows_test,
                                ),
                            )
                        ]
                    },
                }
            )
        client = mock.Mock()
        events = []
        live_reads = 0

        def request(method, path, *_args, **kwargs):
            nonlocal live_reads
            if path == "/api/auth/login":
                return {"token": "private-test-token"}
            if path == "/api/auth/me":
                return {"spaces": [{"name": "test-space", "id": "space-id"}]}
            if path == "/api/plans":
                return {"id": "plan-id"}
            if path == "/api/runs":
                return {"id": "run-id"}
            if path == "/api/runs/run-id/live":
                if live_reads == len(snapshots):
                    events.append("metadata-live")
                    return {
                        "rp_url": "https://reports.example/launch/42",
                        "launcher": {
                            "sessions": [{"status": "cancelled", "raw": "private-log"}]
                        },
                    }
                snapshot = snapshots[live_reads]
                live_reads += 1
                events.append(f"live-{live_reads}")
                return snapshot
            if path == "/api/runs/run-id":
                if "cancel" in events:
                    events.append("metadata-state")
                    return {"pipeline_status": "cancelled", "tests_status": "cancelled"}
                return {"pipeline_status": "running", "tests_status": "passed"}
            if "/logs/" in path:
                self.assertEqual(events[-1], "cancel")
                self.assertTrue(kwargs["text_response"])
                return (
                    "dependency token=private-secret\n[PASS] (files_exist) out.png\n"
                    if path.endswith("stdout")
                    else ""
                )
            raise AssertionError((method, path))

        client.request.side_effect = request

        def cancel_and_wait(_run_id):
            events.append("cancel")
            return cleanup_confirmed

        client.cancel_and_wait.side_effect = cancel_and_wait
        with tempfile.TemporaryDirectory() as temp:
            plan_path = Path(temp) / "plan.json"
            results_path = Path(temp) / "results.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "PLAN_FILE": str(plan_path),
                        "MODE": "live",
                        "RESULTS_FILE": str(results_path),
                        "LIVE_FILE": str(Path(temp) / "live.json"),
                        "ORCHESTRAI_PORTAL_URL": "https://portal.example",
                        "ORCHESTRAI_USER": "test-user",
                        "ORCHESTRAI_PASSWORD": "test-password",
                        "ORCHESTRAI_SPACE": "test-space",
                        "GITHUB_STEP_SUMMARY": "",
                        "GITHUB_OUTPUT": "",
                    },
                ),
                mock.patch.object(orchestrai_run, "PortalClient", return_value=client),
                mock.patch.object(orchestrai_run, "resolve_stock_os_images"),
                mock.patch.object(orchestrai_run.signal, "signal"),
                mock.patch.object(orchestrai_run.time, "sleep") as sleep,
                mock.patch("sys.stdout", new_callable=io.StringIO),
            ):
                self.assertEqual(orchestrai_run.main(), 0 if cleanup_confirmed else 1)
            manifest = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            events,
            ["live-1", "live-2", "live-3", "cancel", "metadata-state"],
        )
        self.assertEqual(sleep.call_count, 2)
        self.assertTrue(manifest["ready"])
        self.assertEqual(
            manifest["run"]["cleanup_status"],
            "confirmed" if cleanup_confirmed else "unconfirmed",
        )
        self.assertNotIn("private", json.dumps(manifest))
        self.assertEqual(manifest["run"]["pipeline_status"], "running")
        self.assertEqual(manifest["run"]["final_pipeline_status"], "cancelled")
        self.assertTrue(all("report_url" not in item for item in manifest["items"]))
        self.assertEqual(
            [item["status"] for item in manifest["items"]], ["passed", "failed"]
        )
        self.assertEqual(
            manifest["items"][0]["public_log"], ["[PASS] (files_exist) out.png"]
        )

    def test_mock_validates_without_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            plan = Path(temp) / "plan.json"
            summary = Path(temp) / "summary.md"
            plan.write_text(
                json.dumps(reporting_plan()),
                encoding="utf-8",
            )
            results = Path(temp) / "results.json"
            live = Path(temp) / "live.json"
            completed = subprocess.run(
                [sys.executable, str(SCRIPTS / "orchestrai_run.py")],
                text=True,
                capture_output=True,
                env=isolated_subprocess_env(
                    MODE="mock",
                    PLAN_FILE=str(plan),
                    RESULTS_FILE=str(results),
                    LIVE_FILE=str(live),
                    GITHUB_STEP_SUMMARY=str(summary),
                ),
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("would login", completed.stdout)
            self.assertIn(
                "OrchestrAI behavioral evals", summary.read_text(encoding="utf-8")
            )
            manifest = json.loads(results.read_text(encoding="utf-8"))
            self.assertEqual(
                [item["status"] for item in manifest["items"]], ["mock", "mock"]
            )
            self.assertTrue(manifest["ready"])


class PublicLogTests(unittest.TestCase):
    def test_expectation_kind_counts_are_case_associated_and_deduplicated(self) -> None:
        raw = router_grader_output()
        item = {
            "skill": "lemonade-router-builder",
            "public_log": orchestrai_logs.public_test_log(
                {"stdout": raw, "stderr": raw}, skill="lemonade-router-builder"
            ),
        }
        summary = orchestrai_logs.public_behavioral_summary(item)
        self.assertEqual(
            summary["expectation_kinds"],
            {"expected_behavior": {"graded": 21, "met": 20}},
        )
        rendered = orchestrai_logs.behavioral_summary_markdown(item)
        self.assertIn("By expectation type", rendered)
        self.assertIn("| `expected_behavior` | 21 | 20 | 1 | 95% |", rendered)
        self.assertIn("| 67% | 95% |", rendered)

    def test_redacted_checks_are_excluded_from_type_rates(self) -> None:
        raw = "\n".join(
            [
                "[behavioral] local-ai-use: 1 case(s)",
                "[PASS] (files_exist) out.png",
                "[FAIL] (expected_behavior) password=private-secret",
                "[FAIL] generate-image: 1/2 checks in 3s",
                "0/1 cases passed (1/2 individual expectations) on opus (effort high).",
            ]
        )
        item = {
            "skill": "local-ai-use",
            "public_log": orchestrai_logs.public_test_log(
                {"stdout": raw}, skill="local-ai-use"
            ),
        }
        summary = orchestrai_logs.public_behavioral_summary(item)
        self.assertEqual(
            summary["expectation_kinds"], {"files_exist": {"graded": 1, "met": 1}}
        )
        text = orchestrai_logs.expectation_breakdown_markdown(
            [summary], heading="### Types"
        )
        self.assertIn("Redacted, missing, or unassigned checks are excluded", text)
        self.assertNotIn("private-secret", text)
        self.assertNotIn("| `expected_behavior` |", text)

    def test_ambiguous_case_block_does_not_manufacture_type_rates(self) -> None:
        item = {
            "skill": "local-ai-use",
            "public_log": [
                "[behavioral] local-ai-use: 2 case(s)",
                "[PASS] (files_exist) image.png",
                "[PASS] (expected_behavior) Install server",
                "[FAIL] (logs_contain) missing output",
                "[FAIL] second-case: 0/1 checks in 3s",
            ],
        }
        summary = orchestrai_logs.public_behavioral_summary(item)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["expectation_kinds"], {})
        self.assertIn(
            "No publishable expectation-level results",
            orchestrai_logs.expectation_breakdown_markdown(
                [summary], heading="### Types"
            ),
        )

    def test_type_table_rejects_injected_names_counts_and_zero_denominators(
        self,
    ) -> None:
        text = orchestrai_logs.expectation_breakdown_markdown(
            [
                {
                    "expectation_kinds": {
                        "private-secret": {"graded": 1, "met": 1},
                        "expected_behavior": {"graded": "password=secret", "met": 1},
                    }
                },
                {"expectation_kinds": "private-secret"},
                None,
                {
                    "expectation_kinds": {
                        "files_exist": {"graded": 0, "met": 0},
                        "logs_contain": {"graded": 1, "met": 2},
                    }
                },
            ],
            heading="### Types",
        )
        self.assertNotIn("private-secret", text)
        self.assertNotIn("password", text)
        self.assertIn("| `files_exist` | 0 | 0 | 0 | Not reported |", text)
        for met, graded in ((0, 0), (2, 1), (-1, 1)):
            self.assertEqual(orchestrai_logs.observed_rate(met, graded), "Not reported")
        self.assertEqual(orchestrai_logs.observed_rate(2, 3), "67%")

    def test_case_timing_is_deduplicated_and_failed_cases_are_first(self) -> None:
        raw = router_grader_output()
        item = {
            "skill": "lemonade-router-builder",
            "public_log": orchestrai_logs.public_test_log(
                {"stdout": raw, "stderr": raw}, skill="lemonade-router-builder"
            ),
        }
        summary = orchestrai_logs.public_behavioral_summary(item)
        self.assertEqual(summary["elapsed_s"], 247.35)
        self.assertEqual(len(summary["case_results"]), 3)
        breakdown = orchestrai_logs.behavioral_summary_markdown(item).split(
            "##### Case breakdown", 1
        )[1]
        self.assertIn("| `keyword-router` | ❌ failed | 8/9 | 82.45s |", breakdown)
        self.assertLess(
            breakdown.index("keyword-router"), breakdown.index("pii-regex-router")
        )
        self.assertIn("excludes machine acquisition", breakdown)

    def test_numeric_footer_roundtrips_markdown_plaintext_and_timestamps(self) -> None:
        canonical = (
            "2/3 cases passed (20/21 individual expectations) on opus (effort high)."
        )
        for raw in (
            canonical,
            "**2/3 cases passed** (20/21 individual expectations) on `opus` (effort `high`).",
            "[21:02:16] [21:02:16] **2/3 cases passed** (20/21 individual expectations) on `opus` (effort `high`).",
        ):
            with self.subTest(raw=raw):
                lines = orchestrai_logs.public_test_log(
                    {"stdout": raw}, skill="lemonade-router-builder"
                )
                self.assertEqual(lines, [canonical])
                self.assertEqual(
                    lines,
                    orchestrai_logs.public_manifest_log(
                        {"skill": "lemonade-router-builder", "public_log": lines}
                    ),
                )

    def test_rejects_arbitrary_metadata_and_actor_report_tables(self) -> None:
        raw = "\n".join(
            [
                "2/3 cases passed (20/21 individual expectations) on private-model (effort high).",
                "2/3 cases passed (20/21 individual expectations) on https://private.internal (effort high).",
                "2/3 cases passed (20/21 individual expectations) on opus (effort secret).",
                "2/3 cases passed (20/21 individual expectations) on opus (effort high). token=private-secret",
                "| keyword-router | error | raw exception | private-secret |",
                "[behavioral] skill='lemonade-router-builder' model='opus': private-agent-prompt",
            ]
        )
        self.assertEqual(
            orchestrai_logs.public_test_log(
                {"stdout": raw}, skill="lemonade-router-builder"
            ),
            [],
        )

    def test_reconstructs_totals_and_unmet_expectations_for_each_case(self) -> None:
        lines = orchestrai_logs.public_test_log(
            {"stdout": router_grader_output()}, skill="lemonade-router-builder"
        )
        item = {"skill": "lemonade-router-builder", "public_log": lines}
        summary = orchestrai_logs.public_behavioral_summary(item)
        self.assertEqual(
            (
                summary["passed"],
                summary["cases"],
                summary["met"],
                summary["expectations"],
            ),
            (2, 3, 20, 21),
        )
        self.assertTrue(summary["complete"])
        self.assertEqual((summary["model"], summary["effort"]), ("opus", "high"))
        self.assertEqual(len(summary["unmet"]), 1)
        self.assertEqual(summary["unmet"][0]["case"], "keyword-router")
        self.assertIn("no curl commands", summary["unmet"][0]["detail"])
        rendered = orchestrai_logs.behavioral_summary_markdown(item)
        self.assertIn(
            "**2/3 cases passed** (20/21 individual expectations) on `opus` (effort `high`).",
            rendered,
        )
        self.assertIn("| `lemonade-router-builder` | 3 | 2 | 21 | 20 |", rendered)
        self.assertIn(
            "| `keyword-router` | expected_behavior | Output curl commands", rendered
        )
        self.assertNotIn("actor-supplied table", rendered)
        self.assertNotIn("Partial", rendered)

    def test_duplicate_test_streams_do_not_double_count_cases(self) -> None:
        raw = router_grader_output()
        lines = orchestrai_logs.public_test_log(
            {"stdout": raw, "stderr": raw}, skill="lemonade-router-builder"
        )
        summary = orchestrai_logs.public_behavioral_summary(
            {"skill": "lemonade-router-builder", "public_log": lines}
        )
        self.assertEqual(summary["cases"], 3)
        self.assertEqual(summary["expectations"], 21)
        self.assertTrue(summary["complete"])
        self.assertEqual(len(summary["unmet"]), 1)

    def test_partial_or_conflicting_output_never_claims_every_case_passed(self) -> None:
        for lines in (
            [
                "[behavioral] local-ai-use: 2 case(s)",
                "[PASS] generate-image: 1/1 checks in 2s",
            ],
            [
                "[behavioral] local-ai-use: 1 case(s)",
                "[PASS] generate-image: 1/1 checks in 2s",
                "0/2 cases passed (0/2 individual expectations) on opus (effort high).",
            ],
            [
                "[behavioral] local-ai-use: 1 case(s)",
                "[FAIL] (expected_behavior) Write image",
                "[PASS] generate-image: 1/1 checks in 2s",
            ],
            [
                "[behavioral] local-ai-use: 1 case(s)",
                "[PASS] generate-image: 1/1 checks in 2s",
                "[FAIL] (expected_behavior) Unassigned check",
            ],
            [
                "[behavioral] local-ai-use: 1 case(s)",
                "[PASS] generate-image: 2/1 checks in 2s",
            ],
        ):
            with self.subTest(lines=lines):
                item = {"skill": "local-ai-use", "public_log": lines}
                summary = orchestrai_logs.public_behavioral_summary(item)
                self.assertFalse(summary["complete"])
                self.assertNotIn("model", summary)
                rendered = orchestrai_logs.behavioral_summary_markdown(item)
                self.assertIn("Partial grader output", rendered)
                self.assertNotIn("Every behavioral case", rendered)

    def test_runtime_errors_and_redacted_expectations_keep_private_details_out(
        self,
    ) -> None:
        raw = (
            "[behavioral] local-ai-use: 2 case(s)\n"
            "[FAIL] (expected_behavior) token=private-secret\n"
            "[FAIL] generate-image: 0/1 checks in 3s\n"
            "[FAIL] restart-server: 0/0 checks in 4s -- RuntimeError: private-host\n"
            "0/2 cases passed (0/1 individual expectations) on opus (effort high)."
        )
        item = {
            "skill": "local-ai-use",
            "public_log": orchestrai_logs.public_test_log(
                {"stdout": raw}, skill="local-ai-use"
            ),
        }
        summary = orchestrai_logs.public_behavioral_summary(item)
        self.assertEqual(summary["passed"], 0)
        self.assertTrue(summary["complete"])
        rendered = orchestrai_logs.behavioral_summary_markdown(item)
        self.assertIn("| `generate-image` | withheld |", rendered)
        self.assertIn("| `restart-server` | error |", rendered)
        self.assertNotIn("private-secret", rendered)
        self.assertNotIn("private-host", rendered)

    def test_summary_escapes_judge_markdown_and_revalidates_manifest(self) -> None:
        item = {
            "skill": "local-ai-use",
            "behavioral": {
                "model": "private-model",
                "unmet": [{"detail": "private-secret"}],
            },
            "public_log": [
                "[behavioral] local-ai-use: 1 case(s)",
                "[FAIL] (expected_behavior) Use | local API -- llm_judge: <img src='bad'> [click](relative-target) ```",
                "::warning::forged",
                "[FAIL] generate-image: 0/1 checks in 3s",
                "0/1 cases passed (0/1 individual expectations) on opus (effort high).",
            ],
        }
        rendered = orchestrai_logs.behavioral_summary_markdown(item)
        for forbidden in (
            "private-secret",
            "private-model",
            "<img",
            "[click]",
            "```",
            "::warning::",
            "Use | local API",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertIn("Use &#124; local API", rendered)
        self.assertIn("&lt;img", rendered)

    def test_success_and_missing_logs_have_truthful_summary_states(self) -> None:
        item = {
            "skill": "local-ai-use",
            "public_log": [
                "[behavioral] local-ai-use: 1 case(s)",
                "[PASS] generate-image: 1/1 checks in 3s",
                "1/1 cases passed (1/1 individual expectations) on opus (effort high).",
            ],
        }
        self.assertIn(
            "None. Every behavioral case met every expectation.",
            orchestrai_logs.behavioral_summary_markdown(item),
        )
        self.assertEqual(
            orchestrai_logs.public_behavioral_summary(
                {"skill": "local-ai-use", "stdout": "raw private output"}
            ),
            {},
        )
        self.assertEqual(
            orchestrai_logs.behavioral_summary_markdown(
                {
                    "skill": "local-ai-use",
                    "public_log": [
                        "1/1 cases passed (1/1 individual expectations) on opus (effort high)."
                    ],
                }
            ),
            "",
        )

    def test_preserves_timestamp_wrapped_grader_lines_without_actor_output(
        self,
    ) -> None:
        output = (
            "[21:02:16] actor=private-host token=private-secret\n"
            "[21:02:16] [behavioral] lemonade-router-builder: 1 case(s)\n"
            "[21:02:16] [21:02:16]   [PASS] (files_exist) router.yaml\n"
            "[2026-09-30T21:02:17.123Z] [FAIL] (expected_behavior) Use keywords_any -- llm_judge: Regex rules were used instead.\n"
            "[2026-09-30 21:02:18 UTC] [FAIL] build-router: 1/2 checks in 64.25s\n"
            "[private-host] [PASS] (files_exist) private-file\n"
        )
        lines = orchestrai_logs.public_test_log(
            {"stdout": output}, skill="lemonade-router-builder"
        )
        self.assertEqual(
            lines,
            [
                "[behavioral] lemonade-router-builder: 1 case(s)",
                "[PASS] (files_exist) router.yaml",
                "[FAIL] (expected_behavior) Use keywords_any -- llm_judge: Regex rules were used instead.",
                "[FAIL] build-router: 1/2 checks in 64.25s",
            ],
        )
        self.assertEqual(
            lines,
            orchestrai_logs.public_manifest_log(
                {"skill": "lemonade-router-builder", "public_log": lines}
            ),
        )

    def test_timestamp_wrappers_do_not_bypass_redaction_or_content_allowlist(
        self,
    ) -> None:
        output = (
            "[21:02:16] [FAIL] (expected_behavior) token=private-secret\n"
            "[21:02:16] [PASS] (files_exist) /home/private-user/file.txt\n"
            "[21:02:16] [FAIL] build-router: 0/1 checks in 20s -- see https://private.internal/report\n"
            "[21:02:16] [FAIL] build-router: 0/1 checks in 20s -- password=private-secret\n"
            "[21:02:16] ::error::private-controller-output\n"
            "[21:02:16] [behavioral] skill='lemonade-router-builder' model='opus': private-agent-prompt\n"
        )
        lines = orchestrai_logs.public_test_log(
            {"stdout": output}, skill="lemonade-router-builder"
        )
        encoded = "\n".join(lines)
        self.assertEqual(len(lines), 4)
        for forbidden in (
            "private-secret",
            "private-user",
            "private.internal",
            "private-controller-output",
            "private-agent-prompt",
            "::error::",
            "https://",
        ):
            self.assertNotIn(forbidden, encoded)
        self.assertIn(
            "[REDACTED: credential-related grader output; inspect the redacted test log]",
            lines,
        )

    def test_public_log_status_distinguishes_filtering_from_missing_streams(
        self,
    ) -> None:
        self.assertEqual(orchestrai_logs.public_log_status({}, []), "unavailable")
        self.assertEqual(
            orchestrai_logs.public_log_status({"stdout": "   "}, []), "unavailable"
        )
        self.assertEqual(
            orchestrai_logs.public_log_status({"stdout": "private output"}, []),
            "no_recognized_lines",
        )
        self.assertEqual(
            orchestrai_logs.public_log_status({}, ["[PASS] (files_exist) out.png"]),
            "available",
        )

    def test_preserves_case_checks_and_judge_explanations(self) -> None:
        self.assertEqual(
            orchestrai_logs.public_test_log(
                {
                    "stdout": "[21:02:16] [FAIL] build-router: 0/0 checks in 5.0s -- RuntimeError: a-short-private-value\n",
                },
                skill="lemonade-router-builder",
            ),
            [
                "[FAIL] build-router: 0/0 checks in 5.0s -- [Error details withheld; inspect the redacted test log]"
            ],
        )
        output = (
            "[behavioral] local-ai-use: 1 case(s)\n"
            "  [PASS] (files_exist) out.png\n"
            "  [FAIL] (expected_behavior) Include coding keywords -- llm_judge: "
            "Functions and bugs were matched with regex instead of keywords_any.\n"
            "  [FAIL] generate-cat-image: 1/2 checks in 64.25s\n"
        )
        lines = orchestrai_logs.public_test_log(
            {"stdout": output}, skill="local-ai-use"
        )
        self.assertEqual(len(lines), 4)
        self.assertIn("Functions and bugs were matched with regex", lines[2])
        self.assertEqual(
            lines,
            orchestrai_logs.public_manifest_log(
                {
                    "skill": "local-ai-use",
                    "public_log": lines,
                }
            ),
        )

    def test_excludes_dependencies_actor_details_and_agent_transcripts(self) -> None:
        output = (
            "Job started on actor 10.1.2.3\n"
            "[deps] username=private-account token=do-not-export\n"
            "[behavioral] skill='local-ai-use' model='opus': a private prompt\n"
            '{"tool": "Bash", "output": "private-agent-transcript"}\n'
            "[PASS] (files_exist) out.png\n"
            "[evals] JSON report: /home/private-account/report.json\n"
        )
        lines = orchestrai_logs.public_test_log(
            {"stdout": output}, skill="local-ai-use"
        )
        self.assertEqual(lines, ["[PASS] (files_exist) out.png"])

    def test_redacts_private_values_and_identifiers_in_grader_text(self) -> None:
        plan = {"sessions": [{"machine_tags": ["private-fleet-selector"]}]}
        live = {"actors": [{"name": "private-node", "ip": "10.1.2.3"}]}
        private_values = orchestrai_logs.private_log_values(
            plan, live, {"LLM_GATEWAY_KEY": "a-secret-with-spaces"}
        )
        line = (
            "[FAIL] (expected_behavior) Run the server -- llm_judge: "
            "See https://private.internal/log?view=full at 10.1.2.3 "
            "and gateway.corp, host=another-private-node, private-node "
            "on private-fleet-selector, /home/user/work.py or C:\\Users\\private\\run.py "
            "with 0123456789abcdef0123456789abcdef and a-secret-with-spaces "
            "for private.person@company.com."
        )
        result = "\n".join(
            orchestrai_logs.public_test_log(
                {"stdout": line}, skill="local-ai-use", private_values=private_values
            )
        )
        for forbidden in (
            "https://",
            "10.1.2.3",
            "gateway.corp",
            "another-private-node",
            "private-node",
            "private-fleet-selector",
            "/home/user",
            "C:\\Users",
            "0123456789abcdef",
            "a-secret-with-spaces",
            "private.person",
        ):
            self.assertNotIn(forbidden, result)
        self.assertIn("[FAIL] (expected_behavior)", result)

    def test_suppresses_credentials_and_ipv6_even_on_valid_grader_lines(self) -> None:
        for text in (
            "ANTHROPIC_API_KEY=raw-short-key",
            '"Ocp-Apim-Subscription-Key": "raw-short-key"',
            "Authorization: Bearer raw-short-key",
            "fe80::abcd:1234",
        ):
            with self.subTest(text=text):
                lines = orchestrai_logs.public_test_log(
                    {
                        "stdout": "[FAIL] (expected_behavior) Check -- llm_judge: "
                        + text
                    },
                    skill="local-ai-use",
                )
                self.assertEqual(len(lines), 1)
                self.assertNotIn("raw-short-key", "".join(lines))
                self.assertNotIn("fe80", "".join(lines))
                self.assertEqual(
                    lines,
                    orchestrai_logs.public_manifest_log(
                        {
                            "skill": "local-ai-use",
                            "public_log": lines,
                        }
                    ),
                )

    def test_neutralizes_workflow_commands_and_summary_fences(self) -> None:
        lines = orchestrai_logs.public_test_log(
            {
                "stdout": "\x1b[31m[PASS] (files_exist) out.png\x1b[0m\n"
                "[FAIL] (expected_behavior) message ::warning::forged ``` <details>\n"
            },
            skill="local-ai-use",
        )
        rendered = "\n".join(lines)
        self.assertNotIn("::warning::", rendered)
        self.assertNotIn("```", rendered)
        self.assertNotIn("\x1b", rendered)
        self.assertIn("[PASS] (files_exist) out.png", rendered)

    def test_bounds_output_and_rejects_oversized_manifest_lines(self) -> None:
        lines = orchestrai_logs.public_test_log(
            {"stdout": "[PASS] (files_exist) out.png\n" * 1000}, skill="local-ai-use"
        )
        self.assertEqual(len(lines), orchestrai_logs.MAX_PUBLIC_LOG_LINES + 1)
        self.assertIn("truncated", lines[-1])
        self.assertEqual(
            orchestrai_logs.public_manifest_log(
                {
                    "skill": "local-ai-use",
                    "public_log": ["[FAIL] (expected_behavior) " + "x " * 2000],
                }
            ),
            [],
        )

    def test_revalidates_untrusted_manifest_and_ignores_raw_streams(self) -> None:
        lines = orchestrai_logs.public_manifest_log(
            {
                "skill": "local-ai-use",
                "stdout": "do-not-export",
                "public_log": [
                    "do-not-export",
                    "::error::forged",
                    {},
                    "[PASS] (files_exist) out.png",
                ],
            }
        )
        self.assertEqual(lines, ["[PASS] (files_exist) out.png"])


class VerdictTests(unittest.TestCase):
    def test_controller_metadata_exports_fixed_states_only(self) -> None:
        run = {
            "pipeline_status": "running",
            "tests_status": "failed",
            "cleanup_status": "confirmed",
            "final_pipeline_status": "cancelled",
            "id": "private-run-id",
            "portal_url": "https://private.internal",
            "failure_summary": ["password=private-secret"],
        }
        state = orchestrai_verdict.public_controller_state(run)
        self.assertEqual(
            state,
            {
                "pipeline_status": "running",
                "tests_status": "failed",
                "cleanup_status": "confirmed",
                "final_pipeline_status": "cancelled",
            },
        )
        item = {
            "status": "error",
            "error": "OrchestrAI infrastructure ended before the requested test completed.",
        }
        document = orchestrai_verdict._summary_document(
            skill="local-ai-use", os_name="Linux", item=item, run=run, ok=False
        )
        self.assertEqual(document["controller"], state)
        self.assertNotIn("private", json.dumps(document))
        for raw in (
            None,
            "private",
            {
                "pipeline_status": {},
                "tests_status": ["secret"],
                "cleanup_status": "private-secret",
                "final_pipeline_status": "private-state",
            },
        ):
            self.assertEqual(
                set(orchestrai_verdict.public_controller_state(raw).values()),
                {"unknown"},
            )

    def test_job_summary_includes_category_coverage_and_cleanup_context(self) -> None:
        item = {
            "skill": "local-ai-use",
            "os": "Linux",
            "status": "error",
            "error": "OrchestrAI infrastructure ended before the requested test completed.",
            "public_log_status": "unavailable",
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "summary.md"
            with mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(path)}):
                orchestrai_verdict._write_step_summary(
                    item,
                    {"pipeline_status": "failed", "cleanup_status": "not_needed"},
                    ok=False,
                    mock=False,
                )
            text = path.read_text()
        self.assertIn("Execution / infrastructure error", text)
        self.assertIn("Test output unavailable", text)
        self.assertIn("| Parent termination | `not_needed` |", text)
        self.assertIn("not a machine-release verification", text)

    def test_complete_case_report_is_visible_in_job_log_summary_and_artifact(
        self,
    ) -> None:
        item = {
            "skill": "lemonade-router-builder",
            "os": "Linux",
            "status": "failed",
            "error": "One or more behavioral expectations were not met.",
            "report_url": "https://reports.example/launch/42",
            "public_log": orchestrai_logs.public_test_log(
                {"stdout": router_grader_output()}, skill="lemonade-router-builder"
            ),
            "behavioral": {"detail": "private-injected-summary"},
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "results.json"
            manifest.write_text(json.dumps({"items": [item]}), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "orchestrai_verdict.py"),
                    "--results",
                    str(manifest),
                    "--skill",
                    item["skill"],
                    "--os",
                    item["os"],
                    "--output-dir",
                    str(root / "output"),
                ],
                capture_output=True,
                text=True,
                check=False,
                env=isolated_subprocess_env(
                    GITHUB_STEP_SUMMARY=str(root / "summary.md")
                ),
            )
            self.assertEqual(completed.returncode, 1)
            summary_text = (root / "summary.md").read_text(encoding="utf-8")
            for public_output in (completed.stdout, summary_text):
                self.assertIn(
                    "**2/3 cases passed** (20/21 individual expectations) on `opus` (effort `high`).",
                    public_output,
                )
                self.assertIn(
                    "| `keyword-router` | expected_behavior | Output curl commands",
                    public_output,
                )
                self.assertIn("no curl commands", public_output)
                self.assertNotIn("private-injected-summary", public_output)
                self.assertNotIn("actor-supplied table", public_output)
            self.assertLess(
                completed.stdout.index("Unmet expectations"),
                completed.stdout.index("::group::Legacy grader output"),
            )
            self.assertIn("stdout-stderr.log", summary_text)
            document = json.loads((root / "output" / "summary.json").read_text())
            self.assertEqual(document["total_tests"], 1)
            self.assertEqual(document["failed"], 1)
            self.assertEqual(document["behavioral"]["cases"], 3)
            self.assertEqual(document["behavioral"]["expectations"], 21)
            self.assertNotIn("private-injected-summary", json.dumps(document))

    def test_filtered_log_diagnostic_never_echoes_untrusted_fields(self) -> None:
        item = {
            "skill": "local-ai-use",
            "os": "Linux",
            "status": "failed",
            "public_log_status": "no_recognized_lines",
            "stdout": "private-output",
            "stderr": "private-secret",
        }
        with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            orchestrai_verdict._write_job_log(item, ok=False, mock=False)
        self.assertIn("test output was received", output.getvalue())
        self.assertNotIn("private", output.getvalue())
        document = orchestrai_verdict._summary_document(
            skill="local-ai-use", os_name="Linux", item=item, run={}, ok=False
        )
        self.assertEqual(document["public_log_status"], "no_recognized_lines")
        self.assertNotIn("private", json.dumps(document))
        item["public_log_status"] = "private-injected-diagnostic"
        with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
            orchestrai_verdict._write_job_log(item, ok=False, mock=False)
        self.assertIn("test output was unavailable", output.getvalue())
        self.assertNotIn("private", output.getvalue())

    def test_preserves_only_allow_listed_behavioral_diagnostics(self) -> None:
        for diagnostic in sorted(orchestrai_verdict.SAFE_BEHAVIORAL_DIAGNOSTICS):
            with self.subTest(diagnostic=diagnostic):
                self.assertEqual(orchestrai_verdict._safe_error(diagnostic), diagnostic)

        self.assertEqual(
            orchestrai_verdict._safe_error(
                "One or more behavioral expectations were not met on "
                "private-host with token=secret"
            ),
            "The behavioral result could not be verified.",
        )

    def test_confirmed_test_failure_is_not_reported_as_unverified(self) -> None:
        self.assertEqual(
            orchestrai_verdict._safe_error("The behavioral test did not pass."),
            "The behavioral test failed.",
        )

    def test_infrastructure_cause_wins_over_missing_result_wrapper(self) -> None:
        error = (
            "expected exactly one live session, found 0; infrastructure: "
            "Machine acquisition timed out after 2400s."
        )
        self.assertEqual(
            orchestrai_verdict._safe_error(error),
            "Machine acquisition timed out.",
        )

    def test_portal_timeout_is_not_reported_as_a_behavioral_timeout(self) -> None:
        self.assertEqual(
            orchestrai_verdict._safe_error("OrchestrAI Portal request timed out"),
            "The OrchestrAI control plane was unreachable.",
        )

    def test_evaluates_only_the_requested_item(self) -> None:
        manifest = {
            "run": {"reportportal_url": "https://reports.example/launch/42"},
            "items": [
                {"skill": "local-ai-use", "os": "Linux", "status": "passed"},
                {"skill": "local-ai-use", "os": "Windows", "status": "failed"},
            ],
        }
        item, run, ok = orchestrai_verdict.evaluate(
            manifest, skill="local-ai-use", os_name="Linux"
        )
        self.assertTrue(ok)
        self.assertEqual(item["status"], "passed")
        self.assertIsInstance(run, dict)

    def test_command_writes_per_item_summary_and_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = Path(temp) / "results.json"
            output_dir = Path(temp) / "test-results"
            step_summary = Path(temp) / "step-summary.md"
            manifest_path.write_text(
                json.dumps(
                    {
                        "run": {"portal_url": "https://portal.example/runs/42"},
                        "items": [
                            {
                                "skill": "local-ai-use",
                                "os": "Linux",
                                "status": "passed",
                                "session": "private-session-id",
                                "job_id": "private-job-id",
                                "path": "https://internal-controller/path",
                                "error": "",
                                "report_url": "https://reports.example/launch/42",
                                "stdout": "hello from the actor\n",
                                "public_streams": {
                                    "stdout": "hello from the actor\n[PASS] (files_exist) out.png\n[PASS] generate-cat-image: 1/1 checks in 12s\n",
                                    "stderr": "",
                                },
                                "public_log": [
                                    "[PASS] (files_exist) out.png",
                                    "[PASS] generate-cat-image: 1/1 checks in 12s",
                                ],
                                "stderr": "",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "orchestrai_verdict.py"),
                    "--results",
                    str(manifest_path),
                    "--skill",
                    "local-ai-use",
                    "--os",
                    "Linux",
                    "--output-dir",
                    str(output_dir),
                ],
                text=True,
                capture_output=True,
                env=isolated_subprocess_env(GITHUB_STEP_SUMMARY=str(step_summary)),
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Result: passed", completed.stdout)
            self.assertNotIn("reports.example", completed.stdout)
            self.assertIn("hello from the actor", completed.stdout)
            self.assertFalse((output_dir / "stdout.log").exists())
            self.assertIn("[PASS] (files_exist) out.png", completed.stdout)
            self.assertIn("Test stdout/stderr (redacted)", completed.stdout)
            self.assertIn(
                "hello from the actor", (output_dir / "stdout-stderr.log").read_text()
            )
            self.assertIn("out.png", (output_dir / "sanitized.log").read_text())
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["passed"], 1)
            self.assertNotIn("report_url", summary)
            self.assertIn("stdout-stderr.log", step_summary.read_text(encoding="utf-8"))
            self.assertNotIn(
                "reports.example", step_summary.read_text(encoding="utf-8")
            )
            self.assertNotIn("stdout", summary["results"][0])
            encoded = json.dumps(summary)
            self.assertNotIn("private-session-id", encoded)
            self.assertNotIn("private-job-id", encoded)
            self.assertNotIn("internal-controller", encoded)

    def test_verdict_classifies_untrusted_manifest_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = Path(temp) / "results.json"
            output_dir = Path(temp) / "test-results"
            manifest_path.write_text(
                json.dumps(
                    {
                        "run": {
                            "id": "private-run-id",
                            "reportportal_url": "https://reports.example/launch/42",
                        },
                        "items": [
                            {
                                "skill": "local-ai-use",
                                "os": "Linux",
                                "status": "server-private-status",
                                "duration": "host=internal-machine",
                                "job_id": "private-job-id",
                                "report_url": "https://user:secret@reports.example/launch/42",
                                "error": "password=hunter2 on internal-machine",
                                "public_log": [
                                    "raw actor output from internal-machine",
                                    "[FAIL] (expected_behavior) Failure password=hunter2",
                                    "[FAIL] (expected_behavior) See https://internal-machine/path",
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "orchestrai_verdict.py"),
                    "--results",
                    str(manifest_path),
                    "--skill",
                    "local-ai-use",
                    "--os",
                    "Linux",
                    "--output-dir",
                    str(output_dir),
                ],
                text=True,
                capture_output=True,
                env=isolated_subprocess_env(),
                check=False,
            )
            self.assertEqual(completed.returncode, 1)
            encoded = (output_dir / "summary.json").read_text(encoding="utf-8")
            encoded += completed.stdout + completed.stderr
            if (output_dir / "sanitized.log").exists():
                encoded += (output_dir / "sanitized.log").read_text(encoding="utf-8")
            for forbidden in (
                "private-run-id",
                "private-job-id",
                "internal-machine",
                "hunter2",
                "server-private-status",
                "reports.example",
            ):
                self.assertNotIn(forbidden, encoded)

    def test_failed_verdict_shows_the_actual_sanitized_grader_reason(self) -> None:
        explanation = (
            "Functions and bugs were matched via regex instead of keywords_any."
        )
        item = {
            "skill": "lemonade-router-builder",
            "os": "Windows",
            "status": "failed",
            "error": "One or more behavioral expectations were not met.",
            "report_url": "https://reports.example/launch/42",
            "public_log": [
                "[FAIL] (expected_behavior) Include keywords_any condition "
                "matching coding-related terms -- llm_judge: " + explanation,
                "[FAIL] keyword-router: 8/9 checks in 64.25s",
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = Path(temp) / "results.json"
            manifest_path.write_text(json.dumps({"items": [item]}), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "orchestrai_verdict.py"),
                    "--results",
                    str(manifest_path),
                    "--skill",
                    item["skill"],
                    "--os",
                    item["os"],
                    "--output-dir",
                    str(Path(temp) / "output"),
                ],
                capture_output=True,
                text=True,
                check=False,
                env=isolated_subprocess_env(
                    GITHUB_STEP_SUMMARY=str(Path(temp) / "summary.md")
                ),
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn(explanation, completed.stdout)
            self.assertNotIn("https://reports.example/launch/42", completed.stdout)
            self.assertIn(explanation, (Path(temp) / "summary.md").read_text())
            self.assertIn(
                explanation, (Path(temp) / "output" / "sanitized.log").read_text()
            )

    def test_missing_manifest_writes_a_failed_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output_dir = Path(temp) / "test-results"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "orchestrai_verdict.py"),
                    "--results",
                    str(Path(temp) / "missing.json"),
                    "--skill",
                    "local-ai-use",
                    "--os",
                    "Linux",
                    "--output-dir",
                    str(output_dir),
                ],
                text=True,
                capture_output=True,
                env=isolated_subprocess_env(),
                check=False,
            )
            self.assertEqual(completed.returncode, 1)
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(summary["status"], "error")
            self.assertNotIn(str(Path(temp) / "missing.json"), summary["error"])


class ReportTests(unittest.TestCase):
    def test_routing_style_tables_use_behavioral_rates_not_routing_categories(
        self,
    ) -> None:
        rows = {
            ("lemonade-router-builder", "Windows"): {
                "status": "failed",
                "public_log": orchestrai_logs.public_test_log(
                    {"stdout": router_grader_output()}, skill="lemonade-router-builder"
                ),
            },
            ("lemonade-router-builder", "Linux"): {
                "status": "passed",
                "public_log": [
                    "[behavioral] lemonade-router-builder: 1 case(s)",
                    "[PASS] (files_exist) router.json",
                    "[PASS] simple-router: 1/1 checks in 3s",
                    "1/1 cases passed (1/1 individual expectations) on opus (effort high).",
                ],
            },
        }
        text = orchestrai_report.render(expected=list(rows), rows=rows)
        self.assertIn("| Verdict | Count | Meaning |", text)
        self.assertIn("| `passed` | 1 |", text)
        self.assertIn("| `failed` | 1 |", text)
        self.assertIn("By expectation type", text)
        self.assertIn("| `expected_behavior` | 21 | 20 | 1 | 95% |", text)
        self.assertIn(
            "| `lemonade-router-builder` | 2 | 1 | 3/4 | 21/22 | 75% | 95% | 2/2 complete |",
            text,
        )
        self.assertIn("Evaluated separately", text)
        self.assertIn("not routing recall or precision", text)
        for misleading in (
            "correct_trigger",
            "true_negative",
            "near_miss",
            "Installed together:",
        ):
            self.assertNotIn(misleading, text)
        self.assertLess(
            text.index("### Verdicts"), text.index("### By expectation type")
        )
        self.assertLess(
            text.index("### By expectation type"), text.index("### Per skill")
        )

    def test_verdict_table_counts_every_requested_state_once_without_fake_rates(
        self,
    ) -> None:
        statuses = [
            "passed",
            "failed",
            "error",
            "mock",
            "skipped",
            "cancelled",
            "aborted",
            "missing",
            "unknown",
        ]
        rows = {
            (f"skill-{index}", "Linux"): {"status": status}
            for index, status in enumerate(statuses)
        }
        text = orchestrai_report.render(expected=list(rows), rows=rows)
        self.assertIn("| `cancelled / aborted` | 2 |", text)
        self.assertIn("| `missing / unknown` | 2 |", text)
        self.assertIn("| `mock` | 1 |", text)
        self.assertIn("| `error` | 1 |", text)
        self.assertIn("No publishable expectation-level results", text)
        self.assertNotIn("100%", text)

    def test_overview_separates_verdicts_cases_and_missing_reports(self) -> None:
        failed = {
            "skill": "lemonade-router-builder",
            "os": "Windows",
            "status": "failed",
            "error": "One or more behavioral expectations were not met.",
            "public_log": orchestrai_logs.public_test_log(
                {"stdout": router_grader_output()}, skill="lemonade-router-builder"
            ),
        }
        rows = {
            ("lemonade-router-builder", "Windows"): failed,
            ("local-ai-use", "Windows"): {
                "status": "passed",
                "public_log_status": "unavailable",
            },
        }
        with mock.patch.dict(
            os.environ,
            {
                "ORCHESTRAI_CONTROLLER_RESULT": "success",
                "ORCHESTRAI_VERDICT_RESULT": "failure",
            },
        ):
            text = orchestrai_report.render(
                expected=[
                    ("lemonade-router-builder", "Windows"),
                    ("local-ai-use", "Windows"),
                    ("local-ai-use", "Linux"),
                ],
                rows=rows,
            )
        self.assertIn("3 skill/OS evaluations requested", text)
        self.assertIn("Passed: 1 · test failures: 1 · execution errors: 1", text)
        self.assertIn("Observed graded cases:** 2/3 passed", text)
        self.assertIn("expectations:** 20/21 met", text)
        self.assertIn("1 complete · 0 partial · 2 unavailable", text)
        self.assertIn("| Linux | 1 | 0 | 0 | 1 | 0 |", text)
        self.assertIn("| Windows | 2 | 1 | 1 | 0 | 0 |", text)
        self.assertIn("| OrchestrAI controller | `success` |", text)
        self.assertIn("| Per-skill verdicts | `failure` |", text)
        self.assertLess(text.index("Needs attention"), text.index("All skill results"))
        self.assertLess(text.index("no curl commands"), text.index("All skill results"))
        self.assertIn("Reporting / result error", text)

    def test_report_revalidates_untrusted_artifact_and_environment_fields(self) -> None:
        row = {
            "skill": "private-host",
            "os": "private-os",
            "status": "private-status",
            "duration": "password=private-secret",
            "error": "password=private-secret on private-host at 10.1.2.3",
            "controller": {"pipeline_status": "private-state", "cleanup_status": {}},
            "behavioral": {"cases": 1000, "model": "private-model"},
            "public_log_status": "private-message",
        }
        with mock.patch.dict(
            os.environ,
            {
                "ORCHESTRAI_CONTROLLER_RESULT": "private-controller",
                "ORCHESTRAI_VERDICT_RESULT": "private-verdict",
            },
        ):
            text = orchestrai_report.render(
                expected=[("local-ai-use", "Linux")],
                rows={("local-ai-use", "Linux"): row},
            )
        for value in (
            "private-host",
            "private-os",
            "private-status",
            "private-secret",
            "10.1.2.3",
            "private-model",
            "private-message",
            "private-controller",
            "private-verdict",
        ):
            self.assertNotIn(value, text)
        self.assertIn("Missing / unverified result", text)
        self.assertIn("Not reported", text)
        self.assertNotIn("0/0", text)

    def test_partial_and_plan_only_counts_are_not_reported_as_complete_tests(
        self,
    ) -> None:
        partial = {
            "status": "passed",
            "public_log": [
                "[behavioral] local-ai-use: 2 case(s)",
                "[PASS] generate-image: 1/1 checks in 3s",
            ],
        }
        mock_item = {
            "status": "mock",
            "public_log": orchestrai_logs.public_test_log(
                {"stdout": router_grader_output()}, skill="lemonade-router-builder"
            ),
        }
        text = orchestrai_report.render(
            expected=[
                ("local-ai-use", "Linux"),
                ("lemonade-router-builder", "Windows"),
            ],
            rows={
                ("local-ai-use", "Linux"): partial,
                ("lemonade-router-builder", "Windows"): mock_item,
            },
        )
        self.assertIn("0 complete · 1 partial · 0 unavailable · 1 not run", text)
        self.assertIn("Observed graded cases:** 1/1 passed", text)
        self.assertIn("1/1 (partial)", text)
        self.assertIn("No unmet expectations in the recorded cases", text)
        self.assertNotIn("Every behavioral case", text)
        self.assertNotIn("2/3", text)

    def test_controller_snapshots_do_not_imply_machine_release(self) -> None:
        rows = {
            ("local-ai-use", "Linux"): {
                "status": "passed",
                "controller": {
                    "pipeline_status": "running",
                    "tests_status": "passed",
                    "cleanup_status": "confirmed",
                    "machine": "private-host",
                },
            }
        }
        text = orchestrai_report.render(expected=list(rows), rows=rows)
        self.assertIn("Controller snapshots", text)
        self.assertIn("| Linux | `running` | `passed` | `confirmed` |", text)
        self.assertIn("machine release remains managed by OrchestrAI", text)
        self.assertNotIn("private-host", text)

    def test_aggregate_includes_case_report_from_revalidated_public_lines(self) -> None:
        row = {
            "skill": "lemonade-router-builder",
            "os": "Linux",
            "status": "failed",
            "public_log": orchestrai_logs.public_test_log(
                {"stdout": router_grader_output()}, skill="lemonade-router-builder"
            ),
            "behavioral": {"detail": "private-injected-summary"},
        }
        rendered = orchestrai_report.render(
            expected=[("lemonade-router-builder", "Linux")],
            rows={("lemonade-router-builder", "Linux"): row},
        )
        self.assertIn("1 failed, 0 passed", rendered)
        self.assertIn("lemonade-router-builder on Linux — case results", rendered)
        self.assertIn("**2/3 cases passed** (20/21 individual expectations)", rendered)
        self.assertIn(
            "| `keyword-router` | expected_behavior | Output curl commands", rendered
        )
        self.assertNotIn("private-injected-summary", rendered)
        self.assertNotIn("actor-supplied table", rendered)

    def test_aggregate_fails_closed_when_an_expected_artifact_is_missing(self) -> None:
        rows = {
            ("local-ai-use", "Linux"): {
                "skill": "local-ai-use",
                "os": "Linux",
                "status": "passed",
                "report_url": "https://reports.example/launch/42",
            }
        }
        rendered = orchestrai_report.render(
            expected=[
                ("local-ai-use", "Linux"),
                ("local-ai-use", "Windows"),
            ],
            rows=rows,
        )
        self.assertIn("1 failed, 1 passed", rendered)
        self.assertIn("no per-skill result artifact was published", rendered)
        self.assertIn("stdout-stderr.log", rendered)
        self.assertNotIn("reports.example", rendered)

    def test_workflow_never_uploads_the_live_controller_snapshot(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "evals.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("orchestrai-live.json", workflow)

    def test_workflow_masks_private_fleet_tags_as_a_secret(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "evals.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "DEVICE_TAGS_JSON: ${{ secrets.ORCHESTRAI_DEVICE_TAGS }}", workflow
        )
        self.assertNotIn(
            "DEVICE_TAGS_JSON: ${{ vars.ORCHESTRAI_DEVICE_TAGS }}", workflow
        )

    def test_workflow_masks_linux_driver_map_as_a_secret(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "evals.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON: "
            "${{ secrets.ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON }}",
            workflow,
        )
        self.assertNotIn(
            "ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON: "
            "${{ vars.ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON }}",
            workflow,
        )

    def test_workflow_preserves_cancellation_and_required_check_contracts(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "evals.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("run: exec python3 .github/scripts/orchestrai_run.py", workflow)
        self.assertIn("'evals / results'", workflow)
        self.assertIn("!cancelled()", workflow)
        # Both gate labels must start a run: enable_mi_ci is the Instinct gate.
        self.assertIn(
            """contains(fromJSON('["run_behavioral","enable_mi_ci"]'), github.event.label.name)""",
            workflow,
        )
        self.assertNotIn("github.event.label.name == 'run_behavioral'", workflow)

    def test_workflow_tests_the_pr_head_and_fails_closed_without_hardware(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "evals.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "TARGET_SHA: ${{ github.event.pull_request.head.sha || github.sha }}",
            workflow,
        )
        self.assertNotIn("git ls-remote", workflow)
        self.assertIn(
            "Strix behavioral cases require the run_behavioral label.",
            workflow,
        )

    def test_discovery_and_orchestrai_use_the_same_commit(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "evals.yml").read_text(
            encoding="utf-8"
        )
        discover = workflow.split("\n  discover:\n", 1)[1].split(
            "\n  external-references:\n", 1
        )[0]
        checkout = discover.split("- name: Check out repository", 1)[1].split(
            "- name: Check out Skillscope", 1
        )[0]
        tested_commit = "${{ github.event.pull_request.head.sha || github.sha }}"
        self.assertIn(f"ref: {tested_commit}", checkout)
        self.assertIn("fetch-depth: 0", checkout)
        self.assertIn(f"TARGET_SHA: {tested_commit}", workflow)

    def test_reporting_edits_do_not_select_every_hardware_case(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "evals.yml").read_text(
            encoding="utf-8"
        )
        infra_line = next(
            line for line in workflow.splitlines() if '"--infra-paths"' in line
        )
        self.assertNotIn("orchestrai_report.py", infra_line)
        self.assertNotIn("test_orchestrai_evals.py", infra_line)

    def test_public_workflow_does_not_publish_control_plane_links(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "evals.yml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("PUBLISH_REPORT_LINK", workflow)
        self.assertNotIn("View in OrchestrAI Portal", workflow)
        self.assertNotIn("View in Jenkins", workflow)

    def test_routing_does_not_consume_a_strix_runner(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "evals.yml").read_text(
            encoding="utf-8"
        )
        routing = workflow.split("\n  routing:\n", 1)[1].split(
            "\n  orchestrai-behavioral:\n", 1
        )[0]
        self.assertIn(
            "runs-on: ${{ vars.ORCHESTRAI_CONTROL_RUNNER || 'ubuntu-latest' }}",
            routing,
        )
        self.assertNotIn("strix_halo", routing)

    def test_workflow_uses_playbooks_orchestrai_runner_variables(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "evals.yml").read_text(
            encoding="utf-8"
        )
        controller = workflow.split("\n  orchestrai-behavioral:\n", 1)[1].split(
            "\n  orchestrai-verdict:\n", 1
        )[0]
        verdict = workflow.split("\n  orchestrai-verdict:\n", 1)[1].split(
            "\n  behavior-scoped:\n", 1
        )[0]
        self.assertIn(
            "runs-on: ${{ vars.ORCHESTRAI_CONTROL_RUNNER || 'ubuntu-latest' }}",
            controller,
        )
        self.assertIn(
            "runs-on: ${{ vars.ORCHESTRAI_WAIT_RUNNER || 'ubuntu-latest' }}",
            verdict,
        )

    def test_skillscope_pin_uses_one_release_version(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "evals.yml").read_text(
            encoding="utf-8"
        )
        config = json.loads(
            (ROOT / ".github" / "orchestrai-config.json").read_text(encoding="utf-8")
        )
        self.assertEqual(workflow.count("SKILLSCOPE_VERSION: v0.1.3"), 1)
        self.assertNotIn("SKILLSCOPE_SHA", workflow)
        self.assertNotIn("skillscope_sha", config)

    def test_verdicts_run_after_a_controller_failure(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "evals.yml").read_text(
            encoding="utf-8"
        )
        verdict = workflow.split("\n  orchestrai-verdict:\n", 1)[1].split(
            "\n  behavior-scoped:\n", 1
        )[0]
        self.assertIn("!cancelled()", verdict)
        self.assertNotIn("needs.orchestrai-behavioral.result == 'success'", verdict)


if __name__ == "__main__":
    unittest.main()
