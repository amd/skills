#!/usr/bin/env python3
"""Offline tests for the OrchestrAI eval plan and trigger helper."""

from __future__ import annotations

import io
import json
import os
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
        with self.assertRaisesRegex(
            SystemExit, "ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON"
        ):
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

    def test_fails_before_submission_when_image_discovery_fails(self) -> None:
        class Client:
            @staticmethod
            def request(_method: str, _path: str) -> list[dict[str, str]]:
                raise RuntimeError("portal unavailable")

        plan = {"sessions": [{"os_image": "ubuntu"}]}
        with self.assertRaisesRegex(
            RuntimeError, "could not resolve exact stock OS images"
        ):
            orchestrai_run.resolve_stock_os_images(plan, Client())

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
        self.assertNotIn("run-42", json.dumps(manifest))
        self.assertNotIn("reportportal_url", manifest["run"])
        self.assertNotIn("reports.example", json.dumps(manifest))

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
            "stdout",
            "stderr",
        ):
            self.assertNotIn(private_value, encoded)

    def test_unknown_private_test_output_stays_generic(self) -> None:
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
        self.assertNotIn("private endpoint", json.dumps(manifest))

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

    def test_public_outputs_omit_private_report_links(self) -> None:
        live = {
            "rp_url": "https://reports.example/launch/42",
            "launcher_url": "https://launcher.example",
            "launcher": {"sessions": []},
        }
        manifest = orchestrai_run.build_results_manifest(
            reporting_plan(), live, mode="live"
        )
        sanitized = orchestrai_run.sanitized_live_snapshot(live)
        encoded = json.dumps([manifest, sanitized])
        self.assertNotIn("reports.example", encoded)
        self.assertNotIn("reportportal", encoded.lower())

    def test_reportportal_link_is_limited_to_private_repo_summary(self) -> None:
        live = {"rp_url": "https://reports.example/ui/#project/launches/42"}
        previous = os.environ.get("PUBLISH_REPORT_LINK")
        try:
            os.environ["PUBLISH_REPORT_LINK"] = "false"
            self.assertEqual(orchestrai_run.reportportal_url_for_summary(live), "")

            os.environ["PUBLISH_REPORT_LINK"] = "true"
            self.assertEqual(
                orchestrai_run.reportportal_url_for_summary(live), live["rp_url"]
            )
            for unsafe in (
                "http://reports.example/launch/42",
                "https://user:password@reports.example/launch/42",
                "https://reports.example/launch/42\nmalicious",
            ):
                self.assertEqual(
                    orchestrai_run.reportportal_url_for_summary({"rp_url": unsafe}),
                    "",
                )
        finally:
            if previous is None:
                os.environ.pop("PUBLISH_REPORT_LINK", None)
            else:
                os.environ["PUBLISH_REPORT_LINK"] = previous

    def test_portal_run_link_is_limited_to_private_repo_summary(self) -> None:
        run_id = "caa0c086-43dc-490f-b588-ede44d5eab07"
        previous = os.environ.get("PUBLISH_REPORT_LINK")
        try:
            os.environ["PUBLISH_REPORT_LINK"] = "false"
            self.assertEqual(
                orchestrai_run.portal_run_url_for_summary(
                    "https://portal.example", run_id
                ),
                "",
            )

            os.environ["PUBLISH_REPORT_LINK"] = "true"
            self.assertEqual(
                orchestrai_run.portal_run_url_for_summary(
                    "https://portal.example/", run_id
                ),
                f"https://portal.example/#/runs/{run_id}",
            )
            for unsafe_base in (
                "http://portal.example",
                "https://user:password@portal.example",
                "https://portal.example/private-api-base",
                "https://portal.example?redirect=elsewhere",
                "https://portal.example/#/other",
                "https://portal.example\nmalicious",
            ):
                self.assertEqual(
                    orchestrai_run.portal_run_url_for_summary(
                        unsafe_base, run_id
                    ),
                    "",
                )
            self.assertEqual(
                orchestrai_run.portal_run_url_for_summary(
                    "https://portal.example", "not-a-run-id"
                ),
                "",
            )
        finally:
            if previous is None:
                os.environ.pop("PUBLISH_REPORT_LINK", None)
            else:
                os.environ["PUBLISH_REPORT_LINK"] = previous

    def test_summary_links_to_authenticated_run_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            summary_path = Path(temp) / "summary.md"
            previous = os.environ.get("GITHUB_STEP_SUMMARY")
            os.environ["GITHUB_STEP_SUMMARY"] = str(summary_path)
            try:
                orchestrai_run.summary(
                    mode="live",
                    plan_name="skills-evals-1-1-linux",
                    results={"run": {}, "items": []},
                    portal_url="https://portal.example/#/runs/42",
                    report_url="https://reports.example/ui/#project/launches/42",
                )
            finally:
                if previous is None:
                    os.environ.pop("GITHUB_STEP_SUMMARY", None)
                else:
                    os.environ["GITHUB_STEP_SUMMARY"] = previous
            rendered = summary_path.read_text(encoding="utf-8")
            self.assertIn("View in OrchestrAI Portal", rendered)
            self.assertIn("https&#58;//portal.example/#/runs/42", rendered)
            self.assertNotIn("https://portal.example", rendered)
            self.assertIn("View in ReportPortal", rendered)
            self.assertIn("https://reports.example/ui/#project/launches/42", rendered)

    def test_summary_keeps_portal_link_when_no_test_launch_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            summary_path = Path(temp) / "summary.md"
            previous = os.environ.get("GITHUB_STEP_SUMMARY")
            os.environ["GITHUB_STEP_SUMMARY"] = str(summary_path)
            try:
                orchestrai_run.summary(
                    mode="live",
                    plan_name="skills-evals-1-1-windows",
                    results={"run": {}, "items": []},
                    portal_url="https://portal.example/#/runs/42",
                )
            finally:
                if previous is None:
                    os.environ.pop("GITHUB_STEP_SUMMARY", None)
                else:
                    os.environ["GITHUB_STEP_SUMMARY"] = previous
            rendered = summary_path.read_text(encoding="utf-8")
            self.assertIn("View in OrchestrAI Portal", rendered)
            self.assertNotIn("View in ReportPortal", rendered)

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
        passed = [{"status": "passed"}, {"status": "passed"}]
        one_failed = [{"status": "passed"}, {"status": "failed"}]
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


class VerdictTests(unittest.TestCase):
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
                                "stdout": "hello from the actor\n",
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
                env=isolated_subprocess_env(),
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse((output_dir / "stdout.log").exists())
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["passed"], 1)
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
                                "error": "password=hunter2 on internal-machine",
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
            for forbidden in (
                "private-run-id",
                "private-job-id",
                "internal-machine",
                "hunter2",
                "server-private-status",
                "reports.example",
            ):
                self.assertNotIn(forbidden, encoded)

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
    def test_aggregate_fails_closed_when_an_expected_artifact_is_missing(self) -> None:
        rows = {
            ("local-ai-use", "Linux"): {
                "skill": "local-ai-use",
                "os": "Linux",
                "status": "passed",
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

    def test_workflow_only_publishes_report_links_for_private_repositories(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "evals.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "PUBLISH_REPORT_LINK: ${{ github.event.repository.private }}", workflow
        )


if __name__ == "__main__":
    unittest.main()
