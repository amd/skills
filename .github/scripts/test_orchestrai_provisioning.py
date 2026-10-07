#!/usr/bin/env python3
"""Offline Windows provisioning contracts, without actual fleet secrets."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import build_orchestrai_plan
import orchestrai_run


def plan_args(os_name: str = "Windows"):
    return build_orchestrai_plan._parser().parse_args(
        [
            "--matrix-json",
            json.dumps([{"skill": "local-ai-use", "os": os_name}]),
            "--device-tags-json",
            '{"strix_halo":["fixture-tag"]}',
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


class WindowsProvisioningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = mock.patch.dict(
            os.environ,
            {
                "ORCHESTRAI_WINDOWS_DRIVER_SOURCE": "",
                "ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON": "",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_windows_uses_private_playbooks_driver_payload_and_shared_reboot(self):
        source = r"\\fixture-server\fixture-share\driver build\ATI"
        with mock.patch.dict(os.environ, {"ORCHESTRAI_WINDOWS_DRIVER_SOURCE": source}):
            plan = build_orchestrai_plan.build_plan(plan_args())
        builds = plan["builds_json"]
        self.assertEqual(
            builds,
            {
                "vars": {"driver_source": source, "driver_copy": "direct"},
                "install_scripts": [
                    {
                        "script": "InstallationScripts/gfx/windows.ps1",
                        "reboot_after": True,
                    }
                ],
            },
        )
        orchestrai_run.require_windows_provisioning(plan)
        before = deepcopy(plan)
        self.assertEqual(
            orchestrai_run.build_run_request(plan, "fixture-plan"),
            {
                "plan_id": "fixture-plan",
                "builds_json": builds,
            },
        )
        self.assertEqual(plan, before)
        self.assertNotIn(source, json.dumps(plan["sessions"]))
        self.assertNotIn("driver_source", json.dumps(plan["sessions"]))
        self.assertEqual(
            plan["sessions"][0]["tests"][0]["variables"][
                "SKILLS_WINDOWS_GPU_PREFLIGHT"
            ],
            "required",
        )

    def test_locally_staged_source_is_supported(self):
        source = r"C:\Temp\DriverSource\ATI"
        with mock.patch.dict(os.environ, {"ORCHESTRAI_WINDOWS_DRIVER_SOURCE": source}):
            plan = build_orchestrai_plan.build_plan(plan_args())
        self.assertEqual(plan["builds_json"]["vars"]["driver_source"], source)

    def test_unc_folder_source_accepts_a_trailing_backslash(self):
        source = "\\\\fixture-server\\fixture-share\\ATI\\"
        with mock.patch.dict(os.environ, {"ORCHESTRAI_WINDOWS_DRIVER_SOURCE": source}):
            plan = build_orchestrai_plan.build_plan(plan_args())
        self.assertEqual(plan["builds_json"]["vars"]["driver_source"], source)

    def test_unsupported_sources_fail_without_disclosing_source(self):
        for source in (
            "https://fixture-private.example/driver.zip",
            "relative-fixture-private-path",
            r"\\fixture-private-server\share\..\driver",
            "C:\\fixture-private-path\n\\driver",
        ):
            with self.subTest(source_type=source[:1]):
                with mock.patch.dict(
                    os.environ, {"ORCHESTRAI_WINDOWS_DRIVER_SOURCE": source}
                ):
                    with self.assertRaises(SystemExit) as raised:
                        build_orchestrai_plan.build_plan(plan_args())
                self.assertNotIn(source, str(raised.exception))
                self.assertNotIn("fixture-private", str(raised.exception))

    def test_config_requires_shared_graphics_installer_and_reboot(self):
        config = json.loads(build_orchestrai_plan.DEFAULT_CONFIG.read_text())
        source = r"C:\Fixture\ATI"
        invalid_scripts = (
            [],
            [{"script": "InstallationScripts/gfx/windows.ps1", "reboot_after": False}],
            [
                {
                    "script": "InstallationScripts/../gfx/windows.ps1",
                    "reboot_after": True,
                }
            ],
        )
        for scripts in invalid_scripts:
            with self.subTest(scripts=scripts):
                config["provisioning"]["windows_install_scripts"] = scripts
                with mock.patch.dict(
                    os.environ, {"ORCHESTRAI_WINDOWS_DRIVER_SOURCE": source}
                ):
                    with self.assertRaises(SystemExit):
                        build_orchestrai_plan._windows_builds(config, {"Windows"})

    def test_malformed_provisioning_object_fails_with_fixed_config_error(self):
        for provisioning in (None, 5, [], "fixture-private-value"):
            with self.subTest(provisioning_type=type(provisioning).__name__):
                with mock.patch.dict(
                    os.environ, {"ORCHESTRAI_WINDOWS_DRIVER_SOURCE": r"C:\Fixture\ATI"}
                ):
                    with self.assertRaisesRegex(
                        SystemExit, "provisioning.windows_install_scripts"
                    ):
                        build_orchestrai_plan._windows_builds(
                            {"provisioning": provisioning}, {"Windows"}
                        )

    def test_live_windows_missing_source_fails_before_portal_contact(self):
        plan = build_orchestrai_plan.build_plan(plan_args())
        self.assertNotIn("builds_json", plan)
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "plan.json"
            plan_path.write_text(json.dumps(plan))
            with mock.patch.dict(
                os.environ, {"PLAN_FILE": str(plan_path), "MODE": "live"}
            ):
                with mock.patch.object(orchestrai_run, "PortalClient") as client:
                    with self.assertRaisesRegex(
                        SystemExit, "ORCHESTRAI_WINDOWS_DRIVER_SOURCE"
                    ):
                        orchestrai_run.main()
                    client.assert_not_called()

    def test_live_windows_exact_image_requires_source(self):
        for image in ("windows", "windows11-fixture", "windows/fixture"):
            with self.subTest(image=image):
                with self.assertRaisesRegex(
                    SystemExit, "ORCHESTRAI_WINDOWS_DRIVER_SOURCE"
                ):
                    orchestrai_run.require_windows_provisioning(
                        {"sessions": [{"os_image": image}]}
                    )

    def test_linux_payload_cannot_satisfy_windows_source_requirement(self):
        plan = {
            "sessions": [{"os_image": "windows"}],
            "builds_json": {"vars": {"driver_sources_json": "{}"}},
        }
        with self.assertRaisesRegex(SystemExit, "ORCHESTRAI_WINDOWS_DRIVER_SOURCE"):
            orchestrai_run.require_windows_provisioning(plan)

    def test_live_windows_source_requires_a_nonempty_string(self):
        for source in (None, 5, {}, [], " "):
            with self.subTest(source_type=type(source).__name__):
                with self.assertRaisesRegex(
                    SystemExit, "ORCHESTRAI_WINDOWS_DRIVER_SOURCE"
                ):
                    orchestrai_run.require_windows_provisioning(
                        {
                            "sessions": [{"os_image": "windows"}],
                            "builds_json": {"vars": {"driver_source": source}},
                        }
                    )

    def test_live_windows_requires_rebooting_installer(self):
        with mock.patch.dict(
            os.environ, {"ORCHESTRAI_WINDOWS_DRIVER_SOURCE": r"C:\Fixture\ATI"}
        ):
            plan = build_orchestrai_plan.build_plan(plan_args())
        for change in ("no_reboot", "no_installer", "wrong_copy"):
            broken = deepcopy(plan)
            if change == "no_reboot":
                broken["builds_json"]["install_scripts"][0]["reboot_after"] = False
            elif change == "no_installer":
                broken["builds_json"]["install_scripts"] = []
            else:
                broken["builds_json"]["vars"]["driver_copy"] = "other"
            with self.subTest(change=change):
                with self.assertRaisesRegex(SystemExit, "direct copy and a reboot"):
                    orchestrai_run.require_windows_provisioning(broken)

    def test_mock_windows_needs_no_driver_or_portal_secret(self):
        plan = build_orchestrai_plan.build_plan(plan_args())
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "plan.json"
            results_path = Path(tmp) / "results.json"
            plan_path.write_text(json.dumps(plan))
            with mock.patch.dict(
                os.environ,
                {
                    "PLAN_FILE": str(plan_path),
                    "MODE": "mock",
                    "RESULTS_FILE": str(results_path),
                    "LIVE_FILE": str(Path(tmp) / "live.json"),
                    "GITHUB_OUTPUT": "",
                    "GITHUB_STEP_SUMMARY": "",
                },
            ):
                with mock.patch.object(orchestrai_run, "PortalClient") as client:
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(orchestrai_run.main(), 0)
                    client.assert_not_called()
            self.assertEqual(json.loads(results_path.read_text())["mode"], "mock")

    def test_linux_plan_is_unaffected_by_windows_source(self):
        linux_sources = '{"ubuntu:24.04":"https://drivers.example/fixture.deb"}'
        with mock.patch.dict(
            os.environ,
            {
                "ORCHESTRAI_WINDOWS_DRIVER_SOURCE": "invalid fixture-private-source",
                "ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON": linux_sources,
            },
        ):
            plan = build_orchestrai_plan.build_plan(plan_args("Linux"))
        self.assertEqual(
            plan["builds_json"]["vars"], {"driver_sources_json": linux_sources}
        )
        self.assertTrue(plan["builds_json"]["install_scripts"][-1]["reboot_after"])
        self.assertNotIn(
            "SKILLS_WINDOWS_GPU_PREFLIGHT", plan["sessions"][0]["tests"][0]["variables"]
        )
        orchestrai_run.require_linux_provisioning(plan)
        orchestrai_run.require_windows_provisioning(plan)

    def test_workflow_hands_source_from_secret_to_plan_builder(self):
        workflow = (SCRIPTS.parent / "workflows/evals.yml").read_text()
        self.assertIn(
            "ORCHESTRAI_WINDOWS_DRIVER_SOURCE: ${{ secrets.ORCHESTRAI_WINDOWS_DRIVER_SOURCE }}",
            workflow,
        )
        self.assertNotIn("vars.ORCHESTRAI_WINDOWS_DRIVER_SOURCE", workflow)


if __name__ == "__main__":
    unittest.main()
