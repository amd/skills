"""Reject Windows grades that bypass the required adapter hardware check."""

import unittest

from orchestrai_readiness import (
    enforce_windows_readiness,
    readiness_description,
    safe_readiness,
)
from orchestrai_run import build_results_manifest


def marker(state="ready"):
    return f"[skills-readiness] windows_amd_display={state} compute_backend=not_probed"


class WindowsReadinessTests(unittest.TestCase):
    def item(self, **values):
        return {
            "os": "Windows",
            "status": "passed",
            "gpu_preflight_required": True,
            **values,
        }

    def test_ready_driver_does_not_claim_gpu_execution(self):
        item = self.item()
        enforce_windows_readiness(item, {"stdout": "[12:34:56] " + marker()})
        self.assertEqual(item["status"], "passed")
        self.assertEqual(safe_readiness(item)["compute_backend"], "not_probed")
        self.assertIn("depends on the selected evaluation", readiness_description(item))

    def test_missing_duplicate_malformed_and_quoted_records_are_errors(self):
        samples = (
            "older adapter without a preflight",
            marker() + "\n" + marker(),
            marker() + " extra-inventory",
            "agent said: " + marker(),
            "[skills-readiness] windows_amd_display=unknown compute_backend=not_probed",
        )
        for output in samples:
            with self.subTest(output=output):
                item = self.item()
                enforce_windows_readiness(item, {"stdout": output})
                self.assertEqual(item["status"], "error")
                self.assertNotIn("extra-inventory", str(item))

    def test_unhealthy_preflight_overrides_pass_but_preserves_prior_setup_errors(self):
        for state in ("missing", "unhealthy", "query_failed"):
            item = self.item()
            enforce_windows_readiness(item, {"stderr": marker(state)})
            self.assertEqual(item["status"], "error")
        item = self.item(
            status="error", error="The skills test dependencies could not be installed."
        )
        enforce_windows_readiness(item, {})
        self.assertEqual(
            item["error"], "The skills test dependencies could not be installed."
        )

    def test_ready_marker_does_not_override_failed_grading(self):
        item = self.item(
            status="failed", error="One or more behavioral expectations were not met."
        )
        enforce_windows_readiness(item, {"stdout": marker()})
        self.assertEqual(item["status"], "failed")

    def test_linux_and_mock_are_not_reclassified(self):
        for values in (
            {"os": "Linux"},
            {"status": "mock"},
            {"gpu_preflight_required": False},
        ):
            item = self.item(**values)
            status = item["status"]
            enforce_windows_readiness(item, {})
            self.assertEqual(item["status"], status)

    def test_readiness_revalidation_discards_actor_controlled_fields(self):
        item = {
            "readiness": {
                "windows_amd_display": "ready",
                "compute_backend": "not_probed",
                "machine_name": "private-fixture-name",
                "raw_inventory": "private",
            }
        }
        self.assertEqual(
            set(safe_readiness(item)), {"windows_amd_display", "compute_backend"}
        )
        self.assertNotIn("private", readiness_description(item))
        item["readiness"]["windows_amd_display"] = ["ready"]
        self.assertEqual(safe_readiness(item), {})

    def test_controller_requires_preflight_from_requested_windows_adapter(self):
        path = "L4-sys/skills/windows/sys_func-skills_behavioral"
        plan = {
            "sessions": [
                {
                    "name": "skills-local-ai-use-windows",
                    "tests": [
                        {
                            "path": path,
                            "variables": {
                                "SKILLS": "local-ai-use",
                                "SKILLS_OS": "Windows",
                                "SKILLS_WINDOWS_GPU_PREFLIGHT": "required",
                            },
                        }
                    ],
                }
            ]
        }
        for output, expected in (("old adapter", "error"), (marker(), "passed")):
            with self.subTest(output=output):
                live = {
                    "launcher": {
                        "sessions": [
                            {
                                "name": "skills-local-ai-use-windows",
                                "status": "completed",
                                "tests": [
                                    {"path": path, "status": "passed", "stdout": output}
                                ],
                            }
                        ]
                    }
                }
                result = build_results_manifest(plan, live, mode="live")
                item = result["items"][0]
                self.assertEqual(item["status"], expected)
                self.assertTrue(item["gpu_preflight_required"])
                self.assertEqual(item["readiness"]["compute_backend"], "not_probed")


if __name__ == "__main__":
    unittest.main()
