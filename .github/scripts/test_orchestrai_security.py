"""Regression tests for privileged CI boundaries and the offline preview."""

import base64
import contextlib
import copy
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from build_orchestrai_plan import _linux_builds
from orchestrai_logs import public_test_log, redact_private_values
from orchestrai_stdout import sanitize_stream

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/evals.yml").read_text()
SHA = "a" * 40


def inline_step(name):
    """Exercise the actual Python embedded in YAML without a YAML dependency."""
    section = WORKFLOW.split(f"      - name: {name}\n", 1)[1]
    match = re.search(r"        run: \|\n((?:          .*\n|\n)+)", section)
    if not match:
        raise AssertionError(f"Missing Python step: {name}")
    return textwrap.dedent(match[1])


def execute_step(name, env):
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "output"
        with (
            patch.dict(os.environ, {**env, "GITHUB_OUTPUT": str(output)}, clear=True),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exec(compile(inline_step(name), name, "exec"), {})  # noqa: S102 - execute the workflow fixture being tested
        return dict(line.split("=", 1) for line in output.read_text().splitlines())


class MatrixSecurityTests(unittest.TestCase):
    def select(self, entries, scoped=None, **env):
        plan = {
            "routing": True,
            "extended": False,
            "default": entries,
            "scoped": scoped or [],
            "skipped": [],
        }
        return execute_step(
            "Apply requested mode and emit plan",
            {
                "PLAN": json.dumps(plan),
                "MODE": "both",
                "EVENT": "pull_request",
                "ORCHESTRAI_OS": "both",
                **env,
            },
        )

    def test_normalizes_and_deduplicates_default_matrix(self):
        entry = {"skill": "local-ai-use", "os": "Linux", "runner": "attacker"}
        result = self.select([entry, entry, {"skill": "local-ai-use", "os": "Windows"}])
        self.assertEqual(
            json.loads(result["default"]),
            [
                {"skill": "local-ai-use", "os": "Linux"},
                {"skill": "local-ai-use", "os": "Windows"},
            ],
        )
        self.assertEqual(len(json.loads(result["orchestrai"])), 2)

    def test_rejects_unsupported_os_and_shell_syntax(self):
        for entry in [
            None,
            [],
            "bad",
            {"skill": "x", "os": "Darwin"},
            {"skill": 'x"; echo injected; #', "os": "Linux"},
            {"skill": "$(id)", "os": "Windows"},
            {"skill": "../../x", "os": "Linux"},
            {"skill": "x\n", "os": "Linux"},
            {"skill": 1, "os": "Linux"},
        ]:
            with self.subTest(entry=entry), self.assertRaises(SystemExit):
                self.select([entry])

    def test_os_filter_does_not_hide_invalid_entries(self):
        with self.assertRaises(SystemExit):
            self.select(
                [{"skill": "x", "os": "Darwin"}],
                EVENT="workflow_dispatch",
                ORCHESTRAI_OS="Linux",
            )
        result = self.select(
            [{"skill": "x", "os": "Linux"}, {"skill": "x", "os": "Windows"}],
            EVENT="workflow_dispatch",
            ORCHESTRAI_OS="Linux",
        )
        self.assertEqual(json.loads(result["default"]), [{"skill": "x", "os": "Linux"}])

    def test_scoped_runner_and_environment_are_not_candidate_controlled(self):
        runner = ["rocm", "self-hosted", "Linux", "X64", "mi300x", "gpu"]
        entry = {
            "skill": "serving-llms-on-instinct",
            "os": "Linux",
            "runner": json.dumps(runner),
            "environment": "unprotected",
        }
        result = json.loads(self.select([], [entry])["scoped"])[0]
        self.assertEqual(result["environment"], "behavioral-instinct")
        for bad in [
            '["internal-admin"]',
            "null",
            "{}",
            "invalid",
            None,
            json.dumps([{}, "Linux", "X64", "mi300x", "gpu", "rocm"]),
        ]:
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                self.select([], [{**entry, "runner": bad}])


class EnvironmentPolicyTests(unittest.TestCase):
    def settings(self):
        return {
            "protection_rules": [
                {
                    "type": "required_reviewers",
                    "prevent_self_review": True,
                    "reviewers": [{"type": "Team", "reviewer": {"id": 1}}],
                }
            ],
            "can_admins_bypass": False,
            "deployment_branch_policy": {
                "protected_branches": True,
                "custom_branch_policies": False,
            },
        }

    def run_policy(self, settings, **env):
        with patch(
            "urllib.request.urlopen",
            side_effect=lambda *a, **k: io.StringIO(json.dumps(settings)),
        ) as request:
            result = execute_step(
                "Check approval rules and select trusted controller",
                {
                    "BASE_SHA": SHA,
                    "GH_TOKEN": "fixture-not-a-token",
                    "GITHUB_API_URL": "https://api.example",
                    "GITHUB_REPOSITORY": "fixture/skills",
                    "NEED_MODEL_OR_PORTAL": "true",
                    "NEED_INSTINCT": "true",
                    **env,
                },
            )
        return result, request.call_count

    def test_accepts_protected_environments_and_immutable_controller(self):
        result, calls = self.run_policy(self.settings())
        self.assertEqual(result["sha"], SHA)
        self.assertEqual(calls, 2)
        result, _ = self.run_policy(self.settings(), CONTROLLER_OVERRIDE="B" * 40)
        self.assertEqual(result["sha"], "b" * 40)

    def test_blocks_missing_reviews_self_approval_bypass_and_branch_policy(self):
        cases = []
        for key, value in [
            ("protection_rules", []),
            ("can_admins_bypass", True),
            ("deployment_branch_policy", None),
        ]:
            cases.append({**self.settings(), key: value})
        for key, value in [("reviewers", []), ("prevent_self_review", False)]:
            settings = self.settings()
            settings["protection_rules"][0][key] = value
            cases.append(settings)
        cases.append({})
        for settings in cases:
            with self.subTest(settings=settings), self.assertRaises(SystemExit):
                self.run_policy(settings)

    def test_invalid_controller_ref_and_unreadable_settings_fail_closed(self):
        for ref in ["main", "HEAD", "a" * 39, SHA + "\n"]:
            with self.subTest(ref=ref), self.assertRaises(SystemExit):
                self.run_policy(self.settings(), CONTROLLER_OVERRIDE=ref)
        with patch(
            "urllib.request.urlopen", side_effect=OSError("private-network-detail")
        ):
            with self.assertRaisesRegex(
                SystemExit, "Cannot verify protected CI"
            ) as caught:
                execute_step(
                    "Check approval rules and select trusted controller",
                    {
                        "BASE_SHA": SHA,
                        "GH_TOKEN": "fixture",
                        "GITHUB_API_URL": "https://api.example",
                        "GITHUB_REPOSITORY": "fixture/skills",
                        "NEED_MODEL_OR_PORTAL": "true",
                        "NEED_INSTINCT": "false",
                    },
                )
            self.assertNotIn("private-network-detail", str(caught.exception))

    def test_workflow_keeps_privileges_and_candidate_code_separate(self):
        def job(name):
            return re.split(
                r"\n  [a-z][a-z-]*:\n",
                WORKFLOW.split(f"\n  {name}:\n", 1)[1],
                maxsplit=1,
            )[0]

        self.assertEqual(WORKFLOW.count("id-token: write"), 1)
        self.assertIn("id-token: write", job("behavior-scoped"))
        self.assertNotIn("PRIVILEGED_CI_ENABLED", WORKFLOW)
        policy = job("security-policy")
        self.assertIn("needs.discover.outputs.routing == 'true'", policy)
        self.assertIn("needs.discover.outputs.default_any == 'true'", policy)
        self.assertIn("needs.discover.outputs.scoped_any == 'true'", policy)
        self.assertIn(
            "github.event.pull_request.head.repo.full_name == github.repository",
            policy,
        )
        self.assertIn(
            "github.event.pull_request.base.ref == github.event.repository.default_branch",
            policy,
        )
        self.assertIn(
            "github.ref == format('refs/heads/{0}', github.event.repository.default_branch)",
            policy,
        )
        self.assertIn("no PR-code fallback", job("security-policy"))
        for name in ["routing", "orchestrai-behavioral", "behavior-scoped"]:
            section = job(name)
            self.assertIn("needs.security-policy.result == 'success'", section)
            self.assertIn(
                "github.event.pull_request.head.repo.full_name == github.repository",
                section,
            )
            self.assertIn(
                "ref: ${{ needs.security-policy.outputs.controller_sha }}", section
            )
            self.assertNotIn("environment: ${{", section)
        self.assertIn("environment: skills-ci", job("routing"))
        self.assertIn("environment: skills-ci", job("orchestrai-behavioral"))
        self.assertIn(
            "environment:\n      name: behavioral-instinct", job("behavior-scoped")
        )
        for name in ["routing", "behavior-scoped"]:
            self.assertIn("path: candidate", job(name))
            self.assertIn("python -I -m skillscope --repo candidate", job(name))
        verdict = job("orchestrai-verdict")
        self.assertIn("runs-on: ubuntu-latest", verdict)
        self.assertIn('--skill "$SKILL"', verdict)
        self.assertIn('--os "$SKILL_OS"', verdict)
        self.assertNotIn('--skill "${{', verdict)


class ResultsPolicyTests(unittest.TestCase):
    def run_results(self, **env):
        output = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {
                    "DISCOVER": "success",
                    "EVENT": "pull_request",
                    "HEAD_REPOSITORY": "fixture/skills",
                    "REPOSITORY": "fixture/skills",
                    "LABELS": "run_behavioral",
                    "ORCHESTRAI_MODE": "live",
                    "ORCHESTRAI_OS": "both",
                    **env,
                },
                clear=True,
            ),
            contextlib.redirect_stdout(output),
        ):
            exec(
                compile(
                    inline_step("Verify eval results"), "Verify eval results", "exec"
                ),
                {},
            )  # noqa: S102 - execute the workflow fixture being tested
        return output.getvalue()

    def test_successful_evals_need_no_enable_variable(self):
        output = self.run_results(
            SECURITY_POLICY="success",
            ROUTING_WANTED="true",
            ROUTING="success",
            BEHAVIOR_WANTED="true",
            ORCHESTRAI="success",
            ORCHESTRAI_VERDICTS="success",
            SCOPED_WANTED="true",
            SCOPED="success",
        )
        self.assertIn("All requested and authorized evals passed.", output)

    def test_requested_grading_still_requires_successful_policy(self):
        for requested in ["ROUTING_WANTED", "BEHAVIOR_WANTED", "SCOPED_WANTED"]:
            for policy in ["", "failure", "skipped", "cancelled"]:
                with (
                    self.subTest(requested=requested, policy=policy),
                    self.assertRaisesRegex(
                        SystemExit, "Privileged CI policy did not pass"
                    ),
                ):
                    self.run_results(**{requested: "true", "SECURITY_POLICY": policy})

    def test_references_only_needs_no_privileged_policy(self):
        output = self.run_results(SECURITY_POLICY="skipped")
        self.assertIn("All requested and authorized evals passed.", output)

    def test_successful_policy_does_not_hide_failed_or_skipped_verdicts(self):
        for verdict in ["failure", "skipped", "cancelled"]:
            with (
                self.subTest(verdict=verdict),
                self.assertRaisesRegex(SystemExit, "skill verdicts did not pass"),
            ):
                self.run_results(
                    SECURITY_POLICY="success",
                    BEHAVIOR_WANTED="true",
                    ORCHESTRAI="success",
                    ORCHESTRAI_VERDICTS=verdict,
                )


class ProvisioningSecurityTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / ".github/orchestrai-config.json").read_text())
        self.driver_env = {
            "ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON": '{"ubuntu:24.04":"https://drivers.example/fixture.deb"}'
        }

    def build(self, config):
        with patch.dict(os.environ, self.driver_env, clear=True):
            return _linux_builds(config, {"Linux"})

    def test_approved_installer_keeps_reboot(self):
        result = self.build(self.config)
        self.assertIn(
            {"script": "InstallationScripts/gfx/linux.sh", "reboot_after": True},
            result["install_scripts"],
        )

    def test_rejects_shell_traversal_and_empty_path_segments(self):
        for path in [
            "InstallationScripts/../bad.sh",
            "InstallationScripts/./bad.sh",
            "InstallationScripts//bad.sh",
            "InstallationScripts/gfx/linux.sh;id",
            "InstallationScripts/$(id).sh",
            "InstallationScripts/space name.sh",
            "InstallationScripts/gfx/linux.sh\n",
            "/InstallationScripts/gfx/linux.sh",
        ]:
            config = copy.deepcopy(self.config)
            config["provisioning"]["linux_install_scripts"].append(
                {"script": path, "reboot_after": True}
            )
            with self.subTest(path=path), self.assertRaises(SystemExit):
                self.build(config)

    def test_requires_installer_reboot_and_well_formed_config(self):
        for scripts in [
            [],
            [{"script": "InstallationScripts/gfx/linux.sh", "reboot_after": False}],
            [{"script": "InstallationScripts/other.sh", "reboot_after": True}],
            [{"script": 7, "reboot_after": True}],
        ]:
            config = copy.deepcopy(self.config)
            config["provisioning"]["linux_install_scripts"] = scripts
            with self.subTest(scripts=scripts), self.assertRaises(SystemExit):
                self.build(config)
        for value in [None, [], "invalid"]:
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.build({**self.config, "provisioning": value})


class RedactionSecurityTests(unittest.TestCase):
    def test_known_short_secret_variants_are_redacted_in_both_publishers(self):
        secret = "Plum!42"
        variants = [
            secret,
            secret.lower(),
            secret.upper(),
            "pLuM!42",
            "P%6Cum%2142",
            "%70%6c%75%6d%21%34%32",
            "%2550lum%252142",
            "%252550lum%25252142",
        ]
        for form in [secret, secret.lower(), secret.upper()]:
            for encoder in [base64.b64encode, base64.urlsafe_b64encode]:
                variants += [
                    encoder(form.encode()).decode(),
                    encoder(form.encode()).decode().rstrip("="),
                ]
        for variant in variants:
            with self.subTest(variant=variant):
                line = "[PASS] (files_exist) useful context " + variant
                results = [
                    redact_private_values(line, [secret]),
                    sanitize_stream(line, [secret]),
                    "\n".join(
                        public_test_log(
                            {"stdout": line}, skill="fixture", private_values=[secret]
                        )
                    ),
                ]
                for result in results:
                    self.assertNotIn(variant.lower(), result.lower())
                    self.assertIn("useful context", result)
                    self.assertIn("REDACTED", result)
                self.assertEqual(sanitize_stream(results[1], [secret]), results[1])

    def test_arbitrary_github_repositories_and_encoded_traversal_are_private(self):
        paths = [
            "https://github.com/private-org/private-repo/releases/tag/driver",
            "https://raw.githubusercontent.com/private-org/private-repo/main/driver.json",
            "https://objects.githubusercontent.com/private-driver-file",
            "https://github.com/amd/skills-private/file",
            "https://github.com/amd/skills/../../private/repo",
            "https://github.com/amd/skills/%252e%252e/private",
            "https://github.com:8443/amd/skills",
            "https://user:pass@github.com/amd/skills",
            "http://github.com/amd/skills",
        ]
        for url in paths:
            with self.subTest(url=url):
                self.assertEqual(sanitize_stream(url), "[PRIVATE URL REDACTED]")
        public = "https://github.com/amd/skillscope/tree/v0.1.3"
        self.assertEqual(sanitize_stream(public), public)
        self.assertEqual(
            sanitize_stream("RuntimeError: useful diagnostic"),
            "RuntimeError: useful diagnostic",
        )


class OfflinePreviewTests(unittest.TestCase):
    def test_preview_is_hosted_and_has_no_service_credentials(self):
        workflow = (
            ROOT / ".github/workflows/orchestrai-security-preview.yml"
        ).read_text()
        self.assertIn("runs-on: ubuntu-latest", workflow)
        self.assertIn("id-token: none", workflow)
        self.assertNotIn("secrets.", workflow)
        self.assertNotIn("environment:", workflow)
        self.assertNotIn("self-hosted", workflow)

    def test_mock_preview_exercises_both_operating_systems_without_hardware(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / ".github/scripts/orchestrai_security_preview.py"),
                    "--output",
                    temp,
                ],
                capture_output=True,
                text=True,
                check=False,
                env={
                    **os.environ,
                    "GITHUB_STEP_SUMMARY": "",
                    "ORCHESTRAI_PASSWORD": "must-not-be-inherited",
                },
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            files = [path for path in Path(temp).rglob("*") if path.is_file()]
            combined = "\n".join(path.read_text() for path in files)
            self.assertIn("no hardware", combined)
            self.assertIn("Linux", combined)
            self.assertIn("Windows", combined)
            self.assertNotIn("must-not-be-inherited", combined)
            self.assertNotIn("Plum!42", combined)
            self.assertFalse(any("plan" in path.name for path in files))


if __name__ == "__main__":
    unittest.main()
