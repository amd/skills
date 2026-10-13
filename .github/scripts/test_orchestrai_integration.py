"""Regression tests for OAI integration, upstream policy parity and redaction."""

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

import orchestrai_preview as preview
from build_orchestrai_plan import _linux_builds
from orchestrai_logs import public_test_log, redact_private_values
from orchestrai_stdout import sanitize_stream

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / ".github/workflows/evals.yml").read_text()
SHA = "a" * 40
CONTROL_RUNNER = "runs-on: ${{ vars.ORCHESTRAI_CONTROL_RUNNER || 'ubuntu-latest' }}"


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

    def test_scoped_matrix_is_passed_through_without_oai_policy(self):
        # Skillscope owns labels/environment selection. Do not force one
        # particular hardware model or make scoped jobs depend on OAI.
        entries = [
            {
                "skill": "serving-llms-on-instinct",
                "os": "Linux",
                "runner": '["self-hosted","Linux","X64","mi300x","gpu","rocm"]',
                "environment": "behavioral-instinct",
            },
            {
                "skill": "fixture",
                "os": "Linux",
                "runner": '["self-hosted","Linux","X64","other-gpu"]',
                "environment": "behavioral-instinct",
            },
        ]
        self.assertEqual(json.loads(self.select([], entries)["scoped"]), entries)


def job(name):
    return re.split(
        r"\n  [a-z][a-z-]*:\n",
        WORKFLOW.split(f"\n  {name}:\n", 1)[1],
        maxsplit=1,
    )[0]


class UpstreamPolicyParityTests(unittest.TestCase):
    def test_trigger_and_concurrency_contract(self):
        self.assertIn("types: [opened, synchronize, reopened, labeled]", WORKFLOW)
        self.assertIn('cron: "0 12 * * 1"', WORKFLOW)
        self.assertIn("options: [both, routing, behavioral, references]", WORKFLOW)
        self.assertIn(
            "group: evals-${{ github.event.pull_request.number || github.ref }}\n",
            WORKFLOW,
        )
        self.assertIn("cancel-in-progress: true", WORKFLOW)
        self.assertNotIn("github.event.label.name", WORKFLOW)
        self.assertNotIn("run_behavioral", WORKFLOW)
        self.assertNotIn("pull_request_target:", WORKFLOW)
        self.assertNotIn("github.event.repository.default_branch", WORKFLOW)

    def test_routing_and_instinct_do_not_depend_on_oai(self):
        for name, selected in [
            ("routing", "routing"),
            ("behavior-scoped", "scoped_any"),
        ]:
            section = job(name)
            self.assertIn("needs: discover\n", section)
            self.assertIn(f"if: needs.discover.outputs.{selected} == 'true'\n", section)
            self.assertNotIn("security-policy", section)
            self.assertNotIn("orchestrai-source", section)
            self.assertNotIn("head.repo.full_name", section)
            self.assertNotIn("controller_sha", section)
        scoped = job("behavior-scoped")
        self.assertIn("runs-on: ${{ fromJSON(matrix.runner) }}", scoped)
        self.assertIn("name: ${{ matrix.environment }}", scoped)
        self.assertIn("ENVIRONMENT: ${{ matrix.environment }}", scoped)
        self.assertIn("id-token: write", scoped)
        self.assertIn("FEDERATION_RULE_ID:", scoped)
        self.assertNotIn("ORCHESTRAI_", scoped)
        self.assertNotIn("environment:", job("routing"))
        self.assertIn("secrets.ORCHESTR_API_KEY", job("routing"))

    def test_new_authorization_policy_is_not_part_of_integration(self):
        for value in [
            "security-policy",
            "ORCHESTRAI_TRUSTED_CONTROLLER_SHA",
            "PRIVILEGED_CI_ENABLED",
            "environment: skills-ci",
            "protection_rules",
            "can_admins_bypass",
        ]:
            self.assertNotIn(value, WORKFLOW)
        # New OAI transport does not need an OIDC token. Instinct still does.
        self.assertEqual(WORKFLOW.count("id-token: write"), 1)
        self.assertIn("id-token: none", job("orchestrai-behavioral"))
        self.assertIn("needs: discover", job("orchestrai-behavioral"))
        self.assertIn(
            "if: needs.discover.outputs.default_any == 'true'",
            job("orchestrai-behavioral"),
        )
        self.assertNotIn("environment:", job("orchestrai-behavioral"))

    def test_public_candidate_is_tested_directly_without_a_mirror(self):
        for token in (
            "orchestrai-source",
            "verify_orchestrai_source",
            "orchestrai-public-source",
        ):
            self.assertNotIn(token, WORKFLOW)
        for path in (
            ".github/orchestrai-public-source.json",
            ".github/scripts/verify_orchestrai_source.py",
            ".github/scripts/test_orchestrai_source.py",
        ):
            self.assertFalse((ROOT / path).exists())
        candidate = "${{ github.event.pull_request.head.sha || github.sha }}"
        self.assertIn(f"ref: {candidate}", job("discover"))
        self.assertIn(f"ref: {candidate}", job("orchestrai-behavioral"))
        self.assertIn(f"TARGET_SHA: {candidate}", job("orchestrai-behavioral"))
        self.assertIn("TARGET_REPOSITORY: ${{ github.repository }}", WORKFLOW)
        self.assertIn(
            "TARGET_REF: ${{ github.event.pull_request.head.ref || github.ref_name }}",
            WORKFLOW,
        )
        self.assertIn('--repository "$TARGET_REPOSITORY"', WORKFLOW)
        self.assertIn('--sha "$TARGET_SHA"', WORKFLOW)
        for name in ("orchestrai-behavioral", "orchestrai-verdict"):
            self.assertIn(f"PUBLIC_SKILLS_COMMIT: {candidate}", job(name))
        self.assertIn('--skill "$SKILL"', job("orchestrai-verdict"))
        self.assertIn('--os "$SKILL_OS"', job("orchestrai-verdict"))
        self.assertNotIn("git ls-remote", WORKFLOW)

    def test_selection_keeps_pr_gates_but_explicit_requests_ignore_them(self):
        common = {
            "SKILLS": "",
            "LABELS": "",
            "BASE": "a" * 40,
            "HEAD": "b" * 40,
            "ROUTING_ROOM_VALUE": "local-ai-use",
        }
        pr_args = execute_step(
            "Work out how to select", {**common, "EVENT": "pull_request"}
        )["args"]
        self.assertIn("--since", pr_args)
        self.assertIn("--labels", pr_args)
        self.assertNotIn("--ignore-gates", pr_args)
        self.assertIn("--scoped-gate enable_mi_ci", pr_args)
        self.assertIn("--scoped-environment behavioral-instinct", pr_args)
        for event in ("workflow_dispatch", "schedule"):
            args = execute_step(
                "Work out how to select",
                {**common, "EVENT": event, "SKILLS": "local-ai-use"},
            )["args"]
            self.assertIn("--names local-ai-use", args)
            self.assertIn("--ignore-gates", args)

    def test_references_mode_selects_no_hardware(self):
        result = MatrixSecurityTests().select(
            [{"skill": "local-ai-use", "os": "Linux"}],
            [{"skill": "serving-llms-on-instinct"}],
            MODE="references",
        )
        self.assertEqual(result["routing"], "false")
        for name in ("default", "scoped", "orchestrai"):
            self.assertEqual(json.loads(result[name]), [])
        self.assertIn("github.event_name == 'schedule' && 'references'", WORKFLOW)

    def test_external_runner_settings_are_preserved(self):
        for name in (
            "discover",
            "external-references",
            "results",
            "orchestrai-verdict",
        ):
            self.assertIn("runs-on: ubuntu-latest", job(name))
        for name in ("routing", "orchestrai-behavioral"):
            self.assertIn(CONTROL_RUNNER, job(name))


class ResultsParityTests(unittest.TestCase):
    def run_results(self, **env):
        output = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {
                    "DISCOVER": "success",
                    "ORCHESTRAI_MODE": "live",
                    "ORCHESTRAI_OS": "both",
                    **env,
                },
                clear=True,
            ),
            contextlib.redirect_stdout(output),
        ):
            exec(  # noqa: S102 - exercise the actual workflow result gate under fixture inputs
                compile(
                    inline_step("Verify eval results"), "Verify eval results", "exec"
                ),
                {},
            )
        return output.getvalue()

    def test_successful_evals_need_no_new_label_policy_or_approval(self):
        output = self.run_results(
            ROUTING_WANTED="true",
            ROUTING="success",
            BEHAVIOR_WANTED="true",
            ORCHESTRAI="success",
            ORCHESTRAI_VERDICTS="success",
            SCOPED_WANTED="true",
            SCOPED="success",
        )
        self.assertIn("All requested evals passed.", output)

    def test_scoped_only_does_not_need_oai_or_source_check(self):
        output = self.run_results(
            SCOPED_WANTED="true",
            SCOPED="success",
            ORCHESTRAI="skipped",
            ORCHESTRAI_VERDICTS="skipped",
        )
        self.assertIn("All requested evals passed.", output)

    def test_missing_instinct_label_warns_without_failing(self):
        output = self.run_results(
            SCOPED_WANTED="false",
            SCOPED="skipped",
            SKIPPED='[{"skill":"serving-llms-on-instinct","reason":"enable_mi_ci"}]',
        )
        self.assertIn("::warning::Scoped behavioral cases were held back", output)
        self.assertIn("All requested evals passed.", output)

    def test_each_requested_leg_must_pass(self):
        success = {
            "ROUTING_WANTED": "true",
            "ROUTING": "success",
            "BEHAVIOR_WANTED": "true",
            "ORCHESTRAI": "success",
            "ORCHESTRAI_VERDICTS": "success",
            "SCOPED_WANTED": "true",
            "SCOPED": "success",
        }
        for key in ("ROUTING", "ORCHESTRAI", "ORCHESTRAI_VERDICTS", "SCOPED"):
            for state in ("failure", "skipped", "cancelled"):
                with self.subTest(key=key, state=state), self.assertRaises(SystemExit):
                    self.run_results(**{**success, key: state})

    def test_references_only_is_advisory_and_needs_no_graded_jobs(self):
        self.assertIn("All requested evals passed.", self.run_results())
        self.assertNotIn("external-references", job("results").split("steps:", 1)[0])

    def test_mock_or_one_os_cannot_claim_complete_hardware_coverage(self):
        for extra in [{"ORCHESTRAI_MODE": "mock"}, {"ORCHESTRAI_OS": "Linux"}]:
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.run_results(
                    BEHAVIOR_WANTED="true",
                    ORCHESTRAI="success",
                    ORCHESTRAI_VERDICTS="success",
                    **extra,
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
    def test_child_environment_preserves_runtime_but_not_credentials(self):
        private = {
            name: "must-not-be-inherited"
            for name in (
                "GH_TOKEN",
                "ORCHESTRAI_PASSWORD",
                "ORCHESTRAI_PORTAL_URL",
                "LLM_GATEWAY_KEY",
                "ORCHESTR_API_KEY",
                "HOME",
                "PYTHONPATH",
                "PYTHONHOME",
                "LD_PRELOAD",
            )
        }
        with patch.dict(
            os.environ,
            {
                **private,
                "PATH": "/runtime/bin",
                "LD_LIBRARY_PATH": "/runtime/lib",
                "ORCHESTRAI_WINDOWS_DRIVER_SOURCE": "private-driver",
                "ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON": "private-driver",
            },
            clear=True,
        ):
            env = preview.child_environment()
        self.assertEqual(env["LD_LIBRARY_PATH"], "/runtime/lib")
        self.assertEqual(env["PATH"], "/runtime/bin")
        self.assertEqual(env["MODE"], "mock")
        self.assertFalse(set(private) & env.keys())
        self.assertNotIn("private-driver", str(env))
        with patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("LD_LIBRARY_PATH", preview.child_environment())

    def test_all_preview_children_receive_the_runtime_environment(self):
        original_run = subprocess.run
        calls = []

        def run(command, **kwargs):
            calls.append(Path(command[1]).name)
            self.assertEqual(kwargs["env"]["LD_LIBRARY_PATH"], library_path)
            self.assertNotIn("ORCHESTRAI_PASSWORD", kwargs["env"])
            return original_run(command, **kwargs)

        library_path = os.environ.get("LD_LIBRARY_PATH") or "/synthetic/runtime/lib"
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(
                os.environ,
                {
                    "LD_LIBRARY_PATH": library_path,
                    "ORCHESTRAI_PASSWORD": "must-not-be-inherited",
                    "GITHUB_STEP_SUMMARY": "",
                },
            ),
            patch.object(sys, "argv", ["preview", "--output", temp]),
            patch.object(preview.subprocess, "run", side_effect=run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(preview.main(), 0)
        self.assertEqual(
            calls,
            [
                "build_orchestrai_plan.py",
                "orchestrai_run.py",
                "orchestrai_verdict.py",
                "build_orchestrai_plan.py",
                "orchestrai_run.py",
                "orchestrai_verdict.py",
                "orchestrai_report.py",
            ],
        )

    def test_loader_failure_diagnostic_does_not_print_raw_stderr(self):
        stderr = "private-host: error while loading shared libraries: libpython3.12.so: private-path"
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(sys, "argv", ["preview", "--output", temp]),
            patch.object(
                preview.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    [],
                    127,
                    stdout="private-output",
                    stderr=stderr,
                ),
            ),
            self.assertRaises(SystemExit) as raised,
        ):
            preview.main()
        self.assertIn("check LD_LIBRARY_PATH", str(raised.exception))
        self.assertNotIn("private-", str(raised.exception))

    def test_preview_is_hosted_and_has_no_service_credentials(
        self,
    ):
        workflow = (ROOT / ".github/workflows/orchestrai-preview.yml").read_text()
        self.assertIn("runs-on: ubuntu-latest", workflow)
        self.assertNotIn("orchestrai-source", workflow)
        self.assertNotIn("verify_orchestrai_source", workflow)
        self.assertIn("id-token: none", workflow)
        self.assertNotIn("secrets.", workflow)
        self.assertNotIn("environment:", workflow)

    def test_mock_preview_exercises_both_operating_systems_without_hardware(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / ".github/scripts/orchestrai_preview.py"),
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
