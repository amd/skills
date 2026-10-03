"""Full-stream publication and its public-output security boundary."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrai_logs import private_log_values
from orchestrai_stdout import (
    MAX_STREAM_BYTES,
    UNAVAILABLE,
    manifest_streams,
    public_commits,
    public_streams,
    sanitize_stream,
    stream_coverage,
)


def pem_marker(kind, *, end=False):
    # Build synthetic test markers at runtime, without embedding a PEM block
    # that repository push protection would mistake for real key material.
    return "-----" + ("END" if end else "BEGIN") + " " + kind + " KEY-----"


class FullStreamTests(unittest.TestCase):
    def test_workflow_passes_verified_revisions_to_both_redaction_boundaries(self):
        workflow = (Path(__file__).parents[1] / "workflows" / "evals.yml").read_text()
        self.assertIn("git -C .skillscope-action rev-parse HEAD", workflow)
        self.assertIn(
            "public_skillscope_commit: ${{ steps.skillscope_commit.outputs.sha }}",
            workflow,
        )
        self.assertEqual(
            workflow.count(
                "PUBLIC_SKILLSCOPE_COMMIT: ${{ needs.discover.outputs.public_skillscope_commit }}"
            ),
            2,
        )
        self.assertEqual(
            workflow.count(
                "PUBLIC_SKILLS_COMMIT: ${{ github.event.pull_request.head.sha || github.sha }}"
            ),
            2,
        )

    def test_paths_keep_filenames_not_private_roots_or_hierarchy(self):
        samples = {
            r"C:\Users\private-login\work\setup-skills.ps1": "<local>/setup-skills.ps1",
            r"C:\Program Files\Python312\python.exe": "<local>/python.exe",
            r"C:\Users\Private Login\work\run.ps1": "<local>/run.ps1",
            '"C:\\Users\\Private Login\\private directory\\run.ps1"': '"<local>/run.ps1"',
            "/home/private-login/private-project/setup-skills.sh": "<local>/setup-skills.sh",
            '"/tmp/private directory/failure.py"': '"<local>/failure.py"',
            "/tmp/private-run/failure.py:23)": "<local>/failure.py:23)",
            "/opt/private-cache/venv": "<local>/venv",
            "/var/private-pipeline/private-directory": "<local>",
            r"\\private-host\private-share\private-directory\stderr.log": "<local>/stderr.log",
            "/tmp/private-run/tmp.py": "<local>/tmp.py",
        }
        for raw, expected in samples.items():
            with self.subTest(raw=raw):
                sanitized = sanitize_stream(raw)
                self.assertEqual(sanitized, expected)
                self.assertEqual(sanitize_stream(sanitized), sanitized)

    def test_public_test_paths_are_not_credential_shaped(self):
        path = "L4-sys/skills/windows/sys_func-skills_behavioral"
        self.assertEqual(
            sanitize_stream(f"Path: {path}\nTest: {path}\n"),
            f"Path: {path}\nTest: {path}\n",
        )
        self.assertEqual(
            sanitize_stream("/tmp/private-checkout/testcases/" + path),
            "<local>/testcases/" + path,
        )
        self.assertEqual(
            sanitize_stream("https://github.com/amd/skills/blob/main/home/example.py"),
            "https://github.com/amd/skills/blob/main/home/example.py",
        )
        self.assertNotIn("aA1" * 12, sanitize_stream("aA1" * 12 + "/aA1" * 12))

    def test_only_verified_public_commits_are_kept(self):
        skills_sha = "1a" * 20
        harness_sha = "2b" * 20
        unknown_key = "3c" * 20
        environ = {
            "PUBLIC_SKILLS_COMMIT": skills_sha,
            "PUBLIC_SKILLSCOPE_COMMIT": harness_sha,
        }
        known = public_commits(environ)
        raw = f"verifying {skills_sha}\nNote: switching to {harness_sha}\ncommit: {unknown_key}\n"
        sanitized = sanitize_stream(raw, verified_commits=known)
        self.assertIn(skills_sha, sanitized)
        self.assertIn(harness_sha, sanitized)
        self.assertNotIn(unknown_key, sanitized)
        item = {
            "public_streams": {"stdout": sanitized},
            "verified_commits": [unknown_key],
        }
        self.assertEqual(manifest_streams(item, environ)["stdout"], sanitized)
        self.assertNotIn(skills_sha, manifest_streams(item, {})["stdout"])
        self.assertNotIn(
            unknown_key,
            sanitize_stream(f"api_key={unknown_key}", verified_commits={unknown_key}),
        )
        self.assertNotIn(
            skills_sha,
            sanitize_stream(skills_sha, [skills_sha], verified_commits=known),
        )
        self.assertEqual(
            public_commits({"PUBLIC_SKILLS_COMMIT": "not-a-commit"}), set()
        )

    def test_credentials_and_identities_in_filenames_remain_redacted(self):
        key = "0123456789abcdef" * 2
        self.assertNotIn(key, sanitize_stream(f"/tmp/private-run/{key}.json"))
        self.assertNotIn(key, sanitize_stream(f"/tmp/private-run/run_{key}.json"))
        self.assertNotIn(
            "11111111-2222",
            sanitize_stream(
                "/tmp/private-run/run_11111111-2222-4333-8444-555555555555.log"
            ),
        )
        self.assertNotIn(
            "private-login",
            sanitize_stream("/tmp/private-run/private-login.py", ["private-login"]),
        )
        self.assertNotIn(
            "LAB-UCICD-DT123", sanitize_stream("/tmp/private-run/LAB-UCICD-DT123.log")
        )
        self.assertEqual(
            sanitize_stream("/root/private-project/failure.py", ["root"]),
            "<local>/failure.py",
        )
        self.assertEqual(
            sanitize_stream(
                "/home/private-login/private-project/failure.py",
                ["/home/private-login"],
            ),
            "<local>/failure.py",
        )
        self.assertEqual(
            sanitize_stream(
                r"C:\Users\private-login\private-project\failure.py",
                [r"C:\Users\private-login"],
            ),
            "<local>/failure.py",
        )

    def test_unrecognized_lines_long_lines_and_entire_stream_are_retained(self):
        raw = "dependency setup warning\n" * 1000
        raw += "x" * 5000 + "\nRuntimeError: newly introduced harness error\n"
        raw += "[behavioral] tracelens-analysis-orchestrator\n"
        self.assertEqual(sanitize_stream(raw), raw)
        streams = public_streams({"stdout": raw, "stderr": "arbitrary stderr\n"})
        self.assertEqual(streams["stdout"], raw)
        self.assertEqual(streams["stderr"], "arbitrary stderr\n")

    def test_private_data_is_redacted_but_context_survives(self):
        raw = (
            "Hostname : LAB-UCICD-DT123\nExecuting user: private-login\n"
            "workspace /home/private-login/work/test.py C:\\Users\\private-login\\test.py\n"
            "endpoint https://private-service.amd.com/tenant?token=dummy\n"
            "private-service.amd.com 10.23.45.67 2001:db8::1234\n"
            "Authorization: Bearer dummy-short-key\n"
            "LLM_GATEWAY_KEY=dummy-short-key\n"
            "AWS_SECRET_ACCESS_KEY=dummy-aws-key\nCookie: dummy-cookie\n"
            "command --password dummy-cli-password\n"
            '{"api_key": "dummy-json-key"}\n'
            "unlabelled 0123456789abcdef0123456789abcdef\n"
            "opaque run 11111111-2222-4333-8444-555555555555\n"
            "known value dummy-secret-value\n"
            f"{pem_marker('PRIVATE')}\nprivate-key-body\n{pem_marker('PRIVATE', end=True)}\n"
            "RuntimeError: connection failed during model download\n"
        )
        result = sanitize_stream(raw, ["dummy-secret-value"])
        for private in (
            "LAB-UCICD-DT123",
            "private-login",
            "private-service",
            "10.23.45.67",
            "2001:db8",
            "dummy-short-key",
            "dummy-aws-key",
            "dummy-cookie",
            "dummy-cli-password",
            "dummy-json-key",
            "0123456789abcdef",
            "11111111-2222",
            "dummy-secret-value",
            "private-key-body",
        ):
            self.assertNotIn(private, result)
        self.assertIn("connection failed during model download", result)
        self.assertIn("workspace", result)
        self.assertEqual(sanitize_stream(result), result)

    def test_known_multiline_secret_and_driver_map_values(self):
        environ = {
            "ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON": json.dumps(
                {"noble": {"url": "https://github.com/private-release/driver.zip"}}
            ),
            "LLM_GATEWAY_KEY": "dummy\nmultiline-secret",
            "ORCHESTRAI_MODE": "live",
            "ORCHESTRAI_OS": "Linux",
        }
        values = private_log_values({}, {}, environ)
        raw = "Linux live https://github.com/private-release/driver.zip dummy\nmultiline-secret\n"
        result = sanitize_stream(raw, values)
        self.assertIn("Linux live", result)
        self.assertNotIn("driver.zip", result)
        self.assertNotIn("multiline-secret", result)
        # The controller receives driver variables via the private plan, not
        # necessarily through its own step environment.
        plan = {
            "builds_json": {
                "vars": {
                    "driver_sources_json": environ[
                        "ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON"
                    ]
                }
            }
        }
        self.assertNotIn(
            "driver.zip", sanitize_stream(raw, private_log_values(plan, {}, {}))
        )

    def test_public_urls_versions_localhost_and_api_key_discussion_survive(self):
        raw = (
            "Python 3.12.13 Claude 2.1.278\n"
            "Install https://github.com/amd/skillscope https://deb.nodesource.com/setup_22.x\n"
            "Use http://localhost:8000/v1 http://127.0.0.1:8000/v1\n"
            "Check OPENAI_API_KEY is not required; local-mode-without-api-key: 2/2 checks\n"
            "Unexpected RuntimeError from a future Skillscope version\n"
        )
        self.assertEqual(sanitize_stream(raw), raw)
        self.assertNotIn(
            "signed-secret",
            sanitize_stream("https://github.com/amd/skills?token=signed-secret"),
        )

    def test_commands_ansi_bidi_and_unterminated_private_keys(self):
        raw = "\x1b[31m::error::forged\x1b[0m\n\u202esecret\n"
        result = sanitize_stream(raw)
        self.assertNotIn("::error::", result)
        self.assertNotIn("\x1b", result)
        self.assertNotIn("\u202e", result)
        self.assertNotIn(
            "private material",
            sanitize_stream(pem_marker("RSA PRIVATE") + "\nprivate material"),
        )

    def test_transport_limit_is_explicit_not_a_tail(self):
        self.assertEqual(
            sanitize_stream("x" * (MAX_STREAM_BYTES + 1)), UNAVAILABLE + "\n"
        )
        self.assertIn("Incomplete", stream_coverage({"stdout": UNAVAILABLE}))
        self.assertIn("other stream unavailable", stream_coverage({"stdout": ""}))
        self.assertIn("stdout + stderr", stream_coverage({"stdout": "", "stderr": ""}))
        self.assertEqual(stream_coverage({}), "Test streams unavailable")

    def test_verdict_revalidates_manifest_and_publishes_full_log_without_rp(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            item = {
                "skill": "local-ai-use",
                "os": "Linux",
                "status": "failed",
                "error": "The behavioral test did not pass.",
                "report_url": "https://private-reports.amd.com/launch/secret-id",
                "public_streams": {
                    "stdout": "new dependency warning\npassword=dummy-private-value\n::error::forged\n",
                    "stderr": "RuntimeError: formerly invisible failure\n",
                },
            }
            manifest = root / "results.json"
            manifest.write_text(json.dumps({"items": [item], "run": {}}))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("orchestrai_verdict.py")),
                    "--results",
                    str(manifest),
                    "--skill",
                    "local-ai-use",
                    "--os",
                    "Linux",
                    "--output-dir",
                    str(root / "output"),
                ],
                env={**os.environ, "GITHUB_STEP_SUMMARY": str(root / "summary.md")},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 1, completed.stderr)
            log = (root / "output" / "stdout-stderr.log").read_text()
            for published in (log, completed.stdout):
                self.assertIn("new dependency warning", published)
                self.assertIn("formerly invisible failure", published)
                self.assertNotIn("dummy-private-value", published)
                self.assertNotIn("::error::forged", published)
                self.assertNotIn("private-reports", published)
            summary = (root / "summary.md").read_text()
            self.assertIn("stdout-stderr.log", summary)
            self.assertNotIn("private-reports", summary)
            self.assertNotIn("formerly invisible failure", summary)
            self.assertNotIn(
                "report_url", json.loads((root / "output" / "summary.json").read_text())
            )

    def test_manifest_cannot_add_other_streams_or_bypass_secret_redaction(self):
        item = {
            "public_streams": {
                "stdout": "known dummy-bound-secret",
                "stderr": "",
                "console": "private console",
            }
        }
        with mock.patch.dict(os.environ, {"LLM_GATEWAY_KEY": "dummy-bound-secret"}):
            result = manifest_streams(item, dict(os.environ))
        self.assertEqual(set(result), {"stdout", "stderr"})
        self.assertNotIn("dummy-bound-secret", result["stdout"])


if __name__ == "__main__":
    unittest.main()
