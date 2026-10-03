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
    public_streams,
    sanitize_stream,
    stream_coverage,
)


def pem_marker(kind, *, end=False):
    # Build synthetic test markers at runtime, without embedding a PEM block
    # that repository push protection would mistake for real key material.
    return "-----" + ("END" if end else "BEGIN") + " " + kind + " KEY-----"


class FullStreamTests(unittest.TestCase):
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
