"""Full-stream publication and its public-output security boundary."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from orchestrai_logs import private_log_values, public_manifest_log, public_test_log
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
        # Public-hosted private driver URLs and escaped known values retain the
        # same precedence when a logger serializes them as JSON strings.
        escaped = json.dumps(
            {
                "debug": "https://github.com/private-release/driver.zip",
                "value": "dummy\nmultiline-secret",
                "result": "pass",
            }
        ).replace("/", "\\/")
        result = sanitize_stream(escaped, values)
        self.assertNotIn("driver.zip", result)
        self.assertNotIn("multiline-secret", result)
        self.assertEqual(json.loads(result)["result"], "pass")
        self.assertEqual(sanitize_stream(result, values), result)
        self.assertEqual(
            manifest_streams({"public_streams": {"stdout": escaped}}, environ)[
                "stdout"
            ],
            result,
        )

    def test_any_scheme_urls_remove_userinfo_and_private_hosts(self):
        # Synthetic values, deliberately not real credentials or endpoints.
        for scheme in (
            "postgres",
            "postgresql",
            "mysql",
            "redis",
            "mongodb",
            "mongodb+srv",
            "amqp",
            "sftp",
            "vendor+driver-v2.1",
        ):
            for authority in (
                "dummy-user:dummy-url-secret@dbhost:3306",
                "dummy%2Duser:dummy%3Aurl%40secret@github.com",
                "dummy%3Aurl%40secret@localhost:8000",
                "dummy-user%3Adummy-url-secret%40github.com",
                "dummy-user@github.com",
                "dbhost:3306",
            ):
                with self.subTest(scheme=scheme, authority=authority):
                    raw = f"could not connect to {scheme}://{authority}/prod\nnext diagnostic\n"
                    expected = (
                        "could not connect to [PRIVATE URL REDACTED]\nnext diagnostic\n"
                    )
                    self.assertEqual(sanitize_stream(raw), expected)
                    self.assertEqual(sanitize_stream(expected), expected)
        self.assertNotIn(
            "dummy-url-secret",
            sanitize_stream("vendor://dummy:dummy-url-secret@[malformed-ipv6]/prod"),
        )
        escaped = '{"dsn":"postgres:\\/\\/dummy-user:dummy-url-secret@dbhost/prod", "result":"pass"}'
        result = sanitize_stream(escaped)
        self.assertNotIn("dummy-url-secret", result)
        self.assertEqual(json.loads(result)["result"], "pass")
        self.assertEqual(sanitize_stream(result), result)
        public = '{"url":"https:\\/\\/github.com\\/amd\\/skills"}'
        self.assertEqual(sanitize_stream(public), public)

    def test_known_json_encoded_values_stay_private_in_both_log_paths(self):
        samples = (
            ({}, {"LLM_GATEWAY_KEY": "dummy\nmultiline-secret"}, "dummy\nmultiline-secret", "multiline-secret"),
            (
                {"builds_json": {"vars": {"driver_source": r"\\fixture-host\Private Driver Folder\fixture-driver.zip"}}},
                {}, r"\\fixture-host\Private Driver Folder\fixture-driver.zip", "Driver Folder",
            ),
        )
        for plan, environ, private, suffix in samples:
            with self.subTest(kind="windows-source" if plan else "multiline-secret"):
                values = private_log_values(plan, {}, environ)
                self.assertIn(private, values)
                raw = (
                    "[PASS] (expected_behavior) retained diagnostic -- llm_judge: "
                    + json.dumps(private)
                )
                full = public_streams({"stdout": raw}, values)
                lines = public_test_log(
                    {"stdout": raw}, skill="fixture-skill", private_values=values
                )
                item = {"skill": "fixture-skill", "public_log": lines, "public_streams": full}
                summary = public_manifest_log(item)
                self.assertTrue(summary)
                for published in (*full.values(), *lines, *summary):
                    self.assertNotIn(suffix, published)
                    self.assertIn("retained diagnostic", published)
                    self.assertIn("[REDACTED]", published)
                self.assertEqual(public_manifest_log({**item, "public_log": summary}), summary)
                self.assertEqual(manifest_streams(item, environ), full)

    def test_hyphenated_headers_and_keys_redact_arbitrary_short_values(self):
        keys = (
            "X-Amz-Security-Token",
            "x-amz-security-token",
            "x-custom-api-key",
            "x-api-key",
            "db-api-key",
            "custom-api-key",
            "client-api-key",
            "service-password",
            "client-secret",
            "access-key-id",
            "private-key",
            "Set-Cookie",
            "Ocp-Apim-Subscription-Key",
        )
        for key in keys:
            for separator in (": ", "="):
                with self.subTest(key=key, separator=separator):
                    raw = f"{key}{separator}dummyshortlowercase\nnext diagnostic\n"
                    result = sanitize_stream(raw)
                    self.assertNotIn("dummyshortlowercase", result)
                    self.assertIn(key, result)
                    self.assertIn("next diagnostic", result)
                    self.assertEqual(result.count("\n"), raw.count("\n"))
                    self.assertEqual(sanitize_stream(result), result)

    def test_structured_values_preserve_safe_adjacent_fields(self):
        records = (
            {"user": "dummy-user", "result": "pass", "items": 5},
            {"host": "build-node-42", "result": "pass", "message": "download done"},
            {"api_key": "dummy-json-key", "result": "pass"},
            {"X-Amz-Security-Token": "dummy-json-token", "result": "pass"},
            {
                "credentials": {"login": "dummy-user", "value": "dummy-nested-secret"},
                "result": "pass",
            },
            {"device_tags": ["dummy-private-tag", "build-node-42"], "result": "pass"},
            {"session_id": 42, "result": "pass"},
            {"user": 'dummy user with "escaped quotes"', "result": "pass"},
            {"user": "dummy\\private\\user", "result": "pass"},
        )
        for record in records:
            with self.subTest(record=record):
                raw = json.dumps(record)
                result = sanitize_stream(raw)
                decoded = json.loads(result)
                self.assertEqual(decoded["result"], "pass")
                self.assertEqual(set(decoded), set(record))
                for key in set(record) - {"result", "items", "message"}:
                    self.assertIn("REDACTED", decoded[key])
                self.assertEqual(sanitize_stream(result), result)
        self.assertEqual(
            sanitize_stream("host=build-node-42 result=pass\n"),
            "host=[IDENTITY REDACTED] result=pass\n",
        )

    def test_multiline_values_and_folded_headers_preserve_line_boundaries(self):
        samples = (
            'api_key="dummy-first-line\ndummy-second-line"\nresult=pass\n',
            "X-Amz-Security-Token: dummy-first-line\n  dummy-second-line\nresult: pass\n",
            'X-Amz-Security-Token: "dummy-first-line"\n  dummy-second-line\nresult: pass\n',
            "client-secret: |\n  dummy-first-line\n  dummy-second-line\nresult: pass\n",
            "password: >-\r\n  dummy-first-line\r\n  dummy-second-line\r\nresult: pass\r\n",
            '{"api_key":\n  "dummy-first-line\\ndummy-second-line",\n  "result": "pass"}\n',
            '{"device_tags": [\n "dummy-first-line",\n "dummy-second-line"\n], "result": "pass"}\n',
        )
        for raw in samples:
            with self.subTest(raw=raw):
                result = sanitize_stream(raw)
                self.assertNotIn("dummy-first-line", result)
                self.assertNotIn("dummy-second-line", result)
                self.assertIn("pass", result)
                self.assertEqual(result.count("\n"), raw.count("\n"))
                self.assertEqual(sanitize_stream(result), result)
                if raw.startswith("{"):
                    self.assertEqual(json.loads(result)["result"], "pass")

    def test_unstructured_credential_punctuation_does_not_end_redaction(self):
        for raw in (
            "Cookie: session=dummy-first; other=dummy-second\nnext diagnostic\n",
            "custom-api-key: dummy-first,dummy-second\nnext diagnostic\n",
            'password="dummy-first" "dummy-second"\nnext diagnostic\n',
            "x-amz-security-token: dummy-first]dummy-second\nnext diagnostic\n",
            "Authorization: dummy-first; dummy-second\nnext diagnostic\n",
            'credentials={"value":"dummy-first"}; dummy-second\nnext diagnostic\n',
            '"X-Amz-Security-Token": dummy-first,dummy-second\nnext diagnostic\n',
            '"Cookie": session=dummy-first; csrf=dummy-second\nnext diagnostic\n',
            'Cookie": dummy-first; csrf=dummy-second\nnext diagnostic\n',
            '"custom-api-key": "dummy-first",dummy-second\nnext diagnostic\n',
            '{"custom-api-key": dummy-first,dummy-second}\nnext diagnostic\n',
        ):
            with self.subTest(raw=raw):
                result = sanitize_stream(raw)
                self.assertNotIn("dummy-first", result)
                self.assertNotIn("dummy-second", result)
                self.assertIn("next diagnostic", result)
                self.assertEqual(sanitize_stream(result), result)

    def test_deep_or_unterminated_credential_containers_fail_closed(self):
        for value in (
            "[" * 2000 + '"dummy-nested-secret"' + "]" * 2000,
            '{"value":"dummy-nested-secret"',
            '"dummy-nested-secret',
        ):
            with self.subTest(shape=value[:20]):
                raw = '{"credentials": ' + value + ', "result": "pass"}\n'
                result = sanitize_stream(raw)
                self.assertNotIn("dummy-nested-secret", result)
                self.assertEqual(sanitize_stream(result), result)

    def test_unknown_bare_machine_names_are_redacted(self):
        for machine in (
            "build-node-42",
            "runner-linux-a1",
            "rack-node-dt123",
            "Build-Node-42",
        ):
            with self.subTest(machine=machine):
                raw = f"connection from {machine} failed; retry pending\n"
                result = sanitize_stream(raw)
                self.assertNotIn(machine, result)
                self.assertIn("failed; retry pending", result)
                self.assertEqual(sanitize_stream(result), result)
        self.assertNotIn(
            "build-node-42", sanitize_stream("/tmp/private-run/build-node-42.log")
        )

    def test_lowercase_nonhex_tokens_do_not_need_a_large_alphabet(self):
        token = "abcdefghijklmnopqrstuvwxyzghijkl"
        digit_token = "abcdefghijklmno0123456789qrstuv"
        for raw in (
            f"unlabelled {token}\n",
            f"/tmp/private-run/run_{token}.json",
            token.upper(),
            digit_token,
            digit_token.upper(),
            f"/tmp/private-run/run_{digit_token}.json",
            "gh" * 16,
            "ghijk" * 7,
        ):
            with self.subTest(raw=raw):
                result = sanitize_stream(raw)
                self.assertNotIn(token, result.lower())
                self.assertNotIn(digit_token, result.lower())
                self.assertNotIn("gh" * 16, result)
                self.assertNotIn("ghijk" * 7, result)
                self.assertEqual(sanitize_stream(result), result)
        # Repetitive output and ordinary skill/file names are not random keys.
        raw = (
            "x" * 5000 + "\n"
            "tracelens-analysis-orchestrator.py\nsetup-skills.ps1\n"
            "https://files.pythonhosted.org/packages/onnx-runtime-1.2.3.whl\n"
            "https://github.com/amd/skills/blob/main/local-ai-use/SKILL.md\n"
        )
        self.assertEqual(sanitize_stream(raw), raw)

    def test_controller_and_verdict_boundaries_reject_combined_exploits(self):
        token = "abcdefghijklmnopqrstuvwxyzghijkl"
        raw = (
            "::error::could not connect to new+db://dummy:dummy-url-secret@dbhost/app\n"
            "x-amz-security-token: dummyshortlowercase\n"
            '{"user": "dummy-private-user", "result": "pass"}\n'
            f"failed on build-node-42 with unlabelled {token}\n"
            "RuntimeError: useful diagnostic retained\n"
        )
        controller = public_streams({"stdout": raw, "stderr": raw})
        verdict = manifest_streams({"public_streams": controller}, {})
        # Also revalidate a forged manifest containing the unsanitized stream.
        forged = manifest_streams({"public_streams": {"stdout": raw}}, {})
        for result in (*controller.values(), *verdict.values(), *forged.values()):
            for private in (
                "dummy-url-secret",
                "dummyshortlowercase",
                "dummy-private-user",
                "build-node-42",
                token,
                "::error::",
            ):
                self.assertNotIn(private, result)
            self.assertIn('"result": "pass"', result)
            self.assertIn("useful diagnostic retained", result)
            self.assertEqual(result.count("\n"), raw.count("\n"))
            self.assertEqual(sanitize_stream(result), result)
        self.assertEqual(verdict, controller)

    def test_only_exact_case_footers_exempt_credential_shaped_case_names(self):
        raw = "[PASS] local-mode-without-api-key: 2/2 checks in 1.0s\n"
        self.assertEqual(sanitize_stream(raw), raw)
        raw_failed = "[FAIL] custom-api-key: 1/2 checks in 1.0s\n"
        self.assertEqual(sanitize_stream(raw_failed), raw_failed)
        self.assertNotIn("local-mode", sanitize_stream(raw, ["local-mode"]))
        for raw in (
            "custom-api-key: shortsecret\n",
            "local-mode-without-api-key: shortsecret\n",
            "[PASS] custom-api-key: shortsecret\n",
            '{"custom-api-key": "shortsecret", "result": "pass"}\n',
        ):
            with self.subTest(raw=raw):
                self.assertNotIn("shortsecret", sanitize_stream(raw))

    def test_public_urls_versions_localhost_and_api_key_discussion_survive(self):
        raw = (
            "Python 3.12.13 Claude 2.1.278\n"
            "Install https://github.com/amd/skillscope https://deb.nodesource.com/setup_22.x\n"
            "Use http://localhost:8000/v1 http://127.0.0.1:8000/v1\n"
            "Check OPENAI_API_KEY is not required\n"
            "[PASS] local-mode-without-api-key: 2/2 checks in 1.0s\n"
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
                    "stdout": (
                        "new dependency warning\npassword=dummy-private-value\n::error::forged\n"
                        "vendor+db://dummy:dummy-url-secret@dbhost/prod\n"
                        "x-amz-security-token: dummyshortlowercase\n"
                        '{"user": "dummy-private-user", "result": "pass"}\n'
                        "failed on build-node-42 with abcdefghijklmnopqrstuvwxyzghijkl\n"
                    ),
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
                for private in (
                    "dummy-url-secret",
                    "dummyshortlowercase",
                    "dummy-private-user",
                    "build-node-42",
                    "abcdefghijklmnopqrstuvwxyzghijkl",
                ):
                    self.assertNotIn(private, published)
                self.assertIn('"result": "pass"', published)
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
