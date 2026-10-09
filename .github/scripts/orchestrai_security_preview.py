"""Exercise mock transport/reporting with synthetic inputs, never lab credentials."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from orchestrai_stdout import sanitize_stream

SCRIPTS = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Do not inherit any service inputs even if someone runs this locally from
    # a credentialed shell. The preview needs only Python and a minimal PATH.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONUTF8": "1",
        "MODE": "mock",
        "GITHUB_OUTPUT": "",
        "GITHUB_STEP_SUMMARY": "",
        "ORCHESTRAI_WINDOWS_DRIVER_SOURCE": r"C:\Fixture\DriverSource\ATI",
        "ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON": '{"ubuntu:24.04":"https://drivers.example/fixture.deb"}',
    }
    matrix = [{"skill": "local-ai-use", "os": name} for name in ("Linux", "Windows")]

    def run(script: str, *arguments: str, extra: dict | None = None) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPTS / script), *arguments],
            env={**env, **(extra or {})},
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            raise SystemExit(
                f"Synthetic preview failed in {script} (exit {completed.returncode})."
            )

    with tempfile.TemporaryDirectory(prefix="skills-security-preview-") as temp:
        for entry in matrix:
            os_name = entry["os"]
            plan = Path(temp) / f"plan-{os_name}.json"
            manifest = Path(temp) / f"results-{os_name}.json"
            run(
                "build_orchestrai_plan.py",
                "--matrix-json",
                json.dumps([entry]),
                "--device-tags-json",
                '{"strix_halo":["fixture-tag"]}',
                "--repository",
                "amd/skills",
                "--ref",
                "fixture-only",
                "--sha",
                "a" * 40,
                "--skillscope-ref",
                "v0.1.3",
                "--extended-flag=--no-extended",
                "--run-id",
                "fixture",
                "--output",
                str(plan),
            )
            run(
                "orchestrai_run.py",
                extra={
                    "PLAN_FILE": str(plan),
                    "RESULTS_FILE": str(manifest),
                    "LIVE_FILE": str(Path(temp) / f"live-{os_name}.json"),
                },
            )
            run(
                "orchestrai_verdict.py",
                "--results",
                str(manifest),
                "--skill",
                entry["skill"],
                "--os",
                os_name,
                "--output-dir",
                str(output / f"test-results-local-ai-use-{os_name}"),
            )

    # Deliberately short fake value: the long-token heuristic alone cannot mask it.
    fake = "Plum!42"
    variants = [fake.lower(), "P%6Cum%2142", base64.b64encode(fake.encode()).decode()]
    sample = "\n".join("Synthetic diagnostic: " + value for value in variants)
    sample += "\nRuntimeError: useful debugging context stays visible.\n"
    sanitized = sanitize_stream(sample, [fake])
    if any(value in sanitized for value in variants):
        raise SystemExit("Synthetic encoded-secret redaction did not pass.")
    (output / "redacted-fixture.log").write_text(sanitized, encoding="utf-8")
    run(
        "orchestrai_report.py",
        "--artifacts",
        str(output),
        "--expected-json",
        json.dumps(matrix),
        "--output",
        str(output / "orchestrai-report.md"),
        extra={
            "ORCHESTRAI_CONTROLLER_RESULT": "success",
            "ORCHESTRAI_VERDICT_RESULT": "success",
        },
    )
    overview = (
        "## OrchestrAI security preview — no hardware\n\n"
        "This is a synthetic test, not a skill-quality or GPU verdict. No Portal "
        "credentials, acquired machines, model calls, or private source checkout were used.\n\n"
        "- Linux and Windows plan-only controller/verdict/report flow: passed.\n"
        "- Encoded synthetic secret redaction: passed.\n"
        "- This preview neither enables nor verifies live privileged CI access.\n\n"
    )
    report = (output / "orchestrai-report.md").read_text(encoding="utf-8")
    (output / "README.md").write_text(overview, encoding="utf-8")
    print(overview + report)
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write(overview + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
