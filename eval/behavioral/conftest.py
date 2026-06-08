"""pytest wiring for the behavioral harness.

- Adds this directory to ``sys.path`` so tests can ``from harness import ...``.
- Skips the whole module unless the prerequisites (the ``claude`` CLI and a
  reachable Lemonade Server) are present -- these are real, local-only tests.
- Resets the per-test run registry at setup, and at teardown evaluates the
  recorded expectations, writes a results JSON, cleans up temp workspaces, and
  fails the test if any item failed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import harness  # noqa: E402

RESULTS_DIR = THIS_DIR / "results"


@pytest.fixture(scope="session", autouse=True)
def _require_prereqs() -> None:
    """Skip the behavioral suite unless claude + Lemonade are available locally."""
    if not harness.claude_available():
        pytest.skip("'claude' CLI not found on PATH; behavioral tests are local-only.",
                    allow_module_level=False)
    if not harness.lemonade_server_reachable():
        pytest.skip(
            f"Lemonade Server not reachable at "
            f"http://{harness.LEMONADE_HOST}:{harness.LEMONADE_PORT}; "
            "start it before running behavioral tests.",
            allow_module_level=False,
        )


@pytest.fixture(autouse=True)
def _behavioral_runs(request: pytest.FixtureRequest):
    """Reset the run registry, then finalize (report + write + assert) at teardown."""
    harness.reset_runs()
    yield
    runs = harness.drain_runs()
    if not runs:
        return

    test_name = request.node.name
    report = {
        "test": test_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "runs": [],
    }
    failures: list[str] = []

    try:
        for run in runs:
            results = [asdict(r) for r in run.results]
            report["runs"].append({
                "prompt": run.prompt_text,
                "skill": run.skill,
                "model": run.model,
                "effort": run.effort,
                "wall_time_s": round(run.wall_time_s, 2),
                "new_models": sorted(run.evidence.new_models),
                "files": run.evidence.files,
                "tools_used": sorted(run.evidence.tool_names),
                "checks": results,
            })
            for r in run.results:
                if r.status == "fail":
                    failures.append(f"[{r.kind}] {r.description}: {r.reason}")
    finally:
        for run in runs:
            run.cleanup()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"{stamp}-{test_name}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[behavioral] results written to {out_path}", flush=True)

    if failures:
        joined = "\n".join(f"  - {f}" for f in failures)
        pytest.fail(
            f"{len(failures)} behavioral expectation(s) failed for {test_name}:\n{joined}",
            pytrace=False,
        )
