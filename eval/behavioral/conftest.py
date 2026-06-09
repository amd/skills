"""pytest wiring for the behavioral harness.

- Adds this directory to ``sys.path`` so tests can ``from harness import ...``.
- Skips the whole suite unless the prerequisites (the ``claude`` CLI and a
  reachable Lemonade Server) are present -- these are real, local-only tests.

Assertions and reporting live on the `Run` object itself (each check prints a
``[PASS]``/``[FAIL]`` line and raises on failure), so there is no per-test
finalizer here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import harness  # noqa: E402


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
