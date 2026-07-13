# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""pytest wiring for the behavioral harness.

Adds this directory to ``sys.path`` so tests can ``from harness import ...``,
and runs a one-time API preflight so the (expensive) behavioral runs fail
fast with a clear message when the `claude` API isn't reachable -- e.g.
when you're not connected to the network that can reach it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import DEFAULT_MODEL, check_api_reachable  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _require_api_reachable() -> None:
    """Fail the suite up front if the `claude` API can't be reached."""
    ok, detail = check_api_reachable(DEFAULT_MODEL)
    if not ok:
        pytest.fail(
            f"claude API not reachable -- are you on the right network? ({detail})"
        )
