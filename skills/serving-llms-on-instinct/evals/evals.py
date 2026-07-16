# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Behavioral test for the `serving-llms-on-instinct` skill.

Serves a deliberately tiny model (``Qwen/Qwen3-0.6B``) end-to-end on real AMD
Instinct hardware so the run stays fast: no HF token needed (Apache 2.0), the
weights are a few hundred MB, and it fits on a single MI300X with room to
spare. The test grants launch approval up front so the agent does not stall on
the skill's "confirm before launching" step.

This test only makes sense on a machine with an AMD Instinct GPU, ROCm, and
Docker -- CI routes it to the self-hosted MI300X runner (see
``.github/workflows/behavioral.yml``). Run locally the same way as the other
behavioral tests:

    cd eval/behavioral
    python -m pytest -c pytest.ini -p conftest \
        ../../skills/serving-llms-on-instinct/evals/evals.py

Each check on `run` prints a `[PASS]`/`[FAIL]` line and raises on failure, so
the test fails at the first unmet expectation. `logs_contains` is
deterministic; `should` / `should_not` are graded by an LLM judge over the
captured evidence.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest

from harness import claude

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="serving-llms-on-instinct requires a Linux + ROCm host",
)

# Small, ungated, single-GPU-friendly model keeps the serve fast.
MODEL_ID = "Qwen/Qwen3-0.6B"


def _cleanup_test_containers() -> None:
    """Best-effort removal of vLLM containers this test may have started.

    The behavioral harness cleans up the temp workspace but knows nothing
    about Docker, so on a shared runner we tear down the container the agent
    launched. Scoped to names the skill derives from the tiny test model so we
    never touch someone else's endpoint.
    """
    docker = shutil.which("docker")
    if not docker:
        return
    try:
        ids = subprocess.run(
            [docker, "ps", "-aq", "--filter", "name=qwen3-0.6b"],
            capture_output=True, text=True, timeout=30,
        ).stdout.split()
        if ids:
            subprocess.run([docker, "rm", "-f", *ids], capture_output=True, timeout=60)
    except (subprocess.SubprocessError, OSError):
        pass


def test_serve_tiny_model_on_instinct():
    try:
        with claude("opus", skill="serving-llms-on-instinct") as agent:
            run = agent.prompt(
                "Use the serving-llms-on-instinct skill to serve "
                f"{MODEL_ID} on this AMD Instinct GPU with vLLM. This is an "
                "automated test on a machine I own: you have my approval to "
                "launch -- do not wait for confirmation. Keep it minimal and "
                "fast, then verify the endpoint is healthy and report the "
                "connection details."
            )

            # Deterministic: the skill was actually engaged.
            run.logs_contains("serving-llms-on-instinct")

            # Positive behavioral expectations.
            run.should("Detect the AMD Instinct GPU before configuring vLLM")
            run.should(
                "Launch the model with vLLM inside a Docker container on the "
                "AMD GPU"
            )
            run.should("Verify the vLLM endpoint is healthy after launching")

            # Negative behavioral expectations.
            run.should_not(
                "Fall back to a cloud LLM provider or an NVIDIA/CUDA code path"
            )
            run.should_not(
                "Serve a different, larger model than the one that was requested"
            )
    finally:
        _cleanup_test_containers()
