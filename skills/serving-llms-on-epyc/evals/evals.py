# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Behavioral spot checks for the `serving-llms-on-epyc` skill.

These cover what is observable without an EPYC host, so they run on the
non-EPYC self-hosted runners (see ``.github/workflows/behavioral.yml``).

Run locally:

    cd eval/behavioral
    python -m pytest -c pytest.ini -p conftest \
        ../../skills/serving-llms-on-epyc/evals/evals.py
"""

import sys

import pytest

from harness import claude

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="serving-llms-on-epyc is Linux-only (lscpu, numactl, container runtime)",
)

MODEL_ID = "Qwen/Qwen3-0.6B"


def test_non_epyc_host_is_rejected_before_anything_launches():
    """The skill's Step 1 guard has to fire on a non-EPYC host.

    A sharp negative case: ``detect.py`` can report ``avx512: true`` alongside
    ``is_amd_epyc: false`` and still exits 0, so stopping means the agent
    honored the skill rather than a failing exit code. Nothing is downloaded.
    """
    with claude("opus", skill="serving-llms-on-epyc") as agent:
        run = agent.prompt(
            f"Serve {MODEL_ID} on this box with vLLM and zentorch. This is an "
            "automated test on a machine I own: you have my approval to run "
            "detection and validation commands without asking."
        )

        run.logs_contains("serving-llms-on-epyc")
        run.logs_contains("detect.py")

        run.should(
            "Inspect this host's CPU with the skill's detect.py script before "
            "configuring anything"
        )
        run.should(
            "Report that this host's CPU is not an AMD EPYC 9000 series and stop there, "
            "naming the CPU it actually found"
        )
        run.should_not(
            "Pull a container image, install vLLM, or start vllm serve on this "
            "unsupported host"
        )
        run.should_not("Claim that a working vLLM endpoint is up")


def test_epyc_launch_plan_follows_the_documented_recipe():
    """Planning for an EPYC host the agent is not sitting on.

    The pinned zentorch image, no ``--device cpu``, and single-socket sizing
    are exactly what an agent without the skill gets wrong.
    """
    with claude("opus", skill="serving-llms-on-epyc") as agent:
        run = agent.prompt(
            "I am deploying to a separate dual-socket AMD EPYC 9004 (Genoa) "
            "server later this week. It is not this machine, so run no commands "
            "and do not probe the local host. This is a planning question. "
            "Using the serving-llms-on-epyc skill, show me the exact container "
            f"and `vllm serve` command you would run there for {MODEL_ID}, and "
            "how you would size it to that hardware."
        )

        run.logs_contains("serving-llms-on-epyc")
        run.logs_contains("amdih/zendnn_zentorch")

        run.should(
            "Take the container image tag from the skill's data/epyc.json "
            "instead of inventing or recalling one"
        )
        run.should(
            "Size the endpoint to a single socket: bind threads to that "
            "socket's physical cores and size the KV cache from that socket's "
            "local memory rather than whole-system RAM"
        )
        run.should_not("Include --device cpu in the proposed vllm serve command")
        run.should_not("Propose a GPU, CUDA, or ROCm image for this CPU launch")
