# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Behavioral smoke tests for the `hyperloom-workload-optimizer` skill.

Run locally (needs the `claude` CLI authenticated):

    pip install -r eval/behavioral/requirements.txt
    cd eval/behavioral
    python -m pytest -c pytest.ini -p conftest \
        ../../skills/hyperloom-workload-optimizer/evals/evals.py
"""

from harness import claude


def test_routes_optimize_vllm_throughput_request():
    with claude("opus", skill="hyperloom-workload-optimizer") as agent:
        run = agent.prompt(
            "I want to optimize vLLM inference throughput on an MI300X. "
            "What are the first steps before launching Hyperloom?"
        )

        run.logs_contains("hyperloom-workload-optimizer")

        run.should(
            "Mention installing the Hyperloom wheel or checking for the hyperloom package"
        )
        run.should(
            "Mention workspace bootstrap such as hyperloom-setup or setup.md"
        )
        run.should(
            "Describe a phased flow where workload parameters (model path, "
            "framework, TP, concurrency, ISL, OSL, precision, time budget) are "
            "collected in a later workload-intake phase, not during bootstrap"
        )
        run.should(
            "Say it will present a launch plan and get user confirmation before "
            "launching the optimizer"
        )
        run.should(
            "Mention a GPU preflight check for stale serving processes or VRAM "
            "in use (IR-1)"
        )
        run.should(
            "Explain that confirmed workload values are persisted (e.g. to a "
            "workload.env file) and sourced at launch, since agent shells do not "
            "keep exports between calls"
        )

        run.should_not(
            "Start a plain vLLM docker serve as the primary answer without the optimization loop"
        )
        run.should_not(
            "Launch the optimizer immediately after setup without collecting TP, "
            "concurrency, ISL, OSL, and precision or confirming a launch plan"
        )


def test_states_install_gate_before_launch():
    # Asked on its own, and answered in prose, so the reply stays inside the
    # window the judge sees; buried in a long plan this gate gets cut off.
    with claude("opus", skill="hyperloom-workload-optimizer") as agent:
        run = agent.prompt(
            "Before Hyperloom's optimize command is launched, what has to be "
            "run and sourced in that same shell, and why does it have to be the "
            "same shell? Answer in two or three sentences. Do not run anything."
        )

        run.should(
            "Mention running install.sh and sourcing kernel-agent.env.sh (IR-2) "
            "before launching the optimizer"
        )


def test_phase_discipline_bootstrap_first():
    # The skill stops for approval before it installs, and `claude -p` has no
    # user to answer, so the approval is granted in the prompt.
    with claude("opus", skill="hyperloom-workload-optimizer") as agent:
        run = agent.prompt(
            "I have a fresh empty workspace. Help me get Hyperloom set up from "
            "scratch so I can optimize a model later. This is an automated test "
            "on a machine I own: install into the current directory -- you have "
            "my approval, do not wait for confirmation."
        )

        run.should(
            "Focus on bootstrap first: confirm the install directory, install "
            "the wheel, and run hyperloom-setup for credentials and run mode"
        )
        run.should_not(
            "Ask for workload parameters like model path, TP, ISL, OSL, or "
            "precision in the same turn as install-directory or run-mode setup"
        )
        run.should_not(
            "Launch hyperloom.inference_optimizer.cli optimize before the "
            "environment is prepared and a launch plan is confirmed"
        )
