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

        run.should_not(
            "Start a plain vLLM docker serve as the primary answer without the optimization loop"
        )
        run.should_not(
            "Launch the optimizer immediately after setup without collecting TP, "
            "concurrency, ISL, OSL, and precision or confirming a launch plan"
        )


def test_states_launcher_gates_before_launch():
    with claude("opus", skill="hyperloom-workload-optimizer") as agent:
        run = agent.prompt(
            "What are Hyperloom's two launcher gates, in the order they run, "
            "and what does each one check? Answer in three or four sentences. "
            "Do not run anything."
        )

        run.should(
            "Mention running install.sh and sourcing kernel-agent.env.sh (IR-2) "
            "before launching the optimizer"
        )
        run.should(
            "Mention a GPU preflight check for stale serving processes or VRAM "
            "in use (IR-1)"
        )


def test_collects_workload_values_before_launch():
    with claude("opus", skill="hyperloom-workload-optimizer") as agent:
        run = agent.prompt(
            "The environment is already set up and I want to start optimizing. "
            "Which workload values do you need from me, and what happens "
            "between me giving them and the optimizer starting? Answer in four "
            "or five sentences. Do not run anything."
        )

        run.should(
            "Name the workload values it needs -- model path, framework, TP, "
            "concurrency, ISL, OSL, precision and time budget -- as its own "
            "intake step"
        )
        run.should(
            "Say it will present a launch plan and get user confirmation before "
            "launching the optimizer"
        )
        run.should(
            "Explain that confirmed workload values are persisted (e.g. to a "
            "workload.env file) and sourced at launch, since agent shells do not "
            "keep exports between calls"
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
