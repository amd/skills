# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Behavioral tests for the ``magpie-kernel-evaluator`` skill.

Run locally (requires an authenticated ``claude`` CLI):

    cd eval/behavioral
    python -m pytest -c pytest.ini -p conftest \
        ../../skills/magpie-kernel-evaluator/evals/evals.py

The HIP fixture is based on Magpie's ``examples/simple_hip_test``. Its analyze
config adds an explicit compile command so the fixture is self-contained. These
tests validate skill activation and the opening workflow without compiling a
kernel, running a GPU workload, installing dependencies, or fabricating results.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from harness import Agent, claude


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "simple_hip_test"


def _copy_analyze_fixture(agent: Agent) -> None:
    """Stage the self-contained version of Magpie's simple HIP example."""
    assert agent.workspace is not None
    shutil.copytree(FIXTURE_DIR, agent.workspace / "examples" / "simple_hip_test")


def _copy_compare_fixture(agent: Agent) -> None:
    """Stage independent baseline and candidate copies of vector_add."""
    assert agent.workspace is not None
    source = (FIXTURE_DIR / "vector_add.hip").read_text(encoding="utf-8")
    candidate = source.replace("int blockSize = 256;", "int blockSize = 128;")
    assert candidate != source, "simple HIP fixture no longer contains blockSize 256"
    variants = {
        "baseline": source,
        "candidate": candidate,
    }
    for name, contents in variants.items():
        variant_dir = agent.workspace / "simple_hip_compare" / name
        variant_dir.mkdir(parents=True)
        (variant_dir / "vector_add.hip").write_text(contents, encoding="utf-8")


def test_analyze_simple_hip_kernel():
    """Trigger: prepare Magpie analyze for the official simple HIP example."""
    with claude("opus", skill="magpie-kernel-evaluator") as agent:
        _copy_analyze_fixture(agent)
        run = agent.prompt(
            """
            Use Magpie to prepare an analyze run for the existing
            examples/simple_hip_test/analyze_default.yaml fixture. Inspect the
            fixture and write analyze_plan.md with the exact command and an
            explanation of what the testcase validates.

            Do not compile or execute the kernel, install dependencies, or
            change the fixture. Do not claim that correctness or performance
            was measured.
            """
        )

        run.logs_contains("magpie-kernel-evaluator")
        run.workspace_contains("analyze_plan.md")
        run.should(
            "Use Magpie analyze with examples/simple_hip_test/analyze_default.yaml "
            "and explain that the testcase exit code validates vector-add correctness"
        )
        run.should_not(
            "Compile or execute the HIP kernel, install dependencies, or claim "
            "measured correctness or performance results"
        )


def test_compare_simple_hip_variants():
    """Trigger: configure a correctness-gated comparison of two HIP variants."""
    with claude("opus", skill="magpie-kernel-evaluator") as agent:
        _copy_compare_fixture(agent)
        run = agent.prompt(
            """
            Use Magpie to prepare a comparison of these existing HIP kernels:

            - simple_hip_compare/baseline/vector_add.hip
            - simple_hip_compare/candidate/vector_add.hip

            Create simple_hip_compare/compare.yaml. Give each implementation
            its own working_dir, hipcc compile_command, and testcase_command so
            Magpie builds and tests two different binaries. In your final
            response, give the exact Magpie command and treat correctness as a
            hard gate before performance ranking. Do not pass --baseline; keep
            the baseline as a named config entry because the current comparison
            report does not consume that CLI selection.

            Do not compile or execute the kernels and do not claim a winner.
            """
        )

        run.logs_contains("magpie-kernel-evaluator")
        run.workspace_contains("simple_hip_compare/compare.yaml")
        run.should(
            "Configure two independent Magpie compare candidates with separate "
            "working directories, compile commands, and testcase commands, and "
            "require correctness before performance ranking without relying "
            "on the --baseline CLI option"
        )
        run.should_not(
            "Compile or execute either HIP kernel, or claim measured results or a winner"
        )


def test_benchmark_optimization_plan():
    """Trigger: plan the benchmark-to-kernel optimization workflow."""
    with claude("opus", skill="magpie-kernel-evaluator") as agent:
        run = agent.prompt(
            """
            I have a slow vLLM workload on AMD GPUs. Use Magpie to write
            benchmark_plan.md describing the shortest safe workflow from a
            clean baseline benchmark to profiling and TraceLens or gap analysis,
            then kernel analyze/compare and a final clean benchmark.

            This environment has no model weights or runnable benchmark setup.
            Do not install, download, or execute anything, and do not invent
            measured results.
            """
        )

        run.logs_contains("magpie-kernel-evaluator")
        run.workspace_contains("benchmark_plan.md")
        run.should(
            "Plan a clean baseline, a separate profiled run with TraceLens or "
            "gap analysis, correctness-gated kernel evaluation, and a final "
            "clean end-to-end benchmark"
        )
        run.should_not(
            "Run a benchmark, install or download anything, or claim measured results"
        )


def test_unrelated_programming_request_does_not_trigger_magpie():
    """Non-trigger: a general programming task should not invoke Magpie."""
    with claude("opus", skill="magpie-kernel-evaluator") as agent:
        run = agent.prompt(
            "Write a Python function that reverses a singly linked list and "
            "include unit tests."
        )

        run.should_not(
            "Invoke Magpie or create a benchmark, profiling, kernel analyze, "
            "or kernel comparison workflow"
        )
