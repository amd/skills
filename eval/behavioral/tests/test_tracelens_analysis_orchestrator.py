# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Behavioral tests for the `tracelens-analysis-orchestrator` skill.

Runs the first standalone repeatability case from TraceLens
(``gemm_01_compute_few_tiles`` in ``combined_traces_standalone.csv``) — the
same Phase-1 agent workflow that ``run_repeatability_parallel.sh`` schedules
first — then validates output with ``workflow_scripted_evals.py`` (the first
Phase-2 eval in that script).

Prerequisites (local run):

    pip install -r eval/behavioral/requirements.txt
    git, python3, and network access to clone AMD-AGI/TraceLens and install deps

    pytest eval/behavioral/tests/test_tracelens_analysis_orchestrator.py -s

This test is slow (TraceLens install + full orchestrator workflow). Outside CI,
``TRACELENS_BEHAVIORAL_MODEL`` defaults to ``opus``; CI coerces to ``sonnet``.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from harness import _is_automated_env, claude

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="TraceLens analysis orchestrator behavioral test requires a Unix environment",
)

TRACELENS_REPO_URL = os.environ.get(
    "TRACELENS_REPO_URL", "https://github.com/AMD-AGI/TraceLens.git"
)
TRACELENS_REF = os.environ.get("TRACELENS_REF", "").strip()
UNIT_TESTS_ARCHIVE = "unit_tests_standalone.tar.gz"
COMBINED_TRACES_CSV = (
    "agent_evals/Analysis/analysis_tests/combined_traces_standalone.csv"
)


@dataclass(frozen=True)
class RepeatabilityCase:
    """First row of ``combined_traces_standalone.csv`` (default repeatability order)."""

    test_id: str
    sub_category: str
    trace_path: Path
    reference_dir: Path
    platform: str


def _analysis_model() -> str:
    if _is_automated_env():
        return "sonnet"
    return os.environ.get(
        "TRACELENS_BEHAVIORAL_MODEL",
        os.environ.get("BEHAVIORAL_MODEL", "opus"),
    )


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def _clone_tracelens(workspace: Path) -> Path:
    dest = workspace / "TraceLens"
    if dest.exists():
        return dest

    clone_cmd = ["git", "clone", "--depth", "1", TRACELENS_REPO_URL, str(dest)]
    if TRACELENS_REF:
        clone_cmd = [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            TRACELENS_REF,
            TRACELENS_REPO_URL,
            str(dest),
        ]
    _run(clone_cmd)
    return dest


def _extract_unit_tests(tracelens_dir: Path) -> None:
    archive = tracelens_dir / "agent_evals/Analysis/analysis_tests" / UNIT_TESTS_ARCHIVE
    if not archive.is_file():
        raise FileNotFoundError(f"unit test archive not found: {archive}")

    target_root = tracelens_dir / "agent_evals/Analysis/analysis_tests"
    marker = target_root / "unit_tests_standalone"
    if marker.is_dir():
        return

    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=target_root)


def _load_first_repeatability_case(tracelens_dir: Path) -> RepeatabilityCase:
    csv_path = tracelens_dir / COMBINED_TRACES_CSV
    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    trace_rel = row["trace_path"]
    reference_rel = row["reference_dir"]
    trace_path = tracelens_dir / trace_rel
    if not trace_path.is_file():
        raise FileNotFoundError(f"trace file missing after extract: {trace_path}")

    return RepeatabilityCase(
        test_id=row["id"],
        sub_category=row["sub_category"],
        trace_path=trace_path.resolve(),
        reference_dir=(tracelens_dir / reference_rel).resolve(),
        platform=row["platform"],
    )


def _install_tracelens_venv(workspace: Path, tracelens_dir: Path) -> Path:
    venv_dir = workspace / ".venv"
    if not venv_dir.exists():
        _run([sys.executable, "-m", "venv", str(venv_dir)], cwd=workspace)

    pip = venv_dir / "bin" / "pip"
    python = venv_dir / "bin" / "python"
    _run([str(pip), "install", "--upgrade", "pip"], cwd=workspace)
    _run([str(pip), "install", "-e", str(tracelens_dir)], cwd=workspace)

    _run([str(python), "-c", "import TraceLens; print('TRACELOK')"], cwd=workspace)
    return venv_dir


def _bootstrap_repeatability_case(workspace: Path) -> tuple[Path, Path, RepeatabilityCase, Path]:
    tracelens_dir = _clone_tracelens(workspace)
    _extract_unit_tests(tracelens_dir)
    case = _load_first_repeatability_case(tracelens_dir)
    venv_dir = _install_tracelens_venv(workspace, tracelens_dir)
    output_dir = workspace / "analysis_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return tracelens_dir, venv_dir, case, output_dir


def _repeatability_prompt(
    *,
    case: RepeatabilityCase,
    trace_path: Path,
    output_dir: Path,
    tracelens_dir: Path,
    venv_dir: Path,
) -> str:
    # Mirrors the standalone Phase-1 prompt in run_repeatability_parallel.sh for
    # the first scheduled test case, adapted for the staged AMD Skills orchestrator.
    return (
        "Follow the tracelens-analysis-orchestrator skill and run the full agentic "
        f"analysis workflow on {trace_path} with platform {case.platform}, "
        "analysis mode default, running locally on the host (no cluster), "
        f"output to {output_dir}. "
        f"TraceLens is cloned at {tracelens_dir} and installed in the virtual "
        f"environment at {venv_dir}. Use that venv for all Python and TraceLens CLI "
        "commands."
    )


def _run_workflow_scripted_eval(
    *,
    tracelens_dir: Path,
    venv_python: Path,
    output_dir: Path,
) -> Path:
    """First Phase-2 eval from run_repeatability_parallel.sh."""
    results_csv = output_dir / "workflow_scripted_results.csv"
    eval_script = tracelens_dir / "agent_evals/Analysis/eval_utils/workflow_scripted_evals.py"
    _run(
        [
            str(venv_python),
            str(eval_script),
            "--output-dir",
            str(output_dir),
            "--results",
            str(results_csv),
            "--comparison-scope",
            "standalone",
        ],
        cwd=tracelens_dir,
    )
    return results_csv


def _assert_workflow_eval_csv_passes(results_csv: Path) -> None:
    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows, f"workflow eval produced no rows: {results_csv}"
    failures = [row for row in rows if row.get("result") != "PASS"]
    assert not failures, (
        "workflow_scripted_evals.py reported failures:\n"
        + "\n".join(
            f"  - {row.get('issue_summary')}: {row.get('details')}"
            for row in failures[:10]
        )
    )


def test_gemm_01_repeatability_first_case():
    """First standalone repeatability case: gemm_01_compute_few_tiles."""
    model = _analysis_model()

    with claude(model, skill="tracelens-analysis-orchestrator", effort="high") as agent:
        tracelens_dir, venv_dir, case, output_dir = _bootstrap_repeatability_case(
            agent.workspace
        )
        venv_python = venv_dir / "bin" / "python"

        assert case.test_id == "gemm_01_compute_few_tiles"

        run = agent.prompt(
            _repeatability_prompt(
                case=case,
                trace_path=case.trace_path,
                output_dir=output_dir.resolve(),
                tracelens_dir=tracelens_dir.resolve(),
                venv_dir=venv_dir.resolve(),
            )
        )

        run.logs_contains("tracelens-analysis-orchestrator")
        run.should(
            "Install or use TraceLens in a Python virtual environment before "
            "running the orchestrator workflow"
        )
        run.should(
            "Run the TraceLens analysis orchestrator workflow steps to produce "
            "analysis output artifacts"
        )

        analysis_md = output_dir / "analysis.md"
        assert analysis_md.is_file(), (
            f"analysis.md not found under {output_dir} "
            f"(workspace files: {run.files})"
        )
        assert analysis_md.stat().st_size >= 100, "analysis.md is too small"

        results_csv = _run_workflow_scripted_eval(
            tracelens_dir=tracelens_dir,
            venv_python=venv_python,
            output_dir=output_dir,
        )
        _assert_workflow_eval_csv_passes(results_csv)
