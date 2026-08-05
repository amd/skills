# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Behavioral tests for the `hrr-replay-analysis` skill.

Two tiers, because triage and replay have different hardware needs:

* **Recorded-evidence tier** (default). Runs against replay logs checked into
  ``evals/fixtures/``, so it grades the agent's *reasoning* -- does it name the
  right kernel, the right fault address, the right fault class -- on a machine
  with no GPU. These are the tests CI can run today.
* **Live-replay tier**. Replays a real archive on a ROCm host. Skipped unless
  ``/dev/kfd``, an ``hrr-playback`` binary, and ``HRR_EVAL_ARCHIVE`` are all
  present, following the same skip-when-unavailable pattern as `local-ai-use`.

Run locally (needs the `claude` CLI authenticated):

    cd eval/behavioral
    python -m pytest -c pytest.ini -p conftest \
        ../../skills/hrr-replay-analysis/evals/evals.py

Each check on `run` prints a `[PASS]`/`[FAIL]` line and raises on failure.
`logs_contains` / `workspace_contains` are deterministic; `should` /
`should_not` are graded by an LLM judge over the captured evidence.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from harness import claude

SKILL = "hrr-replay-analysis"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# The fault recorded in fixtures/read_only_page_fault: a StreamK GEMM writing
# past the tile edge into a read-only page. The agent has to recover all three.
FAULT_ADDRESS = "0x7f2a4c000000"
FAULT_KERNEL = "Cijk_Ailk_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_SK3_WGM8"
FAULT_CLASS = "read_only_page_fault"


def _stage(agent, fixture: str, name: str = "replay.log") -> Path:
    """Copy a recorded fixture into the agent's workspace."""
    dest = agent.workspace / name
    shutil.copyfile(FIXTURES / fixture / "replay.log", dest)
    return dest


def _finding_text(agent, filename: str = "finding.md") -> str:
    path = agent.workspace / filename
    assert path.is_file(), f"{filename} was not written to {agent.workspace}"
    return path.read_text(encoding="utf-8", errors="replace")


def _live_replay_skip_reason() -> str | None:
    """Why the live-replay tier cannot run here, or None when it can."""
    if not Path("/dev/kfd").exists():
        return "no /dev/kfd: this host has no ROCm GPU"
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    if not (shutil.which("hrr-playback") or (rocm / "bin" / "hrr-playback").is_file()):
        return "hrr-playback not found on PATH or under $ROCM_PATH/bin"
    archive = os.environ.get("HRR_EVAL_ARCHIVE", "").strip()
    if not archive:
        return "HRR_EVAL_ARCHIVE is not set to an HRR archive directory"
    if not Path(archive).is_dir():
        return f"HRR_EVAL_ARCHIVE does not exist: {archive}"
    return None


def test_triage_read_only_page_fault_from_log():
    """The core case: a recorded fault log becomes a correct structured finding."""
    with claude("opus", skill=SKILL) as agent:
        _stage(agent, "read_only_page_fault")

        run = agent.prompt(
            "Analyze this replay log from an HRR run and write the structured "
            "finding to finding.md: replay.log"
        )

        run.logs_contains(SKILL)
        run.workspace_contains("finding.md")

        finding = _finding_text(agent)
        assert FAULT_CLASS in finding, f"fault class missing from finding:\n{finding}"
        assert FAULT_ADDRESS in finding, f"fault address missing from finding:\n{finding}"
        assert FAULT_KERNEL in finding, f"faulting kernel missing from finding:\n{finding}"

        run.should("Report the failure as a write to a read-only page")
        run.should(f"Name {FAULT_KERNEL} as the implicated kernel")
        run.should(f"Report the faulting address {FAULT_ADDRESS}")
        run.should("Explain the finding in plain language to the user")

        run.should_not("Re-record the workload instead of analyzing the log provided")
        run.should_not("Attempt to patch, rebuild or replace the kernel")
        run.should_not("Invent a root cause the log does not support")


def test_version_mismatch_is_not_reported_as_a_workload_failure():
    """False-positive guard: an unreadable archive is a tooling problem.

    The skill is explicit that a format mismatch means nothing was replayed, so
    reporting it as a crash would send a kernel developer after a bug that this
    run never observed.
    """
    with claude("opus", skill=SKILL) as agent:
        _stage(agent, "version_mismatch")

        run = agent.prompt("Analyze this HRR replay log and tell me what happened: replay.log")

        run.logs_contains(SKILL)

        run.should("Report that the archive format and the hrr-playback build disagree")
        run.should("State that nothing was replayed")
        run.should("Say that a matching hrr-playback build or a fresh capture is needed")

        run.should_not("Report a GPU fault, a crash, or any workload-level failure")
        run.should_not("Name a faulting kernel")


def test_clean_replay_asks_for_the_original_failure_signature():
    """A clean replay of a workload the user says crashed is itself a finding."""
    with claude("opus", skill=SKILL) as agent:
        _stage(agent, "clean_pass")

        run = agent.prompt(
            "This workload crashed for our customer with a memory access fault, "
            "but here is the replay log from the capture: replay.log. "
            "What happened?"
        )

        run.logs_contains(SKILL)

        run.should("Report the replay as a clean pass with all device-to-host checks passing")
        run.should(
            "Treat the clean replay of a crashing workload as a finding, for example by "
            "asking for the original failure signature or noting the fault did not reproduce"
        )

        run.should_not("Claim to have reproduced the customer's memory access fault")
        # The archive listing in this fixture contains a GEMM kernel. A clean
        # replay implicates none of them, so naming one would read as a culprit.
        run.should_not("Name any kernel as implicated in a failure")


def test_aten_chevron_fault_is_reported_but_not_asserted_as_a_customer_bug():
    """The ATen fault is a real finding that must not be stated as a verdict.

    ATen kernels launched through `<<<>>>` pass device pointers inside by-value
    structs. Replay translates those, so such a fault can be genuine, but the
    translation is heuristic and older archives lack the recorded offsets, so it
    can equally be an artefact of the recording. The agent must report the fault
    it observed and ask for the original failure signature, rather than resolving
    the ambiguity in either direction on its own.
    """
    with claude("opus", skill=SKILL) as agent:
        _stage(agent, "aten_chevron_limitation")

        run = agent.prompt("Analyze this HRR replay log and tell me what happened: replay.log")

        run.logs_contains(SKILL)

        run.should("Report the memory fault and name the faulting ATen kernel")
        run.should("Flag that a <<<>>>-launched kernel faulting at replay may come from the recording rather than the workload")
        run.should("Ask for the original failure signature before concluding anything about the workload")

        run.should_not("State as settled fact that the customer's workload has a memory bug")
        run.should_not("State as settled fact that the fault is only an artefact of the recording")


def test_skill_does_not_activate_on_an_unrelated_prompt():
    """False-activation screen: an unrelated request must not pull in the skill."""
    with claude("opus", skill=SKILL) as agent:
        (agent.workspace / "notes.txt").write_text(
            "Sprint notes: rotate the API token, then update the changelog.\n",
            encoding="utf-8",
        )

        run = agent.prompt("Read notes.txt and summarize it in one sentence.")

        run.should_not("Run the HRR replay analysis scripts")
        run.should_not("Treat notes.txt as a GPU capture, replay log or HRR archive")


@pytest.mark.skipif(
    _live_replay_skip_reason() is not None,
    reason=f"live replay unavailable: {_live_replay_skip_reason()}",
)
def test_live_replay_of_a_real_archive():
    """Live-replay tier: replay a real archive end to end on a ROCm host.

    Set ``HRR_EVAL_ARCHIVE`` to an archive root or a ``pid-*`` directory. The
    archive is not checked in, so this stays opt-in and out of CI until a GPU
    runner that can reach the agent API exists.
    """
    archive = os.environ["HRR_EVAL_ARCHIVE"]

    with claude("opus", skill=SKILL) as agent:
        run = agent.prompt(f"Replay and analyze this HRR archive: {archive}")

        run.logs_contains(SKILL)

        run.should("Read the archive metadata with --info before replaying it")
        run.should("Produce a structured finding with an outcome and a fault class")
        run.should("Record which hrr-playback build produced the result")

        run.should_not("Ask the user for a GPU index, a ROCm path or a Docker command")
