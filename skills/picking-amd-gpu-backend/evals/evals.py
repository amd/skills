# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Behavioral tests for the `picking-amd-gpu-backend` skill.

Run locally (needs the `claude` CLI authenticated; an AMD GPU and a reachable
Lemonade Server make the checks meaningful, otherwise the agent can only
reason about the decision tree):

    cd eval/behavioral
    python -m pytest -c pytest.ini -p conftest ../../skills/picking-amd-gpu-backend/evals/evals.py

On a Windows console with a non-UTF-8 codepage (CJK locales default to GBK,
Shift-JIS, etc.), the harness crashes with `UnicodeEncodeError` while *printing*
a passing result, because judge explanations can contain characters like `2048²`.
The check itself succeeded; only the report failed. Force UTF-8 output:

    PYTHONIOENCODING=utf-8 python -m pytest -c pytest.ini -p conftest ../../skills/picking-amd-gpu-backend/evals/evals.py

Each check on `run` prints a `[PASS]`/`[FAIL]` line and raises on failure, so
the test fails at the first unmet expectation. `logs_contains` /
`workspace_contains` are deterministic; `should` / `should_not` are graded by
an LLM judge over the captured evidence.

The three cases below cover the failure modes this skill exists to prevent:
picking a backend that cannot install, accepting a backend without checking
which GPU actually ran the work, and misdiagnosing a per-allocation cap as a
capacity problem.

Known instability, measured rather than assumed
-----------------------------------------------

Run the cases **one at a time**. Every check here passes in isolation, often
repeatedly. Running all three in one pytest invocation produced 3 failures on
each of two consecutive runs — but a *different* set of checks failed each
time, including checks that had just passed in isolation. Same prompts, same
skill, so the variance is in the judging layer (roughly 18 judge calls in five
minutes), not in the agent's answers.

Treat a whole-suite failure as inconclusive and re-run the specific case alone
before concluding the skill regressed:

    PYTHONIOENCODING=utf-8 python -m pytest -c pytest.ini -p conftest \\
      ../../skills/picking-amd-gpu-backend/evals/evals.py -k test_picks_vulkan_on_windows

Writing checks that survive an LLM judge
----------------------------------------

Two wording bugs in earlier versions of this file failed *correct* answers:

- A ``should_not`` must be a **simple, positive** description of one wrong
  behavior. Compound or negated forms ("state X *without* doing Y") become
  double negatives inside ``should_not`` and were judged wrong on every run.
- A ``should_not`` must not be satisfiable by a **dismissal** of the thing it
  names. "Recommend GGML_VK_FORCE_MAX_ALLOCATION_SIZE as a way to raise the cap"
  tripped when the agent named the variable to explain that it *cannot* raise
  the cap. Phrase the false claim, not the mention.
- A ``should`` should describe the **substance**, not one phrasing of it.
  Requiring "VRAM rising on load and falling on unload" passed once and failed
  the next run, because the agent described the same check differently.
"""

from harness import claude


def test_picks_vulkan_on_windows():
    """The headline decision: do not send the user down the ROCm path on Windows."""
    agent_configs = [(claude, "opus")]
    for agent, model in agent_configs:
        with agent(model, skill="picking-amd-gpu-backend") as agent:
            run = agent.prompt(
                "I have a Radeon RX 7600M XT on Windows and I want to generate "
                "images locally. Which GPU backend should I install, and why?"
            )

            run.logs_contains("picking-amd-gpu-backend")

            run.should("Recommend the vulkan backend for this Windows setup")
            run.should(
                "Explain that the ROCm backend install currently fails on Windows, "
                "and that this is an upstream packaging problem rather than the "
                "user's hardware being unsupported"
            )
            run.should(
                "Note that Vulkan still reaches the GPU's matrix cores, so choosing "
                "it is not a performance compromise"
            )

            run.should_not(
                "Tell the user to install the ROCm or HIP backend as the primary path"
            )
            run.should_not(
                "Claim the GPU is unsupported or that the user needs different hardware"
            )


def test_verifies_which_gpu_ran_the_work():
    """A backend that 'works' may still be running on the integrated GPU.

    The prompt asks both how to *target* the discrete card and how to *confirm*
    it afterwards. Asking only about confirmation does not reliably elicit the
    device-ordering warning, which made an earlier version of this test fail on
    a correct answer.
    """
    agent_configs = [(claude, "opus")]
    for agent, model in agent_configs:
        with agent(model, skill="picking-amd-gpu-backend") as agent:
            run = agent.prompt(
                "I installed the vulkan backend on a laptop that has both an "
                "integrated GPU and a discrete Radeon. Generation works but feels "
                "slow. How do I make sure the discrete card is the one being used, "
                "and how do I confirm it afterwards?"
            )

            # Phrased loosely on purpose. An earlier version required the
            # load/unload delta specifically and passed on one run, failed on the
            # next, because the agent described the same check differently. What
            # matters is that dedicated VRAM is the signal, not how it is observed.
            run.should(
                "Use dedicated VRAM on the discrete adapter as the signal for "
                "which GPU is actually running the model"
            )
            run.should(
                "Explain that an integrated GPU is UMA and reports zero dedicated "
                "memory, so it can be told apart from the discrete card"
            )
            run.should(
                "Point out that the first enumerated device is not necessarily the "
                "discrete GPU, so the device list should be inspected rather than "
                "assumed"
            )

            # A should_not must be a simple, positive description of one wrong
            # behavior. An earlier version read "State that the discrete GPU is
            # being used *without* giving the user any way to check" — a compound
            # negation, which becomes a double negative inside should_not and
            # failed on two consecutive runs against a correct answer.
            run.should_not(
                "Claim that the vulkan backend automatically selects the discrete "
                "GPU, so no check is needed"
            )


def test_diagnoses_allocation_cap_not_capacity():
    """The error names a byte count, not a concept; the wrong fix wastes hours.

    The negative expectations below are phrased as the *false claim* rather than
    the mention. An earlier version said "Recommend GGML_VK_FORCE_MAX_ALLOCATION_SIZE
    as a way to raise the cap", which the judge marked as observed simply because
    the agent named the variable while explaining that it cannot raise the cap —
    failing a correct answer. A `should_not` must not be satisfiable by a
    dismissal of the thing it names.
    """
    agent_configs = [(claude, "opus")]
    for agent, model in agent_configs:
        with agent(model, skill="picking-amd-gpu-backend") as agent:
            run = agent.prompt(
                "Image generation works at 2048x2048 on my 8GB Radeon but fails at "
                "3072x3072 with 'ggml_vulkan: Device memory allocation of size "
                "5368709120 failed'. How do I get more VRAM available so it fits?"
            )

            run.should(
                "Identify 5368709120 as a 5 GiB per-allocation limit rather than "
                "the card running out of total VRAM"
            )
            run.should(
                "Explain that adding available memory will not raise this ceiling"
            )
            run.should(
                "Recommend generating at a smaller size and running a separate "
                "upscale pass instead"
            )

            run.should_not(
                "Tell the user that enabling enable_dgpu_gtt will make the "
                "3072x3072 generation succeed"
            )
            run.should_not(
                "Tell the user that setting GGML_VK_FORCE_MAX_ALLOCATION_SIZE will "
                "let them exceed the 5 GiB allocation limit"
            )
            run.should_not(
                "Tell the user that switching to a smaller or quantized model will "
                "make the 3072x3072 generation succeed"
            )
