# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Behavioral tests for the `local-ai-use` skill.

Run locally (needs the `claude` CLI authenticated and a reachable Lemonade
Server -- otherwise the suite skips):

    pytest skills/local-ai-use/evals/evals.py -s

Each check on `run` prints a `[PASS]`/`[FAIL]` line and raises on failure, so
the test fails at the first unmet expectation. `logs_contains` /
`workspace_contains` are deterministic; `should` / `should_not` are graded by
an LLM judge over the captured evidence.
"""

from harness import claude


def test_generate_image_of_a_cat():
    agent_configs = [(claude, "sonnet")]
    for agent, model in agent_configs:
        with agent(model, skill="local-ai-use") as agent:
            run = agent.prompt(
                "Learn how to generate images locally, then generate an image of a "
                "cat and save it to out.png."
            )

            # Programmatic expectations
            run.logs_contains("local-ai-use")
            run.workspace_contains("AGENTS.md")
            run.workspace_contains("out.png")

            # Positive behavioral expectations
            run.should("Install Lemonade Server if it is not already installed")
            run.should("Download the SD-Turbo model if the model is not already downloaded")
            run.should("Add a 'Local AI Use' block to AGENTS.md")

            # Negative behavioral expectations
            run.should_not("Pull unrelated modalities for this image generation task")
            run.should_not("Reach for a cloud image path instead of local Lemonade")
