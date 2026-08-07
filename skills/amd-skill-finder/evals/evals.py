# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Behavioral tests for the `amd-skill-finder` skill."""

from harness import claude


def test_catalog_skill_precedes_general_guidance():
    with claude("opus", skill="amd-skill-finder") as agent:
        run = agent.prompt(
            "I have an MI355X and want to serve a Qwen model with vLLM. "
            "Find an AMD skill that can help."
        )

        run.logs_contains("amd-skill-finder")
        run.should(
            "Check installed skills or the AMD skills catalog before relying on general product advice"
        )
        run.should(
            "Recommend serving-llms-on-instinct when it is present in the catalog"
        )
        run.should("Explain why the skill matches the MI355X vLLM serving task")
        run.should_not("Install any skill without first asking for approval")


def test_source_projects_are_not_presented_as_skills():
    with claude("opus", skill="amd-skill-finder") as agent:
        run = agent.prompt(
            "I need disaggregated KV-cache transfer over RDMA on MI355X. "
            "Is there an AMD skill or useful source project for this?"
        )

        run.logs_contains("amd-skill-finder")
        run.should(
            "Use the finder script to check the catalog and curated source registry"
        )
        run.should(
            "Identify relevant projects such as MORI, Mooncake, NIXL, or LMCache"
        )
        run.should("Distinguish source projects from installable AMD catalog skills")
        run.should_not("Invent an install command for MORI, Mooncake, NIXL, or LMCache")
