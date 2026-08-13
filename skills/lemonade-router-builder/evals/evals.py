# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Behavioral tests for the `lemonade-router-builder` skill.

Run locally (needs the `claude` CLI authenticated):

    cd eval/behavioral
    python -m pytest -c pytest.ini -p conftest ../../skills/lemonade-router-builder/evals/evals.py

Each check on `run` prints a `[PASS]`/`[FAIL]` line and raises on failure.
`logs_contains` / `workspace_contains` are deterministic; `should` /
`should_not` are graded by an LLM judge over the captured evidence.
"""

from harness import claude


def test_keyword_router_generation():
    """Trigger: user asks to route coding questions to a bigger model."""
    agent_configs = [(claude, "opus")]
    for agent, model in agent_configs:
        with agent(model, skill="lemonade-router-builder") as run:
            run = run.prompt(
                "Route coding questions - anything mentioning functions, bugs, "
                "or stack traces - to Qwen3.5-9B-GGUF. Everything else goes to "
                "Qwen3.5-2B-GGUF."
            )

            # Deterministic: skill must produce a JSON file and validate it
            run.logs_contains("lemonade-router-builder")
            run.workspace_contains("router.json")

            # Positive behavioral expectations
            run.should("Produce a collection.router JSON with recipe collection.router")
            run.should("Include keywords_any condition matching coding-related terms")
            run.should("Set Qwen3.5-2B-GGUF as default_model")
            run.should("Run the offline validator scripts/validate.py and report it passed")
            run.should("Output curl commands for the user to register and test the policy")

            # Negative behavioral expectations
            run.should_not("Execute POST /api/v1/pull by making an actual HTTP call with a tool")
            run.should_not("Invent model names not provided by the user")
            run.should_not("Include a routing.router key anywhere in router.json alongside routing.rules")


def test_pii_regex_router_generation():
    """Trigger: user asks to keep PII local using regex patterns."""
    agent_configs = [(claude, "opus")]
    for agent, model in agent_configs:
        with agent(model, skill="lemonade-router-builder") as run:
            run = run.prompt(
                "Any message with a Social Security number or email address must "
                "stay on Qwen3.5-9B-GGUF. Everything else can go to "
                "Qwen3.5-9B-NoThinking."
            )

            run.logs_contains("lemonade-router-builder")
            run.workspace_contains("router.json")

            run.should("Include a regex condition matching SSN patterns")
            run.should("Include a regex condition matching email addresses")
            run.should("Place the PII rule before any other rules")
            run.should("Run the offline validator and confirm the JSON is ready")

            run.should_not("Execute any live Lemonade server API call by making an actual HTTP call with a tool")
            run.should("Use routing.rules (not routing.router) to implement this request")


def test_llm_as_router_generation():
    """Trigger: user describes intent only by meaning with no concrete signals."""
    agent_configs = [(claude, "opus")]
    for agent, model in agent_configs:
        with agent(model, skill="lemonade-router-builder") as run:
            run = run.prompt(
                "I want sensitive queries to go to Qwen3.5-9B-GGUF and everything "
                "else to Qwen3.5-9B-NoThinking. Use the local model as the router."
            )

            run.logs_contains("lemonade-router-builder")
            run.workspace_contains("router.json")

            run.should("Use routing.router block instead of routing.rules")
            run.should("Set type to llm inside the router block")
            run.should("Write a prompt that describes routing intent only, not reply format")
            run.should("Contain only routing.router with no routing.rules key present in router.json")
            run.should("Write a routing.router prompt that describes only when to pick each model, with no instruction about reply format or output format")


def test_non_trigger_general_question():
    """Non-trigger: general question unrelated to routing should not activate skill."""
    agent_configs = [(claude, "opus")]
    for agent, model in agent_configs:
        with agent(model, skill="lemonade-router-builder") as run:
            run = run.prompt("What is the capital of France?")

            run.should_not("Produce a collection.router JSON")
            run.should_not("Run scripts/validate.py")
            run.should_not("Mention routing.rules or routing.candidates")


def test_non_trigger_unrelated_code_task():
    """Non-trigger: coding task unrelated to Lemonade routing."""
    agent_configs = [(claude, "opus")]
    for agent, model in agent_configs:
        with agent(model, skill="lemonade-router-builder") as run:
            run = run.prompt("Write a Python function to reverse a linked list.")

            run.should_not("Generate a collection.router policy")
            run.should_not("Ask about Lemonade model candidates")
