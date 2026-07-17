# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Behavioral tests for the `local-ai-app-integration` skill.

Run locally (needs the `claude` CLI authenticated):

    cd eval/behavioral
    python -m pytest -c pytest.ini -p conftest \
        ../../skills/local-ai-app-integration/evals/evals.py

Prompts are scoped to code-generation only ("Do not download or install
anything") to avoid the agent attempting the GitHub download, which hangs
indefinitely. Checks prefer `logs_contains` / `workspace_contains` (instant)
over `should` / `should_not` (spawns a judge subprocess) wherever possible.
"""

from harness import claude

_STUB = "from openai import OpenAI\nclient = OpenAI()\n"


def test_launcher_module_written():
    with claude("opus", skill="local-ai-app-integration") as agent:
        (agent.workspace / "main.py").write_text(_STUB)

        run = agent.prompt(
            "Help me add local AI to this Python app using embeddable lemonade "
            "so it replaces the OpenAI API with a local backend. "
            "Do not download or install anything — just write the launcher file."
        )

        run.logs_contains("local-ai-app-integration")  # skill triggered by description
        run.workspace_contains("lemond_launcher.py")
        run.logs_contains("secrets")      # random API key generation
        run.logs_contains("socket")       # dynamic port via socket bind
        run.logs_contains("subprocess")   # lemond spawned as subprocess


def test_http_client_timeout_is_120s():
    with claude("opus", skill="local-ai-app-integration") as agent:
        (agent.workspace / "main.py").write_text(_STUB)

        run = agent.prompt(
            "Update main.py to re-point the OpenAI client at a local lemond "
            "instance. Do not download or install anything — just edit the file. "
            "Use the local-ai-app-integration skill."
        )

        run.logs_contains("local-ai-app-integration")  # skill was invoked
        run.workspace_contains("main.py")
        run.logs_contains("120")          # 120s timeout present in written code


def test_health_check_uses_http_not_stdout():
    with claude("opus", skill="local-ai-app-integration") as agent:
        (agent.workspace / "main.py").write_text(_STUB)

        run = agent.prompt(
            "Write a health-check helper for lemond in this Python app. "
            "Do not download or install anything — just write the code. "
            "Use the local-ai-app-integration skill."
        )

        run.logs_contains("local-ai-app-integration")  # skill was invoked
        run.logs_contains("/api/v1/health")
        run.should_not("Read or parse lemond's stdout or stderr to detect readiness")


def test_no_preload_call_in_written_code():
    with claude("opus", skill="local-ai-app-integration") as agent:
        (agent.workspace / "main.py").write_text(_STUB)

        run = agent.prompt(
            "Write a lemond launcher for this Python app that waits for the "
            "server to be ready. Do not download or install anything. "
            "Use the local-ai-app-integration skill."
        )

        run.logs_contains("local-ai-app-integration")  # skill was invoked
        run.logs_contains("/api/v1/health")
        run.should_not("Call POST /api/v1/load to pre-load the model at startup")


def test_api_key_gate_bypassed_in_local_mode():
    with claude("opus", skill="local-ai-app-integration") as agent:
        (agent.workspace / "main.py").write_text(
            "import os\n"
            "from openai import OpenAI\n\n"
            "api_key = os.environ.get('OPENAI_API_KEY', '')\n"
            "if not api_key:\n"
            "    raise SystemExit('No API key set. Please enter your OpenAI key.')\n\n"
            "client = OpenAI(api_key=api_key)\n"
        )

        run = agent.prompt(
            "Edit main.py so it works in local mode without an OPENAI_API_KEY. "
            "Do not download or install anything — just edit the file. "
            "Use the local-ai-app-integration skill."
        )

        run.logs_contains("local-ai-app-integration")  # skill was invoked
        run.workspace_contains("main.py")
        run.should(
            "Remove or bypass the API-key guard so the app starts in local mode "
            "without requiring OPENAI_API_KEY to be set"
        )
