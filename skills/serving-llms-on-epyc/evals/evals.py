# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Behavioral and hardware-policy tests for `serving-llms-on-epyc`.

Run locally (behavioral tests need an authenticated `claude` CLI):

    cd eval/behavioral
    python -m pytest -c pytest.ini -p conftest \
        ../../skills/serving-llms-on-epyc/evals/evals.py
"""

import json
import sys
from pathlib import Path

from harness import claude

SKILL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_DIR / "scripts"))

from check_model import (  # noqa: E402
    DEFAULT_VLLM_VERSION,
    classify_template_field,
    endpoints_for,
)
from detect import SUPPORTED_EPYC_GENERATIONS, _epyc_generation  # noqa: E402
from validate import stack_compatibility  # noqa: E402


def _stack(vllm, zen_active=True, platform="vllm.platforms.zen_cpu.ZenCpuPlatform"):
    return {"vllm": vllm, "zentorch": "2.11.0.3", "torch": "2.11.0",
            "platform": platform, "zen_active": zen_active, "source": "container"}


def test_default_stack_pin_is_consistent():
    data = json.loads((SKILL_DIR / "data" / "epyc.json").read_text(encoding="utf-8"))
    image = data["container"]["image"]

    assert data["vllm_version"] == "0.25.1"
    assert DEFAULT_VLLM_VERSION == data["vllm_version"]
    assert "vllm_v0.25.1" in image
    assert "zentorch_v2.11.0.3" in image
    assert "ubuntu22.04_2026_ww30" in image


def test_detector_recognizes_venice():
    generation, zen_arch = _epyc_generation(
        "AMD EPYC 9996 256-Core Processor"
    )

    assert generation == "Venice"
    assert zen_arch == "Zen6"
    assert generation in SUPPORTED_EPYC_GENERATIONS


def test_detector_does_not_treat_epyc_4000_as_supported():
    generation, zen_arch = _epyc_generation(
        "AMD EPYC 4564P 16-Core Processor"
    )

    assert generation == "EPYC 4004"
    assert zen_arch == "Zen4"
    assert generation not in SUPPORTED_EPYC_GENERATIONS


# --- Venice stack-compatibility gate (pure policy) ---

def test_venice_on_default_vllm_proceeds_without_warning():
    verdict = stack_compatibility("Venice", _stack(DEFAULT_VLLM_VERSION))
    assert verdict["status"] == "proceed"


def test_venice_on_default_vllm_local_suffix_normalizes():
    verdict = stack_compatibility("Venice", _stack(f"{DEFAULT_VLLM_VERSION}+cpu"))
    assert verdict["status"] == "proceed"


def test_venice_on_other_vllm_requires_confirmation():
    verdict = stack_compatibility("Venice", _stack("0.24.0"))
    assert verdict["status"] == "confirmation_required"
    assert DEFAULT_VLLM_VERSION in verdict["message"]


def test_non_venice_on_other_vllm_still_proceeds():
    verdict = stack_compatibility("Turin", _stack("0.24.0"))
    assert verdict["status"] == "proceed"


def test_stock_cpu_platform_is_blocked():
    verdict = stack_compatibility(
        "Turin", _stack("0.25.1", zen_active=False, platform="vllm.platforms.cpu.CpuPlatform")
    )
    assert verdict["status"] == "blocked"


def test_no_stack_yields_no_verdict():
    assert stack_compatibility("Venice", None) is None


# --- chat-template classification + endpoint selection (pure) ---

def test_string_template_is_present():
    assert classify_template_field("{{ messages }}") == ("present", None, [])


def test_named_templates_with_default_are_present():
    ct = [{"name": "default", "template": "a"}, {"name": "tool_use", "template": "b"}]
    status, selected, names = classify_template_field(ct)
    assert status == "present"
    assert selected == "default"
    assert names == ["default", "tool_use"]


def test_single_named_template_is_present():
    status, selected, names = classify_template_field([{"name": "chat", "template": "a"}])
    assert status == "present"
    assert selected == "chat"


def test_multiple_named_templates_without_default_are_ambiguous():
    ct = [{"name": "a", "template": "x"}, {"name": "b", "template": "y"}]
    status, selected, names = classify_template_field(ct)
    assert status == "ambiguous"
    assert selected is None
    assert names == ["a", "b"]


def test_missing_template_is_absent():
    assert classify_template_field(None) == ("absent", None, [])
    assert classify_template_field("") == ("absent", None, [])
    assert classify_template_field([]) == ("absent", None, [])


def test_text_with_template_prefers_chat():
    eps, primary, needs = endpoints_for("text", "present")
    assert primary == "chat_completions"
    assert eps == ["chat_completions", "completions"]
    assert needs is False


def test_text_without_template_uses_completions():
    for status in ("absent", "ambiguous", "unknown"):
        eps, primary, needs = endpoints_for("text", status)
        assert primary == "completions"
        assert eps == ["completions"]
        assert needs is False


def test_multimodal_without_template_is_not_launchable():
    eps, primary, needs = endpoints_for("multimodal", "absent")
    assert primary is None
    assert eps == []
    assert needs is True


def test_multimodal_with_template_uses_chat():
    eps, primary, needs = endpoints_for("multimodal", "present")
    assert primary == "chat_completions"
    assert needs is False


def test_epyc_4000_is_rejected_even_with_avx512():
    with claude("opus", skill="serving-llms-on-epyc") as agent:
        run = agent.prompt(
            "I have an AMD EPYC 4564P with AVX-512 and want to serve Qwen with "
            "vLLM and zentorch. Assess whether this skill supports that CPU. "
            "Do not execute commands or launch anything."
        )

        run.logs_contains("serving-llms-on-epyc")
        run.should(
            "State that EPYC 4004 is outside this skill's documented ZenDNN "
            "server targets and that AVX-512 alone is not sufficient"
        )
        run.should_not("Recommend proceeding with the launch on EPYC 4004")


def test_cpu_endpoint_is_allowed_on_epyc_plus_instinct_host():
    with claude("opus", skill="serving-llms-on-epyc") as agent:
        run = agent.prompt(
            "This host has an AMD EPYC 9965 CPU and an AMD Instinct MI350X GPU. "
            "I explicitly want a Qwen vLLM endpoint on the CPU so a separate GPU "
            "engine can remain available. Give me the plan only; do not execute "
            "commands or launch anything."
        )

        run.logs_contains("serving-llms-on-epyc")
        run.should(
            "Use the EPYC CPU-serving workflow despite the co-installed Instinct GPU"
        )
        run.should(
            "Plan a single-socket CPU endpoint and require confirmation before launch"
        )
        run.should_not(
            "Reject the CPU-serving request merely because an Instinct GPU is present"
        )


def test_venice_on_old_vllm_pauses_for_confirmation():
    with claude("opus", skill="serving-llms-on-epyc") as agent:
        run = agent.prompt(
            "On a 6th Gen AMD EPYC Venice host, my conda env has vLLM 0.24.0 with "
            "zentorch already active (a Zen CPU platform). I want to serve Qwen. "
            "Should I just launch? Plan only; do not execute anything."
        )

        run.logs_contains("serving-llms-on-epyc")
        run.should(
            "Warn that Venice on vLLM 0.24.0 is not validated by this recipe, stop for "
            "explicit confirmation, and recommend the pinned vLLM 0.25.1 image"
        )
        run.should_not("Launch immediately without any Venice-version warning")


def test_venice_on_default_vllm_proceeds_cleanly():
    with claude("opus", skill="serving-llms-on-epyc") as agent:
        run = agent.prompt(
            "On a 6th Gen AMD EPYC Venice host using the pinned amdih/zendnn_zentorch "
            "vLLM 0.25.1 container from data/epyc.json, I want to serve Qwen. Plan only; "
            "do not execute anything."
        )

        run.logs_contains("serving-llms-on-epyc")
        run.should(
            "Proceed on Venice with the pinned vLLM 0.25.1 stack without raising a "
            "Venice-version compatibility warning"
        )


def test_base_model_hands_off_completions_not_chat():
    with claude("opus", skill="serving-llms-on-epyc") as agent:
        run = agent.prompt(
            "Serve allenai/OLMo-2-0425-1B on this AMD EPYC box with vLLM and zentorch. It is "
            "a base model with no chat template. Plan the client handoff only; do not execute."
        )

        run.logs_contains("serving-llms-on-epyc")
        run.should(
            "Hand off the /v1/completions endpoint with a raw prompt for this base model"
        )
        run.should_not(
            "Present /v1/chat/completions with messages as if a chat template exists"
        )


def test_chat_model_selects_chat_endpoint():
    with claude("opus", skill="serving-llms-on-epyc") as agent:
        run = agent.prompt(
            "Serve Qwen/Qwen3-0.6B on this AMD EPYC box with vLLM and zentorch. Plan the "
            "client handoff only; do not execute anything."
        )

        run.logs_contains("serving-llms-on-epyc")
        run.should(
            "Choose /v1/chat/completions with messages as the client endpoint for this "
            "chat model (it ships a chat template)"
        )
        run.should_not(
            "Hand off /v1/completions with a raw prompt as the primary call for this chat model"
        )
