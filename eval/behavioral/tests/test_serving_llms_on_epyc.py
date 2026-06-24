"""Behavioral tests for the `serving-llms-on-epyc` skill.

Run locally (needs the `claude` CLI authenticated; the agent does not actually
launch a server in the judge's sandbox, so this grades the *plan/behavior*, not
a live endpoint):

    pytest eval/behavioral/tests/test_serving_llms_on_epyc.py -s

`logs_contains` is deterministic; `should` / `should_not` are graded by an LLM
judge over the captured evidence (tool calls + outputs), so the agent's prose
cannot fake a pass.
"""

from harness import claude


def test_serve_model_on_epyc():
    with claude("sonnet", skill="serving-llms-on-epyc") as agent:
        run = agent.prompt(
            "Serve Qwen/Qwen3-0.6B on this AMD EPYC box with vLLM and zentorch. "
            "Use the default settings."
        )

        # Programmatic expectation: the skill was actually loaded.
        run.logs_contains("serving-llms-on-epyc")

        # Positive behavioral expectations (the state machine).
        run.should("Detect the CPU and confirm it is an AMD EPYC host before serving (e.g. runs detect.py)")
        run.should("Validate the container runtime (docker or podman) or the conda path before launching (e.g. runs validate.py)")
        run.should("Take validate.py's environment advisories into account -- the tcmalloc / OpenMP (LD_PRELOAD) perf-library recommendation and, when the image is already pulled, the in-image vllm+zentorch check -- surfacing any that apply")
        run.should("Check that vLLM supports the model before serving (e.g. runs check_model.py), rather than refusing it just for being multimodal")
        run.should("Check that the model fits in host RAM (e.g. runs estimate_memory.py)")
        run.should("Size CPU threads / KV-cache from the hardware rather than using a fixed guess (e.g. runs cpu_tune.py)")
        run.should("Present a sized plan and ask the user to confirm before launching the server")
        run.should("Plan to launch with 'vllm serve' and poll until /health is healthy")

        # Negative behavioral expectations (the explicit Don'ts).
        run.should_not("Pass '--device cpu' to vllm serve")
        run.should_not("Launch the server before the user has confirmed the plan")
        run.should_not("Enter a debugging loop or retry after a launch failure")
        run.should_not("Attempt GPU, ROCm, or Instinct serving")
