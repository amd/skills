"""Behavioral tests for the `vllm-multiinstance` skill.

Run locally (needs the `claude` CLI authenticated and on a network that can
reach the API):

    pytest eval/behavioral/tests/test_vllm_multiinstance.py -s

A real sweep needs podman + ansible + a model + many physical cores and runs
for ~10+ minutes, so these tests do NOT launch the stack. Every prompt is
scoped to planning / explanation ("Do not run anything") and asserts the
agent's *decisions and guardrails* drawn from the skill: sweep sizing, the
"read scores from guidellm.log, not benchmarks.json" rule, and the host
preflight / rootless fail-fast behavior the harness implements.

Each check on `run` prints a `[PASS]`/`[FAIL]` line and raises on failure.
`logs_contains` is deterministic; `should` / `should_not` are graded by an
LLM judge over the captured evidence.
"""

from harness import claude

_NO_RUN = "Do not run any containers, podman, ansible, or scripts -- just answer."


def test_skill_activates_and_sizes_the_sweep():
    with claude("sonnet", skill="vllm-multiinstance") as agent:
        run = agent.prompt(
            "I want to benchmark a vLLM CPU image with the vllm-multiinstance "
            "skill on a single-socket AMD EPYC with 128 physical cores. How "
            f"many vLLM instances should I run and which cores does each get? {_NO_RUN}"
        )

        run.logs_contains("vllm-multiinstance")

        run.should("Recommend running 3 vLLM instances for a 128-physical-core host")
        run.should(
            "Pin the three instances to cores 32-63, 64-95, and 96-127 "
            "(CORES_PER_INSTANCE=32, all on one socket)"
        )
        run.should_not(
            "Spread the instances across both sockets or use a "
            "CORES_PER_INSTANCE other than 32"
        )


def test_reads_throughput_from_guidellm_log_not_json():
    with claude("sonnet", skill="vllm-multiinstance") as agent:
        run = agent.prompt(
            "Using the vllm-multiinstance skill: after a run completes, where do "
            "I read the server throughput, and which number should I NOT trust? "
            f"{_NO_RUN}"
        )

        run.logs_contains("vllm-multiinstance")

        run.should(
            "Read server throughput from guidellm.log (the 'Server Throughput "
            "Statistics' table), which is the server-aggregate number"
        )
        run.should_not(
            "Recommend reporting requests_per_second or output_tokens_per_second "
            "from benchmarks.json as the server throughput"
        )


def test_host_preflight_fails_fast_on_blockers():
    with claude("sonnet", skill="vllm-multiinstance") as agent:
        run = agent.prompt(
            "Using the vllm-multiinstance skill: I'm on a rootless podman 3.4.4 / "
            "CNI 0.9.1 host. What host-level problems will the harness catch "
            "before a long run, and how does it avoid the 20-minute health-wait "
            f"hang? {_NO_RUN}"
        )

        run.logs_contains("vllm-multiinstance")

        run.should(
            "Mention the host preflight (check-host.sh) catches an unresolvable "
            "image short-name, missing rootless cgroup cpuset delegation, and a "
            "CNI cniVersion mismatch"
        )
        run.should(
            "Explain that the harness exits early / fails fast with actionable "
            "guidance instead of hanging the full health-check timeout"
        )


def test_image_short_name_remediation():
    with claude("sonnet", skill="vllm-multiinstance") as agent:
        run = agent.prompt(
            "Using the vllm-multiinstance skill: the default image "
            "amdih/zendnn_zentorch:... won't resolve on my host (no "
            f"unqualified-search registries). What should I do? {_NO_RUN}"
        )

        run.logs_contains("vllm-multiinstance")

        run.should(
            "Recommend using a fully-qualified image name, e.g. prefixing it "
            "with docker.io/ (or pre-pulling that fully-qualified image)"
        )


def test_rootless_runs_without_passwordless_sudo():
    with claude("sonnet", skill="vllm-multiinstance") as agent:
        run = agent.prompt(
            "Using the vllm-multiinstance skill: my host has no passwordless "
            "sudo. Can I still run the guidellm benchmark, and how does the "
            f"harness handle it? {_NO_RUN}"
        )

        run.logs_contains("vllm-multiinstance")

        run.should(
            "Explain the harness can run ansible (incl. guidellm) rootless via "
            "ansible_become=false -- auto-detected when passwordless sudo is "
            "missing, or forced with --no-become / ANSIBLE_NO_BECOME=1"
        )
        run.should_not(
            "Claim the benchmark simply cannot run without passwordless sudo"
        )
