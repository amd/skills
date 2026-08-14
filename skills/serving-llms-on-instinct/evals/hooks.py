# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Docker teardown for the `serving-llms-on-instinct` behavior case.

Prompts and expectations live in ``evals.json``; this file holds only the
environment plumbing the dataset format cannot express. The runner calls
``setup`` before each case and ``teardown`` after it.

The eval runner cleans up the agent's temp workspace but knows nothing about
Docker, so on a shared MI300X runner we remove the container the agent
launched. Cleanup also runs *before* the case, so a run starts from a clean
machine rather than inheriting a half-dead endpoint from a crashed earlier one.
A leftover is reported and not fatal: the dataset grades the endpoint the case
ends with, so an agent that reuses a healthy container still answers correctly.
"""

from __future__ import annotations

import re
import shutil
import subprocess

MODEL_ID = "Qwen/Qwen3-0.6B"

# Matches the model however it shows up in a container's name, image, or
# command: vllm-qwen3-0.6b, vllm-qwen3-0-6b, --model Qwen/Qwen3-0.6B, ...
_MODEL_MARKER = re.compile(r"qwen3[-_./]?0[-_.]?6b")


def _docker_rows() -> list[tuple[str, str]]:
    """Return ``(container_id, searchable_text)`` for every container."""
    docker = shutil.which("docker")
    if not docker:
        return []
    try:
        out = subprocess.run(
            [docker, "ps", "-a", "--no-trunc",
             "--format", "{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Command}}"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []

    rows = []
    for line in out.splitlines():
        container_id, _, rest = line.partition("\t")
        if container_id.strip():
            rows.append((container_id.strip(), rest.lower()))
    return rows


def _cleanup_test_containers() -> None:
    """Best-effort removal of vLLM containers serving the tiny test model.

    Matching is scoped to that model -- by name, image, or the ``--model``
    argument in the container command -- so a shared runner's other endpoints
    are never touched, while containers the agent named differently than the
    skill's template suggests are still caught.
    """
    docker = shutil.which("docker")
    if not docker:
        return
    ids = [cid for cid, text in _docker_rows() if _MODEL_MARKER.search(text)]
    if not ids:
        return
    print(f"  [cleanup] removing {len(ids)} stale {MODEL_ID} container(s)", flush=True)
    try:
        subprocess.run([docker, "rm", "-f", *ids], capture_output=True, timeout=120)
    except (subprocess.SubprocessError, OSError):
        pass


def setup(workspace, case, ctx) -> None:
    """Clear stale test containers so the case starts from a clean machine."""
    _cleanup_test_containers()
    leftover = [cid for cid, text in _docker_rows() if _MODEL_MARKER.search(text)]
    if leftover:
        print(
            f"  [cleanup] warning: {MODEL_ID} container(s) {leftover} survived "
            "removal; the agent may serve this case from the running endpoint",
            flush=True,
        )


def teardown(workspace, case, ctx) -> None:
    _cleanup_test_containers()
