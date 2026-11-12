# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Cluster and script plumbing for the behavioral cases.

Prompts and expectations live in ``evals.json``. This file holds only what the
dataset format cannot say.

The runner stages each case in a temporary workspace holding the skill and the
case's ``workspace`` fixture, which here is a checkout-shaped tree: the real
``helm/INSTALL.md`` and the chart directories it names. Two things are still
missing from that workspace and neither can be expressed as a fixture file:

* **The skill's own probe scripts.** ``SKILL.md`` invokes them at the path they
  occupy in the product repo. Copying them there from the installed skill keeps
  the documented command working without a second copy that could drift.
* **A cluster.** ``kubectl`` and ``helm`` are stubbed, because a graded run has
  no cluster and a skill that cannot probe one stops at preflight, which grades
  nothing about what it does afterwards.

The harness runs a skill's cases one at a time in a single process, and the
agent is started with a copy of ``os.environ`` taken at that moment, so
prepending to the process-global PATH is the only way to put a ``kubectl`` in
front of it. That is safe exactly while cases stay serial, so ``setup`` refuses
to start a second case while one is still staged rather than leaking one case's
cluster into the next.

The stubs live outside the workspace and ``KUBECONFIG`` is pointed at a file
that names no cluster. Both are deliberate. A ``bin`` directory sitting in the
agent's own tree is something it reads, and on a developer machine with a real
``kubectl`` it then takes the stub for the fault it was asked to diagnose,
drops the directory from PATH and grades a narrative about the harness. The
run has to look like a cluster that is simply not reachable any other way.

The stubs do not pretend everything is well. ``install-aiwb-combined`` gets a
cluster that passes preflight and then fails to answer one verification query,
so "report an inconclusive result rather than success" is a thing the run can
actually be judged on. ``troubleshoot-aiwb-install`` gets a deployment stuck at
zero ready replicas, which is the fault its prompt describes.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent

# Where SKILL.md tells the agent the probe scripts live.
SCRIPTS_AT = Path(".agents/skills/guided-workbench-install/scripts")

# Reachable cluster, permissions, storage, operators and schedulable GPU nodes:
# enough for preflight-check.sh to reach RESULT: READY. Kept in step with the
# PREFLIGHT_BASE arms in tests/run-tests.sh, which assert that same verdict.
PREFLIGHT_READY = """
  *"cluster-info") exit 0 ;;
  *"config current-context") echo ctx ;;
  *"auth can-i create namespaces") echo yes ;;
  *"get storageclass"*) echo default ;;
  *"get crd aimclusterruntimeconfigs.aim.eai.amd.com") exit 0 ;;
  *"get crd"*) exit 0 ;;
  *"get gateway"*) exit 0 ;;
  *"get clustersecretstores.external-secrets.io"*"Ready"*) echo True ;;
  *"get deploy -A"*"API_KEY_AUTH_ANNOTATION"*) echo "envoy-ai-gateway-system/ai-gateway-discovery=True|API_KEY_AUTH_ANNOTATION|WORKLOADS_AUTHZ_PORT" ;;
  *"get nodes"*unschedulable*) printf 'n1=/True/8\\nn2=/True/8\\nn3=/True/8\\n' ;;
  *"get nodes"*) echo 8 ;;
"""

# Routes, databases and the message broker all answer healthily.
WORKLOADS_HEALTHY = """
  *"get httproutes"*Accepted*) echo "r=True," ;;
  *"get httproutes"*"metadata.name"*) echo "r" ;;
  *"get clusters.postgresql.cnpg.io"*"status.phase"*) echo "db=Cluster in healthy state" ;;
  *"get clusters.postgresql.cnpg.io"*) echo "db" ;;
  *"get rabbitmqclusters"*AllReplicasReady*) echo "rmq=True" ;;
  *"get rabbitmqclusters"*"metadata.name"*) echo "rmq" ;;
"""

DEPLOYMENTS_READY = """
  *"get deploy"*"{.spec.replicas}"*) echo 1 ;;
  *"get deploy"*"{.status.readyReplicas}"*) echo 1 ;;
  *"get deploy"*) exit 0 ;;
"""

# One replica asked for, none ready: the symptom `troubleshoot-aiwb-install`
# describes, and a state verification must not call healthy.
DEPLOYMENT_NEVER_READY = """
  *"get deploy"*"{.spec.replicas}"*) echo 1 ;;
  *"get deploy"*"{.status.readyReplicas}"*) echo 0 ;;
  *"get deploy"*) exit 0 ;;
"""

# A cause the skill can actually find and name. A deployment stuck at zero
# ready replicas with nothing behind it has no root cause to report, and an
# agent that refuses to invent one is behaving correctly, so grading "name what
# is wrong and give a concrete fix" against that is grading the fixture's
# silence. The image tag is wrong, which the pod list, the events and the
# describe output all say, and which is fixable from the chart values the
# runbook sets.
IMAGE_TAG_WRONG = """
  *"get pod"*jsonpath*image*) echo "amdenterpriseai/aiwb-api:v9.9.9" ;;
  *"get pod"*jsonpath*reason*) echo "ImagePullBackOff" ;;
  *"get pod"*-o?json*) cat <<'JSON'
{
  "apiVersion": "v1",
  "kind": "List",
  "items": [
    {
      "apiVersion": "v1",
      "kind": "Pod",
      "metadata": { "name": "aiwb-api-7d9c4b6f84-2xkp", "namespace": "aiwb" },
      "spec": { "containers": [ { "name": "aiwb-api", "image": "amdenterpriseai/aiwb-api:v9.9.9" } ] },
      "status": {
        "phase": "Pending",
        "containerStatuses": [
          {
            "name": "aiwb-api",
            "ready": false,
            "image": "amdenterpriseai/aiwb-api:v9.9.9",
            "state": { "waiting": { "reason": "ImagePullBackOff", "message": "Back-off pulling image \\"amdenterpriseai/aiwb-api:v9.9.9\\"" } }
          }
        ]
      }
    }
  ]
}
JSON
  ;;
  *"get pod"*) printf 'NAME                        READY   STATUS             RESTARTS   AGE\naiwb-api-7d9c4b6f84-2xkp    0/1     ImagePullBackOff   0          6m\n' ;;
  *"describe pod"*) printf 'Events:\n  Type     Reason     Age                 From     Message\n  ----     ------     ----                ----     -------\n  Normal   Pulling    6m (x4 over 6m)     kubelet  Pulling image "amdenterpriseai/aiwb-api:v9.9.9"\n  Warning  Failed     6m (x4 over 6m)     kubelet  Failed to pull image "amdenterpriseai/aiwb-api:v9.9.9": manifest unknown: manifest tagged by "v9.9.9" is not found\n  Warning  Failed     5m (x6 over 6m)     kubelet  Error: ImagePullBackOff\n' ;;
  *"get events"*) printf 'LAST SEEN   TYPE      REASON   OBJECT                           MESSAGE\n6m          Warning   Failed   pod/aiwb-api-7d9c4b6f84-2xkp     Failed to pull image "amdenterpriseai/aiwb-api:v9.9.9": manifest unknown\n' ;;
  *"logs"*) echo "Error from server (BadRequest): container \\"aiwb-api\\" in pod is waiting to start: image can't be pulled" >&2; exit 1 ;;
"""

# The onboarding Job's status query errors rather than answering. Health cannot
# be confirmed or denied, which is the case the skill must not round up to
# success.
ONBOARDING_UNANSWERABLE = """
  *"get job"*"status.succeeded"*) echo "Error from server: etcdserver: request timed out" >&2; exit 1 ;;
"""

# Namespaces absent: a first install rather than a repair.
FRESH_NAMESPACES = """
  *"get ns"*) echo "Error from server (NotFound): namespace not found" >&2; exit 1 ;;
"""

KUBECTL = """#!/usr/bin/env bash
case "$*" in
{arms}
  *) exit 0 ;;
esac
"""

# A fresh cluster: helm knows the chart repositories and has deployed nothing.
HELM_NOTHING_INSTALLED = """
  list) ;;
  status) echo "Error: release: not found" >&2; exit 1 ;;
"""

# An install that ran and left releases behind, with the bad tag recorded in
# the values the agent can read back. Without this an agent asking how far the
# install got finds no release at all, which says the opposite of the prompt.
HELM_INSTALLED_WITH_BAD_TAG = """
  list)
    case "$*" in
      *json*) printf '[{"name":"aiwb","namespace":"aiwb","revision":"1","status":"deployed","chart":"aiwb-2.1.0"},{"name":"airm","namespace":"airm","revision":"1","status":"deployed","chart":"airm-2.1.0"}]\n' ;;
      *) printf 'NAME\tNAMESPACE\tREVISION\tSTATUS\t\tCHART\naiwb\taiwb\t\t1\t\tdeployed\taiwb-2.1.0\nairm\tairm\t\t1\t\tdeployed\tairm-2.1.0\n' ;;
    esac
    ;;
  status) printf 'NAME: aiwb\nLAST DEPLOYED: Thu Nov  6 09:14:02 2026\nNAMESPACE: aiwb\nSTATUS: deployed\nREVISION: 1\n' ;;
  get)
    case "$*" in
      *values*json*) printf '{"backend":{"image":{"repository":"amdenterpriseai/aiwb-api","tag":"v9.9.9"}}}\n' ;;
      *values*) printf 'backend:\n  image:\n    repository: amdenterpriseai/aiwb-api\n    tag: v9.9.9\n' ;;
      *) ;;
    esac
    ;;
"""

HELM = """#!/usr/bin/env bash
# Enough of helm to be recognised, to let an install command return, and to
# answer for what is already installed. Nothing is deployed; the stub kubectl
# decides what the cluster then looks like.
case "$1" in
  version) echo "v3.16.2" ;;
  install|upgrade) echo "STATUS: deployed" ;;
{arms}
  *) ;;
esac
exit 0
"""

HELM_ARMS = {
    "install-aiwb-combined": HELM_NOTHING_INSTALLED,
    "troubleshoot-aiwb-install": HELM_INSTALLED_WITH_BAD_TAG,
}

CASE_ARMS = {
    "install-aiwb-combined": (
        FRESH_NAMESPACES + PREFLIGHT_READY + WORKLOADS_HEALTHY
        + DEPLOYMENTS_READY + ONBOARDING_UNANSWERABLE
    ),
    "troubleshoot-aiwb-install": (
        IMAGE_TAG_WRONG + PREFLIGHT_READY + WORKLOADS_HEALTHY + DEPLOYMENT_NEVER_READY
    ),
}

# kubectl's own default for an absent resource: nothing on stdout, exit 0. The
# probe scripts read that as "not present" rather than as an error, so an arm
# this fixture does not model degrades to absent instead of failing the run.
# Set between setup() and teardown(). Guards the PATH assumption above.
_staged: str | None = None


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def setup(workspace, case, ctx):  # noqa: ARG001 -- runner passes ctx for placeholders
    """Stage the probe scripts and a stubbed cluster, then put it on PATH."""
    global _staged
    if _staged is not None:
        raise RuntimeError(
            f"case '{_staged}' is still staged, so this harness is running cases "
            f"concurrently. The PATH these hooks set is process-global and would "
            f"leak one case's cluster into '{case.id}'."
        )
    _staged = case.id
    workspace = Path(workspace)

    scripts_dst = workspace / SCRIPTS_AT
    scripts_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SKILL_ROOT / "scripts", scripts_dst, dirs_exist_ok=True)
    for script in scripts_dst.glob("*.sh"):
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    arms = CASE_ARMS.get(case.id, PREFLIGHT_READY + WORKLOADS_HEALTHY + DEPLOYMENTS_READY)
    bin_dir = Path(tempfile.mkdtemp(prefix="cluster-"))
    _write_executable(bin_dir / "kubectl", KUBECTL.format(arms=arms.strip("\n")))
    helm_arms = HELM_ARMS.get(case.id, HELM_NOTHING_INSTALLED)
    _write_executable(bin_dir / "helm", HELM.format(arms=helm_arms.strip("\n")))

    # A kubeconfig naming no cluster, so a kubectl found anywhere else cannot
    # reach a real one and quietly answer for this fixture.
    kubeconfig = bin_dir / "kubeconfig"
    kubeconfig.write_text("apiVersion: v1\nkind: Config\nclusters: []\ncontexts: []\nusers: []\n", encoding="utf-8")

    ctx["_bin_dir"] = str(bin_dir)
    ctx["_env_before"] = {k: os.environ.get(k) for k in ("PATH", "KUBECONFIG")}
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ["KUBECONFIG"] = str(kubeconfig)
    return {}


def teardown(workspace, case, ctx):  # noqa: ARG001 -- symmetry with setup
    """Undo the environment and remove the stubs the case ran against."""
    global _staged
    if _staged == case.id:
        _staged = None
    for key, value in (ctx.get("_env_before") or {}).items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    bin_dir = ctx.get("_bin_dir")
    if bin_dir:
        shutil.rmtree(bin_dir, ignore_errors=True)
