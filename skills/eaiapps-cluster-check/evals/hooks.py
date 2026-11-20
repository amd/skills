# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""A stand-in cluster for the behavioral cases.

Prompts and expectations live in ``evals.json``. This file holds only what the
dataset format cannot say: that each case needs a cluster, and which one.

Every graded case here asks the agent to check a cluster, and ``SKILL.md`` tells
it to confirm the kubeconfig context and then run
``scripts/environment-check.sh``. The runner stages the skill into a temporary
workspace and nothing else, so without this hook there is no cluster to check:
on a clean runner ``kubectl`` finds no kubeconfig and the checker can only answer
CANNOT REACH CLUSTER, which grades none of the expectations. Worse, on a
self-hosted runner inside a cluster there *is* a kubeconfig, and the run would
grade whatever that cluster happens to look like today.

So the hook supplies the cluster:

* **The probe scripts**, copied to the path ``SKILL.md`` names, because the agent
  is told to run them there and the workspace holds the skill under
  ``.claude/skills/`` instead.
* **A fixture-backed ``kubectl``** on PATH, reading a transcript assembled from
  ``fixtures/healthy.json`` and the case's own overlay. The checker shells out to
  ``kubectl`` for everything, so a transcript is the whole cluster.

The stubs live outside the workspace, and ``KUBECONFIG`` points at a file naming
no cluster. Both are deliberate, and both come from
``guided-workbench-install/evals/hooks.py``, which solved this first. A ``bin``
directory inside the agent's own tree is something it reads, and it then takes
the stub for the fault it was asked to diagnose. The run has to look like a
cluster that is not reachable any other way.

The fixtures are not all healthy, or there would be nothing to diagnose. Each
one is the cluster its prompt describes, and each was checked by running the
checker against it:

* ``downloads_hang`` -- Cilium MTU unpinned and no GPU series arriving, on an
  otherwise clean cluster: PROBLEMS, with the pod checks passing, which is the
  prompt's "nothing looks wrong in the pod list".
* ``not_installed_yet`` -- infrastructure healthy, nothing installed on top:
  INCONCLUSIVE and no warning anywhere, every post-install check saying it was
  not deployed yet, so "those are inconclusive, not failures" is a thing the
  run can be judged on.
* ``read_only`` -- several reads forbidden: INCONCLUSIVE and no warning
  anywhere, each blocked check naming what it could not read, which is "not
  proven healthy" rather than "healthy".

The verdicts are named, not the counts. A fixture declares only what differs
from the baseline, so a check added upstream lands in these runs too, and a
count here would be wrong the first time that happened without anything being
broken.

The harness runs a skill's cases one at a time in one process, and starts the
agent with a copy of ``os.environ`` taken at that moment, so prepending to the
process-global PATH is the only way to put a ``kubectl`` in front of it. That is
safe exactly while cases stay serial, so ``setup`` refuses to start a second case
while one is still staged rather than leaking one case's cluster into the next.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = SKILL_ROOT / "evals" / "fixtures"

# Where SKILL.md tells the agent the probe scripts live.
SCRIPTS_AT = Path(".agents/skills/cluster-environment-check/scripts")

# Which cluster each case gets. A case not named here gets the healthy baseline,
# so a new prompt grades against a clean cluster rather than against whichever
# fault the previous case happened to need.
CASE_FIXTURES = {
    "diagnose-silent-platform-problems": "downloads_hang",
    "pre-install-readiness": "not_installed_yet",
    "limited-access-not-proven-healthy": "read_only",
}

# Set between setup() and teardown(). Guards the PATH assumption above.
_staged: str | None = None


def _transcript(name: str) -> str:
    """The healthy baseline with ``name`` laid over it, tokens left unresolved.

    The same composition the repository's own suite does, for the same reason: a fault
    fixture says only what differs. The ``__NOW__`` tokens stay in, because the stub
    resolves them per call and the ages several checks judge by are only meaningful
    against the clock at the moment of the read.
    """
    transcript = json.loads((FIXTURES / "healthy.json").read_text(encoding="utf-8"))
    if name != "healthy":
        transcript.update(json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8")))
    return json.dumps(transcript)


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

    bin_dir = Path(tempfile.mkdtemp(prefix="cluster-"))
    kubectl = bin_dir / "kubectl"
    shutil.copy2(FIXTURES / "fake_kubectl.py", kubectl)
    kubectl.chmod(kubectl.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    fixture = bin_dir / "fixture.json"
    fixture.write_text(_transcript(CASE_FIXTURES.get(case.id, "healthy")), encoding="utf-8")

    # A kubeconfig naming no cluster, so a kubectl found anywhere else cannot
    # reach a real one and quietly answer for this fixture.
    kubeconfig = bin_dir / "kubeconfig"
    kubeconfig.write_text(
        "apiVersion: v1\nkind: Config\nclusters: []\ncontexts: []\nusers: []\n", encoding="utf-8"
    )

    ctx["_bin_dir"] = str(bin_dir)
    keys = ("PATH", "KUBECONFIG", "CLUSTER_CHECK_FIXTURE", "CLUSTER_CHECK_FIXTURE_STRICT")
    ctx["_env_before"] = {k: os.environ.get(k) for k in keys}
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ["KUBECONFIG"] = str(kubeconfig)
    os.environ["CLUSTER_CHECK_FIXTURE"] = str(fixture)
    # An agent reads the cluster around the checker, and the transcript only
    # models what the checker asks. Strict mode would answer those with exit 99
    # on stderr, which is louder than anything else in the run and would be what
    # the judge grades. See fixtures/fake_kubectl.py.
    os.environ["CLUSTER_CHECK_FIXTURE_STRICT"] = "0"
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
