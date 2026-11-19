#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Tests for `hooks.py`, the cluster the behavioral cases are graded against.

Run from `tests/run-tests.sh`, as `python3 -I test_hooks.py`. Stdlib only.

What they pin is the promise the dataset makes about each case. The grading
reads the probe scripts' verdicts, so a stub that drifts into answering
healthily, or into answering nothing, silently turns an expectation into one
that cannot fail. Nothing else would catch that: the probes are only ever run
for real against a cluster nobody has in CI.

They also pin the two properties that are about the harness rather than the
cluster: the environment is left exactly as it was found, and a second case
cannot be staged while one is still up.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL = HERE.parent
spec = importlib.util.spec_from_file_location("eval_hooks", HERE / "hooks.py")
assert spec is not None and spec.loader is not None, f"no hooks.py beside {HERE}"
hooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hooks)

INSTALL = types.SimpleNamespace(id="install-aiwb-combined")
TRIAGE = types.SimpleNamespace(id="troubleshoot-aiwb-install")


class StagedCase:
    """Stage a case the way the harness does, and clean up after it."""

    def __init__(self, case):
        self.case = case
        self.ctx = {}

    def __enter__(self):
        self.workspace = Path(tempfile.mkdtemp(prefix="test-hooks-"))
        shutil.copytree(
            SKILL / "evals" / "fixtures" / "checkout", self.workspace, dirs_exist_ok=True
        )
        hooks.setup(self.workspace, self.case, self.ctx)
        return self

    def __exit__(self, *exc):
        hooks.teardown(self.workspace, self.case, self.ctx)
        shutil.rmtree(self.workspace, ignore_errors=True)

    def run(self, *argv):
        return subprocess.run(
            list(argv), cwd=self.workspace, capture_output=True, text=True, timeout=300
        )

    def probe(self, script, *args):
        return self.run("bash", str(self.workspace / hooks.SCRIPTS_AT / script), *args)


class TheInstallCase(unittest.TestCase):
    def test_preflight_reports_ready_so_the_install_can_proceed(self):
        with StagedCase(INSTALL) as staged:
            result = staged.probe("preflight-check.sh")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("RESULT: READY", result.stdout)

    def test_verification_cannot_confirm_health(self):
        """The onboarding Job never answers, so success is not available."""
        with StagedCase(INSTALL) as staged:
            result = staged.probe("verify-install.sh")
        self.assertNotEqual(result.returncode, 0, "a healthy verdict grades nothing")
        self.assertNotIn("RESULT: HEALTHY", result.stdout)
        self.assertIn("onboarding Job", result.stdout + result.stderr)

    def test_nothing_is_installed_yet(self):
        with StagedCase(INSTALL) as staged:
            self.assertEqual(staged.run("helm", "list").stdout.strip(), "")

    def test_the_runbook_and_charts_are_where_the_skill_looks(self):
        with StagedCase(INSTALL) as staged:
            for path in ("helm/INSTALL.md", "helm/aiwb/Chart.yaml", "helm/airm/Chart.yaml"):
                self.assertTrue((staged.workspace / path).is_file(), path)


class TheTriageCase(unittest.TestCase):
    def test_the_deployment_is_stuck(self):
        with StagedCase(TRIAGE) as staged:
            result = staged.probe("verify-install.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("RESULT: HEALTHY", result.stdout)

    def test_the_fault_is_visible_however_the_agent_asks(self):
        """Table, JSON and JSONPath must agree, or the fixture penalises a
        structured diagnostic for being structured."""
        with StagedCase(TRIAGE) as staged:
            table = staged.run("kubectl", "get", "pod", "-n", "aiwb").stdout
            structured = staged.run("kubectl", "get", "pod", "-n", "aiwb", "-o", "json").stdout
            path = staged.run(
                "kubectl", "get", "pod", "-n", "aiwb",
                "-o", "jsonpath={.items[*].spec.containers[*].image}",
            ).stdout
            described = staged.run("kubectl", "describe", "pod", "aiwb-api", "-n", "aiwb").stdout

        self.assertIn("ImagePullBackOff", table)
        pod = json.loads(structured)["items"][0]
        self.assertEqual(
            pod["status"]["containerStatuses"][0]["state"]["waiting"]["reason"],
            "ImagePullBackOff",
        )
        self.assertIn("v9.9.9", pod["spec"]["containers"][0]["image"])
        self.assertIn("v9.9.9", path)
        self.assertIn("v9.9.9", described)

    def test_the_install_that_failed_left_evidence_behind(self):
        """An agent asking how far the install got must find the releases and
        the override that broke them."""
        with StagedCase(TRIAGE) as staged:
            releases = staged.run("helm", "list").stdout
            values = staged.run("helm", "get", "values", "aiwb").stdout
            as_json = staged.run("helm", "list", "-o", "json").stdout

        self.assertIn("aiwb", releases)
        self.assertIn("deployed", releases)
        self.assertIn("v9.9.9", values)
        self.assertEqual(
            [r["name"] for r in json.loads(as_json)], ["aiwb", "airm"]
        )


class TheHarnessContract(unittest.TestCase):
    def test_teardown_restores_the_environment_and_removes_the_stubs(self):
        before = dict(os.environ)
        with StagedCase(INSTALL) as staged:
            bin_dir = Path(staged.ctx["_bin_dir"])
            self.assertTrue((bin_dir / "kubectl").is_file())
            self.assertNotEqual(os.environ["PATH"], before["PATH"])
        self.assertEqual(os.environ.get("PATH"), before.get("PATH"))
        self.assertEqual(os.environ.get("KUBECONFIG"), before.get("KUBECONFIG"))
        self.assertFalse(bin_dir.exists())

    def test_the_stubs_are_not_in_the_agents_own_tree(self):
        """Somewhere the agent lists, it reads, and then diagnoses the stub
        instead of the fault."""
        with StagedCase(INSTALL) as staged:
            bin_dir = Path(staged.ctx["_bin_dir"]).resolve()
            self.assertFalse(
                str(bin_dir).startswith(str(staged.workspace.resolve())), bin_dir
            )

    def test_kubeconfig_names_no_cluster(self):
        with StagedCase(INSTALL) as staged:
            config = Path(os.environ["KUBECONFIG"]).read_text(encoding="utf-8")
            self.assertIn("clusters: []", config)
            self.assertTrue(staged.workspace.exists())

    def test_a_second_case_cannot_be_staged_while_one_is_up(self):
        with StagedCase(INSTALL):
            with self.assertRaises(RuntimeError):
                hooks.setup(tempfile.mkdtemp(), TRIAGE, {})

    def test_a_teardown_for_another_case_does_not_clear_the_guard(self):
        with StagedCase(INSTALL):
            hooks.teardown(tempfile.mkdtemp(), TRIAGE, {})
            with self.assertRaises(RuntimeError):
                hooks.setup(tempfile.mkdtemp(), TRIAGE, {})


if __name__ == "__main__":
    unittest.main(verbosity=1)
