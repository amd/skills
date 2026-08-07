"""Deterministic tests for amd-skill-finder."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "find_skills.py"
SPEC = importlib.util.spec_from_file_location("amd_skill_finder", SCRIPT)
assert SPEC and SPEC.loader
finder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finder)


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = finder.load_registry()
        cls.projects = {project["id"]: project for project in cls.registry["projects"]}

    def test_requested_projects_and_canonical_repositories_are_present(self):
        required = {
            "quark": "amd/Quark",
            "mori": "ROCm/mori",
            "hyperloom": "AMD-AGI/Hyperloom",
            "pytorch": "pytorch/pytorch",
            "jax": "jax-ml/jax",
            "triton": "triton-lang/triton",
            "torchtitan": "pytorch/torchtitan",
            "lmcache": "LMCache/LMCache",
            "mooncake": "kvcache-ai/Mooncake",
            "tilelang": "tile-ai/tilelang",
            "nixl": "ai-dynamo/nixl",
            "miles": "radixark/miles",
            "verl": "verl-project/verl",
            "vime": "vllm-project/vime",
        }
        for project_id, repository in required.items():
            with self.subTest(project=project_id):
                repos = {
                    entry["repo"] for entry in self.projects[project_id]["repositories"]
                }
                self.assertIn(repository, repos)

    def test_dynamo_repository_is_excluded(self):
        excluded = {repo.lower() for repo in self.registry["excluded_repositories"]}
        all_repositories = {
            repo["repo"].lower()
            for project in self.registry["projects"]
            for repo in project["repositories"]
        }
        self.assertIn("ai-dynamo/dynamo", excluded)
        self.assertNotIn("ai-dynamo/dynamo", all_repositories)
        self.assertIn("ai-dynamo/nixl", all_repositories)

    def test_project_ids_and_repositories_are_unique(self):
        ids = [project["id"] for project in self.registry["projects"]]
        self.assertEqual(len(ids), len(set(ids)))
        for project in self.registry["projects"]:
            repos = [repo["repo"].lower() for repo in project["repositories"]]
            self.assertEqual(len(repos), len(set(repos)), project["id"])

    def test_fork_pairs_are_grouped(self):
        expected = {
            "pytorch": {"ROCm/pytorch", "pytorch/pytorch"},
            "jax": {"ROCm/jax", "jax-ml/jax"},
            "triton": {"ROCm/triton", "triton-lang/triton"},
        }
        for project_id, repositories in expected.items():
            actual = {
                repo["repo"] for repo in self.projects[project_id]["repositories"]
            }
            self.assertEqual(actual, repositories)


class RoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = finder.load_registry()

    def test_kv_transfer_routes_to_official_mori_first(self):
        results = finder.route_projects(
            "disaggregated KV-cache transfer over RDMA on MI355X",
            self.registry,
            limit=6,
        )
        ids = [result["id"] for result in results]
        self.assertEqual(ids[0], "mori")
        self.assertTrue({"mori", "lmcache", "mooncake", "nixl"}.issubset(ids))
        self.assertNotIn("triton", ids)
        self.assertTrue(all(result["installable"] is False for result in results))

    def test_quark_quantization_routes_to_official_quark_first(self):
        self.assertTrue(finder.has_amd_signal("Quark PTQ", self.registry))
        results = finder.route_projects(
            "Use AMD Quark PTQ to quantize a Qwen model to FP8",
            self.registry,
            limit=5,
        )
        self.assertEqual(results[0]["id"], "quark")
        self.assertEqual(results[0]["repositories"][0]["repo"], "amd/Quark")
        self.assertEqual(results[0]["repositories"][0]["tier"], "amd-official")

    def test_post_training_routes_to_three_requested_frameworks(self):
        results = finder.route_projects(
            "RL post-training with vLLM on ROCm",
            self.registry,
            limit=10,
        )
        ids = {result["id"] for result in results}
        self.assertTrue({"miles", "verl", "vime"}.issubset(ids))

    def test_triton_is_one_project_with_two_repositories(self):
        results = finder.route_projects(
            "optimize a Triton kernel on MI355X",
            self.registry,
            limit=6,
        )
        triton = next(result for result in results if result["id"] == "triton")
        self.assertEqual(
            [repo["repo"] for repo in triton["repositories"]],
            ["ROCm/triton", "triton-lang/triton"],
        )

    def test_generic_frontend_optimization_does_not_route(self):
        results = finder.route_projects(
            "Optimize my React page load and reduce the JavaScript bundle size",
            self.registry,
        )
        self.assertEqual(results, [])

    def test_amd_scope_excludes_upstream_only_projects(self):
        results = finder.route_projects(
            "RL post-training with vLLM on ROCm",
            self.registry,
            scope="amd",
            limit=20,
        )
        ids = {result["id"] for result in results}
        self.assertFalse({"miles", "verl", "vime"} & ids)


class CatalogAndLiveSearchTests(unittest.TestCase):
    def test_active_finder_does_not_recommend_itself(self):
        result = finder.find(
            "optimize a Triton kernel on MI355X",
            offline=True,
        )
        names = {
            entry["name"]
            for key in ("installed_skills", "catalog_skills")
            for entry in result[key]
        }
        self.assertNotIn("amd-skill-finder", names)

    def test_catalog_browse_lists_published_skills(self):
        result = finder.find("Show AMD skills", offline=True, scope="catalog")
        names = {entry["name"] for entry in result["catalog_skills"]}
        self.assertIn("serving-llms-on-instinct", names)
        self.assertNotIn("amd-skill-finder", names)

    def test_installed_skill_ranks_above_catalog_copy(self):
        entry = {
            "name": "serving-llms-on-instinct",
            "description": "Serve vLLM on AMD Instinct MI355X GPUs.",
            "origin": "test",
            "url": "test",
        }
        installed = finder._rank_skills(
            "serve vLLM on MI355X", [entry], installed=True
        )[0]
        catalog = finder._rank_skills("serve vLLM on MI355X", [entry], installed=False)[
            0
        ]
        self.assertGreater(installed["score"], catalog["score"])
        self.assertEqual(installed["type"], "installed_skill")
        self.assertNotIn("install_command", installed)
        self.assertEqual(catalog["type"], "installable_skill")

    def test_live_skill_file_is_embedded_not_installable(self):
        project = {
            "id": "hyperloom",
            "repositories": [{"repo": "AMD-AGI/Hyperloom", "tier": "amd-official"}],
        }
        payload = [
            {
                "path": "src/hyperloom/inference_optimizer/SKILL.md",
                "url": "https://example.invalid/SKILL.md",
                "textMatches": [{"fragment": "GPU optimization skill"}],
            }
        ]
        with mock.patch.object(finder, "_gh_json", return_value=payload):
            results, warnings = finder.live_code_search(
                "Hyperloom optimization", [project]
            )
        self.assertEqual(warnings, [])
        self.assertEqual(results[0]["type"], "embedded_skill")
        self.assertFalse(results[0]["installable"])

    def test_secret_like_query_is_rejected(self):
        with self.assertRaises(finder.FinderError):
            finder.reject_secrets("search github with ghp_abcdefghijklmnopqrstuvwxyz")

    def test_offline_cli_emits_json_without_dynamo_repository(self):
        process = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "disaggregated KV cache RDMA on MI355X",
                "--offline",
                "--kind",
                "sources",
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        payload = json.loads(process.stdout)
        repositories = {
            repo["repo"].lower()
            for project in payload["source_projects"]
            for repo in project["repositories"]
        }
        self.assertNotIn("ai-dynamo/dynamo", repositories)
        self.assertIn("rocm/mori", repositories)


if __name__ == "__main__":
    unittest.main()
