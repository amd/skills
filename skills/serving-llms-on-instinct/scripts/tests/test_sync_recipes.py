#!/usr/bin/env python3
"""Regression tests for recipe-cache freshness and provenance selection."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "sync_recipes.py"
SPEC = importlib.util.spec_from_file_location("sync_recipes", SCRIPT)
sync_recipes = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(sync_recipes)


class CacheStatusTests(unittest.TestCase):
    def test_missing_cache_is_not_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, fresh = sync_recipes._cache_status(
                str(Path(tmp) / "missing.json")
            )
        self.assertFalse(fresh)
        self.assertEqual(status["status"], "missing")

    def test_fresh_and_stale_cache_are_distinguished(self):
        now = datetime(2026, 8, 30, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "recipes.json"
            cache.write_text(json.dumps({
                "fetched_at": (now - timedelta(hours=2)).isoformat(),
                "recipes_commit": "abc123",
                "docker_image_pinned": "vllm/image@sha256:123",
            }), encoding="utf-8")

            status, fresh = sync_recipes._cache_status(
                str(cache), max_age_hours=24, now=now
            )
            self.assertTrue(fresh)
            self.assertEqual(status["status"], "fresh")

            status, fresh = sync_recipes._cache_status(
                str(cache), max_age_hours=1, now=now
            )
            self.assertFalse(fresh)
            self.assertEqual(status["status"], "stale")

    def test_malformed_cache_is_not_fresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "recipes.json"
            cache.write_text("not json", encoding="utf-8")
            status, fresh = sync_recipes._cache_status(str(cache))
        self.assertFalse(fresh)
        self.assertEqual(status["status"], "invalid")


class CachePathTests(unittest.TestCase):
    def test_environment_cache_path_is_resolved_at_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "AMD_SKILLS_RECIPE_CACHE": "$RECIPE_CACHE_ROOT/env-cache.json",
                "RECIPE_CACHE_ROOT": tmp,
            }
            with mock.patch.dict(os.environ, env, clear=False):
                resolved = sync_recipes._runtime_cache_file()
        self.assertEqual(resolved, str(Path(tmp, "env-cache.json").resolve()))

    def test_cache_file_cli_override_wins_and_expands_variables(self):
        with tempfile.TemporaryDirectory() as tmp:
            explicit = "$RECIPE_CACHE_ROOT/cli-cache.json"
            expected = str(Path(tmp, "cli-cache.json").resolve())
            env = {
                "AMD_SKILLS_RECIPE_CACHE": str(Path(tmp, "env-cache.json")),
                "RECIPE_CACHE_ROOT": tmp,
            }
            argv = ["sync_recipes.py", "--check", "--cache-file", explicit]
            with (
                mock.patch.dict(os.environ, env, clear=False),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    sync_recipes,
                    "_cache_status",
                    return_value=({"status": "fresh"}, True),
                ) as cache_status,
                redirect_stdout(io.StringIO()),
            ):
                result = sync_recipes.main()

        self.assertEqual(result, 0)
        cache_status.assert_called_once_with(
            expected, sync_recipes.DEFAULT_MAX_AGE_HOURS
        )


class CacheWriteTests(unittest.TestCase):
    def test_cache_is_replaced_and_temporary_file_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_file = Path(tmp, "recipes.json")
            sync_recipes._write_cache(str(cache_file), {"fetched_at": "now"})
            written = json.loads(cache_file.read_text(encoding="utf-8"))
            leftovers = list(Path(tmp).glob(".recipes-cache-*.tmp"))

        self.assertEqual(written, {"fetched_at": "now"})
        self.assertEqual(leftovers, [])


class RecipeParsingTests(unittest.TestCase):
    @unittest.skipUnless(sync_recipes.HAS_YAML, "PyYAML is not installed")
    def test_yaml_is_read_as_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            recipe = Path(tmp) / "recipe.yaml"
            recipe.write_text("meta:\n  description: 日本語\n", encoding="utf-8")
            parsed = sync_recipes._parse_yaml(str(recipe))
        self.assertEqual(parsed["meta"]["description"], "日本語")


class DockerTagTests(unittest.TestCase):
    def test_highest_stable_semver_wins_over_latest_and_nightly(self):
        tags = [
            {"name": "latest", "digest": "sha256:latest"},
            {"name": "nightly", "digest": "sha256:nightly"},
            {"name": "v0.9.2", "digest": "sha256:old"},
            {
                "name": "v0.28.0",
                "last_updated": "2026-08-26T00:00:00Z",
                "images": [{"digest": "sha256:new"}],
            },
            {"name": "v0.22.0", "digest": "sha256:middle"},
        ]
        tag, updated, digest = sync_recipes._select_docker_tag(tags)
        self.assertEqual(tag, "v0.28.0")
        self.assertEqual(updated, "2026-08-26T00:00:00Z")
        self.assertEqual(digest, "sha256:new")

    def test_missing_stable_tag_fails(self):
        with self.assertRaises(RuntimeError):
            sync_recipes._select_docker_tag([
                {"name": "latest", "digest": "sha256:latest"}
            ])

    def test_fetch_follows_pagination_before_selecting(self):
        next_url = f"{sync_recipes.DOCKERHUB_URL}?page=2"
        responses = [
            sync_recipes.subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout=json.dumps({
                    "results": [{"name": "v0.22.0", "digest": "sha256:old"}],
                    "next": next_url,
                }),
                stderr="",
            ),
            sync_recipes.subprocess.CompletedProcess(
                args=[], returncode=0,
                stdout=json.dumps({
                    "results": [{"name": "v0.28.0", "digest": "sha256:new"}],
                    "next": None,
                }),
                stderr="",
            ),
        ]
        with mock.patch.object(
            sync_recipes.subprocess, "run", side_effect=responses
        ) as run:
            tag, _, digest = sync_recipes._fetch_docker_tag()

        self.assertEqual(tag, "v0.28.0")
        self.assertEqual(digest, "sha256:new")
        self.assertEqual(run.call_count, 2)
        self.assertIn("page_size=100", run.call_args_list[0].args[0][-1])
        self.assertEqual(run.call_args_list[1].args[0][-1], next_url)


if __name__ == "__main__":
    unittest.main()
