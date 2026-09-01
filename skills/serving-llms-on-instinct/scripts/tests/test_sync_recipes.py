#!/usr/bin/env python3
"""Regression tests for recipe-cache freshness and provenance selection."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
