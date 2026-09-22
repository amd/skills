#!/usr/bin/env python3
"""
Sync vLLM recipes and Docker Hub tags into a local cache.

Fetches:
  1. Shallow clone of vllm-project/recipes from GitHub
  2. Reads all model YAML files from models/<org>/<model>.yaml
  3. Latest stable Docker image tag from Docker Hub API

Writes output to a writable runtime cache. The path is reported as JSON.

Usage:
    python3 scripts/sync_recipes.py                 # refresh runtime cache
    python3 scripts/sync_recipes.py --check         # require a fresh cache
    python3 scripts/sync_recipes.py --verbose       # show progress
    python3 scripts/sync_recipes.py --cache-file F  # explicit cache path

Refresh failures exit nonzero. The bundled cache remains a fallback for an
agent that explicitly discloses its age; it is never overwritten at runtime.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

REPO_URL = "https://github.com/vllm-project/recipes.git"
DOCKERHUB_URL = "https://hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags"

SKILL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUNDLED_CACHE_FILE = os.path.join(SKILL_DIR, "data", "recipes_cache.json")
DEFAULT_CACHE_FILE = os.path.join(
    tempfile.gettempdir(), "amd-skills", "serving-llms-on-instinct",
    "recipes_cache.json",
)
DEFAULT_MAX_AGE_HOURS = 24.0
MAX_DOCKERHUB_PAGES = 100
_STABLE_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _resolve_cache_file(path):
    """Expand user/environment syntax and return an absolute cache path."""
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def _runtime_cache_file():
    """Resolve the environment override at call time, not module import time."""
    return _resolve_cache_file(
        os.environ.get("AMD_SKILLS_RECIPE_CACHE", DEFAULT_CACHE_FILE)
    )


def _log(msg, verbose):
    if verbose:
        print(f"  [sync] {msg}", file=sys.stderr, flush=True)


def _parse_yaml(path):
    """Parse a YAML file. Requires PyYAML."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clone_recipes(verbose=False):
    """Shallow clone the recipes repo. Return ``(path, commit)``."""
    tmpdir = tempfile.mkdtemp(prefix="vllm-recipes-")
    _log(f"Cloning {REPO_URL} (shallow)...", verbose)
    r = subprocess.run(
        ["git", "clone", "--depth=1", "--single-branch", "--filter=blob:none",
         REPO_URL, tmpdir],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
    )
    if r.returncode != 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f"git clone failed: {r.stderr[:200]}")
    commit = subprocess.run(
        ["git", "-C", tmpdir, "rev-parse", "HEAD"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10,
    )
    if commit.returncode != 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f"git rev-parse failed: {commit.stderr[:200]}")
    return tmpdir, commit.stdout.strip()


def _read_all_recipes(repo_dir, verbose=False):
    """Read all model YAML files from the cloned repo."""
    models_dir = os.path.join(repo_dir, "models")
    if not os.path.isdir(models_dir):
        raise RuntimeError(f"No models/ directory in cloned repo")

    recipes = {}
    yaml_files = glob.glob(os.path.join(models_dir, "*", "*.yaml"))
    _log(f"Found {len(yaml_files)} model YAML files", verbose)

    for path in sorted(yaml_files):
        org = os.path.basename(os.path.dirname(path))
        model = os.path.splitext(os.path.basename(path))[0]
        hf_id = f"{org}/{model}"

        try:
            recipe = _parse_yaml(path)
            if not recipe:
                continue

            meta = recipe.get("meta", {})
            model_section = recipe.get("model", {})

            recipes[hf_id] = {
                "hf_id": hf_id,
                "meta": {
                    "title": meta.get("title", model),
                    "provider": meta.get("provider", org),
                    "description": meta.get("description", ""),
                    "tasks": meta.get("tasks", []),
                    "hardware": meta.get("hardware", {}),
                },
                "model_info": {
                    "architecture": model_section.get("architecture", "dense"),
                    "parameter_count": model_section.get("parameter_count", ""),
                },
                "recipe": recipe,
            }
        except Exception as e:
            _log(f"Failed to parse {hf_id}: {e}", verbose)

    return recipes


def _select_docker_tag(tags):
    """Select the highest stable semantic version and its manifest digest."""
    candidates = []
    for tag in tags:
        match = _STABLE_TAG.fullmatch(tag.get("name", ""))
        if match:
            candidates.append((tuple(map(int, match.groups())), tag))
    if not candidates:
        raise RuntimeError("Docker Hub returned no stable vLLM ROCm tag")

    selected = max(candidates, key=lambda item: item[0])[1]
    digest = selected.get("digest", "")
    if not digest:
        images = selected.get("images") or []
        digest = next((image.get("digest", "") for image in images
                       if image.get("digest")), "")
    if not digest:
        raise RuntimeError(
            f"Docker Hub returned no digest for {selected['name']}"
        )
    return selected["name"], selected.get("last_updated", ""), digest


def _fetch_docker_tag(verbose=False):
    """Fetch the highest stable vLLM ROCm tag and digest from Docker Hub."""
    _log("Fetching Docker Hub tags...", verbose)
    url = f"{DOCKERHUB_URL}?page_size=100&ordering=last_updated"
    tags = []
    for page in range(1, MAX_DOCKERHUB_PAGES + 1):
        r = subprocess.run(
            ["curl", "-sf", "--max-time", "5", url],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=10,
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"Docker Hub request failed on page {page}: {r.stderr[:200]}"
            )

        data = json.loads(r.stdout)
        results = data.get("results", [])
        if not isinstance(results, list):
            raise RuntimeError(f"Docker Hub returned invalid results on page {page}")
        tags.extend(results)

        next_url = data.get("next")
        if not next_url:
            return _select_docker_tag(tags)
        if not isinstance(next_url, str) or not next_url.startswith(DOCKERHUB_URL):
            raise RuntimeError("Docker Hub returned an invalid pagination URL")
        url = next_url

    raise RuntimeError(
        f"Docker Hub pagination exceeded {MAX_DOCKERHUB_PAGES} pages"
    )


def _cache_status(cache_file, max_age_hours=DEFAULT_MAX_AGE_HOURS, now=None):
    """Return machine-readable cache freshness and whether it is usable."""
    result = {"cache": os.path.abspath(cache_file)}
    if not os.path.isfile(cache_file):
        return {**result, "status": "missing"}, False
    try:
        with open(cache_file, encoding="utf-8") as f:
            cache = json.load(f)
        fetched_at = datetime.fromisoformat(cache["fetched_at"].replace("Z", "+00:00"))
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        return {**result, "status": "invalid", "error": str(e)}, False

    now = now or datetime.now(timezone.utc)
    age_hours = max(0.0, (now - fetched_at).total_seconds() / 3600)
    fresh = age_hours <= max_age_hours
    return {
        **result,
        "status": "fresh" if fresh else "stale",
        "fetched_at": fetched_at.isoformat(),
        "age_hours": round(age_hours, 2),
        "max_age_hours": max_age_hours,
        "recipes_commit": cache.get("recipes_commit", ""),
        "docker_image": cache.get("docker_image_pinned",
                                  cache.get("docker_image", "")),
    }, fresh


def _write_cache(cache_file, cache):
    """Atomically replace a cache so interruption cannot leave partial JSON."""
    cache_dir = os.path.dirname(os.path.abspath(cache_file))
    os.makedirs(cache_dir, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=".recipes-cache-", suffix=".tmp",
        dir=cache_dir, delete=False,
    )
    tmp_path = tmp.name
    try:
        with tmp:
            json.dump(cache, tmp, indent=2, default=str)
            tmp.write("\n")
        os.replace(tmp_path, cache_file)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def sync(verbose=False, cache_file=None):
    cache_file = _resolve_cache_file(cache_file or _runtime_cache_file())
    if not HAS_YAML:
        raise RuntimeError(
            "PyYAML is required to sync recipes; install it with "
            "`python3 -m pip install PyYAML`"
        )

    # Step 1: Clone the repo
    repo_dir, recipes_commit = _clone_recipes(verbose)

    try:
        # Step 2: Read all YAML recipes
        recipes = _read_all_recipes(repo_dir, verbose)
        _log(f"Parsed {len(recipes)} models", verbose)
    finally:
        shutil.rmtree(repo_dir, ignore_errors=True)

    if not recipes:
        raise RuntimeError("No recipes found in cloned repo")

    # Step 3: Fetch Docker Hub tag
    docker_tag, docker_date, docker_digest = _fetch_docker_tag(verbose)
    _log(f"Latest stable ROCm tag: {docker_tag} ({docker_date})", verbose)

    # Step 4: Write cache
    cache = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "recipes_source": REPO_URL,
        "recipes_commit": recipes_commit,
        "docker_image": f"vllm/vllm-openai-rocm:{docker_tag}",
        "docker_image_pinned": (
            f"vllm/vllm-openai-rocm@{docker_digest}"
        ),
        "docker_digest": docker_digest,
        "docker_tag": docker_tag,
        "docker_tag_date": docker_date,
        "model_count": len(recipes),
        "models": recipes,
    }

    _write_cache(cache_file, cache)
    _log(f"Cache written: {len(recipes)} models, tag={docker_tag}", verbose)
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="exit nonzero unless the cache is fresh")
    parser.add_argument("--max-age-hours", type=float,
                        default=DEFAULT_MAX_AGE_HOURS)
    parser.add_argument("--cache-file")
    args = parser.parse_args()
    cache_file = (
        _resolve_cache_file(args.cache_file)
        if args.cache_file
        else _runtime_cache_file()
    )

    if args.check:
        status, fresh = _cache_status(cache_file, args.max_age_hours)
        print(json.dumps(status))
        return 0 if fresh else 1

    try:
        cache = sync(verbose=args.verbose, cache_file=cache_file)
        print(json.dumps({
            "status": "ok",
            "cache": cache_file,
            "recipes_commit": cache["recipes_commit"],
            "docker_image": cache["docker_image_pinned"],
        }))
        return 0
    except Exception as e:
        print(f"WARN: sync_recipes failed: {e}", file=sys.stderr)
        print(json.dumps({
            "status": "failed",
            "error": str(e),
            "cache": cache_file,
            "bundled_fallback": os.path.abspath(BUNDLED_CACHE_FILE),
        }))
        return 1


if __name__ == "__main__":
    sys.exit(main())
