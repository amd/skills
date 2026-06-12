#!/usr/bin/env python3
"""
Sync vLLM recipes and Docker Hub tags into a local cache.

Fetches:
  1. Shallow clone of vllm-project/recipes from GitHub
  2. Reads all model YAML files from models/<org>/<model>.yaml
  3. Latest stable Docker image tag from Docker Hub API

Writes output to: data/recipes_cache.json

Usage:
    python3 scripts/sync_recipes.py              # refresh cache
    python3 scripts/sync_recipes.py --verbose     # show progress

Exit 0 always (callers fall back to existing cache on failure).
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

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CACHE_FILE = os.path.join(CACHE_DIR, "recipes_cache.json")


def _log(msg, verbose):
    if verbose:
        print(f"  [sync] {msg}", file=sys.stderr, flush=True)


def _parse_yaml(path):
    """Parse a YAML file. Requires PyYAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def _clone_recipes(verbose=False):
    """Shallow clone the recipes repo into a temp directory. Returns path."""
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
    return tmpdir


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


def _fetch_docker_tag(verbose=False):
    """Fetch the latest stable vllm-openai-rocm tag from Docker Hub."""
    _log("Fetching Docker Hub tags...", verbose)
    url = f"{DOCKERHUB_URL}?page_size=50&ordering=last_updated"
    r = subprocess.run(
        ["curl", "-sf", "--max-time", "5", url],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10,
    )
    if r.returncode != 0:
        return "latest", ""

    data = json.loads(r.stdout)
    for tag in data.get("results", []):
        name = tag["name"]
        if "nightly" in name or "base" in name:
            continue
        if name.startswith("v") and re.match(r"v\d+\.\d+", name):
            return name, tag.get("last_updated", "")
        if name == "latest":
            return name, tag.get("last_updated", "")

    return "latest", ""


def sync(verbose=False):
    if not HAS_YAML:
        print("WARN: PyYAML not installed, cannot sync recipes", file=sys.stderr)
        return False

    # Step 1: Clone the repo
    repo_dir = _clone_recipes(verbose)

    try:
        # Step 2: Read all YAML recipes
        recipes = _read_all_recipes(repo_dir, verbose)
        _log(f"Parsed {len(recipes)} models", verbose)
    finally:
        shutil.rmtree(repo_dir, ignore_errors=True)

    if not recipes:
        print("WARN: No recipes found in cloned repo", file=sys.stderr)
        return False

    # Step 3: Fetch Docker Hub tag
    docker_tag, docker_date = "latest", ""
    try:
        docker_tag, docker_date = _fetch_docker_tag(verbose)
        _log(f"Latest stable ROCm tag: {docker_tag} ({docker_date})", verbose)
    except Exception as e:
        _log(f"Docker Hub fetch failed: {e}", verbose)

    # Step 4: Write cache
    cache = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "docker_image": f"vllm/vllm-openai-rocm:{docker_tag}",
        "docker_tag": docker_tag,
        "docker_tag_date": docker_date,
        "model_count": len(recipes),
        "models": recipes,
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2, default=str)

    _log(f"Cache written: {len(recipes)} models, tag={docker_tag}", verbose)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    try:
        ok = sync(verbose=args.verbose)
        if ok:
            print(json.dumps({"status": "ok", "cache": CACHE_FILE}))
        else:
            print(json.dumps({"status": "partial", "cache": CACHE_FILE}))
    except Exception as e:
        print(f"WARN: sync_recipes failed: {e}", file=sys.stderr)
        print(json.dumps({"status": "failed", "error": str(e)}))
    sys.exit(0)


if __name__ == "__main__":
    main()
