#!/usr/bin/env python3
"""Build an OrchestrAI plan from Skillscope's default behavioral matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

EXPECTED_REPOSITORY = "amd/skills"
EXPECTED_SKILLS_SOURCE = "https://github.com/amd/skills.git"
SKILL_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
REF_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
DRIVER_RELEASE_RE = re.compile(r"^[a-z0-9._-]+:[0-9]+(?:\.[0-9]+)*$")
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "orchestrai-config.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--device-tags-json",
        default=os.environ.get("ORCHESTRAI_DEVICE_TAGS", ""),
        help='fleet map, for example {"strix_halo":["<broker-tag>"]}',
    )
    parser.add_argument("--matrix-json", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument(
        "--extended-flag", choices=("--extended", "--no-extended"), required=True
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", default="1")
    parser.add_argument("--source-run-url", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _validate_source(repository: str, ref: str, sha: str) -> None:
    if repository != EXPECTED_REPOSITORY:
        raise SystemExit(f"repository must be {EXPECTED_REPOSITORY}")
    if not SHA_RE.fullmatch(sha):
        raise SystemExit("sha must be a full 40-character commit SHA")
    if not REF_RE.fullmatch(ref) or ref.startswith("-") or ".." in ref or "@{" in ref:
        raise SystemExit("ref contains unsafe characters")


def _load_config(path: Path, device_tags_json: str) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid OrchestrAI config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit("OrchestrAI config must be a JSON object")
    if device_tags_json.strip():
        try:
            device_tags = json.loads(device_tags_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"ORCHESTRAI_DEVICE_TAGS is invalid JSON: {exc}") from exc
        if not isinstance(device_tags, dict):
            raise SystemExit("ORCHESTRAI_DEVICE_TAGS must be a JSON object")
        config["device_to_tags"] = device_tags
    required_objects = ("skills_source", "test_paths", "os_images", "run_settings")
    missing = [key for key in required_objects if not isinstance(config.get(key), dict)]
    if missing:
        raise SystemExit(
            f"OrchestrAI config is missing object(s): {', '.join(missing)}"
        )
    tags = (config.get("device_to_tags") or {}).get("strix_halo")
    if (
        not isinstance(tags, list)
        or not tags
        or not all(isinstance(tag, str) and tag for tag in tags)
    ):
        raise SystemExit(
            "ORCHESTRAI_DEVICE_TAGS must define a non-empty strix_halo tag list"
        )
    return config


def _linux_builds(config: dict, selected_os: set[str]) -> dict | None:
    """Build Playbooks-compatible per-release Linux provisioning input.

    The GitHub value can be line-wrapped by the repository-variable UI.  Match
    the Playbooks trigger's recovery behavior, but never print the value or its
    URLs because this repository and its workflow logs are public.
    """
    if selected_os != {"Linux"}:
        return None

    raw = os.environ.get("ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON", "").strip()
    if not raw:
        return None

    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        inner = raw[1:-1].strip()
        try:
            json.loads(inner)
            raw = inner
        except json.JSONDecodeError:
            pass

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Raw newlines and tabs cannot be part of a URL.  Playbooks accepts the
        # same repair for values pasted into the GitHub variable editor.
        try:
            parsed = json.loads(re.sub(r"[\n\r\t]+", "", raw))
        except json.JSONDecodeError as exc:
            raise SystemExit(
                "ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON is not valid JSON"
            ) from exc

    if not isinstance(parsed, dict) or not parsed:
        raise SystemExit(
            "ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON must be a non-empty object"
        )
    for release, value in parsed.items():
        if not isinstance(release, str) or not DRIVER_RELEASE_RE.fullmatch(release):
            raise SystemExit(
                "ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON contains an invalid release key"
            )
        if not isinstance(value, str) or not value.strip().startswith("https://"):
            raise SystemExit(
                "ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON values must be HTTPS URLs"
            )
        # A repaired line break can leave indentation inside the URL.
        parsed[release] = re.sub(r"\s+", "", value)

    scripts = (config.get("provisioning") or {}).get("linux_install_scripts")
    if not isinstance(scripts, list) or not scripts:
        raise SystemExit("provisioning.linux_install_scripts must be a non-empty list")
    normalized_scripts = []
    for item in scripts:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("script"), str)
            or not item["script"].startswith("InstallationScripts/")
            or not isinstance(item.get("reboot_after"), bool)
        ):
            raise SystemExit("provisioning.linux_install_scripts is invalid")
        normalized_scripts.append(
            {"script": item["script"], "reboot_after": item["reboot_after"]}
        )

    return {
        "vars": {
            "driver_sources_json": json.dumps(parsed, separators=(",", ":")),
        },
        "install_scripts": normalized_scripts,
    }


def build_plan(args: argparse.Namespace) -> dict:
    _validate_source(args.repository, args.ref, args.sha)
    config = _load_config(args.config, args.device_tags_json)
    test_paths = config["test_paths"]
    os_images = config["os_images"]
    skills_source = config["skills_source"]
    run_settings = config["run_settings"]
    acquire_timeout = run_settings.get("acquire_timeout")
    max_duration = run_settings.get("max_duration")
    if (
        isinstance(acquire_timeout, bool)
        or not isinstance(acquire_timeout, int)
        or acquire_timeout <= 0
    ):
        raise SystemExit("run_settings.acquire_timeout must be a positive integer")
    if (
        isinstance(max_duration, bool)
        or not isinstance(max_duration, int)
        or max_duration <= acquire_timeout
    ):
        raise SystemExit(
            "run_settings.max_duration must be a positive integer greater than "
            "run_settings.acquire_timeout"
        )
    machine_tags = config["device_to_tags"]["strix_halo"]
    skillscope_sha = str(config.get("skillscope_sha") or "")
    if str(skills_source.get("repository") or "") != EXPECTED_SKILLS_SOURCE:
        raise SystemExit(f"skills_source.repository must be {EXPECTED_SKILLS_SOURCE}")
    if not SHA_RE.fullmatch(skillscope_sha):
        raise SystemExit("skillscope_sha must be a full 40-character commit SHA")
    try:
        matrix = json.loads(args.matrix_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"matrix is not valid JSON: {exc}") from exc
    if not isinstance(matrix, list):
        raise SystemExit("matrix must be a JSON list")

    selected: set[tuple[str, str]] = set()
    for entry in matrix:
        if not isinstance(entry, dict):
            raise SystemExit("each matrix entry must be an object")
        os_name = entry.get("os")
        skill = entry.get("skill")
        if os_name not in test_paths or os_name not in os_images:
            raise SystemExit(f"unsupported behavioral OS: {os_name!r}")
        if not isinstance(skill, str) or not SKILL_RE.fullmatch(skill):
            raise SystemExit(f"invalid skill name: {skill!r}")
        selected.add((os_name, skill))

    sessions = []
    for os_name, test_path in test_paths.items():
        for skill in sorted(
            skill for selected_os, skill in selected if selected_os == os_name
        ):
            variables = {
                "SKILLS_REPO": str(skills_source.get("repository") or ""),
                # The workflow resolves the current public branch tip just
                # before building this plan. The adapter uses that resolution
                # only to keep the remote checkout stable while the run starts.
                "SKILLS_REF": args.ref,
                "SKILLS_SHA": args.sha.lower(),
                "SKILLSCOPE_SHA": skillscope_sha,
                "SKILLS": skill,
                "SKILLS_OS": os_name,
                "SKILLS_EXTENDED": str(args.extended_flag == "--extended").lower(),
                "SKILLS_TIMEOUT": "7200",
                "SOURCE_REPOSITORY": args.repository,
                "SOURCE_REF": args.ref,
                "SOURCE_SHA": args.sha.lower(),
                "SOURCE_GITHUB_RUN_URL": args.source_run_url,
            }
            sessions.append(
                {
                    "name": f"skills-{skill}-{os_name.lower()}",
                    # Variables live on the test object because current Portal
                    # session-to-group conversion preserves test objects end to end.
                    "tests": [{"path": test_path, "variables": variables}],
                    "machine_tags": machine_tags,
                    "os_image": str(os_images[os_name]),
                    "allocation": "single",
                    # Keep the launcher wait aligned with the broker-acquire wait.
                    # Portal run settings use minutes, while launcher durations use
                    # an explicit unit-bearing string.
                    "waiting_timeout": f"{acquire_timeout}m",
                }
            )

    if not sessions:
        raise SystemExit("matrix selected no default behavioral sessions")

    selected_os_names = {os_name for os_name, _skill in selected}
    selected_os = sorted(os_name.lower() for os_name in selected_os_names)
    os_suffix = "-".join(selected_os)
    name = f"skills-evals-{args.run_id}-{args.run_attempt}-{os_suffix}"
    plan = {
        "name": name,
        "description": (
            "Skills behavioral evaluations dispatched by "
            f"{args.source_run_url or 'GitHub Actions'}"
        ),
        "tags": ["skills", "github-actions", "strix-halo"],
        "sessions": sessions,
        "run_settings": dict(run_settings),
    }
    builds = _linux_builds(config, selected_os_names)
    if builds:
        # This field is consumed only by orchestrai_run.py and is never included
        # in the public result artifacts or the persisted plan definition.
        plan["builds_json"] = builds
    return plan


def main() -> int:
    args = _parser().parse_args()
    plan = build_plan(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(plan['sessions'])} OrchestrAI session(s) to {args.output}")
    for session in plan["sessions"]:
        print(f"  {session['name']}: {session['tests'][0]['variables']['SKILLS']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
