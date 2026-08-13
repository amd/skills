#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["pyyaml>=6.0"]
# ///
"""Decide what the `evals` workflow should run for a change.

Emits one JSON object on stdout::

    {
      "routing": true,
      "default": [
        {"skill": "local-ai-use", "os": "Linux",
         "runner": "[\\"self-hosted\\",\\"strix_halo\\",\\"Linux\\"]", "gate": ""}
      ],
      "scoped": [
        {"skill": "serving-llms-on-instinct", "os": "Linux",
         "runner": "[\\"self-hosted\\",\\"Linux\\",\\"mi300x\\"]",
         "environment": "behavioral-instinct", "gate": "enable_mi_ci"}
      ],
      "skipped": [{"skill": "serving-llms-on-instinct", "gate": "enable_mi_ci"}],
      "gates": ["enable_mi_ci"]
    }

``default`` and ``scoped`` are GitHub Actions matrices. Which class of machine
a skill needs comes from its ``evals/machine.yml``, and everything that class
implies is resolved by ``datasets.machine_plan``, so adding a skill that needs
unusual hardware never means editing this file or the workflow. A hardcoded
list of which skills need which runner lives in the wrong place: the person
who knows about the hardware is the skill's owner, not whoever last touched
CI, and the two drift silently.

The split is by credentials, not by hardware: a runner class with its own
``environment`` reads that environment's scoped secrets, and one without uses
the repo-wide key for AMD's internal gateway. They are separate jobs because a
job's credentials have to be fixed before its matrix expands.

``skipped`` holds legs whose gate label is missing from the pull request. A
gate comes with the runner class -- the Instinct pool always requires
``enable_mi_ci`` -- rather than being something a skill opts into. They are
reported so the gate job can warn that a change shipped without ever running
on real hardware, rather than failing a PR for a test it deliberately did not
request.

Usage::

    .github/scripts/select_evals.py --all
    .github/scripts/select_evals.py --names "local-ai-use,serving-llms-on-epyc"
    git diff --name-only BASE HEAD | .github/scripts/select_evals.py --changed \
        --labels "enable_mi_ci"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval"))

import datasets  # noqa: E402

# Touching any of these changes the shared engine rather than one skill, so
# every skill is re-run rather than guessing at the blast radius. Repo-root
# relative with forward slashes, to match `git diff --name-only`.
INFRA_FILES = {
    "eval/agent.py",
    "eval/datasets.py",
    "eval/routing.py",
    "eval/run_evals.py",
    "eval/negatives.json",
    ".github/scripts/select_evals.py",
    ".github/workflows/evals.yml",
    # Publishing or unpublishing a skill changes who competes for every prompt,
    # so it re-scores the whole catalog rather than just the skill it names.
    ".claude-plugin/marketplace.json",
}


def has_behavior_cases(skill: str) -> bool:
    """Whether this skill asserts anything a behavior run could grade."""
    return any(case.has_behavior for case in datasets.load_dataset(skill))


def matrix_entries(
    skills: list[str], labels: set[str], ignore_gates: bool = False
) -> tuple[list[dict], list[dict]]:
    """Split `skills` into matrix legs to run and legs held back by a gate."""
    include: list[dict] = []
    skipped: list[dict] = []

    for skill in skills:
        if not has_behavior_cases(skill):
            continue
        plan = datasets.machine_plan(skill)
        gate = plan["gate"]
        if gate and not ignore_gates and gate not in labels:
            skipped.append({"skill": skill, "gate": gate})
            continue

        for os_name in plan["os"]:
            leg = {
                "skill": skill,
                "os": os_name,
                "runner": json.dumps(datasets.runner_labels(plan, os_name)),
                "gate": gate,
            }
            if plan["environment"]:
                leg["environment"] = plan["environment"]
            include.append(leg)
    return include, skipped


def routing_needed(changed: set[str]) -> bool:
    """Whether the change can move a routing decision.

    A skill's description and its prompts are the only inputs to routing, so a
    PR that only edits a reference file or a helper script under a skill does
    not need to pay for a catalog-wide run.

    An unpublished skill's description is not an input either: routing installs
    the published bundle, so that skill is not in the room to win or lose a
    prompt. Its dataset still counts, because its near-miss prompts are graded
    against the skills that are.
    """
    if changed & INFRA_FILES:
        return True
    published = set(datasets.routing_catalog())
    for path in changed:
        if path.endswith("/evals/evals.json"):
            return True
        if path.endswith("/SKILL.md"):
            parts = path.split("/")
            if len(parts) < 2 or parts[0] != "skills" or parts[1] in published:
                return True
    return False


def select_from_changes(changed: set[str]) -> list[str]:
    """Skills whose behavior tests a set of changed paths should re-run."""
    if changed & INFRA_FILES:
        return datasets.skills_with_datasets()
    selected = {
        path.split("/")[1]
        for path in changed
        if path.startswith("skills/") and len(path.split("/")) >= 2
    }
    return [skill for skill in datasets.skills_with_datasets() if skill in selected]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="Every skill with a dataset.")
    mode.add_argument("--changed", action="store_true", help="Read changed paths from stdin.")
    mode.add_argument("--names", metavar="A,B,C", help="An explicit comma-separated skill list.")
    parser.add_argument(
        "--labels", default="",
        help="Comma-separated pull-request labels, used to satisfy runner gates.",
    )
    parser.add_argument(
        "--ignore-gates", action="store_true",
        help="Run gated skills without their label. For workflow_dispatch, which is already explicit human intent.",
    )
    args = parser.parse_args(argv)

    available = datasets.skills_with_datasets()
    changed: set[str] = set()
    if args.all:
        skills = available
        routing = True
    elif args.names is not None:
        requested = [n.strip() for n in args.names.split(",") if n.strip()]
        unknown = [n for n in requested if n not in available]
        if unknown:
            print(f"error: no eval dataset for: {', '.join(unknown)}", file=sys.stderr)
            return 1
        skills = requested
        routing = True
    else:
        changed = {
            line.strip().replace("\\", "/")
            for line in sys.stdin.read().splitlines()
            if line.strip()
        }
        skills = select_from_changes(changed)
        routing = routing_needed(changed)

    labels = {token.strip() for token in args.labels.split(",") if token.strip()}
    include, skipped = matrix_entries(skills, labels, ignore_gates=args.ignore_gates)

    print(
        json.dumps(
            {
                "routing": routing,
                "default": [leg for leg in include if "environment" not in leg],
                "scoped": [leg for leg in include if "environment" in leg],
                "skipped": skipped,
                "gates": sorted({s["gate"] for s in skipped}),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
