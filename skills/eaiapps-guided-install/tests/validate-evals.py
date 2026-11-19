#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Validate the catalog evaluation dataset without running an agent."""

import json
from pathlib import Path


if not __debug__:
    raise RuntimeError("run validate-evals.py without python -O / PYTHONOPTIMIZE")


skill_root = Path(__file__).parents[1]
evals_path = skill_root / "evals" / "evals.json"
data = json.loads(evals_path.read_text(encoding="utf-8"))
assert isinstance(data, dict), "evaluation dataset root must be an object"
evaluations = data.get("evaluations")
assert isinstance(evaluations, list), "evaluations must be an array"

ids: set[str] = set()
triggering = 0
non_triggering = 0
behavioral = 0
optional_fields = ("expected_behavior", "unexpected_behavior", "logs_contain", "files_exist")

for case in evaluations:
    assert isinstance(case, dict), "each evaluation must be an object"
    case_id = case.get("id")
    prompt = case.get("prompt")
    should_trigger = case.get("skill_should_trigger")
    assert isinstance(case_id, str) and case_id, "each evaluation needs a non-empty id"
    assert case_id not in ids, f"duplicate evaluation id: {case_id}"
    ids.add(case_id)
    assert isinstance(prompt, str) and prompt, f"{case_id}: prompt must be non-empty"
    assert isinstance(should_trigger, bool), f"{case_id}: skill_should_trigger must be boolean"

    if should_trigger:
        triggering += 1
    else:
        non_triggering += 1

    for field in optional_fields:
        if field not in case:
            continue
        value = case[field]
        assert should_trigger, f"{case_id}: {field} is only valid for triggering cases"
        assert isinstance(value, list) and value, f"{case_id}: {field} must be a non-empty array"
        assert all(isinstance(item, str) and item for item in value), f"{case_id}: {field} entries must be strings"
        behavioral += field in ("expected_behavior", "unexpected_behavior")

    # A `workspace` naming a directory that is not there fails the case at
    # stage time, deep inside the harness, where it reads as the harness being
    # broken rather than this dataset.
    workspace = case.get("workspace")
    if workspace is not None:
        assert isinstance(workspace, str) and workspace, f"{case_id}: workspace must be a path"
        assert should_trigger, f"{case_id}: workspace is only valid for triggering cases"
        staged = skill_root / workspace
        assert staged.is_dir(), f"{case_id}: workspace {workspace} is not a directory"
        assert any(staged.rglob("*")), f"{case_id}: workspace {workspace} is empty"

assert triggering >= 3, "evaluation dataset needs at least three triggering cases"
assert non_triggering >= 2, "evaluation dataset needs at least two non-triggering cases"
assert behavioral >= 1, "evaluation dataset needs at least one behavioral expectation"
print(f"validated {len(evaluations)} skill evaluations")
