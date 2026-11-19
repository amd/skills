#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Assert the fixture runbook's install commands still match the real one.

The behavioral evaluations stage a shortened `helm/INSTALL.md` into the agent's
workspace and grade whether the skill grounds its commands in it. That is only
worth grading while those commands are the product's actual ones, and nothing
else would notice them going stale: the catalog grades the vendored skill,
where the real runbook does not exist.

The prose in the fixture is deliberately its own. Only the shell blocks holding
a `helm install` or `helm upgrade` are compared, and each must appear verbatim
in the real runbook.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE.parent / "evals" / "fixtures" / "checkout" / "helm" / "INSTALL.md"
REAL = HERE.parents[3] / "helm" / "INSTALL.md"

BLOCK = re.compile(r"```bash\n(.*?)```", re.DOTALL)


def main() -> int:
    if not REAL.is_file():
        print(f"skipped: {REAL} not present")
        return 0
    if not FIXTURE.is_file():
        print(f"fixture runbook missing: {FIXTURE}")
        return 1

    real = REAL.read_text(encoding="utf-8")
    commands = [
        block.strip("\n")
        for block in BLOCK.findall(FIXTURE.read_text(encoding="utf-8"))
        if re.search(r"^\s*helm (install|upgrade)", block, re.MULTILINE)
    ]
    if not commands:
        print("fixture runbook carries no helm commands to compare")
        return 1

    drifted = [c for c in commands if c not in real]
    if drifted:
        for command in drifted:
            first = command.splitlines()[0]
            print(f"drifted from helm/INSTALL.md: {first}")
        print(f"{len(drifted)} of {len(commands)} command(s) no longer match the real runbook")
        return 1

    print(f"validated {len(commands)} install commands against helm/INSTALL.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
