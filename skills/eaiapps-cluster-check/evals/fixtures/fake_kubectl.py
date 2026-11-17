#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Fixture-backed kubectl executable, used by the black-box Robot suite and the skill's
behavioral evals.

Fixture keys are matched against the full normalized argv (all arguments after the
program name, space-joined), so a fixture only answers the exact command it declares.
A generic key like ``get --raw`` therefore cannot satisfy an arbitrary
``kubectl get --raw <other-path>`` invocation — an unmodelled command fails loudly.
The one exception is the in-cluster gateway probe pod: its name carries a random suffix per
call, so every ``run e2e-gateway-probe-<hex> ...`` is answered by the ``run e2e-gateway-probe`` key.
"""

import datetime
import hashlib
import json
import os
import re
import sys
from pathlib import Path


def main() -> int:
    fixture_path = os.environ.get("CLUSTER_CHECK_FIXTURE")
    if not fixture_path:
        print("CLUSTER_CHECK_FIXTURE is not set", file=sys.stderr)
        return 99
    raw = Path(fixture_path).read_text()
    # Ages are judged against the clock at read time, not at write time. A transcript whose
    # timestamps were resolved when it was staged goes stale while the caller is still
    # deciding to run: the CSI lease is only fresh for 120s, so a graded agent that spends
    # longer than that reading its instructions would find healthy storage reported as a
    # warning. The Robot suite resolves these before writing and leaves nothing to replace.
    now = datetime.datetime.now(datetime.timezone.utc)
    for token, when in (
        ("__NOW__", now),
        ("__FIVE_MINUTES_AGO__", now - datetime.timedelta(minutes=5)),
    ):
        raw = raw.replace(token, when.strftime("%Y-%m-%dT%H:%M:%SZ"))
    transcript = json.loads(raw)
    command = re.sub(r"^run e2e-gateway-probe-[0-9a-f]{8} .*", "run e2e-gateway-probe", " ".join(sys.argv[1:]), flags=re.S)
    # The probe pod's cleanup runs after every probe and removes a pod --rm already removed.
    if re.match(r"^delete pod e2e-gateway-probe-[0-9a-f]{8} ", command):
        return 0
    entry = transcript.get(command)
    if entry is None:
        # Only evals/hooks.py sets this. The Robot suite drives the checker and nothing else,
        # so every command it issues is one a fixture should declare and an unmodelled one is
        # a fixture bug worth shouting about. A graded agent is not bounded that way: it reads
        # the cluster around the checker, and exit 99 on stderr would be the loudest thing in
        # the transcript, so the judge would grade the harness instead of the diagnosis. The
        # lenient answer is what kubectl says about a resource that is not there.
        if os.environ.get("CLUSTER_CHECK_FIXTURE_STRICT") == "0":
            return 0
        print(f"unmodelled kubectl command: {command}", file=sys.stderr)
        return 99
    if isinstance(entry, dict):
        # {"sequence": [...]}: successive identical commands get successive answers.
        counter = Path(fixture_path).with_suffix(f".{hashlib.sha256(command.encode()).hexdigest()[:12]}.count")
        index = int(counter.read_text()) if counter.exists() else 0
        counter.write_text(str(index + 1))
        entry = entry["sequence"][min(index, len(entry["sequence"]) - 1)]
    rc, stdout, stderr = entry
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    return int(rc)


if __name__ == "__main__":
    sys.exit(main())
