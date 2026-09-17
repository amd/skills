"""Whether the Instinct sandbox is what these two skills need.

Four runs have now timed out there, twice before a compose file existed and
twice after. No report survives a timeout, so nothing says whether the
container even starts, whether it sees the card, or whether it can reach the
network -- the three things that would make the agent retry until the clock
runs out. Ask directly, before spending another two hours.
"""

from __future__ import annotations

import os
import subprocess
import sys

# The behavioral legs pass `--skills-dir 'skills/*'`; without the same here the
# default glob matches nothing and every lookup reports the skill missing,
# which looks like a finding and is only this script being misconfigured.
os.environ.setdefault("SKILLSCOPE_SKILLS", "skills/*")

SKILLS = ("hyperloom-workload-optimizer", "serving-llms-on-instinct")


def declared() -> None:
    """Does skillscope resolve the compose file we added?"""
    from skillscope import config
    from skillscope.engine import sandbox

    for skill in SKILLS:
        try:
            spec = sandbox.for_skill(skill)
        except SystemExit as exc:
            print(f"{skill:32s} -> REFUSED: {exc}")
            continue
        print(f"{skill:32s} -> {spec}")
        path = config.active().skill_path(skill)
        print(f"{'':32s}    skill at {path}")


def container() -> None:
    """Start the declared sandbox by hand and ask it three questions."""
    compose = f"skills/{SKILLS[0]}/evals/compose.yaml"
    up = subprocess.run(
        ["docker", "compose", "-f", compose, "run", "--rm", "--quiet-pull", "default",
         "sh", "-lc",
         "echo DEVICES:; ls -l /dev/kfd /dev/dri 2>&1 | head -5; "
         "echo NET:; (getent hosts huggingface.co || echo no-dns) 2>&1; "
         "echo ROCM:; (rocm-smi --showid 2>&1 | head -3 || echo no-rocm-smi)"],
        capture_output=True, text=True, timeout=900,
    )
    print(f"exit {up.returncode}")
    print(up.stdout[-2000:] or "(no stdout)")
    if up.returncode != 0:
        # Pull progress is thousands of identical lines and buries the cause,
        # so keep only what is not it.
        noise = ("Extracting", "Downloading", "Pulling", "Waiting", "Verifying")
        lines = [
            ln for ln in up.stderr.splitlines()
            if ln.strip() and not any(n in ln for n in noise)
        ]
        print("--- stderr (pull progress removed) ---")
        print("\n".join(lines[-25:]) or "(nothing but pull progress)")


if __name__ == "__main__":
    print("=== what skillscope resolves ===")
    declared()
    print()
    print("=== what the container actually has ===")
    try:
        container()
    except Exception as exc:  # noqa: BLE001 -- the failure is the result
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(0)
