"""Whether the Instinct sandbox is what these two skills need.

Four runs have now timed out there, twice before a compose file existed and
twice after. No report survives a timeout, so nothing says whether the
container even starts, whether it sees the card, or whether it can reach the
network -- the three things that would make the agent retry until the clock
runs out. Ask directly, before spending another two hours.
"""

from __future__ import annotations

import subprocess
import sys

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
        ["docker", "compose", "-f", compose, "run", "--rm", "default",
         "sh", "-lc",
         "echo DEVICES:; ls -l /dev/kfd /dev/dri 2>&1 | head -5; "
         "echo NET:; (getent hosts huggingface.co || echo no-dns) 2>&1; "
         "echo ROCM:; (rocm-smi --showid 2>&1 | head -3 || echo no-rocm-smi)"],
        capture_output=True, text=True, timeout=300,
    )
    print(f"exit {up.returncode}")
    print(up.stdout[-2000:] or "(no stdout)")
    if up.returncode != 0:
        print("--- stderr ---")
        print(up.stderr[-1500:])


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
