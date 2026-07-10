"""Execute a skill's JSON behavioral evals (``skills/<skill>/evals/evals.json``).

This is the JSON-driven counterpart to the hand-written pytest harness: instead
of a Python file per skill, each skill declares its behavioral tests as data and
this runner executes them through the same `harness` primitives. It keeps our
own check vocabulary -- ``logs_contains`` / ``workspace_contains`` (deterministic)
and ``should`` / ``should_not`` (LLM-judged) -- rather than any external schema.

Usage:
    python run_evals.py <skill>
    python run_evals.py <skill> --runner-target strix_halo+Windows
    python run_evals.py <skill> --test api-key-gate-bypassed-in-local-mode

``--runner-target`` restricts the run to the tests whose ``runners`` resolve to
that target id (see evals_common.py). CI passes it so each self-hosted runner
executes only the tests that asked for its device/OS labels; omitting it runs
every test in the file. Exit code is 0 only if every executed check passed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evals_common as ec  # noqa: E402
from harness import DEFAULT_MODEL, check_api_reachable, claude  # noqa: E402

# Fixed check order so output is stable and cheap deterministic checks run
# before the (billed) LLM-judged ones.
CHECK_ORDER = ("logs_contains", "workspace_contains", "should", "should_not")


def _run_one_test(skill: str, test: dict, model: str | None) -> bool:
    """Execute a single test case; return True iff all its checks passed."""
    test_id = test.get("id", "<unnamed>")
    prompt = test.get("prompt")
    if not prompt:
        print(f"  [FAIL] test {test_id!r}: missing 'prompt'", flush=True)
        return False

    expect = test.get("expect") or {}
    checks: list[tuple[str, str]] = []
    for kind in CHECK_ORDER:
        for arg in expect.get(kind, []) or []:
            checks.append((kind, arg))
    if not checks:
        print(f"  [FAIL] test {test_id!r}: no expectations declared", flush=True)
        return False

    print(f"\n=== [{skill}] {test_id} ===", flush=True)
    with claude(model, skill=skill) as agent:
        setup_files = ((test.get("setup") or {}).get("files")) or {}
        if setup_files:
            agent.stage_files(setup_files)
        run = agent.prompt(prompt)

        passed = True
        for kind, arg in checks:
            ok, detail = run.evaluate(kind, arg)
            print(f"  [{'PASS' if ok else 'FAIL'}] ({kind}) {detail}", flush=True)
            passed = passed and ok
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skill", help="Skill name (folder under skills/).")
    parser.add_argument(
        "--runner-target",
        help="Only run tests whose runners resolve to this target id "
        "(e.g. 'strix_halo+Windows'). Omit to run every test.",
    )
    parser.add_argument("--test", help="Run only the test with this id.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Agent + judge model (default: {DEFAULT_MODEL}; capped to sonnet in CI).",
    )
    args = parser.parse_args(argv)

    if not ec.has_evals(args.skill):
        print(
            f"error: no evals for skill {args.skill!r} "
            f"(expected {ec.evals_path(args.skill)})",
            file=sys.stderr,
        )
        return 2

    data = ec.load_evals(args.skill)
    tests = data["tests"]

    if args.runner_target:
        tests = ec.tests_for_target(data, args.runner_target)
        if not tests:
            print(
                f"error: no tests in {args.skill!r} target "
                f"'{args.runner_target}'. The CI matrix and evals.json have "
                f"drifted; regenerate the matrix.",
                file=sys.stderr,
            )
            return 2
    if args.test:
        tests = [t for t in tests if t.get("id") == args.test]
        if not tests:
            print(f"error: no test with id {args.test!r}", file=sys.stderr)
            return 2

    # Preflight once: the runs below make real (billed) API calls, so fail fast
    # with a clear message when the API is unreachable (e.g. off-network).
    ok, detail = check_api_reachable(args.model)
    if not ok:
        print(f"error: claude API not reachable -- {detail}", file=sys.stderr)
        return 1

    target_note = f" [target {args.runner_target}]" if args.runner_target else ""
    print(f"Running {len(tests)} test(s) for {args.skill!r}{target_note}", flush=True)

    results: list[tuple[str, bool]] = []
    for test in tests:
        results.append((test.get("id", "<unnamed>"), _run_one_test(args.skill, test, args.model)))

    print("\n--- summary ---", flush=True)
    for test_id, passed in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {test_id}", flush=True)
    failed = [tid for tid, ok in results if not ok]
    if failed:
        print(f"\n{len(failed)}/{len(results)} test(s) FAILED: {', '.join(failed)}", flush=True)
        return 1
    print(f"\nAll {len(results)} test(s) passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
