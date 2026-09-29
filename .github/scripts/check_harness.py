# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Refuse to measure with a harness that is not the one under test.

Every job in the inspect-engine trial installs skillscope from a branch and
then grades skills with it. A stale or partial install does not fail loudly --
it produces a clean-looking report of the behaviour the branch was meant to
change, which is worse than no report at all. So each job checks first, and
this is that check.

One file, not one block per job. The Linux and Windows jobs each carried their
own copy, they drifted, and the Windows one went on asserting the pre-removal
engine dispatch until it failed a run on hardware that is slow to book. A
second copy of a check is a second thing to forget.

It also has to survive PowerShell. The Windows job has no heredoc, so its copy
lived on a single `python -c` line, and an f-string's braces read to PowerShell
as a ScriptBlock -- which cost another run. A file has no quoting problem.

Exits non-zero with the reason on the first failed expectation.
"""

from __future__ import annotations

import inspect
import sys


def _fail(reason: str) -> None:
    print(f"error: {reason}", file=sys.stderr)
    raise SystemExit(1)


def check() -> None:
    import inspect_ai

    from skillscope import cli
    from skillscope.engine import no_sandbox, tools, verify
    from skillscope.engine import routing as engine_routing
    from skillscope.engine import verify as engine_verify

    # --- the engine list -------------------------------------------------
    if "legacy" in cli.ENGINES:
        _fail("the legacy engine is still present")
    if "inspect" in cli.ENGINES:
        _fail("the harness-independent engine is still present")
    if set(cli.INSPECT_ENGINES) != set(cli.ENGINES):
        _fail("not every engine runs under inspect_ai")
    if set(cli.ROUTING_ENGINES) != set(cli.ENGINES):
        _fail("routing does not run on every engine")
    if cli.DEFAULT_ENGINE != "claude-code-no-sandbox":
        _fail(
            f"the default engine is {cli.DEFAULT_ENGINE!r}; Windows can only run "
            "claude-code-no-sandbox, so a different default is a Linux-only "
            "assumption"
        )

    # --- the dispatch ----------------------------------------------------
    #
    # A branch list that falls behind the choices list is the defect this
    # guards: runs asked for one agent, got another, and the report named the
    # one they had asked for.
    behavioral = inspect.getsource(cli.cmd_behavioral)
    for engine in cli.ENGINES:
        if 'args.engine == "%s"' % engine not in behavioral:
            _fail(f"no behavioral branch for {engine}")
    if "has no behavioral leg" not in behavioral:
        _fail("the behavioral dispatch still has a silent default")

    # --- where a case runs, and in what --------------------------------
    if not hasattr(tools, "workdir_path"):
        _fail("no workdir_path: the sandboxed leg cannot be told where to work")
    if "cwd=" not in inspect.getsource(verify.build_task):
        _fail("the claude-code leg still runs at the container root")
    if not hasattr(no_sandbox, "install_skill"):
        _fail("the host leg runs the CLI with no skill installed")
    if "FileNotFoundError" not in inspect.getsource(tools.shell_prefix):
        _fail("the shell probe dies where there is no bash")

    # --- what this trial exists to measure -------------------------------
    if not hasattr(engine_routing, "routing_decision"):
        _fail("routing cannot stop at the decision")
    if not hasattr(engine_routing, "host_stop_when_factory"):
        _fail("the host leg still works on after it decides")
    if engine_routing.SANDBOX_BUDGET_FACTOR != 2:
        _fail("the routing budget predates its recalibration")
    if "effort" not in inspect.signature(engine_verify.build_task).parameters:
        _fail("the sandboxed behavioral leg drops --effort")
    if "effort=" not in inspect.getsource(engine_routing._solver):
        _fail("the sandboxed routing leg drops --effort")
    if not hasattr(engine_verify, "files_to_restore"):
        _fail("the sandboxed room is missing most of each skill")

    # A hook that does not run leaves no trace: the case is graded as though
    # its setup happened. On a shared GPU runner that means leaked containers
    # holding memory into whatever runs next.
    from skillscope.engine import hooks as engine_hooks

    if not hasattr(engine_hooks, "setup_solver"):
        _fail("evals/hooks.py setup and teardown would be skipped")
    behavioral_src = inspect.getsource(
        __import__("skillscope.engine.behavioral", fromlist=["x"]).build_task
    )
    if "cleanup=hooks.cleanup_fn" not in behavioral_src:
        _fail("the behavioral task does not wire teardown to Task.cleanup")

    # --- counting, checked by behaviour ----------------------------------
    #
    # A bridged turn arrives on the sample and in the transcript as two
    # objects with different message ids; keying on those counted every call
    # twice. There is no version to read, and the count is the run's output.
    class _Call:
        def __init__(self, ident: str) -> None:
            self.id, self.function, self.arguments = ident, "Bash", {}

    class _Msg:
        def __init__(self, ident: str, calls: list) -> None:
            self.id, self.role, self.tool_calls = ident, "assistant", calls

    class _Event:
        def __init__(self, message) -> None:
            self.output = type("O", (), {"message": message})()

    class _Sample:
        def __init__(self, messages, events) -> None:
            self.messages, self.events = messages, events

    duplicated = _Sample(
        [_Msg("adopted", [_Call("toolu_1")])],
        [_Event(_Msg("transcript", [_Call("toolu_1")]))],
    )
    if len(list(engine_routing._tool_calls(duplicated))) != 1:
        _fail("one bridged turn is still counted twice")

    print(f"harness is current (inspect_ai {inspect_ai.__version__})")


if __name__ == "__main__":
    check()
