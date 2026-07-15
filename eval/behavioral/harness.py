# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Behavioral-test harness for repo skills (local, pytest-based, non-CI).

A behavioral test runs a skill-driven prompt through the agent **once**, then
asserts what the agent *should* and *should not* have done. Tests read like:

    from harness import claude

    def test_image_generation():
        with claude("sonnet", skill="local-ai-use") as agent:
            run = agent.prompt("Use local AI, then generate a cat to out.png.")

            # Deterministic checks (cheap, fail fast).
            run.logs_contains("local-ai-use")
            run.workspace_contains("out.png")

            # Natural-language expectations (graded by an LLM judge).
            run.should("Download the SD-Turbo model")
            run.should_not("Use the GenerateImage tool")

`claude(model, skill=...)` returns an `Agent` context manager. Entering it
stages an isolated temp workspace (skill copied under
`<tmp>/.claude/skills/<skill>/`); leaving it deletes that workspace. `prompt()`
runs the agent once with tool permissions bypassed and returns a `Run`.

Every assertion on `Run` raises `AssertionError` on failure and prints a
`[PASS]`/`[FAIL]` line for visibility under `-s`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from claude_eval import SKILLS_DIR  # noqa: E402

DEFAULT_SKILL = os.environ.get("BEHAVIORAL_SKILL", "local-ai-use")

# Where per-run logs (transcript + check outcomes) are written so they can be
# uploaded and investigated after the fact -- on pass OR fail. Defaults to
# ``eval/behavioral/results/`` (git-ignored); override with the env var. The
# path is anchored to this file, not the cwd, so it is stable no matter where
# pytest is invoked from.
RESULTS_DIR = Path(
    os.environ.get(
        "BEHAVIORAL_RESULTS_DIR",
        str(Path(__file__).resolve().parent / "results"),
    )
)

# Distinguishes multiple prompt() calls within a single test so their logs
# don't clobber each other.
_RUN_COUNTERS: dict[str, int] = {}
DEFAULT_MODEL = os.environ.get("BEHAVIORAL_MODEL", "sonnet")
DEFAULT_EFFORT = os.environ.get("BEHAVIORAL_EFFORT", "high")

# Automated runs are capped at sonnet: a behavioral run makes real cloud calls
# (agent run + LLM judge), so a workflow picking an expensive model can quietly
# run up a large bill. No override -- the cap is non-negotiable in CI.
AUTOMATED_MODEL = "sonnet"
_TRUTHY = {"1", "true", "yes", "on"}


def _is_automated_env() -> bool:
    """True under CI / an automated workflow (GitHub Actions sets both)."""
    return any(
        os.environ.get(var, "").strip().lower() in _TRUTHY
        for var in ("CI", "GITHUB_ACTIONS")
    )


def _enforce_model_policy(model: str | None) -> str | None:
    """Coerce non-sonnet models to sonnet in CI; pass through otherwise."""
    if model is None or not _is_automated_env() or "sonnet" in model.lower():
        return model
    print(
        f"[behavioral] automated run: coercing model '{model}' -> "
        f"'{AUTOMATED_MODEL}' to cap token usage.",
        flush=True,
    )
    return AUTOMATED_MODEL


def _claude_env() -> dict[str, str]:
    """Environment for `claude` subprocesses.

    Disable the CLI's internal retry loop by default so a network/auth
    problem (e.g. not connected to the network that can reach the API)
    fails fast instead of being retried into a long, confusing hang. The
    caller can still override by exporting ``CLAUDE_CODE_MAX_RETRIES``.
    """
    env = dict(os.environ)
    env.setdefault("CLAUDE_CODE_MAX_RETRIES", "0")
    return env


def check_api_reachable(model: str | None = DEFAULT_MODEL, timeout: int = 60) -> tuple[bool, str]:
    """Preflight: confirm the `claude` CLI can actually reach the API.

    Runs a trivial prompt with retries disabled so an unreachable API fails
    fast. Returns ``(ok, detail)`` where ``detail`` is a short human-readable
    reason on failure. This is meant to be called once before the (expensive)
    behavioral runs so the suite can skip cleanly when off-network.
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        return False, "'claude' CLI not found on PATH"

    model = _enforce_model_policy(model)
    cmd = [claude_bin, "-p", "--output-format", "json"]
    if model:
        cmd += ["--model", model]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            input="Reply with the single word: ok", timeout=timeout, env=_claude_env(),
        )
    except subprocess.TimeoutExpired:
        return False, f"API preflight timed out after {timeout}s (is the network reachable?)"

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or f"exit code {proc.returncode}").strip()
        return False, detail[:500]
    return True, "ok"


def _stage_workspace(skill: str) -> Path:
    """Copy ``skill`` into an isolated temp workspace and return its path."""
    skill_src = SKILLS_DIR / skill
    if not (skill_src / "SKILL.md").is_file():
        raise FileNotFoundError(f"skill '{skill}' not found at {skill_src / 'SKILL.md'}")

    workspace = Path(tempfile.mkdtemp(prefix=f"behavioral-{skill}-"))
    dest = workspace / ".claude" / "skills" / skill
    dest.parent.mkdir(parents=True, exist_ok=True)
    # The skill's behavioral tests live under ``skills/<skill>/evals/``. They are
    # part of the skill folder but must NOT be staged into the agent's sandbox --
    # copying them in would pollute the workspace the agent (and the LLM judge)
    # sees. Exclude them (and stray caches) so the sandbox holds only the skill.
    shutil.copytree(skill_src, dest, ignore=shutil.ignore_patterns("evals", "__pycache__"))
    return workspace


def _run_agent(prompt_text: str, workspace: Path, model: str | None, effort: str | None) -> list[dict]:
    """Run the agent once in ``workspace`` and return the stream-json events."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError("'claude' CLI not found on PATH")

    cmd = [
        claude_bin, "-p",
        "--output-format", "stream-json", "--verbose",
        "--dangerously-skip-permissions",
        "--add-dir", str(workspace),
    ]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]

    proc = subprocess.run(
        cmd, cwd=str(workspace), capture_output=True, text=True,
        encoding="utf-8", input=prompt_text, env=_claude_env(),
    )

    events: list[dict] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not events:
        raise RuntimeError(
            f"claude exited with code {proc.returncode} and produced no "
            f"parseable stream-json output. stderr:\n{proc.stderr}"
        )
    return events


def _walk(obj, tool_uses, tool_results) -> None:
    """Collect (tool name, tool input) pairs and tool-result text from events."""
    if isinstance(obj, dict):
        otype = obj.get("type")
        if otype == "tool_use":
            tool_uses.append((str(obj.get("name", "")), json.dumps(obj.get("input", {}), ensure_ascii=False)))
        elif otype == "tool_result":
            content = obj.get("content")
            if isinstance(content, str):
                tool_results.append(content)
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and isinstance(c.get("text"), str):
                        tool_results.append(c["text"])
        for v in obj.values():
            _walk(v, tool_uses, tool_results)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, tool_uses, tool_results)


def _list_workspace_files(workspace: Path) -> list[str]:
    files: list[str] = []
    for p in sorted(workspace.rglob("*")):
        if ".claude" in p.relative_to(workspace).parts:
            continue
        if p.is_file():
            files.append(str(p.relative_to(workspace)).replace("\\", "/"))
    return files


def _grade_with_llm(statement: str, run: "Run", judge_model: str | None) -> tuple[bool, str]:
    """Ask a grader LLM whether ``statement`` is TRUE given the run's evidence.

    The grader may read files in the workspace (e.g. open out.png), so the
    workspace is added and tool permissions are bypassed for the grader too.
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        return False, "llm_judge skipped: 'claude' CLI not on PATH"

    cmd_text = run.command_text
    if len(cmd_text) > 4000:
        cmd_text = cmd_text[:4000] + "\n...[truncated]..."
    evidence = (
        f"Files in workspace:   {run.files or 'none'}\n"
        f"Tools the agent used: {sorted(run.tool_names) or 'none'}\n"
        f"--- Agent final message ---\n{run.result_text[:1500]}\n"
        f"--- Transcript commands/outputs (truncated) ---\n{cmd_text}\n"
    )
    prompt_text = (
        "You are grading whether a coding agent's run satisfied a specific "
        "expectation. Decide if the following statement is TRUE based on the "
        "evidence and (if needed) by reading files in the provided workspace "
        f"directory: {run.workspace}\n\n"
        f"STATEMENT TO EVALUATE:\n{statement}\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        "Respond with ONLY a single-line JSON object and nothing else: "
        '{"pass": true|false, "reason": "<one short sentence>"}'
    )
    cmd = [
        claude_bin, "-p",
        "--output-format", "json",
        "--dangerously-skip-permissions",
        "--add-dir", str(run.workspace),
    ]
    if judge_model:
        cmd += ["--model", judge_model]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            input=prompt_text, timeout=180, env=_claude_env(),
        )
    except subprocess.TimeoutExpired:
        return False, "llm_judge timed out after 180s"

    try:
        payload = json.loads((proc.stdout or "").strip())
        verdict_text = payload.get("result", "") if isinstance(payload, dict) else ""
    except json.JSONDecodeError:
        verdict_text = (proc.stdout or "").strip()

    match = re.search(r"\{.*\}", verdict_text, re.DOTALL)
    if not match:
        return False, f"llm_judge gave no JSON verdict: {verdict_text[:200]!r}"
    try:
        verdict = json.loads(match.group(0))
    except json.JSONDecodeError:
        return False, f"llm_judge verdict not valid JSON: {match.group(0)[:200]!r}"

    passed = bool(verdict.get("pass"))
    reason = str(verdict.get("reason", "")).strip() or "(no reason given)"
    return passed, f"llm_judge: {reason}"


def _run_log_dir(skill: str) -> Path:
    """A unique per-run directory under RESULTS_DIR for this test's logs.

    Named ``<skill>__<test_function>`` from ``PYTEST_CURRENT_TEST`` so logs are
    easy to trace back to their test. A second prompt() call in the same test
    gets a ``-2``, ``-3``, ... suffix so it doesn't clobber the first.
    """
    node = os.environ.get("PYTEST_CURRENT_TEST", "").split(" (")[0]
    func = node.split("::")[-1] if node else "run"
    func = re.sub(r"[^A-Za-z0-9._-]+", "_", func).strip("_") or "run"

    key = f"{skill}/{func}"
    idx = _RUN_COUNTERS.get(key, 0)
    _RUN_COUNTERS[key] = idx + 1
    suffix = "" if idx == 0 else f"-{idx + 1}"

    run_dir = RESULTS_DIR / f"{skill}__{func}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class Run:
    """The captured result of one agent run, with inline-asserting checks.

    Each check prints a ``[PASS]``/``[FAIL]`` line and raises ``AssertionError``
    on failure, so the owning pytest test fails at that line.

    Every run also writes its transcript and check outcomes under
    ``RESULTS_DIR`` up front, so a failing (raising) assertion still leaves a
    complete log behind for later investigation.
    """

    def __init__(
        self,
        *,
        workspace: Path,
        events: list[dict],
        judge_model: str | None,
        skill: str = DEFAULT_SKILL,
        prompt_text: str = "",
    ) -> None:
        tool_uses: list[tuple[str, str]] = []
        tool_results: list[str] = []
        for ev in events:
            _walk(ev, tool_uses, tool_results)

        result_text = ""
        for ev in events:
            if ev.get("type") == "result" and isinstance(ev.get("result"), str):
                result_text = ev["result"]

        self.workspace = workspace
        self.judge_model = judge_model
        self.skill = skill
        self.files = _list_workspace_files(workspace)
        self.tool_names = {name for name, _ in tool_uses if name}
        self.result_text = result_text

        # `command_text` is what the agent actually did (tool inputs + outputs),
        # used by the judge so the agent's prose ("I won't call DALL-E") cannot
        # create false signals.
        self.command_text = "\n".join([inp for _, inp in tool_uses] + tool_results)

        # `logs` is the full raw transcript, searchable for skill activation,
        # tool names, command strings, etc.
        self.logs = "\n".join(json.dumps(ev, ensure_ascii=False) for ev in events)

        # Persist the transcript immediately (before any check can raise) so the
        # log is always uploadable, pass or fail. `checks.log` is appended to as
        # each check runs.
        self._checks_path = self._persist(events, prompt_text)

    def logs_contains(self, text: str) -> "Run":
        ok = text.lower() in self.logs.lower()
        self._report(ok, "logs_contains", f"transcript contains '{text}'")
        return self

    def workspace_contains(self, path: str) -> "Run":
        ok = (self.workspace / path).is_file()
        detail = f"workspace contains '{path}'"
        if not ok:
            detail += f" (files: {self.files or 'none'})"
        self._report(ok, "workspace_contains", detail)
        return self

    def should(self, statement: str) -> "Run":
        observed, reason = _grade_with_llm(statement, self, self.judge_model)
        self._report(observed, "should", f"{statement} -- {reason}")
        return self

    def should_not(self, statement: str) -> "Run":
        observed, reason = _grade_with_llm(statement, self, self.judge_model)
        self._report(not observed, "should_not", f"{statement} -- {reason}")
        return self

    def _persist(self, events: list[dict], prompt_text: str) -> Path | None:
        """Write the transcript + a summary to this run's log dir.

        Returns the path to ``checks.log`` (created on first check) or ``None``
        if the results dir can't be written -- logging must never break a test.
        """
        try:
            run_dir = _run_log_dir(self.skill)
            (run_dir / "transcript.jsonl").write_text(
                "\n".join(json.dumps(ev, ensure_ascii=False) for ev in events) + "\n",
                encoding="utf-8",
            )
            summary = (
                f"skill:   {self.skill}\n"
                f"model:   {self.judge_model}\n"
                f"time:    {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"prompt:  {prompt_text}\n"
                f"tools:   {sorted(self.tool_names) or 'none'}\n"
                f"files:   {self.files or 'none'}\n"
                f"--- final message ---\n{self.result_text}\n"
            )
            (run_dir / "summary.txt").write_text(summary, encoding="utf-8")
            return run_dir / "checks.log"
        except OSError as exc:
            print(f"  [warn] could not write run logs: {exc}", flush=True)
            return None

    def _report(self, passed: bool, kind: str, detail: str) -> None:
        line = f"[{'PASS' if passed else 'FAIL'}] ({kind}) {detail}"
        print(f"  {line}", flush=True)
        # Append the outcome before asserting so a failing check is still
        # recorded in the uploaded log.
        if self._checks_path is not None:
            try:
                with self._checks_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass
        assert passed, f"({kind}) {detail}"


class Agent:
    """A single agent session bound to an isolated, skill-staged workspace.

    Use as a context manager so the temp workspace is always cleaned up::

        with claude("sonnet", skill="local-ai-use") as agent:
            run = agent.prompt("...")
    """

    def __init__(
        self,
        model: str | None = DEFAULT_MODEL,
        *,
        skill: str = DEFAULT_SKILL,
        effort: str | None = DEFAULT_EFFORT,
    ) -> None:
        # Coerce here so the agent run and the LLM judge share the capped model.
        self.model = _enforce_model_policy(model)
        self.skill = skill
        self.effort = effort
        self.workspace: Path | None = None

    def __enter__(self) -> "Agent":
        self.workspace = _stage_workspace(self.skill)
        return self

    def __exit__(self, *exc) -> None:
        if self.workspace is not None:
            shutil.rmtree(self.workspace, ignore_errors=True)
            self.workspace = None

    def prompt(self, text: str) -> Run:
        """Run ``text`` through the agent once and return a Run to assert on."""
        if self.workspace is None:
            raise RuntimeError("Agent.prompt() must be called inside a 'with' block")

        print(f"\n[behavioral] skill='{self.skill}' model='{self.model}': {text}", flush=True)
        events = _run_agent(text, self.workspace, self.model, self.effort)
        return Run(
            workspace=self.workspace,
            events=events,
            judge_model=self.model,
            skill=self.skill,
            prompt_text=text,
        )


def claude(
    model: str | None = DEFAULT_MODEL,
    *,
    skill: str = DEFAULT_SKILL,
    effort: str | None = DEFAULT_EFFORT,
) -> Agent:
    """Factory for a Claude-backed `Agent` (the only agent backend today)."""
    return Agent(model, skill=skill, effort=effort)
