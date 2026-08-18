# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Agent staging and grading for behavior-mode eval runs.

One skill is copied into an isolated temp workspace, one prompt is run to
completion, and the result is graded against a case's expectations::

    from agent import claude

    with claude("opus", skill="local-ai-use") as agent:
        run = agent.prompt("Use local AI, then generate a cat to out.png.")
        checks = run.evaluate(
            files_exist=["out.png"],
            expected_behavior=["Download the SD-Turbo model"],
            unexpected_behavior=["Use the GenerateImage tool"],
        )

``evaluate`` reports every expectation instead of raising at the first
failure, because a run that took minutes and real tokens should not have to
be repeated to discover the second thing wrong with it. The asserting
variants (``logs_contains``, ``expects``, ...) are still here for skills whose
``evals/hooks.py`` needs to express a check the dataset format cannot.

Two things are deliberately *not* graded here. Routing is not: behavior mode
installs a single skill, so "did the right one fire" is unanswerable and
belongs to routing mode, which installs the whole catalog. And nothing checks
that the skill name appears in the transcript, which was the old stand-in for
a routing assertion and only ever proved the staged skill was visible.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from datasets import SKILLS_DIR  # noqa: E402

DEFAULT_MODEL = os.environ.get("EVAL_MODEL", "opus")
DEFAULT_EFFORT = os.environ.get("EVAL_EFFORT", "high")

# Automated runs are pinned to opus: a behavior run makes real cloud calls
# (agent run + LLM judge), so pinning the model keeps CI results comparable
# between runs. No override -- the pin is non-negotiable in CI.
AUTOMATED_MODEL = "opus"
_TRUTHY = {"1", "true", "yes", "on"}


def _safe_print(text: str) -> None:
    """Print, falling back to `errors="replace"` if the console can't encode `text`."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding), flush=True)


def is_automated_env() -> bool:
    """True under CI / an automated workflow (GitHub Actions sets both)."""
    return any(
        os.environ.get(var, "").strip().lower() in _TRUTHY
        for var in ("CI", "GITHUB_ACTIONS")
    )


def enforce_model_policy(model: str | None) -> str | None:
    """Coerce non-opus models to opus in CI; pass through otherwise."""
    if model is None or not is_automated_env() or "opus" in model.lower():
        return model
    _safe_print(
        f"[evals] automated run: coercing model '{model}' -> "
        f"'{AUTOMATED_MODEL}' to pin the CI model."
    )
    return AUTOMATED_MODEL


def claude_env() -> dict[str, str]:
    """Environment for `claude` subprocesses.

    Disable the CLI's internal retry loop by default so a network/auth problem
    (e.g. not connected to the network that can reach the API) fails fast
    instead of being retried into a long, confusing hang. The caller can still
    override by exporting ``CLAUDE_CODE_MAX_RETRIES``.
    """
    env = dict(os.environ)
    env.setdefault("CLAUDE_CODE_MAX_RETRIES", "0")
    return env


def check_api_reachable(model: str | None = DEFAULT_MODEL, timeout: int = 60) -> tuple[bool, str]:
    """Preflight: confirm the `claude` CLI can actually reach the API.

    Runs a trivial prompt with retries disabled so an unreachable API fails
    fast. Returns ``(ok, detail)`` where ``detail`` is a short human-readable
    reason on failure. Called once before the (expensive) runs so a suite can
    fail cleanly when off-network.
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        return False, "'claude' CLI not found on PATH"

    model = enforce_model_policy(model)
    cmd = [claude_bin, "-p", "--output-format", "json"]
    if model:
        cmd += ["--model", model]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            input="Reply with the single word: ok", timeout=timeout, env=claude_env(),
        )
    except subprocess.TimeoutExpired:
        return False, f"API preflight timed out after {timeout}s (is the network reachable?)"

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or f"exit code {proc.returncode}").strip()
        return False, detail[:500]
    return True, "ok"


def _stage_workspace(skill: str, seed: Path | None = None) -> Path:
    """Copy ``skill`` into an isolated temp workspace and return its path.

    ``seed`` is a directory of fixture files (a case's ``workspace``) whose
    *contents* land at the workspace root, so a case can hand the agent a
    starting file to edit rather than describing one in prose.
    """
    skill_src = SKILLS_DIR / skill
    if not (skill_src / "SKILL.md").is_file():
        raise FileNotFoundError(f"skill '{skill}' not found at {skill_src / 'SKILL.md'}")

    workspace = Path(tempfile.mkdtemp(prefix=f"behavior-{skill}-"))
    dest = workspace / ".claude" / "skills" / skill
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(skill_src, dest)

    if seed is not None:
        if not seed.is_dir():
            raise FileNotFoundError(f"workspace fixture directory not found: {seed}")
        shutil.copytree(seed, workspace, dirs_exist_ok=True)

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
        encoding="utf-8", input=prompt_text, env=claude_env(),
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


def _find_file(files: list[str], expected: str) -> str | None:
    """Return the workspace file that satisfies ``expected``, or None.

    An expectation matches anywhere in the tree: ``analyze_plan.md`` is
    satisfied by ``examples/simple_hip_test/analyze_plan.md``, and
    ``out/report.md`` by ``run-1/out/report.md``. Only whole path segments
    count, so ``plan.md`` does not match ``analyze_plan.md``.

    A case asserts that an artifact was produced; which directory the agent
    chose for it is usually its own call, and a plan written beside the fixture
    it describes is not a failed run. Pin the location down in the prompt when
    it matters, and the judged expectations can grade whether it was honored.
    """
    wanted = expected.replace("\\", "/").strip("/")
    if wanted.startswith("./"):
        wanted = wanted[2:]
    for rel in files:
        if rel == wanted or rel.endswith("/" + wanted):
            return rel
    return None


def _grade_with_llm(
    statement: str, run: "Run", judge_model: str | None, *, must_happen: bool
) -> tuple[bool, str]:
    """Ask a grader LLM whether the run satisfied a requirement.

    ``must_happen`` selects the polarity: ``True`` means the agent was required
    to do ``statement``, ``False`` means it was required *not* to. The judge
    grades the requirement itself and returns ``True`` when it is satisfied, so
    callers must never negate this verdict -- a judge shown a "must not"
    expectation reports the desired behavior as a pass, and negating that turns
    a correct run into a failure.

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
    if must_happen:
        requirement = f"The agent MUST have done this:\n{statement}"
        pass_means = 'Set "pass" to true if the agent did it, false if it did not.'
    else:
        requirement = f"The agent MUST NOT have done this:\n{statement}"
        pass_means = (
            'Set "pass" to true if the agent avoided it, false if the agent '
            "did it anyway. Absence of evidence that the agent did it counts "
            "as avoiding it, so the default verdict is true."
        )

    prompt_text = (
        "You are grading whether a coding agent's run satisfied one "
        "requirement. Judge only from the evidence below and (if needed) by "
        "reading files in the provided workspace directory: "
        f"{run.workspace}\n\n"
        f"REQUIREMENT:\n{requirement}\n\n"
        f"EVIDENCE:\n{evidence}\n\n"
        f'"pass" reports whether the requirement is satisfied. {pass_means} '
        "Do not invert the verdict for any reason.\n"
        "Respond with ONLY a single-line JSON object and nothing else: "
        '{"pass": true|false, "reason": "<one short sentence, no braces>"}'
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
            input=prompt_text, timeout=180, env=claude_env(),
        )
    except subprocess.TimeoutExpired:
        return False, "llm_judge timed out after 180s"

    try:
        payload = json.loads((proc.stdout or "").strip())
        verdict_text = payload.get("result", "") if isinstance(payload, dict) else ""
    except json.JSONDecodeError:
        verdict_text = (proc.stdout or "").strip()

    # A chatty judge may wrap the verdict in prose, and its reason may itself
    # contain braces (a regex quantifier, a quoted JSON snippet), so let the
    # decoder find object boundaries rather than matching braces textually.
    # Keep scanning so the last verdict-shaped object wins.
    decoder = json.JSONDecoder()
    verdict = None
    for i, ch in enumerate(verdict_text):
        if ch != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(verdict_text[i:])
        except ValueError:
            continue
        if isinstance(parsed, dict) and "pass" in parsed:
            verdict = parsed
    if verdict is None:
        return False, f"llm_judge gave no JSON verdict: {verdict_text[:200]!r}"

    satisfied = bool(verdict.get("pass"))
    reason = str(verdict.get("reason", "")).strip() or "(no reason given)"
    return satisfied, f"llm_judge: {reason}"


@dataclass
class Check:
    """One graded expectation from a case."""

    kind: str
    expectation: str
    passed: bool
    detail: str = ""


class Run:
    """The captured result of one agent run."""

    def __init__(self, *, workspace: Path, events: list[dict], judge_model: str | None) -> None:
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
        self.files = _list_workspace_files(workspace)
        self.tool_names = {name for name, _ in tool_uses if name}
        self.result_text = result_text

        # `command_text` is what the agent actually did (tool inputs + outputs),
        # used by the judge so the agent's prose ("I won't call DALL-E") cannot
        # create false signals.
        self.command_text = "\n".join([inp for _, inp in tool_uses] + tool_results)

        # `logs` is the full raw transcript, searchable for tool names, command
        # strings, and anything else a case wants to pin down.
        self.logs = "\n".join(json.dumps(ev, ensure_ascii=False) for ev in events)

    def evaluate(
        self,
        *,
        logs_contain: list[str] | tuple[str, ...] = (),
        files_exist: list[str] | tuple[str, ...] = (),
        expected_behavior: list[str] | tuple[str, ...] = (),
        unexpected_behavior: list[str] | tuple[str, ...] = (),
    ) -> list[Check]:
        """Grade every expectation and return all results, raising nothing.

        Deterministic checks run first so their output is on screen before the
        judge calls (which take a few seconds each) start.
        """
        checks: list[Check] = []

        for text in logs_contain:
            ok = text.lower() in self.logs.lower()
            checks.append(Check("logs_contain", text, ok))

        for path in files_exist:
            found = _find_file(self.files, path)
            if found is None:
                detail = f"workspace holds: {self.files or 'nothing'}"
            else:
                detail = "" if found == path else f"found at {found}"
            checks.append(Check("files_exist", path, found is not None, detail))

        for statement in expected_behavior:
            ok, reason = _grade_with_llm(statement, self, self.judge_model, must_happen=True)
            checks.append(Check("expected_behavior", statement, ok, reason))

        for statement in unexpected_behavior:
            ok, reason = _grade_with_llm(statement, self, self.judge_model, must_happen=False)
            checks.append(Check("unexpected_behavior", statement, ok, reason))

        for check in checks:
            suffix = f" -- {check.detail}" if check.detail else ""
            _safe_print(
                f"  [{'PASS' if check.passed else 'FAIL'}] "
                f"({check.kind}) {check.expectation}{suffix}"
            )
        return checks

    # Asserting variants, for an `evals/hooks.py` that needs a check the
    # dataset format cannot express. Each raises AssertionError on failure.

    def logs_contains(self, text: str) -> "Run":
        ok = text.lower() in self.logs.lower()
        self._report(ok, "logs_contains", f"transcript contains '{text}'")
        return self

    def workspace_contains(self, path: str) -> "Run":
        found = _find_file(self.files, path)
        detail = f"workspace contains '{path}'"
        if found is None:
            detail += f" (files: {self.files or 'none'})"
        self._report(found is not None, "workspace_contains", detail)
        return self

    def expects(self, statement: str) -> "Run":
        satisfied, reason = _grade_with_llm(statement, self, self.judge_model, must_happen=True)
        self._report(satisfied, "expected_behavior", f"{statement} -- {reason}")
        return self

    def expects_not(self, statement: str) -> "Run":
        satisfied, reason = _grade_with_llm(statement, self, self.judge_model, must_happen=False)
        self._report(satisfied, "unexpected_behavior", f"{statement} -- {reason}")
        return self

    def _report(self, passed: bool, kind: str, detail: str) -> None:
        _safe_print(f"  [{'PASS' if passed else 'FAIL'}] ({kind}) {detail}")
        assert passed, f"({kind}) {detail}"


class Agent:
    """A single agent session bound to an isolated, skill-staged workspace.

    Use as a context manager so the temp workspace is always cleaned up::

        with claude("opus", skill="local-ai-use") as agent:
            run = agent.prompt("...")
    """

    def __init__(
        self,
        model: str | None = DEFAULT_MODEL,
        *,
        skill: str,
        effort: str | None = DEFAULT_EFFORT,
        seed: Path | None = None,
    ) -> None:
        # Coerce here so the agent run and the LLM judge share the capped model.
        self.model = enforce_model_policy(model)
        self.skill = skill
        self.effort = effort
        self.seed = seed
        self.workspace: Path | None = None

    def __enter__(self) -> "Agent":
        self.workspace = _stage_workspace(self.skill, self.seed)
        return self

    def __exit__(self, *exc) -> None:
        if self.workspace is not None:
            shutil.rmtree(self.workspace, ignore_errors=True)
            self.workspace = None

    def prompt(self, text: str) -> Run:
        """Run ``text`` through the agent once and return a Run to grade."""
        if self.workspace is None:
            raise RuntimeError("Agent.prompt() must be called inside a 'with' block")

        _safe_print(f"\n[behavior] skill='{self.skill}' model='{self.model}': {text}")
        events = _run_agent(text, self.workspace, self.model, self.effort)
        return Run(workspace=self.workspace, events=events, judge_model=self.model)


def claude(
    model: str | None = DEFAULT_MODEL,
    *,
    skill: str,
    effort: str | None = DEFAULT_EFFORT,
    seed: Path | None = None,
) -> Agent:
    """Factory for a Claude-backed `Agent` (the only agent backend today)."""
    return Agent(model, skill=skill, effort=effort, seed=seed)
