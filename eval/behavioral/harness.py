"""Behavioral-test harness for repo skills (local, pytest-based, non-CI).

A behavioral test runs a skill-driven prompt through the agent **once**, then
asserts what the agent *should* and *should not* have done. Tests read like:

    from harness import claude

    def test_image_generation():
        with claude("sonnet", skill="local-ai-use") as agent:
            run = agent.prompt("Use local AI, then generate a cat to out.png.")

            # Programmatic expectations (cheap, deterministic, fail fast).
            run.logs_contains("local-ai-use")
            run.workspace_contains("out.png")

            # Natural-language expectations (graded by an LLM judge).
            run.should("Download the SD-Turbo model")
            run.should_not("Use the GenerateImage tool")

`claude(model, skill=...)` returns an `Agent` context manager. Entering it
stages an isolated temp workspace (skill copied under
`<tmp>/.claude/skills/<skill>/`); leaving it deletes that workspace. `prompt()`
runs the agent once with tool permissions bypassed and returns a `Run`.

Every assertion on `Run` raises `AssertionError` on failure (so the test fails
at that line) and prints a `[PASS]`/`[FAIL]` line for visibility under `-s`:

  * `logs_contains(text)` / `workspace_contains(path)` -- deterministic checks
    against the captured transcript and the workspace files.
  * `should(statement)` / `should_not(statement)` -- a natural-language claim
    graded by an LLM judge over the captured evidence (transcript, downloaded
    model deltas, workspace files); polarity lives in the verb.

Lemonade model *downloads* are global to the Lemonade install and are not
isolated -- that is intentional, since "did the agent download model X?" is
part of what we grade.
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
import urllib.error
import urllib.request
from pathlib import Path

# Reuse the skill location + listing helpers from the existing eval harness.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from claude_eval import SKILLS_DIR, list_available_skills  # noqa: E402

DEFAULT_SKILL = os.environ.get("BEHAVIORAL_SKILL", "local-ai-use")
DEFAULT_MODEL = os.environ.get("BEHAVIORAL_MODEL", "sonnet")
DEFAULT_EFFORT = os.environ.get("BEHAVIORAL_EFFORT", "high")
DEFAULT_JUDGE_MODEL = os.environ.get("BEHAVIORAL_JUDGE_MODEL") or DEFAULT_MODEL

LEMONADE_HOST = os.environ.get("LEMONADE_HOST", "127.0.0.1")
LEMONADE_PORT = int(os.environ.get("LEMONADE_PORT", "13305"))


# --------------------------------------------------------------------------- #
# Lemonade state + prerequisites                                              #
# --------------------------------------------------------------------------- #
def _http_get(url: str, timeout_s: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout_s) as r:  # noqa: S310
        return r.status, r.read()


def claude_available() -> bool:
    return shutil.which("claude") is not None


def lemonade_server_reachable(host: str = LEMONADE_HOST, port: int = LEMONADE_PORT) -> bool:
    try:
        status, _ = _http_get(f"http://{host}:{port}/api/v1/health", timeout_s=3.0)
        return status == 200
    except (urllib.error.URLError, OSError):
        return False


def lemonade_downloaded_models(host: str = LEMONADE_HOST, port: int = LEMONADE_PORT) -> set[str]:
    """Set of locally-downloaded model IDs (CLI first, HTTP fallback)."""
    try:
        out = subprocess.run(
            ["lemonade", "list", "--downloaded", "--json"],
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout
        data = json.loads(out)
        if isinstance(data, list):
            return {m.get("id", "") for m in data if isinstance(m, dict) and m.get("id")}
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError, OSError):
        pass

    try:
        status, body = _http_get(f"http://{host}:{port}/api/v1/models", timeout_s=5)
        if status == 200:
            data = json.loads(body)
            return {
                m.get("id", "") for m in data.get("data", [])
                if isinstance(m, dict) and m.get("downloaded") and m.get("id")
            }
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        pass

    return set()


# --------------------------------------------------------------------------- #
# Workspace staging + agent run                                               #
# --------------------------------------------------------------------------- #
def _stage_workspace(skill: str) -> Path:
    skill_src = SKILLS_DIR / skill
    if not (skill_src / "SKILL.md").is_file():
        available = list_available_skills()
        hint = f" Available: {', '.join(available)}." if available else ""
        raise FileNotFoundError(f"skill '{skill}' not found at {skill_src / 'SKILL.md'}.{hint}")

    workspace = Path(tempfile.mkdtemp(prefix=f"behavioral-{skill}-"))
    dest = workspace / ".claude" / "skills" / skill
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(skill_src, dest)
    return workspace


def _run_agent(
    prompt_text: str,
    workspace: Path,
    model: str | None,
    effort: str | None,
) -> tuple[float, list[dict]]:
    """Run the agent once in ``workspace``; return (wall_s, stream-json events)."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError("'claude' CLI not found on PATH")

    cmd = [
        claude_bin, "-p", prompt_text,
        "--output-format", "stream-json", "--verbose",
        "--dangerously-skip-permissions",
        "--add-dir", str(workspace),
    ]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]

    start = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=str(workspace), capture_output=True, text=True,
        encoding="utf-8", stdin=subprocess.DEVNULL,
    )
    elapsed = time.perf_counter() - start

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
    return elapsed, events


# --------------------------------------------------------------------------- #
# Transcript parsing                                                          #
# --------------------------------------------------------------------------- #
def _walk(obj, tool_uses, tool_results, texts) -> None:
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
        elif otype == "text" and isinstance(obj.get("text"), str):
            texts.append(obj["text"])
        for v in obj.values():
            _walk(v, tool_uses, tool_results, texts)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, tool_uses, tool_results, texts)


def _list_workspace_files(workspace: Path) -> list[str]:
    files: list[str] = []
    for p in sorted(workspace.rglob("*")):
        if ".claude" in p.relative_to(workspace).parts:
            continue
        if p.is_file():
            files.append(str(p.relative_to(workspace)).replace("\\", "/"))
    return files


# --------------------------------------------------------------------------- #
# LLM-judge                                                                   #
# --------------------------------------------------------------------------- #
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
        f"Downloaded models before run: {sorted(run.pre_models) or 'unknown'}\n"
        f"Downloaded models after run:  {sorted(run.post_models) or 'unknown'}\n"
        f"Newly downloaded this run:    {sorted(run.new_models) or 'none'}\n"
        f"Files in workspace:           {run.files or 'none'}\n"
        f"Tools the agent used:         {sorted(run.tool_names) or 'none'}\n"
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
        claude_bin, "-p", prompt_text,
        "--output-format", "json",
        "--dangerously-skip-permissions",
        "--add-dir", str(run.workspace),
    ]
    if judge_model:
        cmd += ["--model", judge_model]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            stdin=subprocess.DEVNULL, timeout=180,
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


# --------------------------------------------------------------------------- #
# Run: the assertion surface                                                  #
# --------------------------------------------------------------------------- #
class Run:
    """The captured result of one agent run, with inline-asserting checks.

    Each check prints a ``[PASS]``/``[FAIL]`` line and raises ``AssertionError``
    on failure, so the owning pytest test fails at that line.
    """

    def __init__(
        self,
        *,
        workspace: Path,
        events: list[dict],
        pre_models: set[str],
        post_models: set[str],
        wall_time_s: float,
        judge_model: str | None,
    ) -> None:
        tool_uses: list[tuple[str, str]] = []
        tool_results: list[str] = []
        texts: list[str] = []
        for ev in events:
            _walk(ev, tool_uses, tool_results, texts)

        result_text = ""
        for ev in events:
            if ev.get("type") == "result" and isinstance(ev.get("result"), str):
                result_text = ev["result"]

        self.workspace = workspace
        self.wall_time_s = wall_time_s
        self.judge_model = judge_model

        self.pre_models = pre_models
        self.post_models = post_models
        self.new_models = post_models - pre_models

        self.files = _list_workspace_files(workspace)
        self.tool_names = {name for name, _ in tool_uses if name}
        self.result_text = result_text
        self.assistant_text = "\n".join(texts)

        # `command_text` is what the agent actually did (tool inputs + outputs),
        # used by the judge so the agent's prose ("I won't call DALL-E") cannot
        # create false signals.
        self.command_text = "\n".join([inp for _, inp in tool_uses] + tool_results)

        # `logs` is the full raw transcript, searchable for skill activation,
        # tool names, command strings, etc.
        self.logs = "\n".join(json.dumps(ev, ensure_ascii=False) for ev in events)

    # -- deterministic checks ------------------------------------------------ #
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

    # -- LLM-judged checks --------------------------------------------------- #
    def should(self, statement: str) -> "Run":
        observed, reason = _grade_with_llm(statement, self, self.judge_model)
        self._report(observed, "should", f"{statement} -- {reason}")
        return self

    def should_not(self, statement: str) -> "Run":
        observed, reason = _grade_with_llm(statement, self, self.judge_model)
        self._report(not observed, "should_not", f"{statement} -- {reason}")
        return self

    # -- internals ----------------------------------------------------------- #
    def _report(self, passed: bool, kind: str, detail: str) -> None:
        print(f"  [{'PASS' if passed else 'FAIL'}] ({kind}) {detail}", flush=True)
        assert passed, f"({kind}) {detail}"


# --------------------------------------------------------------------------- #
# Agent: the context manager that owns an isolated workspace                  #
# --------------------------------------------------------------------------- #
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
        judge_model: str | None = DEFAULT_JUDGE_MODEL,
    ) -> None:
        self.model = model
        self.skill = skill
        self.effort = effort
        self.judge_model = judge_model
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
        pre_models = lemonade_downloaded_models()
        wall_s, events = _run_agent(text, self.workspace, self.model, self.effort)
        post_models = lemonade_downloaded_models()

        return Run(
            workspace=self.workspace,
            events=events,
            pre_models=pre_models,
            post_models=post_models,
            wall_time_s=wall_s,
            judge_model=self.judge_model,
        )


def claude(
    model: str | None = DEFAULT_MODEL,
    *,
    skill: str = DEFAULT_SKILL,
    effort: str | None = DEFAULT_EFFORT,
    judge_model: str | None = DEFAULT_JUDGE_MODEL,
) -> Agent:
    """Factory for a Claude-backed `Agent` (the only agent backend today)."""
    return Agent(model, skill=skill, effort=effort, judge_model=judge_model)
