"""Behavioral-test harness for repo skills (local, pytest-based, non-CI).

A behavioral test runs a skill-driven prompt through the agent **once**, then
asserts what the agent *should* and *should not* have done. Tests read like:

    from harness import prompt, model_downloaded, file_is_png

    def test_image_generation():
        run = prompt("Use local AI, then generate an image of a cat to out.png.")
        run.should(model_downloaded("SD-Turbo"))
        run.should_not(model_newly_downloaded("kokoro-v1"))
        run.should(file_is_png("out.png"))
        run.should("the generated image actually depicts a cat")

`should(...)` / `should_not(...)` accept either:

  * a natural-language string -> graded by an LLM judge over the captured
    evidence (transcript, downloaded-model deltas, workspace files, ...), or
  * a deterministic `Check` helper (model_downloaded, file_is_png, tool_used,
    transcript_matches, ...) -> graded by inspecting real state.

Grading polarity lives in the verb: `should(x)` passes when x is true,
`should_not(x)` passes when x is false. Each call records a result; the
`conftest.py` finalizer prints a per-item table, writes a results JSON, and
fails the test if any item failed.

The agent runs in an **isolated temp workspace** (skill staged under
`<tmp>/.claude/skills/<skill>/`) with tool permissions bypassed, so the work
actually happens and any AGENTS.md edits / generated assets stay out of the
repo. Lemonade model *downloads* are global to the Lemonade install and are not
isolated -- that is intentional, since "did the agent download model X?" is part
of what we grade.
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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Reuse the skill location + listing helpers from the existing eval harness.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from claude_eval import SKILLS_DIR, list_available_skills  # noqa: E402

DEFAULT_SKILL = "local-ai-use"
DEFAULT_MODEL = os.environ.get("BEHAVIORAL_MODEL", "sonnet")
DEFAULT_EFFORT = os.environ.get("BEHAVIORAL_EFFORT", "high")
DEFAULT_JUDGE_MODEL = os.environ.get("BEHAVIORAL_JUDGE_MODEL") or DEFAULT_MODEL

LEMONADE_HOST = os.environ.get("LEMONADE_HOST", "127.0.0.1")
LEMONADE_PORT = int(os.environ.get("LEMONADE_PORT", "13305"))

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# Runs created during the currently-executing test. The conftest autouse fixture
# resets this at setup and drains it (evaluate + report + assert) at teardown.
_ACTIVE_RUNS: list["Run"] = []


# --------------------------------------------------------------------------- #
# Lemonade state + prerequisites                                              #
# --------------------------------------------------------------------------- #
def _http_get(url: str, timeout_s: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout_s) as r:  # noqa: S310
        return r.status, r.read()


def lemonade_server_reachable(host: str = LEMONADE_HOST, port: int = LEMONADE_PORT) -> bool:
    try:
        status, _ = _http_get(f"http://{host}:{port}/api/v1/health", timeout_s=3.0)
        return status == 200
    except (urllib.error.URLError, OSError):
        return False


def claude_available() -> bool:
    return shutil.which("claude") is not None


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


def _model_matches(target: str, models: set[str]) -> bool:
    """Case-insensitive match of ``target`` against model IDs (exact or substring)."""
    t = target.lower()
    for m in models:
        ml = m.lower()
        if t == ml or t in ml or ml in t:
            return True
    return False


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
# Transcript parsing + evidence                                               #
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


@dataclass
class Evidence:
    workspace: Path
    pre_models: set[str]
    post_models: set[str]
    new_models: set[str]
    files: list[str]
    agents_md_text: str
    tool_names: set[str]
    command_text: str
    result_text: str
    assistant_text: str


def _build_evidence(
    workspace: Path,
    pre_models: set[str],
    post_models: set[str],
    events: list[dict],
) -> Evidence:
    tool_uses: list[tuple[str, str]] = []
    tool_results: list[str] = []
    texts: list[str] = []
    for ev in events:
        _walk(ev, tool_uses, tool_results, texts)

    result_text = ""
    for ev in events:
        if ev.get("type") == "result" and isinstance(ev.get("result"), str):
            result_text = ev["result"]

    agents_md = workspace / "AGENTS.md"
    # command_text is what the agent actually did (tool inputs + outputs) so the
    # agent's prose ("I won't call DALL-E") doesn't create false signals.
    command_text = "\n".join([inp for _, inp in tool_uses] + tool_results)
    return Evidence(
        workspace=workspace,
        pre_models=pre_models,
        post_models=post_models,
        new_models=post_models - pre_models,
        files=_list_workspace_files(workspace),
        agents_md_text=agents_md.read_text(encoding="utf-8", errors="replace") if agents_md.is_file() else "",
        tool_names={name for name, _ in tool_uses if name},
        command_text=command_text,
        result_text=result_text,
        assistant_text="\n".join(texts),
    )


# --------------------------------------------------------------------------- #
# Deterministic checks (positive predicates: True == "the thing is true")     #
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    """A deterministic expectation: ``fn(evidence) -> (observed_true, reason)``."""
    description: str
    fn: Callable[[Evidence], tuple[bool, str]]


def model_downloaded(model: str) -> Check:
    def fn(ev: Evidence) -> tuple[bool, str]:
        ok = _model_matches(model, ev.post_models)
        return ok, f"{model} {'present' if ok else 'absent'} in downloaded models"
    return Check(f"model '{model}' is downloaded locally", fn)


def model_newly_downloaded(model: str) -> Check:
    def fn(ev: Evidence) -> tuple[bool, str]:
        ok = _model_matches(model, ev.new_models)
        return ok, (
            f"{model} {'was' if ok else 'was not'} newly downloaded "
            f"(new this run: {sorted(ev.new_models) or 'none'})"
        )
    return Check(f"model '{model}' was downloaded during the run", fn)


def file_exists(path: str) -> Check:
    def fn(ev: Evidence) -> tuple[bool, str]:
        ok = (ev.workspace / path).is_file()
        return ok, f"{path} {'exists' if ok else 'missing'}"
    return Check(f"file '{path}' exists", fn)


def file_is_png(path: str) -> Check:
    def fn(ev: Evidence) -> tuple[bool, str]:
        target = ev.workspace / path
        if not target.is_file():
            return False, f"{path} missing"
        try:
            head = target.read_bytes()[:8]
        except OSError as exc:
            return False, f"{path} unreadable: {exc}"
        ok = head == PNG_MAGIC
        return ok, f"{path} {'is a valid PNG' if ok else 'is not a PNG (bad magic bytes)'}"
    return Check(f"file '{path}' is a valid PNG", fn)


def agents_md_contains(text: str) -> Check:
    def fn(ev: Evidence) -> tuple[bool, str]:
        ok = text in ev.agents_md_text
        return ok, f"AGENTS.md {'contains' if ok else 'missing'} '{text}'"
    return Check(f"AGENTS.md contains '{text}'", fn)


def tool_used(tool: str) -> Check:
    def fn(ev: Evidence) -> tuple[bool, str]:
        ok = any(tool.lower() == t.lower() for t in ev.tool_names)
        return ok, (
            f"tool '{tool}' {'was used' if ok else 'not used'} "
            f"(tools seen: {sorted(ev.tool_names) or 'none'})"
        )
    return Check(f"tool '{tool}' was used", fn)


def transcript_matches(pattern: str) -> Check:
    def fn(ev: Evidence) -> tuple[bool, str]:
        ok = re.search(pattern, ev.command_text, re.IGNORECASE) is not None
        return ok, f"pattern /{pattern}/ {'found' if ok else 'not found'} in transcript commands"
    return Check(f"transcript commands match /{pattern}/", fn)


# --------------------------------------------------------------------------- #
# LLM-judge fallback                                                          #
# --------------------------------------------------------------------------- #
def _evidence_summary(ev: Evidence, max_chars: int = 4000) -> str:
    cmd = ev.command_text
    if len(cmd) > max_chars:
        cmd = cmd[:max_chars] + "\n...[truncated]..."
    return (
        f"Downloaded models before run: {sorted(ev.pre_models) or 'unknown'}\n"
        f"Downloaded models after run:  {sorted(ev.post_models) or 'unknown'}\n"
        f"Newly downloaded this run:    {sorted(ev.new_models) or 'none'}\n"
        f"Files in workspace:           {ev.files or 'none'}\n"
        f"Tools the agent used:         {sorted(ev.tool_names) or 'none'}\n"
        f"--- Agent final message ---\n{ev.result_text[:1500]}\n"
        f"--- Transcript commands/outputs (truncated) ---\n{cmd}\n"
    )


def _grade_with_llm(statement: str, ev: Evidence, judge_model: str | None) -> tuple[bool, str]:
    """Ask a grader LLM whether ``statement`` is TRUE given the evidence.

    The grader may read files in the workspace (e.g. open out.png), so the
    workspace is added and tool permissions are bypassed for the grader too.
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        return False, "llm_judge skipped: 'claude' CLI not on PATH"

    workspace = ev.workspace
    prompt_text = (
        "You are grading whether a coding agent's run satisfied a specific "
        "expectation. Decide if the following statement is TRUE based on the "
        "evidence and (if needed) by reading files in the provided workspace "
        f"directory: {workspace}\n\n"
        f"STATEMENT TO EVALUATE:\n{statement}\n\n"
        f"EVIDENCE:\n{_evidence_summary(ev)}\n\n"
        "Respond with ONLY a single-line JSON object and nothing else: "
        '{"pass": true|false, "reason": "<one short sentence>"}'
    )
    cmd = [
        claude_bin, "-p", prompt_text,
        "--output-format", "json",
        "--dangerously-skip-permissions",
        "--add-dir", str(workspace),
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
# Results + Run                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    kind: str          # "should" | "should_not"
    description: str
    method: str        # "deterministic" | "llm_judge"
    status: str        # "pass" | "fail"
    reason: str


@dataclass
class Run:
    prompt_text: str
    skill: str
    model: str | None
    effort: str | None
    judge_model: str | None
    evidence: Evidence
    wall_time_s: float
    results: list[Result] = field(default_factory=list)

    def _expect(self, kind: str, expectation: "str | Check") -> Result:
        if isinstance(expectation, Check):
            observed, reason = expectation.fn(self.evidence)
            description, method = expectation.description, "deterministic"
        else:
            observed, reason = _grade_with_llm(expectation, self.evidence, self.judge_model)
            description, method = expectation, "llm_judge"

        # Polarity from the verb: should -> pass if true; should_not -> pass if false.
        passed = observed if kind == "should" else (not observed)
        result = Result(kind, description, method, "pass" if passed else "fail", reason)
        self.results.append(result)
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] ({'judge' if method == 'llm_judge' else kind}) {description}: {reason}", flush=True)
        return result

    def should(self, expectation: "str | Check") -> Result:
        return self._expect("should", expectation)

    def should_not(self, expectation: "str | Check") -> Result:
        return self._expect("should_not", expectation)

    def cleanup(self) -> None:
        shutil.rmtree(self.evidence.workspace, ignore_errors=True)


def prompt(
    text: str,
    *,
    skill: str = DEFAULT_SKILL,
    model: str | None = DEFAULT_MODEL,
    effort: str | None = DEFAULT_EFFORT,
    judge_model: str | None = DEFAULT_JUDGE_MODEL,
) -> Run:
    """Run ``text`` through the agent once with ``skill`` staged, return a Run.

    Snapshots downloaded models before/after, executes in an isolated workspace,
    and captures evidence. The returned Run is registered for the current test;
    the conftest finalizer evaluates its expectations, reports, and cleans up.
    """
    print(f"\n[behavioral] running skill='{skill}' model='{model}': {text}", flush=True)
    pre_models = lemonade_downloaded_models()
    workspace = _stage_workspace(skill)
    try:
        wall_s, events = _run_agent(text, workspace, model, effort)
    except BaseException:
        shutil.rmtree(workspace, ignore_errors=True)
        raise
    post_models = lemonade_downloaded_models()
    evidence = _build_evidence(workspace, pre_models, post_models, events)

    run = Run(
        prompt_text=text, skill=skill, model=model, effort=effort,
        judge_model=judge_model, evidence=evidence, wall_time_s=wall_s,
    )
    _ACTIVE_RUNS.append(run)
    return run


# --------------------------------------------------------------------------- #
# Registry helpers (used by conftest)                                         #
# --------------------------------------------------------------------------- #
def reset_runs() -> None:
    _ACTIVE_RUNS.clear()


def drain_runs() -> list[Run]:
    runs = list(_ACTIVE_RUNS)
    _ACTIVE_RUNS.clear()
    return runs
