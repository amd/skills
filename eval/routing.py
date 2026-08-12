# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Routing engine: does the right skill fire, and only then?

Behavior mode asks "once this skill runs, does it do the job?". This asks the
question that comes first: **given every skill in the catalog installed side
by side, does the agent pick the right one?** It grades the routing decision
only, so it catches the four failure modes a description can cause:

  * correct trigger -- the expected skill activated.
  * missed trigger  -- a skill was expected and none activated (under-triggering).
  * wrong skill     -- a skill activated, but not the expected one (two
                       descriptions overlap and the agent picked the wrong side).
  * false trigger   -- no skill was expected and one activated (over-triggering).

Every case installs **all** skills under ``skills/``, because routing is only
meaningful against the whole catalog: a skill tested alone will happily answer
prompts that belong to its neighbor. Cases are pooled across every skill's
dataset, so a positive case for skill Y is automatically a negative for skill
X and the confusion matrix fills itself in.

Cost control: each run is killed the moment the routing decision is observable
-- the first skill activation, the final result event, or a small budget of
tool calls that are neither bookkeeping nor a survey of the installed catalog
-- so no case pays for the work the skill would have gone on to do.
``max_budget_usd`` is a second, independent backstop.

The CLI lives in ``eval/run_evals.py``; this module is the engine.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agent import claude_env  # noqa: E402
from datasets import SKILLS_DIR, Case  # noqa: E402

# Tools that carry no routing signal. An agent often opens with a todo list or
# a plan before deciding anything, and spending the non-skill tool budget on
# that would cut the run off before the real decision.
BOOKKEEPING_TOOLS = {"todowrite", "todoread", "exitplanmode"}

# Tool names Claude Code uses to activate a skill. `Skill` is current; older
# builds routed skills through the slash-command tool.
SKILL_TOOLS = {"skill", "slashcommand"}

# Where the staged catalog lives, as it appears in a tool argument.
CATALOG_DIR = ".claude/skills"

VERDICTS = ("correct_trigger", "true_negative", "missed_trigger", "wrong_skill", "false_trigger", "error")
PASSING_VERDICTS = {"correct_trigger", "true_negative"}

# Stop reasons that leave the routing decision unknown rather than observed.
INCONCLUSIVE_STOPS = {"completed", "timeout"}


@dataclass
class RoutingConfig:
    """Everything ``run_case`` needs that is not the case itself."""

    model: str = "opus"
    effort: str = "high"
    timeout: float = 240.0
    max_tool_calls: int = 4
    max_inspection_calls: int = 8
    max_budget_usd: float = 0.75
    keep_logs: str = ""
    available_flags: set[str] = field(default_factory=set)
    isolate_config: bool = False


@dataclass
class Outcome:
    id: str
    category: str
    skill: str | None
    prompt: str
    expect: str | None
    observed: str | None
    verdict: str
    passed: bool
    stop_reason: str
    elapsed_s: float
    tool_calls: int
    inspection_calls: int = 0
    visible_skills: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)
    error: str | None = None


def stage_workspace(skills: list[str]) -> Path:
    """Install every catalog skill into a fresh temp workspace.

    Claude Code loads ``.claude/skills/`` from a directory passed with
    ``--add-dir``, which registers each skill's name and description in the
    system prompt without injecting its body -- exactly the state a routing
    decision is made from. One workspace per case keeps cases isolated (and
    lets them run concurrently).
    """
    workspace = Path(tempfile.mkdtemp(prefix="routing-"))
    dest_root = workspace / ".claude" / "skills"
    dest_root.mkdir(parents=True, exist_ok=True)
    for skill in skills:
        shutil.copytree(SKILLS_DIR / skill, dest_root / skill)
    return workspace


def supported_flags(flags: list[str]) -> set[str]:
    """Which of `flags` the installed `claude` build advertises in --help.

    The two cost-control flags this eval likes to pass are recent additions. An
    older CLI would reject them and every case would fail identically, which
    reads like a routing collapse rather than a flag problem -- so check once
    (free, no tokens) and drop what isn't there.
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        return set()
    try:
        proc = subprocess.run(
            [claude_bin, "--help"], capture_output=True, text=True, encoding="utf-8", timeout=60
        )
    except (subprocess.SubprocessError, OSError):
        return set()
    text = (proc.stdout or "") + (proc.stderr or "")
    return {flag for flag in flags if flag in text}


def can_isolate_config() -> bool:
    """Whether the runner's own ``~/.claude`` can be kept out of the session.

    User-level skills are registered next to the staged ones and change every
    routing decision, so the catalog has to be exactly what the repo ships.
    Pointing the CLI at a throwaway config dir achieves that, but only when
    auth comes from the environment -- if the login lives in the real config
    dir, hiding it means no case even starts.
    """
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def _iter_tool_uses(obj) -> list[tuple[str, str]]:
    """Every (tool name, JSON-encoded tool input) pair nested anywhere in `obj`."""
    found: list[tuple[str, str]] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            if node.get("type") == "tool_use":
                found.append(
                    (
                        str(node.get("name", "")),
                        json.dumps(node.get("input", {}), ensure_ascii=False),
                    )
                )
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(obj)
    return found


def _match_skill(text: str, skills: list[str]) -> str | None:
    """Longest skill name mentioned in `text`, or None.

    Longest-first matters because one skill name can be a prefix of another
    (`local-ai-use` vs `local-ai-app-integration` share a stem today, and a
    future skill could nest outright).
    """
    lowered = text.lower()
    for skill in sorted(skills, key=len, reverse=True):
        if skill.lower() in lowered:
            return skill
    return None


def _skill_from_body_path(text: str, skills: list[str]) -> str | None:
    """The skill whose own ``SKILL.md`` path appears in `text`, or None.

    Matching the joined ``skills/<name>/skill.md`` path -- never the bare
    filename, never the bare skill name -- is what separates "this skill's
    body was loaded" from "this text happens to mention the skill". Two or
    more matches mean the text enumerates the catalog, which is a listing
    rather than a decision, so that is not an activation either.

    Reading one ``SKILL.md`` is only evidence of activation on a build that
    has no way to activate a skill except by reading it; see
    ``detect_activation``.
    """
    haystack = text.lower().replace("\\\\", "/").replace("\\", "/")
    hits = [skill for skill in skills if f"skills/{skill.lower()}/skill.md" in haystack]
    return hits[0] if len(hits) == 1 else None


def _is_catalog_inspection(tool_input: str, skills: list[str]) -> bool:
    """True when a tool call is only looking at the installed skills tree.

    Surveying the catalog is part of making the routing decision, not the
    agent starting the work itself, so these calls must not spend the
    non-skill tool budget: ending a run mid-survey scored deliberation as a
    missed trigger.
    """
    haystack = tool_input.lower().replace("\\\\", "/").replace("\\", "/")
    if CATALOG_DIR in haystack:
        return True
    return any(f"skills/{skill.lower()}/" in haystack for skill in skills)


def detect_activation(event: dict, skills: list[str], allow_body_path: bool = True) -> str | None:
    """The skill this event activates, or None.

    Only the agent's own tool calls count. Tool *results* and assistant prose
    are deliberately excluded: the staged workspace holds nothing but the
    skills tree, so any prompt that sends the agent looking for a file it
    cannot find gets a recursive listing of every ``SKILL.md`` back. Scoring
    that as an activation credited the longest skill name in the catalog with
    a false trigger on unrelated prompts, and -- worse -- scored a correct
    trigger whenever an expected skill's prompt named a path that did not
    exist, hiding real misses behind the file hunt.

    ``allow_body_path`` carries the same distinction for tool *inputs*. On a
    build that exposes the ``Skill`` tool, an agent that opens a ``SKILL.md``
    is reading the catalog to choose from it, so treating that read as an
    activation just credits whichever skill the directory listing happened to
    put first. The path fallback therefore stays off unless the session has no
    skill tool at all, which is the only case it was written for.

    Returns ``"other:<name>"`` when a skill outside the catalog fires -- that
    is a contaminated runner, not a routing result, and the report should say
    so rather than silently scoring it as a miss.
    """
    for name, tool_input in _iter_tool_uses(event):
        lowered = name.lower()
        if lowered in SKILL_TOOLS:
            hit = _match_skill(tool_input, skills)
            if hit:
                return hit
            try:
                parsed = json.loads(tool_input)
            except json.JSONDecodeError:
                parsed = {}
            invoked = ""
            for key in ("command", "skill", "name", "skill_name"):
                value = parsed.get(key) if isinstance(parsed, dict) else None
                if isinstance(value, str) and value.strip():
                    invoked = value.strip().lstrip("/")
                    break
            return f"other:{invoked or 'unknown'}"

        # Fallback for builds that load a skill body by reading the file
        # instead of going through the Skill tool. The call itself has to
        # target that skill's own SKILL.md; merely touching the skills
        # directory (`ls .claude/skills`) is not a routing decision.
        if allow_body_path:
            hit = _skill_from_body_path(tool_input, skills)
            if hit:
                return hit

    # Some builds announce an activation as a system event instead of a tool
    # call. Same joined-path rule, and init is excluded because it enumerates
    # the whole catalog by design.
    if allow_body_path and event.get("type") == "system" and event.get("subtype") != "init":
        return _skill_from_body_path(json.dumps(event, ensure_ascii=False), skills)
    return None


def _init_skills(event: dict, skills: list[str]) -> list[str] | None:
    """Skill names the CLI reported at session init, if this is that event.

    Used to prove the agent really saw the whole catalog (and nothing extra):
    a stray user-level skill on the runner would change every routing decision.
    """
    if event.get("type") != "system" or event.get("subtype") != "init":
        return None
    seen: list[str] = []
    for key in ("skills", "slash_commands", "slashCommands", "commands"):
        entries = event.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            text = entry if isinstance(entry, str) else json.dumps(entry, ensure_ascii=False)
            hit = _match_skill(text, skills)
            if hit and hit not in seen:
                seen.append(hit)
    return seen


def _init_tools(event: dict) -> set[str] | None:
    """Tool names the CLI reported at session init, if this is that event.

    Used to decide whether the SKILL.md-path fallback in ``detect_activation``
    applies to this build. An init event without a tool list leaves the
    fallback on, which is how older builds behaved.
    """
    if event.get("type") != "system" or event.get("subtype") != "init":
        return None
    tools = event.get("tools")
    if not isinstance(tools, list):
        return set()
    return {str(tool).lower() for tool in tools}


def _init_extra_skills(event: dict, skills: list[str]) -> list[str] | None:
    """Skills the CLI reported at init that this eval did not install.

    A user-level skill on the runner is registered alongside the staged
    catalog and competes for every prompt, so the routing numbers describe a
    catalog nobody ships. The ``other:`` check only notices such a skill when
    it actually fires; this notices it being installed at all.
    """
    if event.get("type") != "system" or event.get("subtype") != "init":
        return None
    entries = event.get("skills")
    if not isinstance(entries, list):
        return []
    known = {skill.lower() for skill in skills}
    extra: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            name = entry
        elif isinstance(entry, dict):
            name = str(entry.get("name") or "")
        else:
            continue
        name = name.strip().lstrip("/")
        if name and name.lower() not in known and name not in extra:
            extra.append(name)
    return extra


def _pump(stream, sink: queue.Queue) -> None:
    try:
        for line in stream:
            sink.put(line)
    finally:
        sink.put(None)


def _terminate(proc: subprocess.Popen) -> None:
    """Kill the CLI and its children.

    The `claude` process spawns helpers, so killing only the parent can leave
    an orphan holding the API call open -- which is the cost this eval exists
    to avoid. Kill the whole group/tree.
    """
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass


def classify(expect: str | None, observed: str | None) -> str:
    if observed is None:
        return "true_negative" if expect is None else "missed_trigger"
    if expect is None:
        return "false_trigger"
    return "correct_trigger" if observed == expect else "wrong_skill"


def run_case(case: Case, skills: list[str], config: RoutingConfig) -> Outcome:
    """Run one prompt, stopping as soon as the routing decision is known."""
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise SystemExit("error: 'claude' CLI not found on PATH")

    workspace = stage_workspace(skills)
    # Outside the workspace: the agent can list its own cwd, and a config dir
    # sitting in there would be one more thing for it to find.
    config_dir = (
        Path(tempfile.mkdtemp(prefix="routing-config-")) if config.isolate_config else None
    )
    cmd = [
        claude_bin,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
        "--add-dir",
        str(workspace),
        "--model",
        config.model,
    ]
    if config.effort:
        cmd += ["--effort", config.effort]
    # Dozens of throwaway sessions per run; don't leave them on disk.
    if "--no-session-persistence" in config.available_flags:
        cmd += ["--no-session-persistence"]
    if config.max_budget_usd > 0 and "--max-budget-usd" in config.available_flags:
        cmd += ["--max-budget-usd", str(config.max_budget_usd)]

    spawn: dict = {}
    if os.name == "nt":
        spawn["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        spawn["start_new_session"] = True

    env = claude_env()
    if config_dir is not None:
        env["CLAUDE_CONFIG_DIR"] = str(config_dir)

    events: list[dict] = []
    observed: str | None = None
    visible: list[str] = []
    extra: list[str] = []
    stop_reason = "completed"
    tool_calls = 0
    inspection_calls = 0
    allow_body_path = True
    error: str | None = None
    stderr_lines: list[str] = []

    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(workspace),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        **spawn,
    )
    try:
        assert proc.stdin is not None
        proc.stdin.write(case.prompt)
        proc.stdin.close()

        stdout_q: queue.Queue = queue.Queue()
        threading.Thread(target=_pump, args=(proc.stdout, stdout_q), daemon=True).start()
        threading.Thread(
            target=lambda: stderr_lines.extend(proc.stderr.readlines()), daemon=True
        ).start()

        deadline = time.perf_counter() + config.timeout
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                stop_reason = "timeout"
                break
            try:
                line = stdout_q.get(timeout=min(1.0, remaining))
            except queue.Empty:
                continue
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)

            reported = _init_skills(event, skills)
            if reported is not None:
                visible = reported
            uninstalled = _init_extra_skills(event, skills)
            if uninstalled is not None:
                extra = uninstalled
            tools = _init_tools(event)
            if tools is not None:
                allow_body_path = not (tools & SKILL_TOOLS)

            hit = detect_activation(event, skills, allow_body_path=allow_body_path)
            if hit:
                observed = hit
                stop_reason = "skill_activated"
                break

            if event.get("type") == "result":
                stop_reason = "result"
                if event.get("is_error"):
                    error = str(event.get("result") or "result event reported an error")[:400]
                break

            for name, tool_input in _iter_tool_uses(event):
                if name.lower() in BOOKKEEPING_TOOLS:
                    continue
                if _is_catalog_inspection(tool_input, skills):
                    inspection_calls += 1
                else:
                    tool_calls += 1
            # Inspection is exempt from the tool budget but not unbounded: an
            # agent that has read the whole catalog and still called no skill
            # has made its decision, and the run should not idle to timeout.
            if tool_calls >= config.max_tool_calls or inspection_calls >= config.max_inspection_calls:
                stop_reason = "tool_budget"
                break
    finally:
        _terminate(proc)
        elapsed = time.perf_counter() - start
        if config.keep_logs:
            logs_dir = Path(config.keep_logs)
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / f"{case.id}.jsonl").write_text(
                "\n".join(json.dumps(e, ensure_ascii=False) for e in events),
                encoding="utf-8",
            )
        shutil.rmtree(workspace, ignore_errors=True)
        if config_dir is not None:
            shutil.rmtree(config_dir, ignore_errors=True)

    if not events:
        error = ("".join(stderr_lines).strip() or "claude produced no stream-json output")[:400]

    # "no skill activated" is only a real finding when the run got far enough to
    # show a decision: the agent answered (`result`) or started doing the work
    # itself (`tool_budget`). A stream that just ends, or a hang, means the run
    # never made a routing decision -- grading that as a missed trigger would
    # invent a result out of an infrastructure failure.
    if observed is None and stop_reason in INCONCLUSIVE_STOPS:
        verdict = "error"
        detail = "".join(stderr_lines).strip()
        error = error or (
            f"run ended without a routing decision (stopped after: {stop_reason})"
            + (f"; stderr: {detail[:300]}" if detail else "")
        )
    elif error and observed is None:
        verdict = "error"
    else:
        verdict = classify(case.expect_skill, observed)

    outcome = Outcome(
        id=case.id,
        category=case.category,
        skill=case.skill,
        prompt=case.prompt,
        expect=case.expect_skill,
        observed=observed,
        verdict=verdict,
        passed=verdict in PASSING_VERDICTS,
        stop_reason=stop_reason,
        elapsed_s=round(elapsed, 2),
        tool_calls=tool_calls,
        inspection_calls=inspection_calls,
        visible_skills=visible,
        extra_skills=extra,
        error=error,
    )
    print(
        f"  [{'PASS' if outcome.passed else 'FAIL'}] {case.id}: "
        f"expected {case.expect_skill or 'no skill'} -> got {observed or 'no skill'} "
        f"({verdict}, {stop_reason}, {outcome.elapsed_s}s)",
        flush=True,
    )
    return outcome


def summarize(outcomes: list[Outcome], skills: list[str], meta: dict) -> dict:
    verdicts = Counter(o.verdict for o in outcomes)
    graded = [o for o in outcomes if o.verdict != "error"]
    passed = [o for o in graded if o.passed]

    by_category: dict[str, dict] = {}
    for category in sorted({o.category for o in outcomes}):
        subset = [o for o in graded if o.category == category]
        hits = sum(1 for o in subset if o.passed)
        by_category[category] = {
            "graded": len(subset),
            "passed": hits,
            "accuracy": round(hits / len(subset), 3) if subset else None,
        }

    per_skill: dict[str, dict] = {}
    for skill in skills:
        expected = [o for o in graded if o.expect == skill]
        correct = sum(1 for o in expected if o.observed == skill)
        fired = [o for o in graded if o.observed == skill]
        false_fires = sum(1 for o in fired if o.expect != skill)
        per_skill[skill] = {
            "expected": len(expected),
            "correct": correct,
            "missed": sum(1 for o in expected if o.observed is None),
            "lost_to_other_skill": sum(
                1 for o in expected if o.observed is not None and o.observed != skill
            ),
            "fired_total": len(fired),
            "fired_when_not_expected": false_fires,
            "recall": round(correct / len(expected), 3) if expected else None,
            "precision": round(correct / len(fired), 3) if fired else None,
        }

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for outcome in graded:
        confusion[outcome.expect or "(no skill)"][outcome.observed or "(no skill)"] += 1

    contaminated = sorted(
        {o.observed for o in outcomes if o.observed and o.observed.startswith("other:")}
    )
    catalog_gaps = sorted(
        {
            skill
            for o in outcomes
            if o.visible_skills
            for skill in skills
            if skill not in o.visible_skills
        }
    )
    catalog_extras = sorted({skill for o in outcomes for skill in o.extra_skills})

    return {
        "meta": meta,
        "totals": {
            "cases": len(outcomes),
            "graded": len(graded),
            "passed": len(passed),
            "errors": verdicts.get("error", 0),
            "accuracy": round(len(passed) / len(graded), 3) if graded else None,
            # How many runs activated any skill at all. Zero across a set that
            # expects activations means the skills were never installed or the
            # activation detector no longer matches the CLI's output -- either
            # way the numbers are an artifact, not a result.
            "activations": sum(1 for o in graded if o.observed),
            "activations_expected": sum(1 for o in graded if o.expect),
        },
        "verdicts": {name: verdicts.get(name, 0) for name in VERDICTS},
        "by_category": by_category,
        "per_skill": per_skill,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "unexpected_skills": contaminated,
        "skills_missing_from_session": catalog_gaps,
        "extra_skills_in_session": catalog_extras,
        "cases": [asdict(o) for o in outcomes],
    }


def render_markdown(summary: dict) -> str:
    totals = summary["totals"]
    verdicts = summary["verdicts"]
    meta = summary["meta"]
    accuracy = totals["accuracy"]
    lines = [
        "## Skill routing",
        "",
        f"**{totals['passed']}/{totals['graded']} correct "
        f"({'n/a' if accuracy is None else f'{accuracy:.1%}'})** across "
        f"{totals['cases']} prompts with {len(meta['skills'])} skills "
        f"installed together, on `{meta['model']}` (effort `{meta['effort']}`).",
        "",
        "| Verdict | Count | Meaning |",
        "| --- | --- | --- |",
        f"| correct_trigger | {verdicts['correct_trigger']} | expected skill activated |",
        f"| true_negative | {verdicts['true_negative']} | no skill expected, none activated |",
        f"| missed_trigger | {verdicts['missed_trigger']} | skill expected, nothing activated |",
        f"| wrong_skill | {verdicts['wrong_skill']} | a skill activated, but the wrong one |",
        f"| false_trigger | {verdicts['false_trigger']} | no skill expected, one activated |",
        f"| error | {verdicts['error']} | the run failed; excluded from accuracy |",
        "",
        "### By prompt category",
        "",
        "Categories are derived, not declared: `skill_should_trigger: true` is "
        "`positive`, `false` in a skill's own dataset is that skill's "
        "`near_miss`, and a prompt from the shared pool is `unrelated`.",
        "",
        "| Category | Graded | Correct | Accuracy |",
        "| --- | --- | --- | --- |",
    ]
    for category, stats in summary["by_category"].items():
        acc = stats["accuracy"]
        lines.append(
            f"| {category} | {stats['graded']} | {stats['passed']} | "
            f"{'n/a' if acc is None else f'{acc:.0%}'} |"
        )

    lines += [
        "",
        "### Per skill",
        "",
        "| Skill | Expected | Correct | Missed | Lost to another skill | Fired when not expected | Recall | Precision |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for skill, stats in summary["per_skill"].items():
        recall = stats["recall"]
        precision = stats["precision"]
        lines.append(
            f"| `{skill}` | {stats['expected']} | {stats['correct']} | {stats['missed']} | "
            f"{stats['lost_to_other_skill']} | {stats['fired_when_not_expected']} | "
            f"{'n/a' if recall is None else f'{recall:.0%}'} | "
            f"{'n/a' if precision is None else f'{precision:.0%}'} |"
        )

    # Errors get their own section: a crashed run says nothing about routing,
    # so listing it as a routing failure would be misleading.
    failures = [c for c in summary["cases"] if not c["passed"] and c["verdict"] != "error"]
    lines += ["", "### Routing failures", ""]
    if not failures:
        lines.append("None. Every graded prompt routed as expected.")
    else:
        lines += [
            "| Case | Category | Expected | Observed | Verdict | Prompt |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for case in failures:
            prompt = case["prompt"].replace("|", "\\|")
            if len(prompt) > 110:
                prompt = prompt[:110] + "..."
            lines.append(
                f"| `{case['id']}` | {case['category']} | {case['expect'] or '(no skill)'} | "
                f"{case['observed'] or '(no skill)'} | {case['verdict']} | {prompt} |"
            )

    errored = [c for c in summary["cases"] if c["verdict"] == "error"]
    if errored:
        lines += [
            "",
            "### Errored cases (not graded)",
            "",
            "| Case | Stopped after | Error |",
            "| --- | --- | --- |",
        ]
        for case in errored:
            detail = (case["error"] or "unknown").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{case['id']}` | {case['stop_reason']} | {detail[:160]} |")

    lines += [
        "",
        "<details><summary>All cases</summary>",
        "",
        "| Case | Expected | Observed | Verdict | Stopped after | Seconds |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in summary["cases"]:
        lines.append(
            f"| `{case['id']}` | {case['expect'] or '(no skill)'} | "
            f"{case['observed'] or '(no skill)'} | {case['verdict']} | "
            f"{case['stop_reason']} | {case['elapsed_s']} |"
        )
    lines += ["", "</details>"]

    if totals["activations"] == 0 and totals["activations_expected"]:
        lines += [
            "",
            "> **Not a valid result:** no skill activated in any case, including "
            f"the {totals['activations_expected']} that expected one. The skills "
            "were probably not installed for the session, or the activation "
            "detector no longer matches this `claude` build. Re-run with "
            "`--keep-logs` and inspect a transcript before trusting these numbers.",
        ]
    if summary["unexpected_skills"]:
        lines += [
            "",
            f"> **Warning:** a skill outside the catalog activated "
            f"({', '.join(summary['unexpected_skills'])}). The runner has extra "
            f"skills installed, so these routing results are not trustworthy.",
        ]
    if summary["skills_missing_from_session"]:
        lines += [
            "",
            f"> **Warning:** the CLI did not report these catalog skills at "
            f"session init: {', '.join(summary['skills_missing_from_session'])}. "
            f"They may not have been installed for the run.",
        ]
    if summary["extra_skills_in_session"]:
        extras = summary["extra_skills_in_session"]
        shown = ", ".join(f"`{name}`" for name in extras[:12])
        if len(extras) > 12:
            shown += f", and {len(extras) - 12} more"
        lines += [
            "",
            f"> **Warning:** {len(extras)} skill(s) beyond this repo's catalog "
            f"were registered for these sessions ({shown}). They come from the "
            f"runner's own config (usually `~/.claude/skills`) and compete for "
            f"every prompt, so the catalog measured here is not the one users "
            f"install. Set `ANTHROPIC_API_KEY` so the run can use an isolated "
            f"config dir, or remove them from the runner.",
        ]
    return "\n".join(lines) + "\n"
