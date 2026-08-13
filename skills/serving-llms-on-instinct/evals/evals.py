"""Behavioral tests for the `serving-llms-on-instinct` skill.

Run locally (needs the `claude` CLI authenticated and, for the live cases, a
reachable AMD Instinct GPU host via ROCM_SSH_HOST -- otherwise those cases skip):

    pip install -r eval/behavioral/requirements.txt
    export ROCM_SSH_HOST=root@10.0.0.5        # or ROCM_SSH_HOST + ROCM_SSH_USER
    cd eval/behavioral
    python -m pytest -c pytest.ini -p conftest ../../skills/serving-llms-on-instinct/evals/evals.py

Everything for this skill's behavioral suite lives in this single file (the repo
convention is one `evals/evals.py` per skill). It has three layers:

  1. LIVE behavioral cases -- run a real agent against the skill once and grade
     what it did. They `skip` unless ROCM_SSH_HOST is reachable. Checks use
     the stock harness's deterministic (`run.logs_contains`) and LLM-judged
     (`should`/`should_not`) surface, plus one in-file helper the stock `Run`
     doesn't provide: `logs_matches` (regex over the transcript) and a compliance
     ORACLE that grades the agent's structured tool calls.
  2. A deterministic compliance ORACLE (`evaluate`) -- pure (events in, verdict
     out): NOT_ENGAGED / NONCOMPLIANT / COMPLIANT. No LLM judgement.
  3. No-hardware ORACLE SELF-TESTS -- feed the oracle synthetic transcripts to
     prove its failure detectors fire. These need no GPU, no API, no agent, so
     they run wherever the file is collected (and via `python evals.py`).

Shim note: the stock shared harness (`eval/behavioral/harness.py`) exposes
`run.logs` -- the full stream-json transcript, one JSON event per line. We
re-parse it here into the structured tool-call view the oracle keys on, so the
shared harness stays untouched. See `_parse_events` / `parse_events`.

Reliability: hard invariants (one correct answer -- ROCm image, health poll,
required override) go through `logs_matches` + the oracle, never the LLM judge.
`should`/`should_not` grade only genuinely-fuzzy behaviour.
"""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# NOTE: `pytest` and the shared `harness` are imported lazily (inside the live
# tests / _require_host) so the no-hardware oracle self-tests -- and `python
# evals.py` -- run with zero external dependencies. Under the repo's pytest
# invocation, conftest puts `harness` on sys.path before the live tests run.

# evals/ -> skill root. Scripts and data are read from the real skill dir (not
# the agent's staged temp copy), so the oracle computes what the agent SHOULD do.
SKILL_NAME = "serving-llms-on-instinct"
SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
DATA = SKILL_ROOT / "data"

_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=accept-new"]

# --- probe configuration (generic model test) --- #
MODEL_ID = os.environ.get("SERVE_LLM_TARGET_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
PORT = int(os.environ.get("SERVE_LLM_TARGET_PORT", "8000"))
PROBE_Q = os.environ.get("SERVE_LLM_PROBE_Q",
                         "What is the capital of France? Answer with only the city name.")
# Comma-separated acceptable substrings; ANY match passes.
PROBE_EXPECT = [s.strip().lower() for s in
                os.environ.get("SERVE_LLM_PROBE_EXPECT", "paris").split(",") if s.strip()]
# 32 tokens is enough for "Paris" but too little for a poem, and reasoning models
# spend most of the budget on hidden reasoning before visible content -- so be generous.
PROBE_MAXTOK = int(os.environ.get("SERVE_LLM_PROBE_MAXTOK", "256"))
# Completion-style coherence prompt for base models (a factual lead-in).
COMPLETION_Q = os.environ.get("SERVE_LLM_COMPLETION_Q", "The capital of France is")

SERVE_TEMPLATE = (
    "Serve {model} for inference on the AMD Instinct GPU server "
    "(the host is in the ROCM_SSH_HOST environment variable). "
    "Expose the OpenAI-compatible API on port {port}. "
    "First stop and remove any existing Docker containers on the server to free GPU memory. "
    "After the endpoint is healthy, send a test chat completion request to verify it produces output. "
    "Do not remove the container when done. "
    "Finally, summarize: the docker image used, the exact docker run command, whether the "
    "health check passed, and whether inference produced output."
)


# =========================================================================== #
# Shim over the stock harness's `Run`
#
# The stock `Run` exposes `run.logs` (the raw stream-json transcript, one JSON
# event per line) but not the structured tool-call list the oracle needs, nor a
# regex check. We recover both here without touching the shared harness.
# =========================================================================== #

def _parse_events(logs):
    """Structured view of the transcript: a list of tool calls with results.

    Returns [{tool, input, command, path, result, is_error}], where:
      * tool      -- tool name (e.g. "Bash", "Read")
      * input     -- the raw tool_use input dict
      * command   -- input["command"] for Bash (the shell line), else ""
      * path      -- input["file_path"] for Read, else ""
      * result    -- the matching tool_result text ("" if none seen)
      * is_error  -- True if the tool_result was flagged is_error

    `logs` is the stock `Run.logs` string -- exactly the stream-json events
    (`json.dumps(ev)` per line), so json.loads recovers each event verbatim.
    This is the deterministic substrate the compliance oracle keys on: it lets a
    check assert "the agent actually RAN detect.py and it returned JSON", not
    merely "the string detect.py appears somewhere in the blob".
    """
    by_id = {}
    order = []
    for line in (logs or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = obj.get("type")
        if t == "assistant":
            for block in obj.get("message", {}).get("content", []) or []:
                if block.get("type") == "tool_use":
                    ev_id = block.get("id")
                    if not ev_id:
                        continue  # skip malformed events with no id to avoid None-key collisions
                    inp = block.get("input", {}) or {}
                    ev = {
                        "id": ev_id,
                        "tool": block.get("name", ""),
                        "input": inp,
                        "command": inp.get("command", "") if isinstance(inp, dict) else "",
                        "path": inp.get("file_path", "") if isinstance(inp, dict) else "",
                        "result": "",
                        "is_error": False,
                    }
                    by_id[ev_id] = ev
                    order.append(ev)
        elif t == "user":
            content = obj.get("message", {}).get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        ev = by_id.get(block.get("tool_use_id"))
                        if ev is None:
                            continue
                        c = block.get("content")
                        if isinstance(c, list):
                            c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
                        ev["result"] = str(c or "")
                        ev["is_error"] = bool(block.get("is_error"))
    return order


def parse_events(run):
    """Structured tool calls for a completed stock-harness `Run`."""
    return _parse_events(run.logs)


def report_pass(label, detail=""):
    print(f"  [PASS] {label}" + (f" -- {detail}" if detail else ""), flush=True)


def report_fail(label, detail=""):
    print(f"  [FAIL] {label}" + (f" -- {detail}" if detail else ""), flush=True)
    raise AssertionError(f"{label}" + (f": {detail}" if detail else ""))


def logs_matches(run, pattern, flags=0):
    """Deterministic regex check over the transcript (a hard invariant).

    The stock harness's `run.logs_contains(text)` handles substring checks;
    this adds regex support that the stock harness doesn't provide.
    """
    ok = bool(re.search(pattern, run.logs, flags))
    run._report(ok, "logs_matches", f"transcript matches {pattern!r}")


# =========================================================================== #
# Live-host helpers (this skill drives a remote AMD Instinct box over SSH)
# =========================================================================== #

def _gpu_host():
    host = os.environ.get("ROCM_SSH_HOST", "").strip()
    user = os.environ.get("ROCM_SSH_USER", "").strip()
    if host and user and "@" not in host:
        return f"{user}@{host}"
    return host


def _require_host():
    """Return the resolved ssh target, or skip the (live) test when unreachable."""
    import pytest
    host = _gpu_host()
    if not host:
        pytest.skip("ROCM_SSH_HOST not set -- live serve test needs an AMD Instinct host")
    try:
        r = subprocess.run(_SSH + [host, "echo serve-llm-ok"],
                           capture_output=True, text=True, timeout=25)
    except Exception as e:  # noqa: BLE001 - any ssh failure means "not runnable"
        pytest.skip(f"GPU host {host} unreachable: {e}")
    if r.returncode != 0 or "serve-llm-ok" not in r.stdout:
        pytest.skip(f"GPU host {host} unreachable over SSH: {r.stderr.strip()[:200]}")
    return host


def _purge_containers(host):
    """Stop and remove ALL Docker containers on the GPU host before each test.

    A pre-existing container for the model under test would let the agent skip
    the launch entirely, hiding whether it applies the correct configuration
    (e.g. VLLM_ROCM_USE_AITER_MOE=0). Tests must start from a clean slate.
    """
    r = subprocess.run(
        _SSH + [host, "ids=$(docker ps -aq); [ -n \"$ids\" ] && docker rm -f $ids "
                "&& echo removed || echo none"],
        capture_output=True, text=True, timeout=60,
    )
    print(f"  [setup] container purge on {host}: "
          f"{r.stdout.strip() or r.stderr.strip()[:120]}", flush=True)


# =========================================================================== #
# SKILL.md compliance ORACLE  (deterministic; pure events -> verdict)
#
# Given the STRUCTURED tool calls of a run, decide one of three outcomes:
#   NOT_ENGAGED  -- agent never used the skill (free-solved). Engagement failure.
#   NONCOMPLIANT -- skill engaged but the SKILL.md procedure was broken. THE main
#                   failure this suite exists to catch.
#   COMPLIANT    -- engaged AND every required step performed correctly.
# =========================================================================== #

# Assets that exist ONLY because of the skill. Touching one proves engagement.
SKILL_SCRIPTS = ["detect.py", "validate.py", "check_overrides.py",
                 "estimate_vram.py", "sync_recipes.py"]
SKILL_DATA = ["recipes_cache.json", "gpu_overrides.json", "blacklist.json"]

# A launched container must use a vLLM ROCm image. These patterns flag the
# opposite -- a CUDA/NVIDIA image, i.e. the wrong environment entirely.
_ROCM_IMAGE_RX = re.compile(r"(vllm[/-]openai[/-]rocm|rocm/vllm)", re.IGNORECASE)
_CUDA_IMAGE_RX = re.compile(r"(nvidia/|/cuda|vllm/vllm-openai:|nvcr\.io|nvidia-docker)",
                            re.IGNORECASE)

# A single REQUIRED violation makes the run NONCOMPLIANT; EXPECTED violations are
# recorded as warnings and (by default) do not fail the run.
REQUIRED = "required"
EXPECTED = "expected"


class ComplianceReport:
    def __init__(self):
        self.engaged = False
        self.steps = {}          # step name -> short status string
        self.violations = []     # list of (step, severity, reason)
        self.notes = []          # informational
        self.expected_overrides = {}   # required env overrides for this model+gfx

    # Canonical SKILL.md procedure, in order. The "override" check is only shown
    # when the model+gfx actually requires one.
    _PROCEDURE = [
        ("detect",          "Step 1  detect GPU (detect.py)"),
        ("validate",        "Step 2  validate env (validate.py)"),
        ("check_overrides", "Step 3b check model+gfx overrides (check_overrides.py)"),
        ("construct",       "Step 4  read recipe/gpu data files"),
        ("estimate_vram",   "Step 4  VRAM fit check (estimate_vram.py)"),
        ("launch",          "Step 6  launch ROCm container (docker run)"),
        ("override",        "Step 6  required override applied in launch"),
        ("health",          "Step 6  poll /health"),
    ]

    def checks(self):
        """Ordered per-step results for display: list of
        {label, ok, severity, reason}. Engagement is the first line; when the
        agent never engaged, only that line is returned."""
        eng = {"label": "Engagement: used the skill's scripts/data",
               "ok": self.engaged, "severity": REQUIRED,
               "reason": "" if self.engaged else "free-solved -- no skill assets touched"}
        if not self.engaged:
            return [eng]
        failed = {s: (sev, reason) for s, sev, reason in self.violations}
        out = [eng]
        for key, label in self._PROCEDURE:
            if key == "override" and not self.expected_overrides:
                continue
            if key in failed:
                sev, reason = failed[key]
                out.append({"label": label, "ok": False, "severity": sev, "reason": reason})
            else:
                out.append({"label": label, "ok": True, "severity": REQUIRED, "reason": ""})
        return out

    def _add(self, step, severity, reason):
        self.violations.append((step, severity, reason))

    @property
    def required_violations(self):
        return [v for v in self.violations if v[1] == REQUIRED]

    @property
    def verdict(self):
        if not self.engaged:
            return "NOT_ENGAGED"
        return "NONCOMPLIANT" if self.required_violations else "COMPLIANT"

    def summary(self):
        if self.verdict == "NOT_ENGAGED":
            return "NOT_ENGAGED -- agent used no skill scripts or data files (free-solved)"
        if self.verdict == "COMPLIANT":
            warn = [v for v in self.violations if v[1] == EXPECTED]
            tail = f" ({len(warn)} non-fatal warning(s))" if warn else ""
            return "COMPLIANT -- followed the SKILL.md procedure" + tail
        rv = "; ".join(f"{s}: {r}" for s, sev, r in self.required_violations)
        return f"NONCOMPLIANT -- {rv}"


# ------------------------- low-level event queries ------------------------- #

def _script_calls(events, name):
    """Bash events whose command invokes the named skill script."""
    return [e for e in events
            if e.get("tool") == "Bash" and name in (e.get("command") or "")]


def _read_calls(events, name):
    """Events that read the named file -- via the Read tool OR a cat/grep in Bash."""
    out = []
    for e in events:
        if e.get("tool") == "Read" and name in (e.get("path") or ""):
            out.append(e)
        elif e.get("tool") == "Bash" and name in (e.get("command") or ""):
            out.append(e)
    return out


def _launch_calls(events):
    """Bash events that launch a container (`docker run`), local or over ssh."""
    return [e for e in events
            if e.get("tool") == "Bash" and re.search(r"docker\s+run", e.get("command") or "")]


def _all_commands(events):
    return "\n".join(e.get("command") or "" for e in events if e.get("tool") == "Bash")


def tool_trace(events, width=88):
    """Human-readable, ordered list of the tool calls the agent made. Each line is
    '<n>. [Tool] <command-or-path> [ERR]'."""
    lines = []
    for i, e in enumerate(events, 1):
        tool = e.get("tool", "?")
        body = (e.get("command") or e.get("path") or "").strip().replace("\n", " ")
        body = re.sub(r"\s+", " ", body)
        if len(body) > width:
            body = body[:width - 1] + "…"
        err = "  [ERR]" if e.get("is_error") else ""
        lines.append(f"{i:>2}. [{tool}] {body}{err}")
    return lines


def check_lines(report):
    """Per-step PASS/FAIL/WARN lines for display (uses report.checks())."""
    out = []
    for c in report.checks():
        if c["ok"]:
            mark = "PASS"
        else:
            mark = "FAIL" if c["severity"] == REQUIRED else "WARN"
        line = f"    [{mark}] {c['label']}"
        if c["reason"]:
            line += f" -- {c['reason']}"
        out.append(line)
    return out


def evaluate_environment(events, model, gfx, expected_env_set=None):
    """Did the run launch the RIGHT container/environment? Inspects the ACTUAL
    `docker run` command, with NO dependency on whether the skill was used -- so a
    free-solve is judged exactly like a skill run. Encodes the guarantee the user
    cares about: 'picking the wrong container/environment must never happen.'

    A wrong environment can pass a naive health probe yet silently corrupt output
    (e.g. gpt-oss on gfx950 without VLLM_ROCM_USE_AITER_MOE=0), so this is
    separate from -- and required in addition to -- the live probe.

    Returns {launched, env_ok, issues:[(category, reason)]} where category is
    WRONG_CONTAINER (bad image) or WRONG_ENV (missing override).
    """
    expected_env_set = expected_env_set or {}
    launches = _launch_calls(events)
    if not launches:
        return {"launched": False, "env_ok": False,
                "issues": [("NO_LAUNCH", "no `docker run` was ever issued")]}
    text = "\n".join(e.get("command") or "" for e in launches)
    issues = []
    if _CUDA_IMAGE_RX.search(text):
        issues.append(("WRONG_CONTAINER",
                       "launched a CUDA/NVIDIA image -- wrong environment for AMD Instinct"))
    if not _ROCM_IMAGE_RX.search(text):
        issues.append(("WRONG_CONTAINER", "launch did not use a vLLM ROCm image"))
    for k, v in expected_env_set.items():
        if not re.search(rf"{re.escape(k)}\s*=\s*{re.escape(str(v))}", text):
            issues.append(("WRONG_ENV",
                           f"required override {k}={v} for {model} on {gfx} not applied "
                           f"(silent-corruption risk)"))
    return {"launched": True, "env_ok": not issues, "issues": issues}


def evaluate(events, model, gfx, expected_env_set=None, strict=False):
    """Produce a ComplianceReport.

    events           -- structured tool calls (parse_events output)
    model            -- the HF model id the agent was asked to serve
    gfx              -- the gfx version detected on the target host (oracle truth)
    expected_env_set -- {ENV: val} check_overrides.py prescribes for this
                        model+gfx; each MUST appear in the launch command. Pass {}
                        when no override is required.
    strict           -- if True, EXPECTED-severity items also fail the run.
    """
    expected_env_set = expected_env_set or {}
    r = ComplianceReport()
    r.expected_overrides = dict(expected_env_set)

    # ---- Engagement: did the agent touch ANY skill asset? ---- #
    touched = []
    for s in SKILL_SCRIPTS:
        if _script_calls(events, s):
            touched.append(s)
    for d in SKILL_DATA:
        if _read_calls(events, d):
            touched.append(d)
    r.engaged = bool(touched)
    r.notes.append(f"skill assets touched: {touched or 'NONE'}")
    if not r.engaged:
        return r  # pure engagement failure; nothing more to grade

    # ---- Step 1: GPU detection ---- #
    det = _script_calls(events, "detect.py")
    if not det:
        r.steps["detect"] = "missing"
        r._add("detect", REQUIRED, "detect.py was never run (Step 1)")
    elif all(e.get("is_error") for e in det):
        r.steps["detect"] = "errored"
        r._add("detect", REQUIRED, "detect.py ran but every call returned an error")
    else:
        r.steps["detect"] = "ok"

    # ---- Step 2: environment validation ---- #
    val = _script_calls(events, "validate.py")
    if not val:
        r.steps["validate"] = "missing"
        r._add("validate", EXPECTED, "validate.py was never run (Step 2)")
    else:
        r.steps["validate"] = "ok"

    # ---- Step 3b: model+GPU overrides (MANDATORY) ---- #
    ov = _script_calls(events, "check_overrides.py")
    if not ov:
        r.steps["check_overrides"] = "missing"
        r._add("check_overrides", REQUIRED,
               "check_overrides.py was never run (Step 3b is mandatory)")
    else:
        before = len(r.violations)
        cmds = " ".join(e.get("command") or "" for e in ov)
        m = re.search(r"--gfx\s+(\S+)", cmds)
        if gfx and m:
            if m.group(1) != gfx:
                r._add("check_overrides", REQUIRED,
                       f"check_overrides.py called with --gfx {m.group(1)} but host is {gfx}")
        elif gfx:
            r._add("check_overrides", REQUIRED,
                   "check_overrides.py run without a --gfx argument")
        r.steps["check_overrides"] = "ok" if len(r.violations) == before else "wrong-args"

    # ---- Step 4: construct from recipe + GPU data ---- #
    read_recipes = _read_calls(events, "recipes_cache.json")
    read_gpu = _read_calls(events, "gpu_overrides.json")
    if not (read_recipes or read_gpu):
        r.steps["construct"] = "no-data-read"
        r._add("construct", EXPECTED,
               "neither recipes_cache.json nor gpu_overrides.json was read (Step 4)")
    else:
        r.steps["construct"] = "ok"

    # ---- Step 4: VRAM fit estimate ---- #
    if not _script_calls(events, "estimate_vram.py"):
        r.steps["estimate_vram"] = "missing"
        r._add("estimate_vram", EXPECTED, "estimate_vram.py was never run (Step 4 fit check)")
    else:
        r.steps["estimate_vram"] = "ok"

    # ---- Step 6: launch the correct container/environment ---- #
    env = evaluate_environment(events, model, gfx, expected_env_set)
    if not env["launched"]:
        r.steps["launch"] = "missing"
        r._add("launch", REQUIRED, "no `docker run` was ever issued (Step 6)")
    else:
        container_issues = [reason for cat, reason in env["issues"] if cat == "WRONG_CONTAINER"]
        env_issues = [reason for cat, reason in env["issues"] if cat == "WRONG_ENV"]
        for reason in container_issues:
            r._add("launch", REQUIRED, reason)
        for reason in env_issues:   # missing mandatory override
            r._add("override", REQUIRED, reason)
        r.steps["launch"] = "ok" if not container_issues else "wrong-image"

    # ---- Step 6: health verification ---- #
    if "/health" not in _all_commands(events):
        r.steps["health"] = "missing"
        r._add("health", REQUIRED, "the /health endpoint was never polled (Step 6)")
    else:
        r.steps["health"] = "ok"

    if strict:
        r.violations = [(s, REQUIRED if sev == EXPECTED else sev, reason)
                        for s, sev, reason in r.violations]

    return r


# =========================================================================== #
# Live grading helpers: derive expected behavior from the skill's own oracles
# and probe the real endpoint. Keeps the generic test discriminating without
# hardcoding per-model facts.
# =========================================================================== #

def _detect_gfx(host):
    """Detected gfx version of the target host (amd-smi over SSH; no network)."""
    try:
        r = subprocess.run(["python3", str(SCRIPTS / "detect.py"), "--host", host],
                           capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout).get("gfx_version", "")
    except Exception:  # noqa: BLE001
        return ""


def _expected_overrides(model, gfx):
    """env_set that check_overrides.py prescribes for this model+gfx (no network)."""
    try:
        r = subprocess.run(
            ["python3", str(SCRIPTS / "check_overrides.py"), "--model", model, "--gfx", gfx],
            capture_output=True, text=True, timeout=30)
        return json.loads(r.stdout).get("env_set", {}) or {}
    except Exception:  # noqa: BLE001
        return {}


def _blacklist_category(model):
    """Return the blacklist category name if this model can't be served, else None."""
    try:
        bl = json.loads((DATA / "blacklist.json").read_text())
    except Exception:  # noqa: BLE001
        return None
    for cat, body in bl.items():
        if isinstance(body, dict) and model in (body.get("models") or []):
            return cat
    return None


def _endpoint_serving(host, port, model, timeout=30):
    """Ground truth: is a live endpoint actually serving `model` on the host?

    Checks /health == 200 AND /v1/models lists the model id. Used for the
    blacklist case so the verdict rests on what's really running.
    """
    remote = (f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{port}/health; "
              f"echo '|'; curl -s http://localhost:{port}/v1/models")
    try:
        r = subprocess.run(_SSH + [host, remote], capture_output=True, text=True, timeout=timeout)
        code, _, models = r.stdout.partition("|")
        return code.strip() == "200" and model in models
    except Exception:  # noqa: BLE001
        return False


def _served_model_name(host, port, fallback, timeout=20):
    """The model id the endpoint actually serves (from /v1/models). The agent may
    serve under a different name than requested, so probe the served name."""
    remote = f"curl -s http://localhost:{port}/v1/models"
    try:
        r = subprocess.run(_SSH + [host, remote], capture_output=True, text=True, timeout=timeout)
        data = json.loads(r.stdout).get("data") or []
        return data[0].get("id") or fallback
    except Exception:  # noqa: BLE001
        return fallback


def _probe_chat(host, port, model, question, timeout=180):
    """Hit the live endpoint and return the model's answer. "" on any failure."""
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "max_tokens": PROBE_MAXTOK, "temperature": 0,
    })
    remote = (f"curl -s http://localhost:{port}/v1/chat/completions "
              f"-H 'Content-Type: application/json' -d {shlex.quote(payload)}")
    try:
        r = subprocess.run(_SSH + [host, remote], capture_output=True, text=True, timeout=timeout)
        msg = json.loads(r.stdout)["choices"][0]["message"]
        # Reasoning models may leave `content` null and put text in reasoning_content.
        return msg.get("content") or msg.get("reasoning_content") or ""
    except Exception:  # noqa: BLE001
        return ""


def _probe_completion(host, port, model, prompt, timeout=180):
    """Raw /v1/completions probe -- works for BASE models with no chat template."""
    payload = json.dumps({"model": model, "prompt": prompt,
                          "max_tokens": PROBE_MAXTOK, "temperature": 0})
    remote = (f"curl -s http://localhost:{port}/v1/completions "
              f"-H 'Content-Type: application/json' -d {shlex.quote(payload)}")
    try:
        r = subprocess.run(_SSH + [host, remote], capture_output=True, text=True, timeout=timeout)
        return json.loads(r.stdout)["choices"][0].get("text") or ""
    except Exception:  # noqa: BLE001
        return ""


# Name-based classification: does this model expose a usable chat/instruct
# interface? Errs SAFE -- a chat model misread as base only loses the coherence
# bonus; a base model is very unlikely to carry a chat keyword.
_CHAT_HINTS = ("instruct", "-it", "_it", "chat", "thinking", "qwq", "gpt-oss",
               "-r1", "reasoner", "reasoning", "distill", "dolphin", "venice",
               "kimi", "glm", "minimax", "apriel", "command", "vibethinker",
               "deepseek-r1", "deepseek-v3", "deepseek-v2", "-rl-", "moe-chat")


def _is_chat_model(model_id):
    m = (model_id or "").lower()
    return any(h in m for h in _CHAT_HINTS)


def _is_degenerate(text, prompt=""):
    """True if the output is empty, just echoes the prompt, or is a repetition
    loop -- i.e. the endpoint serves but the output is broken."""
    t = (text or "").strip()
    if len(t) < 2:
        return True
    if prompt and t.replace(prompt.strip(), "").strip() == "":
        return True  # only echoed the prompt back
    for n in (40, 24, 12):
        if len(t) >= n * 3 and t.count(t[:n]) >= 3:
            return True
    return False


def _endpoint_up(host, port, timeout=20):
    """Is there a healthy endpoint on this port (independent of how it launched)?"""
    remote = f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{port}/health"
    try:
        r = subprocess.run(_SSH + [host, remote], capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() == "200"
    except Exception:  # noqa: BLE001
        return False


def _grade_outcome(host, port, model):
    """OUTCOME AXIS: did the run leave a working endpoint? Probed the SAME way
    regardless of whether the skill was used. Returns (served, detail, answer,
    served_name), served being one of:
        PASS       chat model answered the coherence probe correctly
        SERVED_OK  base model produced real, non-degenerate completion output
        PROBE_FAIL endpoint is up but output is empty/degenerate/incoherent
        DOWN       no healthy endpoint at all
    """
    if not _endpoint_up(host, port):
        return "DOWN", f"no healthy endpoint on port {port}", "", model
    served = _served_model_name(host, port, model)
    if _is_chat_model(model):
        answer = _probe_chat(host, port, served, PROBE_Q)
        hit = next((e for e in PROBE_EXPECT if e in answer.lower()), None)
        if hit:
            return "PASS", f"chat answer contains {hit!r}", answer, served
        return "PROBE_FAIL", (f"chat answer wrong: expected any of {PROBE_EXPECT}, "
                              f"got {answer!r}"), answer, served
    answer = _probe_completion(host, port, served, COMPLETION_Q)
    if _is_degenerate(answer, COMPLETION_Q):
        return "PROBE_FAIL", f"base output empty/degenerate: {answer!r}", answer, served
    return "SERVED_OK", f"base completion: {answer.strip()[:60]!r}", answer, served


class Grade:
    """One run, measured the SAME way whether or not the skill was used.

    SUCCESS = the run left a working endpoint AND used the correct
    container/environment. Failures are bucketed so the focus failure -- wrong
    container/environment -- is visible.
    """
    def __init__(self, model, served=None, served_detail="", procedure=None,
                 procedure_detail="", report=None, answer=None, served_name=None,
                 env=None, blacklisted=False):
        self.model = model
        self.served = served
        self.served_detail = served_detail
        self.procedure = procedure
        self.procedure_detail = procedure_detail
        self.report = report
        self.answer = answer
        self.served_name = served_name
        self.env = env or {}
        self.blacklisted = blacklisted

    @property
    def served_ok(self):
        return self.served in ("PASS", "SERVED_OK")

    @property
    def env_ok(self):
        return bool(self.env.get("env_ok"))

    @property
    def env_issues(self):
        return self.env.get("issues") or []

    @property
    def engaged(self):
        return self.procedure not in ("NOT_ENGAGED", None)

    @property
    def compliant(self):
        return self.procedure in ("COMPLIANT", "REFUSED_OK")

    @property
    def success(self):
        if self.blacklisted:
            return self.procedure == "REFUSED_OK"
        return self.served_ok and self.env_ok

    passed = success  # alias for existing callers

    @property
    def failure_category(self):
        if self.success:
            return None
        if self.blacklisted:
            return "SERVED_BLACKLISTED"
        cats = [c for c, _ in self.env_issues]
        if "WRONG_CONTAINER" in cats:
            return "WRONG_CONTAINER"   # the focus failure
        if "WRONG_ENV" in cats:
            return "WRONG_ENV"         # missing mandatory override (silent corruption)
        if self.served == "DOWN" or not self.env.get("launched"):
            return "DOWN"
        if self.served == "PROBE_FAIL":
            return "BAD_OUTPUT"
        return "FAIL"

    @property
    def verdict(self):
        if self.blacklisted:
            return self.procedure
        return "SUCCESS" if self.success else f"FAIL:{self.failure_category}"

    @property
    def detail(self):
        if self.blacklisted:
            return self.procedure_detail
        if self.success:
            return f"served={self.served}, environment correct"
        bits = "; ".join(reason for _, reason in self.env_issues) or self.served_detail
        return f"{self.failure_category}: {bits}"

    def __str__(self):
        return f"{self.model}: {self.verdict} -- {self.detail}"


def grade_serve_run(events, host, model, port, category=None):
    """Grade one completed agent run (structured `events`) on both axes
    (outcome + skill procedure). Takes the event list so it works for both a live
    stock-harness Run (via parse_events) and synthetic self-test events."""
    # ----- Blacklisted (non-LLM) model: expect refusal, not a launch ----- #
    if category:
        if _endpoint_serving(host, port, model):
            return Grade(model, blacklisted=True, procedure="REFUSED_BAD",
                         procedure_detail=f"stood up a live endpoint serving blacklisted "
                                          f"model ({category})")
        return Grade(model, blacklisted=True, procedure="REFUSED_OK",
                     procedure_detail=f"no live endpoint; correctly refused ({category})")

    gfx = _detect_gfx(host)
    env_set = _expected_overrides(model, gfx)

    # ENVIRONMENT: did it launch the RIGHT container/env? (condition-independent)
    env = evaluate_environment(events, model, gfx, expected_env_set=env_set)
    # SKILL (secondary): engagement + procedure from the tool calls.
    report = evaluate(events, model, gfx, expected_env_set=env_set)
    # OUTCOME: always probe the live endpoint, the SAME way for both.
    served, sdetail, answer, sname = _grade_outcome(host, port, model)

    return Grade(model, served=served, served_detail=sdetail,
                 procedure=report.verdict, procedure_detail=report.summary(),
                 report=report, answer=answer, served_name=sname, env=env)


# =========================================================================== #
# LIVE behavioral tests (skip unless ROCM_SSH_HOST is reachable)
# =========================================================================== #

def _agent(**kw):
    """Lazy accessor for the shared harness's `claude` factory (imported only
    when a live test actually runs, so oracle-only paths need no harness)."""
    from harness import claude
    return claude(**kw)


def test_serve_qwen3_on_instinct():
    """Happy path: a concrete, well-specified model served end-to-end."""
    host = _require_host()
    _purge_containers(host)
    with _agent(skill=SKILL_NAME) as agent:
        run = agent.prompt(SERVE_TEMPLATE.format(model="Qwen/Qwen3-8B", port=8000))

        # Deterministic checks — must appear regardless of the agent's path.
        logs_matches(run, r"vllm[/-]openai[/-]rocm")   # ROCm image, not CUDA
        logs_matches(run, r"docker\s+run")             # a container was launched
        logs_matches(run, r"/health")                  # health endpoint polled
        logs_matches(run, r"gfx9[45]\d")               # gfx detected (gfx942/gfx950)
        logs_matches(run, r"Qwen/Qwen3-8B")            # correct model targeted
        logs_matches(run, r"(200|HEALTHY|health.*ok)", re.IGNORECASE)  # health passed

        # Positive behavioral expectations (LLM judge).
        run.should("Detect the GPU hardware (gfx version and VRAM) before configuring vLLM")
        run.should("Launch a vLLM ROCm Docker container and poll /health until the endpoint is ready")
        run.should("Send a chat completion request to the running endpoint and receive non-empty model output")
        run.should("Leave the vLLM container running at the end rather than removing it")

        # Negative behavioral expectations.
        run.should_not("Use an NVIDIA/CUDA Docker image or the nvidia-docker runtime")
        run.should_not("Remove or stop the vLLM container it just launched")


def test_refuses_blacklisted_model():
    """A blacklisted (non-LLM) model must be refused with an explanation, not launched."""
    host = _require_host()
    _purge_containers(host)
    with _agent(skill=SKILL_NAME) as agent:
        run = agent.prompt(
            "Serve stabilityai/stable-diffusion-3.5-medium as an OpenAI-compatible "
            "LLM chat endpoint on the AMD Instinct GPU server (host in ROCM_SSH_HOST) "
            "on port 8000. I need to call it via the /v1/chat/completions API."
        )

        run.logs_contains("stable-diffusion")

        run.should(
            "Decline to serve the model and explain that stable-diffusion-3.5-medium "
            "is a text-to-image diffusion model that cannot be exposed as an "
            "OpenAI-compatible LLM chat endpoint"
        )
        run.should("Suggest at least one alternative model that can be served as an LLM endpoint")

        run.should_not(
            "Launch or attempt to launch any Docker container to serve stable-diffusion "
            "in any form (including via diffusers, FastAPI, or a custom server)"
        )
        run.should_not("Treat the request as achievable and proceed to set up any inference service")


def test_ambiguous_gpt_oss_request():
    """Ambiguous 'gpt-oss' (no size). Claims tolerate either variant; the gfx950
    AITER fix is asserted deterministically via check_overrides.py."""
    host = _require_host()
    _purge_containers(host)
    with _agent(skill=SKILL_NAME) as agent:
        run = agent.prompt(
            "Serve gpt-oss for inference on the AMD Instinct GPU server "
            "(host in ROCM_SSH_HOST). It is an MI350X (gfx950). Expose the API on port 8000. "
            "Before constructing the Docker command, run "
            "`python3 scripts/check_overrides.py --model <chosen-model-id> --gfx gfx950` "
            "and apply every env var it returns. "
            "After the endpoint is healthy, send a test chat completion and confirm the answer is coherent. "
            "Do not remove the container. Summarize the docker image, the exact docker run "
            "command, the overrides applied, the health-check result, and the inference output."
        )

        logs_matches(run, r"openai/gpt-oss")           # a gpt-oss variant targeted
        run.logs_contains("check_overrides.py")       # Step 3b was run
        # Hard invariant: AITER MoE path MUST be disabled for gpt-oss on gfx950.
        logs_matches(run, r"VLLM_ROCM_USE_AITER_MOE\s*=\s*0")
        logs_matches(run, r"vllm[/-]openai[/-]rocm")   # ROCm image, not CUDA
        logs_matches(run, r"/health")
        logs_matches(run, r"(200|HEALTHY|health.*ok)", re.IGNORECASE)

        run.should(
            "Resolve the ambiguous 'gpt-oss' request by selecting a specific gpt-oss variant "
            "(e.g. 20b or 120b) that fits the detected GPU VRAM, and state which one it chose"
        )
        run.should(
            "Apply the gfx950 override that sets VLLM_ROCM_USE_AITER_MOE=0 for gpt-oss, "
            "and explain that this prevents corrupted output on MI350X"
        )
        run.should_not("Leave the AITER MoE path enabled for gpt-oss on gfx950")
        run.should_not("Serve a gpt-oss variant that does not fit the detected GPU's VRAM")


def test_serve_kimi_k2_mxfp4():
    """AMD-quantized Kimi-K2.5-MXFP4: large MoE, MXFP4 weights already cached.
    Tests handling of AMD-quantized variants and multi-GPU MoE."""
    host = _require_host()
    _purge_containers(host)
    with _agent(skill=SKILL_NAME) as agent:
        run = agent.prompt(
            "Serve amd/Kimi-K2.5-MXFP4 for inference on the AMD Instinct GPU server "
            "(host in ROCM_SSH_HOST). Expose the API on port 8000. "
            "Before constructing the Docker command, run "
            "`python3 scripts/check_overrides.py --model amd/Kimi-K2.5-MXFP4 --gfx <detected-gfx>` "
            "and apply any overrides it returns. "
            "After the endpoint is healthy, send a test chat completion and confirm output is coherent. "
            "Do not remove the container. Summarize the docker image, exact docker run command, "
            "tensor-parallel degree used, any overrides applied, health-check result, and inference output."
        )

        logs_matches(run, r"amd/Kimi-K2.5-MXFP4|Kimi-K2\.5")
        run.logs_contains("check_overrides.py")
        logs_matches(run, r"vllm[/-]openai[/-]rocm")
        logs_matches(run, r"docker\s+run")
        logs_matches(run, r"/health")
        logs_matches(run, r"(200|HEALTHY|health.*ok)", re.IGNORECASE)

        run.should(
            "Use --distributed-executor-backend mp if serving across multiple GPUs, "
            "as required for MoE models on ROCm"
        )
        run.should("Send a chat completion request and receive non-empty coherent output")
        run.should("Leave the container running at the end")
        run.should_not("Use an NVIDIA/CUDA Docker image or the nvidia-docker runtime")


def test_serve_target_model():
    """Generic, model-agnostic case: derives expected behavior from the skill's
    own oracles and grades a THREE-WAY verdict + live endpoint probe. Point it at
    any model via SERVE_LLM_TARGET_MODEL."""
    host = _require_host()
    _purge_containers(host)
    category = _blacklist_category(MODEL_ID)

    with _agent(skill=SKILL_NAME) as agent:
        run = agent.prompt(SERVE_TEMPLATE.format(model=MODEL_ID, port=PORT))
        events = parse_events(run)

        # The agent must always acknowledge the exact model it was asked about.
        logs_matches(run, re.escape(MODEL_ID), re.IGNORECASE)

        print("  --- steps the agent took ---", flush=True)
        for ln in tool_trace(events):
            print("    " + ln, flush=True)

        grade = grade_serve_run(events, host, MODEL_ID, PORT, category)

        if grade.report is not None:
            print("  --- SKILL.md compliance ---", flush=True)
            for ln in check_lines(grade.report):
                print(ln, flush=True)
        print(f"  [environment] env_ok={grade.env_ok} -- "
              f"{'correct container + overrides' if grade.env_ok else grade.env_issues}", flush=True)
        print(f"  [outcome    ] served={grade.served} -- {grade.served_detail}", flush=True)
        print(f"  [skill      ] procedure={grade.procedure} -- {grade.procedure_detail}", flush=True)
        print(f"  [VERDICT    ] {grade.verdict}", flush=True)

        # Refusal class (only reached for blacklisted models).
        if grade.blacklisted:
            if grade.procedure == "REFUSED_BAD":
                report_fail(f"refuse({MODEL_ID})", grade.procedure_detail)
            report_pass(f"refuse({MODEL_ID})", grade.procedure_detail)
            return

        # THE focus failure: wrong container/environment must never happen.
        if not grade.env_ok:
            report_fail("correct_container_environment",
                        f"{grade.failure_category}: {grade.env_issues}")
        report_pass("correct_container_environment", "correct ROCm image + required overrides")

        # Outcome reality: a working endpoint (PASS chat / SERVED_OK base).
        if not grade.served_ok:
            report_fail("live_endpoint_output", grade.served_detail)
        report_pass("live_endpoint_output", grade.served_detail)

        # Skill engagement (secondary; this test runs WITH the skill).
        if not grade.engaged:
            report_fail("skill_engaged",
                        "agent used no skill scripts or data files (free-solved the task)")
        report_pass("skill_engaged", "agent used the skill's scripts/data")
        if grade.procedure == "NONCOMPLIANT":
            report_fail("skill_md_compliance", grade.procedure_detail)
        report_pass("skill_md_compliance", "followed the SKILL.md procedure")


# =========================================================================== #
# No-hardware ORACLE SELF-TESTS (synthetic events -> verdicts). These prove the
# failure detectors fire; they need no GPU, no API, no agent.
# =========================================================================== #

_MOD = sys.modules[__name__]  # for monkeypatching the live probes in self-tests


def bash(cmd, result="", is_error=False):
    return {"tool": "Bash", "command": cmd, "path": "", "result": result, "is_error": is_error}


def read(path):
    return {"tool": "Read", "command": "", "path": path, "result": "{}", "is_error": False}


def good_run(model="Qwen/Qwen3-8B", gfx="gfx942", image="vllm/vllm-openai-rocm:latest",
             extra_env="", include=("detect", "validate", "overrides", "data",
                                     "vram", "launch", "health")):
    """A procedure-following run. Drop steps via `include` to build failures."""
    ev = []
    if "detect" in include:
        ev.append(bash("python3 scripts/detect.py --host root@h",
                       result='{"gfx_version": "%s", "gpu_count": 8}' % gfx))
    if "validate" in include:
        ev.append(bash("python3 scripts/validate.py --auto-fix --host root@h",
                       result='{"ready": true}'))
    if "overrides" in include:
        ev.append(bash(f"python3 scripts/check_overrides.py --model {model} --gfx {gfx}",
                       result='{"env_set": {}}'))
    if "data" in include:
        ev.append(read("data/recipes_cache.json"))
        ev.append(read("data/gpu_overrides.json"))
    if "vram" in include:
        ev.append(bash(f"python3 scripts/estimate_vram.py --model-id {model} --vram-gb 192 --tp 1",
                       result='{"fit": {"weights_fit": true}}'))
    if "launch" in include:
        env = f" --env {extra_env}" if extra_env else ""
        ev.append(bash(f"ssh root@h 'docker run -d --name vllm{env} {image} "
                       f"--model {model} --port 8000'"))
    if "health" in include:
        ev.append(bash("ssh root@h 'curl -sf http://localhost:8000/health'", result="200"))
    return ev


def test_good_run_is_compliant():
    r = evaluate(good_run(), "Qwen/Qwen3-8B", "gfx942", expected_env_set={})
    assert r.engaged is True
    assert r.verdict == "COMPLIANT", r.summary()
    assert not r.required_violations


def test_free_solve_is_not_engaged():
    ev = [bash("ssh root@h 'docker run -d vllm/vllm-openai-rocm:latest --model X --port 8000'"),
          bash("ssh root@h 'curl http://localhost:8000/health'", result="200")]
    r = evaluate(ev, "X", "gfx942", expected_env_set={})
    assert r.engaged is False
    assert r.verdict == "NOT_ENGAGED", r.summary()
    assert r.verdict != "NONCOMPLIANT"


def test_skipped_check_overrides_is_noncompliant():
    ev = good_run(include=("detect", "validate", "data", "vram", "launch", "health"))
    r = evaluate(ev, "Qwen/Qwen3-8B", "gfx942", expected_env_set={})
    assert r.engaged is True
    assert r.verdict == "NONCOMPLIANT"
    assert any(s == "check_overrides" and sev == REQUIRED for s, sev, _ in r.violations), r.summary()


def test_cuda_image_is_noncompliant():
    ev = good_run(image="vllm/vllm-openai:latest")  # CUDA image, wrong environment
    r = evaluate(ev, "Qwen/Qwen3-8B", "gfx942", expected_env_set={})
    assert r.verdict == "NONCOMPLIANT"
    assert any(s == "launch" for s, sev, _ in r.required_violations), r.summary()


def test_nvidia_runtime_image_is_noncompliant():
    ev = good_run(image="nvidia/cuda:12.4.0-runtime")
    r = evaluate(ev, "Qwen/Qwen3-8B", "gfx942", expected_env_set={})
    assert r.verdict == "NONCOMPLIANT"
    assert any(s == "launch" for s, sev, _ in r.required_violations), r.summary()


def test_missing_aiter_override_is_noncompliant():
    # gpt-oss on gfx950 REQUIRES VLLM_ROCM_USE_AITER_MOE=0; launch omits it.
    ev = good_run(model="openai/gpt-oss-120b", gfx="gfx950")
    r = evaluate(ev, "openai/gpt-oss-120b", "gfx950",
                 expected_env_set={"VLLM_ROCM_USE_AITER_MOE": "0"})
    assert r.verdict == "NONCOMPLIANT"
    assert any(s == "override" for s, sev, _ in r.required_violations), r.summary()


def test_applied_aiter_override_is_compliant():
    ev = good_run(model="openai/gpt-oss-120b", gfx="gfx950",
                  extra_env="VLLM_ROCM_USE_AITER_MOE=0")
    r = evaluate(ev, "openai/gpt-oss-120b", "gfx950",
                 expected_env_set={"VLLM_ROCM_USE_AITER_MOE": "0"})
    assert r.verdict == "COMPLIANT", r.summary()


def test_wrong_gfx_in_check_overrides_is_noncompliant():
    # Host is gfx950 but the agent queried overrides for gfx942 -> wrong environment.
    ev = good_run(model="openai/gpt-oss-120b", gfx="gfx950",
                  extra_env="VLLM_ROCM_USE_AITER_MOE=0")
    for e in ev:
        if "check_overrides.py" in e["command"]:
            e["command"] = "python3 scripts/check_overrides.py --model openai/gpt-oss-120b --gfx gfx942"
    r = evaluate(ev, "openai/gpt-oss-120b", "gfx950",
                 expected_env_set={"VLLM_ROCM_USE_AITER_MOE": "0"})
    assert r.verdict == "NONCOMPLIANT"
    assert any(s == "check_overrides" for s, sev, _ in r.required_violations), r.summary()


def test_missing_health_is_noncompliant():
    ev = good_run(include=("detect", "validate", "overrides", "data", "vram", "launch"))
    r = evaluate(ev, "Qwen/Qwen3-8B", "gfx942", expected_env_set={})
    assert r.verdict == "NONCOMPLIANT"
    assert any(s == "health" for s, sev, _ in r.required_violations), r.summary()


def test_no_launch_is_noncompliant():
    ev = good_run(include=("detect", "validate", "overrides", "data", "vram", "health"))
    r = evaluate(ev, "Qwen/Qwen3-8B", "gfx942", expected_env_set={})
    assert r.verdict == "NONCOMPLIANT"
    assert any(s == "launch" for s, sev, _ in r.required_violations), r.summary()


def test_detect_errored_is_noncompliant():
    ev = good_run()
    for e in ev:
        if "detect.py" in e["command"]:
            e["is_error"] = True
            e["result"] = "ssh: connect to host h port 22: Connection refused"
    r = evaluate(ev, "Qwen/Qwen3-8B", "gfx942", expected_env_set={})
    assert r.verdict == "NONCOMPLIANT"
    assert any(s == "detect" for s, sev, _ in r.required_violations), r.summary()


def test_expected_warnings_do_not_fail_by_default_but_do_in_strict():
    # Drop validate (EXPECTED severity). Default: still COMPLIANT. Strict: fails.
    ev = good_run(include=("detect", "overrides", "data", "vram", "launch", "health"))
    r = evaluate(ev, "Qwen/Qwen3-8B", "gfx942", expected_env_set={})
    assert r.verdict == "COMPLIANT", r.summary()
    assert any(s == "validate" for s, _, _ in r.violations)
    r_strict = evaluate(ev, "Qwen/Qwen3-8B", "gfx942", expected_env_set={}, strict=True)
    assert r_strict.verdict == "NONCOMPLIANT", r_strict.summary()


def test_grade_serve_run_not_engaged_without_hardware():
    # Stub the outcome probe (no SSH) so the test is instant; endpoint is DOWN.
    ev = [bash("ssh root@h 'docker run -d vllm/vllm-openai-rocm:latest --model X --port 8000'")]
    orig = _MOD._endpoint_up
    _MOD._endpoint_up = lambda *a, **k: False
    try:
        g = grade_serve_run(ev, host="", model="Qwen/Qwen3-8B", port=8000, category=None)
        assert g.procedure == "NOT_ENGAGED", g
        assert not g.engaged
        assert g.served == "DOWN"
        assert not g.passed
    finally:
        _MOD._endpoint_up = orig


def test_grade_serve_run_noncompliant_without_hardware():
    ev = good_run(include=("detect", "validate", "data", "vram", "launch", "health"))  # no check_overrides
    orig = _MOD._endpoint_up
    _MOD._endpoint_up = lambda *a, **k: False
    try:
        g = grade_serve_run(ev, host="", model="Qwen/Qwen3-8B", port=8000, category=None)
        assert g.procedure == "NONCOMPLIANT", g
        assert "check_overrides" in g.procedure_detail
    finally:
        _MOD._endpoint_up = orig


def test_freesolve_correct_container_is_success_even_if_not_engaged():
    # A free-solve that serves a working endpoint with the RIGHT ROCm container is
    # a SUCCESS (success = working endpoint + correct env). Engagement is separate.
    ev = [bash("ssh root@h 'docker run -d vllm/vllm-openai-rocm:latest --model X --port 8000'")]
    orig = (_MOD._endpoint_up, _MOD._served_model_name, _MOD._probe_completion)
    _MOD._endpoint_up = lambda *a, **k: True
    _MOD._served_model_name = lambda *a, **k: "meta-llama/Llama-3.1-8B"
    _MOD._probe_completion = lambda *a, **k: " Paris, the capital of France."
    try:
        g = grade_serve_run(ev, host="", model="meta-llama/Llama-3.1-8B", port=8000)
        assert g.success and g.verdict == "SUCCESS", g
        assert g.env_ok and g.served_ok
        assert g.procedure == "NOT_ENGAGED" and not g.engaged  # secondary
    finally:
        _MOD._endpoint_up, _MOD._served_model_name, _MOD._probe_completion = orig


def test_evaluate_environment_flags_cuda_image():
    ev = [bash("ssh h 'docker run -d nvidia/cuda:12.4.0 --model X --port 8000'")]
    env = evaluate_environment(ev, "Qwen/Qwen3-8B", "gfx942", {})
    assert not env["env_ok"]
    assert any(c == "WRONG_CONTAINER" for c, _ in env["issues"])


def test_evaluate_environment_flags_missing_override():
    ev = [bash("ssh h 'docker run -d vllm/vllm-openai-rocm:latest --model openai/gpt-oss-120b --port 8000'")]
    env = evaluate_environment(ev, "openai/gpt-oss-120b", "gfx950",
                               {"VLLM_ROCM_USE_AITER_MOE": "0"})
    assert not env["env_ok"]
    assert any(c == "WRONG_ENV" for c, _ in env["issues"])


def test_evaluate_environment_accepts_correct_setup():
    ev = [bash("ssh h 'docker run -d --env VLLM_ROCM_USE_AITER_MOE=0 "
               "vllm/vllm-openai-rocm:latest --model openai/gpt-oss-120b --port 8000'")]
    env = evaluate_environment(ev, "openai/gpt-oss-120b", "gfx950",
                               {"VLLM_ROCM_USE_AITER_MOE": "0"})
    assert env["env_ok"], env


def test_grade_wrong_env_is_failure_even_when_endpoint_looks_healthy():
    # THE killer case: a free-solve serves a HEALTHY endpoint that even answers the
    # probe, but it skipped the mandatory override -> wrong environment -> FAIL.
    ev = [bash("ssh h 'docker run -d vllm/vllm-openai-rocm:latest --model openai/gpt-oss-120b --port 8000'")]
    orig = (_MOD._endpoint_up, _MOD._served_model_name, _MOD._probe_chat,
            _MOD._detect_gfx, _MOD._expected_overrides)
    _MOD._endpoint_up = lambda *a, **k: True
    _MOD._served_model_name = lambda *a, **k: "openai/gpt-oss-120b"
    _MOD._probe_chat = lambda *a, **k: "Paris"          # endpoint LOOKS fine
    _MOD._detect_gfx = lambda *a, **k: "gfx950"
    _MOD._expected_overrides = lambda *a, **k: {"VLLM_ROCM_USE_AITER_MOE": "0"}
    try:
        g = grade_serve_run(ev, host="", model="openai/gpt-oss-120b", port=8000)
        assert g.served_ok            # probe passed...
        assert not g.env_ok           # ...but environment is wrong
        assert not g.success          # so the run FAILS
        assert g.failure_category == "WRONG_ENV", g
        assert g.verdict == "FAIL:WRONG_ENV"
    finally:
        (_MOD._endpoint_up, _MOD._served_model_name, _MOD._probe_chat,
         _MOD._detect_gfx, _MOD._expected_overrides) = orig


def test_chat_vs_base_classification():
    for m in ["Qwen/Qwen2.5-1.5B-Instruct", "google/gemma-2-2b-it",
              "meta-llama/Llama-3.3-70B-Instruct", "deepseek-ai/DeepSeek-R1",
              "Qwen/QwQ-32B", "openai/gpt-oss-120b", "moonshotai/Kimi-K2-Thinking"]:
        assert _is_chat_model(m), f"should be chat: {m}"
    for m in ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.1-8B",
              "meta-llama/Meta-Llama-3-8B", "openai-community/gpt2",
              "openbmb/InfLLM-V2-Short-Dense-Base"]:
        assert not _is_chat_model(m), f"should be base: {m}"


def test_degenerate_output_detection():
    junk = ("What is the capital of France? Answer with only the city name.edReader\n\n" * 12)
    assert _is_degenerate(junk, "What is the capital of France? Answer with only the city name.")
    assert _is_degenerate("", "x")
    assert _is_degenerate("The capital of France is", "The capital of France is")  # echo only
    assert not _is_degenerate(" Paris, the largest city in France.", "The capital of France is")


def test_base_model_served_ok_path_without_hardware():
    # A COMPLIANT base model that serves real output -> served=SERVED_OK AND
    # procedure=COMPLIANT -> full pass.
    ev = good_run(model="meta-llama/Llama-3.2-1B", gfx="gfx942")
    orig = (_MOD._endpoint_up, _MOD._served_model_name, _MOD._probe_completion)
    _MOD._endpoint_up = lambda *a, **k: True
    _MOD._served_model_name = lambda *a, **k: "meta-llama/Llama-3.2-1B"
    _MOD._probe_completion = lambda *a, **k: " Paris, a city in northern France."
    try:
        g = grade_serve_run(ev, host="", model="meta-llama/Llama-3.2-1B", port=8000)
        assert g.served == "SERVED_OK", g
        assert g.procedure == "COMPLIANT", g
        assert g.passed
    finally:
        _MOD._endpoint_up, _MOD._served_model_name, _MOD._probe_completion = orig


# Live tests need a reachable GPU host + the `claude` CLI; everything else is a
# no-hardware oracle self-test.
_LIVE_TESTS = {
    "test_serve_qwen3_on_instinct", "test_refuses_blacklisted_model",
    "test_ambiguous_gpt_oss_request", "test_serve_kimi_k2_mxfp4",
    "test_serve_target_model",
}

# Convenience: `python evals.py` runs the no-hardware oracle self-tests only
# (the live cases need a GPU host and the `claude` CLI, and skip via _require_host).
if __name__ == "__main__":
    _oracle = [v for k, v in sorted(globals().items())
               if k.startswith("test_") and callable(v) and k not in _LIVE_TESTS]
    failed = 0
    for fn in _oracle:
        try:
            fn()
            print(f"[PASS] {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {fn.__name__}: {e}")
    print(f"\n{len(_oracle) - failed}/{len(_oracle)} oracle self-tests passed")
    sys.exit(1 if failed else 0)
