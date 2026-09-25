#!/usr/bin/env python3
"""Create, trigger, poll, and (on interruption) cancel an OrchestrAI run."""

from __future__ import annotations

import json
import os
import re
import signal
import time
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

TERMINAL_PIPELINE_STATES = {
    "passed",
    "failed",
    "unstable",
    "error",
    "cancelled",
    "aborted",
}
PASS_TEST_STATES = {"passed", "completed", "finished"}
TERMINAL_TEST_STATES = PASS_TEST_STATES | {
    "failed",
    "error",
    "skipped",
    "cancelled",
    "aborted",
}
PASS_SESSION_STATES = {"completed", "finished", "passed"}
TERMINAL_SESSION_STATES = PASS_SESSION_STATES | {
    "failed",
    "error",
    "cancelled",
    "aborted",
}
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
JENKINS_TIMESTAMP_RE = re.compile(
    r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z\]\s*"
)
FAILURE_LINE_RE = re.compile(
    r"(?:\bERROR\b|timed? out|acquisition failed|rolling back|"
    r"unexpected exit|pipeline-level failure|actor offline|"
    r"infrastructure polling failed|no machines? available)",
    re.IGNORECASE,
)
TRUSTED_FAILURE_PREFIX_RE = re.compile(
    r"^(?:\[(?:provision|report|launcher|poll|05-[^\]]+)\]|"
    r"L[1-5](?:-[A-Za-z0-9]+)?:)",
    re.IGNORECASE,
)
SENSITIVE_VALUE_RE = re.compile(
    r"(?i)\b(token|password|passwd|secret|api[_-]?key|authorization)"
    r"(\s*[:=]\s*)([^\s,;]+)"
)
MAX_FAILURE_LINES = 20
SAFE_SESSION_STATES = TERMINAL_SESSION_STATES | {
    "queued",
    "pending",
    "running",
    "starting",
    "unknown",
}
SAFE_TEST_STATES = TERMINAL_TEST_STATES | {
    "queued",
    "pending",
    "running",
    "starting",
    "unknown",
}
SAFE_PIPELINE_STATES = TERMINAL_PIPELINE_STATES | {
    "created",
    "queued",
    "pending",
    "running",
    "starting",
    "provisioning",
    "unknown",
}

UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
)


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def log(message: str) -> None:
    print(f"[orchestrai] {message}", flush=True)


def _safe_private_summary_url(raw: str) -> str:
    """Validate a private HTTPS link before writing it to a job summary."""
    raw = raw.strip()
    if (
        env("PUBLISH_REPORT_LINK").lower() not in {"1", "true", "yes"}
        or not raw
        or len(raw) > 2048
        or any(character.isspace() or character in '<>"' for character in raw)
    ):
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ""
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return ""
    return raw


def reportportal_url_for_summary(live: dict) -> str:
    """Return a safe ReportPortal link only for a private repository summary."""
    raw = str(live.get("rp_url") or "").strip()
    return _safe_private_summary_url(raw)


def portal_run_url_for_summary(base_url: str, run_id: str) -> str:
    """Build the authenticated Portal run route for a private job summary."""
    base_url = _safe_private_summary_url(base_url)
    if not base_url or not UUID_RE.fullmatch(run_id):
        return ""
    try:
        parsed = urlsplit(base_url)
    except ValueError:
        return ""
    # Only publish an origin. A path in the API secret could itself be private,
    # and it would not be a valid base for the Portal's browser route anyway.
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        return ""
    return f"{base_url.rstrip('/')}/#/runs/{run_id}"


def secret_mask_resistant_markdown_target(url: str) -> str:
    """Encode the scheme delimiter without changing the rendered link target.

    GitHub masks an Actions secret wherever its exact value occurs, including
    inside GITHUB_STEP_SUMMARY. The Portal API origin is configured as a secret,
    so an ordinary Markdown URL is rewritten to ``***`` and becomes a broken
    relative link. CommonMark decodes the numeric entity in the generated href,
    while the raw summary never contains the exact secret value.

    Callers must pass only a URL already accepted by
    :func:`portal_run_url_for_summary`.
    """
    return url.replace(":", "&#58;", 1)


def require_linux_provisioning(plan: dict) -> None:
    """Fail before allocation when a live Linux run has no driver map."""
    linux_requested = any(
        str(session.get("os_image") or "") == "ubuntu"
        or str(session.get("os_image") or "").startswith("ubuntu/")
        for session in plan.get("sessions", [])
        if isinstance(session, dict)
    )
    if linux_requested and not isinstance(plan.get("builds_json"), dict):
        raise SystemExit(
            "missing required secret(s): ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON"
        )


def build_run_request(plan: dict, plan_id: str) -> dict[str, Any]:
    """Return the private trigger body without logging or exporting it."""
    request: dict[str, Any] = {"plan_id": plan_id}
    if isinstance(plan.get("builds_json"), dict):
        request["builds_json"] = plan["builds_json"]
    return request


def safe_status(value: object, allowed: set[str]) -> str:
    status = str(value or "unknown").strip().lower()
    return status if status in allowed else "unknown"


def safe_duration(value: object) -> str:
    duration = str(value or "").strip().lower()
    if re.fullmatch(
        r"\d+(?:\.\d+)?\s*(?:ms|s|sec|secs|seconds?|m|min|mins|minutes?)", duration
    ):
        return duration
    return ""


def classify_launcher_failure(value: object, default: str) -> str:
    """Map launcher-owned text to a fixed message before exporting it."""
    lowered = str(value or "").lower()
    if "timed out" in lowered or "timeout" in lowered:
        return "The behavioral test timed out."
    if "offline" in lowered:
        return "The allocated actor went offline."
    if "unreachable" in lowered or "connection" in lowered:
        return "The behavioral test launcher was unreachable."
    if "cancel" in lowered or "abort" in lowered:
        return "The behavioral test was cancelled."
    if "setup" in lowered or "install" in lowered or "dependency" in lowered:
        return "Behavioral test setup failed."
    return default


def classify_behavioral_test_output(test: dict[str, Any]) -> str:
    """Map private test output to one fixed, credential-free diagnostic.

    Launcher stdout and stderr can contain credentials, private hostnames,
    machine identities, paths, and full agent transcripts. They are therefore
    inspected only inside the controller and never copied into its artifacts.
    Keep this allow-list exact and return only constant strings.
    """
    output = "\n".join(
        str(test.get(stream) or "")[-1_000_000:] for stream in ("stdout", "stderr")
    ).lower()
    if not output.strip():
        return ""

    if "skillscope setup artifacts are missing" in output:
        return "Skillscope setup artifacts were missing on the test machine."
    if "checked-out skills commit does not match skills_sha" in output:
        return "The checked-out skills commit did not match the requested commit."
    if any(
        marker in output
        for marker in (
            "secure node variable llm_gateway_key is unavailable",
            "secure node variable llm_gateway_url is unavailable",
        )
    ):
        return "LLM gateway configuration was unavailable on the test machine."
    if any(
        marker in output
        for marker in (
            "dependency installation failed",
            "install steps failed for 'lib.amd.skills-runner'",
        )
    ):
        return "The skills test dependencies could not be installed."
    if "unprivileged skills runner is unavailable" in output:
        return "An unprivileged Linux test account was unavailable."
    if "could not prepare the skills artifact directory" in output:
        return "The Linux skills workspace could not be prepared."
    if "error: claude api not reachable" in output:
        if any(
            marker in output
            for marker in (
                "authentication_error",
                "authentication failed",
                "unauthorized",
                "invalid api key",
                "invalid x-api-key",
                "status code 401",
                "http 401",
            )
        ):
            return "The Claude API preflight could not authenticate through the LLM gateway."
        if "timed out" in output or "timeout" in output:
            return "The Claude API preflight timed out."
        return "The Claude API preflight could not reach the LLM gateway."
    if any(
        marker in output
        for marker in (
            "behavioral exceeded --timeout",
            "agent timed out after",
        )
    ):
        return "The behavioral test timed out."
    if "produced no parseable stream-json output" in output:
        if "cannot be used with root/sudo privileges" in output:
            return "Claude Code could not run under the Linux test account."
        if any(
            marker in output
            for marker in (
                "unknown option",
                "unknown argument",
                "unrecognized option",
                "unrecognized argument",
                "unexpected argument",
                "invalid value for '--output-format'",
            )
        ):
            return "Claude Code rejected the behavioral invocation."
        if any(
            marker in output
            for marker in (
                "authentication_error",
                "authentication failed",
                "unauthorized",
                "invalid api key",
                "invalid x-api-key",
                "status code 401",
                "http 401",
            )
        ):
            return "The Claude agent could not authenticate through the LLM gateway."
        if any(
            marker in output
            for marker in (
                "rate_limit_error",
                "rate limit",
                "rate-limit",
                "status code 429",
                "http 429",
            )
        ):
            return "The Claude agent was rate limited by the LLM gateway."
        if any(
            marker in output
            for marker in (
                "overloaded_error",
                "service unavailable",
                "status code 529",
                "http 529",
            )
        ):
            return "The Claude service was temporarily unavailable."
        if any(
            marker in output
            for marker in (
                "model_not_found",
                "model not found",
                "unknown model",
                "unsupported model",
                "does not have access to model",
            )
        ):
            return "The requested Claude model was unavailable through the LLM gateway."
        if any(
            marker in output
            for marker in (
                "context_length_exceeded",
                "prompt is too long",
                "request too large",
                "maximum context length",
            )
        ):
            return "The Claude agent exceeded the gateway context limit."
        if any(
            marker in output
            for marker in (
                "connection refused",
                "connection reset",
                "network error",
                "socket hang up",
                "fetch failed",
                "econnrefused",
                "econnreset",
                "enotfound",
            )
        ):
            return "The Claude agent lost connectivity to the LLM gateway."
        if any(
            marker in output
            for marker in (
                "illegal instruction",
                "segmentation fault",
                "symbol lookup error",
            )
        ):
            return "The Claude Code process crashed before returning a result."
        return "The Claude agent did not return a usable result."
    if "[fail]" in output and " checks in " in output:
        return "One or more behavioral expectations were not met."
    return ""


def classify_controller_exception(value: BaseException) -> str:
    """Return a useful fixed error without exporting a remote response body."""
    message = str(value)
    if isinstance(value, json.JSONDecodeError):
        return "The OrchestrAI Portal returned an invalid response."
    safe_patterns = (
        r"^OrchestrAI Portal request failed with HTTP \d{3}; "
        r"inspect the Portal logs for details$",
        r"^OrchestrAI Portal was unreachable$",
        r"^OrchestrAI Portal request timed out$",
        r"^could not resolve exact stock OS images: OrchestrAI Portal .+$",
        r"^could not resolve stock OS images: no images were returned$",
        r"^could not resolve generic '(?:ubuntu|windows)' to a compatible image$",
        r"^login response did not contain a token$",
        r"^plan creation response did not contain an id$",
        r"^run trigger response did not contain an id$",
        r"^run timed out after \d+ seconds$",
    )
    if any(re.fullmatch(pattern, message) for pattern in safe_patterns):
        return message
    if "space" in message.lower():
        return (
            "The configured OrchestrAI space was not available to the service account."
        )
    return "The OrchestrAI controller failed; inspect the Portal logs for details."


def poll_outage_exhausted(
    first_failure_at: float, *, now: float, timeout_seconds: int
) -> bool:
    """Use elapsed time, not request count, for the control-plane retry budget."""
    return now - first_failure_at >= timeout_seconds


def write_output(name: str, value: str) -> None:
    if path := env("GITHUB_OUTPUT"):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def classify_failure_line(line: str) -> str:
    """Map a trusted console failure to a fixed message with no remote payload."""
    lowered = line.lower()
    fixed_messages = {
        "Machine acquisition failed.",
        "The allocated actor went offline.",
        "Infrastructure polling failed.",
        "The pipeline rolled back after an infrastructure failure.",
        "An infrastructure process exited unexpectedly.",
        "Provisioning failed.",
        "The launcher failed.",
        "Report generation failed.",
        "The OrchestrAI pipeline reported an infrastructure failure.",
    }
    if line in fixed_messages or re.fullmatch(
        r"(?:Machine acquisition|An infrastructure operation) timed out"
        r"(?: after \d+(?:\.\d+)?\s*(?:seconds?|secs?|s|minutes?|mins?|m))?\.",
        line,
    ):
        return line
    if "timed out" in lowered or "timeout" in lowered:
        duration = re.search(
            r"\b\d+(?:\.\d+)?\s*(?:seconds?|secs?|s|minutes?|mins?|m)\b",
            lowered,
        )
        suffix = f" after {duration.group(0)}" if duration else ""
        if "machine" in lowered or "acqui" in lowered or "provision" in lowered:
            return f"Machine acquisition timed out{suffix}."
        return f"An infrastructure operation timed out{suffix}."
    if "no machine" in lowered or "acquisition failed" in lowered:
        return "Machine acquisition failed."
    if "actor offline" in lowered:
        return "The allocated actor went offline."
    if "polling failed" in lowered:
        return "Infrastructure polling failed."
    if "rolling back" in lowered:
        return "The pipeline rolled back after an infrastructure failure."
    if "unexpected exit" in lowered:
        return "An infrastructure process exited unexpectedly."
    if line.lower().startswith("[provision]"):
        return "Provisioning failed."
    if line.lower().startswith("[launcher]"):
        return "The launcher failed."
    if line.lower().startswith("[report]"):
        return "Report generation failed."
    if line.lower().startswith("[poll]"):
        return "Infrastructure polling failed."
    return "The OrchestrAI pipeline reported an infrastructure failure."


def safe_failure_message(value: object) -> str:
    """Allow only messages generated by this controller's fixed classifiers."""
    line = ANSI_ESCAPE_RE.sub("", str(value or "")).strip()
    synthetic = {
        "The OrchestrAI pipeline failed before any requested test session produced a result.",
        "The OrchestrAI pipeline failed before any requested test session produced a result; the launcher was unreachable.",
    }
    if line in synthetic:
        return line
    controller = classify_controller_exception(RuntimeError(line))
    if controller == line:
        return line
    return classify_failure_line(line)


class PortalClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = ""
        self.space_id = ""

    def request(self, method: str, path: str, payload: dict | None = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.space_id:
            headers["X-Space-Id"] = self.space_id
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            # Consume the response so the connection can be reused, but never
            # copy a server-owned body into public GitHub logs or artifacts.
            exc.read(2000)
            raise RuntimeError(
                f"OrchestrAI Portal request failed with HTTP {exc.code}; "
                "inspect the Portal logs for details"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("OrchestrAI Portal was unreachable") from exc
        except TimeoutError as exc:
            raise RuntimeError("OrchestrAI Portal request timed out") from exc
        return json.loads(body) if body else {}

    def cancel(self, run_id: str) -> None:
        try:
            self.request("POST", f"/api/runs/{run_id}/cancel", {})
            log("cancellation requested for the active run")
        except Exception:  # best effort while the process is being interrupted
            log("warning: could not cancel the active run")


def validate_plan(path: Path) -> dict:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid plan file {path}: {exc}") from exc
    sessions = plan.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise SystemExit("plan must contain at least one session")
    if not all(isinstance(s, dict) and s.get("tests") for s in sessions):
        raise SystemExit("every plan session must contain at least one test")
    return plan


def _is_windows_client_image(image: str) -> bool:
    version = image[len("windows") : len("windows") + 1]
    return "/" not in image and image.startswith("windows") and version.isdigit()


def resolve_stock_os_images(plan: dict, client: PortalClient) -> None:
    """Replace generic OS selectors with exact broker-allowlisted images."""
    generic_sessions = [
        session
        for session in plan.get("sessions", [])
        if str(session.get("os_image") or "") in {"ubuntu", "windows"}
    ]
    if not generic_sessions:
        return

    try:
        resources = client.request("GET", "/api/playground/boot-resources")
    except RuntimeError as exc:
        raise RuntimeError(f"could not resolve exact stock OS images: {exc}") from exc
    if not isinstance(resources, list) or not resources:
        raise RuntimeError("could not resolve stock OS images: no images were returned")

    available_images = [
        (str(item.get("os")), str(item.get("type") or ""))
        for item in resources
        if isinstance(item, dict) and item.get("os")
    ]
    for session in generic_sessions:
        requested = str(session.get("os_image") or "")
        if requested == "windows":
            # A generic Windows behavioral run needs a normal client image.
            # Broker client images are named windows11-* at the top level;
            # windows/azure-host-* and windows/vhlk-controller-* are
            # specialized images and must not be selected.
            candidates = [
                (image, image_type)
                for image, image_type in available_images
                if _is_windows_client_image(image)
            ]
        else:
            candidates = [
                (image, image_type)
                for image, image_type in available_images
                if image.startswith("ubuntu/")
            ]
        if not candidates:
            raise RuntimeError(
                f"could not resolve generic {requested!r} to a compatible image"
            )

        # Prefer a warm ratio image. Fall back to a broker-allowlisted
        # on-demand image when the requested OS has no ratio entry.
        candidates.sort(key=lambda item: (item[1] != "ratio", item[0]))
        session["os_image"] = candidates[0][0]
        log(f"resolved {requested} to an available stock image")


def expected_items(plan: dict) -> list[dict[str, str]]:
    """Return the stable skill/OS identity expected from every plan session."""
    items = []
    names: set[str] = set()
    identities: set[tuple[str, str]] = set()
    for session in plan.get("sessions", []):
        name = str(session.get("name") or "")
        tests = session.get("tests") or []
        if not name or name in names:
            raise ValueError(
                f"plan contains a missing or duplicate session name: {name!r}"
            )
        if (
            not isinstance(tests, list)
            or len(tests) != 1
            or not isinstance(tests[0], dict)
        ):
            raise ValueError(f"session {name!r} must contain exactly one test object")
        test = tests[0]
        variables = test.get("variables") or {}
        skill = str(variables.get("SKILLS") or "")
        os_name = str(variables.get("SKILLS_OS") or "")
        path = str(test.get("path") or "")
        if not skill or "," in skill or os_name not in {"Linux", "Windows"} or not path:
            raise ValueError(f"session {name!r} has invalid reporting metadata")
        identity = (skill, os_name)
        if identity in identities:
            raise ValueError(f"plan contains duplicate skill/OS identity: {identity!r}")
        names.add(name)
        identities.add(identity)
        items.append({"session": name, "skill": skill, "os": os_name, "path": path})
    return items


def _matches_session(live_name: str, expected_name: str) -> bool:
    return live_name == expected_name or live_name.startswith(expected_name + " #")


def jenkins_failure_summary(live: dict, *, limit: int = 20) -> list[str]:
    """Return bounded, redacted failure lines from the Jenkins console."""
    if not isinstance(live, dict):
        return []
    jenkins = live.get("jenkins") or {}
    lines: list[str] = []
    retained = jenkins.get("safe_failure_summary") if isinstance(jenkins, dict) else []
    if isinstance(retained, list):
        lines.extend(str(line) for line in retained if str(line).strip())
    log_text = jenkins.get("log_text") if isinstance(jenkins, dict) else ""
    if not isinstance(log_text, str):
        log_text = ""
    for raw_line in log_text.splitlines():
        line = ANSI_ESCAPE_RE.sub("", raw_line).strip()
        # Jenkins consoleText prefixes each line with an ISO timestamp. Strip
        # only that exact shape before applying the trusted-component allowlist.
        line = JENKINS_TIMESTAMP_RE.sub("", line, count=1)
        if (
            not line
            or not TRUSTED_FAILURE_PREFIX_RE.search(line)
            or not FAILURE_LINE_RE.search(line)
        ):
            continue
        lines.append(classify_failure_line(line))

    sanitized: list[str] = []
    for line in lines:
        line = safe_failure_message(line)
        line = re.sub(r"(?i)\bBearer\s+\S+", "Bearer ***", line)
        line = re.sub(r"https://[^/\s:@]+:[^@\s/]+@", "https://***@", line)
        line = SENSITIVE_VALUE_RE.sub(r"\1\2***", line)[:500]
        if line and line not in sanitized:
            sanitized.append(line)
    return sanitized[-max(limit, 1) :]


def merge_live_snapshot(previous: dict, current: dict) -> dict:
    """Keep the latest live state while retaining only safe failure context.

    The Portal live endpoint advances its Jenkins log cursor. A later response can
    therefore omit the failure line returned by an earlier response. Persist the
    redacted summary, never the full console, across those polls.
    """
    merged = deepcopy(current)
    failure_lines = jenkins_failure_summary(previous) + jenkins_failure_summary(current)
    deduplicated: list[str] = []
    for line in failure_lines:
        if line not in deduplicated:
            deduplicated.append(line)
    if deduplicated:
        jenkins = merged.setdefault("jenkins", {})
        if isinstance(jenkins, dict):
            jenkins["safe_failure_summary"] = deduplicated[-MAX_FAILURE_LINES:]
            jenkins.pop("log_text", None)
    return merged


def build_results_manifest(
    plan: dict,
    live: dict,
    *,
    mode: str,
    pipeline_status: str = "",
    tests_status: str = "",
) -> dict:
    """Normalize the Portal live snapshot into one verdict per skill and OS."""
    launcher = (live.get("launcher") if isinstance(live, dict) else {}) or {}
    if not isinstance(launcher, dict):
        launcher = {}
    live_sessions = launcher.get("sessions") or []
    if not isinstance(live_sessions, list):
        live_sessions = []
    failure_summary = jenkins_failure_summary(live)
    launcher_reachable = launcher.get("reachable")
    if (
        not failure_summary
        and pipeline_status in {"failed", "unstable", "error", "cancelled", "aborted"}
        and not live_sessions
    ):
        reachability = (
            "; the launcher was unreachable" if launcher_reachable is False else ""
        )
        failure_summary = [
            "The OrchestrAI pipeline failed before any requested test session "
            f"produced a result{reachability}."
        ]

    items = []
    for expected in expected_items(plan):
        item: dict[str, Any] = {
            **expected,
            "status": "missing",
            "session_status": "missing",
            "duration": "",
            "error": "",
            "terminal": False,
        }
        matches = [
            session
            for session in live_sessions
            if isinstance(session, dict)
            and _matches_session(str(session.get("name") or ""), expected["session"])
        ]
        if len(matches) != 1:
            if failure_summary and pipeline_status in TERMINAL_PIPELINE_STATES:
                item["status"] = "error"
            item["error"] = f"expected exactly one live session, found {len(matches)}"
            if failure_summary:
                item["error"] += f"; infrastructure: {failure_summary[-1]}"
            items.append(item)
            continue

        session = matches[0]
        session_status = safe_status(session.get("status"), SAFE_SESSION_STATES)
        item["session_status"] = session_status
        tests = session.get("tests") or []
        matching_tests = [
            test
            for test in tests
            if isinstance(test, dict)
            and expected["path"]
            in {str(test.get("path") or ""), str(test.get("name") or "")}
        ]
        if len(matching_tests) != 1:
            item["error"] = (
                f"expected exactly one result for {expected['path']!r}, "
                f"found {len(matching_tests)}"
            )
            items.append(item)
            continue

        test = matching_tests[0]
        raw_status = safe_status(test.get("status"), SAFE_TEST_STATES)
        item["duration"] = safe_duration(test.get("duration"))
        item["terminal"] = (
            raw_status in TERMINAL_TEST_STATES
            and session_status in TERMINAL_SESSION_STATES
        )
        if raw_status in PASS_TEST_STATES and session_status in PASS_SESSION_STATES:
            item["status"] = "passed"
        elif raw_status in PASS_TEST_STATES:
            item["status"] = "error"
            item["error"] = classify_launcher_failure(
                session.get("error"), "The behavioral test session did not complete."
            )
        else:
            item["status"] = (
                raw_status if raw_status in TERMINAL_TEST_STATES else "error"
            )
            item["error"] = (
                classify_launcher_failure(session.get("error") or test.get("error"), "")
                or classify_behavioral_test_output(test)
                or "The behavioral test did not pass."
            )
        items.append(item)

    return {
        "schema_version": 1,
        "mode": mode,
        "ready": bool(items) and all(item["terminal"] for item in items),
        "run": {
            "pipeline_status": pipeline_status,
            "tests_status": tests_status,
            "failure_summary": failure_summary,
        },
        "items": items,
    }


def mock_results_manifest(plan: dict) -> dict:
    items = [
        {
            **item,
            "status": "mock",
            "session_status": "mock",
            "duration": "",
            "error": "",
            "terminal": True,
        }
        for item in expected_items(plan)
    ]
    return {
        "schema_version": 1,
        "mode": "mock",
        "ready": True,
        "run": {
            "pipeline_status": "mock",
            "tests_status": "mock",
            "failure_summary": [],
        },
        "items": items,
    }


def infrastructure_failure_manifest(plan: dict, *, mode: str, error: str) -> dict:
    return {
        "schema_version": 1,
        "mode": mode,
        "ready": False,
        "run": {
            "pipeline_status": "error",
            "tests_status": "unknown",
            "failure_summary": [error],
        },
        "items": [
            {
                **item,
                "status": "error",
                "session_status": "missing",
                "duration": "",
                "error": error,
                "terminal": False,
            }
            for item in expected_items(plan)
        ],
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_failure_artifacts(
    plan: dict,
    *,
    mode: str,
    error: str,
    results_path: Path,
    live_path: Path,
    live: dict,
) -> dict:
    """Replace partial controller output with a safe, fail-closed result pair."""
    results = infrastructure_failure_manifest(plan, mode=mode, error=error)
    _write_json(results_path, results)
    _write_json(live_path, sanitized_live_snapshot(live))
    return results


def sanitized_live_snapshot(live: dict) -> dict:
    """Return a bounded diagnostic snapshot with no raw logs or plan variables."""
    if not isinstance(live, dict):
        return {}
    jenkins = live.get("jenkins") or {}
    launcher = live.get("launcher") or {}
    if not isinstance(jenkins, dict):
        jenkins = {}
    if not isinstance(launcher, dict):
        launcher = {}
    clean_sessions = []
    raw_sessions = launcher.get("sessions") or []
    if not isinstance(raw_sessions, list):
        raw_sessions = []
    for session in raw_sessions:
        if not isinstance(session, dict):
            continue
        clean_tests = []
        for test in session.get("tests") or []:
            if not isinstance(test, dict):
                continue
            raw_path = str(test.get("path") or test.get("name") or "")
            clean_test: dict[str, Any] = {
                "status": safe_status(test.get("status"), SAFE_TEST_STATES)
            }
            if re.fullmatch(
                r"L4-sys/skills/(?:linux|windows)/sys_func-skills_behavioral",
                raw_path,
            ):
                clean_test["path"] = raw_path
            if duration := safe_duration(test.get("duration")):
                clean_test["duration"] = duration
            if clean_test["status"] not in PASS_TEST_STATES:
                clean_test["failure"] = classify_launcher_failure(
                    test.get("error"), "The behavioral test did not pass."
                )
            clean_tests.append(clean_test)
        raw_name = re.sub(r"\s+#.*$", "", str(session.get("name") or ""))
        clean_session: dict[str, Any] = {
            "status": safe_status(session.get("status"), SAFE_SESSION_STATES),
            "tests": clean_tests,
        }
        if re.fullmatch(r"skills-[a-z0-9-]+-(?:linux|windows)", raw_name):
            clean_session["name"] = raw_name
        if clean_session["status"] not in PASS_SESSION_STATES:
            clean_session["failure"] = classify_launcher_failure(
                session.get("error"), "The behavioral test session did not complete."
            )
        clean_sessions.append(clean_session)
    clean_stages = []
    raw_stages = jenkins.get("stages") or []
    if not isinstance(raw_stages, list):
        raw_stages = []
    for stage in raw_stages:
        if not isinstance(stage, dict):
            continue
        clean_stage: dict[str, Any] = {
            "status": safe_status(stage.get("status"), SAFE_PIPELINE_STATES)
        }
        name = str(stage.get("name") or "")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _./:-]{0,79}", name):
            clean_stage["name"] = name
        duration_ms = stage.get("duration_ms")
        if isinstance(duration_ms, int) and 0 <= duration_ms <= 86_400_000:
            clean_stage["duration_ms"] = duration_ms
        clean_stages.append(clean_stage)
    clean: dict[str, Any] = {}
    for key in ("status", "pipeline_status", "tests_status"):
        if live.get(key) not in (None, ""):
            clean[key] = safe_status(live.get(key), SAFE_PIPELINE_STATES)
    clean["jenkins"] = {
        "stages": clean_stages,
        "failure_summary": jenkins_failure_summary(live),
    }
    launcher_status = launcher.get("status") or {}
    if not isinstance(launcher_status, dict):
        launcher_status = {}
    clean["launcher"] = {
        "reachable": launcher.get("reachable"),
        "status": {
            key: value
            for key in ("active_sessions", "done_sessions", "total_sessions")
            if isinstance((value := launcher_status.get(key)), int)
            and 0 <= value <= 10_000
        },
        "sessions": clean_sessions,
    }
    clean["reporting"] = {"launcher_available": bool(live.get("launcher_url"))}
    return clean


def collect_live_results(
    client: PortalClient,
    plan: dict,
    *,
    run_id: str,
    pipeline_status: str,
    tests_status: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> tuple[dict, dict]:
    """Wait briefly for the terminal launcher snapshot and normalize it."""
    deadline = time.monotonic() + max(timeout_seconds, 0)
    latest_live: dict = {}
    latest_manifest = build_results_manifest(
        plan,
        latest_live,
        mode="live",
        pipeline_status=pipeline_status,
        tests_status=tests_status,
    )
    last_error = ""
    while True:
        try:
            response = client.request("GET", f"/api/runs/{run_id}/live")
            if not isinstance(response, dict):
                raise RuntimeError("live endpoint returned a non-object response")
            latest_live = merge_live_snapshot(latest_live, response)
            latest_manifest = build_results_manifest(
                plan,
                latest_live,
                mode="live",
                pipeline_status=pipeline_status,
                tests_status=tests_status,
            )
            last_error = ""
            if latest_manifest["ready"]:
                return latest_manifest, latest_live
        except RuntimeError as exc:
            last_error = str(exc)

        if time.monotonic() >= deadline:
            if last_error:
                for item in latest_manifest["items"]:
                    if not item["terminal"]:
                        item["status"] = "error"
                        item["error"] = (
                            f"could not read terminal live results: {last_error}"
                        )
            return latest_manifest, latest_live
        time.sleep(max(poll_seconds, 1))


def pipeline_reporting_completed(pipeline_status: str, items: list[dict]) -> bool:
    """Separate aggregate test failures from pipeline infrastructure failures."""
    if pipeline_status == "passed":
        return True
    if pipeline_status in {"failed", "unstable"}:
        # The shared pipeline becomes non-green when any of its sessions fails.
        # If all requested tests nevertheless passed, the non-green result came
        # from outside the graded sessions and must remain an infrastructure
        # failure. Individual failed sessions are gated by the verdict matrix.
        return any(item.get("status") != "passed" for item in items)
    return False


def summary(
    *,
    mode: str,
    plan_name: str,
    pipeline_status: str = "",
    tests_status: str = "",
    results: dict | None = None,
    portal_url: str = "",
    report_url: str = "",
) -> None:
    path = env("GITHUB_STEP_SUMMARY")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("## OrchestrAI behavioral evals\n\n")
        handle.write(f"- Mode: `{mode}`\n- Plan: `{plan_name}`\n")
        if pipeline_status:
            handle.write(f"- Pipeline: `{pipeline_status or 'unknown'}`\n")
            handle.write(f"- Tests: `{tests_status or 'unknown'}`\n")
        if portal_url:
            portal_target = secret_mask_resistant_markdown_target(portal_url)
            handle.write(
                f"- Pipeline logs: [View in OrchestrAI Portal](<{portal_target}>)\n"
            )
        if report_url:
            handle.write(
                f"- Test results: [View in ReportPortal](<{report_url}>)\n"
            )
        failure_summary = ((results or {}).get("run") or {}).get(
            "failure_summary"
        ) or []
        if failure_summary:
            handle.write("\n### Infrastructure failure\n\n```text\n")
            for line in failure_summary:
                handle.write(safe_failure_message(line) + "\n")
            handle.write("```\n")
        items = (results or {}).get("items") or []
        if items:
            handle.write("\n| Skill | OS | Result |\n")
            handle.write("|---|---|---|\n")
            for item in items:
                status = str(item.get("status") or "unknown")
                icon = (
                    "✅" if status == "passed" else ("🧪" if status == "mock" else "❌")
                )
                handle.write(
                    f"| `{item.get('skill', '')}` | {item.get('os', '')} | "
                    f"{icon} `{status}` |\n"
                )


def main() -> int:
    plan_path = Path(env("PLAN_FILE"))
    if not str(plan_path):
        raise SystemExit("PLAN_FILE is required")
    plan = validate_plan(plan_path)
    plan_name = str(plan.get("name") or "skills-evals")
    mode = env("MODE", "live").lower()
    if mode not in {"live", "mock"}:
        raise SystemExit("MODE must be live or mock")
    results_path = Path(env("RESULTS_FILE", "orchestrai-results.json"))
    live_path = Path(env("LIVE_FILE", "orchestrai-live.json"))
    # Validate the one-session-per-item reporting contract before contacting
    # the Portal or allocating hardware.
    expected_items(plan)

    if mode == "mock":
        results = mock_results_manifest(plan)
        _write_json(results_path, results)
        _write_json(live_path, {"mode": "mock", "launcher": {"sessions": []}})
        log(
            f"MOCK: validated plan '{plan_name}' with {len(plan['sessions'])} session(s)"
        )
        log(
            "MOCK: would login, resolve the Space, create the plan, trigger it, and poll it"
        )
        summary(mode=mode, plan_name=plan_name, results=results)
        return 0

    require_linux_provisioning(plan)

    required = (
        "ORCHESTRAI_PORTAL_URL",
        "ORCHESTRAI_USER",
        "ORCHESTRAI_PASSWORD",
        "ORCHESTRAI_SPACE",
    )
    missing = [name for name in required if not env(name)]
    if missing:
        raise SystemExit("missing required secret(s): " + ", ".join(missing))

    client = PortalClient(env("ORCHESTRAI_PORTAL_URL"))
    run_id = ""
    interrupted = False
    state: dict[str, Any] = {}
    results: dict | None = None
    latest_live: dict = {}

    def stop(_signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        if run_id:
            client.cancel(run_id)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    try:
        log("authenticating to OrchestrAI Portal")
        login = client.request(
            "POST",
            "/api/auth/login",
            {
                "username": env("ORCHESTRAI_USER"),
                "password": env("ORCHESTRAI_PASSWORD"),
            },
        )
        client.token = str(login.get("token") or "")
        if not client.token:
            raise RuntimeError("login response did not contain a token")

        me = client.request("GET", "/api/auth/me")
        matches = [
            space
            for space in me.get("spaces", [])
            if space.get("name") == env("ORCHESTRAI_SPACE")
        ]
        if not matches:
            raise RuntimeError("the configured OrchestrAI space was not available")
        client.space_id = str(matches[0]["id"])
        resolve_stock_os_images(plan, client)

        log(f"creating plan '{plan_name}'")
        created = client.request(
            "POST",
            "/api/plans",
            {
                "name": plan_name,
                "description": str(plan.get("description") or ""),
                "plan_json": {
                    "sessions": plan["sessions"],
                    "run_settings": plan.get("run_settings") or {},
                },
                "tags": plan.get("tags") or [],
            },
        )
        plan_id = str(created.get("id") or "")
        if not plan_id:
            raise RuntimeError("plan creation response did not contain an id")

        # This POST is deliberately not retried: an uncertain retry could launch
        # duplicate hardware sessions. A GitHub rerun is a new intentional run.
        run_request = build_run_request(plan, plan_id)
        triggered = client.request("POST", "/api/runs", run_request)
        run_id = str(triggered.get("id") or "")
        if not run_id:
            raise RuntimeError("run trigger response did not contain an id")
        log("run submitted")

        timeout_seconds = int(env("POLL_TIMEOUT_SEC", "14400"))
        deadline = time.monotonic() + timeout_seconds
        poll_errors = 0
        poll_failure_started: float | None = None
        poll_error_timeout = int(env("POLL_ERROR_TIMEOUT_SEC", "300"))
        if poll_error_timeout <= 0:
            raise ValueError("POLL_ERROR_TIMEOUT_SEC must be a positive integer")
        while time.monotonic() < deadline:
            try:
                state = client.request("GET", f"/api/runs/{run_id}")
                poll_errors = 0
                poll_failure_started = None
            except RuntimeError as exc:
                now = time.monotonic()
                if poll_failure_started is None:
                    poll_failure_started = now
                poll_errors += 1
                if poll_outage_exhausted(
                    poll_failure_started,
                    now=now,
                    timeout_seconds=poll_error_timeout,
                ):
                    raise
                safe_error = classify_controller_exception(exc)
                log(
                    "poll temporarily unavailable "
                    f"(attempt {poll_errors}; retrying for up to "
                    f"{poll_error_timeout} seconds): {safe_error}"
                )
                time.sleep(15)
                continue

            pipeline_status = safe_status(
                state.get("pipeline_status") or state.get("status"),
                SAFE_PIPELINE_STATES,
            )
            tests_status = safe_status(state.get("tests_status"), SAFE_PIPELINE_STATES)
            log(f"status={pipeline_status} tests={tests_status}")
            if pipeline_status in TERMINAL_PIPELINE_STATES:
                break
            time.sleep(15)
        else:
            client.cancel(run_id)
            raise RuntimeError(f"run timed out after {timeout_seconds} seconds")

        results, latest_live = collect_live_results(
            client,
            plan,
            run_id=run_id,
            pipeline_status=pipeline_status,
            tests_status=tests_status,
            timeout_seconds=int(env("RESULT_TIMEOUT_SEC", "90")),
            poll_seconds=int(env("RESULT_POLL_SEC", "5")),
        )
        _write_json(results_path, results)
        _write_json(live_path, sanitized_live_snapshot(latest_live))
        summary(
            mode=mode,
            plan_name=plan_name,
            pipeline_status=pipeline_status,
            tests_status=tests_status,
            results=results,
            portal_url=portal_run_url_for_summary(client.base_url, run_id),
            report_url=reportportal_url_for_summary(latest_live),
        )
        write_output("pipeline_status", pipeline_status)
        write_output("tests_status", tests_status)

        if not results["ready"]:
            log(
                "terminal Portal snapshot did not contain every requested skill/OS verdict"
            )
            return 1
        failed_items = [item for item in results["items"] if item["status"] != "passed"]
        if not pipeline_reporting_completed(pipeline_status, results["items"]):
            log(
                f"pipeline did not complete cleanly; pipeline={pipeline_status}, "
                f"tests={tests_status}; inspect the private controller job for details"
            )
            return 1
        if failed_items:
            log(
                f"captured {len(failed_items)} failed skill/OS verdict(s); "
                "GitHub matrix jobs will report them individually"
            )
        else:
            log("all OrchestrAI behavioral sessions passed")
        return 0
    except KeyboardInterrupt:
        return 130 if interrupted else 1
    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
        safe_error = classify_controller_exception(exc)
        log(f"error: {safe_error}")
        if run_id:
            client.cancel(run_id)
        try:
            results = write_failure_artifacts(
                plan,
                mode=mode,
                error=safe_error,
                results_path=results_path,
                live_path=live_path,
                live=latest_live or state,
            )
        except (OSError, ValueError, TypeError):
            log("warning: could not write failure result artifacts")
        if run_id:
            summary(
                mode=mode,
                plan_name=plan_name,
                pipeline_status="error",
                tests_status="unknown",
                results=results,
                portal_url=portal_run_url_for_summary(client.base_url, run_id),
                report_url=reportportal_url_for_summary(latest_live or state),
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
