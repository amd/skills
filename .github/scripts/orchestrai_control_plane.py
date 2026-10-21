"""OrchestrAI V2 CI API and read-only result transport (Playbooks #1120).

Only the Control Plane starts/cancels runs. Pipeline access reads archived
session results; it never schedules work or controls machine allocation.
Remote response bodies, URLs, IDs and credentials must not reach public logs.
"""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from copy import deepcopy
from pathlib import Path

TERMINAL = {"passed", "failed", "unstable", "error", "cancelled"}
MAX_JSON = 32 * 1024 * 1024
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z")


class ControlPlaneError(RuntimeError):
    """Only locally generated, credential-free messages belong here."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def https_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or any(c.isspace() for c in value)
        or "\\" in value
        or "%" in parsed.path
        or ".." in parsed.path.split("/")
    ):
        raise ControlPlaneError("OrchestrAI V2 requires a valid HTTPS endpoint.")
    return value.rstrip("/")


def read_response(request, *, limit=MAX_JSON, timeout=30):
    # Explicitly prohibit redirects so neither OIDC nor Basic auth can be sent
    # to a different service. No exception from urllib is printed to the log.
    with urllib.request.build_opener(NoRedirect()).open(
        request, timeout=timeout
    ) as response:
        body = response.read(limit + 1)
        if len(body) > limit:
            raise ControlPlaneError(
                "OrchestrAI V2 response exceeded the transport limit."
            )
        return json.loads(body) if body else None


def inline_request(plan: dict) -> dict:
    document = {
        "sessions": deepcopy(plan["sessions"]),
        "run_settings": {**plan.get("run_settings", {}), "executor": "jenkins"},
    }
    # Generic ubuntu/windows aliases are resolved by the Control Plane. Do not
    # call the retired Portal boot-resources API or choose an arbitrary image.
    for session in document["sessions"]:
        session.setdefault("level", session["tests"][0]["path"].split("/", 1)[0])
        variable = "ORCHESTRAI_V2_" + (
            "WINDOWS_IMAGE" if session.get("os_image") == "windows" else "LINUX_IMAGE"
        )
        if os.environ.get(variable):
            session["os_image"] = os.environ[variable]
    images = {s.get("os_image") for s in document["sessions"] if s.get("os_image")}
    if len(images) == 1:
        document["run_settings"]["os_image"] = images.pop()
    payload = {
        "require_capacity": "queue",
        "plan_document": {
            "name": plan["name"],
            "description": plan.get("description", ""),
            "document": document,
        },
        "builds_json": deepcopy(plan.get("builds_json", {})),
    }
    if len(json.dumps(payload).encode()) > 60 * 1024:
        raise ControlPlaneError(
            "OrchestrAI V2 inline plan exceeds the CI API size limit."
        )
    return payload


def idempotency_key(plan: dict) -> str:
    identity = [
        os.environ.get(k, "")
        for k in (
            "GITHUB_REPOSITORY",
            "GITHUB_RUN_ID",
            "GITHUB_RUN_ATTEMPT",
            "GITHUB_JOB",
        )
    ] + [plan["name"]]
    if not all(identity[:3]):
        raise ControlPlaneError("OrchestrAI V2 requires the GitHub run identity.")
    return "skills-" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()


class ControlPlaneClient:
    def __init__(self):
        self.base = https_url(os.environ["ORCHESTRAI_V2_CI_API_URL"])
        if not self.base.endswith("/ci/v1"):
            raise ControlPlaneError("OrchestrAI V2 CI API URL must end in /ci/v1.")
        self.audience = os.environ["ORCHESTRAI_V2_OIDC_AUDIENCE"]
        https_url(
            self.audience
        )  # Validate, but preserve the exact audience (including /).
        self.workspace = os.environ["ORCHESTRAI_V2_WORKSPACE"]
        if not ID.fullmatch(self.workspace):
            raise ControlPlaneError("OrchestrAI V2 workspace must be a workspace slug.")
        self.token = ""
        self.expires = 0.0

    def oidc(self, *, refresh=False):
        if self.token and not refresh and time.time() < self.expires - 60:
            return self.token
        url = os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
        parsed = urllib.parse.urlsplit(url)
        https_url(urllib.parse.urlunsplit(parsed._replace(query="")))
        query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        query = [(k, v) for k, v in query if k != "audience"]
        query.append(("audience", self.audience))
        url = urllib.parse.urlunsplit(
            parsed._replace(query=urllib.parse.urlencode(query))
        )
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": "Bearer "
                + os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"],
                "Accept": "application/json",
            },
        )
        data = read_response(request, limit=1024 * 1024)
        token = data.get("value") if isinstance(data, dict) else None
        if not isinstance(token, str) or not token or any(c.isspace() for c in token):
            raise ControlPlaneError("GitHub did not return an OIDC token.")
        # GitHub's token is used only for authentication. Its claims are not
        # locally trusted to authorize the workflow; that is the CP's job.
        self.token = token
        self.expires = time.time() + 120
        try:
            part = token.split(".")[1]
            claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
            self.expires = float(claims["exp"])
        except (IndexError, ValueError, KeyError, TypeError):
            pass
        print("::add-mask::" + token, flush=True)
        return token

    def request(self, method, path, payload=None, *, key="", attempts=5, timeout=30):
        body = None if payload is None else json.dumps(payload, sort_keys=True).encode()
        refreshed = False
        for attempt in range(attempts):
            try:
                headers = {
                    "Authorization": "Bearer " + self.oidc(),
                    "X-OrchestrAI-Workspace": self.workspace,
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                }
                if key:
                    headers["Idempotency-Key"] = key
                response = read_response(
                    urllib.request.Request(
                        self.base + path, data=body, headers=headers, method=method
                    ),
                    limit=4 * 1024 * 1024,
                    timeout=timeout,
                )
                if response is None and path.endswith("/cancel"):
                    return {}
                if not isinstance(response, dict):
                    raise ValueError("not an object")
                return response
            except urllib.error.HTTPError as exc:
                status = exc.code
                # Examine only the fixed API code. Never propagate the remote
                # message/detail/body, which can include private infrastructure.
                try:
                    code = json.loads(exc.read(65536)).get("error", {}).get("code", "")
                except (ValueError, AttributeError, OSError, http.client.HTTPException):
                    code = ""
                if status == 401 and not refreshed:
                    self.token = ""
                    refreshed = True
                elif status == 403:
                    raise ControlPlaneError(
                        "OrchestrAI V2 denied this workflow. Check workspace CI access "
                        "rules, audience, repository, workflow ref, event and scopes."
                    ) from None
                elif status not in {408, 429, 500, 502, 503, 504} and not (
                    status == 409 and code == "idempotency_in_progress"
                ):
                    raise ControlPlaneError(
                        f"OrchestrAI V2 CI request failed with HTTP {status}; "
                        "inspect the private Control Plane diagnostics."
                    ) from None
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                http.client.HTTPException,
                ssl.SSLError,
                ValueError,
            ):
                pass
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 15))
        raise ControlPlaneError(
            "OrchestrAI V2 CI API was unavailable after bounded retries."
        )

    @staticmethod
    def run(response):
        run = response.get("run")
        if not isinstance(run, dict) or not ID.fullmatch(str(run.get("id", ""))):
            raise ControlPlaneError("OrchestrAI V2 returned an invalid run response.")
        return run

    def start(self, plan, state_path, *, attempts=5):
        payload = inline_request(plan)
        key = idempotency_key(plan)
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        state_path = Path(state_path)
        if state_path.exists():
            saved = json.loads(state_path.read_text())
            if saved != {"key": key, "digest": digest}:
                raise ControlPlaneError(
                    "OrchestrAI V2 refuses to replay changed run inputs."
                )
        else:
            # Private local state only, never an upload-artifact input. A retry
            # after an uncertain POST reuses the exact same key and body.
            fd = os.open(state_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as handle:
                json.dump({"key": key, "digest": digest}, handle)
        return self.run(
            self.request("POST", "/runs", payload, key=key, attempts=attempts)
        )

    def status(self, run_id):
        run = self.run(self.request("GET", f"/runs/{run_id}"))
        if run["id"] != run_id:
            raise ControlPlaneError("OrchestrAI V2 returned a mismatched run identity.")
        return run

    def cancel_and_wait(self, run_id, *, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                run = self.run(
                    self.request("GET", f"/runs/{run_id}", attempts=1, timeout=5)
                )
                if run["id"] == run_id and run.get("status") in TERMINAL:
                    return True
            except (ControlPlaneError, KeyError):
                pass
            try:
                self.request(
                    "POST", f"/runs/{run_id}/cancel", {}, attempts=1, timeout=5
                )
            except ControlPlaneError:
                # A lost cancellation response is not proof it was rejected.
                # The next GET must still run and can confirm termination.
                pass
            time.sleep(3)
        return False


class PipelineResults:
    """Read session evidence from the single CP-selected build, never consoleText.

    The configured base confines Basic auth to the intended pipeline service
    AND folder. Artifact paths come from a narrow allowlist, not arbitrary URLs
    supplied by test output. Encrypted evidence is not decoded or published.
    """

    def __init__(self):
        self.base = https_url(os.environ["ORCHESTRAI_PIPELINE_URL"]) + "/"
        raw = (
            os.environ["ORCHESTRAI_PIPELINE_USER"]
            + ":"
            + os.environ["ORCHESTRAI_PIPELINE_TOKEN"]
        )
        self.auth = "Basic " + base64.b64encode(raw.encode()).decode()

    def read(self, build_url, path):
        build = https_url(build_url) + "/"
        if not build.startswith(self.base) or not re.fullmatch(
            r"(?:job/[A-Za-z0-9_.-]+/)+[0-9]+/", build[len(self.base) :]
        ):
            raise ControlPlaneError(
                "OrchestrAI V2 returned an untrusted pipeline build URL."
            )
        try:
            return read_response(
                urllib.request.Request(
                    build + path,
                    headers={
                        "Authorization": self.auth,
                        "Accept": "application/json",
                    },
                )
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise ControlPlaneError(
                f"OrchestrAI V2 result artifact request failed with HTTP {exc.code}."
            ) from None
        except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError):
            raise ControlPlaneError(
                "OrchestrAI V2 result artifacts were unreadable."
            ) from None

    def snapshot(self, build_url):
        listing = self.read(build_url, "api/json?tree=artifacts[relativePath]")
        if listing is None:
            return {"launcher": {"sessions": []}}
        if not isinstance(listing, dict) or not isinstance(
            listing.get("artifacts"), list
        ):
            raise ControlPlaneError("OrchestrAI V2 result artifact index was invalid.")
        paths = [
            a.get("relativePath", "")
            for a in listing["artifacts"]
            if isinstance(a, dict)
        ]
        paths = [
            p
            for p in paths
            if isinstance(p, str)
            and re.fullmatch(r"\.pipeline/results/[A-Za-z0-9_-]+_results\.json", p)
        ]
        if len(paths) > 256 or len(paths) != len(set(paths)):
            raise ControlPlaneError("OrchestrAI V2 result artifact index was invalid.")
        sessions = []
        total_bytes = 0
        for path in paths:
            session = self.read(build_url, "artifact/" + path)
            if not isinstance(session, dict):
                raise ControlPlaneError(
                    "OrchestrAI V2 session result artifact was invalid."
                )
            total_bytes += len(json.dumps(session).encode())
            if total_bytes > 64 * 1024 * 1024:
                raise ControlPlaneError(
                    "OrchestrAI V2 result artifacts exceeded the transport limit."
                )
            tests = session.get("tests") or session.get("jobs") or []
            if not isinstance(tests, list) or not all(
                isinstance(t, dict) for t in tests
            ):
                raise ControlPlaneError(
                    "OrchestrAI V2 session test results were invalid."
                )
            normalized = []
            for test in tests:
                test = dict(test)
                test["path"] = (
                    test.get("path") or test.get("test_path") or test.get("name")
                )
                test["job_id"] = test.get("job_id") or test.get("id")
                if isinstance(test.get("duration_ms"), (int, float)):
                    test["duration"] = f"{max(0, test['duration_ms']) / 1000:g}s"
                normalized.append(test)
            sessions.append({**session, "tests": normalized})
        return {"launcher": {"sessions": sessions}}
