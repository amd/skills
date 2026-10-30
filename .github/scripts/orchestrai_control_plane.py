"""OrchestrAI V2 CI API transport using the GitHub job's OIDC identity.

All service requests use the Control Plane CI API; no executor credentials
or executor URLs are consumed. Detailed evidence access is an OAI API gap.
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
RESULT_IDENTITY_GAP = (
    "OrchestrAI CI results cannot distinguish multiple skills using the same "
    "test adapter. No machines were requested. OAI must provide authenticated "
    "per-session results before this selection can run."
)

# These are local messages keyed by documented API enums, not remote prose.
# In particular, never echo error.message, request IDs, rule/project names,
# rejected claim values, or an unrecognized error.code into public Actions logs.
AUTH_DIAGNOSTICS = {
    "no_matching_rule": "No enabled workspace CI rule matches the job's identity.",
    "scope_missing": "The matched CI rule lacks a required API permission.",
    "project_not_allowed": "The CI rule does not allow the requested or default project.",
    "rule_turned_off": "The matching CI rule is disabled.",
    "rule_needs_owner": "A current workspace owner must review and save the CI rule again.",
    "rule_token_ended": "The credential used to save the CI rule has expired or been revoked.",
    "rule_grants_nothing": "The CI rule grants no currently allowed permissions.",
    "workspace_missing": "The configured CI workspace is no longer available.",
    "audience_retired": "The CI rule's audience is no longer accepted by this deployment.",
    "ci_token_forbidden": "The CI API requires the GitHub job OIDC token, not an exchanged token.",
    "run_not_started_by_rule": "This CI rule cannot access that run.",
    "portal_refused": "The Portal refused the Control Plane operation after CI authentication.",
}
REQUIRED_SCOPES = frozenset(
    {
        "plans:read",
        "plans:write",
        "plans:publish",
        "runs:start",
        "runs:read",
        "runs:cancel",
        "results:read",
    }
)
GENERIC_AUTH_DIAGNOSTIC = (
    "OrchestrAI V2 denied this workflow. Check workspace CI access "
    "rules, audience, repository, workflow ref, event and scopes."
)


def authorization_diagnostic(code, scope=""):
    if not isinstance(code, str) or code not in AUTH_DIAGNOSTICS:
        return GENERIC_AUTH_DIAGNOSTIC
    message = f"OrchestrAI V2 authorization denied ({code}). {AUTH_DIAGNOSTICS[code]}"
    if code == "scope_missing" and isinstance(scope, str) and scope in REQUIRED_SCOPES:
        message += f" Required scope: {scope}."
    return message


SAFE_AUTH_DIAGNOSTICS = frozenset(
    [GENERIC_AUTH_DIAGNOSTIC]
    + [authorization_diagnostic(code) for code in AUTH_DIAGNOSTICS]
    + [authorization_diagnostic("scope_missing", scope) for scope in REQUIRED_SCOPES]
)
REQUEST_DIAGNOSTICS = {
    (400, "invalid_request"): (
        "OrchestrAI V2 CI request failed with HTTP 400 (invalid_request). "
        "Check the CI API plan request schema."
    ),
    (400, "portal_refused"): (
        "OrchestrAI V2 CI request failed with HTTP 400 (portal_refused). "
        "The supplied plan was rejected; check private plan and catalog validation."
    ),
}


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
    # Explicitly prohibit redirects so OIDC authentication cannot be sent
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
        "run_settings": deepcopy(plan.get("run_settings", {})),
    }
    # Generic ubuntu/windows aliases are resolved by the Control Plane. Do not
    # call the retired Portal boot-resources API or choose an arbitrary image.
    for session in document["sessions"]:
        # V2 plan drafts require string test references, as Playbooks submits.
        # Keep the local per-test envelope for result attribution, but move its
        # variables onto the one-test session at the API boundary. Never merge
        # several skills' variables onto a shared adapter invocation.
        tests = session.get("tests")
        if (
            not isinstance(tests, list)
            or len(tests) != 1
            or not isinstance(tests[0], dict)
            or not isinstance(tests[0].get("path"), str)
            or not tests[0]["path"]
            or not isinstance(tests[0].get("variables", {}), dict)
            or not isinstance(session.get("variables", {}), dict)
        ):
            raise ControlPlaneError(
                "OrchestrAI V2 requires one test adapter with variables per skill session."
            )
        test = tests[0]
        variables = session.get("variables", {})
        for name, value in test.get("variables", {}).items():
            if name in variables and variables[name] != value:
                raise ControlPlaneError(
                    "OrchestrAI V2 refuses conflicting session and test variables."
                )
            variables[name] = value
        session["tests"] = [test["path"]]
        session["variables"] = variables
        session.setdefault("level", test["path"].split("/", 1)[0])
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


def require_result_identity(plan: dict) -> None:
    """The current /tests compatibility API identifies tests, not sessions.

    Never fan out one adapter outcome to several skills or zip rows to the
    plan order. Stop before admission when that public contract is ambiguous.
    OAI's detailed execution/read-access contract must replace this restriction.
    """
    paths = [test["path"] for session in plan["sessions"] for test in session["tests"]]
    if len(paths) != len(set(paths)):
        raise ControlPlaneError(RESULT_IDENTITY_GAP)


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
                # Only documented enum values may select a local diagnostic.
                # Malformed or private response fields are never published.
                code, scope = "", ""
                try:
                    envelope = json.loads(exc.read(65536))
                    error = (
                        envelope.get("error") if isinstance(envelope, dict) else None
                    )
                    if isinstance(error, dict):
                        code = error.get("code", "")
                        scope = error.get("scope", "")
                except (ValueError, AttributeError, OSError, http.client.HTTPException):
                    pass
                if status == 401 and not refreshed:
                    self.token = ""
                    refreshed = True
                elif status == 403:
                    raise ControlPlaneError(
                        authorization_diagnostic(code, scope)
                    ) from None
                elif status not in {408, 429, 500, 502, 503, 504} and not (
                    status == 409 and code == "idempotency_in_progress"
                ):
                    raise ControlPlaneError(
                        REQUEST_DIAGNOSTICS.get((status, code), "")
                        if isinstance(code, str)
                        and (status, code) in REQUEST_DIAGNOSTICS
                        else f"OrchestrAI V2 CI request failed with HTTP {status}; "
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

    def snapshot(self, run_id, plan):
        """Read the existing CI API's status-only results without inferring IDs.

        A unique planned test path can be attributed to its one skill. Multiple
        executions of that path (including ambiguous retries) are refused. The
        legacy normalizer envelope here is not a launcher response or proof of
        session/cleanup completion; the controller separately waits for the run.
        Unknown fields, raw documents and storage locations are not consumed.
        """
        require_result_identity(plan)
        if not isinstance(run_id, str) or not ID.fullmatch(run_id):
            raise ControlPlaneError("OrchestrAI V2 returned an invalid run identity.")
        response = self.request("GET", f"/runs/{run_id}/tests")
        tests = response.get("tests")
        if (
            response.get("api_version") != "ci/v1"
            or response.get("run_id") != run_id
            or not isinstance(tests, list)
            or not all(isinstance(test, dict) for test in tests)
            or type(response.get("total")) is not int
            or response["total"] != len(tests)
            or response.get("next_cursor")
            or len(tests) > 1000
        ):
            raise ControlPlaneError(
                "OrchestrAI V2 returned invalid or incomplete test results."
            )
        seen = set()
        for test in tests:
            execution_id = test.get("execution_id")
            if (
                not isinstance(execution_id, str)
                or not 1 <= len(execution_id) <= 512
                or any(c.isspace() or ord(c) < 32 for c in execution_id)
                or execution_id in seen
            ):
                raise ControlPlaneError(
                    "OrchestrAI V2 returned missing or duplicate execution identities."
                )
            seen.add(execution_id)
        sessions = []
        for session in plan["sessions"]:
            path = session["tests"][0]["path"]
            matches = [
                test for test in tests if (test.get("path") or test.get("name")) == path
            ]
            if not matches:
                continue  # Publication can lag run completion; the caller retries.
            if len(matches) != 1:
                raise ControlPlaneError(
                    "OrchestrAI V2 returned ambiguous results for a test adapter."
                )
            test = matches[0]
            outcome = test.get("outcome")
            # The compatibility API does not expose attempt/superseded metadata.
            # Never select the last row or use the parent's green status.
            status = (
                outcome
                if isinstance(outcome, str)
                and outcome
                in {"passed", "failed", "error", "skipped", "cancelled", "aborted"}
                else "error"
            )
            error = test.get("error")
            if error is not None and not isinstance(error, str):
                raise ControlPlaneError("OrchestrAI V2 returned an invalid test error.")
            error = error or ""
            if status != outcome:
                error = "OrchestrAI did not publish a recognized terminal test outcome."
            exit_code = test.get("exit_code")
            if exit_code is not None and type(exit_code) is not int:
                raise ControlPlaneError(
                    "OrchestrAI V2 returned an invalid test exit code."
                )
            if status == "passed" and (exit_code not in (None, 0) or error):
                status, error = (
                    "error",
                    "OrchestrAI published a passing outcome with failure evidence.",
                )
            sessions.append(
                {
                    "name": session["name"],
                    "status": "completed",
                    "tests": [{"path": path, "status": status, "error": error}],
                }
            )
        return {"launcher": {"sessions": sessions}}

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
