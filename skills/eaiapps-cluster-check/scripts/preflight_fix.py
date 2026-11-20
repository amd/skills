#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Repair the known faults that stop an end-to-end run before its health check, then prove each repair.

environment_check.py only reads the cluster. This is its counterpart for a shared test cluster: each
fixer detects one fault, repairs it the one way that is known to work, and verifies the result. Every
fixer is idempotent and prints exactly what it changed, so a clean cluster is left untouched and the
log of a repaired one says what was done to it.

What it never does: change anything unless the caller holds the cluster lock (the workflow acquires it
around this step; the lease is checked again before every change, so one lost or taken over mid-run stops
the repairs), write the cluster lock, or edit Argo CD or Git configuration. The two cleanups additionally
touch only what belongs to a CI run GitHub reports as completed (an unknown run counts as running) and,
for AIRM projects, only this cluster's. The SeaweedFS repair is not run-scoped: it acts on the cluster's
own state, and only on positive evidence of a lost leader. A fault it cannot repair safely is reported as
refused or unknown and left for the health check to name.

Fixers, in the order they run:
  seaweedfs-raft      SeaweedFS masters with no agreed raft leader: delete every master pod at once so the
                      StatefulSet recreates them together (a rolling restart replaces one at a time and can
                      stall on a master that never turns Ready), wait for a leader, then restart the filer
                      and S3 gateway that cache the old one.
  finished-run-leftovers
                      Test projects, namespaces and Kaiwo queue entries of a finished run. Order is the
                      point: AIRM projects first (through the AIRM API), then namespaces, then queue
                      entries whose namespaces are gone. Removing namespaces or entries while the project
                      still exists makes the AIRM agent re-add the entries and fails the queue config.
  restore-snapshot    A ClusterForge restore snapshot marked unverified by a finished run blocks every
                      later run. It is deleted only once the service sources and parent sync policy equal
                      it and both services have stayed Synced and Healthy with no operation over repeated
                      samples. Deployment images may differ, unlike in cluster-forge-override.sh: a run
                      changes services only through the Application source, so once that source is the
                      recorded one and Argo reports it synced, the running images are what Git holds for
                      the cluster (another slot's deploy), not something the run left behind.
  longhorn            Shared storage that cannot serve a new volume stalls every model download. Longhorn
                      components that are not Ready are restarted, and a crash-looping share-manager or an
                      orphan volume is handled only when it belongs to a finished run; then a fresh RWX
                      volume must bind and mount in a probe pod, which is removed afterwards.

Exit code: 0 when every fixer ended clean, repaired or refused; 1 when a repair failed or a fixer could
not read the state it needs. --json writes the per-fixer results and every change made.
"""
from __future__ import annotations

import argparse
import atexit
import base64
import http.client
import json
import os
import re
import ssl
import subprocess
import tempfile
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

CLEAN, REPAIRED, REFUSED, FAILED, UNKNOWN = "clean", "repaired", "refused", "failed", "unknown"

SEAWEED_NS = "seaweedfs-instance"
COMPONENT_LABEL = "app.kubernetes.io/component"
SEAWEED_MASTER_SELECTOR = "app.kubernetes.io/component=master"
SEAWEED_CACHE_SELECTOR = "app.kubernetes.io/component in (filer,s3)"
SEAWEED_STATUS_URL = "http://127.0.0.1:9333/cluster/status"

# Every namespace, project and queue a CI suite creates is named e2e-ci<run id>-...
RUN_RESOURCE = re.compile(r"^e2e-ci(\d+)-")
QUEUE_CONFIG = "kaiwo"
RUNNER_NS = "e2e-test-runner"

# The lock every E2E lane holds while it uses the cluster; only read here, never written.
LOCK_NS, LOCK_NAME = "default", "e2e-cluster-lock"
# A lease that ends sooner than this is not trusted for a change: the change and its wait could outlive it.
LOCK_MARGIN_SECONDS = 120
# AIRM labels every namespace it creates for a project with that project's id.
PROJECT_ID_LABEL = "airm.silogen.ai/project-id"

ARGOCD_NS = "argocd"
PARENT_APP = "cluster-forge"
CHILD_APPS = ("airm", "aiwb")
SNAPSHOT_PREFIX = "e2e-cf-override-"
UNVERIFIED_LABEL = "e2e.silogen.ai/restore-unverified"
OWNER_ANNOTATION = "e2e.silogen.ai/override-owner"
PAUSE_ANNOTATION = "argocd.argoproj.io/skip-reconcile"

LONGHORN_NS = "longhorn"
# The probe proves a fresh shared (RWX) volume can be provisioned, attached and mounted, which is what an
# AIM model download needs. Its namespace is not an e2e-ci<run> name, so no run's cleanup ever takes it.
PROBE_NS = "e2e-preflight-storage"
PROBE_IMAGE = "alpine:3.22"
# The env var lets the black-box suite shorten the wait for a volume that never mounts.
PROBE_SECONDS = int(os.environ.get("PREFLIGHT_PROBE_SECONDS") or 180)
LONGHORN_PROVISIONER = "driver.longhorn.io"
SHARE_MANAGER_RESTARTS = 3

# The waits sum to 32 minutes (20 for the first three fixers, 12 for Longhorn's restart and its probe with
# cleanup on both sides), inside the 40-minute workflow step that runs every fixer.
LEADER_WAIT_SECONDS = 300
POD_READY_SECONDS = int(os.environ.get("PREFLIGHT_POD_READY_SECONDS") or 180)  # each of the two restarts
PROJECT_DELETE_SECONDS = 240
NAMESPACE_DELETE_SECONDS = 180
QUEUE_READY_SECONDS = 120
POLL_SECONDS = 10
# A CI runner may not reach the cluster's public ingress at all, so AIRM and Keycloak are addressed by
# their Service names and reached through a kubectl port-forward. A request that does not arrive, or
# that the proxy in front of the service answers as unavailable, gets this many attempts, this far apart.
HTTP_TIMEOUT_SECONDS = 30
HTTP_ATTEMPTS = 3
HTTP_RETRY_SECONDS = 10
PORT_FORWARD_SECONDS = 30
UNAVAILABLE = (502, 503, 504)
AIRM_SERVICE_URL = "http://airm-api.airm.svc.cluster.local"
# A restore is only believed after this many matching, settled samples, as cluster-forge-override.sh does.
SETTLED_SAMPLES = 3
SETTLED_SAMPLE_SECONDS = 10

# Certificate verification is off by default, as in the E2E suites, so self-signed clusters authenticate
# the same way here; VERIFY_SSL=true turns it on.
VERIFY_TLS = os.environ.get("VERIFY_SSL", "").strip().lower() in ("1", "true", "yes", "on")

# Seams the tests replace: the clocks, the sleep, the process runner and the HTTP client.
now: Callable[[], float] = time.monotonic
wall_clock: Callable[[], float] = time.time
sleep: Callable[[float], None] = time.sleep


class ReadError(Exception):
    """The state a decision depends on could not be read."""


@dataclass
class FixResult:
    name: str
    status: str
    detail: str
    changes: list[str] = field(default_factory=list)


def log(message: str) -> None:
    print(message, flush=True)


def run(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s: {' '.join(cmd)}"


def kubectl(*args: str, timeout: int = 60) -> tuple[int, str, str]:
    return run(["kubectl", *args], timeout)


def kubectl_read(*args: str) -> Any:
    rc, out, err = kubectl("--request-timeout=20s", *args, "-o", "json", timeout=40)
    if rc != 0:
        raise ReadError(f"kubectl {' '.join(args)}: {err[:200]}")
    try:
        return json.loads(out)
    except json.JSONDecodeError as error:
        raise ReadError(f"kubectl {' '.join(args)}: unparseable output: {error}") from error


def not_found(error: ReadError) -> bool:
    text = str(error).lower()
    return "command not found" not in text and ("notfound" in text.replace(" ", "") or "not found" in text)


class Lease:
    """The cluster lock as held by this caller. Every change asks it first; it is never written here."""

    owner = ""

    @classmethod
    def holder(cls) -> tuple[str, int]:
        rc, out, err = kubectl("--request-timeout=20s", "get", "configmap", LOCK_NAME, "-n", LOCK_NS, "--ignore-not-found", "-o", "json", timeout=40)
        if rc != 0:
            raise ReadError(f"cannot read the cluster lock {LOCK_NS}/{LOCK_NAME}: {err[:200]}")
        if not out:
            return "", 0
        try:
            data = json.loads(out).get("data") or {}
            return str(data.get("owner") or ""), int(data.get("expires") or 0)
        except (json.JSONDecodeError, ValueError, AttributeError) as error:
            raise ReadError(f"cannot parse the cluster lock {LOCK_NS}/{LOCK_NAME}: {error}") from error

    @classmethod
    def problem(cls) -> str | None:
        """Why a change may not be made now, or None while this caller's lease has time left."""
        if not cls.owner:
            return "no cluster lock owner was given, so nothing may be changed"
        holder, expires = cls.holder()
        if holder != cls.owner:
            return f"the cluster lock is held by {holder or 'nobody'}, not {cls.owner}"
        if expires - wall_clock() < LOCK_MARGIN_SECONDS:
            return f"the cluster lock lease of {cls.owner} ends within {LOCK_MARGIN_SECONDS}s"
        return None

    @classmethod
    def require(cls) -> None:
        reason = cls.problem()
        if reason:
            raise RuntimeError(f"refusing the change: {reason}")


def kubectl_change(change: str, changes: list[str], *args: str, timeout: int = 60) -> None:
    """Run a mutating kubectl command, log it, and record it; a failure raises so the fixer stops."""
    Lease.require()
    log(f"  change: {change}")
    rc, _, err = kubectl(*args, timeout=timeout)
    if rc != 0:
        raise RuntimeError(f"{change} failed: {err[:300]}")
    changes.append(change)


def wait_until(condition: Callable[[], bool], seconds: int) -> bool:
    """Poll until the condition holds or the time is up; a read that fails counts as not yet."""
    deadline = now() + seconds
    while True:
        try:
            if condition():
                return True
        except ReadError as error:
            log(f"  still waiting: {error}")
        remaining = deadline - now()
        if remaining <= 0:
            return False
        sleep(min(POLL_SECONDS, remaining))


def _opener(context: ssl.SSLContext, service_host: str | None) -> urllib.request.OpenerDirector:
    if service_host is None:
        return urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))

    class _ServiceHttps(http.client.HTTPSConnection):
        def connect(self) -> None:
            http.client.HTTPConnection.connect(self)
            self.sock = context.wrap_socket(self.sock, server_hostname=service_host)

    class _ServiceHttpsHandler(urllib.request.HTTPSHandler):
        def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
            return self.do_open(_ServiceHttps, req, context=context)

    # An empty ProxyHandler replaces the environment's: a proxy would reach the loopback of its own host,
    # not this host's forward, and would see the request, sign-in password included.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _ServiceHttpsHandler(context=context))


def http_json(method: str, url: str, headers: dict[str, str] | None = None, form: dict[str, str] | None = None,
              verify: bool = VERIFY_TLS, service_host: str | None = None) -> tuple[int, Any]:
    """(status, parsed body). An unreachable endpoint or a body that is not JSON is a ReadError; an HTTP error
    status is returned, with its body parsed when it can be.

    `service_host` marks a request sent to a port-forward on this host for that Service: it bypasses any
    proxy, and TLS is checked against the Service's name, which is what its certificate carries."""
    data = urllib.parse.urlencode(form).encode() if form is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    context = ssl.create_default_context() if verify else ssl._create_unverified_context()  # noqa: S323 - cluster services may be self-signed
    try:
        with _opener(context, service_host).open(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            status, raw = response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        raw = error.read().decode(errors="replace")
        try:
            return error.code, json.loads(raw or "null")
        except json.JSONDecodeError:
            return error.code, raw
    except (OSError, ValueError, http.client.HTTPException) as error:
        raise ReadError(f"{method} {url}: {error}") from error
    try:
        return status, json.loads(raw or "null")
    except json.JSONDecodeError as error:
        raise ReadError(f"{method} {url}: answered {status} with a body that is not JSON") from error


def service_target(url: str) -> tuple[str, str, int] | None:
    """(namespace, service, port) when the URL names a cluster Service, else None."""
    try:
        parts = urllib.parse.urlsplit(url)
        labels = (parts.hostname or "").split(".")
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or len(labels) < 2 or not all(labels[:2]):
        return None
    # Only an explicit Service suffix: a bare two-label name is as likely a public host (example.com).
    if labels[2:] not in (["svc"], ["svc", "cluster", "local"]):
        return None
    return labels[1], labels[0], port


_forwards: dict[tuple[str, str, int], tuple[subprocess.Popen[str], int]] = {}


def _stop_forward(key: tuple[str, str, int]) -> None:
    entry = _forwards.pop(key, None)
    if entry is None:
        return
    process = entry[0]
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def stop_forwards() -> None:
    for key in list(_forwards):
        _stop_forward(key)


atexit.register(stop_forwards)


def forward(namespace: str, service: str, port: int) -> int:
    """A local port forwarded to the Service by kubectl, started once and reused for the run."""
    key = (namespace, service, port)
    entry = _forwards.get(key)
    if entry and entry[0].poll() is None:
        return entry[1]
    _stop_forward(key)
    try:
        process = subprocess.Popen(["kubectl", "port-forward", "-n", namespace, f"svc/{service}", f":{port}",
                                    "--address", "127.0.0.1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as error:
        raise ReadError(f"cannot start kubectl port-forward to svc/{service}: {error}") from error
    # kubectl announces the port it bound on stdout; a refusal goes to stderr and it exits.
    line: list[str] = []
    reader = threading.Thread(target=lambda: line.append(process.stdout.readline()), daemon=True)  # type: ignore[union-attr]
    reader.start()
    reader.join(PORT_FORWARD_SECONDS)
    match = re.search(r"Forwarding from 127\.0\.0\.1:(\d+)", line[0] if line else "")
    if not match:
        process.kill()
        process.wait(timeout=5)
        reason = (process.stderr.read() if process.stderr else "").strip()[:200]  # type: ignore[union-attr]
        raise ReadError(f"kubectl port-forward to svc/{service} in {namespace} did not start: {reason or 'no port reported'}")
    _forwards[key] = (process, int(match.group(1)))
    return int(match.group(1))


def reachable(url: str) -> str:
    """The URL to use from here: a cluster Service URL goes through a port-forward, any other as is."""
    target = service_target(url)
    if target is None:
        return url
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, f"127.0.0.1:{forward(*target)}", parts.path, parts.query, ""))


def drop_forward(url: str) -> None:
    """Restart the URL's forward on next use: a forward that lost its pod stops carrying requests."""
    target = service_target(url)
    if target is not None:
        _stop_forward(target)


def http_retrying(method: str, url: str, headers: dict[str, str] | None = None, form: dict[str, str] | None = None,
                  before: Callable[[], None] = lambda: None) -> tuple[int, Any]:
    """http_json to a cluster endpoint, retried when the request does not reach it. `before` runs right
    before every attempt, so a retried change re-checks the lease."""
    for attempt in range(1, HTTP_ATTEMPTS + 1):
        try:
            target = reachable(url)
            before()
            service_host = urllib.parse.urlsplit(url).hostname if target != url else None
            code, body = http_json(method, target, headers, form, service_host=service_host)
            if code not in UNAVAILABLE or attempt == HTTP_ATTEMPTS:
                return code, body
            problem = f"answered HTTP {code}"
        except ReadError as error:
            if attempt == HTTP_ATTEMPTS:
                raise ReadError(f"{error} (after {HTTP_ATTEMPTS} attempts)") from error
            problem = str(error)
        # A retried DELETE that had in fact arrived is answered 404 or 409, which the caller accepts.
        log(f"  {method} {url} failed ({problem}); retrying in {HTTP_RETRY_SECONDS}s")
        drop_forward(url)
        sleep(HTTP_RETRY_SECONDS)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- GitHub runs


class RunStates:
    """Whether a CI run is finished, asked of GitHub once per run id. Anything but completed is in use."""

    def __init__(self, repository: str, token: str, api_url: str) -> None:
        self._repository, self._token, self._api_url = repository, token, api_url.rstrip("/")
        self._cache: dict[str, str] = {}

    def status(self, run_id: str) -> str:
        if run_id not in self._cache:
            self._cache[run_id] = self._status(run_id)
            log(f"  run {run_id}: {self._cache[run_id]}")
        return self._cache[run_id]

    def finished(self, run_id: str) -> bool:
        return self.status(run_id) == "completed"

    def unknown(self, run_id: str) -> bool:
        return self.status(run_id).startswith("unknown")

    def _status(self, run_id: str) -> str:
        if not self._repository or not self._token:
            return "unknown (no repository or token to ask GitHub)"
        headers = {"Authorization": f"Bearer {self._token}", "Accept": "application/vnd.github+json"}
        try:
            # The token authorizes reading the answer that decides deletions, so GitHub is always verified,
            # whatever VERIFY_SSL says about the cluster's own services.
            code, body = http_json("GET", f"{self._api_url}/repos/{self._repository}/actions/runs/{run_id}", headers, verify=True)
        except ReadError as error:
            return f"unknown ({error})"
        if code != 200 or not isinstance(body, dict):
            # A run of another repository sharing the cluster is not ours to judge.
            return f"unknown (GitHub answered {code})"
        return str(body.get("status") or "unknown")


# ---------------------------------------------------------------- SeaweedFS raft


def _ready(pod: dict[str, Any]) -> bool:
    return any(c.get("type") == "Ready" and c.get("status") == "True" for c in (pod.get("status") or {}).get("conditions") or [])


def _pods(selector: str) -> list[dict[str, Any]]:
    """Live pods matching the selector; one already being deleted is not counted."""
    items = kubectl_read("get", "pods", "-n", SEAWEED_NS, "-l", selector).get("items") or []
    return [p for p in items if not (p.get("metadata") or {}).get("deletionTimestamp")]


def _seaweed_leader() -> tuple[str, str]:
    """(state, detail): healthy, no-leader or absent. Raises ReadError when the masters cannot be judged.

    A restart needs positive evidence: masters that answer and disagree or report no leader, or too few Ready
    masters for a quorum. A Ready master that cannot be queried (a timeout, a pod being replaced) proves
    nothing, so the raft is unknown rather than lost."""
    try:
        pods = _pods(SEAWEED_MASTER_SELECTOR)
    except ReadError as error:
        if not_found(error):
            return "absent", f"no {SEAWEED_NS} namespace"
        raise
    if not pods:
        return "absent", "no SeaweedFS masters"
    leaders: dict[str, str] = {}
    down: list[str] = []
    for pod in pods:
        name = pod["metadata"]["name"]
        if not _ready(pod):
            down.append(name)
            continue
        rc, out, err = kubectl("exec", "-n", SEAWEED_NS, name, "--", "wget", "-qO-", SEAWEED_STATUS_URL, timeout=30)
        if rc != 0:
            raise ReadError(f"cannot query Ready master {name}: {err[:200] or f'exit {rc}'}")
        try:
            leaders[name] = str((json.loads(out) or {}).get("Leader") or "")
        except (json.JSONDecodeError, AttributeError) as error:
            raise ReadError(f"master {name} answered an unparseable status") from error
    agreed = {leader for leader in leaders.values() if leader}
    quorum = len(pods) // 2 + 1
    detail = f"masters report {json.dumps(leaders, sort_keys=True)}" + (f"; not Ready: {', '.join(down)}" if down else "")
    # One master down is tolerated while a quorum of the others agree; that is a node fault, not a lost raft.
    if len(agreed) == 1 and all(leaders.values()) and len(leaders) >= quorum:
        return "healthy", detail
    return "no-leader", detail


def _replaced(selector: str, old_uids: set[str], expected: dict[str, int]) -> bool:
    """Every component has as many Ready replacements as it had pods before, none of them an old pod."""
    pods = _pods(selector)
    if any(p["metadata"]["uid"] in old_uids for p in pods):
        return False
    ready: dict[str, int] = {}
    for pod in pods:
        if _ready(pod):
            component = (pod["metadata"].get("labels") or {}).get(COMPONENT_LABEL, "")
            ready[component] = ready.get(component, 0) + 1
    return all(ready.get(component, 0) >= count for component, count in expected.items())


def _restart(selector: str, what: str, changes: list[str]) -> bool:
    """Delete every pod the selector matches at once, then wait for Ready replacements of each component."""
    pods = _pods(selector)
    if not pods:
        return True
    expected: dict[str, int] = {}
    for pod in pods:
        component = (pod["metadata"].get("labels") or {}).get(COMPONENT_LABEL, "")
        expected[component] = expected.get(component, 0) + 1
    names = ", ".join(sorted(p["metadata"]["name"] for p in pods))
    kubectl_change(f"delete {what} pods {names} -n {SEAWEED_NS}", changes,
                   "delete", "pods", "-n", SEAWEED_NS, "-l", selector, "--wait=false")
    old_uids = {p["metadata"]["uid"] for p in pods}
    return wait_until(lambda: _replaced(selector, old_uids, expected), POD_READY_SECONDS)


def fix_seaweedfs_raft(dry_run: bool) -> FixResult:
    name = "seaweedfs-raft"
    try:
        state, detail = _seaweed_leader()
    except ReadError as error:
        return FixResult(name, UNKNOWN, str(error))
    if state in ("healthy", "absent"):
        return FixResult(name, CLEAN, detail)
    log(f"[{name}] no agreed raft leader: {detail}")
    if dry_run:
        return FixResult(name, REFUSED, f"dry run; would restart the masters: {detail}")
    changes: list[str] = []
    try:
        # Restarting a single master does not restore the raft, and a rolling restart is one master at a time:
        # it waits for each to turn Ready, which a master without a leader may never do. All masters go at once.
        if not _restart(SEAWEED_MASTER_SELECTOR, "SeaweedFS master", changes):
            return FixResult(name, FAILED, "the masters did not all come back Ready", changes)
        if not wait_until(lambda: _seaweed_leader()[0] == "healthy", LEADER_WAIT_SECONDS):
            return FixResult(name, FAILED, f"no leader after the restart: {_safe_leader_detail()}", changes)
        # The filer and S3 gateway keep the old leader's address; until both restart, S3 calls fail.
        if not _restart(SEAWEED_CACHE_SELECTOR, "SeaweedFS filer and S3", changes):
            return FixResult(name, FAILED, "the filer and S3 gateway did not both come back Ready", changes)
        state, detail = _seaweed_leader()
    except (ReadError, RuntimeError) as error:
        return FixResult(name, FAILED, str(error), changes)
    if state != "healthy":
        return FixResult(name, FAILED, f"leader lost again after the repair: {detail}", changes)
    return FixResult(name, REPAIRED, f"raft leader restored: {detail}", changes)


def _safe_leader_detail() -> str:
    try:
        return _seaweed_leader()[1]
    except ReadError as error:
        return str(error)


# ---------------------------------------------------------------- finished-run leftovers


class Airm:
    """The AIRM project API, as the E2E admin whose credentials the runner namespace holds."""

    def __init__(self, api_url: str, keycloak_url: str) -> None:
        self._api_url = (api_url or AIRM_SERVICE_URL).rstrip("/")
        self._keycloak_url = keycloak_url
        self._headers: dict[str, str] | None = None

    def _secret(self, secret: str, key: str) -> str:
        data = kubectl_read("get", "secret", secret, "-n", RUNNER_NS).get("data") or {}
        try:
            return base64.b64decode(data.get(key, ""), validate=True).decode()
        except ValueError as error:
            raise ReadError(f"secret {RUNNER_NS}/{secret} key {key} is not valid base64 text") from error

    def headers(self) -> dict[str, str]:
        if self._headers is None:
            cfg = kubectl_read("get", "configmap", "keycloak-config", "-n", RUNNER_NS).get("data") or {}
            missing = [k for k in ("KEYCLOAK_CLIENT_ID", "E2E_USERNAME", "KEYCLOAK_REALM") if not cfg.get(k)]
            if missing or not (self._keycloak_url or cfg.get("KEYCLOAK_SERVER_URL")):
                raise ReadError(f"{RUNNER_NS}/keycloak-config lacks {', '.join(missing) or 'KEYCLOAK_SERVER_URL'}")
            form = {
                "grant_type": "password",
                "client_id": cfg["KEYCLOAK_CLIENT_ID"],
                "username": cfg["E2E_USERNAME"],
                "password": self._secret("e2e-user-credentials", "E2E_PASSWORD"),
            }
            client_secret = self._secret("e2e-keycloak-client-secret", "KEYCLOAK_CLIENT_SECRET")
            if client_secret:
                form["client_secret"] = client_secret
            keycloak = (self._keycloak_url or cfg["KEYCLOAK_SERVER_URL"]).rstrip("/")
            code, body = http_retrying("POST", f"{keycloak}/realms/{cfg['KEYCLOAK_REALM']}/protocol/openid-connect/token", form=form)
            token = body.get("access_token") if isinstance(body, dict) else None
            if code != 200 or not isinstance(token, str) or not token.strip():
                raise ReadError(f"the E2E admin could not sign in to Keycloak (HTTP {code})")
            self._headers = {"Authorization": f"Bearer {token}"}
        return self._headers

    def _call(self, method: str, path: str, before: Callable[[], None] = lambda: None) -> tuple[int, Any]:
        # Realm tokens are short-lived and a slow repair can outlast one, so an expired token is renewed once.
        # `before` runs after signing in and immediately ahead of every attempt, so a slow sign-in cannot
        # outlast the lease it checked.
        for attempt in range(2):
            headers = self.headers()
            code, body = http_retrying(method, f"{self._api_url}{path}", headers, before=before)
            if code != 401 or attempt:
                return code, body
            self._headers = None
        raise AssertionError("unreachable")

    def projects(self) -> list[dict[str, Any]]:
        code, body = self._call("GET", "/v1/projects")
        if code != 200 or not isinstance(body, dict) or not isinstance(body.get("data"), list):
            raise ReadError(f"cannot list AIRM projects (HTTP {code})")
        return body["data"]

    def delete_project(self, project_id: str) -> None:
        code, body = self._call("DELETE", f"/v1/projects/{project_id}", Lease.require)
        # 409: a deletion is already under way, which is all this wants.
        if code not in (200, 202, 204, 404, 409):
            raise RuntimeError(f"AIRM refused to delete project {project_id} (HTTP {code}): {str(body)[:200]}")


def _run_id(name: str) -> str | None:
    match = RUN_RESOURCE.match(name)
    return match.group(1) if match else None


def _queue_config() -> dict[str, Any] | None:
    try:
        return kubectl_read("get", "kaiwoqueueconfig", QUEUE_CONFIG)
    except ReadError as error:
        if not_found(error):
            return None
        raise


def _namespace_projects() -> dict[str, str]:
    """Every namespace and the AIRM project id it is labelled with ("" when unlabelled)."""
    return {
        n["metadata"]["name"]: str((n["metadata"].get("labels") or {}).get(PROJECT_ID_LABEL) or "")
        for n in kubectl_read("get", "namespaces").get("items") or []
    }


def _run_namespaces() -> set[str]:
    return {name for name in _namespace_projects() if _run_id(name)}


def project_cluster_id(project: dict[str, Any]) -> str:
    return str((project.get("cluster") or {}).get("id") or project.get("clusterId") or "")


def this_cluster_id(projects: list[dict[str, Any]], labelled: dict[str, str]) -> str:
    """This cluster's AIRM id: the cluster of the projects whose namespaces exist here.

    AIRM lists every cluster's projects, and a name alone does not say which cluster one belongs to, so a
    project is only this cluster's when a namespace here carries its id. Those projects must all name one
    cluster; anything else is refused rather than guessed."""
    by_id = {str(p.get("id")): p for p in projects}
    ids = {project_cluster_id(by_id[pid]) for pid in labelled.values() if pid in by_id}
    if len(ids) != 1 or "" in ids:
        raise ReadError(f"cannot tell this cluster's AIRM id from its labelled namespaces (found {sorted(ids) or 'none'})")
    return ids.pop()


def _all_namespaces() -> set[str]:
    return {n["metadata"]["name"] for n in kubectl_read("get", "namespaces").get("items") or []}


def _queue_entries(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    return list(((config or {}).get("spec") or {}).get("clusterQueues") or [])


def _queue_ready(entry_names_gone: set[str]) -> bool:
    config = _queue_config()
    present = {e.get("name") for e in _queue_entries(config)}
    return ((config or {}).get("status") or {}).get("status") == "READY" and not (present & entry_names_gone)


def fix_finished_run_leftovers(runs: RunStates, airm: Airm, dry_run: bool) -> FixResult:
    name = "finished-run-leftovers"
    changes: list[str] = []
    try:
        namespaces = _run_namespaces()
        config = _queue_config()
        entries = [e for e in _queue_entries(config) if _run_id(str(e.get("name", "")))]
        # A run's project can outlive its namespace and queue entry, so AIRM is asked too. AIRM lists every
        # cluster's projects; without this cluster's identity none of them is deleted.
        listed = airm.projects()
        labels = _namespace_projects()
        run_projects = [p for p in listed if _run_id(str(p.get("name", "")))]
        identity_error: ReadError | None = None
        try:
            cluster_id = this_cluster_id(listed, {n: pid for n, pid in labels.items() if pid}) if run_projects else ""
        except ReadError as error:
            # Without an identity no project is deleted, but a leftover namespace or queue entry with no
            # project of its name is still this cluster's and can go.
            cluster_id, identity_error = "", error
        found = namespaces | {str(e["name"]) for e in entries} | (set() if identity_error else {
            str(p["name"]) for p in run_projects if project_cluster_id(p) == cluster_id})
        if not found:
            if identity_error:
                return FixResult(name, UNKNOWN, f"AIRM holds CI-run projects, but {identity_error}")
            return FixResult(name, CLEAN, "no test projects, namespaces or queue entries of any CI run")
        finished = {r for r in sorted({_run_id(n) or "" for n in found}) if r and runs.finished(r)}
        kept = sorted(n for n in found if _run_id(n) not in finished)
        targets = {n for n in found if _run_id(n) in finished}
        unconfirmed = sorted({r for n in kept if (r := _run_id(n)) and runs.unknown(r)})
        if not targets:
            if unconfirmed:
                return FixResult(name, REFUSED, f"GitHub could not confirm runs {', '.join(unconfirmed)} finished, so their resources were kept: {', '.join(kept)}")
            return FixResult(name, CLEAN, f"only resources of runs still in use: {', '.join(kept)}")
        log(f"[{name}] leftovers of finished runs {', '.join(sorted(finished))}: {', '.join(sorted(targets))}")
        if kept:
            log(f"  leaving resources of runs still in use: {', '.join(kept)}")
        if dry_run:
            return FixResult(name, REFUSED, f"dry run; would remove {', '.join(sorted(targets))}")

        # 1. Projects first: AIRM owns the namespaces and queue entries and re-creates both while a project lives.
        # Only this cluster's projects are touched, and a project whose namespace here is labelled with another
        # id is not the one that namespace belongs to.
        named = [p for p in listed if str(p.get("name", "")) in targets]
        if named and identity_error:
            raise identity_error

        def ours(project: dict[str, Any]) -> bool:
            name = str(project.get("name", ""))
            return (bool(cluster_id) and name in targets and project_cluster_id(project) == cluster_id
                    and labels.get(name, "") in ("", str(project.get("id"))))

        projects = [p for p in listed if ours(p)]
        for project in named:
            if project not in projects:
                log(f"  keeping AIRM project {project.get('name')} ({project.get('id')}): it belongs to cluster {project_cluster_id(project) or 'unknown'}, not this one ({cluster_id})")
        for project in projects:
            if project.get("status") == "Deleting":
                log(f"  project {project['name']} is already being deleted")
                continue
            log(f"  change: delete AIRM project {project['name']} ({project['id']})")
            airm.delete_project(str(project["id"]))
            changes.append(f"delete AIRM project {project['name']} ({project['id']})")
        if projects and not wait_until(lambda: not [p for p in airm.projects() if ours(p)], PROJECT_DELETE_SECONDS):
            remaining = sorted(str(p.get("name")) for p in airm.projects() if ours(p))
            return FixResult(name, FAILED, f"AIRM projects still present, so nothing else was removed: {', '.join(remaining)}", changes)

        # 2. Namespaces the project deletion did not take with it.
        for namespace in sorted(_run_namespaces() & targets):
            kubectl_change(f"delete namespace {namespace}", changes, "delete", "namespace", namespace, "--wait=false")
        if not wait_until(lambda: not (_run_namespaces() & targets), NAMESPACE_DELETE_SECONDS):
            log(f"  namespaces still terminating: {', '.join(sorted(_run_namespaces() & targets))}")

        # 3. Queue entries whose namespaces are all gone; one for a namespace that still exists is left in place.
        existing = _all_namespaces()
        config = _queue_config()
        stale = [
            (index, entry) for index, entry in enumerate(_queue_entries(config))
            if str(entry.get("name", "")) in targets and not set(entry.get("namespaces") or [entry.get("name")]) & existing
        ]
        if stale:
            # Highest index first so earlier removals do not shift later ones; each `test` refuses the patch
            # if the list changed since it was read.
            patch: list[dict[str, Any]] = []
            for index, entry in sorted(stale, key=lambda item: item[0], reverse=True):
                patch += [{"op": "test", "path": f"/spec/clusterQueues/{index}/name", "value": entry["name"]},
                          {"op": "remove", "path": f"/spec/clusterQueues/{index}"}]
            kubectl_change(f"remove KaiwoQueueConfig {QUEUE_CONFIG} entries {', '.join(sorted(str(e['name']) for _, e in stale))}",
                           changes, "patch", "kaiwoqueueconfig", QUEUE_CONFIG, "--type=json", "-p", json.dumps(patch))
        gone = {str(e["name"]) for _, e in stale}
        if config is not None and not wait_until(lambda: _queue_ready(gone), QUEUE_READY_SECONDS):
            return FixResult(name, FAILED, f"KaiwoQueueConfig {QUEUE_CONFIG} is not READY after the cleanup, or an entry came back", changes)
        left = sorted((_run_namespaces() | {str(e.get('name')) for e in _queue_entries(_queue_config())}) & targets)
    except ReadError as error:
        return FixResult(name, FAILED if changes else UNKNOWN, str(error), changes)
    except RuntimeError as error:
        return FixResult(name, FAILED, str(error), changes)
    if left:
        return FixResult(name, FAILED, f"still present after the cleanup: {', '.join(left)}", changes)
    return FixResult(name, REPAIRED, f"removed the leftovers of finished runs {', '.join(sorted(finished))}", changes)


# ---------------------------------------------------------------- unverified restore snapshots


def _snapshot_mismatches(snapshot: dict[str, Any], parent: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(blocking mismatches, image differences). Service sources and the parent's sync policy must equal the
    snapshot exactly; deployment images may differ, because another slot's deploy is not this run's state."""
    data = snapshot.get("data") or {}
    mismatches: list[str] = []
    for app in CHILD_APPS:
        # Source and settledness come from one read, so the status judged is the status of that source.
        application = kubectl_read("get", "application", app, "-n", ARGOCD_NS)
        unsettled = _unsettled(application)
        if unsettled:
            mismatches.append(f"Application {app} is not settled: {unsettled}")
        recorded = data.get(f"{app}-source.json")
        if not recorded:
            mismatches.append(f"the snapshot has no {app} source")
            continue
        live = (application.get("spec") or {}).get("source")
        if json.loads(recorded) != live:
            mismatches.append(f"Application {app} source {json.dumps(live, sort_keys=True)} is not the recorded {recorded}")
    recorded_policy = data.get("parent-automated.json")
    live_policy = ((parent.get("spec") or {}).get("syncPolicy") or {}).get("automated")
    if not recorded_policy or json.loads(recorded_policy) != live_policy:
        mismatches.append(f"{PARENT_APP} automated sync {json.dumps(live_policy)} is not the recorded {recorded_policy}")
    differences: list[str] = []
    for item in json.loads(data.get("deployment-images.json") or "[]"):
        deployment = kubectl_read("get", "deployment", item["deployment"], "-n", item["namespace"])
        images = [c.get("image") for c in deployment["spec"]["template"]["spec"]["containers"] if c.get("name") == item["container"]]
        if images != [item["image"]]:
            differences.append(f"{item['namespace']}/{item['deployment']} runs {images} instead of the recorded {item['image']}")
    return mismatches, differences


def _unsettled(app: dict[str, Any]) -> str:
    """Why an Application is still rolling out, or "" once it is Healthy with no operation running."""
    status = app.get("status") or {}
    phase = (status.get("operationState") or {}).get("phase") or ""
    health = (status.get("health") or {}).get("status") or "unknown"
    if "operation" in app:
        return "a sync operation is pending"
    sync = (status.get("sync") or {}).get("status") or "unknown"
    if sync != "Synced":
        return f"it is {sync}, not Synced"
    # Synced and Healthy describe the source ArgoCD last compared, which lags a source just changed.
    if ((status.get("sync") or {}).get("comparedTo") or {}).get("source") != (app.get("spec") or {}).get("source"):
        return "ArgoCD has not yet compared its current source"
    if phase in ("Running", "Terminating"):
        return f"its sync is {phase}"
    if health != "Healthy":
        return f"its health is {health}"
    return ""


def _stable_match(snapshot: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Mismatches over a sampling window: one instant of equality can be the middle of a rollout."""
    differences: list[str] = []
    for sample in range(SETTLED_SAMPLES):
        parent = kubectl_read("get", "application", PARENT_APP, "-n", ARGOCD_NS)
        try:
            mismatches, differences = _snapshot_mismatches(snapshot, parent)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
            # A snapshot or object this does not understand is not evidence of a restore; keep the snapshot.
            raise ReadError(f"cannot compare {snapshot.get('metadata', {}).get('name')} with the cluster: {error!r}") from error
        if mismatches:
            return mismatches, differences
        if sample < SETTLED_SAMPLES - 1:
            sleep(SETTLED_SAMPLE_SECONDS)
    return [], differences


def fix_restore_snapshots(runs: RunStates, dry_run: bool) -> FixResult:
    name = "restore-snapshot"
    changes: list[str] = []
    try:
        snapshots = kubectl_read("get", "configmaps", "-n", ARGOCD_NS, "-l", f"{UNVERIFIED_LABEL}=true").get("items") or []
        if not snapshots:
            return FixResult(name, CLEAN, "no unverified ClusterForge restore snapshot")
        parent = kubectl_read("get", "application", PARENT_APP, "-n", ARGOCD_NS)
        annotations = (parent.get("metadata") or {}).get("annotations") or {}
        if annotations.get(OWNER_ANNOTATION):
            return FixResult(name, REFUSED, f"{PARENT_APP} is leased by {annotations[OWNER_ANNOTATION]}; its override is in use")
        if PAUSE_ANNOTATION in annotations:
            return FixResult(name, REFUSED, f"{PARENT_APP} is paused without an owner; its restore has not happened")
        refused: list[str] = []
        for snapshot in snapshots:
            cm = snapshot["metadata"]["name"]
            owner = cm.removeprefix(SNAPSHOT_PREFIX)
            if not runs.finished(owner.split("-")[0]):
                refused.append(f"{cm}: its run is not finished")
                continue
            mismatches, differences = _stable_match(snapshot)
            if mismatches:
                refused.append(f"{cm}: {'; '.join(mismatches)}")
                continue
            log(f"[{name}] {cm}: service sources and {PARENT_APP} sync policy equal the snapshot and the services stayed settled over {SETTLED_SAMPLES} samples")
            for difference in differences:
                log(f"  image differs (another slot's deploy, not this restore): {difference}")
            if dry_run:
                refused.append(f"{cm}: dry run; would delete it")
                continue
            kubectl_change(f"delete configmap {cm} -n {ARGOCD_NS}", changes, "delete", "configmap", cm, "-n", ARGOCD_NS)
            rc, out, err = kubectl("--request-timeout=20s", "get", "configmap", cm, "-n", ARGOCD_NS, "--ignore-not-found", "-o", "name", timeout=40)
            if rc != 0 or out:
                return FixResult(name, FAILED, f"{cm} is still present after its deletion{': ' + err[:200] if err else ''}", changes)
    except ReadError as error:
        return FixResult(name, FAILED if changes else UNKNOWN, str(error), changes)
    except RuntimeError as error:
        return FixResult(name, FAILED, str(error), changes)
    if refused:
        return FixResult(name, REFUSED, "; ".join(refused), changes)
    return FixResult(name, REPAIRED, f"deleted {len(changes)} verified snapshot(s)", changes)


# ---------------------------------------------------------------- entry point

FIXERS = ("seaweedfs-raft", "finished-run-leftovers", "restore-snapshot", "longhorn")


# ---------------------------------------------------------------- Longhorn shared storage


def _pod_name(pod: dict[str, Any]) -> str:
    return str((pod.get("metadata") or {}).get("name", "?"))


def _labels(pod: dict[str, Any]) -> dict[str, str]:
    return (pod.get("metadata") or {}).get("labels") or {}


def _restart_count(pod: dict[str, Any]) -> int:
    return sum(int(c.get("restartCount") or 0) for c in (pod.get("status") or {}).get("containerStatuses") or [])


def _longhorn_component(pod: dict[str, Any]) -> str | None:
    labels = _labels(pod)
    if labels.get("app") in ("longhorn-manager", "longhorn-csi-plugin"):
        return labels["app"]
    if labels.get("longhorn.io/component") == "instance-manager":
        return "instance-manager"
    return None


def _default_longhorn_class() -> str | None:
    """The default StorageClass when Longhorn provisions it; the probe uses what an AIM download uses."""
    for sc in kubectl_read("get", "storageclass").get("items") or []:
        annotations = (sc.get("metadata") or {}).get("annotations") or {}
        if annotations.get("storageclass.kubernetes.io/is-default-class") == "true":
            return str(sc["metadata"]["name"]) if sc.get("provisioner") == LONGHORN_PROVISIONER else None
    return None


def _probe_manifest(storage_class: str) -> str:
    claim = {"apiVersion": "v1", "kind": "PersistentVolumeClaim", "metadata": {"name": "rwx-probe", "namespace": PROBE_NS},
             "spec": {"accessModes": ["ReadWriteMany"], "storageClassName": storage_class, "resources": {"requests": {"storage": "1Gi"}}}}
    pod = {"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "rwx-probe", "namespace": PROBE_NS},
           "spec": {"restartPolicy": "Never", "terminationGracePeriodSeconds": 0,
                    "securityContext": {"runAsNonRoot": True, "runAsUser": 65534, "seccompProfile": {"type": "RuntimeDefault"}},
                    "containers": [{"name": "probe", "image": PROBE_IMAGE, "command": ["ls", "/data"], "volumeMounts": [{"name": "data", "mountPath": "/data"}],
                                    "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
                                    "resources": {"requests": {"cpu": "10m", "memory": "16Mi"}, "limits": {"memory": "32Mi"}}}],
                    "volumes": [{"name": "data", "persistentVolumeClaim": {"claimName": "rwx-probe"}}]}}
    namespace = {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": PROBE_NS}}
    return json.dumps({"apiVersion": "v1", "kind": "List", "items": [namespace, claim, pod]})


def _remove_probe() -> bool:
    """Delete the probe namespace and wait until it is gone, so the next probe starts from nothing.

    Deleting it is a change like any other, so it is made only while the lease is still ours."""
    if not _namespace_exists(PROBE_NS):
        return True
    Lease.require()
    rc, _, err = kubectl("delete", "namespace", PROBE_NS, "--ignore-not-found", "--wait=false")
    if rc != 0:
        raise ReadError(f"cannot remove the storage probe namespace {PROBE_NS}: {err[:200]}")
    return wait_until(lambda: not _namespace_exists(PROBE_NS), NAMESPACE_DELETE_SECONDS)


def _namespace_exists(name: str) -> bool:
    try:
        kubectl_read("get", "namespace", name)
    except ReadError as error:
        if not_found(error):
            return False
        raise
    return True


def _probe_rwx(storage_class: str) -> str:
    """"" when a fresh RWX volume binds and a pod mounts it within PROBE_SECONDS and the probe is then
    removed again, else why not."""
    if not _remove_probe():
        return f"a previous storage probe namespace {PROBE_NS} is still terminating"
    Lease.require()
    try:
        problem = _mount_probe(storage_class)
    finally:
        # Also after a failed apply: part of the probe may exist. A lease lost meanwhile raises here.
        removed = _remove_probe()
    if not removed:
        cleanup = f"the storage probe namespace {PROBE_NS} was not removed within {NAMESPACE_DELETE_SECONDS}s"
        return f"{problem}; {cleanup}" if problem else cleanup
    _reclaim_probe_volumes()
    return problem


def _reclaim_probe_volumes() -> None:
    """Delete a PV the probe's claim left behind, and its Longhorn volume.

    A class with reclaimPolicy Retain keeps the PV once the probe namespace is gone, and no later
    cleanup would collect it, so every nightly would leave another volume. Only PVs whose claim was
    in the probe namespace are touched; the StorageClass is not changed. The Longhorn volume goes first
    and the PV last: the PV is what names the volume, so a cleanup stopped in between leaves it for the
    next run to find and finish."""
    for pv in kubectl_read("get", "pv").get("items") or []:
        if ((pv.get("spec") or {}).get("claimRef") or {}).get("namespace") != PROBE_NS:
            continue
        name = str((pv.get("metadata") or {}).get("name"))
        handle = ((pv.get("spec") or {}).get("csi") or {}).get("volumeHandle")
        if handle:
            Lease.require()
            rc, _, err = kubectl("delete", "volumes.longhorn.io", str(handle), "-n", LONGHORN_NS, "--ignore-not-found", "--wait=false")
            if rc != 0:
                raise ReadError(f"cannot remove the storage probe's Longhorn volume {handle}: {err[:200]}")
        Lease.require()
        rc, _, err = kubectl("delete", "pv", name, "--ignore-not-found", "--wait=false")
        if rc != 0:
            raise ReadError(f"cannot remove the storage probe's volume {name}: {err[:200]}")


def _mount_probe(storage_class: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        handle.write(_probe_manifest(storage_class))
    try:
        rc, _, err = kubectl("apply", "-f", handle.name)
    finally:
        os.unlink(handle.name)
    if rc != 0:
        return f"cannot create the storage probe: {err[:200]}"

    def finished() -> bool:
        phase = ((kubectl_read("get", "pod", "rwx-probe", "-n", PROBE_NS).get("status") or {}).get("phase")) or ""
        if phase == "Failed":
            raise ProbeFailed("the probe pod mounted the volume but its command failed")
        return phase == "Succeeded"
    try:
        if wait_until(finished, PROBE_SECONDS):
            return ""
    except ProbeFailed as error:
        return str(error)
    events = kubectl_read("get", "events", "-n", PROBE_NS).get("items") or []
    warnings = sorted({f"{e.get('reason')}: {str(e.get('message', ''))[:160]}" for e in events if e.get("type") == "Warning"})
    return f"a fresh {storage_class} RWX volume was not mounted within {PROBE_SECONDS}s" + (f" ({'; '.join(warnings[-3:])})" if warnings else "")


class ProbeFailed(Exception):
    """The probe pod ran but its command failed."""


def fix_longhorn(runs: RunStates, dry_run: bool) -> FixResult:
    """Longhorn components that are not Ready, share-managers and volumes left by finished runs, then a probe.

    Volumes are only touched when they belong to a finished e2e-ci run; components are restarted by
    deleting their pod, which their DaemonSet or the Longhorn manager recreates."""
    name = "longhorn"
    changes: list[str] = []
    try:
        if not _namespace_exists(LONGHORN_NS):
            return FixResult(name, CLEAN, "Longhorn is not installed")
        pods = [p for p in kubectl_read("get", "pods", "-n", LONGHORN_NS).get("items") or []
                if not (p.get("metadata") or {}).get("deletionTimestamp")]
        volumes = kubectl_read("get", "volumes.longhorn.io", "-n", LONGHORN_NS).get("items") or []
        # A PV references its Longhorn volume by CSI volumeHandle; its own name differs for a static or
        # imported PV. Either reference keeps the volume, whatever the PV's phase.
        bound: set[str] = set()
        for pv in kubectl_read("get", "pv").get("items") or []:
            bound.add(str((pv.get("metadata") or {}).get("name")))
            handle = ((pv.get("spec") or {}).get("csi") or {}).get("volumeHandle")
            if handle:
                bound.add(str(handle))
        volume_ns = {str(v["metadata"]["name"]): str(((v.get("status") or {}).get("kubernetesStatus") or {}).get("namespace") or "") for v in volumes}

        def finished_run_volume(volume: str) -> bool:
            run = _run_id(volume_ns.get(volume, ""))
            return run is not None and runs.finished(run)

        not_ready = sorted((_longhorn_component(p) or "", _pod_name(p)) for p in pods if _longhorn_component(p) and not _ready(p))
        looping = sorted(str(_labels(p).get("longhorn.io/share-manager")) for p in pods
                         if _labels(p).get("longhorn.io/component") == "share-manager" and not _ready(p) and _restart_count(p) >= SHARE_MANAGER_RESTARTS)
        stuck_shares = [v for v in looping if finished_run_volume(v)]
        orphans = sorted(v for v in volume_ns if v not in bound and finished_run_volume(v) and not _namespace_exists(volume_ns[v]))
        left = [v for v in looping if v not in stuck_shares]
        if dry_run:
            found = [f"not Ready: {n}" for _, n in not_ready] + [f"crash-looping share-manager {v}" for v in stuck_shares] + [f"orphan volume {v}" for v in orphans]
            skipped = f"; left alone (not a finished run's volume): crash-looping share-manager(s) {', '.join(left)}" if left else ""
            preview = f"dry run; would repair: {', '.join(found)}" if found else "dry run; would probe a fresh RWX volume"
            return FixResult(name, REFUSED, preview + skipped)

        for volume in orphans:
            kubectl_change(f"delete Longhorn volume {volume} of finished run namespace {volume_ns[volume]}", changes,
                           "delete", "volumes.longhorn.io", volume, "-n", LONGHORN_NS, "--wait=false")
        for volume in stuck_shares:
            kubectl_change(f"restart share-manager of {volume} ({volume_ns[volume]}, a finished run)", changes,
                           "delete", "pod", f"share-manager-{volume}", "-n", LONGHORN_NS, "--wait=false")
        # The footprint before the restart: a Ready sibling on another node must not stand in for a
        # restarted pod whose replacement has not come up yet.
        footprint: dict[str, int] = {}
        for pod in pods:
            component = _longhorn_component(pod)
            if component:
                footprint[component] = footprint.get(component, 0) + 1
        # By UID, not name: the Longhorn manager recreates an instance-manager pod under the same name.
        restarted = {str((p.get("metadata") or {}).get("uid")) for p in pods if _longhorn_component(p) and not _ready(p)}
        for component, pod in not_ready:
            kubectl_change(f"restart {component} pod {pod}", changes, "delete", "pod", pod, "-n", LONGHORN_NS, "--wait=false")
        if not_ready:
            def components_ready() -> bool:
                live = [p for p in kubectl_read("get", "pods", "-n", LONGHORN_NS).get("items") or []
                        if _longhorn_component(p) and not (p.get("metadata") or {}).get("deletionTimestamp")]
                if any(str((p.get("metadata") or {}).get("uid")) in restarted for p in live):
                    return False
                ready: dict[str, int] = {}
                for p in live:
                    if _ready(p):
                        ready[str(_longhorn_component(p))] = ready.get(str(_longhorn_component(p)), 0) + 1
                return all(ready.get(c, 0) >= count for c, count in footprint.items())
            if not wait_until(components_ready, POD_READY_SECONDS):
                return FixResult(name, FAILED, f"Longhorn components still not Ready after a restart: {', '.join(n for _, n in not_ready)}", changes)

        storage_class = _default_longhorn_class()
        if storage_class is None:
            detail = "the default StorageClass is not Longhorn, so no RWX probe was made"
            return FixResult(name, REPAIRED if changes else CLEAN, detail, changes)
        problem = _probe_rwx(storage_class)
    except ReadError as error:
        return FixResult(name, FAILED if changes else UNKNOWN, str(error), changes)
    except RuntimeError as error:
        return FixResult(name, FAILED, str(error), changes)
    note = f"; crash-looping share-manager(s) of volumes outside finished runs left alone: {', '.join(left)}" if left else ""
    if problem:
        return FixResult(name, FAILED, f"shared storage cannot serve a new volume: {problem}{note}", changes)
    if changes:
        return FixResult(name, REPAIRED, f"repaired Longhorn; a fresh {storage_class} RWX volume then mounted{note}", changes)
    return FixResult(name, CLEAN, f"a fresh {storage_class} RWX volume bound and mounted{note}")


def run_fixers(selected: list[str], dry_run: bool, runs: RunStates, airm: Airm, lock_owner: str = "",
               report: Callable[[list[FixResult]], None] = lambda results: None) -> list[FixResult]:
    """Run the selected fixers in order. `report` is called after each one, so a later crash or a killed step
    still leaves a record of every change already made."""
    results: list[FixResult] = []
    Lease.owner = lock_owner
    if not dry_run:
        try:
            reason = Lease.problem()
        except ReadError as error:
            log(f"refusing every repair: {error}")
            return [FixResult(f, UNKNOWN, str(error)) for f in FIXERS if f in selected]
        if reason:
            log(f"refusing every repair: {reason}")
            return [FixResult(f, REFUSED, reason) for f in FIXERS if f in selected]
    for fixer in FIXERS:
        if fixer not in selected:
            continue
        log(f"== {fixer}")
        try:
            if fixer == "seaweedfs-raft":
                result = fix_seaweedfs_raft(dry_run)
            elif fixer == "finished-run-leftovers":
                result = fix_finished_run_leftovers(runs, airm, dry_run)
            elif fixer == "restore-snapshot":
                result = fix_restore_snapshots(runs, dry_run)
            else:
                result = fix_longhorn(runs, dry_run)
        except Exception as error:  # noqa: BLE001 - one fixer's defect must not stop the others or the report
            result = FixResult(fixer, FAILED, f"unexpected error: {error!r}")
        log(f"[{result.name}] {result.status}: {result.detail}")
        results.append(result)
        report(results)
    return results


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="detect and report only; change nothing")
    parser.add_argument("--only", default=",".join(FIXERS), help=f"comma-separated fixers to run (default: all of {', '.join(FIXERS)})")
    parser.add_argument("--json", dest="json_out", help="write the per-fixer results and changes to this file")
    parser.add_argument("--lock-owner", default="", help="the owner this caller holds the cluster lock as; required for any change")
    args = parser.parse_args(argv)
    selected = [f.strip() for f in args.only.split(",") if f.strip()]
    unknown = sorted(set(selected) - set(FIXERS))
    if unknown:
        parser.error(f"unknown fixer(s): {', '.join(unknown)}")
    runs = RunStates(os.environ.get("GITHUB_REPOSITORY", ""), os.environ.get("GITHUB_TOKEN", ""),
                     os.environ.get("GITHUB_API_URL", "https://api.github.com"))
    airm = Airm(os.environ.get("AIRM_API_URL", ""), os.environ.get("KEYCLOAK_SERVER_URL", ""))

    def report(results: list[FixResult]) -> None:
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as handle:
                json.dump({"dryRun": args.dry_run, "fixers": [asdict(r) for r in results]}, handle, indent=2)

    report([])
    results = run_fixers(selected, args.dry_run, runs, airm, args.lock_owner, report)
    report(results)
    return 1 if any(r.status in (FAILED, UNKNOWN) for r in results) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
