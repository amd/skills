# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Primitives and cross-cutting helpers every check builds on.

The single side-effecting primitive is run(); the socket and HTTP stack sits here with it so
that stubbing one module stubs every caller, which is what the test suite relies on.
"""
from __future__ import annotations

import contextlib
import datetime
import http.client
import io
import ipaddress
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any



OK, WARN, INCONCLUSIVE = "ok", "warn", "inconclusive"

@dataclass
class CheckResult:
    name: str
    status: str
    verdict: str
    recommendation: str = ""
    # False in --mode e2e for a check that is shown but cannot decide whether the suites can run.
    gating: bool = True

def run(cmd: list[str], timeout: int = 30, stdin: str | None = None) -> tuple[int, str, str]:
    """Run a command read-only and return (returncode, stdout, stderr), all stripped.

    A missing binary is returncode 127 and a timeout is 124 — both are surfaced to the caller
    as an inconclusive read rather than being mistaken for a clean result.
    """
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, input=stdin)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s"

# Several checks read the same expensive cluster state (every running pod, a UI's logs); one run
# fetches each once. run_all_checks empties it so a run never sees an earlier run's state.
_RUN_CACHE: dict[tuple[Any, ...], Any] = {}

def _once_per_run(key: tuple[Any, ...], fetch: Any) -> Any:
    if key not in _RUN_CACHE:
        _RUN_CACHE[key] = fetch()
    return _RUN_CACHE[key]

def _running_pods() -> tuple[bool, Any, str]:
    """kubectl_json of every Running pod in the cluster, fetched once per run."""
    return _once_per_run(("running-pods",), lambda: kubectl_json(["get", "pods", "-A", "--field-selector=status.phase=Running"]))

def kubectl_json(args: list[str], timeout: int = 30) -> tuple[bool, Any, str]:
    """Run `kubectl <args> -o json` and parse it. Returns (read_ok, parsed_or_None, error).

    read_ok is False when the command failed OR the output did not parse — either way the state
    could not be read, which every caller turns into INCONCLUSIVE (never a pass).
    """
    rc, out, err = run(["kubectl", *args, "-o", "json"], timeout)
    if rc != 0:
        return False, None, err[:160]
    try:
        return True, json.loads(out), ""
    except json.JSONDecodeError as e:
        return False, None, f"unparseable output: {e}"

def _is_not_found(err: str) -> bool:
    """True when a kubectl error definitively means the object does not exist (vs an access/API
    error). A forbidden/timeout read is NOT a confirmed absence and must stay inconclusive."""
    low = err.lower()
    # run() emits "not found: <binary>" when the command itself is missing (e.g. kubectl not on
    # PATH). That is a tooling failure, NOT an object-absent signal — misreading it would drive a
    # check down the "not applicable / OK" path and violate the anti-silence contract.
    if low.startswith("not found:"):
        return False
    # kubectl reports an unregistered resource type (e.g. a CRD whose operator is not installed)
    # as "the server doesn't have a resource type \"X\"" — a confirmed absence, not an access error.
    if "doesn't have a resource type" in low or "could not find the requested resource" in low:
        return True
    return "not found" in low or "notfound" in low.replace(" ", "")

def crd_state(name: str) -> str:
    """Return 'present' | 'absent' | 'unreadable' for a CRD.

    A read that fails for any reason other than a confirmed not-found (RBAC forbidden, timeout,
    missing kubectl, API 500) is 'unreadable' — NOT 'absent'. Collapsing those into a single
    boolean is how a failed read turns into a false "not applicable / healthy" verdict, which the
    anti-silence contract forbids.
    """
    rc, _, err = run(["kubectl", "get", "crd", name], 20)
    if rc == 0:
        return "present"
    return "absent" if _is_not_found(err) else "unreadable"

# --- Null-safe accessors: kubectl -o json fields can be null or the wrong type (e.g. status:null,
# conditions:null while a resource reconciles). Iterating those raises and would abort the whole
# run, breaking the stable --json contract — so every parse goes through these. ---
def _dict(x: Any) -> dict[str, Any]:
    return x if isinstance(x, dict) else {}

def _list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else []

def _conditions(obj: Any) -> list[dict[str, Any]]:
    """Return obj.status.conditions as a list of dict conditions, tolerating null/non-list/non-dict."""
    return [c for c in _list(_dict(_dict(obj).get("status")).get("conditions")) if isinstance(c, dict)]

def _condition_is_true(obj: Any, cond_type: str) -> bool:
    return any(c.get("type") == cond_type and c.get("status") == "True" for c in _conditions(obj))

def _rfc3339_age_seconds(ts: str) -> float | None:
    """Age in seconds of an RFC3339 timestamp (fractional seconds / offsets tolerated), or None."""
    if not ts:
        return None
    try:
        when = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.timezone.utc)
    return (datetime.datetime.now(datetime.timezone.utc) - when).total_seconds()

def crd_gate(name: str, check: str, absent_msg: str) -> CheckResult | None:
    """Shared CRD precondition: return an INCONCLUSIVE CheckResult when the CRD is absent (with
    absent_msg) or unreadable (distinct message so the two are never conflated), or None when the
    CRD is present and the caller should proceed."""
    state = crd_state(name)
    if state == "unreadable":
        return CheckResult(check, INCONCLUSIVE, f"cannot read the {name} CRD (RBAC/API error) — state unconfirmed")
    if state == "absent":
        return CheckResult(check, INCONCLUSIVE, absent_msg)
    return None

# --------------------------------------------------------------------------------------------
# Image existence — handles PRIVATE registries correctly: an auth-required response is
# INCONCLUSIVE, not a failure (a Docker Hub tag-API 404 on a private repo is not proof the tag
# is missing). Exposed for reuse by other lanes; do not duplicate this logic.
# --------------------------------------------------------------------------------------------
def image_exists(ref: str) -> tuple[str, str]:
    """Return (status, detail) for a container image reference.

    Prefers a real manifest read (skopeo/crane/docker) and only reports WARN on a definitive
    manifest-unknown/not-found; anything auth-related or otherwise unreadable is INCONCLUSIVE.
    """
    if shutil.which("skopeo"):
        rc, _, err = run(["skopeo", "inspect", f"docker://{ref}"], 60)
        if rc == 0:
            return OK, "manifest present (skopeo)"
        low = err.lower()
        if "manifest unknown" in low or "not found" in low or "manifest for" in low:
            return WARN, f"skopeo: {err[:140]}"
        return INCONCLUSIVE, f"skopeo could not confirm (auth/network?): {err[:140]}"
    if shutil.which("crane"):
        rc, _, err = run(["crane", "manifest", ref], 60)
        if rc == 0:
            return OK, "manifest present (crane)"
        low = err.lower()
        if "manifest unknown" in low or "not found" in low:
            return WARN, f"crane: {err[:140]}"
        return INCONCLUSIVE, f"crane could not confirm (auth/network?): {err[:140]}"
    if shutil.which("docker"):
        rc, _, err = run(["docker", "manifest", "inspect", ref], 60)
        if rc == 0:
            return OK, "manifest present (docker)"
        low = err.lower()
        if "no such manifest" in low or "manifest unknown" in low or "not found" in low:
            return WARN, f"docker manifest: {err[:140]}"
        if "unauthorized" in low or "denied" in low or "authentication" in low:
            return INCONCLUSIVE, "registry auth required (private?) — cannot confirm from here"
        return INCONCLUSIVE, f"docker manifest inconclusive: {err[:140]}"
    # Last resort: the Docker Hub public tag API. Only meaningful for public repos; a 404 here
    # may just mean the repo is private, so it is INCONCLUSIVE, never WARN. A digest ref
    # (name@sha256:...) can't be validated by the tag API — only a manifest tool can — so skip it.
    if ":" in ref and "@" not in ref:
        path, tag = ref.rsplit(":", 1)
        path = path.replace("docker.io/", "").replace("registry-1.docker.io/", "")
        first = path.split("/")[0]
        # Map the ref to a Hub repository path, or None if it isn't a Hub ref:
        #   owner/name -> owner/name   (first component must not be a registry host)
        #   name       -> library/name (official image with no owner)
        # A first component with "." or ":" is a registry host (e.g. ghcr.io, localhost:5000), which
        # the public Hub tag API can't answer for.
        hub_path: str | None = None
        if "/" in path:
            if "." not in first and ":" not in first:
                hub_path = path
        elif "." not in path and ":" not in path:
            hub_path = f"library/{path}"
        if hub_path:
            url = f"https://hub.docker.com/v2/repositories/{hub_path}/tags/{tag}"
            try:
                with urllib.request.urlopen(url, timeout=15) as r:  # noqa: S310 (fixed https host)
                    if r.status == 200:
                        return OK, "tag present (hub api)"
                    return INCONCLUSIVE, f"hub api {r.status}"
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return INCONCLUSIVE, "hub api 404 — repo may be PRIVATE; cannot confirm without registry creds"
                return INCONCLUSIVE, f"hub api {e.code}"
            except urllib.error.URLError as e:
                return INCONCLUSIVE, f"hub api error: {e.reason}"
    return INCONCLUSIVE, "no image-inspection tool available (install skopeo/crane) and not a public hub ref"

def check_image(ref: str) -> CheckResult:
    status, detail = image_exists(ref)
    rec = ""
    if status == WARN:
        rec = "Publish/retag this image or fix the reference to a published tag."
    elif status == INCONCLUSIVE:
        rec = "Confirm with registry creds (skopeo/crane inspect) or verify it pulls on-cluster."
    return CheckResult(f"image-exists:{ref}", status, f"{ref}: {detail}", rec)

# ============================ CLUSTER PLATFORM CHECKS ============================
def check_cilium_mtu() -> CheckResult:
    n = "cilium-mtu"
    state = crd_state("ciliumnodes.cilium.io")
    if state == "unreadable":
        return CheckResult(n, INCONCLUSIVE, "cannot read the ciliumnodes CRD (RBAC/API error) — Cilium MTU state unconfirmed")
    if state == "absent":
        # Confirmed not a Cilium cluster — the download-hang bug is Cilium-specific, so it does not
        # apply. A clean pass (nothing to detect), not an unread check that would block HEALTHY.
        return CheckResult(n, OK, "Cilium not in use (no ciliumnodes.cilium.io CRD) — MTU divergence bug not applicable")
    # MTU divergence is a disagreement between nodes; a single node has nothing to disagree with.
    nodes_ok, nodes, _ = _once_per_run(("nodes",), lambda: kubectl_json(["get", "nodes"]))
    if nodes_ok and len(_list(_dict(nodes).get("items"))) == 1:
        return CheckResult(n, OK, "single-node cluster — per-node MTU divergence is not possible")
    # The reliable, API-observable signal is whether MTU is pinned cluster-wide in cilium-config.
    # A pinned value eliminates the per-node auto-detect divergence that hangs downloads. The
    # per-node effective MTU itself lives on the node OS (not exposed on ciliumnodes — verified),
    # so an unpinned configuration cannot be certified healthy from the Kubernetes API alone.
    ok, cm, err = kubectl_json(["get", "configmap", "cilium-config", "-n", "kube-system"])
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot read cilium-config: {err}", "Grant read access to kube-system, then re-run.")
    mtu = _dict(_dict(cm).get("data")).get("mtu", "")
    if mtu and mtu != "0":
        return CheckResult(n, OK, f"MTU pinned cluster-wide to {mtu} in cilium-config — no per-node auto-detect divergence possible")
    return CheckResult(
        n, INCONCLUSIVE,
        "Cilium MTU is auto-detected (cilium-config mtu unset) — effective per-node MTU is not exposed by the Kubernetes API, so divergence and model-download hangs cannot be ruled out",
        "Compare MTU on every node and pin cilium-config mtu cluster-wide before treating the cluster as proven healthy.",
    )

def _in_e2e_footprint(namespace: str, test_namespace: str) -> bool:
    """True for the namespaces an E2E run creates or depends on: the test namespace, every e2e-* one,
    and the platform namespaces the suites need (app, identity, object storage)."""
    return namespace == test_namespace or namespace.startswith("e2e-") or namespace in _E2E_PLATFORM_NAMESPACES

# ============================ E2E-READINESS CHECKS ============================
def _deploy_env(name: str, namespace: str) -> tuple[bool, dict[str, str], str]:
    """Read a deployment's container env without hiding an unreadable deployment."""
    ok, obj, err = kubectl_json(["get", "deploy", name, "-n", namespace])
    if not ok:
        return False, {}, err
    env: dict[str, str] = {}
    containers = _list(_dict(_dict(_dict(_dict(obj).get("spec")).get("template")).get("spec")).get("containers"))
    for c in containers:
        for e in _list(_dict(c).get("env")):
            if isinstance(e, dict) and e.get("name") and e.get("value") is not None:
                env[e["name"]] = e["value"]
    return True, env, ""

def _secret_state(name: str, namespace: str) -> tuple[str, str]:
    """Return (state, detail) for a secret read, distinguishing present / absent / unreadable.

    state is one of "present", "absent", "unreadable" — an unreadable secret (forbidden, API
    error) is never reported as absent, so a missing-secret WARN is only raised on a confirmed
    not-found.
    """
    rc, out, err = run(["kubectl", "get", "secret", name, "-n", namespace, "-o", "name"], 20)
    if rc == 0 and out:
        return "present", ""
    if rc != 0 and not _is_not_found(err):
        return "unreadable", err[:120]
    return "absent", ""

_SEAWEED_NS = "seaweedfs-instance"

# The runner's own namespaces never hold a LocalQueue, so they are not test namespaces.
_RUNNER_NAMESPACES = frozenset({"e2e-runner", "e2e-test-runner"})

_CONFIGURED_TEST_NAMESPACE = "e2e-apiserver-testing"

_NO_E2E_NAMESPACE = (
    "no e2e-* namespace exists yet (the E2E runs create their own), so there is no namespace to check; "
    "pass --test-namespace NS to check a specific one"
)

def _discover_test_namespace() -> tuple[bool, str | None, str]:
    """(read_ok, namespace, error) for the E2E namespace to check when --test-namespace is not given.

    The runner's own namespaces and namespaces being deleted are never candidates: lanes create
    and tear down per-run project namespaces (e2e-apiserver-proj-*) that have no queue, and the
    health gate runs while the previous lane's are still terminating. A namespace that already
    holds a LocalQueue is preferred, since that is what the Kueue check reads; otherwise only the
    configured test namespace is reported, never an arbitrary per-run project. None means there
    is no namespace to check.
    """
    def fetch() -> tuple[bool, str | None, str]:
        ok, obj, err = kubectl_json(["get", "namespaces"])
        if not ok:
            return False, None, err
        e2e = []
        for item in _list(_dict(obj).get("items")):
            meta = _dict(_dict(item).get("metadata"))
            name = str(meta.get("name", ""))
            terminating = bool(meta.get("deletionTimestamp")) or _dict(_dict(item).get("status")).get("phase") == "Terminating"
            if name.startswith("e2e-") and name not in _RUNNER_NAMESPACES and not terminating:
                e2e.append(name)
        e2e.sort()
        if not e2e:
            return True, None, ""
        lq_ok, lqs, lq_err = kubectl_json(["get", "localqueues.kueue.x-k8s.io", "-A"])
        queued = {str(_dict(_dict(q).get("metadata")).get("namespace", "")) for q in _list(_dict(lqs).get("items"))} if lq_ok else set()
        chosen = next((name for name in e2e if name in queued), None)
        if chosen is None and _CONFIGURED_TEST_NAMESPACE in e2e:
            chosen = _CONFIGURED_TEST_NAMESPACE
        if chosen is None and not lq_ok:
            return False, None, f"cannot list LocalQueues to choose among {', '.join(e2e)}: {lq_err}"
        return True, chosen, ""
    result: tuple[bool, str | None, str] = _once_per_run(("test-namespace",), fetch)
    return result

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow redirects. urllib's default handler replays the Authorization header across
    origins (unlike curl/requests), so a 3xx to another host would leak the bearer key. Returning
    None makes urllib raise the 3xx as an HTTPError instead, which the caller surfaces as a status."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        return None

# Every connect failure of the current _http_status call, on every address, in order. urllib's own
# socket.create_connection keeps only the last one, so a refused address followed by one that
# timed out would read as silence; this list keeps the refusal visible.
_CONNECT_FAILURES = threading.local()

class _AllAddressesFailed(OSError):
    """No address of the host accepted a connection; every failure is in _CONNECT_FAILURES."""

def _connect_each_address(address: tuple[str, int], timeout: Any = socket._GLOBAL_DEFAULT_TIMEOUT,  # type: ignore[attr-defined]
                          source_address: Any = None) -> socket.socket:
    """socket.create_connection, but recording every address's failure, not only the last."""
    failures: list[tuple[str, OSError]] = getattr(_CONNECT_FAILURES, "failures", [])
    host, port = address
    lookup_timeout = timeout if isinstance(timeout, (int, float)) else 20
    infos, err = _bounded_getaddrinfo(host, port, lookup_timeout)
    if infos is None:
        # A lookup that never answered is a timeout; one that answered "no such name" is not.
        raise (TimeoutError if isinstance(err, _Silence) else OSError)(err)
    for family, socktype, proto, _, sockaddr in infos:
        sock = None
        try:
            sock = socket.socket(family, socktype, proto)
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:  # type: ignore[attr-defined]
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as e:
            failures.append((str(sockaddr[0]), e))
            if sock is not None:
                sock.close()
    raise _AllAddressesFailed("; ".join(f"{a}: {e}" for a, e in failures) or f"{host} has no address")

class _EachAddressConnections(urllib.request.HTTPHandler, urllib.request.HTTPSHandler):
    """Opens HTTP and HTTPS connections through _connect_each_address."""

    def do_open(self, http_class: Any, req: Any, **http_conn_args: Any) -> Any:  # type: ignore[override]
        def connection(host: str, **kwargs: Any) -> Any:
            conn = http_class(host, **kwargs)
            conn._create_connection = _connect_each_address
            conn.response_class = _noticing_responses(lambda: setattr(_CONNECT_FAILURES, "answered", True))
            return conn

        return super().do_open(connection, req, **http_conn_args)

# Pass an instance (not the class): build_opener replaces the default HTTPRedirectHandler with any
# subclass it is given, so redirects are suppressed either way, but an instance is unambiguous.
_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect(), _EachAddressConnections())

def _http_status(url: str, bearer: str, timeout: int = 20) -> tuple[int | None, str]:
    """GET url with a Bearer token and return (status_code, error). status_code is None when no
    HTTP response was obtained at all (DNS/TLS/connection failure or a malformed URL), which is
    not a 5xx. Redirects are not followed, so the bearer key never crosses an origin.

    The error is a _Silence only when nothing answered at all: every address of every connection
    timed out, and so did whatever came after."""
    failures: list[tuple[str, OSError]] = []
    _CONNECT_FAILURES.failures = failures
    _CONNECT_FAILURES.answered = False

    def unanswered(e: BaseException) -> str:
        met = [(f"{a}: {f}", _is_silence(f)) for a, f in failures]
        message = str(e) or "timed out"
        if getattr(_CONNECT_FAILURES, "answered", False):
            # The gateway started its answer and then stopped: that is an answer, not silence.
            met.append(("the gateway started answering but did not finish", False))
        if isinstance(e, _AllAddressesFailed):
            # Its message already names every address's failure, and it adds none of its own.
            return _unanswered(message, met)
        if met:
            message = f"{message} (after {'; '.join(m for m, _ in met)})"
        return _unanswered(message, [*met, (message, _is_silence(e))])

    try:
        # Build the Request inside the try: an invalid URL raises ValueError here, not at open().
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {bearer}"}, method="GET")
        with _NO_REDIRECT_OPENER.open(req, timeout=timeout) as r:  # noqa: S310 (scheme validated by caller)
            return int(r.status), ""
    except urllib.error.HTTPError as e:
        # A refused redirect arrives here as its 3xx code; a real 4xx/5xx likewise.
        return int(e.code), ""
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLCertVerificationError):
            return None, _FinalError(str(e.reason))
        return None, unanswered(e.reason) if isinstance(e.reason, BaseException) else str(e.reason)
    # A malformed URL (e.g. a non-numeric port) is rejected before any connection is attempted.
    except (http.client.InvalidURL, ValueError) as e:
        return None, _FinalError(str(e))
    # http.client.HTTPException (BadStatusLine, InvalidURL, ...) and ValueError (a malformed URL from
    # --gateway-url / AI_GATEWAY_URL) are NOT OSError, so without these an odd endpoint would raise
    # out of the eager run_all_checks list and destroy the whole --json contract.
    except (TimeoutError, OSError, http.client.HTTPException, ValueError) as e:
        return None, unanswered(e)
    finally:
        _CONNECT_FAILURES.failures = []
        _CONNECT_FAILURES.answered = False

# A runner's request to a public endpoint occasionally gets no answer while the endpoint itself
# answers promptly from elsewhere; one unanswered probe must not leave a healthy cluster unproven.
# Only an unanswered probe is retried: any HTTP status is the endpoint's real answer.
# The attempt count and backoff are documented in references/checks.md; keep the two in step.
_NETWORK_BACKOFF_SECONDS = (2.0, 5.0)

# Discovery verdicts are cut to one 300-character line; the attempt list is sized to fit inside it.
_ATTEMPT_SUMMARY_CHARS = 280

class _FinalError(str):
    """An unanswered probe's error that is decided before any request leaves (a malformed URL, an
    untrusted certificate, a refused address): the next attempt would fail identically, so it is
    never retried. The helpers return it from the exception types that mean exactly that."""

class _Silence(str):
    """An unanswered probe's error where the only failure was a timeout: nothing on the route said
    anything. That is what a host with no route to the endpoint sees, and it is the only failure an
    answer from another vantage point may stand in for. A certificate, lookup, refusal or reset is
    the route answering badly, and must stay a failure. The type is set where the exception is
    caught and carried through retries, so truncating or combining the text cannot change it."""

class _FirstBytes(io.RawIOBase):
    """The raw socket stream beneath a response's buffered reader, calling `on_first` the first
    time the peer sends anything. It sits below the buffering, so a peer that sends part of a
    status line and then stalls is known to have answered even though no line ever completed."""

    def __init__(self, raw: Any, on_first: Any) -> None:
        super().__init__()
        self._raw, self._on_first = raw, on_first

    def readable(self) -> bool:
        return True

    def readinto(self, b: Any) -> int | None:
        n = self._raw.readinto(b)
        if n and self._on_first is not None:
            on_first, self._on_first = self._on_first, None
            on_first()
        return n

    def fileno(self) -> int:
        return self._raw.fileno()

    def close(self) -> None:
        self._raw.close()
        super().close()

def _noticing_responses(on_first: Any) -> Any:
    """An http.client response class whose stream reports its first received bytes to `on_first`."""

    class _Response(http.client.HTTPResponse):
        def __init__(self, sock: Any, *a: Any, **k: Any) -> None:
            super().__init__(sock, *a, **k)
            # http.client reads the status line and headers from fp. Rebuild it over the same socket
            # stream with the notice beneath the buffer: nothing has been read from it yet. Detaching
            # stops the old reader from closing that stream when it is collected.
            self.fp = io.BufferedReader(_FirstBytes(self.fp.detach(), on_first))

    return _Response

def _is_silence(e: BaseException) -> bool:
    """A timeout is silence; everything else is the route answering. A TLS failure is never a
    TimeoutError (ssl.SSLError is a separate OSError subclass), so it is never silence."""
    return isinstance(e, TimeoutError)

def _unanswered(message: str, failures: list[tuple[str, bool]]) -> str:
    """The error of a probe that got no answer, from every failure it met as (description, whether
    it was silence). It is a _Silence only when there was a failure and every one was silence. This
    is the one place that decision is made: each probe path collects its failures and calls it, so
    no path can classify silence differently."""
    return _Silence(message) if failures and all(silent for _, silent in failures) else message

# Errors reported only as text by the resolver and the certificate check, with no exception type
# left to classify; they are as final as a _FinalError.
_FINAL_UNANSWERED = ("certificate verify failed", "non-public address")

def _retry_unanswered(probe: Any) -> tuple[Any, ...]:
    """Run `probe` (returning a tuple whose first item is the status, None when unanswered, and
    whose last item is the error) until it is answered, with backoff. When every attempt goes
    unanswered the error names each attempt, so the verdict shows the endpoint was given a fair try.
    The summary is a _Silence only when every attempt was one."""
    errors: list[tuple[str, bool]] = []
    for delay in (0.0, *_NETWORK_BACKOFF_SECONDS):
        if delay:
            time.sleep(delay)
        result = probe()
        err = result[-1]
        if result[0] is not None or isinstance(err, _FinalError) or any(
                m in str(err).lower().replace("_", " ") for m in _FINAL_UNANSWERED):
            return result
        errors.append((str(err), isinstance(err, _Silence)))
    return (*result[:-1], _unanswered(_attempt_summary([e for e, _ in errors]), errors))

def _attempt_summary(errors: list[str]) -> str:
    """Name every attempt within _ATTEMPT_SUMMARY_CHARS: runs of the same error share one label
    ("attempts 1-3: ...") and each distinct error gets an equal share of the space, so a long error
    can never push a later attempt out of a truncated verdict."""
    groups: list[tuple[int, int, str]] = []
    for i, e in enumerate(errors, 1):
        if groups and groups[-1][2] == e:
            groups[-1] = (groups[-1][0], i, e)
        else:
            groups.append((i, i, e))
    head = f"no answer on any of {len(errors)} attempts — "
    share = max((_ATTEMPT_SUMMARY_CHARS - len(head)) // len(groups) - len("attempts 1-3: ; "), 20)
    parts = [f"attempt{'s' if a != b else ''} {a}{f'-{b}' if a != b else ''}: {_one_line(e, share)}" for a, b, e in groups]
    return head + "; ".join(parts)

# Where the gateway is probed from inside the cluster: the E2E lanes reach the gateway from there, so
# that answer is the one the run depends on, whatever this host's egress allows. A short-lived pod of
# its own in the E2E runner namespace, running as that namespace's default account, rather than an
# exec into a product workload, which would run the probe with that workload's credentials.
_IN_CLUSTER_PROBE_NAMESPACE = "e2e-test-runner"
# Python, pinned like the other in-cluster helpers (the lane-evidence collector uses the same image).
_IN_CLUSTER_PROBE_IMAGE = "alpine/k8s:1.34.1@sha256:ec714df3813b5405292860f8a1c55c5727bf8c33c88992f1e981efad8065547f"
_IN_CLUSTER_PROBE_POD = "e2e-gateway-probe"
_IN_CLUSTER_PROBE_WORKLOAD = f"{_IN_CLUSTER_PROBE_NAMESPACE}/{_IN_CLUSTER_PROBE_POD}"

# Runs inside that workload's own Python: the same no-redirect GET as _http_status, retried the same
# way, printing one JSON line {"status": N} or {"error": "...", "final": bool}. As in
# _retry_unanswered, only silence is retried: an untrusted certificate or a malformed URL stops at
# once and is reported final, so a later answer from another backend can never hide it.
# url, bearer and backoff are prepended by _in_cluster_http_status.
_IN_CLUSTER_PROBE_SCRIPT = """
import http.client, json, ssl, sys, time, urllib.error, urllib.request
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None
opener = urllib.request.build_opener(NoRedirect())
errors = []
for delay in [0, *backoff]:
    time.sleep(delay)
    try:
        req = urllib.request.Request(url, headers={"Authorization": "Bearer " + bearer})
        with opener.open(req, timeout=20) as r:
            print(json.dumps({"status": r.status})); sys.exit(0)
    except urllib.error.HTTPError as e:
        print(json.dumps({"status": e.code})); sys.exit(0)
    except (http.client.InvalidURL, ValueError) as e:
        print(json.dumps({"error": str(e), "final": True})); sys.exit(0)
    except Exception as e:
        reason = getattr(e, "reason", e)
        if isinstance(reason, ssl.SSLCertVerificationError):
            print(json.dumps({"error": str(reason), "final": True})); sys.exit(0)
        errors.append(str(reason))
print(json.dumps({"error": "; ".join(errors), "final": False}))
"""

def _in_cluster_http_status(url: str, bearer: str) -> tuple[int | None, str]:
    """_http_status, but sent from inside the cluster by a probe pod in _IN_CLUSTER_PROBE_NAMESPACE.
    (status, error); status is None when neither the probe nor the gateway gave an answer."""
    ns = _IN_CLUSTER_PROBE_NAMESPACE
    target = f"{_IN_CLUSTER_PROBE_POD}-{os.urandom(4).hex()}"
    # The arguments travel inside the script on stdin as Python literals (repr), never through a
    # shell or argv, so a URL from cluster configuration cannot become code in the workload and a
    # key never shows in the process list.
    script = f"url, bearer, backoff = {url!r}, {bearer!r}, {list(_NETWORK_BACKOFF_SECONDS)!r}\n{_IN_CLUSTER_PROBE_SCRIPT}"
    # The pod runs as the namespace default account with no privileges and is removed when the
    # script ends; --rm also removes it when the attach fails. When the timeout kills kubectl before
    # --rm runs, the delete below removes it, and the deadline stops it should that delete fail too.
    overrides = json.dumps({"spec": {"automountServiceAccountToken": False, "activeDeadlineSeconds": 240, "securityContext": {
        "runAsNonRoot": True, "runAsUser": 65534, "seccompProfile": {"type": "RuntimeDefault"}},
        "containers": [{"name": target, "image": _IN_CLUSTER_PROBE_IMAGE, "command": ["python3", "-"],
                        "stdin": True, "stdinOnce": True,
                        "securityContext": {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
                        "resources": {"requests": {"cpu": "10m", "memory": "32Mi"}, "limits": {"memory": "128Mi"}}}]}})
    cmd = ["kubectl", "run", target, "-n", ns, "-i", "--rm", "--quiet", "--restart=Never",
           f"--image={_IN_CLUSTER_PROBE_IMAGE}", "--overrides", overrides, "--pod-running-timeout=90s"]
    try:
        rc, out, err = run(cmd, 180, stdin=script)
    finally:
        run(["kubectl", "delete", "pod", target, "-n", ns, "--ignore-not-found", "--wait=false"], 30)
    try:
        answer = _dict(json.loads(out.splitlines()[-1])) if out else {}
    except json.JSONDecodeError:
        answer = {}
    if isinstance(answer.get("status"), int):
        return int(answer["status"]), ""
    if rc != 0:
        return None, f"could not run the probe pod {ns}/{target}: {_one_line(err or out, 200)}"
    message = _one_line(str(answer.get("error") or out or "no output"), 200)
    return None, _FinalError(message) if answer.get("final") is True else message

_APP_NAMESPACES = ("aiwb", "airm")

_MINIO_NS = "minio-tenant-default"

_E2E_PLATFORM_NAMESPACES = frozenset({*_APP_NAMESPACES, "keycloak", _SEAWEED_NS, _MINIO_NS})

_CLOCK_SKEW_SECONDS = 300

def _hf_get(url: str, token: str) -> tuple[int | None, str, str]:
    try:
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}, method="GET")
        with _NO_REDIRECT_OPENER.open(request, timeout=5) as response:  # noqa: S310 (fixed HF origin)
            return int(response.status), response.read(1 << 20).decode("utf-8", "replace"), ""
    except urllib.error.HTTPError as error:
        return int(error.code), error.read(1 << 20).decode("utf-8", "replace"), ""
    # HTTPException (BadStatusLine, ...) is not an OSError; uncaught it would abort every other check.
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException, ValueError) as error:
        return None, "", str(error)

def _load_setup_module(core_repo: str) -> Any:
    import importlib.util
    path = Path(core_repo) / "tooling" / "e2e" / "setup_e2e_prerequisites.py"
    spec = importlib.util.spec_from_file_location("setup_e2e_prerequisites", path)
    if spec is None or spec.loader is None or not path.is_file():
        return None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves the executing module through sys.modules, so it must be registered first.
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return module

def _as_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0

def _selector_arg(selector: Any) -> str | None:
    """Convert a Deployment's matchLabels selector to kubectl's stable key=value form."""
    labels = _dict(_dict(selector).get("matchLabels"))
    if not labels or _dict(selector).get("matchExpressions"):
        return None
    return ",".join(f"{key}={labels[key]}" for key in sorted(labels))

# (Deployment, namespace, role). The charts name each UI Service after its Deployment and give it a
# single port, so the API-server service proxy reaches it by name alone. airm-api has its own role:
# it validates tokens through its own Keycloak client, not the shared OIDC_* validator AIWB uses.
_PRODUCT_DEPLOYMENTS: tuple[tuple[str, str, str], ...] = (
    ("airm-ui", "airm", "ui"),
    ("aiwb-ui", "aiwb", "ui"),
    ("airm-api", "airm", "airm-api"),
    ("aiwb-api", "aiwb", "api"),
)

# The homepage of a UI whose login is broken still answers 200/307; these routes run the auth stack.
_UI_AUTH_PATHS = ("/api/auth/providers", "/api/auth/session")

# kubectl reports a non-2xx `get --raw` answer as `Error from server (<Reason>)`. client-go turns any
# 5xx without a Kubernetes Status body into InternalError, so the reason recovers the status class.
_RAW_REASON_STATUS = {
    "BadRequest": 400, "Unauthorized": 401, "Forbidden": 403, "NotFound": 404, "MethodNotAllowed": 405,
    "NotAcceptable": 406, "Conflict": 409, "UnsupportedMediaType": 415, "Invalid": 422,
    "TooManyRequests": 429, "InternalError": 500, "ServiceUnavailable": 503, "Timeout": 504,
    "Gone": 410, "Expired": 410, "RequestEntityTooLarge": 413, "ServerTimeout": 504,
}

# For a code it has no reason name for, client-go says "the server responded with the status code N".
_RAW_STATUS_CODE = re.compile(r"the server responded with the status code (\d{3})")

# Answers the API server gives itself when it cannot hand the request to the service at all. They say
# nothing about how the application behaves, so they must not be read as its answer. The API server
# reports a failed dial to the pod as an InternalError too, so these are matched whatever the reason.
# `dial tcp` and `Trying to reach:` are Go transport wording, which the Node UIs never answer with.
_PROXY_UNDELIVERED = re.compile(
    r'no endpoints available|error trying to reach service|Trying to reach: |dial tcp |'
    r'resource "services/proxy"|services "[^"]+" not found',
    re.IGNORECASE,
)

_SERVICE_ABSENT = re.compile(r'services "[^"]+" not found', re.IGNORECASE)

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

def _one_line(text: str, limit: int = 300) -> str:
    flat = " ".join(_ANSI_ESCAPE.sub("", text).split())
    return flat if len(flat) <= limit else f"{flat[:limit]}…"

def _answered(status: int, err: str) -> str:
    """How to name a proxied answer's status. client-go reports every 5xx it has no name for (500,
    502, ...) as InternalError, so that one is a server error of unknown code, not a 500."""
    return "answered a server error (5xx)" if "Error from server (InternalError)" in err else f"answered {status}"

def _service_proxy_get(namespace: str, service: str, path: str) -> tuple[int | None, str, str]:
    """GET a path of an in-cluster Service through the API-server proxy: (status, body, error).

    status is None when the API server could not deliver the request (no such Service, no ready
    endpoint, no services/proxy permission) — that is not an answer from the application. Going
    through kubectl reaches the Service wherever the operator's kubeconfig reaches the cluster, with
    no dependency on public DNS, TLS trust, or the gateway."""
    rc, out, err = run(["kubectl", "get", "--raw", f"/api/v1/namespaces/{namespace}/services/{service}/proxy{path}"], 30)
    if rc == 0:
        return 200, out, ""
    match = re.search(r"Error from server \((\w+)\)", err)
    reason = match.group(1) if match else ""
    status = _RAW_REASON_STATUS.get(reason)
    if status is None and (code := _RAW_STATUS_CODE.search(err)):
        status = int(code.group(1))
    if status is None or _PROXY_UNDELIVERED.search(err):
        return None, "", err
    return status, "", err

_RESOLVER_SLOTS = threading.BoundedSemaphore(4)

# Only the ranges operators actually number a cluster's own services from; ip.is_private would also
# accept reserved and documentation ranges (240.0.0.0/4, 192.0.2.0/24, 2001:db8::/32).
_PRIVATE_UNICAST = tuple(ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7"))

_IPV4_COMPATIBLE = ipaddress.ip_network("::/96")

_NAT64 = ipaddress.ip_network("64:ff9b::/96")

def _fold_embedded_ipv4(ip: Any) -> Any:
    """The IPv4 address an IPv6 form embeds (IPv4-mapped, IPv4-compatible, NAT64), else `ip` itself."""
    if ip.version == 6:
        if ip.ipv4_mapped is not None:
            return ip.ipv4_mapped
        if ip in _IPV4_COMPATIBLE or ip in _NAT64:
            return ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
    return ip

def _bounded_getaddrinfo(host: str, port: int, timeout: float) -> tuple[list[Any] | None, str]:
    """(getaddrinfo's answer, error) with a deadline. A resolver that does not answer in time is
    silence; one that answers with an error (no such name) is not."""
    # getaddrinfo has no timeout of its own; a daemon thread lets a hung resolver be abandoned. An
    # abandoned lookup keeps its thread until the resolver gives up, so a cap on lookups still in
    # flight stops a hung resolver from accumulating threads across fetches.
    if not _RESOLVER_SLOTS.acquire(blocking=False):
        message = f"cannot resolve {host}: earlier lookups are still hung, so name resolution is not answering"
        return None, _unanswered(message, [(message, True)])
    answer: list[Any] = []

    def resolve() -> None:
        try:
            answer.append(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except (OSError, UnicodeError) as e:
            answer.append(e)
        finally:
            _RESOLVER_SLOTS.release()

    worker = threading.Thread(target=resolve, daemon=True)
    worker.start()
    worker.join(timeout)
    if not answer:
        # A resolver that never answers is silence, like a connection that never does; a resolver
        # that answers "no such name" is an answer and is not.
        message = f"cannot resolve {host}: no answer within {timeout:.0f}s"
        return None, _unanswered(message, [(message, True)])
    if isinstance(answer[0], BaseException):
        return None, f"cannot resolve {host}: {answer[0]}"
    return answer[0], ""

def _resolve_public(host: str, port: int, timeout: float, allow_private: bool = False) -> tuple[list[str], str]:
    """(addresses to connect to, error). Every address `host` resolves to must be public (not
    loopback, link-local such as cloud metadata, private or reserved); anything else, a failed
    resolution, or one that takes longer than `timeout`, is refused and no address is returned.

    allow_private additionally accepts private-range unicast addresses (RFC 1918, unique-local), for
    a host the caller trusts to be the cluster's own: a cluster's Keycloak is commonly published on
    a private load-balancer address. Loopback, link-local (cloud metadata) and multicast stay refused."""
    infos, err = _bounded_getaddrinfo(host, port, timeout)
    if infos is None:
        return [], err
    addresses = list(dict.fromkeys(str(info[4][0]).split("%", 1)[0] for info in infos))
    rejected: list[str] = []
    for address in addresses:
        ip = ipaddress.ip_address(address)
        # is_global is true for multicast (224.0.0.0/4, ff00::/8), which is not a unicast server, and
        # for IPv4-compatible (::a.b.c.d) and NAT64 (64:ff9b::a.b.c.d) forms, which reach the IPv4
        # address they embed, so that address is what must be public.
        ip = _fold_embedded_ipv4(ip)
        if allow_private and any(ip in net for net in _PRIVATE_UNICAST):
            continue
        if not ip.is_global or ip.is_multicast:
            rejected.append(address)
    if rejected:
        # Every rejected answer is named, so a caller judging the refusal sees the whole answer set, not
        # whichever address DNS happened to list first.
        return [], f"refused to fetch it from the operator's host: {host} resolves to non-public address {', '.join(rejected)}"
    if not addresses:
        return [], f"cannot resolve {host}: no address"
    return addresses, ""

class _Sockets:
    """Every socket a fetch opens, so the caller can close them all when it abandons the fetch at its
    deadline: closing a socket is what unblocks a worker stuck in connect, a TLS handshake or a read,
    so the thread ends instead of holding the descriptor. A socket registered after close_all() is
    closed at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open: list[socket.socket] = []
        self._closed = False

    def add(self, sock: socket.socket) -> socket.socket:
        with self._lock:
            if not self._closed:
                self._open.append(sock)
                return sock
        sock.close()
        raise TimeoutError("the fetch deadline passed")

    def close_all(self) -> None:
        with self._lock:
            self._closed = True
            doomed, self._open = self._open, []
        for sock in doomed:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()

def _dial(address: tuple[str, int], timeout: float | None, sockets: _Sockets) -> socket.socket:
    """Connect to a vetted numeric address. The socket is registered before connecting, so even a
    connect that never completes can be closed by the caller."""
    family = socket.AF_INET6 if ipaddress.ip_address(address[0]).version == 6 else socket.AF_INET
    sock = sockets.add(socket.socket(family, socket.SOCK_STREAM))
    sock.settimeout(timeout)
    sock.connect(address)
    return sock

class _PinnedHTTPConnection(http.client.HTTPConnection):
    """An HTTP connection to an address resolved and vetted beforehand, so the name cannot resolve
    to another address between the check and the connection."""

    def __init__(self, host: str, address: str, port: int, timeout: float, sockets: _Sockets) -> None:
        super().__init__(host, port, timeout=timeout)
        self._address = address
        self._sockets = sockets

    def connect(self) -> None:
        self.sock = _dial((self._address, self.port), self.timeout, self._sockets)

class _PinnedHTTPSConnection(_PinnedHTTPConnection):
    """As _PinnedHTTPConnection, with TLS verified against the host name (SNI and certificate)."""

    default_port = http.client.HTTPS_PORT

    def connect(self) -> None:
        raw = _dial((self._address, self.port), self.timeout, self._sockets)
        # Wrapping moves the descriptor to the TLS socket, so that is the one to register before
        # the handshake can stall.
        tls = ssl.create_default_context().wrap_socket(raw, server_hostname=self.host, do_handshake_on_connect=False)
        self._sockets.add(tls)
        tls.do_handshake()
        self.sock = tls

def _tighten(conn: http.client.HTTPConnection, deadline: float) -> None:
    """Give the next socket operation only what is left of the fetch's deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("the fetch deadline passed")
    if conn.sock is not None:
        conn.sock.settimeout(remaining)

def _http_get(url: str, timeout: int = 20, allow_private: bool = False) -> tuple[int | None, str, str]:
    """Unauthenticated GET of a public http(s) URL: (status, body, error). status is None when no
    HTTP response was obtained. Redirects are not followed.

    The URL comes from cluster configuration but the request leaves from the operator's host, so it
    only ever connects to a public address, vetted once and then pinned for the connection: a
    Deployment must not be able to make the check probe the operator's localhost, cloud metadata or
    private network, directly or by re-resolving the name to another address."""
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError as e:
        return None, "", _FinalError(str(e))
    scheme, host = parts.scheme.lower(), parts.hostname or ""
    if scheme not in ("http", "https") or not host:
        return None, "", _FinalError(f"unsupported URL {url!r}")
    port = port or (443 if scheme == "https" else 80)
    # One deadline covers resolving, connecting, and every read, so no phase can stretch the fetch.
    deadline = time.monotonic() + timeout
    addresses, err = _resolve_public(host, port, timeout, allow_private)
    if not addresses:
        return None, "", err
    conn_type = _PinnedHTTPSConnection if scheme == "https" else _PinnedHTTPConnection
    path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    # Socket timeouts bound each operation, not the sum: a server trickling bytes (TLS handshake,
    # headers, body) could stretch the fetch indefinitely. The attempts therefore run in a daemon
    # thread that is abandoned, and its connection closed, when the deadline passes.
    if time.monotonic() >= deadline:
        # The whole budget went to a resolver that answered too late: that is silence too.
        message = f"not tried, the {timeout}s deadline had passed while resolving {host}"
        return None, "", _unanswered(message, [(message, True)])
    sockets = _Sockets()
    outcome: list[tuple[int | None, str, str]] = []
    # Failures the worker has already met; at the deadline they decide whether the fetch was silence.
    failures: list[tuple[str, bool]] = []
    worker = threading.Thread(
        target=lambda: outcome.append(_fetch_pinned(conn_type, host, addresses, port, path, deadline, timeout, sockets, failures)),
        daemon=True,
    )
    worker.start()
    worker.join(max(deadline - time.monotonic(), 0))
    # Read what the worker had reached before closing its sockets: closing them wakes a blocked
    # connect or handshake with an error that is cleanup, not anything the peer did.
    finished, earlier = list(outcome), list(failures)
    sockets.close_all()
    if finished:
        return finished[0]
    message = f"no complete answer within {timeout}s"
    # The worker records each failure, and a response that arrived, as it happens; whatever it was
    # doing when abandoned was waiting, which on its own is silence.
    if earlier:
        message = f"{message} (after {'; '.join(m for m, _ in earlier)})"
    return None, "", _unanswered(message, [*earlier, (message, True)])

def _fetch_pinned(conn_type: Any, host: str, addresses: list[str], port: int, path: str, deadline: float,
                  timeout: int, sockets: _Sockets,
                  failures: list[tuple[str, bool]] | None = None) -> tuple[int | None, str, str]:
    """Try each vetted address in turn: one unreachable family (e.g. IPv6 on an IPv4-only host) must
    not make a reachable issuer look unreadable. Each failure is recorded in `failures` with whether
    it was silence, as it happens, so a caller abandoning this at its deadline still knows. Once a
    response has arrived, nothing after it is silence: the endpoint answered, then failed to finish."""
    failures = [] if failures is None else failures
    for address in addresses:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            failures.append((f"{address}: not tried, the {timeout}s deadline had passed", True))
            break
        conn = conn_type(host, address, port, remaining, sockets)
        started: list[bool] = []

        def first_bytes(address: str = address, started: list[bool] = started) -> None:
            started.append(True)
            failures.append((f"{address}: started answering but did not finish in time", False))

        conn.response_class = _noticing_responses(first_bytes)
        try:
            conn.request("GET", path)
            _tighten(conn, deadline)
            response = conn.getresponse()
            if not started:
                first_bytes()
            failures[-1] = (f"{address}: answered {response.status} but did not finish its document in time", False)
            chunks: list[bytes] = []
            size = 0
            while response.status == 200 and size < (1 << 20):
                _tighten(conn, deadline)
                chunk = response.read1(1 << 16)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
            return int(response.status), b"".join(chunks)[: 1 << 20].decode("utf-8", "replace"), ""
        except (TimeoutError, OSError) as e:
            if started:
                failures[-1] = (f"{failures[-1][0].split(' but did not finish')[0]}, then {e}", False)
            else:
                failures.append((f"{address}: {e}", _is_silence(e)))
        except (http.client.HTTPException, ValueError) as e:
            return None, "", str(e)
        finally:
            conn.close()
    return None, "", _unanswered("; ".join(m for m, _ in failures), failures)

def _cluster_service_target(url: str, cluster_domain: str = "cluster.local") -> tuple[str, str, str] | None:
    """(namespace, proxy service name, path with query) for an in-cluster Service URL such as
    http://keycloak.keycloak.svc.cluster.local:8080/realms/airm, or None for any other URL."""
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError:
        return None  # a malformed URL is not a Service; the direct fetch reports it as unreadable
    if parts.scheme not in ("http", "https"):
        return None
    # <service>.<namespace>.svc, optionally followed by the cluster's DNS domain; a public host that
    # merely has "svc" as its third label is not a Service.
    labels = (parts.hostname or "").split(".")
    if len(labels) < 3 or labels[2] != "svc" or labels[3:] not in ([], cluster_domain.strip(".").split(".")):
        return None
    service = labels[0] + (f":{port}" if port else "")
    if parts.scheme == "https":
        service = f"https:{service}"
    return labels[1], service, parts.path + (f"?{parts.query}" if parts.query else "")
