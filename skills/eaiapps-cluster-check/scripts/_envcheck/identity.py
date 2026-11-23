# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Checks for who may use the cluster and what already occupies it: the inference gateway,
OIDC issuers, UI auth, E2E RBAC, project quota and the namespaces earlier runs left behind."""
from __future__ import annotations

import http.client
import ipaddress
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from typing import Any

from .shared import (
    CheckResult,
    INCONCLUSIVE,
    OK,
    WARN,
    _IN_CLUSTER_PROBE_NAMESPACE,
    _IN_CLUSTER_PROBE_WORKLOAD,
    _NO_E2E_NAMESPACE,
    _PRIVATE_UNICAST,
    _PRODUCT_DEPLOYMENTS,
    _SERVICE_ABSENT,
    _Silence,
    _UI_AUTH_PATHS,
    _answered,
    _as_int,
    _cluster_service_target,
    _condition_is_true,
    _deploy_env,
    _dict,
    _discover_test_namespace,
    _fold_embedded_ipv4,
    _http_get,
    _http_status,
    _in_cluster_http_status,
    _is_not_found,
    _list,
    _once_per_run,
    _one_line,
    _retry_unanswered,
    _rfc3339_age_seconds,
    _selector_arg,
    _service_proxy_get,
    crd_gate,
    crd_state,
    kubectl_json,
    run,
)


def check_aimartifact_rbac(test_namespace: str | None) -> CheckResult:
    n = "aimartifact-rbac"
    gate = crd_gate("aimartifacts.aim.eai.amd.com", n, "AIMArtifact CRD not present (aimartifacts.aim.eai.amd.com) — cannot check create permission")
    if gate:
        return gate
    if test_namespace is None:
        ns_ok, test_namespace, ns_err = _discover_test_namespace()
        if not ns_ok:
            return CheckResult(n, INCONCLUSIVE, f"cannot list namespaces to find the E2E namespace: {ns_err}")
        if test_namespace is None:
            return CheckResult(n, OK, _NO_E2E_NAMESPACE)
    # `kubectl auth can-i` prints yes/no on stdout as the authoritative answer (and exits non-zero
    # for a definitive "no"), so the stdout answer — not the exit code — is what we read. Some
    # versions prepend a warning line, so take the last non-empty line; anything that is not
    # yes/no means the access review could not be evaluated, which is inconclusive.
    _, out, err = run(["kubectl", "auth", "can-i", "create", "aimartifacts.aim.eai.amd.com", "-n", test_namespace], 20)
    lines = [ln for ln in out.strip().splitlines() if ln.strip()]
    verdict = lines[-1].strip().lower() if lines else ""
    if verdict not in ("yes", "no"):
        return CheckResult(n, INCONCLUSIVE, f"cannot evaluate AIMArtifact create permission in ns {test_namespace}: {err[:120]}")
    if verdict == "yes":
        return CheckResult(
            n, OK,
            f"current kubectl identity can create AIMArtifacts in ns {test_namespace} — rules out the RBAC-config denial, but not an app-side Forbidden (the check tests the kubectl identity, not the app's runtime path)",
        )
    return CheckResult(
        n, WARN, f"cannot create AIMArtifacts in ns {test_namespace} (Forbidden)",
        "Adapter tests fail with 'Failed to create AIMArtifact CR ...: Forbidden'; grant create on aimartifacts to the test identity in this namespace.",
    )

_E2E_RBAC_ROLE = "e2e-aim-test-author"

# What the E2E suites actually exercise as the test identity, resource -> required verbs. The
# adapter/model/custom-model suites author these CRs with kubectl to stage cases the API cannot
# express, so a missing verb surfaces as a mid-suite Forbidden rather than a setup error.
_E2E_RBAC_REQUIRED: dict[str, set[str]] = {
    "aimartifacts": {"get", "create", "patch"},
    "aimprofiles": {"get", "list"},
    # The custom-model precision rounds read both collections back with kubectl to confirm the
    # project is empty before the next round reuses it, so these need list as well as get.
    "aimmodels": {"get", "list", "create", "patch", "delete"},
    # kubectl patch --subresource=status reads before writing, so the deployed role (and the
    # least-privilege guard) grant get+patch; require both here or a role missing get would pass.
    "aimmodels/status": {"get", "patch"},
    "aimservices": {"get", "list", "create", "patch", "delete"},
}

_AIM_API_GROUP = "aim.eai.amd.com"

def _rule_covers(rule: dict[str, Any], resource: str) -> bool:
    """True when an RBAC rule's apiGroups/resources select this AIM resource.

    Matches the Kubernetes authorizer (ResourceMatches, k8s.io/kubernetes pkg/apis/rbac): a rule
    resource matches on literal ``*``, an exact string match, or the ``*/<subresource>`` form (which
    grants that subresource across all resources). Note there is no ``<parent>/*`` form in RBAC.
    """
    # A resourceNames-restricted rule grants access only to specific named objects. The suites
    # create arbitrary AIM objects (and create itself cannot be name-restricted), so such a rule
    # must not count as satisfying the requirement — otherwise the check is OK while the suite 403s.
    if _list(rule.get("resourceNames")):
        return False
    groups = {str(g) for g in _list(rule.get("apiGroups"))}
    resources = {str(r) for r in _list(rule.get("resources"))}
    if not (_AIM_API_GROUP in groups or "*" in groups):
        return False
    if "*" in resources or resource in resources:
        return True
    # "*/<subresource>" grants that subresource on every resource (the only subresource wildcard RBAC has).
    return "/" in resource and f"*/{resource.split('/', 1)[1]}" in resources

def check_e2e_rbac(expected_oidc_user: str | None = None) -> CheckResult:
    """Assert the E2E test identity is granted what the suites actually do with kubectl.

    A missing ClusterRole here took out ~20 AIWB tests in one night: every case that authors an
    AIM CR failed deep into the suite with a Forbidden that read like a product bug.

    With expected_oidc_user set, the binding must bind that exact user (the cluster's OIDC prefix
    'oidc:' or the legacy 'oidc' form) — otherwise a binding to the WRONG OIDC user would pass. When
    it is None the check keeps the looser "any OIDC user" acceptance for ad-hoc runs.
    """
    n = "eai-e2e-rbac"
    ok, role, err = kubectl_json(["get", "clusterrole", _E2E_RBAC_ROLE])
    if not ok:
        if _is_not_found(err):
            return CheckResult(
                n, WARN,
                f"the E2E test identity has no permissions — ClusterRole '{_E2E_RBAC_ROLE}' does not exist; every test that authors an AIM resource will be refused",
                "Install/upgrade the eai-e2e chart so the E2E RBAC is applied before the suite runs.",
            )
        return CheckResult(n, INCONCLUSIVE, f"cannot read ClusterRole '{_E2E_RBAC_ROLE}': {err}")

    granted: dict[str, set[str]] = {}
    for rule in _list(_dict(role).get("rules")):
        rule = _dict(rule)
        verbs = {str(v) for v in _list(rule.get("verbs"))}
        for resource in _E2E_RBAC_REQUIRED:
            if _rule_covers(rule, resource):
                granted.setdefault(resource, set()).update(verbs)
    missing = {
        resource: sorted(required - granted.get(resource, set()))
        for resource, required in _E2E_RBAC_REQUIRED.items()
        if "*" not in granted.get(resource, set()) and not required <= granted.get(resource, set())
    }
    if missing:
        detail = "; ".join(f"{res} missing {', '.join(verbs)}" for res, verbs in sorted(missing.items()))
        return CheckResult(
            n, WARN,
            f"the E2E test identity is missing permissions the suites use — ClusterRole '{_E2E_RBAC_ROLE}': {detail}",
            "Upgrade the eai-e2e chart so the role grants these verbs; otherwise those tests fail mid-run with Forbidden.",
        )

    ok_b, binding, err_b = kubectl_json(["get", "clusterrolebinding", _E2E_RBAC_ROLE])
    if not ok_b:
        if _is_not_found(err_b):
            return CheckResult(
                n, WARN,
                f"ClusterRole '{_E2E_RBAC_ROLE}' grants the right verbs but no ClusterRoleBinding of that name exists — nothing is bound to the test identity",
                "Install/upgrade the eai-e2e chart so the ClusterRoleBinding is applied.",
            )
        return CheckResult(n, INCONCLUSIVE, f"role is complete, but cannot read ClusterRoleBinding '{_E2E_RBAC_ROLE}': {err_b}")
    ref = _dict(_dict(binding).get("roleRef"))
    if ref.get("kind") != "ClusterRole" or ref.get("name") != _E2E_RBAC_ROLE:
        return CheckResult(
            n, WARN,
            f"ClusterRoleBinding '{_E2E_RBAC_ROLE}' does not reference ClusterRole '{_E2E_RBAC_ROLE}' (points at {ref.get('kind') or '?'}/{ref.get('name') or '?'}) — the test identity does not get these permissions",
            "Correct the binding's roleRef so the E2E permissions actually apply.",
        )
    # The cluster prefixes OIDC users with 'oidc:' (a legacy 'oidc' form is also bound), so any User
    # subject carrying that prefix is an OIDC identity.
    subjects = _list(_dict(binding).get("subjects"))
    users = [str(_dict(s).get("name", "")) for s in subjects if _dict(s).get("kind") == "User"]
    oidc_users = [u for u in users if u.startswith("oidc")]
    if not oidc_users:
        # List every subject (kind/name), not just Users — otherwise a ServiceAccount-only binding
        # renders "(subjects: none)", which misleads someone troubleshooting.
        rendered = ", ".join(f"{_dict(s).get('kind', '?')}/{_dict(s).get('name', '?')}" for s in subjects) or "none"
        return CheckResult(
            n, WARN,
            f"ClusterRoleBinding '{_E2E_RBAC_ROLE}' binds no OIDC user — the E2E test identity is not granted these permissions (subjects: {rendered})",
            "Bind the E2E test user to the role (the eai-e2e chart does this from its configured username).",
        )
    if expected_oidc_user:
        # Require the binding to name THIS user, under either prefix the chart emits. Without this an
        # otherwise-complete binding to a DIFFERENT OIDC user would pass while the real runner 403s.
        wanted = {f"oidc:{expected_oidc_user}", f"oidc{expected_oidc_user}"}
        if not wanted & set(oidc_users):
            return CheckResult(
                n, WARN,
                f"ClusterRoleBinding '{_E2E_RBAC_ROLE}' is not bound to the expected E2E user '{expected_oidc_user}' — bound OIDC users: {', '.join(oidc_users)}",
                "Bind the role to the OIDC user the E2E runner authenticates as, or drop --e2e-oidc-user if any OIDC user is acceptable.",
            )
        return CheckResult(
            n, OK,
            f"the E2E test identity has the permissions the suites use — ClusterRole '{_E2E_RBAC_ROLE}' grants every required verb and is bound to the expected user '{expected_oidc_user}'",
        )
    return CheckResult(
        n, OK,
        f"the E2E test identity has the permissions the suites use — ClusterRole '{_E2E_RBAC_ROLE}' grants every required verb and is bound to {', '.join(oidc_users)}",
    )

_PROJECT_LABEL = "airm.silogen.ai/project-id"

_AIRM_FINALIZER_PREFIX = "airm.silogen.ai/"

_AIRM_DELETION_GRACE_SECONDS = 300

def check_airm_quota_health() -> CheckResult:
    """Detect the dominant AIRM failure path from its residue, without creating anything.

    An AIRM project is a database entity that materialises a labelled namespace plus a Kueue
    ClusterQueue named after it. Two failures strand a whole E2E run: a namespace left half-deleted
    behind an AIRM finalizer PAST a grace window (holding it briefly is normal teardown), and a live
    project whose same-named ClusterQueue is missing or not Active. Both are reported together, and
    an unreadable state is INCONCLUSIVE, never a pass.
    """
    n = "airm-quota-health"
    ok, obj, err = kubectl_json(["get", "namespaces", "-l", _PROJECT_LABEL])
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot list AIRM project namespaces: {err}")
    projects = _list((obj or {}).get("items"))
    if not projects:
        return CheckResult(n, OK, "no AIRM project namespaces present — nothing stuck deleting and no quota to verify")

    problems: list[str] = []
    unconfirmed: list[str] = []
    live: list[str] = []
    deleting = 0
    for ns in projects:
        meta = _dict(_dict(ns).get("metadata"))
        name = str(meta.get("name", "?"))
        deletion_ts = str(meta.get("deletionTimestamp") or "")
        if not deletion_ts:
            live.append(name)
            continue
        deleting += 1
        held = [str(f) for f in _list(meta.get("finalizers")) if str(f).startswith(_AIRM_FINALIZER_PREFIX)]
        if not held:
            continue
        # Holding an AIRM finalizer during deletion is normal; only past the grace is it "stuck".
        age = _rfc3339_age_seconds(deletion_ts)
        if age is None:
            unconfirmed.append(f"{name}(deletion age unreadable)")
        elif age >= _AIRM_DELETION_GRACE_SECONDS:
            problems.append(f"{name}(stuck deleting; {', '.join(held)})")

    if live:
        ok_q, qobj, err_q = kubectl_json(["get", "clusterqueues.kueue.x-k8s.io"])
        if not ok_q:
            detail = (
                "Kueue ClusterQueues are absent (is Kueue installed?)"
                if _is_not_found(err_q)
                else f"ClusterQueues unreadable: {err_q}"
            )
            unconfirmed.append(f"quota for {len(live)} live project(s): {detail}")
        else:
            queues = {
                str(_dict(_dict(q).get("metadata")).get("name", "")): q
                for q in _list((qobj or {}).get("items"))
            }
            # A project's ClusterQueue carries the project name; Active=True is Kueue's own statement
            # that it can admit work. Zero local quota is NOT flagged — every project queue joins the
            # kaiwo cohort with borrowing enabled, so a zero-quota project borrows (quota_preemption).
            for name in live:
                queue = queues.get(name)
                if queue is None:
                    problems.append(f"{name}(no ClusterQueue)")
                elif not _condition_is_true(queue, "Active"):
                    problems.append(f"{name}(ClusterQueue not Active)")

    if problems:
        suffix = f"; also unconfirmed: {', '.join(unconfirmed)}" if unconfirmed else ""
        return CheckResult(
            n, WARN,
            f"AIRM project lifecycle/quota problems will block the suite: {', '.join(sorted(problems))}{suffix}",
            "For stuck deletion, inspect airm-agent finalizer errors. For missing/inactive quota, inspect airm-agent and the Kueue controller before the run.",
        )
    if unconfirmed:
        return CheckResult(
            n, INCONCLUSIVE,
            f"cannot confirm every AIRM project is healthy: {', '.join(unconfirmed)}",
            "Restore the failed read (if ClusterQueues are absent, confirm Kueue is installed and its controller is running) or inspect the malformed deletion timestamp, then re-run; unconfirmed state is never counted healthy.",
        )
    return CheckResult(
        n, OK,
        f"all {len(live)} live AIRM project namespace(s) have an Active ClusterQueue; {deleting} deleting namespace(s) are within the {_AIRM_DELETION_GRACE_SECONDS}s grace or no longer held by AIRM",
    )


_RUN_NAMESPACE = re.compile(r"^e2e-ci(\d+)-")


class RunLookupError(Exception):
    """GitHub could be asked about a run but did not answer."""


def _github_run_status(run_id: str) -> str | None:
    """The run's GitHub status ("completed", "in_progress", ...), or None when no credentials are set
    or GitHub does not know the run.

    Only a CI caller can ask: it needs GITHUB_REPOSITORY and GITHUB_TOKEN, which an operator's shell
    usually lacks. With them set, a failed lookup raises RunLookupError rather than reading as
    "cannot ask", so CI never treats a run it failed to look up as harmless. The request goes to
    GitHub's API with certificate checks on."""
    repo, token = os.environ.get("GITHUB_REPOSITORY", ""), os.environ.get("GITHUB_TOKEN", "")
    if not repo or not token:
        return None
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    request = urllib.request.Request(f"{api}/repos/{repo}/actions/runs/{run_id}",
                                     headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 - fixed https API
            status = json.loads(response.read().decode()).get("status")
    except urllib.error.HTTPError as error:
        # 404: a run of another repository sharing the cluster, or one past retention. The repair leaves
        # such a namespace alone too, so gating on it would hold every nightly with nothing to clear it.
        if error.code == 404:
            return None
        raise RunLookupError(f"run {run_id}: {error}") from error
    except (OSError, ValueError, http.client.HTTPException) as error:
        raise RunLookupError(f"run {run_id}: {error}") from error
    if not status:
        raise RunLookupError(f"run {run_id}: GitHub returned no status")
    return str(status)


def check_finished_run_leftovers() -> CheckResult:
    """Namespaces an earlier CI run left behind.

    The repairs remove them before the check, but a repair that was refused or never got the cluster
    lock leaves them, and a run that starts beside them inherits their quota, queue entries and volumes."""
    n = "finished-run-leftovers"
    ok, obj, err = kubectl_json(["get", "namespaces"])
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot list namespaces: {err}")
    runs: dict[str, list[str]] = {}
    for ns in _list(_dict(obj).get("items")):
        name = str(_dict(_dict(ns).get("metadata")).get("name", ""))
        match = _RUN_NAMESPACE.match(name)
        if match:
            runs.setdefault(match.group(1), []).append(name)
    if not runs:
        return CheckResult(n, OK, "no namespaces of CI runs")
    statuses: dict[str, str | None] = {}
    failed: list[str] = []
    for run in sorted(runs):
        try:
            statuses[run] = _github_run_status(run)
        except RunLookupError as error:
            statuses[run] = None
            failed.append(str(error))
    finished = sorted(ns for run, names in runs.items() if statuses[run] == "completed" for ns in names)
    unknown = sorted(ns for run, names in runs.items() if statuses[run] is None for ns in names)
    note = f"; cannot ask GitHub whether these runs finished, so they are not judged: {', '.join(unknown)}" if unknown else ""
    if finished:
        return CheckResult(
            n, WARN,
            f"namespaces of finished CI runs are still on the cluster: {', '.join(finished)}{note}",
            "Remove them, AIRM project first, then the namespace and its Kaiwo queue entry: "
            "the platform re-adds the rest while the project is still there.",
        )
    if failed:
        return CheckResult(
            n, INCONCLUSIVE,
            f"cannot tell whether these CI runs finished, so their namespaces may be leftovers: {', '.join(unknown)} ({'; '.join(failed)[:300]})",
            "Retry once GitHub answers; a finished run's namespace holds quota and queue entries the next run needs.",
        )
    if unknown:
        return CheckResult(n, OK, f"no namespaces of finished CI runs confirmed{note}", gating=False)
    return CheckResult(n, OK, f"only namespaces of runs still in progress: {', '.join(sorted(ns for names in runs.values() for ns in names))}")

_GATEWAY_MODELS_PATH = "/v1/models"

_AI_GATEWAY_ROUTE_CRD = "aigatewayroutes.aigateway.envoyproxy.io"

# Deliberately not a plausible key: it must be rejected, and it must never collide with a real one.
_GATEWAY_INVALID_KEY = "environment-check-invalid-key-expect-401"

_ENVOY_SECURITY_POLICY = "securitypolicies.gateway.envoyproxy.io"

# Every Gateway API route kind that can attach to a Gateway and so carry inference traffic.
_GATEWAY_ROUTE_KINDS = (
    "httproutes.gateway.networking.k8s.io",
    "grpcroutes.gateway.networking.k8s.io",
)

def _ext_authz_gateways() -> tuple[bool, set[tuple[str, str]]]:
    """(read_ok, {(namespace, name)}) of every Gateway an ext_authz SecurityPolicy protects.

    These are the gateways whose auth the invalid-key probe exercises, identified from the policy
    itself rather than guessed from a naming convention.
    """
    ok, obj, _ = kubectl_json(["get", _ENVOY_SECURITY_POLICY, "-A"])
    if not ok:
        return False, set()
    gateways: set[tuple[str, str]] = set()
    for pol in _list((obj or {}).get("items")):
        spec = _dict(_dict(pol).get("spec"))
        if not spec.get("extAuth"):
            continue
        ns = _dict(_dict(pol).get("metadata")).get("namespace", "")
        refs = _list(spec.get("targetRefs")) or ([spec["targetRef"]] if spec.get("targetRef") else [])
        for ref in refs:
            ref = _dict(ref)
            if ref.get("kind") == "Gateway" and ref.get("name"):
                gateways.add((str(ref.get("namespace") or ns), str(ref["name"])))
    return True, gateways

def _no_inference_routes_published() -> bool:
    """True only when it is CONFIRMED that no route reaches an ext_authz-protected gateway.

    Requires the Envoy AI Gateway route CRD to be present (this cluster demonstrably routes
    inference through it), zero AIGatewayRoutes, at least one ext_authz SecurityPolicy, and zero
    HTTPRoutes or GRPCRoutes whose parentRef names a gateway that policy protects. Any unreadable
    state, or a cluster that does not use this CRD, returns False, so the caller keeps its original
    verdict instead of downgrading a possible auth failure.
    """
    if crd_state(_AI_GATEWAY_ROUTE_CRD) != "present":
        return False
    ok, obj, _ = kubectl_json(["get", _AI_GATEWAY_ROUTE_CRD, "-A"])
    if not ok or _list((obj or {}).get("items")):
        return False
    ok, protected = _ext_authz_gateways()
    if not ok or not protected:
        return False
    for kind in _GATEWAY_ROUTE_KINDS:
        state = crd_state(kind)
        if state == "absent":
            continue
        if state != "present":
            return False
        ok, obj, _ = kubectl_json(["get", kind, "-A"])
        if not ok:
            return False
        for r in _list((obj or {}).get("items")):
            route_ns = _dict(_dict(r).get("metadata")).get("namespace", "")
            for p in _list(_dict(_dict(r).get("spec")).get("parentRefs")):
                p = _dict(p)
                if p.get("kind", "Gateway") != "Gateway":
                    continue
                if (str(p.get("namespace") or route_ns), str(p.get("name", ""))) in protected:
                    return False
    return True

def check_gateway_auth(gateway_url: str | None = None, gateway_key: str | None = None) -> CheckResult:
    """Prove the inference gateway is serving auth decisions rather than failing.

    The invalid-key half needs no credentials: a healthy gateway answers 401. A 5xx instead is the
    stale-certificate breakage, which otherwise surfaces as unexplained inference failures mid-suite.
    """
    n = "gateway-auth"
    if gateway_url:
        url = gateway_url
    else:
        # An unreadable deployment is not the same as one without the variable, and reporting the
        # latter for the former sends the operator looking for a config gap that does not exist.
        env_ok, env, env_err = _deploy_env("aiwb-ui", "aiwb")
        if not env_ok:
            return CheckResult(
                n, INCONCLUSIVE,
                f"cannot read the aiwb-ui deployment to discover the inference gateway address: {env_err} — gateway health unconfirmed",
                "Restore deployment read access or pass --gateway-url <url>.",
            )
        url = env.get("AI_GATEWAY_URL", "")
        if not url:
            # A readable aiwb-ui with no AI_GATEWAY_URL is a confirmed "no inference gateway is
            # wired to this workbench" — there is no gateway to be broken, so this is not-applicable
            # (like cilium-mtu on a non-Cilium cluster), not unproven health. But if the caller
            # supplied a key to validate, they expect a gateway; not finding one is then inconclusive.
            if gateway_key:
                return CheckResult(
                    n, INCONCLUSIVE,
                    "a gateway key was supplied but no inference gateway address could be discovered (aiwb-ui has no AI_GATEWAY_URL) — cannot validate the key",
                    "Pass --gateway-url <url> alongside --gateway-key.",
                )
            return CheckResult(
                n, OK,
                "no inference gateway is configured for this workbench (aiwb-ui has no AI_GATEWAY_URL) — no gateway auth to probe",
            )
    # A scheme-less value (including a bare host:port, which urlparse would misread as a scheme)
    # defaults to https; but a value carrying a NON-http(s) scheme (e.g. file://) must be refused,
    # not passed to the opener — its FileHandler yields a status-less response, and int(None) would
    # raise out of the eager run_all_checks list and destroy the --json contract. A real scheme is
    # exactly the part before "://"; a bare host:port has no "://".
    if url.lower().startswith(("http://", "https://")):
        pass
    elif "://" in url:
        return CheckResult(
            n, INCONCLUSIVE,
            f"the inference gateway address has an unsupported scheme '{url.split('://', 1)[0]}://' — only http/https can be probed",
            "Set --gateway-url / AI_GATEWAY_URL to an http(s) URL.",
        )
    else:
        url = f"https://{url}"
    endpoint = f"{url.rstrip('/')}{_GATEWAY_MODELS_PATH}"

    probe: Any = _http_status
    vantage = ""
    status, err = _retry_unanswered(lambda: _http_status(endpoint, _GATEWAY_INVALID_KEY))
    if status is None and isinstance(err, _Silence):
        # This host may simply have no route to a public gateway the cluster reaches fine; the
        # E2E lanes call it from inside the cluster, so that answer is the one that decides. Only
        # pure silence qualifies: a certificate, lookup or connection failure on any attempt is
        # the route answering badly, and an answer from elsewhere must not stand in for it.
        in_status, in_err = _in_cluster_http_status(endpoint, _GATEWAY_INVALID_KEY)
        if in_status is None and _is_certificate_failure(in_err):
            return _in_cluster_certificate_warning(n, endpoint, f"gave this host no answer ({err}), and", in_err)
        if in_status is None:
            err = f"{err}; not from inside the cluster either ({in_err})"
        else:
            status, probe = in_status, _in_cluster_http_status
            vantage = f" — answered from inside the cluster ({_IN_CLUSTER_PROBE_WORKLOAD}); this host got no answer"
    result = _gateway_auth_verdict(n, endpoint, status, err, probe, gateway_key)
    return replace(result, verdict=result.verdict + vantage) if vantage else result

def _is_certificate_failure(err: str) -> bool:
    return "certificate_verify_failed" in err.lower().replace(" ", "_")

def _in_cluster_certificate_warning(n: str, endpoint: str, context: str, in_err: str) -> CheckResult:
    """The in-cluster probe pod rejected the gateway's certificate. The pod trusts only public CAs,
    not whatever CA the cluster's inference callers are given, so a gateway signed by a private CA
    fails here while serving them fine: the answer is unproven, not a fault. Either probe of the
    gateway can meet it."""
    workload = _IN_CLUSTER_PROBE_WORKLOAD
    return CheckResult(
        n, INCONCLUSIVE,
        f"the inference gateway at {endpoint} {context} from inside the cluster {workload}, which trusts only public CAs, rejected its certificate "
        f"({in_err}); in-cluster inference calls fail the same way unless their callers trust the gateway's CA",
        "Check the gateway's certificate. If a private CA signs it, confirm the workloads calling the gateway trust that CA.",
    )

def _gateway_auth_verdict(n: str, endpoint: str, status: int | None, err: str, probe: Any,
                          gateway_key: str | None) -> CheckResult:
    """gateway-auth's verdict from the invalid-key answer; `probe` sends the valid-key call from the
    same vantage point that answered."""
    if status is None:
        # A TLS verification failure is genuinely ambiguous from here: the gateway's certificate may
        # be bad (a real problem), or this host may simply not trust the issuing CA (not one). Saying
        # which would be a guess, so name the ambiguity and how to settle it rather than warning.
        if _is_certificate_failure(err):
            return CheckResult(
                n, INCONCLUSIVE,
                f"the inference gateway at {endpoint} presented a certificate this host does not trust ({err}) — that is either a gateway certificate problem or a missing CA here, and the check cannot tell which",
                "Re-run where the cluster CA is trusted (or compare with `curl -v` from the E2E runner): if the certificate is genuinely stale, inference tests will fail.",
            )
        return CheckResult(n, INCONCLUSIVE, f"the inference gateway at {endpoint} could not be reached: {err} — gateway health unconfirmed",
                           "Confirm the gateway URL is reachable from here or from inside the cluster (the probe there runs "
                           f"in a pod of its own in {_IN_CLUSTER_PROBE_NAMESPACE} and needs `create pods` and `create pods/attach` there).")
    if status >= 500:
        return CheckResult(
            n, WARN,
            f"the inference gateway is not serving requests correctly — it answered {status} to an invalid key instead of refusing it (a certificate/upstream failure, not an auth decision)",
            "Check the gateway's certificate and upstream health before the run; inference tests will fail unexplained.",
        )
    if status == 404 and _no_inference_routes_published():
        # Envoy answers 404 when no route matches, BEFORE ext_authz runs — so with nothing published
        # (no model deployed) the invalid key never reaches an auth decision. That is not broken
        # enforcement; there is simply nothing to protect yet. Report it as not-applicable, but only
        # on the confirmed zero-route state above, never on the bare 404.
        if gateway_key:
            # A supplied key means the caller expects to call a model through this gateway. With no
            # routes it cannot be served, so the key cannot be validated: that is unproven, not n/a.
            return CheckResult(
                n, INCONCLUSIVE,
                f"a gateway key was supplied but the inference gateway publishes no routes — {endpoint} answers 404 before auth runs, so the key cannot be validated",
                "Deploy a model (so a route exists) before validating a gateway key, or omit --gateway-key.",
            )
        return CheckResult(
            n, OK,
            f"the inference gateway publishes no routes to any ext_authz-protected gateway — {endpoint} answers 404 before auth runs, so there is no gateway auth to probe until a model is deployed",
        )
    if status != 401:
        return CheckResult(
            n, WARN,
            f"the inference gateway did not refuse an invalid key — it answered {status} where 401 was expected, so key enforcement is not behaving as the tests assume",
            "Check the gateway's auth configuration (ext_authz) before relying on inference test results.",
        )
    if not gateway_key:
        return CheckResult(n, OK, f"the inference gateway refuses an invalid key cleanly (401 from {endpoint})")

    # The in-cluster probe retries inside the workload already.
    ok_status, ok_err = probe(endpoint, gateway_key) if probe is _in_cluster_http_status else _retry_unanswered(lambda: probe(endpoint, gateway_key))
    if ok_status is None and probe is _in_cluster_http_status and _is_certificate_failure(ok_err):
        return _in_cluster_certificate_warning(n, endpoint, "refuses invalid keys, but on the valid-key call", ok_err)
    if ok_status is None:
        return CheckResult(n, INCONCLUSIVE, f"invalid keys are refused, but the valid-key probe could not reach {endpoint}: {ok_err}")
    if ok_status >= 500:
        return CheckResult(
            n, WARN,
            f"the inference gateway refuses invalid keys but answered {ok_status} to a valid key — it is failing to serve authorised requests",
            "Check the gateway's upstream/model routing before the run.",
        )
    # 203 is the ext_authz short-circuit the gateway returns for an authorised models call.
    if ok_status not in (200, 203):
        return CheckResult(
            n, WARN,
            f"the inference gateway rejected the supplied valid key ({ok_status}) — authorised inference calls will fail",
            "Confirm the key is current and authorised for this gateway.",
        )
    return CheckResult(n, OK, f"the inference gateway refuses an invalid key (401) and serves a valid one ({ok_status}) at {endpoint}")

def _issuer_at(url: str, cluster_domain: str = "cluster.local", allow_private: bool = False,
               public_url: str | None = None, public_allow_private: bool = False,
               internal_url: str | None = None, via: list[str] | None = None) -> tuple[str | None, str]:
    """The issuer Keycloak's discovery document names for the realm at `base`, fetched the way the
    workload fetches it (in-cluster URLs through the service proxy). (issuer, error).

    When the proxy read cannot be delivered (no services/proxy permission, no ready endpoint) and the
    caller supplies the workload's public issuer URL, the document is read from there instead, so an
    identity that may not proxy to Keycloak still gets a verdict. That read proves less than the
    in-cluster one: Keycloak may name the issuer after the address it was asked on, in which case
    this read cannot fail, so a match means "no contradiction found", not "consistent", and says
    nothing about which Keycloak the workload reaches."""
    target = _cluster_service_target(url, cluster_domain)
    status, body, err = _service_proxy_get(*target) if target else _retry_unanswered(lambda: _http_get(url, allow_private=allow_private))
    # The public counterpart is only a stand-in when it names the same realm and path as the URL the
    # workload reads; a differently configured URL (a stale realm) must not be replaced by the expected one.
    def _path(u: str) -> str | None:
        try:
            return urllib.parse.urlsplit(u).path
        except ValueError:
            return None

    # A public URL the operator's host cannot fetch (it resolves to a private address from there, or
    # it never answers, as when this host has no route to it) is read through the in-cluster Service
    # instead, when the workload's own internal URL for the same realm is known and the identity may
    # proxy to it. The workloads read it in-cluster too, so a runner without public egress must not
    # leave a healthy cluster unproven.
    refused = re.search(r"resolves to non-public address ([0-9A-Fa-f:.]+(?:, [0-9A-Fa-f:.]+)*)", err) if status is None else None
    try:
        refused_private = refused is not None and all(
            any(_fold_embedded_ipv4(ipaddress.ip_address(a)) in net for net in _PRIVATE_UNICAST) for a in refused.group(1).split(", "))
    except ValueError:
        refused_private = False
    # Only silence counts as no route: a failed lookup or handshake may be a mistyped host or a bad
    # certificate on the public route, which reading another document would hide. _Silence is set
    # only when every address on every attempt timed out, so one such failure rules this out.
    unanswered = status is None and refused is None and isinstance(err, _Silence)
    if (refused_private or unanswered) and not target and internal_url and _path(internal_url) is not None and _path(internal_url) == _path(url):
        inner = _cluster_service_target(internal_url, cluster_domain)
        if inner:
            i_status, i_body, i_err = _service_proxy_get(*inner)
            if i_status == 200:
                status, body, err, url = i_status, i_body, "", internal_url
                if via is not None:
                    via.append("in-cluster")
            else:
                combined = f"{_one_line(err)}; the in-cluster document {internal_url} also {_answered(i_status, i_err) if i_status is not None else 'could not be read'}: {_one_line(i_err)}"
                return None, f"the discovery document at {url} could not be read: {combined}"
    if status is None and target and public_url and _path(public_url) is not None and _path(public_url) == _path(url):
        public_status, public_body, public_err = _retry_unanswered(lambda: _http_get(public_url, allow_private=public_allow_private))
        if public_status == 200:
            status, body, err, url = public_status, public_body, "", public_url
            if via is not None:
                via.append("public")
        else:
            combined = f"{_one_line(err)}; the public issuer URL {public_url} also {_answered(public_status, public_err) if public_status is not None else 'could not be read'}: {_one_line(public_err)}"
            return None, f"the discovery document at {url} could not be read: {combined}"
    if status is None and target and public_url and _path(public_url) is None:
        return None, f"the discovery document at {url} could not be read: {_one_line(err)}; the public issuer URL {_one_line(repr(public_url))} is not a valid URL"
    if status != 200:
        answer = _answered(status, err) if status is not None else "could not be read"
        return None, f"the discovery document at {url} {answer}: {_one_line(err)}"
    try:
        issuer = _dict(json.loads(body)).get("issuer")
    except json.JSONDecodeError:
        issuer = None
    if not isinstance(issuer, str) or not issuer:
        return None, f"the discovery document at {url} names no issuer"
    return issuer, ""

# The chart's name for the container that runs the product in each Deployment.
_PRODUCT_CONTAINER = {"airm-ui": "airm-ui", "aiwb-ui": "aiwb-ui", "airm-api": "airm", "aiwb-api": "aiwb"}

def _deploy_container_env(name: str, namespace: str) -> tuple[str, dict[str, str], set[str], bool, str | None, str]:
    """(state, literal env, names set by reference, has envFrom, pod selector, error) of the
    Deployment's product container. state is present / absent / unreadable. A value set through
    valueFrom or envFrom is not visible here, so callers must not treat such a variable as unset."""
    ok, obj, err = kubectl_json(["get", "deploy", name, "-n", namespace])
    if not ok:
        return ("absent" if _is_not_found(err) else "unreadable"), {}, set(), False, None, err
    env: dict[str, str] = {}
    indirect: set[str] = set()
    pod_spec = _dict(_dict(_dict(_dict(obj).get("spec")).get("template")).get("spec"))
    containers = [_dict(c) for c in _list(pod_spec.get("containers"))]
    # Only the product container's configuration is the product's; a sidecar's env says nothing
    # about it and must neither fill in nor overwrite a value.
    # Even a lone container must carry the chart's name: otherwise it may be a sidecar left behind
    # while the product container is gone, and its env would be read as the product's.
    containers = [c for c in containers if c.get("name") == _PRODUCT_CONTAINER.get(name)]
    if len(containers) != 1:
        return "unreadable", {}, set(), False, None, f"no single container named {_PRODUCT_CONTAINER.get(name)!r} to read the product's configuration from"
    c = containers[0]
    env_from = bool(_list(c.get("envFrom")))
    for e in _list(c.get("env")):
        if not isinstance(e, dict) or not e.get("name"):
            continue
        if e.get("valueFrom") is not None:
            indirect.add(e["name"])
        else:
            # An entry without a value sets the variable to "", which the product reads as set.
            env[e["name"]] = str(e.get("value") or "")
    selector = _selector_arg(_dict(_dict(obj).get("spec")).get("selector"))
    return "present", env, indirect, env_from, selector, ""

_UI_ISSUER_VARS = ("OIDC_ISSUER", "KEYCLOAK_ISSUER", "OIDC_ISSUER_INTERNAL_URL", "KEYCLOAK_ISSUER_INTERNAL_URL")

_API_ISSUER_VARS = ("OIDC_ISSUER", "KEYCLOAK_PUBLIC_URL", "KEYCLOAK_REALM", "OIDC_JWKS_URL", "KEYCLOAK_INTERNAL_URL", "OPENID_CONFIGURATION_URL")

_AIRM_API_ISSUER_VARS = ("KEYCLOAK_PUBLIC_URL", "KEYCLOAK_REALM", "KEYCLOAK_INTERNAL_URL", "OPENID_CONFIGURATION_URL")

_JWKS_SUFFIX = "/protocol/openid-connect/certs"

def _effective_issuer(role: str, env: dict[str, str]) -> tuple[str | None, str | None, str | None]:
    """(issuer the workload enforces, realm URL it fetches discovery from, configuration problem).

    Mirrors the product code: the UIs read OIDC_ISSUER, then KEYCLOAK_ISSUER; AIWB's API reads
    OIDC_ISSUER, else KEYCLOAK_PUBLIC_URL/realms/KEYCLOAK_REALM; AIRM's API ignores the OIDC_* values
    and takes its realm from KEYCLOAK_INTERNAL_URL and hands clients (Swagger, generated kubeconfigs)
    KEYCLOAK_PUBLIC_URL/realms/KEYCLOAK_REALM. Unset values fall back to built-in localhost defaults."""
    # The APIs read these with os.getenv(name, default), so one set to "" is used as "" (a realm
    # URL like "/realms/airm"), not replaced by the default; OIDC_ISSUER and OIDC_JWKS_URL are read
    # with `or` and do fall back, as the UIs' `||` does for theirs. AIWB's API only builds its token
    # issuer and key URL from a Keycloak setting when the generic OIDC value is absent; Swagger's
    # default URL uses them regardless, and _swagger_discovery judges that one.
    if role in ("api", "airm-api"):
        used = {"KEYCLOAK_PUBLIC_URL", "KEYCLOAK_INTERNAL_URL", "KEYCLOAK_REALM"}
        if role == "api":
            used = ({"KEYCLOAK_PUBLIC_URL", "KEYCLOAK_REALM"} if not env.get("OIDC_ISSUER") else set()) | (
                {"KEYCLOAK_INTERNAL_URL", "KEYCLOAK_REALM"} if not env.get("OIDC_JWKS_URL") else set())
        emptied = [v for v in ("KEYCLOAK_PUBLIC_URL", "KEYCLOAK_INTERNAL_URL", "KEYCLOAK_REALM") if v in used and env.get(v) == ""]
        if emptied:
            return None, None, f"sets {' and '.join(emptied)} to an empty value, which the product uses as is instead of a Keycloak URL or realm"
    oidc = env.get("OIDC_ISSUER")
    if role == "airm-api":
        airm_realm = env.get("KEYCLOAK_REALM") or "airm"
        kc_public, kc_internal = env.get("KEYCLOAK_PUBLIC_URL"), env.get("KEYCLOAK_INTERNAL_URL")
        if not kc_public or not kc_internal:
            missing = " and ".join(v for v, val in (("KEYCLOAK_PUBLIC_URL", kc_public), ("KEYCLOAK_INTERNAL_URL", kc_internal)) if not val)
            return None, None, (
                f"sets no {missing}, so it uses its built-in default http://localhost:8080 for Keycloak "
                f"(realm {airm_realm})"
            )
        # Verbatim, as the product builds it: a trailing slash on KEYCLOAK_PUBLIC_URL reaches clients.
        return f"{kc_public}/realms/{airm_realm}", f"{kc_internal.rstrip('/')}/realms/{airm_realm}", None
    if role == "ui":
        issuer = oidc or env.get("KEYCLOAK_ISSUER")
        if not issuer:
            return None, None, (
                "sets neither OIDC_ISSUER nor KEYCLOAK_ISSUER, so its issuer is whatever the image "
                "supplies (a development default is http://localhost:8080/realms/airm)"
            )
        base = env.get("OIDC_ISSUER_INTERNAL_URL") or env.get("KEYCLOAK_ISSUER_INTERNAL_URL") or issuer
        # The UIs trim slashes from both their issuer and the discovery document's before comparing.
        return issuer.rstrip("/"), base, None
    # api_common.auth.config: every unset value has a built-in localhost/platform default, and the
    # JWKS URL (the realm whose keys it trusts) derives from KEYCLOAK_INTERNAL_URL independently of
    # OIDC_ISSUER. Relying on either localhost default is a broken deployment, not a guess to verify.
    public, realm = env.get("KEYCLOAK_PUBLIC_URL"), env.get("KEYCLOAK_REALM") or "platform"
    jwks, internal = env.get("OIDC_JWKS_URL"), env.get("KEYCLOAK_INTERNAL_URL")
    if not oidc and not public:
        return None, None, (
            "sets neither OIDC_ISSUER nor KEYCLOAK_PUBLIC_URL, so it checks tokens against its built-in "
            f"default issuer (http://localhost:8080/realms/{realm})"
        )
    if not jwks and not internal:
        return None, None, (
            "sets neither OIDC_JWKS_URL nor KEYCLOAK_INTERNAL_URL, so it fetches signing keys from its "
            f"built-in default http://localhost:8080/realms/{realm}"
        )
    # Verbatim, as the product builds them: PyJWT compares the configured issuer with `iss` exactly,
    # and a trailing slash on KEYCLOAK_INTERNAL_URL reaches the key URL as "//realms".
    issuer = oidc or f"{public}/realms/{realm}"
    jwks = jwks or f"{internal}/realms/{realm}{_JWKS_SUFFIX}"
    # A key URL in Keycloak's realm layout locates the discovery document served beside the keys; the
    # caller compares the enforced issuer with the one that document names. That compares the API
    # with whatever serves its signing keys, Keycloak or not, which is the pairing that must agree.
    # A key URL in any other layout (Dex serves /keys) locates no document: base None, unproven.
    realm_url = jwks[: -len(_JWKS_SUFFIX)] if jwks.endswith(_JWKS_SUFFIX) else None
    return issuer, realm_url, None

# The UI's own report of a discovery document that names another issuer than the one it enforces.
_LOGGED_ISSUER_MISMATCH = re.compile(r"names the issuer ([^\s,]+), but OIDC_ISSUER is ([^\s,\"']+)")

_DISCOVERY_SUFFIX = "/.well-known/openid-configuration"

def _is_loopback_url(url: str) -> bool:
    try:
        host = urllib.parse.urlsplit(url).hostname or ""
    except ValueError:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False

def _swagger_discovery(role: str, env: dict[str, str], issuer: str,
                       issuer_at: Any = None, internal_url: str | None = None) -> tuple[str | None, str | None]:
    """(problem, unproven) for the discovery document an API hands its Swagger login.

    Both APIs hand Swagger OPENID_CONFIGURATION_URL, defaulting to the discovery document of
    KEYCLOAK_PUBLIC_URL (default http://localhost:8080)/realms/KEYCLOAK_REALM independently of the
    issuer they validate, so a stale value logs Swagger users in to another realm while every other
    check agrees. The document is read even when its URL is spelled like the issuer's: Swagger
    fetches it from wherever that URL leads (often a public proxy the token check never uses), and a
    different URL is fine when its document names the same issuer."""
    if role not in ("api", "airm-api"):
        return None, None
    # The APIs read these with os.getenv(name, default): a variable set to "" is used as "", not
    # replaced by the default, so only an absent one falls back.
    url = env.get("OPENID_CONFIGURATION_URL")
    if url is None:
        public = env.get("KEYCLOAK_PUBLIC_URL", "http://localhost:8080")
        realm = env.get("KEYCLOAK_REALM", "airm" if role == "airm-api" else "platform")
        url = f"{public}/realms/{realm}{_DISCOVERY_SUFFIX}"
    try:
        usable = bool(urllib.parse.urlsplit(url).hostname)
    except ValueError:
        usable = False  # e.g. an unmatched IPv6 bracket: no host to fetch from either
    if not usable:
        return f"hands Swagger's login no usable discovery URL ({url!r})", None
    if _is_loopback_url(url):
        return f"points Swagger's login at {url}, which is on the browser's own machine, not Keycloak", None
    public_discovery = f"{issuer.rstrip('/')}{_DISCOVERY_SUFFIX}"
    found, err = (issuer_at or _issuer_at)(url, public_url=public_discovery, internal_url=internal_url if url == public_discovery else None)
    if found is None:
        return None, f"cannot read the discovery document it hands Swagger: {err}"
    if found != issuer:
        return f"points Swagger's login at {url}, whose issuer is {found}, not {issuer}", None
    return None, None

def check_oidc_issuer_consistency(recent_minutes: int = 30, cluster_domain: str = "cluster.local",
                                  trusted_private_hosts: frozenset[str] = frozenset()) -> CheckResult:
    """Every UI/API must enforce the issuer Keycloak's discovery document names for its realm; a
    UI whose issuer disagrees fails every login with a 500 while its pod stays Running.

    The Deployment shows only what the chart sets. A UI image can carry its own OIDC_ISSUER that
    wins over the chart's KEYCLOAK_ISSUER fallback, which no Deployment read reveals, so a UI's recent
    log report of the issuer it actually enforces overrides the Deployment-derived value."""
    n = "oidc-issuer-consistency"
    problems: list[str] = []
    # Swagger's discovery URL is configured independently of the issuer a workload enforces, so its
    # problems need their own sentence and advice: fixing the issuer leaves them unchanged.
    swagger_problems: list[str] = []
    unproven: list[str] = []
    on_unproven_fallback = False
    confirmed: list[str] = []
    issuers: set[str] = set()
    discovery: dict[tuple[str, str | None, str | None], tuple[str | None, str]] = {}
    read_via: dict[tuple[str, str | None, str | None], list[str]] = {}
    reads: list[list[str]] = []
    confirmed_public: list[str] = []
    confirmed_swagger_in_cluster: list[str] = []

    def issuer_at(url: str, public_url: str | None = None, internal_url: str | None = None) -> tuple[str | None, str]:
        # One read per (discovery URL, public counterpart) per run; the Swagger URL often is the realm URL already read.
        key = (url, public_url, internal_url)
        if key not in discovery:
            via: list[str] = []
            def trusted(u: str | None) -> bool:
                try:
                    return (urllib.parse.urlsplit(u or "").hostname or "") in trusted_private_hosts
                except ValueError:
                    return False
            discovery[key] = _issuer_at(url, cluster_domain, allow_private=trusted(url), public_url=public_url, public_allow_private=trusted(public_url), internal_url=internal_url, via=via)
            read_via[key] = via
        reads.append(read_via[key])
        return discovery[key]

    for name, ns, role in _PRODUCT_DEPLOYMENTS:
        state, env, indirect, env_from, selector, err = _deploy_container_env(name, ns)
        if state == "absent":
            continue
        if state == "unreadable":
            unproven.append(f"{ns}/{name}: cannot read the deployment ({_one_line(err)})")
            continue
        relevant = {"ui": _UI_ISSUER_VARS, "api": _API_ISSUER_VARS, "airm-api": _AIRM_API_ISSUER_VARS}[role]
        hidden = sorted(indirect.intersection(relevant))
        # A literal env entry wins over envFrom, but any issuer input left unset literally may come
        # from an envFrom source this check does not resolve.
        via_env_from = sorted(v for v in relevant if v not in env and v not in indirect) if env_from else []
        if hidden or via_env_from:
            what = "; ".join(filter(None, [", ".join(hidden), f"envFrom may set {', '.join(via_env_from)}" if via_env_from else ""]))
            unproven.append(f"{ns}/{name}: sets its issuer through a reference this check does not resolve ({what})")
            continue
        issuer, base, problem = _effective_issuer(role, env)
        if problem:
            problems.append(f"{ns}/{name} {problem}")
        if issuer is None:
            continue
        if base is None:
            unproven.append(f"{ns}/{name}: its signing-key URL is not a Keycloak realm URL, so the realm it trusts cannot be located to compare issuers")
            continue
        # On the KEYCLOAK_ISSUER fallback the Deployment cannot show whether the image's own
        # OIDC_ISSUER (and OIDC_ISSUER_INTERNAL_URL) override it. A logged mismatch proves a
        # problem, but no log line proves the reverse: an image pointing at another, self-consistent
        # server logs nothing. So a fallback UI is never confirmed. A literal OIDC_ISSUER wins over
        # any image default, so for it the log only adds findings.
        on_fallback = role == "ui" and "OIDC_ISSUER" not in env
        if on_fallback and not selector:
            on_unproven_fallback = True
            unproven.append(f"{ns}/{name}: relies on KEYCLOAK_ISSUER and has no pod selector to read the log that would show an image override")
            continue
        if role == "ui" and selector:
            logs_ok, logs, lerr, complete = _recent_logs(ns, selector, recent_minutes)
            if not logs_ok and on_fallback:
                on_unproven_fallback = True
                unproven.append(f"{ns}/{name}: relies on KEYCLOAK_ISSUER, and its log, the only evidence of an image override, cannot be read ({_one_line(lerr)})")
                continue
            logged = _LOGGED_ISSUER_MISMATCH.findall(logs) if logs_ok else []
            if logged:
                published, enforced = logged[-1]
                problems.append(
                    f"{ns}/{name} logs that it enforces issuer {enforced.rstrip('/')} while Keycloak's discovery "
                    f"document names {published.rstrip('/')}"
                    + ("" if "OIDC_ISSUER" in env else " (the image's own OIDC_ISSUER overrides the Deployment's KEYCLOAK_ISSUER)")
                )
                continue
            if on_fallback and not complete:
                on_unproven_fallback = True
                unproven.append(f"{ns}/{name}: relies on KEYCLOAK_ISSUER, and its log in the last {recent_minutes}m exceeds {_LOG_TAIL_LINES} lines, so an image override logged earlier may be unread")
                continue
        if on_fallback:
            on_unproven_fallback = True
            unproven.append(
                f"{ns}/{name}: relies on KEYCLOAK_ISSUER, which the image's own OIDC_ISSUER overrides unseen; "
                "no mismatch is logged, but that cannot show which issuer it enforces (set OIDC_ISSUER on the Deployment)"
            )
            continue
        if _is_loopback_url(base):
            # Inside the pod, localhost is the pod itself; nothing there serves discovery.
            problems.append(f"{ns}/{name} fetches its issuer's discovery document from {base}, which inside the pod is the pod itself")
            continue
        reads.clear()
        found, derr = issuer_at(f"{base.rstrip('/')}{_DISCOVERY_SUFFIX}", f"{issuer.rstrip('/')}{_DISCOVERY_SUFFIX}")
        if found is None:
            unproven.append(f"{ns}/{name}: {derr}")
            continue
        issuers.add(found)
        if issuer != (found.rstrip("/") if role == "ui" else found):
            verb = "hands clients issuer" if role == "airm-api" else "enforces issuer"
            problems.append(f"{ns}/{name} {verb} {issuer} but Keycloak's discovery document names {found}")
            continue
        swagger, swagger_unproven = _swagger_discovery(role, env, issuer, issuer_at, internal_url=f"{base.rstrip('/')}{_DISCOVERY_SUFFIX}")
        if swagger:
            swagger_problems.append(f"{ns}/{name} {swagger}")
        elif swagger_unproven:
            unproven.append(f"{ns}/{name}: {swagger_unproven}")
        else:
            # _effective_issuer never returns both an issuer and a problem, so this one is sound.
            if any("public" in v for v in reads):
                confirmed_public.append(f"{ns}/{name}")
            elif any("in-cluster" in v for v in reads):
                confirmed_swagger_in_cluster.append(f"{ns}/{name}")
            else:
                confirmed.append(f"{ns}/{name}")
    fallback_advice = ("Set OIDC_ISSUER on each UI Deployment that relies on KEYCLOAK_ISSUER, so the issuer it "
                       "enforces is visible and cannot be overridden by the image.")
    if problems or swagger_problems:
        causes: list[str] = []
        advice: list[str] = []
        if problems:
            causes.append(f"OIDC issuer configuration will break sign-in: {'; '.join(problems)}")
            advice.append(
                "Make each workload enforce exactly the issuer Keycloak publishes for the realm (its "
                ".well-known/openid-configuration `issuer`), reachable in-cluster: a UI reads OIDC_ISSUER "
                "and OIDC_ISSUER_INTERNAL_URL, the AIWB API OIDC_ISSUER and OIDC_JWKS_URL, and the AIRM "
                "API KEYCLOAK_PUBLIC_URL, KEYCLOAK_INTERNAL_URL and KEYCLOAK_REALM (it ignores OIDC_*). "
                "A mismatch makes the UI's /api/auth/* answer 500 and the API reject every token.")
        if swagger_problems:
            causes.append(f"Swagger's login is handed the wrong discovery document: {'; '.join(swagger_problems)}")
            advice.append(
                "Set OPENID_CONFIGURATION_URL on the API to the public discovery document of the realm it "
                "validates (unset, it derives from KEYCLOAK_PUBLIC_URL and KEYCLOAK_REALM); the issuer "
                "settings do not change what Swagger is handed.")
        if on_unproven_fallback:
            advice.append(fallback_advice)
        caveat = f" (also unconfirmed: {'; '.join(unproven)})" if unproven else ""
        return CheckResult(n, WARN, f"{'; and '.join(causes)}{caveat}", " ".join(advice))
    if unproven:
        steps = ["Restore read access to the deployments, and grant the checking identity `get` on "
                  "`services/proxy` in the Keycloak namespace so it can read the discovery document the "
                  "workload reads, then re-run."]
        if on_unproven_fallback:
            steps.insert(0, fallback_advice)
        refused = sorted(set(re.findall(r"(\S+) resolves to non-public address", "; ".join(unproven))))
        if refused:
            steps.append(f"The operator's host resolves {', '.join(refused)} to a non-public address, so the discovery document was not "
                         "fetched; if that is the cluster's own Keycloak on a private address, re-run with "
                         + " ".join(f"--trusted-private-host {h}" for h in refused) + " to read it.")
        return CheckResult(n, INCONCLUSIVE, f"cannot confirm the OIDC issuer of: {'; '.join(unproven)}", " ".join(steps))
    if not (confirmed or confirmed_public or confirmed_swagger_in_cluster):
        return CheckResult(n, INCONCLUSIVE, "no AIRM/AIWB UI or API deployment found — no issuer to check (expected before install)")
    parts = []
    if confirmed:
        parts.append(f"{', '.join(confirmed)} enforce the issuer Keycloak's discovery names ({', '.join(sorted(issuers))})")
    if confirmed_public:
        parts.append(f"{', '.join(confirmed_public)}: no contradiction found in the public issuer document "
                     f"({', '.join(sorted(issuers))}); it was read from the public URL, not in-cluster, so it does not show which Keycloak the workload reaches")
    if confirmed_swagger_in_cluster:
        parts.append(f"{', '.join(confirmed_swagger_in_cluster)}: Swagger's discovery URL could not be reached from here (its host "
                     "resolves to a private address or did not answer), so the in-cluster document was read instead; the public route Swagger uses was not checked")
    return CheckResult(n, OK, "; ".join(parts))

def _nextauth_answer_problem(path: str, body: str) -> str:
    """Why a successful answer is still not NextAuth's, or "" when it is. Both routes return a JSON
    object; a signed-out session is legitimately `{}`, but providers with none configured leaves
    the sign-in page nothing to offer."""
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        return f"answered success without the JSON object NextAuth sends ({_one_line(body, 120) or 'empty body'})"
    # Each NextAuth provider entry names its id and the URL that starts its sign-in; any other
    # object (an error payload, a proxy's JSON) offers the sign-in page nothing.
    if path.endswith("/providers") and not any(
            isinstance(p, dict) and all(isinstance(p.get(k), str) and p[k].strip() for k in ("id", "signinUrl"))
            for p in parsed.values()):
        return "answered success but lists no sign-in provider"
    return ""

def _ui_proxy_get(namespace: str, service: str, path: str) -> tuple[int | None, str, str]:
    """_service_proxy_get for a UI Service, naming the port when the bare Service name finds no endpoint.

    The API server matches a proxy request that names no port against endpoint ports with an empty
    name, so a Service whose port is named (`web`) answers "no endpoints available" while its pod
    is Running and its Endpoints are populated."""
    status, body, err = _service_proxy_get(namespace, service, path)
    if status is not None or "no endpoints available" not in err.lower():
        return status, body, err
    ok, svc, _ = kubectl_json(["get", "service", service, "-n", namespace])
    ports = _list(_dict(_dict(svc).get("spec")).get("ports")) if ok else []
    if not ports:
        return status, body, err
    first = _dict(ports[0])
    port = first.get("name") or first.get("port")
    if not port:
        return status, body, err
    return _service_proxy_get(namespace, f"{service}:{port}", path)

def check_ui_auth_endpoints() -> CheckResult:
    """The UIs' auth routes must answer 200 with NextAuth's JSON (providers listing a provider), and
    an installed UI must have the Service that routes them. A UI whose login is broken still serves
    its homepage, so pod readiness and a homepage probe both pass it; a UI whose Service is gone has
    pods that nothing can reach."""
    n = "ui-auth-endpoints"
    problems: list[str] = []
    unrouted: list[str] = []
    unproven: list[str] = []
    answered: list[str] = []
    for name, ns, role in _PRODUCT_DEPLOYMENTS:
        if role != "ui":
            continue
        for path in _UI_AUTH_PATHS:
            status, body, err = _ui_proxy_get(ns, name, path)
            # kubectl reports any 2xx as success without its code; NextAuth answers both routes with a
            # JSON object, so an empty or non-JSON 2xx (e.g. 204) is not a working route.
            if status == 200 and (why := _nextauth_answer_problem(path, body)):
                problems.append(f"{ns}/{name}{path} {why}")
                continue
            if status is None:
                if _SERVICE_ABSENT.search(err):
                    # Only a missing Deployment means the UI is not installed; a Deployment whose
                    # Service is gone serves nobody's sign-in.
                    dok, _, derr = kubectl_json(["get", "deploy", name, "-n", ns])
                    if dok:
                        unrouted.append(f"{ns}/{name}")
                    elif not _is_not_found(derr):
                        unproven.append(f"{ns}/{name}: its Service is missing and its Deployment cannot be read ({_one_line(derr)})")
                    break
                unproven.append(f"{ns}/{name}{path}: request not delivered to the UI ({_one_line(err)})")
            elif status != 200:
                # Both routes answer 200 to an anonymous caller; a 4xx means the auth stack is not
                # serving them (a missing route, a proxy or policy refusing it), which breaks sign-in
                # just as a 5xx does.
                problems.append(f"{ns}/{name}{path} {_answered(status, err)} ({_one_line(err)})")
            else:
                answered.append(f"{ns}/{name}{path}={status}")
    if problems or unrouted:
        # Each cause gets its own sentence and advice: an unrouted UI answered nothing, so the
        # sign-in-error diagnosis would send the operator to logs that cannot show it.
        found: list[str] = []
        advice: list[str] = []
        if problems:
            found.append(f"the auth endpoints answer with an error: {'; '.join(problems)}")
            advice.append("A UI whose auth routes fail lets nobody log in although its pod is Running and its homepage "
                          "loads; read its log (app-startup-errors) — an OIDC issuer that disagrees with Keycloak is "
                          "the known cause (oidc-issuer-consistency).")
        if unrouted:
            found.append(f"the Deployment is installed but its Service is missing, so nothing reaches it: {', '.join(unrouted)}")
            advice.append("A UI with no Service is unreachable; restore the Service from the chart (a failed or "
                          "partial helm upgrade can drop it).")
        caveat = f" (also unconfirmed: {'; '.join(unproven)})" if unproven else ""
        return CheckResult(n, WARN, f"UI sign-in is broken — {'; and '.join(found)}{caveat}", " ".join(advice))
    if unproven:
        return CheckResult(n, INCONCLUSIVE, f"cannot confirm the UI auth endpoints: {'; '.join(unproven)}",
                           "Grant the checking identity `get` on `services/proxy` in the airm and aiwb namespaces (a plain cluster-reader lacks it), or wait for the UI to have a ready endpoint, then re-run.")
    if not answered:
        return CheckResult(n, INCONCLUSIVE, "no AIRM/AIWB UI Service found — no sign-in to check (expected before install)")
    return CheckResult(n, OK, f"UI auth endpoints answer 200: {', '.join(answered)}")

# Lines the product code writes when sign-in cannot work: a discovery document that names another
# issuer or lacks the login endpoints. The UIs' other discovery messages (a transient fetch retry, an
# endpoint outside the issuer, logout proceeding locally) are not failures on their own.
_AUTH_LOG_SIGNATURES = ("OIDC discovery document from",)

# The token library's issuer rejection. One can be a client presenting another realm's token; an API
# whose own issuer is wrong rejects every token, so only a recurring rejection is a fault.
_ISSUER_REJECTION = "Invalid issuer"

_REPEATED_ISSUER_REJECTION_MIN = 3

_UNHANDLED_LOG = re.compile(r"unhandled|Traceback \(most recent call last\)", re.IGNORECASE)

# One unhandled error can be a client disconnect; a recurring one is a fault.
_REPEATED_UNHANDLED_MIN = 3

_LOG_TAIL_LINES = 10000

def _recent_logs(namespace: str, selector: str, recent_minutes: int) -> tuple[bool, str, str, bool]:
    """(read_ok, log text, error, complete) for every container of every pod matching the selector,
    fetched once per run. complete is False when a container logged more than the line cap inside
    the window, so its earliest lines in the window were not read."""
    def fetch() -> tuple[bool, str, str, bool]:
        rc, out, err = run([
            "kubectl", "logs", "-n", namespace, "-l", selector, "--all-containers", "--prefix",
            f"--since={recent_minutes}m", f"--tail={_LOG_TAIL_LINES}", "--max-log-requests=20",
        ], 60)
        per_container: dict[str, int] = {}
        for line in out.splitlines():
            prefix = line.split("] ", 1)[0] if line.startswith("[") else ""
            per_container[prefix] = per_container.get(prefix, 0) + 1
        if rc == 0 and not out.strip() and "No resources found" in err:
            return False, "", f"no pod matches {selector}", True
        return rc == 0, out, err, all(count < _LOG_TAIL_LINES for count in per_container.values())
    result: tuple[bool, str, str, bool] = _once_per_run(("logs", namespace, selector, recent_minutes), fetch)
    return result

def check_app_startup_errors(recent_minutes: int) -> CheckResult:
    """Scan the product UIs' and APIs' recent logs for sign-in errors and recurring unhandled errors
    that leave the pod Running but the product broken."""
    n = "app-startup-errors"
    problems: list[str] = []
    unproven: list[str] = []
    clean: list[str] = []
    quiet_ready: list[str] = []
    for name, ns, _role in _PRODUCT_DEPLOYMENTS:
        ok, obj, err = kubectl_json(["get", "deploy", name, "-n", ns])
        if not ok:
            if not _is_not_found(err):
                unproven.append(f"{ns}/{name}: cannot read the deployment ({_one_line(err)})")
            continue
        selector = _selector_arg(_dict(_dict(obj).get("spec")).get("selector"))
        if not selector:
            unproven.append(f"{ns}/{name}: no matchLabels pod selector to find its pods")
            continue
        logs_ok, out, lerr, complete = _recent_logs(ns, selector, recent_minutes)
        if not logs_ok:
            unproven.append(f"{ns}/{name}: cannot read logs ({_one_line(lerr)})")
            continue
        lines = out.splitlines()
        quiet = not out.strip()
        rejected = [line for line in lines if _ISSUER_REJECTION in line]
        auth = [line for line in lines if any(sig in line for sig in _AUTH_LOG_SIGNATURES)]
        if len(rejected) >= _REPEATED_ISSUER_REJECTION_MIN:
            auth += rejected
        unhandled = [line for line in lines if _UNHANDLED_LOG.search(line)]
        found = False
        if auth:
            problems.append(f"{ns}/{name}: {len(auth)} sign-in error line(s), latest: {_one_line(auth[-1], 600)}")
            found = True
        if len(unhandled) >= _REPEATED_UNHANDLED_MIN:
            problems.append(f"{ns}/{name}: {len(unhandled)} unhandled-error line(s), latest: {_one_line(unhandled[-1], 600)}")
            found = True
        if not found and not complete:
            unproven.append(f"{ns}/{name}: a container logged over {_LOG_TAIL_LINES} lines in the last {recent_minutes}m, so the earliest were not scanned")
        elif not found and quiet:
            # A pod can log nothing in the window (an idle service that has not restarted), leaving
            # nothing to scan; readiness is all that can be said, and the verdict states that limit.
            if _as_int(_dict(_dict(obj).get("status")).get("readyReplicas")) > 0:
                clean.append(f"{ns}/{name}")
                quiet_ready.append(f"{ns}/{name}")
            else:
                unproven.append(f"{ns}/{name}: no log line in the last {recent_minutes}m and no ready replica, so there is nothing to scan")
        elif not found:
            clean.append(f"{ns}/{name}")
    if problems:
        caveat = f" (also unconfirmed: {'; '.join(unproven)})" if unproven else ""
        return CheckResult(
            n, WARN,
            f"product logs from the last {recent_minutes}m show errors: {'; '.join(problems)}{caveat}",
            "An OIDC discovery / issuer error means sign-in fails (see oidc-issuer-consistency); recurring "
            "unhandled errors mean a request path is failing. Read the full log of the named deployment.",
        )
    if unproven:
        return CheckResult(n, INCONCLUSIVE, f"cannot confirm the product logs are clean: {'; '.join(unproven)}")
    if not clean:
        return CheckResult(n, INCONCLUSIVE, "no AIRM/AIWB UI or API deployment found — no logs to scan (expected before install)")
    limit = (f"; {', '.join(quiet_ready)} logged nothing in that window, so for it only readiness, not its log, was checked"
             if quiet_ready else "")
    return CheckResult(n, OK, f"no sign-in or recurring unhandled errors in the last {recent_minutes}m of {', '.join(clean)}{limit}")
