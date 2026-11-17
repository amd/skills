# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Checks for AIM models and the GitOps state that deploys them."""
from __future__ import annotations

import re
from typing import Any

from .shared import (
    CheckResult,
    INCONCLUSIVE,
    OK,
    WARN,
    _as_int,
    _condition_is_true,
    _conditions,
    _deploy_env,
    _dict,
    _in_e2e_footprint,
    _list,
    _once_per_run,
    _one_line,
    crd_gate,
    crd_state,
    kubectl_json,
)


def _model_selected(name: str, models: list[str]) -> bool:
    """With no --model given every model counts; otherwise only a model whose resource name is exactly one of them."""
    return not models or name in models

# aim-profiles and aim-ready read the AIMClusterModel list and aim-model-health also reads the
# namespaced AIMModel; each judges it differently (profile condition, top-level status, namespace/model scope).
def check_aim_profiles(models: list[str] | None = None, e2e: bool = False) -> CheckResult:
    """With models (e2e mode, --model) only those models can fail the check; the rest are noted.

    In e2e mode a model the run did not name that has no profile for the cluster's GPUs is a
    hardware-coverage gap of the catalog, not a cluster fault, so it is shown without gating.
    """
    models = models or []
    n = "aim-profiles"
    gate = crd_gate("aimclustermodels.aim.eai.amd.com", n, "AIM not detected (no aimclustermodels CRD) — cannot check model/GPU profile compatibility")
    if gate:
        return gate
    ok, obj, err = kubectl_json(["get", "aimclustermodels.aim.eai.amd.com", "-A"])
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot query aimclustermodels: {err}")
    items = _list((obj or {}).get("items"))
    # A requested model that matches nothing is not "nothing incompatible": the run cannot use a
    # model that is not on the cluster, so it must never read as affirmatively ready.
    names = {str(_dict(_dict(m).get("metadata")).get("name", "")) for m in items}
    absent = [m for m in models if m not in names]
    if absent:
        return CheckResult(
            n, WARN, f"requested model(s) not found among {len(items)} cluster model(s): {', '.join(absent)}",
            "Onboard the model (or fix the --model name) before the run; a run cannot use a model the cluster does not have.",
        )
    if not items:
        return CheckResult(n, OK, "AIM present, no cluster models defined yet — nothing incompatible")
    unsupported = []       # confirmed NoSupportedProfiles -> WARN
    unproven = []          # condition missing/unparseable or a different not-ready reason -> INCONCLUSIVE
    unused = []            # models the run does not use: noted, never gating
    for m in items:
        # Null-safe against malformed CR data (status: null, conditions: null / not-a-list, non-dict
        # entries) — a parse crash here would abort the whole run and break the --json contract.
        name = _dict(_dict(m).get("metadata")).get("name", "?")
        cond = next((c for c in _conditions(m) if c.get("type") == "ManagedProfilesReady"), None)
        if not _model_selected(str(name), models):
            if cond is None or cond.get("status") != "True":
                unused.append(str(name))
            continue
        if cond is None:
            unproven.append(f"{name}(no ManagedProfilesReady condition)")
        elif cond.get("status") == "True":
            continue  # affirmatively proven ready
        elif cond.get("reason") == "NoSupportedProfiles":
            unsupported.append(name)
        else:
            unproven.append(f"{name}({cond.get('reason') or 'NotReady'})")
    # A mixed catalog keeps the gating result: unresolved readiness must never read as healthy.
    if unsupported and not (e2e and not models and unproven):
        # Informational only while some catalog model is both Ready and profiled: a cluster that can
        # deploy no model must stay a problem.
        deployable = any(
            not _is_base_only_cluster_model(m)
            and _dict(_dict(m).get("status")).get("status") == "Ready"
            and any(c.get("type") == "ManagedProfilesReady" and c.get("status") == "True" for c in _conditions(m))
            for m in items
        )
        gpus_known, gpus = _allocatable_gpus()
        cpu_only = e2e and not models and gpus_known and gpus == 0
        gap = e2e and not models and (deployable or cpu_only)
        reason = " (hardware-coverage gap of the catalog, not a cluster fault: informational in e2e mode, not gating)" if gap else ""
        if cpu_only:
            reason = " (no allocatable GPUs, so no model is expected to deploy: informational in e2e mode, not gating)"
        return CheckResult(
            n, WARN,
            f"models cannot be deployed on this hardware — no supported AIM profile matches the cluster's GPUs{reason}. Models: {', '.join(unsupported)}. GPU device-id(s) on nodes: {_gpu_device_ids() or 'unknown'}",
            "Onboard an AIM profile for that GPU/partition, or deploy on matching hardware.",
            gating=not gap,
        )
    if unproven:
        # Anti-silence: a model whose readiness we could not affirmatively confirm must NOT be
        # counted as healthy — report it as inconclusive rather than a silent OK.
        return CheckResult(
            n, INCONCLUSIVE,
            f"cannot confirm every model has a supported profile — unresolved: {', '.join(unproven)}"
            + (f"; also no supported profile (hardware-coverage gap): {', '.join(unsupported)}" if unsupported else ""),
            "Wait for these aimclustermodels to reconcile, or inspect their ManagedProfilesReady condition.",
        )
    note = f"; {len(unused)} model(s) the run does not use lack a profile (informational, not gating)" if unused else ""
    return CheckResult(n, OK, f"every {'selected ' if models else ''}cluster model is affirmatively ManagedProfilesReady=True{note}")

def _gpu_device_ids() -> str:
    ok, obj, _ = kubectl_json(["get", "nodes"])
    if not ok:
        return ""
    ids = {
        _dict(_dict(_dict(i).get("metadata")).get("labels")).get("amd.com/gpu.device-id", "")
        for i in _list((obj or {}).get("items"))
    }
    return ", ".join(sorted(x for x in ids if x))

def check_aim_route_hostnames() -> CheckResult:
    """Catch the catalog-AIM-deploy failure pre-run: on a multi-listener AIM gateway, aim-engine
    requires spec.routing.hostnames on the effective AIMClusterRuntimeConfig, or the AIMService
    gets no HTTPRoute (ConfigValid=False / RouteHostnameRequired) and catalog AIM deploys fail.
    AIWB's own default runtimeconfig sets hostnames; the failure mode is aim-engine's competing
    default overriding it with none.
    """
    n = "aim-route-hostnames-ready"
    # Resolve gateway + runtimeconfig names from aiwb-api env. Defaults are trustworthy only after
    # the deployment was read successfully; otherwise they could make us inspect unrelated objects.
    env_ok, env, env_err = _deploy_env("aiwb-api", "aiwb")
    if not env_ok:
        return CheckResult(
            n, INCONCLUSIVE,
            f"cannot read aiwb/aiwb-api deployment to resolve gateway/runtimeconfig coordinates: {env_err} — default coordinates were not assumed",
            "Confirm the deployment exists and grant read access, then re-run.",
        )
    gw_ns = env.get("AIM_GATEWAY_NAMESPACE") or "kgateway-system"
    gw_name = env.get("AIM_GATEWAY_NAME") or "https"
    rc_name = env.get("AIM_CLUSTER_RUNTIME_CONFIG_NAME") or "default"

    ok, gw, err = kubectl_json(["get", "gateway", gw_name, "-n", gw_ns])
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot read the AIM gateway {gw_ns}/{gw_name}: {err}",
                           "Confirm the AIM gateway exists and kube access is available.")
    listeners = _list(_dict(_dict(gw).get("spec")).get("listeners"))
    if len(listeners) <= 1:
        # aim-engine only requires per-route hostnames on a multi-listener gateway.
        return CheckResult(n, OK, f"AIM gateway {gw_ns}/{gw_name} is single-listener — route hostnames not required")

    ok2, rcfg, err2 = kubectl_json(["get", "aimclusterruntimeconfig", rc_name])
    if not ok2:
        return CheckResult(n, INCONCLUSIVE, f"AIM gateway is multi-listener but cannot read AIMClusterRuntimeConfig '{rc_name}': {err2}",
                           "AIM CRDs may be absent or unreadable; cannot confirm routing hostnames.")
    routing = _dict(_dict(_dict(rcfg).get("spec")).get("routing"))
    hostnames = _list(routing.get("hostnames"))
    enabled = routing.get("enabled")
    if enabled and not hostnames:
        return CheckResult(
            n, WARN,
            f"AIM gateway {gw_name} has {len(listeners)} listeners but the effective AIMClusterRuntimeConfig '{rc_name}' has routing enabled with NO hostnames — catalog AIM deploys will fail with RouteHostnameRequired (aim-engine issue #142)",
            "The aim-engine default ClusterRuntimeConfig is likely overriding AIWB's hostname-bearing default; set clusterRuntimeConfig.enable=false on aim-engine so AIWB's 'default' is effective.",
        )
    if enabled is None and not hostnames:
        # routing.enabled omitted: the CRD defaults it to true, so an empty hostnames list may still
        # break catalog deploys — but we cannot read the CRD default from here. Never call this OK on
        # unconfirmed state (anti-silence): the WARN case above proves the exact failure mode.
        return CheckResult(
            n, INCONCLUSIVE,
            f"AIM gateway {gw_name} has {len(listeners)} listeners and the effective AIMClusterRuntimeConfig '{rc_name}' omits routing.enabled with NO hostnames — if the CRD defaults enabled to true, catalog AIM deploys will fail with RouteHostnameRequired",
            "Set routing.enabled explicitly on the effective AIMClusterRuntimeConfig (or add hostnames) so readiness can be confirmed.",
        )
    return CheckResult(n, OK, f"AIMClusterRuntimeConfig '{rc_name}' sets {len(hostnames)} routing hostname(s) on a {len(listeners)}-listener gateway")

def check_required_secrets() -> CheckResult:
    n = "required-secrets"
    ok, obj, err = kubectl_json(["get", "externalsecret", "-n", "aiwb"])
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot read ExternalSecrets in ns aiwb: {err}", "Ensure read access to the cluster.")
    items = _list((obj or {}).get("items"))
    if not items:
        return CheckResult(n, INCONCLUSIVE, "no ExternalSecrets in ns aiwb yet (cluster not installed / not fully synced?)")
    bad = []
    for i in items:
        conds = _conditions(i)
        if not _condition_is_true(i, "Ready"):
            reason = next((c.get("reason") for c in conds if c.get("type") == "Ready"), None) or "NotReady"
            bad.append(f"{_dict(_dict(i).get('metadata')).get('name', '?')}({reason})")
    if bad:
        return CheckResult(
            n, WARN,
            f"{len(bad)} ExternalSecret(s) not synced: {', '.join(bad)}",
            "The secret backend (OpenBao/ESO) must mint these keys; a SecretSyncedError here blocks aiwb-api.",
        )
    return CheckResult(n, OK, f"all {len(items)} aiwb ExternalSecret(s) synced")

def _allocatable_gpus() -> tuple[bool, int]:
    ok, obj, _ = kubectl_json(["get", "nodes"])
    if not ok:
        return False, 0
    return True, sum(
        _as_int(_dict(_dict(_dict(node).get("status")).get("allocatable")).get("amd.com/gpu", 0))
        for node in _list(_dict(obj).get("items"))
    )

def _resource_ref(item: Any) -> str:
    meta = _dict(_dict(item).get("metadata"))
    ns = meta.get("namespace")
    return f"{ns}/{meta.get('name', '?')}" if ns else str(meta.get("name", "?"))

def _aim_status(item: Any) -> str:
    return str(_dict(_dict(item).get("status")).get("status") or "Unknown")

_AIM_PROFILE_AGGREGATE_KEYS = frozenset({"base", "ready", "total", "notAvailable", "not_available"})

def _is_base_only_cluster_model(item: Any) -> bool:
    """Mirror of the AIWB API's rule for AIMClusterModels it leaves out of the inference catalog:
    only base profiles, or an unresolved base image. Such a model being Ready serves nothing."""
    status = _dict(_dict(item).get("status"))
    profiles = _dict(status.get("managedProfiles"))
    base = profiles.get("base")
    if isinstance(base, int) and base > 0 and not any(
        isinstance(count, int) and count > 0
        for kind, count in profiles.items() if kind not in _AIM_PROFILE_AGGREGATE_KEYS
    ):
        return True
    model = _dict(_dict(status.get("imageMetadata")).get("model"))
    if "base" not in str(model.get("title") or "").strip().lower():
        return False
    return status.get("aimId") is None and model.get("canonicalName") is None

def check_aim_ready(models: list[str] | None = None) -> CheckResult:
    """On a cluster with GPUs, the AIM catalog must offer at least one Ready model. This is the set
    the inference suites draw from (the API's AIM list filtered to Ready); when it is empty they fail
    with 'No Ready AIMs available in system' long after the run started."""
    n = "aim-ready"
    ok, gpus = _allocatable_gpus()
    if not ok:
        return CheckResult(n, INCONCLUSIVE, "cannot read nodes to count allocatable GPUs")
    if gpus == 0 and not models:
        return CheckResult(n, OK, "no allocatable GPUs — no AIM is expected to be Ready")
    gate = crd_gate("aimclustermodels.aim.eai.amd.com", n,
                    f"AIM not detected (no aimclustermodels CRD) on a cluster with {gpus} allocatable GPU(s) — cannot confirm any model can be deployed")
    if gate:
        return gate
    ok, obj, err = kubectl_json(["get", "aimclustermodels.aim.eai.amd.com", "-A"])
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot list AIMClusterModels: {err}")
    catalog = [i for i in _list(_dict(obj).get("items")) if not _is_base_only_cluster_model(i)]
    ready = [_resource_ref(i) for i in catalog if _aim_status(i) == "Ready"]
    if models:
        # The run names its models: an unrelated Ready model proves nothing about them.
        by_name = {str(_dict(_dict(i).get("metadata")).get("name", "")): _aim_status(i) for i in catalog}
        not_ready = [f"{m}({by_name.get(m, 'not in the catalog')})" for m in models if by_name.get(m) != "Ready"]
        if not_ready:
            return CheckResult(
                n, WARN, f"requested model(s) not Ready on a cluster with {gpus} allocatable GPU(s): {', '.join(not_ready)}",
                "Inference suites that use these models fail until they are Ready; describe the AIMClusterModels to see why.",
            )
        return CheckResult(n, OK, f"every requested model is Ready ({', '.join(models)}) on {gpus} allocatable GPU(s)")
    if ready:
        return CheckResult(n, OK, f"{len(ready)} of {len(catalog)} catalog AIM(s) Ready on {gpus} allocatable GPU(s)")
    states: dict[str, int] = {}
    for i in catalog:
        states[_aim_status(i)] = states.get(_aim_status(i), 0) + 1
    seen = ", ".join(f"{s} ({c})" for s, c in sorted(states.items())) or "the catalog is empty"
    return CheckResult(
        n, WARN,
        f"no catalog AIM is Ready on a cluster with {gpus} allocatable GPU(s) — AIM states: {seen}",
        "Inference suites need a Ready AIM and will fail with 'No Ready AIMs available in system'. "
        "Describe the AIMClusterModels to see why none is Ready (image not cached, no profile for "
        "these GPUs), and fix that before the run.",
    )

_AIM_MODEL_CRDS = ("aimmodels.aim.eai.amd.com", "aimclustermodels.aim.eai.amd.com")

# The engine's states for a model that cannot be served; the product classifies Failed and Error as
# a failed onboarding and Degraded as a partly broken one.
_AIM_UNHEALTHY_STATUSES = frozenset({"Degraded", "Failed", "Error"})

# Every status the engine reports for a model; anything else is missing or malformed.
_AIM_STATUSES = _AIM_UNHEALTHY_STATUSES | {"NotAvailable", "Pending", "Progressing", "Ready"}

def check_aim_model_health(scope_namespace: str | None = None, models: list[str] | None = None) -> CheckResult:
    """Flag AIM models in Degraded, Failed or Error status — a model that cannot be served reliably.

    In e2e mode (scope_namespace set) a namespaced model counts only in the test and platform
    namespaces, an e2e-* one only when named by --model (else it is a leftover of an earlier run),
    and a cluster-scoped one only when named; the rest is noted, not gating.
    """
    models = models or []
    n = "aim-model-health"
    unhealthy: list[str] = []
    unproven: list[str] = []
    total = outside = 0
    installed = False
    for crd in _AIM_MODEL_CRDS:
        state = crd_state(crd)
        if state == "absent":
            continue
        if state == "unreadable":
            unproven.append(f"cannot read the {crd} CRD")
            continue
        installed = True
        ok, obj, err = kubectl_json(["get", crd, "-A"])
        if not ok:
            unproven.append(f"cannot list {crd}: {_one_line(err)}")
            continue
        for item in _list(_dict(obj).get("items")):
            meta = _dict(_dict(item).get("metadata"))
            ns = str(meta.get("namespace") or "")
            # Models in e2e-* namespaces other than the selected test namespace are what earlier
            # (possibly cancelled) runs left behind, so they gate only when --model names them; other namespaced models are scoped by namespace alone.
            name = str(meta.get("name", ""))
            if scope_namespace is not None and (
                ((ns.startswith("e2e-") and ns != scope_namespace and name not in models) or not _in_e2e_footprint(ns, scope_namespace))
                if ns else not _model_selected(name, models)
            ):
                outside += 1
                continue
            total += 1
            status = _aim_status(item)
            if status not in _AIM_STATUSES:
                # No status yet, or one this check does not know: health is not established.
                unproven.append(f"{_resource_ref(item)} reports no known status ({status})")
                continue
            if status not in _AIM_UNHEALTHY_STATUSES:
                continue
            why = next((c for c in _conditions(item) if c.get("status") == "False" and c.get("reason")), None)
            unhealthy.append(f"{_resource_ref(item)}={status}" + (f"({why.get('reason')})" if why else ""))
    if unhealthy:
        return CheckResult(
            n, WARN,
            f"{len(unhealthy)} AIM model(s) Degraded, Failed or Error: {', '.join(unhealthy)}",
            "Such a model cannot be deployed reliably; describe it to see which source or profile "
            "failed, and fix it before running suites that use it.",
        )
    if unproven:
        return CheckResult(n, INCONCLUSIVE, f"cannot confirm AIM model health: {'; '.join(unproven)}")
    if not installed:
        return CheckResult(n, INCONCLUSIVE, "AIM not detected (no aimmodels/aimclustermodels CRD) — no model health to check")
    note = f"; {outside} model(s) outside the E2E scope not examined (informational)" if outside else ""
    return CheckResult(n, OK, f"no Degraded, Failed or Error model among {total} AIM model(s){note}")

# The revisions that pin what an Application deploys: an exact (SemVer-shaped) release or chart
# version, or a commit SHA. Anything else — HEAD, a branch such as main or release/v2.4.x, a chart
# version range — deploys whatever lands next. Whether a bare name is a branch or a tag needs the
# source repository, which this offline check does not read, so only the pinned shapes are trusted.
_PINNED_REVISION = re.compile(r"^v?[0-9]+\.[0-9]+\.[0-9]+(?:[-+][0-9A-Za-z.+-]+)?$|^[0-9a-f]{7,40}$")

_UNRESOLVED_REVISION = re.compile(r"unable to resolve '([^']+)' to a commit SHA", re.IGNORECASE)

def _unresolved_revision(status: dict[str, Any]) -> str | None:
    """The reason ArgoCD names when an Application's targetRevision cannot be resolved to a commit
    (e.g. a branch the repository no longer has), or None. It sits in a status condition."""
    for cond in _list(status.get("conditions")):
        message = str(_dict(cond).get("message", ""))
        if _UNRESOLVED_REVISION.search(message):
            return _one_line(message)
    return None

_KINDS_WITHOUT_HEALTH = frozenset({
    "ConfigMap", "Secret", "ServiceAccount", "Namespace", "Role", "RoleBinding", "ClusterRole", "ClusterRoleBinding",
    "NetworkPolicy", "StorageClass", "ResourceQuota", "LimitRange", "PriorityClass",
})

def _externalsecret_settled(r: dict[str, Any], status: dict[str, Any]) -> bool:
    """An ExternalSecret has no Argo health, so its own sync result stands in: Synced, or a hook
    that the last sync ran to Succeeded (a hook resource carries no sync status in the tree)."""
    if r.get("kind") != "ExternalSecret":
        return False
    if r.get("status") == "Synced":
        return True
    operation = _dict(status.get("operationState"))
    if not (r.get("hook") and r.get("status") is None and operation.get("phase") == "Succeeded"):
        return False
    return any(
        _dict(h).get("hookPhase") == "Succeeded"
        and all(_dict(h).get(k) == r.get(k) for k in ("group", "kind", "namespace", "name"))
        for h in _list(_dict(operation.get("syncResult")).get("resources"))
    )

def _only_waiting_for_first_consumer(app: dict[str, Any], status: dict[str, Any]) -> bool:
    """True when an Application is Progressing only because its PVCs are Pending on a
    WaitForFirstConsumer StorageClass, which bind once a pod uses them: steady state, not a fault."""
    # The PVC and StorageClass inventories are read from the current cluster, so an app that deploys
    # anywhere else (or whose destination is unknown) cannot be judged from them.
    destination = _dict(_dict(app.get("spec")).get("destination"))
    if destination.get("server") != "https://kubernetes.default.svc" and destination.get("name") != "in-cluster":
        return False
    resources = [_dict(r) for r in _list(status.get("resources"))]
    # Argo CD reports no health for kinds it cannot assess, so a missing health is only accepted for those:
    # for anything else (a Deployment, say) it is absent evidence, not proof the resource is Healthy.
    if any(
        r.get("kind") != "PersistentVolumeClaim"
        and _dict(r.get("health")).get("status") != "Healthy"
        and not (_dict(r.get("health")).get("status") is None and r.get("kind") in _KINDS_WITHOUT_HEALTH)
        and not (_dict(r.get("health")).get("status") is None and _externalsecret_settled(r, status))
        for r in resources
    ):
        return False
    claims = {(r.get("namespace"), r.get("name")) for r in resources if r.get("kind") == "PersistentVolumeClaim"}
    if not claims:
        return False
    ok, pvcs, _ = _once_per_run(("pvcs",), lambda: kubectl_json(["get", "pvc", "-A"]))
    ok_sc, scs, _ = _once_per_run(("storageclasses",), lambda: kubectl_json(["get", "storageclass"]))
    if not (ok and ok_sc):
        return False
    waiting = {_dict(_dict(s).get("metadata")).get("name") for s in _list(_dict(scs).get("items"))
               if _dict(s).get("volumeBindingMode") == "WaitForFirstConsumer"}
    pending: set[tuple[Any, Any]] = set()
    bound: set[tuple[Any, Any]] = set()
    for p in _list(_dict(pvcs).get("items")):
        meta = _dict(_dict(p).get("metadata"))
        key = (meta.get("namespace"), meta.get("name"))
        if key not in claims:
            continue
        # A claim being deleted (held by a finalizer) is not steady state, whatever its phase.
        if meta.get("deletionTimestamp"):
            return False
        if _dict(_dict(p).get("status")).get("phase") == "Bound":
            bound.add(key)
            continue
        # A selected-node annotation means a pod already chose a node, and a volumeName means a specific
        # volume is already being bound: either way the claim is no longer waiting for a consumer.
        if (
            _dict(_dict(p).get("status")).get("phase") != "Pending"
            or _dict(_dict(p).get("spec")).get("storageClassName") not in waiting
            or "volume.kubernetes.io/selected-node" in _dict(meta.get("annotations"))
            or _dict(_dict(p).get("spec")).get("volumeName")
        ):
            return False
        pending.add(key)
    # A tracked claim that was not found is unaccounted for, so the wait cannot be called steady state.
    return bool(pending) and all(c in pending or c in bound for c in claims)

def check_argocd_apps() -> CheckResult:
    """All ArgoCD Applications must be Synced and Healthy. The targetRevision mix is reported, and the
    Applications not pinned to a release or commit are named, so one that follows a branch is
    visible in the record of the run."""
    n = "argocd-apps"
    state = crd_state("applications.argoproj.io")
    if state == "absent":
        return CheckResult(n, OK, "ArgoCD is not installed on this cluster — no Applications to check")
    if state == "unreadable":
        return CheckResult(n, INCONCLUSIVE, "cannot read the applications.argoproj.io CRD (RBAC/API error) — sync state unconfirmed")
    ok, obj, err = kubectl_json(["get", "applications.argoproj.io", "-A"])
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot list ArgoCD Applications: {err}")
    apps = _list(_dict(obj).get("items"))
    if not apps:
        return CheckResult(n, OK, "no ArgoCD Applications on this cluster")
    unhealthy: list[str] = []
    unresolved: list[str] = []
    waiting_volume: list[str] = []
    moving: list[str] = []
    sourceless: list[str] = []
    mix: dict[str, int] = {}
    for app in apps:
        name = _dict(_dict(app).get("metadata")).get("name", "?")
        spec, status = _dict(_dict(app).get("spec")), _dict(_dict(app).get("status"))
        sync = _dict(status.get("sync")).get("status") or "Unknown"
        health = _dict(status.get("health")).get("status") or "Unknown"
        reason = _unresolved_revision(status)
        if reason and health == "Healthy":
            unresolved.append(f"{name}: {reason}")
        elif sync == "Synced" and health == "Progressing" and _only_waiting_for_first_consumer(_dict(app), status):
            waiting_volume.append(name)
        elif sync != "Synced" or health != "Healthy":
            unhealthy.append(f"{name}={sync}/{health}")
        sources = ([spec["source"]] if isinstance(spec.get("source"), dict) else []) + _list(spec.get("sources"))
        # A source without a repository deploys nothing identifiable, and one malformed source among
        # valid ones would otherwise be dropped silently, so either leaves the revision unknown.
        if not sources or not all(isinstance(s, dict) and s.get("repoURL") for s in sources):
            sourceless.append(name)
            continue
        revisions = {str(s.get("targetRevision") or "HEAD") for s in sources}
        for rev in revisions:
            mix[rev] = mix.get(rev, 0) + 1
        if any(not _PINNED_REVISION.match(rev) for rev in revisions):
            moving.append(name)
    mix_text = ", ".join(f"{rev} ({count})" for rev, count in sorted(mix.items(), key=lambda kv: (-kv[1], kv[0])))
    unresolved_text = f"; {len(unresolved)} Healthy Application(s) whose targetRevision cannot be resolved, so synchronization is unconfirmed (on its own a warning, not a problem): {'; '.join(sorted(unresolved))}" if unresolved else ""
    tracking = f"; not pinned to a release or commit: {', '.join(sorted(moving))}" if moving else ""
    # An unresolved app's revision comparison failed, so it cannot be counted as Synced.
    counted = f"{len(apps) - len(unresolved)} of {len(apps)}" if unresolved else f"all {len(apps)}"
    healthy = (
        f"{counted} ArgoCD Application(s) Synced+Healthy"
        if not waiting_volume
        else f"{counted} ArgoCD Application(s) Synced and Healthy or waiting only for a workload to claim a "
        f"WaitForFirstConsumer volume ({', '.join(sorted(waiting_volume))})"
    )
    if unhealthy:
        unknown = f"; revision unknown (no readable source): {', '.join(sorted(sourceless))}" if sourceless else ""
        return CheckResult(
            n, WARN,
            f"{len(unhealthy)} of {len(apps)} ArgoCD Application(s) not Synced+Healthy: {', '.join(unhealthy)}; targetRevision mix: {mix_text}{tracking}{unknown}{unresolved_text}",
            "The deployed state is not the declared one; tests would run against a partial rollout. Sync "
            "or fix these Applications first.",
        )
    if sourceless:
        return CheckResult(
            n, INCONCLUSIVE,
            f"{healthy}, but the revision of {', '.join(sorted(sourceless))} "
            f"is unknown (no readable source); targetRevision mix: {mix_text}{tracking}{unresolved_text}",
        )
    if unresolved:
        return CheckResult(n, OK, f"WARNING: {healthy}; targetRevision mix: {mix_text}{tracking}{unresolved_text}")
    return CheckResult(n, OK, f"{healthy}; targetRevision mix: {mix_text}{tracking}")

# Image repository names of the product and the AIM engine, e.g. amdenterpriseai/airm-api.
_PRODUCT_IMAGE = re.compile(r"^(?:airm|aiwb)[-_]|^aim-engine")
