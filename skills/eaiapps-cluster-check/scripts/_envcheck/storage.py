# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Checks for storage: CNI MTU, Longhorn, the provisioner, SeaweedFS and object-store fill."""
from __future__ import annotations

import os
import re
import time
from typing import Any

from .shared import (
    CheckResult,
    INCONCLUSIVE,
    OK,
    WARN,
    _CLOCK_SKEW_SECONDS,
    _MINIO_NS,
    _SEAWEED_NS,
    _condition_is_true,
    _conditions,
    _dict,
    _in_e2e_footprint,
    _is_not_found,
    _list,
    _once_per_run,
    _rfc3339_age_seconds,
    _secret_state,
    crd_state,
    kubectl_json,
    run,
)


def _volume_unhealthy(volume: Any) -> bool:
    """Longhorn reports robustness=unknown for every DETACHED volume (nothing to judge without an
    attached engine), so unknown+detached is idle, not broken. Unknown on an attached volume, a
    blank robustness, degraded and faulted all stay unhealthy."""
    st = _dict(_dict(volume).get("status"))
    robustness, state = st.get("robustness", ""), st.get("state", "")
    if state == "detached" and robustness in ("healthy", "unknown"):
        return False
    return not (robustness == "healthy" and state in ("attached", "detached"))

# One engine-settle window; the check re-reads at most once, so the wait is bounded. The env var lets black-box suites shorten it.
_LONGHORN_RECHECK_SECONDS = 60

def _longhorn_recheck_seconds() -> float:
    """Env override clamped to 0..60; anything unparseable or non-finite falls back to the default."""
    try:
        seconds = float(os.environ.get("ENVIRONMENT_CHECK_LONGHORN_RECHECK_SECONDS", _LONGHORN_RECHECK_SECONDS))
    except ValueError:
        return float(_LONGHORN_RECHECK_SECONDS)
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return float(_LONGHORN_RECHECK_SECONDS)
    return min(max(seconds, 0.0), float(_LONGHORN_RECHECK_SECONDS))

def check_longhorn_health(scope_namespace: str | None = None) -> CheckResult:
    """Cluster-wide volume health; with scope_namespace (e2e mode) only the E2E footprint can fail
    the check and volumes elsewhere are reported as informational."""
    n = "longhorn-health"
    state = crd_state("volumes.longhorn.io")
    if state == "unreadable":
        return CheckResult(n, INCONCLUSIVE, "cannot read the volumes.longhorn.io CRD (RBAC/API error) — state unconfirmed")
    if state == "absent":
        # Without Longhorn, shared storage is only unconfirmed when something other than node-local
        # provisioners could be backing it; a cluster whose every StorageClass is node-local has none.
        sc_ok, sc, _ = kubectl_json(["get", "storageclass"])
        classes = _list(_dict(sc).get("items")) if sc_ok else []
        if classes and all(str(_dict(c).get("provisioner", ""))in _NODE_LOCAL_PROVISIONERS for c in classes):
            return CheckResult(n, OK, "Longhorn not in use (no volumes.longhorn.io CRD) and every StorageClass is node-local — Longhorn health not applicable")
        return CheckResult(n, INCONCLUSIVE, "Longhorn not detected (no volumes.longhorn.io CRD) — shared-storage health unconfirmed via Longhorn")
    ok, obj, err = kubectl_json(["get", "volumes.longhorn.io", "-A"])
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot query Longhorn volumes: {err}")
    items = _list((obj or {}).get("items"))
    if not items:
        return CheckResult(n, OK, "Longhorn present, no volumes provisioned yet — nothing unhealthy")
    def classify(vols: list[Any]) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]], int]:
        gating: list[tuple[str, str, str]] = []
        informational: list[tuple[str, str, str]] = []
        elsewhere_count = 0
        for v in vols:
            st = _dict(_dict(v).get("status"))
            pvc_ns = str(_dict(st.get("kubernetesStatus")).get("namespace") or "")
            # A volume with no bound PVC namespace is unconfirmed, not confirmed elsewhere: it gates.
            elsewhere = scope_namespace is not None and bool(pvc_ns) and not _in_e2e_footprint(pvc_ns, scope_namespace)
            elsewhere_count += int(elsewhere)
            if not _volume_unhealthy(v):
                continue
            entry = (str(_dict(_dict(v).get("metadata")).get("name", "?")), str(st.get("robustness") or ""), str(st.get("state") or ""))
            (informational if elsewhere else gating).append(entry)
        return gating, informational, elsewhere_count

    bad_entries, outside_entries, out_of_scope = classify(items)
    # A volume that was just attached reports unknown robustness until its engine settles; only
    # that shape is re-read. Degraded, faulted and blank robustness gate at once.
    if bad_entries and all(r == "unknown" and s == "attached" for _, r, s in bad_entries):
        time.sleep(_longhorn_recheck_seconds())
        ok2, obj2, err2 = kubectl_json(["get", "volumes.longhorn.io", "-A"])
        if not ok2:
            return CheckResult(
                n, WARN,
                f"shared storage may be unhealthy: volumes attached with unknown robustness and the re-read failed ({err2}): "
                f"{', '.join(f'{a}({b}/{c})' for a, b, c in bad_entries)}",
                "Check the Longhorn UI / nfs-ganesha pods and recover the volumes before deploying workloads.",
            )
        # The second read is authoritative: a volume clears only by turning healthy or vanishing, and any
        # volume bad in it gates, including one that was fine in the first read.
        items = _list((obj2 or {}).get("items"))
        bad_entries, outside_entries, out_of_scope = classify(items)
    bad = [f"{a}({b or 'unknown'}/{c or 'unknown'})" for a, b, c in bad_entries]
    outside = [f"{a}({b or 'unknown'}/{c or 'unknown'})" for a, b, c in outside_entries]
    if bad:
        return CheckResult(
            n, WARN,
            f"shared storage is unhealthy — workloads may fail to start or mount. Unhealthy volumes: {', '.join(bad)}",
            "Check the Longhorn UI / nfs-ganesha pods and recover the volumes before deploying workloads.",
        )
    note = f"; {len(outside)} unhealthy volume(s) outside the E2E namespaces (informational, not gating): {', '.join(outside)}" if outside else ""
    if scope_namespace is None:
        return CheckResult(n, OK, f"all {len(items)} Longhorn volume(s) healthy")
    return CheckResult(n, OK, f"all {len(items) - out_of_scope} Longhorn volume(s) in scope healthy{note}")


LONGHORN_NS = "longhorn"


# Components every volume operation depends on: the manager reconciles volumes, the instance managers run
# their engines and replicas, and the CSI plugin attaches and mounts them on each node.
_LONGHORN_COMPONENTS = {
    "longhorn-manager": lambda labels: labels.get("app") == "longhorn-manager",
    "instance-manager": lambda labels: labels.get("longhorn.io/component") == "instance-manager",
    "longhorn-csi-plugin": lambda labels: labels.get("app") == "longhorn-csi-plugin",
}


# A share-manager serves one RWX volume over NFS; restarting this often means it cannot serve it.
_SHARE_MANAGER_RESTARTS = 3


def _restarts(pod: Any) -> int:
    return sum(int(_dict(c).get("restartCount") or 0) for c in _list(_dict(_dict(pod).get("status")).get("containerStatuses")))


def _crash_looping(pod: Any) -> bool:
    waiting = [_dict(_dict(c).get("state")).get("waiting") for c in _list(_dict(_dict(pod).get("status")).get("containerStatuses"))]
    if any(_dict(w).get("reason") == "CrashLoopBackOff" for w in waiting):
        return True
    return not _pod_ready(pod) and _restarts(pod) >= _SHARE_MANAGER_RESTARTS


def check_longhorn_components(scope_namespace: str | None = None) -> CheckResult:
    """The Longhorn components and nodes RWX and RWO volumes need, and share-managers that keep crashing.

    Volume robustness (longhorn-health) can read healthy while no new volume can attach or mount: a CSI
    plugin or instance manager that is down, a node that refuses replicas, or a share-manager that keeps
    restarting leaves every pod that needs the volume stuck in ContainerCreating."""
    n = "longhorn-components"
    state = crd_state("volumes.longhorn.io")
    if state == "unreadable":
        return CheckResult(n, INCONCLUSIVE, "cannot read the volumes.longhorn.io CRD (RBAC/API error) — Longhorn components unconfirmed")
    if state == "absent":
        return CheckResult(n, OK, "Longhorn not in use (no volumes.longhorn.io CRD) — component check not applicable")
    ok, pods_obj, err = kubectl_json(["get", "pods", "-n", LONGHORN_NS])
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot read the Longhorn pods: {err}")
    ok_n, nodes_obj, err_n = kubectl_json(["get", "nodes.longhorn.io", "-n", LONGHORN_NS])
    if not ok_n:
        return CheckResult(n, INCONCLUSIVE, f"cannot read the Longhorn nodes: {err_n}")
    ok_d, ds_obj, err_d = kubectl_json(["get", "daemonsets", "-n", LONGHORN_NS])
    if not ok_d:
        return CheckResult(n, INCONCLUSIVE, f"cannot read the Longhorn DaemonSets: {err_d}")
    # A terminating pod is counted as not Ready: until its replacement is up, its node has no plugin.
    pods = _list(_dict(pods_obj).get("items"))
    problems: list[str] = []
    for ds in _list(_dict(ds_obj).get("items")):
        name = str(_dict(_dict(ds).get("metadata")).get("name", "?"))
        status = _dict(_dict(ds).get("status"))
        desired, ready = int(status.get("desiredNumberScheduled") or 0), int(status.get("numberReady") or 0)
        if name in ("longhorn-manager", "longhorn-csi-plugin") and ready < desired:
            problems.append(f"{name} Ready on {ready} of {desired} node(s)")
    for component, matches in _LONGHORN_COMPONENTS.items():
        mine = [p for p in pods if matches(_dict(_dict(_dict(p).get("metadata")).get("labels")))]
        not_ready = sorted(str(_dict(_dict(p).get("metadata")).get("name", "?")) for p in mine if not _pod_ready(p))
        if not mine:
            problems.append(f"no {component} pods")
        elif not_ready:
            problems.append(f"{component} not Ready: {', '.join(not_ready)}")
    for node in _list(_dict(nodes_obj).get("items")):
        name = str(_dict(_dict(node).get("metadata")).get("name", "?"))
        # longhorn.io/v1beta2 lists the conditions; v1beta1 keyed them by type.
        raw = _dict(_dict(node).get("status")).get("conditions")
        entries = [{"type": k, **_dict(v)} for k, v in raw.items()] if isinstance(raw, dict) else _list(raw)
        conditions = {str(_dict(c).get("type")): _dict(c).get("status") for c in entries}
        if conditions.get("Ready") != "True" or conditions.get("Schedulable") != "True" or _dict(_dict(node).get("spec")).get("allowScheduling") is False:
            problems.append(f"Longhorn node {name} cannot take volumes (Ready={conditions.get('Ready')}, Schedulable={conditions.get('Schedulable')})")
    looping = [p for p in pods if _dict(_dict(_dict(p).get("metadata")).get("labels")).get("longhorn.io/component") == "share-manager" and _crash_looping(p)]
    outside: list[str] = []
    if looping:
        ok_v, vols_obj, err_v = kubectl_json(["get", "volumes.longhorn.io", "-A"])
        volume_ns = {
            str(_dict(_dict(v).get("metadata")).get("name")): str(_dict(_dict(_dict(v).get("status")).get("kubernetesStatus")).get("namespace") or "")
            for v in _list(_dict(vols_obj).get("items"))
        } if ok_v else {}
        for pod in looping:
            labels = _dict(_dict(_dict(pod).get("metadata")).get("labels"))
            volume = str(labels.get("longhorn.io/share-manager") or "?")
            namespace = volume_ns.get(volume, "")
            entry = f"share-manager {volume} ({namespace or 'namespace unknown'}) restarted {_restarts(pod)} times"
            # In e2e mode only the E2E footprint gates; a volume whose namespace is unknown is not confirmed elsewhere.
            if scope_namespace is not None and namespace and not _in_e2e_footprint(namespace, scope_namespace):
                outside.append(entry)
            else:
                problems.append(entry)
    note = f"; {len(outside)} crash-looping share-manager(s) outside the E2E namespaces (informational, not gating): {', '.join(outside)}" if outside else ""
    if problems:
        return CheckResult(
            n, WARN,
            f"Longhorn cannot reliably attach or mount volumes: {'; '.join(problems)}{note}",
            "Pods that need a volume will stay in ContainerCreating. Restart the named Longhorn pods "
            "(their DaemonSet or the Longhorn manager recreates them) and re-run; for a node, check its "
            "Longhorn node conditions and disks.",
        )
    return CheckResult(n, OK, f"Longhorn manager, instance managers, CSI plugin and {len(_list(_dict(nodes_obj).get('items')))} node(s) ready{note}")

# Provisioners that place a volume on the node that uses it. kubernetes.io/nfs and the like are not
# listed: they are network storage, and kubernetes.io/no-provisioner only disables dynamic provisioning
# (statically created PVs can be network volumes), so neither proves a cluster has no shared storage.
_NODE_LOCAL_PROVISIONERS = frozenset({"rancher.io/local-path", "kubernetes.io/host-path"})

_NON_CSI_PROVISIONER_PREFIXES = ("kubernetes.io/", "rancher.io/local-path")

# A CSI leader-election Lease renews every ~15s; older than this means the holder is not alive.
_LEASE_FRESH_SECONDS = 120

def check_storage_provisioner() -> CheckResult:
    n = "storage-provisioner"
    ok, scobj, err = kubectl_json(["get", "storageclass"])
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot list StorageClasses: {err}")
    sc_items = _list((scobj or {}).get("items"))
    # Map each StorageClass to its provisioner (the CSI driver name it depends on).
    sc_prov = {_dict(_dict(s).get("metadata")).get("name", "?"): str(_dict(s).get("provisioner") or "") for s in sc_items}
    if not sc_prov:
        return CheckResult(n, INCONCLUSIVE, "no StorageClass found — cannot confirm a working storage provisioner (a workbench install needs one)")
    # A StorageClass with a blank/missing provisioner cannot be correlated to any driver; treating
    # it as fine would pass a class we know nothing about. Collect it as a caveat rather than
    # returning early — a real WARN from a *different* StorageClass must still win and be reported
    # (every problem gets surfaced, never masked by an unrelated unreadable one).
    no_prov = sorted(sc for sc, prov in sc_prov.items() if not prov)
    no_prov_note = (
        f" (also: StorageClass(es) with no usable provisioner field, backing driver unidentifiable: {', '.join(no_prov)})"
        if no_prov else ""
    )

    # Correlate each StorageClass to a *registered* CSI driver, so a healthy but unrelated CSI pod
    # cannot vouch for a StorageClass whose own provisioner is absent. Non-CSI/in-tree provisioners
    # (kubernetes.io/*, rancher.io/local-path) have no CSIDriver object and are exempt (not a false
    # pass).
    ok_cd, cdobj, err_cd = kubectl_json(["get", "csidrivers.storage.k8s.io"])
    if not ok_cd:
        return CheckResult(n, INCONCLUSIVE, f"cannot read csidrivers to correlate StorageClass provisioners: {err_cd}")
    registered = {_dict(_dict(d).get("metadata")).get("name", "") for d in _list((cdobj or {}).get("items"))}
    csi_provs = {
        prov for prov in sc_prov.values()
        if prov and not any(prov.startswith(p) for p in _NON_CSI_PROVISIONER_PREFIXES)
    }
    unbacked = sorted(
        f"{sc} (provisioner {prov})"
        for sc, prov in sc_prov.items()
        if prov in csi_provs and prov not in registered
    )
    if unbacked:
        return CheckResult(
            n, WARN,
            f"StorageClass(es) reference a CSI provisioner that is not a registered CSIDriver — new PVCs will stay Pending: {', '.join(unbacked)}{no_prov_note}",
            "Install/repair the CSI driver for that StorageClass before deploying workloads.",
        )

    # Per-provisioner liveness: a CSI provisioner is live only if its node plugin has registered on
    # at least one node. CSINode.spec.drivers[].name lists exactly that, keyed to the driver name —
    # so this ties liveness to each StorageClass's OWN provisioner, not a global csi-driver pod
    # sweep (which an unrelated Ready pod could satisfy). Without CSINodes we cannot establish the
    # mapping, so the result is inconclusive, never a false OK.
    live_csi = [prov for sc, prov in sc_prov.items() if prov in registered]
    if live_csi:
        ok_cn, cnobj, err_cn = kubectl_json(["get", "csinodes.storage.k8s.io"])
        if not ok_cn:
            return CheckResult(n, INCONCLUSIVE, f"StorageClass provisioners are registered, but cannot read csinodes to confirm each is live: {err_cn}")
        node_registered: set[str] = set()
        for cn in _list((cnobj or {}).get("items")):
            for drv in _list(_dict(_dict(cn).get("spec")).get("drivers")):
                name = _dict(drv).get("name")
                if name:
                    node_registered.add(name)
        not_live = sorted(
            f"{sc} (provisioner {prov})"
            for sc, prov in sc_prov.items()
            if prov in registered and prov not in node_registered
        )
        if not_live:
            return CheckResult(
                n, WARN,
                f"a storage provisioner's node plugin is not registered on any node — new PVCs will stay Pending: {', '.join(not_live)}{no_prov_note}",
                "Restart/repair the CSI node plugin (DaemonSet) for that provisioner before deploying workloads.",
            )

        # Controller liveness: the node plugin registering is not enough — the external-provisioner
        # sidecar (which handles CreateVolume) must be alive. It holds a leader-election Lease named
        # after the driver (e.g. driver-longhorn-io / external-provisioner-leader-<driver>) and
        # renews it every ~15s, so a Lease whose renewTime is recent proves the controller is live
        # NOW. No matching Lease means we cannot establish controller liveness -> inconclusive.
        ok_l, lobj, err_l = kubectl_json(["get", "leases.coordination.k8s.io", "-A"])
        if not ok_l:
            return CheckResult(n, INCONCLUSIVE, f"node plugins registered, but cannot read leases to confirm the provisioner controllers are live: {err_l}")
        # Leases are namespaced, so two namespaces can hold a lease with the same name; keep every
        # lease in a list (not a name-keyed dict, which would drop one duplicate) and pair each with
        # its name for the substring match below.
        all_leases = [(_dict(_dict(le).get("metadata")).get("name", ""), le) for le in _list((lobj or {}).get("items"))]
        stale, unproven = [], []
        for prov in sorted(set(live_csi)):
            dashed = prov.replace(".", "-").replace("/", "-")
            # Match ONLY the external-provisioner (CreateVolume) lease — not the attacher/resizer/
            # snapshotter sidecar leases, which also embed the driver name. A fresh attacher lease
            # must not vouch that the provisioner is live. The provisioner lease is either the bare
            # driver name (e.g. driver-longhorn-io) or an explicit external-provisioner-* lease.
            matched = [
                le for name, le in all_leases
                if name == dashed or (dashed in name and "provisioner" in name)
            ]
            if not matched:
                unproven.append(prov)
                continue
            ages = [_rfc3339_age_seconds(str(_dict(_dict(le).get("spec")).get("renewTime") or "")) for le in matched]
            # A renewTime further in the future than clock skew proves nothing about a live holder.
            fresh = [a for a in ages if a is not None and -_CLOCK_SKEW_SECONDS <= a <= _LEASE_FRESH_SECONDS]
            if not fresh:
                stale.append(prov)
        if stale:
            return CheckResult(
                n, WARN,
                f"a storage provisioner's controller is not renewing its leader lease (not live) — CreateVolume will not run: {', '.join(stale)}{no_prov_note}",
                "Restart/repair the CSI controller (external-provisioner) Deployment for that provisioner.",
            )
        if unproven:
            return CheckResult(
                n, INCONCLUSIVE,
                f"node plugins are live, but no leader lease was found to confirm the CreateVolume controller for: {', '.join(unproven)} — controller liveness unproven{no_prov_note}",
                "Confirm the CSI controller (external-provisioner) is running for that provisioner.",
            )
    csi_ok = sorted(set(live_csi))
    intree = sorted({
        prov for prov in sc_prov.values()
        if prov and any(prov.startswith(p) for p in _NON_CSI_PROVISIONER_PREFIXES)
    })
    parts = []
    if csi_ok:
        parts.append(f"CSI provisioner(s) registered with a live node plugin and controller: {', '.join(csi_ok)}")
    if intree:
        parts.append(f"in-tree/non-CSI provisioner(s) present (not CSI-verifiable): {', '.join(intree)}")
    summary = "; ".join(parts) or f"StorageClass provisioner(s) present: {', '.join(sorted(v for v in sc_prov.values() if v))}"
    if no_prov:
        # Nothing worse was found, but a class we could not identify at all remains unproven.
        return CheckResult(
            n, INCONCLUSIVE,
            f"StorageClass(es) have no usable provisioner field, so their backing driver cannot be identified: {', '.join(no_prov)}. Others checked: {summary}",
            "Verify those StorageClasses and their backing provisioner manually.",
        )
    return CheckResult(n, OK, summary)

def _is_model_download(meta: dict[str, Any]) -> bool:
    """Identify timeout evidence owned by the model-download path, not arbitrary Jobs."""
    labels = _dict(meta.get("labels"))
    annotations = _dict(meta.get("annotations"))
    identity = " ".join(
        [
            str(meta.get("name") or ""),
            *(f"{key}={value}" for key, value in labels.items()),
            *(f"{key}={value}" for key, value in annotations.items()),
        ]
    ).lower()
    return any(marker in identity for marker in ("model-download", "model_downloader", "model-downloader", "artifact-download"))

def check_stuck_storage_pods(scope_namespace: str | None = None) -> CheckResult:
    n = "stuck-storage-pods"
    # Events and Jobs are read before the pod inventories: a pod created in between would otherwise
    # be absent from the inventory and its live mount failure discarded as a deleted pod's.
    ok2, eobj, err2 = kubectl_json(["get", "events", "-A", "--field-selector", "reason=FailedMount"])
    if not ok2:
        return CheckResult(n, INCONCLUSIVE, f"cannot list FailedMount events: {err2}")
    ok, pobj, err = kubectl_json(["get", "pods", "-A", "--field-selector=status.phase=Pending"])
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot list pods: {err}")
    ok5, sobj, err5 = kubectl_json(["get", "pods", "-A"])
    if not ok5:
        return CheckResult(n, INCONCLUSIVE, f"cannot list pods to tell which still exist: {err5}")
    ok3, dobj, err3 = kubectl_json(["get", "events", "-A", "--field-selector", "reason=DeadlineExceeded"])
    if not ok3:
        return CheckResult(n, INCONCLUSIVE, f"cannot list DeadlineExceeded events: {err3}")
    ok4, jobj, err4 = kubectl_json(["get", "jobs", "-A"])
    if not ok4:
        return CheckResult(n, INCONCLUSIVE, f"cannot list Jobs for model-download timeouts: {err4}")

    # FailedMount is definitive storage evidence. DeadlineExceeded on an event or failed Job is the
    # durable signal left by a model download that stopped making progress until its active deadline.
    stuck = set()
    stuck_uids: dict[str, set[str]] = {}
    uidless: set[str] = set()
    for e in _list((eobj or {}).get("items")):
        involved = _dict(_dict(e).get("involvedObject"))
        ref = f"{involved.get('namespace', '?')}/{involved.get('name', '?')}"
        stuck.add(ref)
        if involved.get("uid"):
            stuck_uids.setdefault(ref, set()).add(str(involved["uid"]))
        else:
            uidless.add(ref)
    # A pod that ran to completion under a Complete Job is not waiting on storage; its FailedMount
    # events are history. An ownerless Succeeded pod, or one whose Job is not Complete, stays judged.
    complete_jobs = {
        (str(_dict(_dict(j).get("metadata")).get("namespace", "?")), str(_dict(_dict(j).get("metadata")).get("name", "?")))
        for j in _list((jobj or {}).get("items"))
        if any(c.get("type") == "Complete" and c.get("status") == "True" for c in _conditions(j))
    }
    existing: set[str] = set()
    finished = set()
    for p in _list((sobj or {}).get("items")):
        meta = _dict(_dict(p).get("metadata"))
        ns = str(meta.get("namespace", "?"))
        ref = f"{ns}/{meta.get('name', '?')}"
        # A same-name replacement (a StatefulSet pod) is not the pod the event was about.
        if ref in uidless or ref not in stuck_uids or not meta.get("uid") or str(meta["uid"]) in stuck_uids[ref]:
            existing.add(ref)
        if _dict(_dict(p).get("status")).get("phase") == "Succeeded" and any(
            _dict(o).get("kind") == "Job" and (ns, str(_dict(o).get("name"))) in complete_jobs
            for o in _list(meta.get("ownerReferences"))
        ):
            finished.add(ref)
    # FailedMount events outlive the pod they describe, and only a pod that exists can be stuck.
    stuck &= existing
    stuck -= finished
    timed_out = set()
    for event in _list((dobj or {}).get("items")):
        involved = _dict(_dict(event).get("involvedObject"))
        if _is_model_download(involved):
            timed_out.add(f"{involved.get('namespace', '?')}/{involved.get('name', '?')}")
    for job in _list((jobj or {}).get("items")):
        meta = _dict(_dict(job).get("metadata"))
        if _is_model_download(meta) and any(
            c.get("type") == "Failed" and c.get("status") == "True" and c.get("reason") == "DeadlineExceeded"
            for c in _conditions(job)
        ):
            timed_out.add(f"{meta.get('namespace', '?')}/{meta.get('name', '?')}")

    # A Pending pod counts as storage-stuck ONLY when its scheduling condition points at storage
    # (an unbound PVC / volume). A pod Pending for CPU, affinity, or taint reasons is not a storage
    # problem, so it must not be attributed to storage.
    storage_kw = ("persistentvolumeclaim", "unbound", "volume")
    for p in _list((pobj or {}).get("items")):
        for c in _conditions(p):
            if c.get("type") == "PodScheduled" and c.get("status") != "True":
                msg = str(c.get("message") or "").lower()
                if any(k in msg for k in storage_kw):
                    meta = _dict(_dict(p).get("metadata"))
                    stuck.add(f"{meta.get('namespace', '?')}/{meta.get('name', '?')}")
    outside: set[str] = set()
    if scope_namespace is not None:
        # e2e mode: only workloads in the E2E footprint can block a run; the rest is informational.
        outside = {w for w in stuck | timed_out if not _in_e2e_footprint(w.split("/", 1)[0], scope_namespace)}
        stuck, timed_out = stuck - outside, timed_out - outside
    if stuck or timed_out:
        evidence = []
        if stuck:
            evidence.append(f"storage wait (Pending / FailedMount): {', '.join(sorted(stuck))}")
        if timed_out:
            evidence.append(f"DeadlineExceeded (possible stalled model download): {', '.join(sorted(timed_out))}")
        return CheckResult(
            n, WARN,
            f"workloads are stuck: {'; '.join(evidence)}",
            "Resolve the underlying storage problem; for DeadlineExceeded model downloads, also check cilium-mtu for per-node divergence before retrying.",
        )
    if outside:
        return CheckResult(n, OK, f"no E2E workloads stuck on storage; {len(outside)} stuck outside the E2E namespaces (informational, not gating): {', '.join(sorted(outside))}")
    return CheckResult(n, OK, "no workloads stuck waiting on storage or model downloads")

_SEAWEED_S3_SERVICE = "filer-s3"

_SEAWEED_ROLES = ("seaweed-master-", "seaweed-volume-", "seaweed-filer-", "seaweed-s3-")

def _is_forbidden(err: str) -> bool:
    low = err.lower()
    return "forbidden" in low or "unauthorized" in low

def _seaweed_pods() -> tuple[bool, list[dict[str, Any]], str]:
    """(read_ok, SeaweedFS pods, error), fetched once per run; empty when SeaweedFS is not installed."""
    ok, obj, err = _once_per_run(("seaweed-pods",), lambda: kubectl_json(["get", "pods", "-n", _SEAWEED_NS]))
    if not ok and _is_not_found(err):
        # Listing pods in a namespace that does not exist is NotFound: SeaweedFS is not installed, not unreadable.
        return True, [], ""
    pods = [p for p in _list((obj or {}).get("items")) if str(_dict(_dict(p).get("metadata")).get("name", "")).startswith("seaweed-")]
    return ok, pods, err

def _pod_ready(pod: Any) -> bool:
    # Ready=True implies Running; a Terminating pod can still read Ready, so it is excluded explicitly.
    return not _dict(_dict(pod).get("metadata")).get("deletionTimestamp") and _condition_is_true(pod, "Ready")

def _openbao_store_result(n: str, backend: str) -> CheckResult:
    # The per-namespace `minio-credentials` secret is minted PER PROJECT at test time (AIRM's
    # minio-credentials-fetcher ExternalSecret), so it does not exist pre-run and is not a gate.
    # The pre-run signal is that the OpenBao ClusterSecretStore is Ready, so the fetcher CAN mint it.
    ok, store, err = kubectl_json(["get", "clustersecretstores.external-secrets.io", "openbao-secret-store"])
    if not ok:
        if _is_not_found(err):
            return CheckResult(
                n, WARN, "OpenBao ClusterSecretStore 'openbao-secret-store' not found — the per-project minio-credentials fetcher cannot mint S3 creds",
                "Ensure ESO + the openbao-secret-store ClusterSecretStore are installed so AIWB S3 secrets can be minted at test time.",
            )
        return CheckResult(n, INCONCLUSIVE, f"{backend} present, but cannot read the openbao-secret-store ClusterSecretStore: {err}")
    if not _condition_is_true(store, "Ready"):
        return CheckResult(
            n, WARN, "OpenBao ClusterSecretStore 'openbao-secret-store' is not Ready — minio-credentials cannot be minted for projects",
            "Fix the ESO ClusterSecretStore (OpenBao reachability / source keys minio-api-access-key, minio-api-secret-key) before the run.",
        )
    return CheckResult(n, OK, f"{backend} ready and openbao-secret-store Ready (per-project minio-credentials can be minted)")

def check_object_storage() -> CheckResult:
    """Object storage the platform needs: SeaweedFS (current), or a MinIO tenant on clusters that
    still run it, plus the OpenBao store that mints per-project S3 credentials."""
    n = "object-storage"
    ok, pods, err = _seaweed_pods()
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot read the SeaweedFS namespace {_SEAWEED_NS}{' (identity: forbidden)' if _is_forbidden(err) else ''}: {err}")
    if pods:
        missing = [r.rstrip("-") for r in _SEAWEED_ROLES if not any(str(_dict(_dict(p).get("metadata")).get("name", "")).startswith(r) for p in pods)]
        if missing:
            return CheckResult(n, WARN, f"SeaweedFS is missing components: {', '.join(missing)}", "Restore the SeaweedFS instance before S3-dependent tests.")
        notready = sorted(str(_dict(_dict(p).get("metadata")).get("name")) for p in pods if not _pod_ready(p))
        if notready:
            return CheckResult(n, WARN, f"SeaweedFS pods not Ready: {', '.join(notready)}", "Recover the SeaweedFS pods; S3 reads and writes fail while they are down.")
        # Ready pods do not prove the Service workloads dial (filer-s3) has an endpoint.
        ok_ep, ep, ep_err = kubectl_json(["get", "endpoints", _SEAWEED_S3_SERVICE, "-n", _SEAWEED_NS])
        if not ok_ep:
            if _is_not_found(ep_err):
                return CheckResult(n, WARN, f"SeaweedFS S3 Service {_SEAWEED_S3_SERVICE} not found in {_SEAWEED_NS} — workloads cannot reach S3", "Restore the SeaweedFS S3 Service.")
            return CheckResult(n, INCONCLUSIVE, f"cannot read the SeaweedFS S3 Service endpoints: {ep_err}")
        if not any(_list(_dict(sub).get("addresses")) for sub in _list(_dict(ep).get("subsets"))):
            return CheckResult(n, WARN, f"SeaweedFS S3 Service {_SEAWEED_S3_SERVICE} has no ready endpoint — workloads cannot reach S3", "Recover the SeaweedFS S3 pods behind the Service.")
        return _openbao_store_result(n, "SeaweedFS")
    # No SeaweedFS pods. That is also exactly what a SeaweedFS outage looks like, so fall back to
    # MinIO only when the SeaweedFS namespace is confirmed absent; if it exists, SeaweedFS is
    # installed and zero pods is a problem.
    rc, _, ns_err = run(["kubectl", "get", "ns", _SEAWEED_NS], 15)
    if rc == 0:
        return CheckResult(n, WARN, f"SeaweedFS is installed ({_SEAWEED_NS} exists) but has no pods", "Restore the SeaweedFS instance; S3 is unavailable while it is down.")
    if not _is_not_found(ns_err):
        return CheckResult(n, INCONCLUSIVE, f"cannot confirm whether SeaweedFS is installed: {ns_err[:120]}")
    state, state_err = _secret_state("default-minio-tenant-env-configuration", _MINIO_NS)
    if state == "unreadable":
        return CheckResult(n, INCONCLUSIVE, f"SeaweedFS not found; cannot read the MinIO tenant secret: {state_err}")
    if state == "absent":
        rc2, _, err2 = run(["kubectl", "get", "ns", _MINIO_NS], 15)
        if rc2 != 0 and not _is_not_found(err2):
            return CheckResult(n, INCONCLUSIVE, f"cannot confirm MinIO tenant namespace: {err2[:120]}")
        return CheckResult(n, WARN, f"no object storage found: no SeaweedFS in {_SEAWEED_NS} and no MinIO tenant", "Provision SeaweedFS (or the MinIO tenant) or skip S3-data tests.")
    # The secret outlives a broken tenant, so also require the tenant's pods to be Ready.
    ok_m, mpods, merr = kubectl_json(["get", "pods", "-n", _MINIO_NS])
    if not ok_m:
        return CheckResult(n, INCONCLUSIVE, f"MinIO tenant secret present, but cannot read its pods: {merr}")
    mitems = _list(_dict(mpods).get("items"))
    mnotready = [str(_dict(_dict(p).get("metadata")).get("name", "?")) for p in mitems if not _pod_ready(p)]
    if not mitems or mnotready:
        return CheckResult(n, WARN, f"MinIO tenant pods in {_MINIO_NS} not Ready ({', '.join(mnotready) or 'no pods'}) — S3 is unavailable", "Recover the MinIO tenant before S3-data tests.")
    return _openbao_store_result(n, "MinIO tenant")

# Fill thresholds, from the incident where data disks reached ~99% and S3 PUTs failed with
# "failed to find writable volumes".
_FILL_WARN_PCT, _FILL_FAIL_PCT = 85, 95

def _seaweed_data_mounts(pod: Any) -> list[str]:
    """Mount paths of every PVC-backed data volume of the pod."""
    spec = _dict(_dict(pod).get("spec"))
    pvc_volumes = {str(_dict(v).get("name")) for v in _list(spec.get("volumes")) if _dict(_dict(v).get("persistentVolumeClaim"))}
    mounts: list[str] = []
    for c in _list(spec.get("containers")):
        for m in _list(_dict(c).get("volumeMounts")):
            path = str(_dict(m).get("mountPath") or "")
            if _dict(m).get("name") in pvc_volumes and path and path not in mounts:
                mounts.append(path)
    return mounts

_WEED_VOLUME_RE = re.compile(r"volume id:(\d+)\b(.*)")

_WEED_SIZE_LIMIT_RE = re.compile(r"volumeSizeLimit:(\d+)\s*MB")

_WEED_FREE_SLOTS_RE = re.compile(r"^\s*Disk\b.*?\bfree:(\d+)", re.M)

def _weed_free_slots(text: str) -> int | None:
    """Free volume slots from the `Disk ...(volume:N/M active:A free:F ...)` lines of `volume.list`
    (repeated per topology level, so only whether it is above zero is meaningful); None when absent."""
    found = _WEED_FREE_SLOTS_RE.findall(text)
    return sum(int(x) for x in found) if found else None

def _parse_weed_volume_list(text: str, collection: str) -> tuple[int, int] | None:
    """(volumes in the collection, of which writable) from `weed shell volume.list`.

    SeaweedFS stops assigning writes to a volume that is read-only or has reached the volume size
    limit, so writable means neither. None when the output is not something we can judge: no volume
    lines, no size limit, or a volume of the collection whose size is missing, so the check never
    counts a volume it could not measure as writable.
    """
    limit_match = _WEED_SIZE_LIMIT_RE.search(text)
    seen = 0
    # volume.list prints one entry per replica; a volume takes writes only when every replica can.
    replicas_ok: dict[str, bool] = {}
    unmeasured = False
    for line in text.splitlines():
        m = _WEED_VOLUME_RE.search(line)
        if not m:
            continue
        seen += 1
        rest = m.group(2)
        if not re.search(rf'collection:"{re.escape(collection)}"', rest):
            continue
        size = re.search(r"\bsize:(\d+)\b", rest)
        if not size:
            unmeasured = True
            continue
        full = limit_match is not None and int(size.group(1)) >= int(limit_match.group(1)) * 1024 * 1024
        ok = not full and "read_only:true" not in rest
        replicas_ok[m.group(1)] = replicas_ok.get(m.group(1), True) and ok
    total = len(replicas_ok)
    if not limit_match or unmeasured:
        return None
    if not seen:
        # A fresh topology has the size-limit header and no volumes yet: nothing is full, nothing to count.
        return 0, 0
    return total, sum(replicas_ok.values())

def check_object_store_fill(bucket: str = "default-bucket") -> CheckResult:
    """SeaweedFS data-disk use and writable volumes for the bucket collection. A full object store
    fails every S3 PUT while every pod stays Ready, so Ready proves nothing about it."""
    n = "object-store-fill"
    ok, pods, err = _seaweed_pods()
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot read the SeaweedFS namespace: {err}")
    volume_pods = sorted((p for p in pods if str(_dict(_dict(p).get("metadata")).get("name", "")).startswith("seaweed-volume-")),
                         key=lambda p: str(_dict(_dict(p).get("metadata")).get("name")))
    if not volume_pods:
        # Zero volume servers is also what a total outage looks like; only an absent namespace means
        # SeaweedFS is not installed.
        rc, _, ns_err = run(["kubectl", "get", "ns", _SEAWEED_NS], 15)
        if rc == 0:
            return CheckResult(n, WARN, f"SeaweedFS is installed ({_SEAWEED_NS} exists) but has no volume servers — the object store cannot take writes", "Restore the SeaweedFS volume servers.")
        if not _is_not_found(ns_err):
            return CheckResult(n, INCONCLUSIVE, f"cannot confirm whether SeaweedFS is installed: {ns_err[:120]}")
        return CheckResult(n, OK, "no SeaweedFS volume servers on this cluster — nothing to measure")
    worst, usage, unconfirmed = 0, [], []
    for p in volume_pods:
        name = str(_dict(_dict(p).get("metadata")).get("name"))
        mounts = _seaweed_data_mounts(p)
        if not mounts:
            unconfirmed.append(f"cannot find the data mount of {name}")
            continue
        for mount in mounts:
            rc, out, e = run(["kubectl", "exec", "-n", _SEAWEED_NS, name, "--", "df", "-P", mount], 30)
            if rc != 0:
                who = "the current identity may not exec into pods (read-only login)" if _is_forbidden(e) else f"exec failed: {e[:100]}"
                unconfirmed.append(f"cannot measure {name} {mount} disk use — {who}")
                continue
            cols = out.splitlines()[-1].split() if out else []
            pct = cols[4].rstrip("%") if len(cols) >= 5 else ""
            if not pct.isdigit():
                unconfirmed.append(f"unparseable df output from {name} {mount}: {out[-80:]!r}")
                continue
            worst = max(worst, int(pct))
            usage.append(f"{name} {mount} {pct}%")
    writable_note = ""
    total = writable = 0
    exhausted = False
    masters = sorted(str(_dict(_dict(p).get("metadata")).get("name")) for p in pods
                     if str(_dict(_dict(p).get("metadata")).get("name", "")).startswith("seaweed-master-") and _pod_ready(p))
    if not masters:
        unconfirmed.append("no Ready SeaweedFS master to list volumes from")
    else:
        rc, out, e = run(["kubectl", "exec", "-n", _SEAWEED_NS, masters[0], "--", "sh", "-c", "echo volume.list | weed shell"], 60)
        counts = _parse_weed_volume_list(out, bucket) if rc == 0 else None
        if rc != 0:
            who = "the current identity may not exec into pods (read-only login)" if _is_forbidden(e) else f"exec failed: {e[:100]}"
            unconfirmed.append(f"cannot list volumes — {who}")
        elif counts is None:
            unconfirmed.append("could not read `weed shell volume.list` output")
        else:
            total, writable = counts
            writable_note = f"; collection {bucket}: {writable} writable of {total} volume(s)"
            if writable == 0:
                free_slots = _weed_free_slots(out)
                if free_slots is None:
                    # Only a fresh topology (no volumes in any collection) with no slot figure is not a problem.
                    if total or _WEED_VOLUME_RE.search(out):
                        unconfirmed.append("cannot tell whether the collection can grow (no free volume slots reported)")
                elif free_slots > 0:
                    # Full volumes roll over: SeaweedFS allocates a new one while slots remain.
                    writable_note += f"; {free_slots} free volume slot(s) can grow it"
                else:
                    exhausted = True
    summary = f"disk use {', '.join(usage) or 'unmeasured'}{writable_note}"
    # A confirmed breach is a problem whatever else could not be measured; only then does an
    # unconfirmed probe turn the result inconclusive rather than OK.
    caveat = f"; also unconfirmed: {'; '.join(unconfirmed)}" if unconfirmed else ""
    if exhausted:
        return CheckResult(n, WARN, f"{summary} — no writable volume and no free slot to allocate one, S3 PUTs fail{caveat}", "Free or expand the SeaweedFS volume disks; S3-dependent tests fail until a volume is writable.")
    if worst >= _FILL_FAIL_PCT:
        return CheckResult(n, WARN, f"{summary} — at or above {_FILL_FAIL_PCT}%{caveat}", "Free or expand the SeaweedFS volume disks before the run; S3 writes fail when they fill.")
    if worst >= _FILL_WARN_PCT:
        return CheckResult(n, WARN, f"{summary} — at or above {_FILL_WARN_PCT}%{caveat}", "Free or expand the SeaweedFS volume disks soon; the run may fill them.")
    if unconfirmed:
        return CheckResult(n, INCONCLUSIVE, f"{summary}; {'; '.join(unconfirmed)}")
    return CheckResult(n, OK, summary)
