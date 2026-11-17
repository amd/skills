# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Checks for compute and versions: metrics, GPUs, application pods, images and release skew."""
from __future__ import annotations

import base64
import contextlib
import datetime
import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .shared import (
    CheckResult,
    INCONCLUSIVE,
    OK,
    WARN,
    _APP_NAMESPACES,
    _CLOCK_SKEW_SECONDS,
    _NO_E2E_NAMESPACE,
    _as_int,
    _condition_is_true,
    _conditions,
    _dict,
    _discover_test_namespace,
    _hf_get,
    _is_not_found,
    _list,
    _load_setup_module,
    _once_per_run,
    _rfc3339_age_seconds,
    _running_pods,
    _selector_arg,
    _service_proxy_get,
    kubectl_json,
    run,
)


_METRICS_BACKEND_NAMESPACES = ("otel-lgtm-stack", "monitoring", "mimir", "prometheus")

def _discover_metrics_backend() -> str | None:
    """Best-effort discovery of a Prometheus/Mimir query service ('<ns>/<svc>:9090') so a
    kubectl-only probe can query it through the API-server service proxy. None if not found."""
    ok, obj, _ = kubectl_json(["get", "svc", "-A"])
    if not ok:
        return None
    for s in _list((obj or {}).get("items")):
        meta = _dict(_dict(s).get("metadata"))
        ns, name = meta.get("namespace", ""), meta.get("name", "")
        if ns not in _METRICS_BACKEND_NAMESPACES:
            continue
        for port in _list(_dict(_dict(s).get("spec")).get("ports")):
            if _as_int(_dict(port).get("port")) == 9090:
                return f"{ns}/{name}:9090"
    return None

def _prom_instant_query(backend: str, promql: str) -> tuple[bool, Any]:
    """Run an instant PromQL query against '<ns>/<svc>:<port>' via the API-server service proxy
    (read-only, no port-forward). Returns (read_ok, data) where data is Prometheus' `data` object
    (resultType + result); read_ok=False means the query could not be read/parsed."""
    try:
        nssvc, port = backend.rsplit(":", 1)
        ns, svc = nssvc.split("/", 1)
    except ValueError:
        return False, None
    path = (
        f"/api/v1/namespaces/{ns}/services/{svc}:{port}/proxy/api/v1/query"
        f"?query={urllib.parse.quote(promql)}"
    )
    rc, out, _ = run(["kubectl", "get", "--raw", path], 30)
    if rc != 0:
        return False, None
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        return False, None
    if _dict(parsed).get("status") != "success":
        return False, None
    return True, _dict(parsed).get("data")

def _prom_numeric(data: Any) -> float | None:
    """Extract the numeric value from a Prometheus instant-query `data` object, handling both a
    scalar result ([ts, "v"]) and a vector result ([{value: [ts, "v"]}, ...]). None if empty/absent
    — so an empty result vector (no series matched) is distinct from a real 0."""
    rtype = _dict(data).get("resultType")
    result = _dict(data).get("result")
    if rtype == "scalar" and isinstance(result, list) and len(result) == 2:
        return _as_float(result[1])
    if rtype == "vector" and isinstance(result, list) and result:
        val = _dict(result[0]).get("value")
        if isinstance(val, list) and len(val) == 2:
            return _as_float(val[1])
    return None

def _pod_log_signature(namespace: str, pod: str, patterns: tuple[str, ...]) -> tuple[bool, str | None]:
    """Return (read_ok, matched_pattern_or_None). read_ok=False means the logs were unreadable."""
    rc, out, _ = run(["kubectl", "logs", "-n", namespace, pod, "--tail=200"], 30)
    if rc != 0:
        return False, None
    for pat in patterns:
        if pat in out:
            return True, pat
    return True, None

def _exporter_scrape_hosts(exporters: list[Any]) -> tuple[dict[str, set[int]], bool]:
    """host -> metrics ports a collector may scrape an exporter by: pod IPs with the pod's container
    ports, plus the ClusterIPs and DNS names of the Services selecting it with the Service ports
    (scraping through the exporter Service is the documented setup). An empty port set means the
    ports are not declared, so any port counts. The flag is False when the Services could not be
    read, so the map may be missing Service targets."""
    hosts: dict[str, set[int]] = {}

    def add(host: str, ports: set[int]) -> None:
        # A host seen with undeclared ports stays port-agnostic rather than narrowing to one declaration.
        if host in hosts and not (hosts[host] and ports):
            hosts[host] = set()
        else:
            hosts.setdefault(host, set()).update(ports)

    for p in exporters:
        status = _dict(_dict(p).get("status"))
        containers = [_dict(c) for c in _list(_dict(_dict(p).get("spec")).get("containers"))]
        # Only the exporter's own container serves metrics: a sidecar's port is not a scrape target.
        # With several containers and none named like an exporter, the metrics port is unknown, so the
        # pod stays port-agnostic rather than guessing.
        if len(containers) > 1:
            containers = [c for c in containers if "exporter" in str(c.get("name", ""))]
        ports = {
            port
            for c in containers
            for e in _list(c.get("ports"))
            if isinstance(port := _dict(e).get("containerPort"), int)
        }
        for ip in [status.get("podIP"), *(_dict(e).get("ip") for e in _list(status.get("podIPs")))]:
            if ip:
                add(str(ip), ports)
    ok, svcs, _ = kubectl_json(["get", "svc", "-A"])
    if not ok:
        return hosts, False
    for svc in _list((svcs or {}).get("items")):
        meta, spec = _dict(_dict(svc).get("metadata")), _dict(_dict(svc).get("spec"))
        selector = _dict(spec.get("selector"))
        if not selector or not any(
            _dict(_dict(p).get("metadata")).get("namespace") == meta.get("namespace")
            and selector.items() <= _dict(_dict(_dict(p).get("metadata")).get("labels")).items()
            for p in exporters
        ):
            continue
        name, ns = meta.get("name"), meta.get("namespace")
        ports = {port for e in _list(spec.get("ports")) if isinstance(port := _dict(e).get("port"), int)}
        for ip in [spec.get("clusterIP"), *_list(spec.get("clusterIPs"))]:
            if ip and ip != "None":
                add(str(ip), ports)
        # A headless Service's DNS name resolves straight to the pods, so it is scraped on the target
        # port with no Service-port translation; a named target port cannot be resolved here, so that
        # name stays port-agnostic.
        dns_ports = ports
        if spec.get("clusterIP") == "None":
            targets = [_dict(e).get("targetPort", _dict(e).get("port")) for e in _list(spec.get("ports"))]
            dns_ports = {t for t in targets if isinstance(t, int)} if all(isinstance(t, int) for t in targets) else set()
        if name and ns:
            for dns in (f"{name}.{ns}", f"{name}.{ns}.svc", f"{name}.{ns}.svc.cluster.local"):
                add(dns, dns_ports)
    return hosts, True

def _collector_scrape_failure(namespace: str, pod: str, exporter_ips: dict[str, set[int]]) -> tuple[bool, str | None, bool]:
    """(read_ok, a scrape-failure log line, whether other scrape failures were seen) for a scrape of one of the exporter pods. A collector also
    scrapes unrelated targets (e.g. the AIRM custom-metrics job), whose failures say nothing about
    the GPU exporter, so only a failure naming an exporter pod's address counts."""
    rc, out, _ = run(["kubectl", "logs", "-n", namespace, pod, "--tail=200"], 30)
    if rc != 0:
        return False, None, False
    # An IPv6 host is always bracketed in a target and may extend the exporter's address (a longer
    # address sharing its prefix), so it must match through the closing bracket; an IPv4 host ends at
    # the port separator.
    def target(host: str, ports: set[int]) -> str:
        bare = rf"\[{re.escape(host)}\]" if ":" in host else re.escape(host)
        # Matching the port too keeps an unrelated target on a shared host IP (hostNetwork pods
        # share the node's address) from being read as the exporter's.
        return rf"{bare}:(?:{'|'.join(str(p) for p in sorted(ports))})(?![0-9])" if ports else (bare if ":" in host else rf"{bare}(?=[:\\\"])")

    hosts = "|".join(target(host, ports) for host, ports in exporter_ips.items())
    pattern = re.compile(rf"instance=\\?\"(?:{hosts})") if hosts else None
    unclassified = False
    for line in out.splitlines():
        if "Failed to scrape" not in line:
            continue
        if pattern and pattern.search(line):
            return True, line, False
        unclassified = True
    return True, None, unclassified

def _active_gpu_workload_present() -> tuple[bool, bool]:
    """(read_ok, present). present is True when a Running pod requests/limits amd.com/gpu — i.e.
    a GPU workload is actually consuming GPUs, so GPU series are expected to exist. On an idle
    cluster (no such pod) a zero series count is legitimate, not a broken pipeline."""
    ok, obj, _ = _running_pods()
    if not ok:
        return False, False
    for p in _list((obj or {}).get("items")):
        for c in _list(_dict(_dict(p).get("spec")).get("containers")):
            res = _dict(_dict(c).get("resources"))
            for key in ("requests", "limits"):
                if _as_int(_dict(res.get(key)).get("amd.com/gpu", 0)) > 0:
                    return True, True
    return True, False

def _gpu_capacity() -> tuple[bool, int]:
    """(read_ok, total amd.com/gpu capacity across all nodes).

    read_ok is False when the node list could not be read — a failed read is never a confirmed
    "this cluster has no GPUs", so callers must not turn it into a not-applicable pass.
    """
    ok, obj, _ = _once_per_run(("nodes",), lambda: kubectl_json(["get", "nodes"]))
    if not ok:
        return False, 0
    total = 0
    for x in _list((obj or {}).get("items")):
        total += _as_int(_dict(_dict(_dict(x).get("status")).get("capacity")).get("amd.com/gpu", 0))
    return True, total

# The AMD GPU operator labels its exporter pods `metrics-exporter`; older charts used `device-metrics-exporter`.
_GPU_EXPORTER_SELECTOR = "app.kubernetes.io/name in (metrics-exporter,device-metrics-exporter)"

def check_gpu_metrics(metrics_backend: str | None = None, require_gpu: bool = False) -> CheckResult:
    n = "gpu-metrics"
    ok, obj, err = kubectl_json(["get", "pods", "-A", "-l", _GPU_EXPORTER_SELECTOR])
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot query the GPU metrics exporter: {err}")
    exporters = _list((obj or {}).get("items"))
    if not exporters:
        # A missing exporter on a cluster that has no GPUs at all is not an unproven pipeline: there
        # are no GPUs to export metrics for, so there is nothing this check can ever confirm. Settle
        # it on the same fact gpu-schedulable reads (amd.com/gpu capacity is 0) rather than assuming
        # it from the exporter's absence — and only when that read SUCCEEDED and the caller did not
        # assert GPUs are expected, because a failed read or --require-gpu leaves the pipeline
        # genuinely unproven.
        cap_ok, capacity = _gpu_capacity()
        if cap_ok and capacity == 0 and not require_gpu:
            return CheckResult(n, OK, "no GPU nodes on this cluster (amd.com/gpu capacity is 0) — there is no GPU metrics pipeline to check")
        return CheckResult(n, INCONCLUSIVE, "GPU metrics exporter not found — no pipeline to check (expected before install or on a cluster without GPUs)")
    # The exporter is a per-node DaemonSet: read EVERY pod, so a single silently-failing node is
    # not masked by a healthy replica. Ready proves nothing — the failure is in the logs.
    for p in exporters:
        meta = _dict(_dict(p).get("metadata"))
        ns = meta.get("namespace", "?")
        name = meta.get("name", "?")
        read_ok, hit = _pod_log_signature(ns, name, ("i/o timeout", "Failed to watch"))
        if not read_ok:
            return CheckResult(n, INCONCLUSIVE, f"exporter pod {ns}/{name} logs are unreadable — cannot confirm it is exporting metrics (do NOT assume Running == healthy)")
        if hit:
            return CheckResult(
                n, WARN,
                f"the GPU metrics exporter ({ns}/{name}) is Running but failing — it cannot reach the kube-apiserver (log signature: '{hit}'), so metric labels are blank and dashboards show no GPU utilization; broken hop: exporter -> kube-apiserver",
                "Restore node -> control-plane network reachability (check Cilium MTU / network policy) for the exporter pod.",
            )
    # The OpenTelemetry operator names each collector after its CR (otel-collector-metrics-airm-collector),
    # so only its component label is common to all of them.
    ok2, cobj, err2 = kubectl_json(["get", "pods", "-n", "otel-lgtm-stack", "-l", "app.kubernetes.io/component=opentelemetry-collector"])
    if not ok2:
        return CheckResult(n, INCONCLUSIVE, f"cannot query the OTel collector: {err2} — the exporter->collector scrape hop is unconfirmed")
    collectors = _list((cobj or {}).get("items"))
    if not collectors:
        return CheckResult(n, INCONCLUSIVE, "exporter present but no OTel collector found — the scrape hop that carries GPU metrics to the backend is missing; pipeline health unconfirmed")
    exporter_ips, services_read = _exporter_scrape_hosts(exporters)
    unclassifiable = False
    for c in collectors:
        name = _dict(_dict(c).get("metadata")).get("name", "?")
        read_ok, hit, other_failure = _collector_scrape_failure("otel-lgtm-stack", name, exporter_ips)
        if not read_ok:
            return CheckResult(n, INCONCLUSIVE, f"OTel collector {name} logs are unreadable — cannot confirm it is scraping the exporter")
        if hit:
            return CheckResult(
                n, WARN,
                f"the OTel collector ({name}) cannot scrape the GPU metrics exporter (scrape failure), so GPU metrics never reach the backend; broken hop: OTel collector -> exporter",
                "Check the exporter Service/endpoints and the collector's scrape config.",
            )
        unclassifiable = unclassifiable or other_failure
    if unclassifiable and not services_read:
        return CheckResult(
            n, INCONCLUSIVE,
            "an OTel collector logs scrape failures, but the Services could not be read to tell whether any target is the GPU exporter's Service — the exporter->collector scrape hop is unconfirmed",
            "Grant read access to Services (kubectl get svc -A) and re-run.",
        )
    # Absence of the xfail signatures is not positive proof; the anti-silence contract
    # (scenario 7) requires proving series actually reach the backend. Query the metrics backend
    # (Prometheus/Mimir) through the API-server service proxy — a read-only kubectl call, no
    # port-forward — for a current GPU series. Prometheus staleness means a non-empty result IS
    # fresh: OK. Empty result -> the pipeline is not delivering -> WARN. Backend not
    # discoverable/reachable -> INCONCLUSIVE (never a pass we did not prove).
    backend = metrics_backend or _discover_metrics_backend()
    if not backend:
        return CheckResult(
            n, INCONCLUSIVE,
            "exporter and collector present with no failure signature, but the metrics backend could not be found to positively confirm delivery",
            "Pass --metrics-backend <ns>/<svc>:<port> (Prometheus/Mimir) so the check can query GPU series freshness.",
        )
    # Require a POSITIVE count of current GPU series that carry a non-blank workload_id — the
    # incident's signature is the exporter losing kube-apiserver access so labels go blank and the
    # UI shows 0 GPU/VRAM even while series technically exist. A count of 0 (or a scalar zero) is
    # therefore NOT a pass.
    read_ok, data = _prom_instant_query(backend, 'count(gpu_gfx_activity{workload_id!=""})')
    if not read_ok:
        return CheckResult(
            n, INCONCLUSIVE,
            f"exporter and collector present with no failure signature, but the metrics backend query ({backend}) failed — delivery unconfirmed",
            "Confirm the metrics backend is reachable / the --metrics-backend target is correct.",
        )
    count = _prom_numeric(data)
    if count is not None and count > 0:
        return CheckResult(n, OK, f"{int(count)} current GPU series with a non-blank workload_id in the backend ({backend}) — exporter/collector healthy and metrics are arriving")

    workload_read_ok, active_workload = _active_gpu_workload_present()
    if not workload_read_ok:
        return CheckResult(
            n, INCONCLUSIVE,
            f"the metrics backend ({backend}) has no current GPU series, but active GPU workloads cannot be read — zero-series health is unconfirmed",
            "Grant read access to running pods, then re-run so the check can distinguish an idle cluster from a broken pipeline.",
        )
    if not active_workload:
        return CheckResult(
            n, OK,
            f"no active GPU workload is running, so zero workload-labelled GPU series in the backend ({backend}) is expected",
        )
    return CheckResult(
        n, WARN,
        f"an active GPU workload is running but no current GPU metrics with a non-blank workload_id are in the backend ({backend}) — series are not reaching it or labels are blank; the pipeline is not delivering",
        "Check the exporter -> kube-apiserver access and the exporter -> collector -> backend scrape path; GPU dashboards will be empty.",
    )

_WAITING_FATAL = ("CrashLoopBackOff", "CreateContainerConfigError", "ImagePullBackOff", "ErrImagePull")

# A pod that has been NotReady longer than this is a real problem; younger than this it is likely
# still starting up or mid-rolling-update, so a bare NotReady is not yet flagged (a fatal waiting
# reason or a restart count still is, immediately, regardless of age).
_NOTREADY_GRACE_SECONDS = 300

def _pod_age_seconds(pod: dict[str, Any]) -> float | None:
    ts = _dict(_dict(pod).get("status")).get("startTime") or _dict(_dict(pod).get("metadata")).get("creationTimestamp")
    if not ts:
        return None
    # Kubernetes timestamps are RFC3339, usually "...Z" at second precision but sometimes with
    # fractional seconds (RFC3339Nano) or a numeric offset. Parse via fromisoformat (which handles
    # fractional seconds and offsets) after normalizing a trailing Z; a parse failure yields None.
    try:
        started = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=datetime.timezone.utc)
    return (datetime.datetime.now(datetime.timezone.utc) - started).total_seconds()


# How far back a restart count still counts as flapping. A node outage restarts every pod on it
# once or more, and the cumulative count keeps those restarts for the pod's lifetime, so a pod that
# has run cleanly since would block every later run. A pod that is really flapping keeps dying, so
# its last termination stays inside this window; a slow leak that kills it every hour or two does too.
_RESTART_HISTORY_SECONDS = 6 * 3600


def _restarts_are_stale(container: dict[str, Any]) -> bool:
    """True when the container's last termination is known and older than the restart history
    window, so it has run without dying since. An unknown or unparseable time proves nothing."""
    last = _dict(_dict(container.get("lastState")).get("terminated"))
    age = _rfc3339_age_seconds(str(last.get("finishedAt") or ""))
    return age is not None and age > _RESTART_HISTORY_SECONDS

def _recently_restarted(container: dict[str, Any], window_seconds: float) -> str | None:
    """The last termination reason when the container last died inside the window, else None.

    A cumulative restart count cannot tell a pod that flapped last week from one that is failing its
    startup right now; the last termination time can. It is only reported once the container has
    actually been restarted, so a first start is never read as a restart."""
    if _as_int(container.get("restartCount", 0)) < 1:
        return None
    last = _dict(_dict(container.get("lastState")).get("terminated"))
    age = _rfc3339_age_seconds(str(last.get("finishedAt") or ""))
    # A termination in the future beyond clock skew is a bad timestamp, not proof of recency.
    if age is None or age > window_seconds or age < -_CLOCK_SKEW_SECONDS:
        return None
    return str(last.get("reason") or "restarted")

def check_app_pods_stable(restart_threshold: int, recent_minutes: int = 30) -> CheckResult:
    """Flag app pods that are not stably running — a container (init or main) waiting in a fatal
    reason (CrashLoopBackOff / CreateContainerConfigError / ImagePullBackOff / ErrImagePull), a
    pod that has stayed NotReady past the startup grace, a pod that is Ready NOW but has
    crash-looped (restarts over the threshold, last one within the restart history window), or a
    container that restarted within the last `recent_minutes` minutes.

    A Ready pod with a high restart count is the silent case a plain Available check misses: the
    pod recovered but was flapping, which is how a downstream call intermittently 'fetch failed'.
    A recent restart catches the same thing below the threshold: an API that died during startup
    a few times before the release settled.
    """
    n = "app-pods-stable"
    window_seconds = recent_minutes * 60
    problems: list[str] = []
    read_any = False
    for ns in _APP_NAMESPACES:
        ok, obj, err = kubectl_json(["get", "pods", "-n", ns])
        if not ok:
            if _is_not_found(err):
                continue  # namespace not installed yet — nothing to judge here
            return CheckResult(n, INCONCLUSIVE, f"cannot list pods in ns {ns}: {err}")
        read_any = True
        for pod in _list((obj or {}).get("items")):
            meta, status = _dict(_dict(pod).get("metadata")), _dict(_dict(pod).get("status"))
            name = meta.get("name", "?")
            # Skip pods that are meant to run to completion (Jobs), not stay up.
            if status.get("phase") in ("Succeeded", "Failed") and _owned_by_job(meta):
                continue
            # Init containers only complete or fail; a fatal waiting reason on one blocks the pod
            # from ever starting its main containers, so surface it (named as init).
            for c in _list(status.get("initContainerStatuses")):
                wr = _dict(_dict(_dict(c).get("state")).get("waiting")).get("reason", "")
                if wr in _WAITING_FATAL:
                    problems.append(f"{ns}/{name}[init:{_dict(c).get('name', '?')}]={wr}")
                elif recent_reason := _recently_restarted(_dict(c), window_seconds):
                    restarts = _as_int(_dict(c).get("restartCount", 0))
                    problems.append(f"{ns}/{name}[init:{_dict(c).get('name', '?')}]=restarted-within-{recent_minutes}m({recent_reason}, {restarts} restart(s) total)")
            containers = _list(status.get("containerStatuses"))
            if not containers:
                # A pod that never got scheduled has NO containerStatuses at all — the normal shape
                # of a Pending/Unschedulable pod. Skipping it silently would let the check assert
                # "all app pods Running/Ready" while a workload cannot start, so surface it once it
                # is past the startup grace (a just-created pod is still being scheduled).
                # Only when the pod is genuinely NOT scheduled — a scheduled pod with no
                # containerStatuses is running its init containers, already covered above, and
                # labelling it "Unscheduled" would be wrong.
                sched = next((c for c in _conditions(pod) if c.get("type") == "PodScheduled"), None)
                scheduled = sched is not None and sched.get("status") == "True"
                if not scheduled and status.get("phase") not in ("Succeeded", "Failed"):
                    # A pod with no containerStatuses that is not affirmatively scheduled cannot be
                    # certified healthy: it may be genuinely unschedulable or have malformed/empty
                    # conditions. Surface it once past the startup grace rather than falling through
                    # to the all-clear (an old Pending pod with empty conditions must not pass).
                    age = _pod_age_seconds(pod)
                    if age is None or age >= _NOTREADY_GRACE_SECONDS:
                        reason = _dict(sched).get("reason") if sched is not None else None
                        reason = reason or status.get("phase") or "NotScheduled"
                        problems.append(f"{ns}/{name}=Unscheduled({reason})")
                continue
            for c in containers:
                c = _dict(c)
                waiting_reason = _dict(_dict(c.get("state")).get("waiting")).get("reason", "")
                if waiting_reason in _WAITING_FATAL:
                    problems.append(f"{ns}/{name}[{c.get('name', '?')}]={waiting_reason}")
                    continue
                restarts = _as_int(c.get("restartCount", 0))
                recent_reason = _recently_restarted(c, window_seconds)
                if c.get("ready") and restarts >= restart_threshold and not _restarts_are_stale(c):
                    last = _dict(_dict(c.get("lastState")).get("terminated"))
                    reason = last.get("reason", "restarted")
                    problems.append(f"{ns}/{name}[{c.get('name', '?')}]=Ready-but-{restarts}-restarts({reason})")
                elif recent_reason:
                    problems.append(f"{ns}/{name}[{c.get('name', '?')}]=restarted-within-{recent_minutes}m({recent_reason}, {restarts} restart(s) total)")
                elif not c.get("ready") and status.get("phase") not in ("Succeeded",):
                    # Don't flag a pod that is still within its startup grace — that is a rollout in
                    # progress, not a fault. An unknown age (no timestamp) errs toward surfacing.
                    age = _pod_age_seconds(pod)
                    if age is None or age >= _NOTREADY_GRACE_SECONDS:
                        problems.append(f"{ns}/{name}[{c.get('name', '?')}]=NotReady")
    if not read_any:
        return CheckResult(n, INCONCLUSIVE, f"no app namespaces present ({', '.join(_APP_NAMESPACES)}) — nothing to check (expected before install)")
    if problems:
        return CheckResult(
            n, WARN,
            f"app pods not stably running: {', '.join(problems)}",
            "A fatal waiting reason blocks the component; a Ready-but-repeatedly-restarted or recently restarted pod was failing and can cause intermittent downstream failures (e.g. UI 'fetch failed'). Inspect the pod's logs (including --previous) and events, and its dependencies.",
        )
    return CheckResult(n, OK, f"all app pods in {', '.join(_APP_NAMESPACES)} Running/Ready with no crash-loop, fatal waiting state, or restart in the last {recent_minutes}m")

def _owned_by_job(meta: dict[str, Any]) -> bool:
    return any(_dict(ref).get("kind") == "Job" for ref in _list(_dict(meta).get("ownerReferences")))

def _check_deploy_healthy(name: str, namespace: str, check_name: str) -> CheckResult:
    ok, obj, err = kubectl_json(["get", "deploy", name, "-n", namespace])
    if not ok:
        if _is_not_found(err):
            return CheckResult(check_name, INCONCLUSIVE, f"{name} deployment not found in ns {namespace} (expected before install)")
        return CheckResult(check_name, INCONCLUSIVE, f"cannot read {name} deploy: {err}")
    avail = _condition_is_true(obj, "Available")
    ready = _as_int(_dict(_dict(obj).get("status")).get("readyReplicas", 0))
    if avail and ready:
        return CheckResult(check_name, OK, f"{name} Available (readyReplicas={ready})")
    return CheckResult(
        check_name, WARN,
        f"{name} not Available (readyReplicas={ready})",
        f"Check {name} pod (CreateContainerConfigError => a required secret/config is missing).",
    )

def check_aiwb_api() -> CheckResult:
    return _check_deploy_healthy("aiwb-api", "aiwb", "aiwb-api-healthy")

def check_aiwb_ui() -> CheckResult:
    return _check_deploy_healthy("aiwb-ui", "aiwb", "aiwb-ui-healthy")

def _e2e_hf_token() -> tuple[str | None, str]:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token, ""
    ok, secret, err = kubectl_json(["get", "secret", "e2e-hf-token", "-n", "e2e-apiserver-testing"])
    if not ok:
        if _is_not_found(err):
            return None, "no token"
        return None, f"cannot read runner token secret: {err}"
    encoded = _dict(_dict(secret).get("data")).get("token")
    if not isinstance(encoded, str):
        return None, "no token"
    try:
        return base64.b64decode(encoded, validate=True).decode("utf-8"), ""
    except (ValueError, UnicodeDecodeError):
        return None, "runner token secret has no readable token"

def _fine_tuning_models() -> tuple[list[Any] | None, str]:
    """(entries of GET /v1/fine-tuning/models, error), read once per run through the service proxy."""
    status, body, error = _once_per_run(
        ("fine-tuning-models",), lambda: _service_proxy_get("aiwb", "aiwb-api", "/v1/fine-tuning/models")
    )
    if status != 200:
        return None, f"cannot read fine-tuning models: {error or f'HTTP {status}'}"
    try:
        data = _dict(json.loads(body)).get("data")
    except json.JSONDecodeError:
        return None, "fine-tuning models response is not JSON"
    if not isinstance(data, list) or not all(isinstance(model, dict) for model in data):
        return None, "fine-tuning models response is not a list of models under 'data'"
    return data, ""

def check_finetuning_hf_token() -> CheckResult:
    name = "finetuning-hf-token"
    token, error = _e2e_hf_token()
    if token is None:
        return CheckResult(name, WARN if error == "no token" else INCONCLUSIVE, error,
                           "Provide HF_TOKEN or the e2e-apiserver-testing/e2e-hf-token runner secret.")
    status, _, error = _hf_get("https://huggingface.co/api/whoami-v2", token)
    if status != 200:
        if status in (401, 403):
            return CheckResult(name, WARN, "token rejected", "Replace the HF token used by the E2E run.")
        return CheckResult(name, INCONCLUSIVE, f"cannot verify token: {error or f'HTTP {status}'}")
    models, error = _fine_tuning_models()
    if models is None:
        return CheckResult(name, INCONCLUSIVE, error)
    for model in models:
        model_data = _dict(model)
        if model_data.get("hfTokenRequired") is not True:
            continue
        model_name = model_data.get("canonicalName")
        if not isinstance(model_name, str) or not model_name:
            return CheckResult(name, INCONCLUSIVE, "hfTokenRequired model has no canonicalName")
        # auth-check proves permission to download; the model-info endpoint answers 200 with public
        # metadata for a gated repo even when the token is denied its weights.
        status, _, error = _hf_get(
            f"https://huggingface.co/api/models/{urllib.parse.quote(model_name, safe='/')}/auth-check", token
        )
        if status in (401, 403, 404):
            return CheckResult(name, WARN, f"no access to {model_name}", "Grant the E2E HF token access to that model.")
        if status != 200:
            return CheckResult(name, INCONCLUSIVE, f"cannot verify access to {model_name}: {error or f'HTTP {status}'}")
    return CheckResult(name, OK, "HF token is accepted and can read every gated fine-tuning model")

# Tag values that make an AIM speech-to-text capable. Mirrors apps/api/aiwb/specs/libraries/SpeechToText.py.
_TRANSCRIPTION_TAGS = {"transcription", "speech-to-text", "automatic-speech-recognition"}

# Tag values that make an AIM diffusion capable. Mirrors apps/api/aiwb/specs/libraries/Diffusion.py.
_DIFFUSION_TAGS = {"diffusion", "text-to-video"}

# The diffusion suite passes over models also tagged for another capability. Mirrors
# OTHER_CAPABILITY_TAG_VALUES in apps/api/aiwb/specs/libraries/Diffusion.py.
_OTHER_THAN_DIFFUSION_TAGS = {"chat"} | _TRANSCRIPTION_TAGS

def check_e2e_local_queues(test_namespace: str | None) -> CheckResult:
    name = "e2e-local-queues"
    if test_namespace is None:
        ns_ok, test_namespace, ns_err = _discover_test_namespace()
        if not ns_ok:
            return CheckResult(name, INCONCLUSIVE, f"cannot list namespaces to find the E2E namespace: {ns_err}")
        if test_namespace is None:
            return CheckResult(name, OK, _NO_E2E_NAMESPACE)
    ok, obj, err = kubectl_json(["get", "localqueues.kueue.x-k8s.io", "-n", test_namespace])
    if not ok:
        return CheckResult(name, INCONCLUSIVE, f"cannot read LocalQueues in {test_namespace}: {err}")
    if not _list(_dict(obj).get("items")):
        return CheckResult(
            name, WARN, f"namespace {test_namespace} has no Kueue LocalQueue",
            f"Create a LocalQueue in {test_namespace}; without one every Job waits on admission until its check times out.",
        )
    return CheckResult(name, OK, f"namespace {test_namespace} has a Kueue LocalQueue")

def check_fine_tuning_models() -> CheckResult:
    name = "fine-tuning-models"
    models, error = _fine_tuning_models()
    if models is None:
        return CheckResult(name, INCONCLUSIVE, error)
    if not models:
        return CheckResult(
            name, WARN, "GET /v1/fine-tuning/models lists no model",
            "Publish an AIMClusterProfile and a recipe compatible with this cluster's GPUs; every fine-tuning test is unrunnable without one.",
        )
    gated = sorted(str(_dict(m).get("canonicalName")) for m in models if _dict(m).get("hfTokenRequired") is True)
    return CheckResult(
        name, OK,
        f"{len(models)} fine-tuning model(s); hfTokenRequired: {', '.join(gated) if gated else 'none'}",
    )

def check_optional_capability_prerequisites() -> CheckResult:
    """Name the tests that cannot run because the cluster lacks a LoRA recipe, a transcription AIM or a diffusion AIM."""
    name = "optional-capability-prerequisites"
    models, error = _fine_tuning_models()
    if models is None:
        return CheckResult(name, INCONCLUSIVE, error)
    ok, aims, aims_err = kubectl_json(["get", "aimclustermodels.aim.eai.amd.com", "-A"])
    if not ok:
        return CheckResult(name, INCONCLUSIVE, f"cannot read AIM cluster models: {aims_err}")
    missing: list[str] = []
    if not any(_dict(m).get("trainingType") == "lora" for m in models):
        missing.append("no LoRA-capable fine-tuning recipe (adapter fine-tuning and adapter-serving scenarios are not runnable; leave run_adapter_gap_tests off)")
    aim_tags = [
        {t for t in _list(_dict(_dict(_dict(_dict(a).get("status")).get("imageMetadata")).get("model")).get("tags")) if isinstance(t, str)}
        for a in _list(_dict(aims).get("items"))
    ]
    if not any(_TRANSCRIPTION_TAGS & tags for tags in aim_tags):
        missing.append("no Whisper/speech-to-text AIM (the transcription suite is not runnable; leave run_capability_tests off)")
    if not any(_DIFFUSION_TAGS & tags and not _OTHER_THAN_DIFFUSION_TAGS & tags for tags in aim_tags):
        missing.append("no diffusion-only (text-to-video) AIM (the diffusion suite is not runnable; leave run_capability_tests off)")
    if missing:
        return CheckResult(name, WARN, "; ".join(missing), "Publish the missing model, or leave the matching run_* input off so those tests are excluded and counted.")
    return CheckResult(name, OK, "a LoRA-capable recipe, a speech-to-text AIM and a diffusion-only AIM are all present")

def check_team_member_sign_in(core_repo: str) -> CheckResult:
    """Reuse the setup script's own credential resolution and grant, so this proves what the suites will do."""
    name = "team-member-sign-in"
    try:
        setup = _load_setup_module(core_repo)
    except Exception as error:  # noqa: BLE001 - an unloadable helper is unproven, not healthy
        return CheckResult(name, INCONCLUSIVE, f"cannot load the E2E setup helper: {type(error).__name__}")
    if setup is None:
        return CheckResult(name, INCONCLUSIVE, f"tooling/e2e/setup_e2e_prerequisites.py not found under {core_repo}")
    try:
        config_problems, config = setup.verify_runner_config(setup.RUNNER_NAMESPACE)
        credential_problems, credentials = setup.verify_credentials(setup.RUNNER_NAMESPACE)
        problems = [*config_problems, *credential_problems]
        if not problems:
            # verify_team_member prints a progress line; stdout must stay one JSON document under --json.
            with contextlib.redirect_stdout(io.StringIO()):
                problems = setup.verify_team_member(config, credentials)
    except Exception as error:  # noqa: BLE001
        return CheckResult(name, INCONCLUSIVE, f"cannot verify the team member account: {type(error).__name__}")
    if problems:
        return CheckResult(name, WARN, str(problems[0].what), str(problems[0].fix))
    return CheckResult(name, OK, "the team member account signs in")

def check_gpu_schedulable(require_gpu: bool, test_namespace: str | None) -> CheckResult:
    n = "gpu-schedulable"
    ok, obj, err = _once_per_run(("nodes",), lambda: kubectl_json(["get", "nodes"]))
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot read nodes: {err}")
    nodes = _list((obj or {}).get("items"))
    capacity = 0
    allocatable = 0
    for x in nodes:
        st = _dict(_dict(x).get("status"))
        capacity += _as_int(_dict(st.get("capacity")).get("amd.com/gpu", 0))
        allocatable += _as_int(_dict(st.get("allocatable")).get("amd.com/gpu", 0))
    if capacity == 0 and not require_gpu:
        return CheckResult(n, OK, "no GPU nodes on this cluster (amd.com/gpu capacity is 0) — GPU workloads not applicable")
    if allocatable == 0:
        return CheckResult(
            n, WARN,
            f"no amd.com/gpu allocatable (capacity={capacity}) — GPU workloads/tests will stay Pending",
            "Check the AMD GPU Operator: drivers may not have rolled out even though the operator is installed.",
        )
    if test_namespace is None:
        ns_ok, test_namespace, ns_err = _discover_test_namespace()
        if not ns_ok:
            return CheckResult(
                n, INCONCLUSIVE,
                f"{allocatable} amd.com/gpu allocatable, but Kueue admission is unverified (cannot list namespaces to find the E2E namespace: {ns_err})",
            )
        if test_namespace is None:
            return CheckResult(n, OK, f"{allocatable} amd.com/gpu allocatable; Kueue admission not checked: {_NO_E2E_NAMESPACE}")
    # GPUs are allocatable; confirm the E2E namespace's own LocalQueue resolves to a ClusterQueue
    # with GPU quota. An unrelated queue elsewhere in the cluster cannot admit jobs from this ns.
    ok2, lqobj, err2 = kubectl_json(["get", "localqueues.kueue.x-k8s.io", "-n", test_namespace])
    if not ok2:
        return CheckResult(
            n, INCONCLUSIVE,
            f"{allocatable} amd.com/gpu allocatable, but Kueue admission for namespace {test_namespace} is unverified (cannot read LocalQueues: {err2})",
            f"Confirm namespace {test_namespace} has a readable LocalQueue bound to a ClusterQueue with amd.com/gpu quota.",
        )
    local_queues = _list((lqobj or {}).get("items"))
    if not local_queues:
        return CheckResult(
            n, WARN,
            f"{allocatable} amd.com/gpu allocatable, but namespace {test_namespace} has no LocalQueue — GPU jobs submitted there will not admit",
            f"Create a LocalQueue in {test_namespace} bound to a ClusterQueue with amd.com/gpu quota.",
        )
    # AIRM binds workloads to the LocalQueue named after the project (queue name == project ==
    # namespace). Select that queue by name; API list order is not a contract, so an arbitrary
    # local_queues[0] could follow an unrelated queue. If the project-named queue is absent, fall
    # back to a sole queue, but treat several unnamed candidates as ambiguous rather than healthy.
    selected = next(
        (q for q in local_queues if _dict(_dict(q).get("metadata")).get("name") == test_namespace),
        None,
    )
    if selected is None:
        if len(local_queues) == 1:
            selected = local_queues[0]
        else:
            names = ", ".join(sorted(_dict(_dict(q).get("metadata")).get("name", "?") for q in local_queues))
            return CheckResult(
                n, INCONCLUSIVE,
                f"namespace {test_namespace} has multiple LocalQueues ({names}) and none is named for the project — cannot tell which admits its GPU jobs",
                f"Name the project LocalQueue {test_namespace}, or point the workloads at a single queue.",
            )
    lq_name = _dict(_dict(selected).get("metadata")).get("name", "?")
    cluster_queue = str(_dict(_dict(selected).get("spec")).get("clusterQueue") or "")
    if not cluster_queue:
        return CheckResult(
            n, WARN,
            f"namespace {test_namespace} LocalQueue {lq_name} has no ClusterQueue binding — GPU jobs will not admit",
            "Set spec.clusterQueue on the LocalQueue to a ClusterQueue with amd.com/gpu quota.",
        )
    ok3, cq, err3 = kubectl_json(["get", "clusterqueues.kueue.x-k8s.io", cluster_queue])
    if not ok3:
        if _is_not_found(err3):
            return CheckResult(
                n, WARN,
                f"namespace {test_namespace} LocalQueue references missing ClusterQueue {cluster_queue} — GPU jobs will not admit",
                "Create the referenced ClusterQueue or update the LocalQueue binding.",
            )
        return CheckResult(
            n, INCONCLUSIVE,
            f"namespace {test_namespace} LocalQueue references ClusterQueue {cluster_queue}, but it cannot be read: {err3}",
            "Grant read access to ClusterQueues, then re-run.",
        )
    cq_spec = _dict(_dict(cq).get("spec"))
    gpu_quota = 0.0
    for rg in _list(cq_spec.get("resourceGroups")):
        for fl in _list(_dict(rg).get("flavors")):
            for res in _list(_dict(fl).get("resources")):
                if _dict(res).get("name") == "amd.com/gpu":
                    gpu_quota += _as_float(_dict(res).get("nominalQuota", 0))
    if gpu_quota <= 0:
        # Zero local quota is only a failure outside a cohort. AIRM project queues join the kaiwo
        # cohort with borrowing enabled, so a zero-quota project admits GPU jobs by borrowing —
        # check_airm_quota_health treats this same shape as healthy, and flagging it here would be a
        # spurious readiness failure contradicting that sibling check.
        cohort = str(cq_spec.get("cohort") or "")
        if not cohort:
            return CheckResult(
                n, WARN,
                f"{allocatable} amd.com/gpu allocatable, but namespace {test_namespace} is bound to ClusterQueue {cluster_queue} with no amd.com/gpu quota and no cohort to borrow from — GPU jobs will not admit",
                f"Add amd.com/gpu nominalQuota to ClusterQueue {cluster_queue}, or join it to a cohort with GPU quota.",
            )
        return CheckResult(
            n, OK,
            f"{allocatable} amd.com/gpu allocatable and namespace {test_namespace} is bound to ClusterQueue {cluster_queue}, which has no local amd.com/gpu quota but borrows from cohort {cohort}",
        )
    return CheckResult(
        n, OK,
        f"{allocatable} amd.com/gpu allocatable and namespace {test_namespace} is bound to ClusterQueue {cluster_queue} with {gpu_quota:g} GPU quota",
    )

def _as_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0

# ============================ RELEASE-SPEC SKEW ============================
# The E2E specs come from the branch under test, but the services come from whatever release is
# deployed. When the deployed product predates a spec's behaviour, that spec fails permanently on the
# environment. This check proves the relation instead of letting it surface as an assertion failure —
# but only when the caller supplies the spec revision, so it stays opt-in.
#
# For each product, the deployment named below carries the product-code image; its tag is the
# observable deployed marker. It is resolved to a commit in a supplied core checkout and compared by
# ancestry against the spec revision.
_SKEW_PRODUCTS: dict[str, tuple[str, str, str]] = {
    # product: (Deployment, namespace, product container). The chart container names are `aiwb` and
    # `airm`, not the Deployment names. Never fall back to the first container: that can be a sidecar.
    "aiwb": ("aiwb-api", "aiwb", "aiwb"),
    "airm": ("airm-api", "airm", "airm"),
}

# A deployed image tag ending in a git short/long SHA (e.g. main-fdb389c, <branch>-product-de73d50).
# Anchored to the end so it matches the build SHA, not an unrelated hex run earlier in the tag.
_TAG_TRAILING_SHA = re.compile(r"(?:^|[-_./])([0-9a-f]{7,40})$")

# Exact tag-name resolution is allowed only for established release-shaped tags. A bare branch image
# tag (`main`) is mutable and must not resolve to the current branch tip. The repository has used both
# SemVer without a v prefix and legacy calendar releases: vYYYY.MM-rc.N / vYYYY.MM.NNN.
_RELEASE_TAG = re.compile(
    r"^(?:"
    r"[0-9]+\.[0-9]+\.[0-9]+(?:[-.][0-9A-Za-z][0-9A-Za-z.-]*)?"
    r"|v[0-9]{4}\.[0-9]{2}(?:-rc\.[0-9]+|\.[0-9]{3})"
    r")$"
)

def _image_tag(image: str) -> str | None:
    """Return a tag only for an unpinned image reference.

    Any ref containing `@` is digest-pinned. Even `repo:tag@sha256:...` selects the digest, so its tag
    is not provenance for that manifest and cannot be used to infer a source commit.
    """
    if "@" in image:
        return None
    if ":" in image:
        candidate = image.rsplit(":", 1)[1]
        if "/" not in candidate:
            return candidate
    return None

def _pod_is_ready(pod: Any) -> bool:
    return (
        not _dict(_dict(pod).get("metadata")).get("deletionTimestamp")
        and _dict(_dict(pod).get("status")).get("phase") == "Running"
        and _condition_is_true(pod, "Ready")
    )

def _image_digest(image_id: str) -> str | None:
    """Return the sha256 manifest digest from a container status imageID, or None.

    imageID is the manifest the runtime actually pulled (e.g.
    'registry.example.com/product-api@sha256:abcd…' or 'docker-pullable://…@sha256:abcd…'); the
    requested .image tag can drift from it if a registry tag is retargeted. Only the digest identifies
    the running build."""
    match = _IMAGE_DIGEST.search(image_id)
    return f"sha256:{match.group(1)}" if match else None

# A canonical (lowercase, 64-hex) sha256 digest ending the imageID; anything else names no manifest.
_IMAGE_DIGEST = re.compile(r"@sha256:([0-9a-f]{64})$")

def _rollout_complete(deploy: Any) -> bool:
    """True when the Deployment controller has seen the latest spec and every desired replica runs
    the updated template, Ready and available — so no old pod is still serving."""
    meta, spec, status = _dict(_dict(deploy).get("metadata")), _dict(_dict(deploy).get("spec")), _dict(_dict(deploy).get("status"))
    desired = _as_int(spec.get("replicas", 1))
    return (
        desired > 0
        and _as_int(status.get("observedGeneration")) >= _as_int(meta.get("generation"))
        # status.replicas counts old pods too, so a maxSurge rollout still holding one exceeds it.
        and _as_int(status.get("replicas")) == desired
        and _as_int(status.get("updatedReplicas")) == desired
        and _as_int(status.get("readyReplicas")) == desired
        and _as_int(status.get("availableReplicas")) == desired
        and _as_int(status.get("unavailableReplicas")) == 0
    )

def _deployed_product_image(name: str, namespace: str, container_name: str) -> tuple[str | None, str]:
    """Return the image tag actually serving a completed Deployment rollout.

    A Deployment template is only desired state. During a rolling update old Ready pods can still
    serve traffic, so require rollout completion and inspect every selected Ready pod. Both the
    requested .image tag AND the running .imageID manifest digest must be identical across every Ready
    pod: a uniform digest rules out a partial retarget where some pods run an older manifest under the
    same tag text. Any incomplete, ambiguous, or mixed state is unconfirmed and must not produce an
    OK/WARN skew verdict.

    Provenance limit: tag text is then resolved to a source commit via the local checkout, which
    trusts the GitOps (Argo) tag->digest binding. This checker is offline and cannot bind the running
    digest to a commit through build attestations, so a registry tag uniformly retargeted to a
    different build of the same name is a known residual gap (closing it needs registry/attestation
    access).
    """
    ok, obj, err = kubectl_json(["get", "deploy", name, "-n", namespace])
    if not ok:
        return None, err
    spec = _dict(_dict(obj).get("spec"))
    desired = _as_int(spec.get("replicas", 1))
    if not _rollout_complete(obj):
        return None, f"{name} rollout is incomplete; deployed image is not stable"
    selector = _selector_arg(spec.get("selector"))
    if not selector:
        return None, f"{name} has no unambiguous matchLabels pod selector"
    ok_pods, pod_obj, pod_err = kubectl_json(["get", "pods", "-n", namespace, "-l", selector])
    if not ok_pods:
        return None, f"cannot read {name} pods: {pod_err}"
    pods = [pod for pod in _list(_dict(pod_obj).get("items")) if _pod_is_ready(pod)]
    if len(pods) != desired:
        return None, f"{name} has {len(pods)} Ready serving pod(s), expected {desired}; rollout state is ambiguous"
    images: set[str] = set()
    digests: set[str] = set()
    for pod in pods:
        statuses = [
            _dict(item) for item in _list(_dict(_dict(pod).get("status")).get("containerStatuses"))
            if _dict(item).get("name") == container_name
        ]
        pod_name = _dict(_dict(pod).get("metadata")).get("name", "?")
        if len(statuses) != 1 or not statuses[0].get("image"):
            return None, f"Ready pod {pod_name} does not expose exactly one {container_name!r} product container image"
        images.add(str(statuses[0]["image"]))
        digest = _image_digest(str(statuses[0].get("imageID") or ""))
        if digest is None:
            return None, f"Ready pod {pod_name} exposes no running image digest for {container_name!r} — cannot confirm the serving manifest"
        digests.add(digest)
    if len(images) != 1:
        return None, f"{name} Ready pods serve mixed product images: {', '.join(sorted(images))}"
    if len(digests) != 1:
        return None, f"{name} Ready pods serve one tag but different running manifests (retarget in progress): {', '.join(sorted(digests))}"
    return next(iter(images)), ""

def _git(repo: str, args: list[str], timeout: int = 20) -> tuple[bool, str, str]:
    rc, out, err = run(["git", "-C", repo, *args], timeout)
    return rc == 0, out, err

def _resolve_tag_to_commit(repo: str, tag: str) -> tuple[str | None, str]:
    """Resolve a deployed image tag to a commit in the core repo, using only IMMUTABLE markers so a
    later comparison is trustworthy: an annotated/lightweight release tag under refs/tags (e.g.
    2.0.2-rc.3), else a git short/long SHA embedded at the end of the tag (main-<sha>,
    <branch>-product-<sha>). A bare branch name is deliberately NOT accepted — the image workflows
    publish moving branch tags, and resolving `main` to the current branch tip instead of the build
    commit would silently compare the wrong revision. Returns (commit, detail) or (None, reason).
    """
    # refs/tags/<tag> only, and only for release-shaped names: a tag named `main` is itself mutable
    # in image-publishing practice, even though Git stores it under refs/tags.
    if _RELEASE_TAG.fullmatch(tag):
        ok, out, _ = _git(repo, ["rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}"])
        if ok and out:
            return out, f"release tag {tag}"
    m = _TAG_TRAILING_SHA.search(tag)
    if m:
        sha = m.group(1)
        # A short SHA is immutable content, but rev-parse would also happily resolve a branch/tag
        # whose *name* is all hex. Require the resolved object to be a commit reachable as a SHA, and
        # confirm the token really is an object id (not a ref name) by checking it as a commit-ish.
        ok2, out2, _ = _git(repo, ["rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"])
        if ok2 and out2 and out2.startswith(sha):
            return out2, f"commit {sha} (from image tag {tag})"
    return None, f"image tag {tag!r} does not resolve to an immutable release tag or commit in the core repo"

def _is_shallow(repo: str) -> bool | None:
    """Whether the checkout has truncated history. None means Git could not establish it, which is
    also inconclusive because a negative ancestry answer would then be unsafe to trust."""
    ok, out, _ = _git(repo, ["rev-parse", "--is-shallow-repository"])
    if not ok or out.strip() not in ("true", "false"):
        return None
    return out.strip() == "true"

def _is_ancestor(repo: str, ancestor: str, descendant: str, shallow: bool) -> bool | None:
    """True when `ancestor` is an ancestor of (or equal to) `descendant`. None means indeterminate —
    a git error (rc not 0/1), OR a negative answer in a shallow repo, where history beyond the
    boundary could still contain the ancestor. An indeterminate answer must never become a false
    OK/WARN, so callers route it to INCONCLUSIVE."""
    rc, _, _ = run(["git", "-C", repo, "merge-base", "--is-ancestor", ancestor, descendant], 20)
    if rc == 0:
        return True
    if rc == 1:
        # In a full clone rc=1 is definitive "not an ancestor"; in a shallow clone it may only mean
        # "not within the fetched history", which is not proof.
        return None if shallow else False
    return None

def check_release_spec_skew(spec_revision: str, core_repo: str) -> CheckResult:
    """Warn when a deployed product predates the specs about to run against it.

    Opt-in: only invoked when --spec-revision is given. Anti-silence: any unresolvable input, image,
    tag, or comparison is INCONCLUSIVE, never a false OK or WARN.
    """
    n = "release-spec-skew"
    ok_repo, _, err_repo = _git(core_repo, ["rev-parse", "--is-inside-work-tree"])
    if not ok_repo:
        return CheckResult(
            n, INCONCLUSIVE,
            f"--core-repo {core_repo!r} is not a readable git repository: {err_repo[:120]} — cannot compare deployed to spec",
            "Point --core-repo at a core checkout that contains the release history.",
        )
    ok_spec, spec_commit, _ = _git(core_repo, ["rev-parse", "--verify", "--quiet", f"{spec_revision}^{{commit}}"])
    if not (ok_spec and spec_commit):
        return CheckResult(
            n, INCONCLUSIVE,
            f"the spec revision {spec_revision!r} does not resolve in --core-repo — cannot establish what the tests require",
            "Pass a --spec-revision present in --core-repo (fetch it first if the checkout is shallow).",
        )

    shallow = _is_shallow(core_repo)
    if shallow is None:
        return CheckResult(
            n, INCONCLUSIVE,
            "cannot establish whether --core-repo has complete history — ancestry results would be unsafe to trust",
            "Use a readable full core checkout, then re-run.",
        )
    problems: list[str] = []
    unconfirmed: list[str] = []
    confirmed: list[str] = []
    for product, (deploy, ns, container_name) in sorted(_SKEW_PRODUCTS.items()):
        image, ierr = _deployed_product_image(deploy, ns, container_name)
        if image is None:
            if _is_not_found(ierr):
                unconfirmed.append(f"{product}: {deploy} not deployed in ns {ns}")
            else:
                unconfirmed.append(f"{product}: cannot read {deploy} image ({ierr[:80]})")
            continue
        tag = _image_tag(image)
        if not tag:
            unconfirmed.append(f"{product}: {deploy} runs a digest-pinned or untagged image with no trusted commit provenance")
            continue
        deployed_commit, detail = _resolve_tag_to_commit(core_repo, tag)
        if deployed_commit is None:
            unconfirmed.append(f"{product}: {detail}")
            continue
        spec_in_deployed = _is_ancestor(core_repo, spec_commit, deployed_commit, shallow)
        if spec_in_deployed is None:
            reason = "checkout is shallow" if shallow else "git error"
            unconfirmed.append(f"{product}: cannot compare deployed {tag} to the spec revision ({reason})")
            continue
        if spec_in_deployed:
            confirmed.append(f"{product} ({detail})")
            continue
        deployed_in_spec = _is_ancestor(core_repo, deployed_commit, spec_commit, shallow)
        if deployed_in_spec is None:
            reason = "checkout is shallow" if shallow else "git error"
            unconfirmed.append(f"{product}: cannot establish the relationship between deployed {tag} and the spec revision ({reason})")
        elif deployed_in_spec:
            problems.append(
                f"{product}: deployed {tag} ({deployed_commit[:9]}) predates the spec revision "
                f"{spec_revision} ({spec_commit[:9]}) — specs written for newer behaviour will fail"
            )
        else:
            problems.append(
                f"{product}: deployed {tag} ({deployed_commit[:9]}) has diverged from the spec revision "
                f"{spec_revision} ({spec_commit[:9]}) — it does not contain those specs' behaviour"
            )

    if problems:
        suffix = f"; also unconfirmed: {', '.join(unconfirmed)}" if unconfirmed else ""
        return CheckResult(
            n, WARN,
            f"a deployed product is out of sync with the specs about to run: {'; '.join(problems)}{suffix}",
            "Deploy a build that contains the spec revision (or run the specs from the deployed release's branch) before the suite.",
        )
    if unconfirmed:
        return CheckResult(
            n, INCONCLUSIVE,
            f"cannot confirm every deployed product contains the spec revision: {', '.join(unconfirmed)}",
            "Restore the failed read, or ensure --core-repo contains the deployed tags/commits, then re-run.",
        )
    return CheckResult(
        n, OK,
        f"every deployed product contains the spec revision {spec_revision}: {', '.join(confirmed)}",
    )
