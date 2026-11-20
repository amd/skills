#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Read-only environment check for a cluster that runs (or will run) the AI Workbench.

Detects known, *silent* platform problems and reports a plain-language verdict per check,
usable before an install (readiness) and after it (in-flight health). It never mutates the
cluster.

Anti-silence contract (the whole reason this exists): a check that CANNOT read the state it
needs returns INCONCLUSIVE, and the overall result is HEALTHY only when EVERY check actively
passed. An unread check is never counted as healthy, so a silent bug can never pass by omission.

Output: human OK / WARN / INCONCLUSIVE lines by default, or a machine-readable verdict with
--json. The JSON contract is stable and consumed by other skills:

    {
      "overall": "ok" | "problems" | "inconclusive" | "unreachable",
      "checks": [ {"name", "status": "ok"|"warn"|"inconclusive", "verdict", "recommendation"} ]   (an informational e2e check also carries "gating": false)
    }

Exit codes: 0 HEALTHY, 1 PROBLEMS FOUND (>=1 warn), 2 CANNOT REACH CLUSTER, 3 INCONCLUSIVE
(no problems but a check could not confirm its state). Only exit 0 is "all clear".

image-exists is also exposed as a subcommand so other lanes can reuse it without duplicating
the private-registry logic:  environment_check.py image-exists <ref>
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from typing import Any

from _envcheck.shared import (
    CheckResult,
    INCONCLUSIVE,
    OK,
    WARN,
    _RUN_CACHE,
    _as_int,
    _dict,
    _list,
    _one_line,
    _running_pods,
    _selector_arg,
    check_cilium_mtu,
    check_image,
    crd_state,
    kubectl_json,
    run,
)
from _envcheck.storage import (
    check_longhorn_components,
    check_longhorn_health,
    check_object_storage,
    check_object_store_fill,
    check_storage_provisioner,
    check_stuck_storage_pods,
)
from _envcheck.compute import (
    _image_digest,
    _pod_is_ready,
    _rollout_complete,
    check_aiwb_api,
    check_aiwb_ui,
    check_app_pods_stable,
    check_e2e_local_queues,
    check_fine_tuning_models,
    check_finetuning_hf_token,
    check_gpu_metrics,
    check_gpu_schedulable,
    check_optional_capability_prerequisites,
    check_release_spec_skew,
    check_team_member_sign_in,
)
from _envcheck.identity import (
    check_aimartifact_rbac,
    check_airm_quota_health,
    check_app_startup_errors,
    check_e2e_rbac,
    check_finished_run_leftovers,
    check_gateway_auth,
    check_oidc_issuer_consistency,
    check_ui_auth_endpoints,
)
from _envcheck.aim import (
    _PRODUCT_IMAGE,
    _resource_ref,
    check_aim_model_health,
    check_aim_profiles,
    check_aim_ready,
    check_aim_route_hostnames,
    check_argocd_apps,
    check_required_secrets,
)

# Re-exported so this module's namespace stays the skill's whole interface:
# other lanes import check functions and image_exists from here by name.
from _envcheck.shared import (  # noqa: F401
    _ANSI_ESCAPE,
    _APP_NAMESPACES,
    _ATTEMPT_SUMMARY_CHARS,
    _AllAddressesFailed,
    _CLOCK_SKEW_SECONDS,
    _CONFIGURED_TEST_NAMESPACE,
    _CONNECT_FAILURES,
    _E2E_PLATFORM_NAMESPACES,
    _EachAddressConnections,
    _FINAL_UNANSWERED,
    _FinalError,
    _FirstBytes,
    _IN_CLUSTER_PROBE_SCRIPT,
    _IN_CLUSTER_PROBE_NAMESPACE,
    _IN_CLUSTER_PROBE_POD,
    _IN_CLUSTER_PROBE_WORKLOAD,
    _IPV4_COMPATIBLE,
    _MINIO_NS,
    _NAT64,
    _NETWORK_BACKOFF_SECONDS,
    _NO_E2E_NAMESPACE,
    _NO_REDIRECT_OPENER,
    _NoRedirect,
    _PRIVATE_UNICAST,
    _PRODUCT_DEPLOYMENTS,
    _PROXY_UNDELIVERED,
    _PinnedHTTPConnection,
    _PinnedHTTPSConnection,
    _RAW_REASON_STATUS,
    _RAW_STATUS_CODE,
    _RESOLVER_SLOTS,
    _RUNNER_NAMESPACES,
    _SEAWEED_NS,
    _SERVICE_ABSENT,
    _Silence,
    _Sockets,
    _UI_AUTH_PATHS,
    _answered,
    _attempt_summary,
    _bounded_getaddrinfo,
    _cluster_service_target,
    _condition_is_true,
    _conditions,
    _connect_each_address,
    _deploy_env,
    _dial,
    _discover_test_namespace,
    _fetch_pinned,
    _fold_embedded_ipv4,
    _hf_get,
    _http_get,
    _http_status,
    _in_cluster_http_status,
    _in_e2e_footprint,
    _is_not_found,
    _is_silence,
    _load_setup_module,
    _noticing_responses,
    _once_per_run,
    _resolve_public,
    _retry_unanswered,
    _rfc3339_age_seconds,
    _secret_state,
    _service_proxy_get,
    _tighten,
    _unanswered,
    crd_gate,
    image_exists,
)
from _envcheck.storage import (  # noqa: F401
    LONGHORN_NS,
    _FILL_FAIL_PCT,
    _FILL_WARN_PCT,
    _LEASE_FRESH_SECONDS,
    _LONGHORN_COMPONENTS,
    _LONGHORN_RECHECK_SECONDS,
    _NODE_LOCAL_PROVISIONERS,
    _NON_CSI_PROVISIONER_PREFIXES,
    _SEAWEED_ROLES,
    _SEAWEED_S3_SERVICE,
    _SHARE_MANAGER_RESTARTS,
    _WEED_FREE_SLOTS_RE,
    _WEED_SIZE_LIMIT_RE,
    _WEED_VOLUME_RE,
    _crash_looping,
    _is_forbidden,
    _is_model_download,
    _longhorn_recheck_seconds,
    _openbao_store_result,
    _parse_weed_volume_list,
    _pod_ready,
    _restarts,
    _seaweed_data_mounts,
    _seaweed_pods,
    _volume_unhealthy,
    _weed_free_slots,
)
from _envcheck.compute import (  # noqa: F401
    _DIFFUSION_TAGS,
    _GPU_EXPORTER_SELECTOR,
    _IMAGE_DIGEST,
    _METRICS_BACKEND_NAMESPACES,
    _NOTREADY_GRACE_SECONDS,
    _OTHER_THAN_DIFFUSION_TAGS,
    _RELEASE_TAG,
    _RESTART_HISTORY_SECONDS,
    _SKEW_PRODUCTS,
    _TAG_TRAILING_SHA,
    _TRANSCRIPTION_TAGS,
    _WAITING_FATAL,
    _active_gpu_workload_present,
    _as_float,
    _check_deploy_healthy,
    _collector_scrape_failure,
    _deployed_product_image,
    _discover_metrics_backend,
    _e2e_hf_token,
    _exporter_scrape_hosts,
    _fine_tuning_models,
    _git,
    _gpu_capacity,
    _image_tag,
    _is_ancestor,
    _is_shallow,
    _owned_by_job,
    _pod_age_seconds,
    _pod_log_signature,
    _prom_instant_query,
    _prom_numeric,
    _recently_restarted,
    _resolve_tag_to_commit,
    _restarts_are_stale,
)
from _envcheck.identity import (  # noqa: F401
    RunLookupError,
    _AIM_API_GROUP,
    _AIRM_API_ISSUER_VARS,
    _AIRM_DELETION_GRACE_SECONDS,
    _AIRM_FINALIZER_PREFIX,
    _AI_GATEWAY_ROUTE_CRD,
    _API_ISSUER_VARS,
    _AUTH_LOG_SIGNATURES,
    _DISCOVERY_SUFFIX,
    _E2E_RBAC_REQUIRED,
    _E2E_RBAC_ROLE,
    _ENVOY_SECURITY_POLICY,
    _GATEWAY_INVALID_KEY,
    _GATEWAY_MODELS_PATH,
    _GATEWAY_ROUTE_KINDS,
    _ISSUER_REJECTION,
    _JWKS_SUFFIX,
    _LOGGED_ISSUER_MISMATCH,
    _LOG_TAIL_LINES,
    _PRODUCT_CONTAINER,
    _PROJECT_LABEL,
    _REPEATED_ISSUER_REJECTION_MIN,
    _REPEATED_UNHANDLED_MIN,
    _RUN_NAMESPACE,
    _UI_ISSUER_VARS,
    _UNHANDLED_LOG,
    _deploy_container_env,
    _effective_issuer,
    _ext_authz_gateways,
    _gateway_auth_verdict,
    _github_run_status,
    _in_cluster_certificate_warning,
    _is_certificate_failure,
    _is_loopback_url,
    _issuer_at,
    _nextauth_answer_problem,
    _no_inference_routes_published,
    _recent_logs,
    _rule_covers,
    _swagger_discovery,
    _ui_proxy_get,
)
from _envcheck.aim import (  # noqa: F401
    _AIM_MODEL_CRDS,
    _AIM_PROFILE_AGGREGATE_KEYS,
    _AIM_STATUSES,
    _AIM_UNHEALTHY_STATUSES,
    _KINDS_WITHOUT_HEALTH,
    _PINNED_REVISION,
    _UNRESOLVED_REVISION,
    _aim_status,
    _allocatable_gpus,
    _externalsecret_settled,
    _gpu_device_ids,
    _is_base_only_cluster_model,
    _model_selected,
    _only_waiting_for_first_consumer,
    _unresolved_revision,
)


def check_deployed_versions() -> CheckResult:
    """Record which AIRM/AIWB/aim-engine builds are serving, so the output of a pre-E2E run names
    exactly what was tested. Report-only: it never warns, though an unconfirmed record (a rollout in
    progress) is INCONCLUSIVE and so makes the whole run exit 3.

    A Deployment template is only the desired image; mid-rollout, Ready pods can still serve the old
    one. The record therefore comes from the Ready pods' running image and digest of a completed
    rollout, and a Deployment still rolling out or whose Ready pods show no single build (none Ready,
    or a mix) leaves the record unconfirmed."""
    n = "deployed-versions"
    ok, obj, err = kubectl_json(["get", "deploy", "-A"])
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot list deployments to record deployed versions: {err}")
    products: list[tuple[str, str, dict[str, Any], str, str, bool, int]] = []  # (ref, ns, selector, container, desired image, rolled out, replicas)
    for deploy in _list(_dict(obj).get("items")):
        spec = _dict(_dict(deploy).get("spec"))
        # Only a plain matchLabels selector can be matched here; with matchExpressions it stays empty,
        # which leaves the record unconfirmed rather than matching unrelated pods by labels alone.
        selector = _dict(_dict(spec.get("selector")).get("matchLabels")) if _selector_arg(spec.get("selector")) else {}
        ns = str(_dict(_dict(deploy).get("metadata")).get("namespace") or "")
        for c in _list(_dict(_dict(spec.get("template")).get("spec")).get("containers")):
            image = str(_dict(c).get("image") or "")
            repo = image.split("@", 1)[0].rsplit("/", 1)[-1].split(":", 1)[0]
            if _PRODUCT_IMAGE.search(repo):
                products.append((_resource_ref(deploy), ns, selector, str(_dict(c).get("name") or ""), image,
                                 _rollout_complete(deploy), _as_int(spec.get("replicas", 1))))
    if not products:
        return CheckResult(n, INCONCLUSIVE, "no AIRM/AIWB/aim-engine deployment found — no deployed version to record (expected before install)")
    ok, pods_obj, err = _running_pods()
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot list running pods to record the serving builds: {err}")
    pods = [p for p in _list(_dict(pods_obj).get("items")) if _pod_is_ready(p)]
    entries: list[str] = []
    unconfirmed: list[str] = []
    for ref, ns, selector, container, desired, rolled_out, replicas in products:
        running: set[str] = set()
        missing_digest: list[str] = []
        serving_pods = 0
        label = f"{ref}[{container}]"
        # Mid-rollout the Ready pods can all still be old ones, agreeing on a build about to be replaced.
        if not rolled_out:
            unconfirmed.append(f"{label}: rollout incomplete (desired {desired})")
            continue
        if not selector:
            unconfirmed.append(f"{label}: its pod selector is not plain matchLabels, so its pods cannot be told apart")
            continue
        for pod in pods:
            meta = _dict(_dict(pod).get("metadata"))
            labels = _dict(meta.get("labels"))
            if meta.get("namespace") != ns or not selector or any(labels.get(k) != v for k, v in selector.items()):
                continue
            if meta.get("deletionTimestamp"):
                continue  # terminating: leaving, not serving the rollout's build
            serving_pods += 1
            statuses = [cs for cs in _list(_dict(_dict(pod).get("status")).get("containerStatuses")) if _dict(cs).get("name") == container]
            if len(statuses) != 1:
                # A pod that does not show exactly one product container proves nothing about the build.
                missing_digest.append(f"{label}: Ready pod {meta.get('name', '?')} shows {len(statuses)} {container!r} container(s)")
            for cs in statuses:
                image = str(_dict(cs).get("image") or "")
                digest = _image_digest(str(_dict(cs).get("imageID") or ""))
                # Without both, the pod does not prove which manifest it runs.
                if not image or not digest:
                    missing_digest.append(f"{label}: Ready pod {meta.get('name', '?')} exposes no running image digest")
                else:
                    running.add(f"{image}@{digest}")
        if missing_digest:
            unconfirmed.extend(missing_digest)
        elif len(running) == 1 and serving_pods != replicas:
            # The Deployment status lags the pods; a count that disagrees means a replica is missing
            # or extra, so one uniform build among the pods seen does not prove what serves.
            unconfirmed.append(f"{label}: {serving_pods} Ready pod(s) serve it, expected {replicas}")
        elif len(running) == 1:
            # The full digest identifies the build; only the printed record is shortened.
            image, digest = next(iter(running)).rsplit("@", 1)
            entries.append(f"{label}={image} ({digest[:19]})")
        elif not running:
            unconfirmed.append(f"{label}: no Ready pod serves it (desired {desired})")
        else:
            unconfirmed.append(f"{label}: Ready pods serve {len(running)} builds: {', '.join(sorted(running))}")
    record = f"serving: {'; '.join(sorted(entries))}" if entries else "no build confirmed serving"
    if unconfirmed:
        return CheckResult(n, INCONCLUSIVE, f"{record}; unconfirmed: {'; '.join(unconfirmed)}",
                           "Re-run once the named Deployments have finished rolling out, so the record names one serving build each.")
    return CheckResult(n, OK, record)

# ============================ RUNNER ============================
def aggregate_overall(results: list[CheckResult]) -> str:
    statuses = {r.status for r in results if r.gating}
    if WARN in statuses:
        return "problems"
    if INCONCLUSIVE in statuses:
        return "inconclusive"
    return "ok"

_OVERALL_EXIT = {"ok": 0, "problems": 1, "unreachable": 2, "inconclusive": 3}

_HUMAN_LABEL = {OK: "OK          ", WARN: "WARN        ", INCONCLUSIVE: "INCONCLUSIVE"}

# Shown in e2e mode but never gating: they describe node-level or cluster-hygiene state that does
# not decide whether the suites can run. Every other check gates in both modes.
_E2E_INFORMATIONAL = frozenset({"cilium-mtu"})

def _event_time(event: Any) -> datetime.datetime:
    """Latest moment an event was observed: a recurring series updates series.lastObservedTime, not eventTime."""
    e = _dict(event)
    stamps = [_dict(e.get("series")).get("lastObservedTime"), e.get("eventTime"), e.get("lastTimestamp")]
    parsed: list[datetime.datetime] = []
    for ts in stamps:
        try:
            when = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            continue
        parsed.append(when if when.tzinfo else when.replace(tzinfo=datetime.timezone.utc))
    return max(parsed, default=datetime.datetime.min.replace(tzinfo=datetime.timezone.utc))

def check_kaiwo_queue_config() -> CheckResult:
    """A KaiwoQueueConfig whose status is FAILED means its controller cannot reconcile quota.

    A stale entry (e.g. a ClusterQueue for a deleted namespace) leaves the controller erroring
    every reconcile, so every quota operation in the next run fails. The newest event on the
    object is quoted when readable, since status.status alone does not say why.
    """
    n = "kaiwo-queue-config"
    crd = "kaiwoqueueconfigs.kaiwo.silogen.ai"
    state = crd_state(crd)
    if state == "absent":
        return CheckResult(n, OK, "Kaiwo is not installed on this cluster — no KaiwoQueueConfig to check")
    if state == "unreadable":
        return CheckResult(n, INCONCLUSIVE, f"cannot read the {crd} CRD (RBAC/API error) — queue config state unconfirmed")
    ok, obj, err = kubectl_json(["get", crd, "-A"])
    if not ok:
        return CheckResult(n, INCONCLUSIVE, f"cannot list KaiwoQueueConfigs: {_one_line(err)}")
    configs = _list(_dict(obj).get("items"))
    failed = [c for c in configs if _dict(_dict(c).get("status")).get("status") == "FAILED"]
    # An entry naming a deleted namespace is what makes the controller fail, so it is flagged
    # even before the status flips.
    wanted = {
        ns: _resource_ref(c)
        for c in configs
        for q in _list(_dict(_dict(c).get("spec")).get("clusterQueues"))
        for ns in _list(_dict(q).get("namespaces"))
        if isinstance(ns, str)
    }
    missing: dict[str, str] = {}
    unproven: list[str] = []
    if wanted:
        ok_n, nobj, err_n = kubectl_json(["get", "namespaces"])
        if not ok_n:
            unproven.append(f"cannot list namespaces to verify queue entries: {_one_line(err_n)}")
        else:
            present = {str(_dict(_dict(x).get("metadata")).get("name")) for x in _list(_dict(nobj).get("items"))}
            missing = {ns: ref for ns, ref in wanted.items() if ns not in present}
    if not failed and not missing:
        if unproven:
            return CheckResult(n, INCONCLUSIVE, f"{len(configs)} KaiwoQueueConfig(s), none FAILED, but {'; '.join(unproven)}")
        return CheckResult(n, OK, f"{len(configs)} KaiwoQueueConfig(s), none FAILED")
    details: list[str] = []
    if failed:
        ok_e, eobj, err_e = kubectl_json(["get", "events", "-A", "--field-selector", "involvedObject.kind=KaiwoQueueConfig"])
        events = _list(_dict(eobj).get("items")) if ok_e else []
        for cfg in failed:
            ref = _resource_ref(cfg)
            meta = _dict(_dict(cfg).get("metadata"))
            # Events outlive their object, so a replacement with the same name is told apart by uid.
            mine = [
                e for e in events
                if _dict(_dict(e).get("involvedObject")).get("name") == meta.get("name")
                and _dict(_dict(e).get("involvedObject")).get("uid") == meta.get("uid")
            ]
            mine.sort(key=_event_time)
            if mine:
                latest = _dict(mine[-1])
                details.append(f"{ref}=FAILED({latest.get('reason', '?')}: {_one_line(str(latest.get('message', '')))})")
            else:
                why = "" if ok_e else f" (events unreadable: {_one_line(err_e)})"
                details.append(f"{ref}=FAILED{why}")
    details.extend(f"{ref} lists missing namespace {ns}" for ns, ref in sorted(missing.items()))
    return CheckResult(
        n, WARN,
        f"KaiwoQueueConfig not healthy: {', '.join(details)}",
        "The Kaiwo controller cannot reconcile queues, so quota operations fail in every suite. Remove "
        "the stale entry named above from the KaiwoQueueConfig, then re-run.",
    )

def run_all_checks(images: list[str], require_gpu: bool, test_namespace: str | None, restart_threshold: int,
                   metrics_backend: str | None = None, gateway_url: str | None = None,
                   gateway_key: str | None = None, e2e_oidc_user: str | None = None,
                   spec_revision: str | None = None, core_repo: str = ".",
                   recent_minutes: int = 30, cluster_domain: str = "cluster.local", mode: str = "general",
                   s3_bucket: str = "default-bucket", models: list[str] | None = None,
                   require_hf_token: bool = False, e2e_prerequisites: bool = False,
                   trusted_private_hosts: frozenset[str] = frozenset()) -> list[CheckResult]:
    _RUN_CACHE.clear()
    # An undiscovered test namespace still scopes e2e mode: only e2e-* and platform namespaces match.
    scope = (test_namespace or "") if mode == "e2e" else None
    results = [
        check_deployed_versions(),
        check_argocd_apps(),
        check_cilium_mtu(),
        check_longhorn_health(scope),
        check_longhorn_components(scope),
        check_storage_provisioner(),
        check_stuck_storage_pods(scope),
        check_gpu_metrics(metrics_backend, require_gpu),
        check_aim_profiles(models if mode == "e2e" else None, e2e=mode == "e2e"),
        check_aim_model_health(scope, models),
        check_aim_ready(models if mode == "e2e" else None),
        check_app_pods_stable(restart_threshold, recent_minutes),
        # Probing the UIs' auth endpoints makes a UI with a wrong issuer log its mismatch, so the
        # probe runs before the checks that read those logs.
        check_ui_auth_endpoints(),
        check_app_startup_errors(recent_minutes),
        check_required_secrets(),
        check_object_storage(),
        check_object_store_fill(s3_bucket),
        check_aimartifact_rbac(test_namespace),
        check_e2e_rbac(e2e_oidc_user),
        check_airm_quota_health(),
        check_kaiwo_queue_config(),
        check_finished_run_leftovers(),
        check_gateway_auth(gateway_url, gateway_key),
        check_aim_route_hostnames(),
        check_aiwb_api(),
        check_aiwb_ui(),
        check_oidc_issuer_consistency(recent_minutes, cluster_domain, trusted_private_hosts),
        check_gpu_schedulable(require_gpu, test_namespace),
    ]
    # Opt-in: the deployed-vs-spec skew check runs only when the caller states which spec revision
    # will run, so an ordinary readiness run is unaffected.
    if require_hf_token:
        results.append(check_finetuning_hf_token())
    if e2e_prerequisites:
        results.extend([
            check_e2e_local_queues(test_namespace),
            check_fine_tuning_models(),
            check_optional_capability_prerequisites(),
            check_team_member_sign_in(core_repo),
        ])
    if spec_revision:
        results.append(check_release_spec_skew(spec_revision, core_repo))
    results.extend(check_image(ref) for ref in images)
    if mode == "e2e":
        for r in results:
            if r.name in _E2E_INFORMATIONAL:
                r.gating = False
    return results

def _check_json(r: CheckResult) -> dict[str, Any]:
    """The stable per-check contract; `gating: false` is added only for an informational e2e check."""
    out: dict[str, Any] = {"name": r.name, "status": r.status, "verdict": r.verdict, "recommendation": r.recommendation}
    if not r.gating:
        out["gating"] = False
    return out

def emit(results: list[CheckResult], overall: str, as_json: bool) -> None:
    if as_json:
        print(json.dumps(
            {"overall": overall, "checks": [_check_json(r) for r in results]},
            indent=2,
        ))
        return
    print("== Cluster environment check ==")
    for r in results:
        note = "" if r.gating or r.status == OK else " [informational in e2e mode, not gating]"
        print(f"{_HUMAN_LABEL[r.status]} {r.name}: {r.verdict}{note}")
        if r.recommendation and r.status != OK:
            print(f"             -> {r.recommendation}")
    warns = sum(1 for r in results if r.status == WARN and r.gating)
    inconc = sum(1 for r in results if r.status == INCONCLUSIVE and r.gating)
    print(f"\nSUMMARY: warnings={warns} inconclusive={inconc}")
    print(f"RESULT: {overall.upper()}")

def _positive_int(text: str) -> int:
    """argparse type for a count or window: zero or less would silently disable the check it tunes."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value

def main(argv: list[str] | None = None) -> int:
    # --json lives on a shared parent parser so it is accepted both before the subcommand and after
    # `image-exists REF` — a single subparser copy would otherwise shadow the parent value by order.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="Emit a machine-readable verdict.")
    ap = argparse.ArgumentParser(description="Read-only AI Workbench cluster environment check.", parents=[common])
    sub = ap.add_subparsers(dest="subcommand")
    img = sub.add_parser("image-exists", parents=[common], help="Check one container image reference and exit (reusable by other lanes).")
    img.add_argument("ref")
    skew = sub.add_parser("release-spec-skew", parents=[common], help="Check whether deployed AIRM/AIWB builds contain one core spec revision.")
    skew.add_argument("spec_revision", metavar="REV")
    # No default here: --core-repo shares its dest with the top-level flag, so a concrete default would
    # overwrite a value the user passed BEFORE the subcommand. SUPPRESS makes the subparser set
    # core_repo only when --core-repo is given after the subcommand; otherwise the top-level value (or
    # its "." default) stands.
    skew.add_argument("--core-repo", default=argparse.SUPPRESS, metavar="PATH",
                      help="Core git checkout containing REV and deployed release tags/commits (defaults to the top-level --core-repo, i.e. the current directory).")
    ap.add_argument("--image", action="append", default=[], metavar="REF",
                    help="Also check that this image reference resolves (repeatable).")
    ap.add_argument("--mode", choices=("general", "e2e"), default="general",
                    help="general (default): cluster-wide health, every problem reported. e2e: can the E2E suites run "
                         "here? longhorn-health, stuck-storage-pods and aim-model-health are scoped to the test namespace, e2e-* and the platform namespaces (aiwb, airm, keycloak, seaweedfs-instance, minio-tenant-default); problems elsewhere are informational")
    ap.add_argument("--model", action="append", default=[], metavar="NAME",
                    help="e2e mode: a model the run uses (exact model resource name, repeatable). aim-profiles gates only on those models and "
                         "aim-model-health narrows cluster-scoped models to them (namespaced models are scoped by namespace); aim-ready then requires each of them to be Ready; without it every model gates")
    ap.add_argument("--s3-bucket", default="default-bucket", metavar="BUCKET",
                    help="object-store bucket (SeaweedFS collection) whose writable volumes object-store-fill checks")
    ap.add_argument("--require-gpu", action="store_true",
                    help="Treat 0 allocatable GPUs as a problem even on a cluster with no GPU nodes (for GPU E2E lanes).")
    ap.add_argument("--test-namespace", default=None, metavar="NS",
                    help="Namespace the E2E tests run in, checked for AIMArtifact create permission and a GPU-quota LocalQueue (default: an existing e2e-* namespace, preferring one with a LocalQueue; skipped when none exists).")
    ap.add_argument("--restart-threshold", type=_positive_int, default=3, metavar="N",
                    help="app-pods-stable warns when a Ready app pod has restarted at least N times and last died within "
                         "the past 6 hours, or its last termination time is unreadable (default: 3).")
    ap.add_argument("--cluster-domain", default="cluster.local", metavar="DOMAIN",
                    help="The cluster's DNS domain, so <svc>.<ns>.svc.DOMAIN URLs are read through the service proxy (default: cluster.local).")
    ap.add_argument("--trusted-private-host", action="append", default=[], metavar="HOST",
                    help="Issuer host (e.g. the Keycloak host) the operator vouches for; its discovery document may then be read from a private-range address (RFC 1918 / ULA). Repeatable. Without it, any host resolving to a non-public address is refused.")
    ap.add_argument("--recent-minutes", type=_positive_int, default=30, metavar="N",
                    help="Window for 'recent': app-pods-stable warns on a restart, and app-startup-errors and oidc-issuer-consistency scan logs, within the last N minutes (default: 30).")
    ap.add_argument("--metrics-backend", default=None, metavar="NS/SVC:PORT",
                    help="Prometheus/Mimir query service for gpu-metrics to positively confirm GPU series arrive (e.g. otel-lgtm-stack/lgtm-stack:9090). Auto-discovered if omitted.")
    ap.add_argument("--gateway-url", default=None, metavar="URL",
                    help="Inference gateway base URL for gateway-auth. Auto-discovered from the aiwb-ui deployment if omitted.")
    ap.add_argument("--gateway-key", default=None, metavar="KEY",
                    help="A valid inference API key. When given, gateway-auth also confirms an authorised call succeeds (the invalid-key probe needs no key).")
    ap.add_argument("--e2e-oidc-user", default=None, metavar="USER",
                    help="The OIDC username the E2E runner authenticates as. When given, eai-e2e-rbac requires the role binding to name exactly this user (else any OIDC user is accepted).")
    ap.add_argument("--require-hf-token", action="store_true",
                    help="Also check the Hugging Face token the run uses (HF_TOKEN, else the e2e-hf-token runner secret): present, accepted by Hugging Face, and able to read every gated fine-tuning model.")
    ap.add_argument("--e2e-prerequisites", action="store_true",
                    help="Also check what the E2E suites need that is easy to miss: a Kueue LocalQueue in --test-namespace, a non-empty /v1/fine-tuning/models (naming the hfTokenRequired entries), a LoRA recipe and a speech-to-text AIM (else those tests are reported not runnable), and a team-member account that can sign in (needs --core-repo).")
    ap.add_argument("--spec-revision", default=None, metavar="REV",
                    help="The core commit/ref whose specs will run. When given, release-spec-skew warns if a deployed product predates it. Requires --core-repo to contain that revision and the deployed tags.")
    ap.add_argument("--core-repo", default=".", metavar="PATH",
                    help="Path to a core git checkout used to resolve deployed image tags and --spec-revision to commits for the release-spec-skew check (default: current directory).")
    args = ap.parse_args(argv)

    if args.subcommand == "image-exists":
        result = check_image(args.ref)
        overall = {OK: "ok", WARN: "problems", INCONCLUSIVE: "inconclusive"}[result.status]
        emit([result], overall, getattr(args, "json", False))
        return _OVERALL_EXIT[overall]

    if args.subcommand == "release-spec-skew":
        result = check_release_spec_skew(args.spec_revision, args.core_repo)
        overall = {OK: "ok", WARN: "problems", INCONCLUSIVE: "inconclusive"}[result.status]
        emit([result], overall, getattr(args, "json", False))
        return _OVERALL_EXIT[overall]

    # Connectivity gate — no cluster access is reported clearly and NEVER as healthy.
    if run(["kubectl", "cluster-info"], 20)[0] != 0:
        conn = CheckResult(
            "connectivity", INCONCLUSIVE,
            "cannot reach the cluster — kubectl cannot talk to the API server",
            "Fix kubeconfig / context / network access, then re-run. No checks could run, so health is UNKNOWN.",
        )
        emit([conn], "unreachable", getattr(args, "json", False))
        return _OVERALL_EXIT["unreachable"]

    results = run_all_checks(args.image, args.require_gpu, args.test_namespace, args.restart_threshold,
                             args.metrics_backend, args.gateway_url, args.gateway_key, args.e2e_oidc_user,
                             args.spec_revision, args.core_repo, args.recent_minutes, args.cluster_domain,
                             args.mode, args.s3_bucket, args.model,
                             args.require_hf_token, args.e2e_prerequisites,
                             frozenset(h.lower() for h in args.trusted_private_host))
    overall = aggregate_overall(results)
    emit(results, overall, getattr(args, "json", False))
    return _OVERALL_EXIT[overall]

if __name__ == "__main__":
    sys.exit(main())
