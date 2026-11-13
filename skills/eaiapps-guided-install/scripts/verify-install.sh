#!/usr/bin/env bash

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
# Post-install health probe for an AIWB install.
# Confirms the workbench is actually up — primary signal is the aiwb-api Deployment ready.
# Also checks aiwb-ui, CNPG PostgreSQL, RabbitMQ (combined), and HTTPRoutes.
#
# Usage:
#   verify-install.sh [--standalone] [--external-db] [--namespace <ns>]
#                     [--airm-namespace <ns>] [--release <name>] [--airm-release <name>]
#
# --standalone        AIWB-only install (skips AIRM/RabbitMQ checks). Default is combined.
# --external-db       Bring-your-own PostgreSQL: skip the CNPG cluster checks. Use when the
#                     CNPG operator and *-cnpg infra charts were intentionally not installed
#                     and the DB host/secrets point at an external instance (INSTALL.md
#                     "Bringing Your Own Infrastructure"). Without this flag a healthy BYO-DB
#                     install would falsely report INCONCLUSIVE for a missing CNPG cluster.
# --namespace         AIWB Helm release namespace (default: aiwb). This is the namespace
#                     you `helm install -n <ns>` into — the same in standalone and combined.
#                     NOT the standalone defaultNamespace/workbench value, which is a
#                     separate user-workload namespace; the aiwb-api/-ui Deployments and
#                     HTTPRoutes always render into .Release.Namespace.
# --airm-namespace    AIRM namespace (default: airm; combined mode only).
# --release           Helm release name for AIWB (default: aiwb) → Deployments {release}-api/-ui.
# --airm-release      Helm release name for AIRM (default: airm) → Deployments {airm_release}-api/-ui.
# --workloads-authz-port
#                     Port the workloads ext_authz handler serves on and the workloads-extauth
#                     SecurityPolicy routes to (default: 8084). The ai-gateway-discovery chart
#                     exposes this as controller.workloadsAuthzPort; pass the overridden value
#                     here or a healthy policy on a custom port is falsely reported failed.
# --request-timeout   Per-request kubectl timeout (default: 30s). Bounds every cluster call so a
#                     reachable-but-degraded API server surfaces as INCONCLUSIVE instead of
#                     hanging forever — a wedged verification is exactly the silent partial
#                     install this probe exists to prevent. Raise it on a large or distant
#                     cluster.
#
# Exit codes: 0 = healthy; 1 = inconclusive (a component not ready — what to fix);
#             2 = cluster unreachable; 64 = usage error (unknown option or missing option value).
set -uo pipefail

standalone=0
external_db=0
aiwb_ns="aiwb"
airm_ns="airm"
release="aiwb"
airm_release="airm"
workloads_authz_port="8084"
request_timeout="30s"
# Validate an option's value is present and is not itself an option (a bare ${2:?} would fire
# bash's exit 1, colliding with the documented result codes; a "-"-prefixed value means the
# value was omitted and the next flag got swallowed). Usage errors exit 64.
require_value() { # flag  remaining-argc  value
  [ "$2" -ge 2 ] || { echo "$1 requires a value" >&2; exit 64; }
  case "$3" in -*) echo "$1 requires a value (got option '$3')" >&2; exit 64 ;; esac
}
# A Go duration is one or more <quantity><unit> pairs ("30s", "1h30m"). A glob cannot require a
# digit before each unit, so match a real regex: a character-class glob would accept "sm" or "h"
# and let kubectl's own parse error surface as a bogus INCONCLUSIVE verdict. Zero is rejected
# separately because kubectl reads 0 as "no timeout", silently restoring the unbounded call this
# option exists to prevent.
require_duration() { # flag  value
  # Keep this a here-string; do NOT rewrite it as `printf ... | grep -q`. `grep -q` stops at
  # its first match without draining its input, so the writer can still be holding an
  # unwritten tail when the reader exits; it then takes SIGPIPE/EPIPE, and under `pipefail`
  # that becomes the *pipeline's* status, reporting a successful match as a failure. A
  # here-string has no writer process to break. It appends exactly one newline where
  # `printf '%s'` appended none, so a value already ending in a newline gains an empty final
  # line. That cannot change this verdict: the pattern below is anchored, so it matches no
  # empty line, and `grep -q` succeeds if any line matches.
  grep -Eq '^([0-9]+(\.[0-9]+)?(ms|s|m|h))+$' <<<"$2" \
    || { echo "$1 requires a duration like 30s (got '$2')" >&2; exit 64; }
  # "Contains a nonzero digit" needs no regex, so test it in the shell — no reader, no pipe
  # and no subprocess at all. The same bracket expression, applied to the whole string
  # rather than per line, which is equivalent for a single-character class.
  case "$2" in
    *[1-9]*) ;;
    *) echo "$1 must be greater than zero (got '$2'); 0 disables the timeout" >&2; exit 64 ;;
  esac
}
while [ $# -gt 0 ]; do
  case "$1" in
    --standalone) standalone=1; shift ;;
    --external-db) external_db=1; shift ;;
    --namespace) require_value "$1" "$#" "${2-}"; aiwb_ns="$2"; shift 2 ;;
    --airm-namespace) require_value "$1" "$#" "${2-}"; airm_ns="$2"; shift 2 ;;
    --release) require_value "$1" "$#" "${2-}"; release="$2"; shift 2 ;;
    --airm-release) require_value "$1" "$#" "${2-}"; airm_release="$2"; shift 2 ;;
    --workloads-authz-port)
      require_value "$1" "$#" "${2-}"
      case "$2" in ''|*[!0-9]*) echo "$1 requires a numeric port (got '$2')" >&2; exit 64 ;; esac
      workloads_authz_port="$2"; shift 2 ;;
    --request-timeout)
      require_value "$1" "$#" "${2-}"
      require_duration "$1" "$2"
      request_timeout="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

# Every cluster read goes through this wrapper so no call can hang unbounded against a
# reachable-but-degraded API server: a wedged probe is indistinguishable from "still checking"
# and leaves the unresolved partial install this script exists to surface.
kc() { kubectl --request-timeout="$request_timeout" "$@"; }

problems=0
ok()   { printf 'OK    %s\n' "$1"; }
bad()  { printf 'FAIL  %s\n' "$1"; problems=$((problems+1)); }

echo "== AIWB install verification (namespace=$aiwb_ns, mode=$([ $standalone -eq 1 ] && echo standalone || echo combined)) =="

# kubectl absence is a tooling error, not a cluster problem — check it before the
# reachability probe so a missing binary isn't misreported as "cluster unreachable".
if ! command -v kubectl >/dev/null 2>&1; then
  echo "FAIL  kubectl not found on PATH — install kubectl to verify the install."
  echo "RESULT: INCONCLUSIVE / WHAT TO FIX — cannot verify without kubectl."
  exit 1
fi
if ! kc cluster-info >/dev/null 2>&1; then
  echo "BLOCK cluster unreachable — cannot verify install state."
  exit 2
fi

# --- deployment readiness helper: desired replicas all Ready ---
deploy_ready() { # ns deploy
  local ns="$1" d="$2" desired ready err
  # kubectl returns the same non-zero exit for NotFound, Forbidden (RBAC), and transient API
  # errors, so don't assert "not found" — capture stderr and surface the real cause as a query
  # failure. Only an explicit NotFound is reported as the chart likely not being installed.
  if ! err=$(kc get deploy "$d" -n "$ns" 2>&1); then
    case "$err" in
      *[Nn]otFound*|*'not found'*)
        bad "Deployment $d not found in namespace $ns (chart not installed there?)" ;;
      *)
        bad "cannot query Deployment $d in $ns — ${err:-kubectl error} (not necessarily absent)" ;;
    esac
    return 1
  fi
  # Distinguish a failed query (non-zero exit: RBAC/transient error) from an absent field
  # (exit 0, empty output). The deploy exists (checked above), so a query error here is a
  # real fault and must not silently fall back to defaults and read as "not ready".
  if ! desired=$(kc get deploy "$d" -n "$ns" -o jsonpath='{.spec.replicas}' 2>/dev/null) \
     || ! ready=$(kc get deploy "$d" -n "$ns" -o jsonpath='{.status.readyReplicas}' 2>/dev/null); then
    bad "failed to query $d replica status in $ns — cannot confirm readiness"
    return 1
  fi
  # An absent .spec.replicas means 1 in Kubernetes, not 0 — defaulting it to 0 would
  # misreport a Deployment that omits the field as never-ready.
  desired=${desired:-1}; ready=${ready:-0}
  if [ "$ready" -ge "$desired" ] && [ "$desired" -gt 0 ]; then
    ok "$d ready ($ready/$desired)"
    return 0
  fi
  bad "$d not ready ($ready/$desired) — see references/troubleshooting.md"
  return 1
}

# --- primary signal: aiwb-api ---
deploy_ready "$aiwb_ns" "${release}-api"
# --- UI ---
deploy_ready "$aiwb_ns" "${release}-ui"

# --- CNPG PostgreSQL (aiwb, and airm for combined) ---
# An empty item list must NEVER read as healthy: with the CRD installed but no Cluster
# created, the phase scan below has nothing to reject, so count items first.
cnpg_healthy() { # ns
  local ns="$1" count names unhealthy
  if ! kc get clusters.postgresql.cnpg.io -n "$ns" >/dev/null 2>&1; then
    bad "cannot list CNPG clusters in $ns — CRD missing, RBAC denial, or API error (not necessarily absent); API cannot serve without its DB"
    return 1
  fi
  # Capture the names query and check its exit status before counting: piping straight into
  # grep -c masks a failed query as count 0, misreporting a query error as "chart not installed".
  if ! names=$(kc get clusters.postgresql.cnpg.io -n "$ns" \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null); then
    bad "failed to query CNPG cluster list in $ns"
    return 1
  fi
  count=$(printf '%s\n' "$names" | grep -c . || true)
  if [ "${count:-0}" -eq 0 ]; then
    # Name the chart for the conventional namespaces (aiwb-cnpg / airm-cnpg). The verdict holds
    # for a custom namespace too, but the chart name would not, so hedge the hint rather than
    # asserting a chart that may not exist.
    bad "no CNPG PostgreSQL cluster in $ns — its CNPG infra chart (${ns}-cnpg by convention) was never installed"
    return 1
  fi
  # Capture the phase query on its own first: if the jsonpath query itself errors, treat
  # that as a failed check, NOT as "all healthy" — piping a failed query straight into
  # grep (ending in '|| true') would swallow the error and yield a false healthy.
  local phases
  if ! phases=$(kc get clusters.postgresql.cnpg.io -n "$ns" \
      -o jsonpath='{range .items[*]}{.metadata.name}={.status.phase}{"\n"}{end}' 2>/dev/null); then
    bad "failed to query CNPG cluster status in $ns"
    return 1
  fi
  # A healthy CNPG cluster reports phase exactly 'Cluster in healthy state'. Match the whole
  # phase value (anchored to the '=' and end), NOT a substring: 'Cluster in unhealthy state'
  # contains 'healthy state' and a substring filter would wrongly pass it. Blank lines
  # (unreconciled cluster, empty phase → 'name=') fail to match too, so they are flagged.
  unhealthy=$(printf '%s\n' "$phases" | grep -v '^$' | grep -vE '=Cluster in healthy state$' || true)
  if [ -n "$unhealthy" ]; then
    bad "CNPG cluster(s) in $ns not healthy: ${unhealthy//$'\n'/, }"
    return 1
  fi
  ok "CNPG PostgreSQL healthy in $ns ($count cluster(s))"
}
# BYO PostgreSQL (INSTALL.md "Bringing Your Own Infrastructure") intentionally omits the
# CNPG operator and *-cnpg charts, so there is no CNPG cluster to check — skip rather than
# fail. The DB itself is still exercised indirectly: aiwb-api won't reach Ready without it.
if [ "$external_db" -eq 1 ]; then
  echo "SKIP  CNPG check in $aiwb_ns — --external-db (bring-your-own PostgreSQL)"
else
  cnpg_healthy "$aiwb_ns"
fi

# --- HTTPRoutes: must exist AND be Accepted by the Gateway ---
# Existence alone is not reachability — a route whose parent Gateway doesn't match stays
# un-Accepted and the service is silently unreachable.
httproutes_accepted() { # ns
  local ns="$1" count names status notaccepted
  if ! kc get httproutes.gateway.networking.k8s.io -n "$ns" >/dev/null 2>&1; then
    bad "cannot list HTTPRoutes in $ns — Gateway API CRDs missing, RBAC denial, or API error (not necessarily absent)"
    return 1
  fi
  # Capture the names query and check its exit status before counting, so a failed query is
  # reported as such rather than misread as "no HTTPRoutes".
  if ! names=$(kc get httproutes.gateway.networking.k8s.io -n "$ns" \
      -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null); then
    bad "failed to query HTTPRoute list in $ns"
    return 1
  fi
  count=$(printf '%s\n' "$names" | grep -c . || true)
  if [ "${count:-0}" -eq 0 ]; then
    bad "no HTTPRoutes in $ns — services won't be reachable (check Gateway name/namespace)"
    return 1
  fi
  # Capture the status query on its own first: if the jsonpath query itself errors, treat
  # that as a failed check, NOT as "all Accepted" — piping a failed query straight into
  # grep would swallow the error and yield a false healthy. Keep the jsonpath on one line so
  # the expression starts with '{' (a leading newline can break gojsonpath on real clusters).
  if ! status=$(kc get httproutes.gateway.networking.k8s.io -n "$ns" -o jsonpath='{range .items[*]}{.metadata.name}={range .status.parents[*]}{.conditions[?(@.type=="Accepted")].status}{","}{end}{"\n"}{end}' 2>/dev/null); then
    bad "failed to query HTTPRoute Accepted status in $ns"
    return 1
  fi
  # Each route reports one status per parent. Accept one or more True values (each query
  # result has a trailing comma); keep the route if ANY parent is missing Accepted=True.
  # Examples: route=True, and route=True,True, pass; route=True,False, and route= fail.
  notaccepted=$(printf '%s\n' "$status" | grep -v '^$' | grep -Ev '^[^=]+=(True,)+$' || true)
  if [ -n "$notaccepted" ]; then
    bad "HTTPRoute(s) in $ns not Accepted by the Gateway: ${notaccepted//$'\n'/, } — check gateway.gatewayName/namespace"
    return 1
  fi
  ok "HTTPRoutes Accepted in $ns ($count)"
}
httproutes_accepted "$aiwb_ns"

# --- workloads serving-route auth ---
# A fresh workbench may have no AIM model/serving route yet, so no workloads-extauth policy is
# expected and this is a SKIP. Once any model namespace exists, ai-gateway-discovery stamps one
# namespace-wide policy named workloads-extauth; it must be fail-closed and target the dedicated
# workloads auth handler port (8084), not the body-aware AI-gateway handler.
# Each line: <ns>/workloads-extauth=<failOpen>/<backendRef.namespace>/<backendRef.name>/<port>.
# Capturing the backendRef (not just the port) lets the backend-readiness check below verify the
# EXACT Service the policy routes to, rather than any authz Deployment elsewhere in the cluster.
if ! workloads_policies=$(kc get securitypolicies.gateway.envoyproxy.io -A \
    -o jsonpath='{range .items[?(@.metadata.name=="workloads-extauth")]}{.metadata.namespace}/{.metadata.name}={.spec.extAuth.failOpen}/{.spec.extAuth.http.backendRefs[0].namespace}/{.spec.extAuth.http.backendRefs[0].name}/{.spec.extAuth.http.backendRefs[0].port}{"\n"}{end}' 2>/dev/null); then
  bad "cannot query workloads-extauth SecurityPolicies — workloads serving-route auth unverified"
elif [ -z "$workloads_policies" ]; then
  echo "SKIP  no workloads-extauth SecurityPolicy yet (expected before the first AIM serving route; verify again after deploying a model)"
else
  # Must be fail-closed and route to the dedicated workloads handler port (default 8084, override
  # with --workloads-authz-port). The middle backendRef ns/name is free-form here (correlated for
  # readiness below).
  workloads_bad=$(printf '%s\n' "$workloads_policies" | grep -vE "=false/[^/]+/[^/]+/${workloads_authz_port}\$" || true)
  if [ -n "$workloads_bad" ]; then
    bad "workloads-extauth SecurityPolicy not fail-closed on port ${workloads_authz_port}: ${workloads_bad//$'\n'/, }"
  else
    ok "workloads serving-route auth fail-closed (${workloads_policies//$'\n'/, })"
  fi
  # The fail-closed ext_authz authorizer runs as its own HA Deployment behind the
  # backendRef Service. It is the sole authenticator on the inference hot path, so the policy's
  # OWN backend Service must have ready endpoints — an unrelated authz Deployment being up proves
  # nothing. No ready endpoints means every serving request fails closed (denied): a real outage.
  # Correlate to the exact backendRef Service and check its ready addresses. Keep ONLY lines
  # matching the full <ns>/name=<failOpen>/<brNs>/<brName>/<port> shape before extracting: a
  # partial jsonpath line (e.g. backendRef absent mid-reconcile) left unmatched by sed would
  # otherwise yield a garbage Service name and emit misleading extra FAILs. The malformed policy
  # itself is already flagged by the fail-closed/expected-port check above.
  br_refs=$(printf '%s\n' "$workloads_policies" \
    | grep -E '^[^=]+=[^/]+/[^/]+/[^/]+/[0-9]+$' \
    | sed -E 's#^[^=]+=[^/]+/([^/]+)/([^/]+)/([0-9]+)$#\1 \2 \3#' | sort -u || true)
  while read -r br_ns br_name br_port; do
    [ -n "$br_name" ] || continue
    # The Service must expose the backendRef port AND have ready addresses. Ready addresses on
    # some other port don't help: Envoy resolves the backendRef by the Service port ($br_port),
    # so a Service missing that port cannot route even with ready pods.
    if ! svc_ports=$(kc get svc "$br_name" -n "$br_ns" \
        -o jsonpath='{range .spec.ports[*]}{.port}{"\n"}{end}' 2>/dev/null); then
      bad "cannot query Service $br_ns/$br_name for the ext_authz backend — workloads auth handler readiness unverified"
    # Here-string, not `printf ... | grep -Fqx`: `-q` exits at the first match without
    # draining, and under `pipefail` the writer's SIGPIPE would become this test's status.
    # Negated here, so a poisoned status would fabricate a "port not exposed" failure AND
    # skip the endpoint check below. `<<<` feeds the identical bytes `printf '%s\n'` did —
    # both append exactly one newline — so the match is unchanged. `-F` compares the port as
    # a literal string rather than a regex, which is what "is this port in the list" means.
    elif ! grep -Fqx "$br_port" <<<"$svc_ports"; then
      bad "ext_authz backend Service $br_ns/$br_name does not expose port $br_port — Envoy cannot resolve the workloads-extauth backendRef, so serving requests fail closed (denied)"
    elif ! br_eps=$(kc get endpoints "$br_name" -n "$br_ns" \
        -o jsonpath='{range .subsets[*].addresses[*]}{.ip}{"\n"}{end}' 2>/dev/null); then
      bad "cannot query endpoints for ext_authz backend $br_ns/$br_name — workloads auth handler readiness unverified"
    # Strip every whitespace character and ask whether anything is left. Decided entirely in
    # the shell, and it must stay that way: piping this list into `grep -q '[^[:space:]]'`
    # lets the reader exit on the FIRST non-blank line without draining, the writer take
    # SIGPIPE, and `pipefail` hand that failure here — sending a backend that HAS ready
    # endpoints down the `else` below, so the probe reports a serving outage that does not
    # exist. With no pipe and no reader there is no status to poison. The two forms ask the
    # same question: whitespace is exactly what both ignore, so "some line holds a non-space
    # character" and "the string holds a non-space character" agree on every input.
    elif [ -n "${br_eps//[[:space:]]/}" ]; then
      # Any non-empty address is ready — IPv4 and IPv6 are both valid. Do not require a
      # decimal digit: a valid IPv6 address can contain only hex letters and colons.
      ok "ext_authz backend $br_ns/$br_name exposes port $br_port with ready endpoints"
    else
      bad "ext_authz backend Service $br_ns/$br_name has no ready endpoints — workloads-extauth routes to it, so serving requests fail closed (denied)"
    fi
  done <<EOF
$br_refs
EOF
fi

# --- combined-mode extras: AIRM must be up, not merely installed ---
if [ "$standalone" -eq 0 ]; then
  deploy_ready "$airm_ns" "${airm_release}-api"
  deploy_ready "$airm_ns" "${airm_release}-ui"
  if [ "$external_db" -eq 1 ]; then
    echo "SKIP  CNPG check in $airm_ns — --external-db (bring-your-own PostgreSQL)"
  else
    cnpg_healthy "$airm_ns"
  fi
  httproutes_accepted "$airm_ns"

  # RabbitMQ: existence is not readiness — require the operator's Ready condition.
  if kc get rabbitmqclusters.rabbitmq.com -n "$airm_ns" >/dev/null 2>&1; then
    if ! rq_names=$(kc get rabbitmqclusters.rabbitmq.com -n "$airm_ns" \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2>/dev/null); then
      bad "failed to query RabbitMQ cluster list in $airm_ns"
      rn=-1
    else
      rn=$(printf '%s\n' "$rq_names" | grep -c . || true)
    fi
    if [ "${rn:-0}" -eq -1 ]; then
      : # query failure already reported above
    elif [ "${rn:-0}" -eq 0 ]; then
      bad "no RabbitMQ cluster in $airm_ns — AIRM↔Agent messaging won't work"
    elif ! rq_status=$(kc get rabbitmqclusters.rabbitmq.com -n "$airm_ns" -o jsonpath='{range .items[*]}{.metadata.name}={.status.conditions[?(@.type=="AllReplicasReady")].status}{"\n"}{end}' 2>/dev/null); then
      bad "failed to query RabbitMQ readiness in $airm_ns"
    else
      rq_bad=$(printf '%s\n' "$rq_status" | grep -v '^$' | grep -v '=True$' || true)
      if [ -n "$rq_bad" ]; then
        bad "RabbitMQ cluster(s) in $airm_ns not ready: ${rq_bad//$'\n'/, }"
      else
        ok "RabbitMQ ready in $airm_ns ($rn cluster(s))"
      fi
    fi
  else
    bad "cannot list RabbitMQ clusters in $airm_ns — CRD missing, RBAC denial, or API error (not necessarily absent); operator/CRD required for combined mode"
  fi

  # Cluster onboarding (INSTALL.md Step 5). Only present when includeDemoSetup is enabled;
  # absent is reported as a skip, but a present-and-failed Job is a real problem. Only an
  # explicit NotFound may skip: treating an RBAC or transient failure as "absent" would drop
  # the check silently and let the run still report HEALTHY.
  if job_err=$(kc get job "${airm_release}-configure" -n "$airm_ns" 2>&1 >/dev/null); then
    if ! succeeded=$(kc get job "${airm_release}-configure" -n "$airm_ns" \
        -o jsonpath='{.status.succeeded}' 2>/dev/null); then
      bad "failed to query onboarding Job ${airm_release}-configure status in $airm_ns"
    elif [ "${succeeded:-0}" -ge 1 ]; then
      ok "cluster onboarding Job ${airm_release}-configure completed"
    else
      bad "cluster onboarding Job ${airm_release}-configure has not succeeded — cluster not registered (kubectl logs job/${airm_release}-configure -n $airm_ns)"
    fi
  else
    case "$job_err" in
      *[Nn]otFound*|*'not found'*)
        echo "SKIP  no ${airm_release}-configure Job (expected when includeDemoSetup=false; onboard the cluster manually per INSTALL.md Step 5)" ;;
      *)
        bad "cannot query onboarding Job ${airm_release}-configure in $airm_ns — ${job_err:-kubectl error} (not necessarily absent)" ;;
    esac
  fi
fi

echo
if [ "$problems" -eq 0 ]; then
  echo "RESULT: HEALTHY — every check above passed (see the OK lines for exactly what was verified)."
  echo "Access URLs use your appDomain: https://aiwbapi.<domain>, https://aiwbui.<domain> (see references/verification.md)."
  echo "Note: this probe covers the platform's core components, not GPU/model-serving features."
  exit 0
fi
echo "RESULT: INCONCLUSIVE / WHAT TO FIX — $problems component(s) not verified healthy (see FAIL lines above)."
echo "This is NOT a successful install. Triage each FAIL via references/troubleshooting.md before retrying."
exit 1
