#!/usr/bin/env bash

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
# Read-only preflight probe for an AIWB install.
# Reports cluster readiness: reachability, tooling, StorageClass, required CRDs/operators,
# Gateway, GPU capacity, and target namespaces. Never mutates the cluster.
#
# Usage: preflight-check.sh [--secret-store <name>] [--no-openbao]
#                           [--manual-bootstrap-secrets] [--request-timeout <duration>]
#
# --secret-store  ClusterSecretStore configured by both the AIWB app chart
#                 (openBao.secretStoreRef) and external-secrets chart
#                 (externalSecretStore). Default: openbao-secret-store.
# --no-openbao    The install sets openBao.enabled=false (AIWB app chart) and
#                 openbaoToken.enabled=false (external-secrets chart), so there is no
#                 API-key store to require. The ClusterSecretStore is then needed only for
#                 bootstrap-secret sync (INSTALL.md Option A), and the check says so.
# --manual-bootstrap-secrets
#                 The bootstrap Secrets are created by hand (INSTALL.md Option B) and the
#                 external-secrets chart is not installed. Alone this changes nothing: the
#                 ClusterSecretStore is still the API-key store. Together with --no-openbao
#                 the store has no role left, so its check is skipped with a SKIP line.
# --request-timeout
#                 Per-request kubectl timeout (default: 30s). Bounds every cluster call so a
#                 reachable-but-degraded API server surfaces as a reported failure instead of
#                 hanging forever — an unattended agent cannot interrupt a wedged probe. 30s is
#                 chosen to stay above a slow cluster-wide list without stalling a run; raise it
#                 on a large or distant cluster.
#
# Exit codes: 0 = every checked prerequisite present (READY); 1 = a hard blocker OR an
# unconfirmed prerequisite (NOT READY / INCOMPLETE); 2 = cluster unreachable;
# 64 = usage error (unknown option or missing option value).
# Only exit 0 is a green light — a WARN is an unmet prerequisite the operator must resolve
# or explicitly waive for their deployment mode (see references/preflight.md), because
# installing past one is what produces the partial install this skill exists to prevent.
# Findings are printed as OK / NOTE / SKIP / WARN / BLOCK lines (NOTE and SKIP are
# informational and do not affect the result).
set -uo pipefail

secret_store="openbao-secret-store"
openbao_enabled=1
manual_bootstrap_secrets=0
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
# and let kubectl's own parse error surface as a bogus cluster verdict. Zero is rejected
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
    --secret-store) require_value "$1" "$#" "${2-}"; secret_store="$2"; shift 2 ;;
    --no-openbao) openbao_enabled=0; shift ;;
    --manual-bootstrap-secrets) manual_bootstrap_secrets=1; shift ;;
    --request-timeout)
      require_value "$1" "$#" "${2-}"
      require_duration "$1" "$2"
      request_timeout="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

# Every cluster read goes through this wrapper so no call can hang unbounded against a
# reachable-but-degraded API server: the probe must report a failure, not stall an agent that
# has no way to interrupt it.
kc() { kubectl --request-timeout="$request_timeout" "$@"; }

blockers=0
warns=0
ok()    { printf 'OK    %s\n' "$1"; }
warn()  { printf 'WARN  %s\n' "$1"; warns=$((warns+1)); }
block() { printf 'BLOCK %s\n' "$1"; blockers=$((blockers+1)); }
# Purely informational — does NOT count toward warns, so it can never gate READY.
note()  { printf 'NOTE  %s\n' "$1"; }
# A check that the selected install mode makes irrelevant. Printed so the omission is
# visible in the report; never counts toward warns or blockers.
skip()  { printf 'SKIP  %s\n' "$1"; }
# Literal substring test, used below to classify captured kubectl stderr. Deliberately NOT
# `printf ... | grep -q`: `grep -q` returns at its first match without draining its input, so
# the writer can still hold an unwritten tail when the reader exits, take SIGPIPE/EPIPE, and
# — under this script's `pipefail` — hand that failure to the caller as the pipeline's status.
# A match would then read as a non-match, which at every call site below silently downgrades a
# precise verdict into "cannot query ... RBAC or API error". `case` has no reader, no pipe and
# no subprocess, so there is no status to poison. The needle is quoted, so it matches as a
# literal rather than as a pattern, and matching the whole string rather than each line is
# equivalent for needles that contain no newline — none here do.
contains() { # haystack  needle
  case "$1" in *"$2"*) return 0 ;; *) return 1 ;; esac
}

echo "== AIWB install preflight =="

# --- Tooling ---
# kubectl absence is a tooling error, not a cluster problem — fail here with exit 1 so it
# is not misclassified as "unreachable" (exit 2) by the reachability probe below, which
# would itself fail simply because kubectl cannot run.
if ! command -v kubectl >/dev/null 2>&1; then
  block "kubectl not found on PATH — install kubectl before running preflight"
  echo
  echo "SUMMARY: blockers=$blockers warns=$warns"
  echo "RESULT: NOT READY — kubectl is required to check the cluster."
  exit 1
fi
if ! command -v helm >/dev/null 2>&1; then
  block "helm not found on PATH (Helm 3.x or 4.x required)"
elif ! helm_version=$(helm version --short 2>&1); then
  block "helm present but 'helm version --short' failed — broken install or unusable binary: ${helm_version:-no output}"
# Here-string, not `printf ... | grep -qE`: `-q` exits at the first match without draining,
# and under `pipefail` the writer's SIGPIPE would become this test's status, falling through
# to the `else` below and BLOCKing a perfectly good Helm 3.x/4.x. The alternation and anchor
# need a real regex, so grep stays — only the pipe goes. `<<<` feeds the identical bytes
# `printf '%s\n'` did, both appending exactly one newline, so the match is unchanged.
elif grep -qE '(^|[^0-9])v[34]\.[0-9]' <<<"$helm_version"; then
  # Match a v3.x or v4.x token ANYWHERE in the output, not just as the whole blob's prefix: a
  # helm wrapper or plugin hook can prepend a banner/warning line (stderr is merged in above),
  # which a strict prefix check would misread as an unsupported version and falsely BLOCK.
  ok "Helm 3.x or 4.x present ($(printf '%s' "$helm_version" | tr '\n' ' '))"
else
  block "Helm 3.x or 4.x required; found '$(printf '%s' "${helm_version:-unknown version}" | tr '\n' ' ')'"
fi

# --- Cluster reachability (hard gate) ---
if ! kc cluster-info >/dev/null 2>&1; then
  block "cluster unreachable — kubectl cannot reach the API server"
  echo
  echo "Cluster is unreachable; skipping remaining checks. Fix kubeconfig/context/access first."
  echo "SUMMARY: blockers=$blockers warns=$warns"
  exit 2
fi
ok "cluster reachable ($(kc config current-context 2>/dev/null))"

# --- Permissions ---
# `kubectl auth can-i` exits 1 for a legitimate "no", not only on an API error, so the exit
# status alone cannot tell a denial from a fault. Branch on the answer it prints instead:
# "no" is a permission denial, and anything else with a failed query is a real fault.
namespace_access=$(kc auth can-i create namespaces 2>/dev/null)
case "$namespace_access" in
  yes) ok "can create namespaces" ;;
  no)  block "insufficient permissions — cannot create namespaces (need cluster-admin-equivalent)" ;;
  *)   block "cannot check namespace-create permission — RBAC or API error; resolve access before installing" ;;
esac

# --- StorageClass ---
# Distinguish a failed query (RBAC/transient API error) from a genuinely empty list, so a
# transient error is not misreported as the specific "no StorageClass" blocker.
if ! sc_list=$(kc get storageclass -o name 2>/dev/null); then
  block "cannot list StorageClasses — RBAC or API error (not necessarily absent); resolve access before installing"
elif [ -z "$sc_list" ]; then
  block "no StorageClass found — CNPG/RabbitMQ PVCs will never bind"
else
  # Reuse the already-successful names query rather than issuing a second jsonpath call whose
  # failure could print a false `OK StorageClass(es) present:` with an empty list.
  sc_names=$(printf '%s\n' "$sc_list" | sed 's#^storageclass.storage.k8s.io/##' | paste -sd' ' -)
  ok "StorageClass(es) present: ${sc_names}"
  # The list above already proved read access, so a failure here is almost certainly a genuine
  # absence — but report a non-NotFound error as itself rather than sending the operator off to
  # add a per-chart override they may not need.
  if sc_default_error=$(kc get storageclass default 2>&1 >/dev/null); then
    ok "StorageClass 'default' present (chart default; no override needed)"
  elif contains "$sc_default_error" NotFound; then
    warn "no StorageClass named 'default' — pass per-chart override (see StorageClass gotcha in SKILL.md)"
  else
    warn "cannot query StorageClass 'default' — RBAC or API error (not necessarily absent)"
  fi
fi

# --- AIM Engine CRD (hard blocker for every AIWB install) ---
if aim_crd_error=$(kc get crd aimclusterruntimeconfigs.aim.eai.amd.com 2>&1 >/dev/null); then
  ok "AIM Engine CRD AIMClusterRuntimeConfig present"
elif contains "$aim_crd_error" NotFound; then
  block "AIM Engine CRDs missing — AIWB chart creates AIMClusterRuntimeConfig; install AIM Engine CRDs first"
else
  block "cannot query AIM Engine CRD — RBAC or API error (not necessarily absent); resolve access before installing"
fi

# --- Gateway (default: 'https' in envoy-gateway-system) ---
if kc get gateway.gateway.networking.k8s.io -A >/dev/null 2>&1; then
  if kc get gateway.gateway.networking.k8s.io https -n envoy-gateway-system >/dev/null 2>&1; then
    ok "Gateway 'https' present in envoy-gateway-system (chart default)"
  else
    # Capture the list and check its exit status before deciding it is empty: a failed
    # --no-headers query would otherwise yield an empty gw and misreport "no Gateway found"
    # (a BLOCK) on a healthy cluster.
    if ! gw_raw=$(kc get gateway.gateway.networking.k8s.io -A --no-headers 2>/dev/null); then
      block "cannot list Gateway resources — RBAC or API error (not necessarily absent); resolve access before installing"
    else
      # Join with a literal ", " — `paste -sd', '` alternates ',' and ' ' as separate
      # single-char delimiters, which would render "a,b c" instead of "a, b, c".
      gw=$(printf '%s\n' "$gw_raw" | awk 'NF{print $2" (ns "$1")"}' | awk 'NR>1{printf ", "}{printf "%s",$0}END{print ""}')
      if [ -n "$gw" ]; then
        warn "Gateway present but not the default 'https/envoy-gateway-system': ${gw} — override gateway.gatewayName/gateway.namespace at install"
      else
        block "no Gateway resource found — services won't be reachable; install a Gateway API impl + Gateway"
      fi
    fi
  fi
else
  block "cannot list Gateways — Gateway API CRDs missing (install Gateway API + a controller e.g. kgateway/Envoy Gateway), or insufficient RBAC to read them"
fi

# --- Mode-dependent operators (reported as WARN; see references/preflight.md) ---
crd_check() { # crd-name  ok-msg  warn-msg
  local crd_error
  if crd_error=$(kc get crd "$1" 2>&1 >/dev/null); then
    ok "$2"
  elif contains "$crd_error" NotFound; then
    warn "$3"
  else
    warn "cannot query CRD '$1' — RBAC or API error (not necessarily absent)"
  fi
}
crd_check clusters.postgresql.cnpg.io "CNPG operator CRDs present" \
  "CNPG operator CRDs missing — needed unless bringing your own PostgreSQL"
crd_check rabbitmqclusters.rabbitmq.com "RabbitMQ operator CRDs present" \
  "RabbitMQ operator CRDs missing — needed for combined (AIRM) mode"
crd_check clusterpolicies.kyverno.io "Kyverno present" \
  "Kyverno missing — required only for standalone AIWB (standAloneMode=true)"
# ESO serves the API-key ExternalSecrets and the bootstrap-secret sync; the two Step 1 flags
# remove those roles one at a time. The CRD stays a WARN even when both are gone, because
# combined (AIRM) mode still mints per-project credentials through ESO.
if [ "$openbao_enabled" -eq 0 ] && [ "$manual_bootstrap_secrets" -eq 1 ]; then
  eso_warn="External Secrets Operator missing — AIWB has no ExternalSecrets left with --no-openbao and --manual-bootstrap-secrets; combined (AIRM) mode still needs ESO for per-project credentials"
elif [ "$openbao_enabled" -eq 0 ]; then
  eso_warn="External Secrets Operator missing — required for bootstrap-secret sync (INSTALL.md Option A); --no-openbao removes only the API-key ExternalSecrets"
else
  eso_warn="External Secrets Operator missing — required at runtime for AIWB API-key ExternalSecrets; manual Option B replaces bootstrap-secret sync only"
fi
crd_check externalsecrets.external-secrets.io "External Secrets Operator present" "$eso_warn"
crd_check certificates.cert-manager.io "cert-manager present" \
  "cert-manager missing — needed for combined mode (RabbitMQ TLS / agent webhook)"

# Remaining Tier 1-3 dependencies from helm/INSTALL.md. Each is required for the full
# platform; several are waivable on a local dev cluster or when substituting your own
# service, so they are WARN rather than BLOCK — see references/preflight.md.
crd_check opentelemetrycollectors.opentelemetry.io "OpenTelemetry Operator present" \
  "OpenTelemetry Operator missing — AIWB deploys a vLLM metrics collector (INSTALL.md Tier 1)"
crd_check scaledobjects.keda.sh "KEDA present" \
  "KEDA missing — required by KServe to autoscale InferenceService pods (INSTALL.md Tier 1)"
crd_check servicemonitors.monitoring.coreos.com "Prometheus Operator CRDs present" \
  "Prometheus Operator CRDs missing — ServiceMonitor/PodMonitor needed by the observability stack (INSTALL.md Tier 1)"
crd_check keycloaks.k8s.keycloak.org "Keycloak operator present" \
  "Keycloak not detected — AIRM requires the Keycloak Admin API and it is NOT substitutable (INSTALL.md Tier 2)"
crd_check tenants.minio.min.io "MinIO operator present" \
  "MinIO not detected — AIWB needs S3-compatible storage; waivable only if bringing your own S3 (INSTALL.md Tier 2)"
crd_check kaiwojobs.kaiwo.silogen.ai "Kaiwo present" \
  "Kaiwo missing — AI workload orchestration/queueing on GPU clusters (INSTALL.md Tier 3)"
crd_check inferenceservices.serving.kserve.io "KServe present" \
  "KServe missing — needed for model serving; configure for RawDeployment mode (INSTALL.md Tier 3)"

# ClusterSecretStore: the configured OpenBao/ESO store is the sole API-key path
# (helm/aiwb/values.yaml), and with INSTALL.md Option A the external-secrets chart also syncs
# the bootstrap Secrets through it. The store name is configurable in both charts, so probe
# the caller-selected value. Existence is not health: ESO only considers the backend usable
# when its Ready condition is explicitly True. The check is a function of the two Step 1
# choices: --no-openbao drops the API-key role, --manual-bootstrap-secrets drops the sync
# role, and only when both are gone is there nothing left to require.
if [ "$openbao_enabled" -eq 0 ] && [ "$manual_bootstrap_secrets" -eq 1 ]; then
  skip "ClusterSecretStore '$secret_store' check — --no-openbao (no API-key store) and --manual-bootstrap-secrets (no bootstrap-secret sync)"
else
  if [ "$openbao_enabled" -eq 1 ]; then
    store_role="OpenBao API-key store"
  else
    store_role="bootstrap-secret sync store (--no-openbao: no API-key store expected)"
  fi
  if ! store_ready=$(kc get clustersecretstores.external-secrets.io "$secret_store" \
      -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null); then
    warn "cannot query $store_role '$secret_store' — ClusterSecretStore missing, unreadable, or query failed"
  elif [ "$store_ready" = "True" ]; then
    ok "$store_role ready (ClusterSecretStore '$secret_store')"
  elif [ "$openbao_enabled" -eq 1 ]; then
    warn "$store_role '$secret_store' not Ready (status=${store_ready:-unset}) — ESO cannot materialize API keys"
  else
    warn "$store_role '$secret_store' not Ready (status=${store_ready:-unset}) — ESO cannot sync the bootstrap Secrets through it"
  fi
fi

# SECURITY: preflight proves the controller can reconcile both auth paths:
# per-model apiKeyAuth for AI-gateway routes and the namespace-wide fail-closed workloads-extauth
# policy for workloads serving routes. Actual policy state is checked post-install once a model
# namespace/serving route exists; do not infer traffic denial from these prerequisite probes.
if security_policy_error=$(kc get crd securitypolicies.gateway.envoyproxy.io 2>&1 >/dev/null); then
  ok "Envoy Gateway SecurityPolicy CRD present"
elif contains "$security_policy_error" NotFound; then
  warn "Envoy Gateway SecurityPolicy CRD missing — per-model apiKeyAuth cannot be enforced; AIM serving routes would be unauthenticated"
else
  warn "cannot query Envoy Gateway SecurityPolicy CRD — RBAC or API error (not necessarily absent)"
fi

# A Helm release may use any name, so identify the controller by stable env contracts rather
# than a release-name-derived label. Require WORKLOADS_AUTHZ_PORT as well as the older
# API_KEY_AUTH_ANNOTATION contract: an old controller can create per-model policies but lacks
# the namespace-wide fail-closed workloads-route guard.
if ! agd_status=$(kc get deploy -A -o jsonpath='{range .items[*]}{.metadata.namespace}/{.metadata.name}={.status.conditions[?(@.type=="Available")].status}{"|"}{range .spec.template.spec.containers[*].env[?(@.name=="API_KEY_AUTH_ANNOTATION")]}{.name}{end}{"|"}{range .spec.template.spec.containers[*].env[?(@.name=="WORKLOADS_AUTHZ_PORT")]}{.name}{end}{"\n"}{end}' 2>/dev/null); then
  warn "cannot query ai-gateway-discovery readiness — inference-route auth enforcement is unconfirmed"
elif agd_ready=$(printf '%s\n' "$agd_status" | grep -E '=True\|API_KEY_AUTH_ANNOTATION\|WORKLOADS_AUTHZ_PORT$' || true) && [ -n "$agd_ready" ]; then
  ok "ai-gateway-discovery controller ready with AI-gateway + workloads-route auth (${agd_ready//$'\n'/, })"
else
  warn "ai-gateway-discovery controller not ready or missing the WORKLOADS_AUTHZ_PORT contract — upgrade to a version with workloads-route fail-closed auth"
fi

# --- Node schedulability and GPU capacity ---
# One query serves both checks so they can never disagree about which nodes are usable.
# Emits "<name>=<unschedulable>/<Ready>/<allocatable amd.com/gpu>". A usable node is Ready=True
# and not cordoned; .spec.unschedulable is absent (empty) on a normal node but an explicit
# "false" after `kubectl uncordon`, so both count as schedulable — only "true" is cordoned.
# Capture and check the query exit status so a failed list (RBAC/transient) is reported as
# such, not hidden behind "could not determine node count" or misread as zero GPUs.
if ! node_states=$(kc get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"="}{.spec.unschedulable}{"/"}{.status.conditions[?(@.type=="Ready")].status}{"/"}{.status.allocatable.amd\.com/gpu}{"\n"}{end}' 2>/dev/null); then
  warn "cannot list nodes — RBAC or API error; node count and GPU capacity are unconfirmed"
else
  usable_nodes=$(printf '%s\n' "$node_states" | grep -E '=(false)?/True/' || true)
  total_nodes=$(printf '%s\n' "$node_states" | grep -c . || true)
  node_count=$(printf '%s\n' "$usable_nodes" | grep -c . || true)

  # aiwb-cnpg defaults to 3 PostgreSQL instances spread over nodes. Fewer nodes than
  # instances still installs, but they co-schedule (no added resilience, 3x resources) and
  # the resulting PodDisruptionBudget blocks draining that node — so flag it before install.
  if [ "${total_nodes:-0}" -eq 0 ]; then
    warn "could not determine node count"
  elif [ "${node_count:-0}" -ge 3 ]; then
    ok "schedulable nodes: ${node_count}/${total_nodes} (enough to spread the aiwb-cnpg default of 3 instances)"
  else
    warn "only ${node_count} schedulable node(s) of ${total_nodes} — aiwb-cnpg defaults to 3 instances; install with '--set instances=1' on single-node clusters (INSTALL.md 'Node Requirements for CNPG Clusters')"
  fi

  # Sum GPUs on usable nodes ONLY. A cordoned or NotReady node keeps its last reported
  # allocatable value, so counting every node would report capacity no workload can be
  # scheduled onto — the false-healthy claim this probe exists to avoid.
  sum_gpus() { awk -F/ '{s+=$NF} END{print s+0}'; }
  gpu_total=$(printf '%s\n' "$usable_nodes" | sum_gpus)
  gpu_stranded=$(printf '%s\n' "$node_states" | grep -vE '=(false)?/True/' | sum_gpus)
  if [ "${gpu_total:-0}" -gt 0 ]; then
    ok "allocatable AMD GPUs on schedulable nodes: ${gpu_total}"
  else
    warn "0 allocatable amd.com/gpu on schedulable nodes — expected on local dev clusters; on a GPU cluster the GPU operator hasn't rolled out"
  fi
  # Report stranded capacity separately: it is real hardware, but unschedulable until the node
  # is uncordoned or becomes Ready, so it must never be added to the usable total above.
  if [ "${gpu_stranded:-0}" -gt 0 ]; then
    warn "${gpu_stranded} allocatable amd.com/gpu sit on cordoned or NotReady nodes — no workload can schedule onto them until those nodes recover"
  fi
fi

# --- Namespaces (informational only — must not gate READY) ---
# Checks the DEFAULT release namespaces. A pre-existing one is fine (--create-namespace is
# idempotent), so this is a NOTE, not a WARN: scoring it would make every install into a
# pre-existing namespace report INCOMPLETE and never READY. Wording stays non-interpretive —
# the namespace may exist from a prior install or an unrelated reason, and a custom install
# can target a different namespace entirely.
for ns in airm aiwb; do
  if namespace_error=$(kc get ns "$ns" 2>&1 >/dev/null); then
    note "default namespace '$ns' already exists (fine; --create-namespace is idempotent)"
  elif contains "$namespace_error" NotFound; then
    ok "default namespace '$ns' absent"
  else
    warn "cannot query default namespace '$ns' — RBAC or API error (not necessarily absent)"
  fi
done

echo
echo "SUMMARY: blockers=$blockers warns=$warns"
if [ "$blockers" -gt 0 ]; then
  echo "RESULT: NOT READY — resolve BLOCK items before installing (WARN items are mode-dependent; see references/preflight.md)."
  exit 1
fi
if [ "$warns" -gt 0 ]; then
  echo "RESULT: INCOMPLETE — no hard blockers, but $warns prerequisite(s) are unconfirmed (WARN above)."
  echo "This is NOT a green light. Resolve each WARN or explicitly waive it as not-needed for your"
  echo "deployment mode (references/preflight.md) before installing; installing past an unmet"
  echo "prerequisite is what produces a partial install."
  exit 1
fi
echo "RESULT: READY — every checked prerequisite is present; proceed to install."
exit 0
