#!/usr/bin/env bash

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
# Regression tests for the install probes.
#
# The probes' job is to never claim health they did not verify, so each case pins a way
# they previously could report a false pass. Tests use a stub kubectl: no cluster is
# required and nothing is mutated.
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
preflight="${here}/../scripts/preflight-check.sh"
verify="${here}/../scripts/verify-install.sh"

pass=0
fail=0
out=""
code=0

run_with_stub() { # stub_arms  cmd...
  local stub_arms="$1"; shift
  local dir
  dir="$(mktemp -d)" || { echo "FATAL mktemp -d failed" >&2; exit 1; }
  {
    echo '#!/usr/bin/env bash'
    echo 'case "$*" in'
    printf '%s\n' "$stub_arms"
    # Catch-all exit 0 is deliberate: it models kubectl's "resource absent / nothing to
    # report" default (empty stdout, success), which several cases rely on — e.g. no
    # workloads-extauth policy must read as SKIP, not error. Cases that must prove a call is
    # NOT made assert it explicitly (see the "SHOULD-NOT-BE-CALLED" check_count below).
    echo '  *) exit 0 ;;'
    echo 'esac'
  } > "${dir}/kubectl"
  chmod +x "${dir}/kubectl"
  # Default helm stub reports a healthy v3. Override with STUB_HELM_BODY to exercise the
  # version-validation paths (broken binary, wrong major version).
  printf '#!/usr/bin/env bash\n%s\n' "${STUB_HELM_BODY:-echo \"v3.0.0\"}" > "${dir}/helm"
  chmod +x "${dir}/helm"
  out=$(PATH="${dir}:$PATH" "$@" 2>&1)
  code=$?
  rm -rf "${dir}"
}

check() { # description  expected_code  expected_substring
  local desc="$1" want_code="$2" want_text="$3"
  if [ "$code" -ne "$want_code" ]; then
    printf 'FAIL  %s\n      expected exit %s, got %s\n' "$desc" "$want_code" "$code"
    fail=$((fail+1))
    return
  fi
  # Substring test in the shell itself, deliberately NOT `printf ... | grep -qF`. `grep -q`
  # stops at the first match without draining its input, so the writer could be left holding
  # an unwritten tail; with `pipefail` on, the resulting SIGPIPE/EPIPE became the pipeline's
  # status and `!` turned a *successful* match into "output missing". Which assertion lost
  # the race depended on scheduling, so it reported a different, arbitrary healthy test on
  # each occurrence. `case` has no reader, no pipe, and no exit status to poison. The needle
  # is quoted, so it is matched literally (as `-F` did), and no needle here spans lines.
  case "$out" in
    *"$want_text"*) ;;
    *)
      printf 'FAIL  %s\n      output missing: %s\n' "$desc" "$want_text"
      fail=$((fail+1))
      return
      ;;
  esac
  printf 'PASS  %s\n' "$desc"
  pass=$((pass+1))
}

check_count() { # description  expected_substring  expected_occurrences
  local desc="$1" want_text="$2" want_n="$3" got_n
  # Here-string rather than a pipe, for the reason spelled out in `check` above. `grep -c`
  # has to read to EOF to count, so it does not exit early today — but that is a property of
  # one flag, not of the construct: `-m` makes grep stop at the limit, and on a *pipe* that
  # brings the flake straight back. Over a here-string it does not, because there is no
  # reader to lose and no pipeline status to poison — which is the point of reading from one
  # here rather than relying on `-c` alone. It supplies the same single trailing newline
  # `printf '%s\n'` did, so the line count is unchanged.
  got_n=$(grep -cF -- "$want_text" <<<"$out" || true)
  if [ "$got_n" -ne "$want_n" ]; then
    printf 'FAIL  %s\n      expected %s occurrence(s) of "%s", got %s\n' "$desc" "$want_n" "$want_text" "$got_n"
    fail=$((fail+1))
    return
  fi
  printf 'PASS  %s\n' "$desc"
  pass=$((pass+1))
}

# Run a probe with kubectl absent from PATH (only kubectl-containing dirs are dropped, so
# coreutils stay available). If kubectl is already absent, use PATH as-is rather than SKIP —
# that environment can execute the regression directly. Returns 1 only when kubectl cannot
# be isolated without losing required core tools. helm is stubbed so a real helm does not matter.
run_without_kubectl() { # cmd...
  local entry filtered="" dir
  if command -v kubectl >/dev/null 2>&1; then
    # Drop EVERY PATH entry that has an executable kubectl (there can be several), keeping
    # all other dirs so coreutils remain resolvable.
    local IFS=:
    for entry in $PATH; do
      [ -n "$entry" ] || continue
      [ -x "${entry}/kubectl" ] && continue
      filtered="${filtered:+$filtered:}${entry}"
    done
  else
    filtered="$PATH"
  fi
  PATH="$filtered" command -v kubectl >/dev/null 2>&1 && return 1
  # The probes need these core tools too (and bash, which invokes them); if kubectl shared a
  # dir with any of them, dropping that dir would break the probe for the WRONG reason. Bail to
  # SKIP rather than mis-fail.
  local tool
  for tool in bash grep mktemp sed awk paste sort; do
    PATH="$filtered" command -v "$tool" >/dev/null 2>&1 || return 1
  done
  dir="$(mktemp -d)" || { echo "FATAL mktemp -d failed" >&2; exit 1; }
  printf '#!/usr/bin/env bash\necho "v3.0.0"\n' > "${dir}/helm"
  chmod +x "${dir}/helm"
  out=$(PATH="${dir}:${filtered}" "$@" 2>&1)
  code=$?
  rm -rf "${dir}"
  return 0
}

READY_BASE='
  *"cluster-info") exit 0 ;;
  *"get deploy"*"{.spec.replicas}"*) echo 1 ;;
  *"get deploy"*"{.status.readyReplicas}"*) echo 1 ;;
  *"get deploy"*) exit 0 ;;'

HEALTHY_ROUTES='
  *"get httproutes"*Accepted*) echo "r=True," ;;
  *"get httproutes"*"metadata.name"*) echo "r" ;;'

HEALTHY_CNPG='
  *"get clusters.postgresql.cnpg.io"*"status.phase"*) echo "db=Cluster in healthy state" ;;
  *"get clusters.postgresql.cnpg.io"*) echo "db" ;;'

HEALTHY_RABBIT='
  *"get rabbitmqclusters"*AllReplicasReady*) echo "rmq=True" ;;
  *"get rabbitmqclusters"*"metadata.name"*) echo "rmq" ;;'

echo "== verify-install.sh =="

run_with_stub "${READY_BASE}${HEALTHY_ROUTES}
  *\"get clusters.postgresql.cnpg.io\"*) exit 0 ;;" \
  bash "$verify" --standalone
check "empty CNPG list fails instead of vacuously passing" 1 "no CNPG PostgreSQL cluster"

# One cluster listed, but the phase status query itself errors: the query failure must be
# a FAIL, not swallowed by '|| true' into a false healthy.
run_with_stub "${READY_BASE}${HEALTHY_ROUTES}
  *\"get clusters.postgresql.cnpg.io\"*\"status.phase\"*) exit 1 ;;
  *\"get clusters.postgresql.cnpg.io\"*) echo \"db\" ;;" \
  bash "$verify" --standalone
check "CNPG status-query failure fails instead of false healthy" 1 "failed to query CNPG cluster status"

# 'Cluster in unhealthy state' contains the substring 'healthy state'; an exact-match filter
# must still flag it as unhealthy rather than passing it via substring.
run_with_stub "${READY_BASE}${HEALTHY_ROUTES}
  *\"get clusters.postgresql.cnpg.io\"*\"status.phase\"*) echo \"db=Cluster in unhealthy state\" ;;
  *\"get clusters.postgresql.cnpg.io\"*) echo \"db\" ;;" \
  bash "$verify" --standalone
check "CNPG 'unhealthy state' phase is not passed by substring match" 1 "not healthy"

# BYO PostgreSQL (INSTALL.md "Bringing Your Own Infrastructure") has no CNPG cluster; with
# --external-db the CNPG check is skipped, so a healthy external-DB install passes instead
# of failing INCONCLUSIVE for the (correctly) absent CNPG cluster.
# CNPG names query failure (CRD present) must report a query error, not "chart not installed".
run_with_stub "${READY_BASE}${HEALTHY_ROUTES}
  *\"get clusters.postgresql.cnpg.io\"*\"metadata.name\"*) exit 1 ;;
  *\"get clusters.postgresql.cnpg.io\"*) exit 0 ;;" \
  bash "$verify" --standalone
check "CNPG list-query failure is distinct from not-installed" 1 "failed to query CNPG cluster list"

run_with_stub "${READY_BASE}${HEALTHY_ROUTES}
  *\"get clusters.postgresql.cnpg.io\"*) exit 1 ;;" \
  bash "$verify" --standalone --external-db
check "BYO PostgreSQL (--external-db) skips CNPG and still passes" 0 "RESULT: HEALTHY"
check "external-db prints an explicit CNPG skip line" 0 "SKIP  CNPG check in aiwb"

run_with_stub "${READY_BASE}${HEALTHY_CNPG}
  *\"get httproutes\"*Accepted*) echo \"r=False,\" ;;
  *\"get httproutes\"*\"metadata.name\"*) echo \"r\" ;;" \
  bash "$verify" --standalone
check "HTTPRoute that exists but is not Accepted fails" 1 "not Accepted by the Gateway"

# A route whose Accepted status is blank (not yet reconciled) must fail, not be treated
# as Accepted because the value simply isn't the string True.
run_with_stub "${READY_BASE}${HEALTHY_CNPG}
  *\"get httproutes\"*Accepted*) echo \"r=\" ;;
  *\"get httproutes\"*\"metadata.name\"*) echo \"r\" ;;" \
  bash "$verify" --standalone
check "HTTPRoute with blank Accepted status fails" 1 "not Accepted by the Gateway"

# Multi-parent routes report one Accepted value per parent. All True values pass; any
# non-True parent fails.
run_with_stub "${READY_BASE}${HEALTHY_CNPG}
  *\"get httproutes\"*Accepted*) echo \"r=True,True,\" ;;
  *\"get httproutes\"*\"metadata.name\"*) echo \"r\" ;;" \
  bash "$verify" --standalone
check "HTTPRoute with multiple Accepted parents passes" 0 "RESULT: HEALTHY"

run_with_stub "${READY_BASE}${HEALTHY_CNPG}
  *\"get httproutes\"*Accepted*) echo \"r=True,False,\" ;;
  *\"get httproutes\"*\"metadata.name\"*) echo \"r\" ;;" \
  bash "$verify" --standalone
check "HTTPRoute with one rejected parent fails" 1 "not Accepted by the Gateway"

# HTTPRoute list-query failure must report a query error, not "no HTTPRoutes".
run_with_stub "${READY_BASE}${HEALTHY_CNPG}
  *\"get httproutes\"*\"metadata.name\"*) exit 1 ;;
  *\"get httproutes\"*) exit 0 ;;" \
  bash "$verify" --standalone
check "HTTPRoute list-query failure is distinct from none present" 1 "failed to query HTTPRoute list"

# Workloads serving-route auth: a present workloads-extauth policy that is fail-open
# or points at the wrong handler port leaves inference unauthenticated and must FAIL. The policy
# value is <failOpen>/<backendRef.ns>/<backendRef.name>/<port>.
run_with_stub "  *\"get svc ai-gateway-discovery -n agd\"*) echo 8084 ;;
  *\"get endpoints ai-gateway-discovery\"*) echo \"10.0.0.1\" ;;
${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}
  *\"get securitypolicies.gateway.envoyproxy.io\"*) echo \"proj/workloads-extauth=true/agd/ai-gateway-discovery/8084\" ;;" \
  bash "$verify" --standalone
check "fail-open workloads-extauth policy fails" 1 "not fail-closed on port 8084"

# A fail-closed policy on port 8084 whose backendRef Service exposes 8084 and has ready
# endpoints is healthy.
run_with_stub "  *\"get svc ai-gateway-discovery -n agd\"*) echo 8084 ;;
  *\"get endpoints ai-gateway-discovery -n agd\"*) echo \"10.0.0.1\" ;;
${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}
  *\"get securitypolicies.gateway.envoyproxy.io\"*) echo \"proj/workloads-extauth=false/agd/ai-gateway-discovery/8084\" ;;" \
  bash "$verify" --standalone
check "fail-closed workloads-extauth policy passes" 0 "workloads serving-route auth fail-closed"
check "policy backend port+endpoints ready reported healthy" 0 "exposes port 8084 with ready endpoints"

# A custom workloads authz port (controller.workloadsAuthzPort override) is healthy when passed
# via --workloads-authz-port, and is falsely flagged as not-fail-closed without it.
run_with_stub "  *\"get svc ai-gateway-discovery -n agd\"*) echo 9090 ;;
  *\"get endpoints ai-gateway-discovery -n agd\"*) echo \"10.0.0.1\" ;;
${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}
  *\"get securitypolicies.gateway.envoyproxy.io\"*) echo \"proj/workloads-extauth=false/agd/ai-gateway-discovery/9090\" ;;" \
  bash "$verify" --standalone --workloads-authz-port 9090
check "custom workloads authz port passes when declared" 0 "exposes port 9090 with ready endpoints"

run_with_stub "  *\"get svc ai-gateway-discovery -n agd\"*) echo 9090 ;;
  *\"get endpoints ai-gateway-discovery -n agd\"*) echo \"10.0.0.1\" ;;
${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}
  *\"get securitypolicies.gateway.envoyproxy.io\"*) echo \"proj/workloads-extauth=false/agd/ai-gateway-discovery/9090\" ;;" \
  bash "$verify" --standalone
check "custom port not declared is flagged not-fail-closed on default" 1 "not fail-closed on port 8084"

# A non-numeric --workloads-authz-port is a usage error (exit 64).
run_with_stub "${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}" \
  bash "$verify" --standalone --workloads-authz-port abc
check "non-numeric workloads-authz-port is a usage error" 64 "requires a numeric port"

# IPv6-only ready endpoints are valid even if the address contains no decimal digit.
run_with_stub "  *\"get svc ai-gateway-discovery -n agd\"*) echo 8084 ;;
  *\"get endpoints ai-gateway-discovery -n agd\"*) echo \"fdae::beef\" ;;
${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}
  *\"get securitypolicies.gateway.envoyproxy.io\"*) echo \"proj/workloads-extauth=false/agd/ai-gateway-discovery/8084\" ;;" \
  bash "$verify" --standalone
check "IPv6-only backend ready endpoint passes" 0 "exposes port 8084 with ready endpoints"

# The policy's OWN backendRef Service has no ready endpoints (authz down) — every
# serving request fails closed (denied), a real outage, must FAIL.
run_with_stub "  *\"get svc ai-gateway-discovery -n agd\"*) echo 8084 ;;
  *\"get endpoints ai-gateway-discovery -n agd\"*) echo \"\" ;;
${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}
  *\"get securitypolicies.gateway.envoyproxy.io\"*) echo \"proj/workloads-extauth=false/agd/ai-gateway-discovery/8084\" ;;" \
  bash "$verify" --standalone
check "policy backend Service with no ready endpoints fails" 1 "has no ready endpoints"

# The backend Service must expose the policy's backendRef port: ready addresses on some other
# port don't help, because Envoy resolves the backendRef by that Service port.
run_with_stub "  *\"get svc ai-gateway-discovery -n agd\"*) echo 9999 ;;
  *\"get endpoints ai-gateway-discovery -n agd\"*) echo \"10.0.0.1\" ;;
${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}
  *\"get securitypolicies.gateway.envoyproxy.io\"*) echo \"proj/workloads-extauth=false/agd/ai-gateway-discovery/8084\" ;;" \
  bash "$verify" --standalone
check "ready endpoints but Service missing backendRef port fails" 1 "does not expose port 8084"

# The backend check must correlate to the policy's OWN backendRef, not any authz Service
# elsewhere: an unrelated Service being up must NOT rescue a down real backend.
run_with_stub "  *\"get svc ai-gateway-discovery -n agd\"*) echo 8084 ;;
  *\"get endpoints ai-gateway-discovery -n agd\"*) echo \"\" ;;
  *\"get endpoints something-else\"*) echo \"10.0.0.9\" ;;
${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}
  *\"get securitypolicies.gateway.envoyproxy.io\"*) echo \"proj/workloads-extauth=false/agd/ai-gateway-discovery/8084\" ;;" \
  bash "$verify" --standalone
check "unrelated authz endpoints do not satisfy the policy's backendRef" 1 "ai-gateway-discovery has no ready endpoints"

# Before any AIM serving route exists there is no workloads-extauth policy; absence is a SKIP,
# not a failure, so a fresh workbench still reports HEALTHY.
run_with_stub "${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}
  *\"get securitypolicies.gateway.envoyproxy.io\"*) echo \"\" ;;" \
  bash "$verify" --standalone
check "absent workloads-extauth policy is a SKIP, not a failure" 0 "no workloads-extauth SecurityPolicy yet"

# A partial policy line (backendRef absent mid-reconcile) must FAIL the fail-closed check and
# NOT be parsed into a garbage Service query — no endpoints/service call for a bogus name.
run_with_stub "  *\"get endpoints\"*) echo \"SHOULD-NOT-BE-CALLED\" ;;
  *\"get svc\"*) echo \"SHOULD-NOT-BE-CALLED\" ;;
${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}
  *\"get securitypolicies.gateway.envoyproxy.io\"*) echo \"proj/workloads-extauth=false\" ;;" \
  bash "$verify" --standalone
check "malformed workloads-extauth line fails cleanly" 1 "not fail-closed on port 8084"
check_count "malformed policy line is not parsed into a Service query" "SHOULD-NOT-BE-CALLED" 0

run_with_stub "${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}
  *\"get rabbitmqclusters\"*AllReplicasReady*) echo \"rmq=False\" ;;
  *\"get rabbitmqclusters\"*\"metadata.name\"*) echo \"rmq\" ;;
  *\"get job\"*\"status.succeeded\"*) echo 1 ;;" \
  bash "$verify"
check "RabbitMQ present but not ready fails" 1 "not ready"

# RabbitMQ list-query failure must report a query error, not "no RabbitMQ cluster".
run_with_stub "${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}
  *\"get rabbitmqclusters\"*\"metadata.name\"*) exit 1 ;;
  *\"get rabbitmqclusters\"*) exit 0 ;;
  *\"get job\"*\"status.succeeded\"*) echo 1 ;;" \
  bash "$verify"
check "RabbitMQ list-query failure is distinct from none present" 1 "failed to query RabbitMQ cluster list"

run_with_stub "${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}${HEALTHY_RABBIT}
  *\"get job\"*\"status.succeeded\"*) echo \"\" ;;" \
  bash "$verify"
check "unfinished airm-configure Job fails" 1 "has not succeeded"

# Onboarding Job status-query failure must report a query error, not "has not succeeded".
run_with_stub "${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}${HEALTHY_RABBIT}
  *\"get job\"*\"status.succeeded\"*) exit 1 ;;" \
  bash "$verify"
check "onboarding Job status-query failure is distinct from not-succeeded" 1 "failed to query onboarding Job"

run_with_stub "${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}${HEALTHY_RABBIT}
  *\"get job\"*\"status.succeeded\"*) echo 1 ;;" \
  bash "$verify"
check "fully healthy combined install passes" 0 "RESULT: HEALTHY"

# An RBAC/transient failure on the onboarding Job existence check must NOT be read as "Job
# absent" and skipped: that would drop the check and still allow a HEALTHY verdict.
run_with_stub "${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}${HEALTHY_RABBIT}
  *\"get job\"*\"status.succeeded\"*) echo 1 ;;
  *\"get job\"*) echo 'Error from server (Forbidden): jobs is forbidden' >&2; exit 1 ;;" \
  bash "$verify"
check "onboarding Job query failure fails instead of skipping" 1 "cannot query onboarding Job"
check_count "onboarding Job query failure does not report HEALTHY" "RESULT: HEALTHY" 0

# A genuine NotFound is the expected includeDemoSetup=false case and still skips cleanly.
run_with_stub "${READY_BASE}${HEALTHY_CNPG}${HEALTHY_ROUTES}${HEALTHY_RABBIT}
  *\"get job\"*) echo 'Error from server (NotFound): jobs.batch \"airm-configure\" not found' >&2; exit 1 ;;" \
  bash "$verify"
check "absent onboarding Job still skips and passes" 0 "SKIP  no airm-configure Job"

# In combined mode a missing CNPG in the airm namespace must name the airm-cnpg chart,
# not hardcode aiwb-cnpg (cnpg_healthy runs for both namespaces).
run_with_stub "${READY_BASE}${HEALTHY_ROUTES}${HEALTHY_RABBIT}
  *\"get clusters.postgresql.cnpg.io -n airm\"*) exit 0 ;;
  *\"get clusters.postgresql.cnpg.io\"*\"status.phase\"*) echo \"db=Cluster in healthy state\" ;;
  *\"get clusters.postgresql.cnpg.io\"*) echo \"db\" ;;
  *\"get job\"*\"status.succeeded\"*) echo 1 ;;" \
  bash "$verify"
check "missing airm CNPG names the airm-cnpg chart" 1 "its CNPG infra chart (airm-cnpg by convention) was never installed"

# A Deployment that omits .spec.replicas means 1 replica in Kubernetes; with 1 ready it
# must read as ready, not as 0/0 not-ready.
run_with_stub "
  *\"cluster-info\") exit 0 ;;
  *\"get deploy\"*\"{.spec.replicas}\"*) echo \"\" ;;
  *\"get deploy\"*\"{.status.readyReplicas}\"*) echo 1 ;;
  *\"get deploy\"*) exit 0 ;;
  ${HEALTHY_CNPG}${HEALTHY_ROUTES}${HEALTHY_RABBIT}
  *\"get job\"*\"status.succeeded\"*) echo 1 ;;" \
  bash "$verify"
check "Deployment with unset spec.replicas and 1 ready reads as ready" 0 "RESULT: HEALTHY"

# The Deployment exists, but the replica-status jsonpath query itself errors (RBAC/transient).
# That must FAIL as "cannot confirm", not silently default to 0/1 and read as not-ready.
run_with_stub "
  *\"cluster-info\") exit 0 ;;
  *\"get deploy\"*\"{.spec.replicas}\"*) exit 1 ;;
  *\"get deploy\"*) exit 0 ;;" \
  bash "$verify" --standalone
check "deploy replica-status query failure fails, not false not-ready" 1 "failed to query"

# A Forbidden (RBAC) error on the existence check must NOT be reported as "not found"; surface
# the real cause as a query failure. (The stub writes the error to stderr and exits non-zero.)
run_with_stub "
  *\"cluster-info\") exit 0 ;;
  *\"get deploy\"*\"{.\"*) echo 1 ;;
  *\"get deploy\"*) echo 'Error from server (Forbidden): deployments is forbidden' >&2; exit 1 ;;" \
  bash "$verify" --standalone
check "deploy Forbidden error is a query failure, not not-found" 1 "cannot query Deployment"

# A genuine NotFound is still reported as the chart likely not being installed.
run_with_stub "
  *\"cluster-info\") exit 0 ;;
  *\"get deploy\"*\"{.\"*) echo 1 ;;
  *\"get deploy\"*) echo 'Error from server (NotFound): deployments.apps \"aiwb-api\" not found' >&2; exit 1 ;;" \
  bash "$verify" --standalone
check "deploy NotFound still names the chart-not-installed cause" 1 "not found in namespace"

# Capture the Deployment query's combined output in err so the real stderr classifies the
# failure. The old `2>&1 >/dev/null` ordering leaked stderr and left err empty.
run_with_stub "
  *\"cluster-info\") exit 0 ;;
  *\"get deploy\"*\"{.\"*) echo 1 ;;
  *\"get deploy\"*) echo 'Error from server (Forbidden): deployments is forbidden' >&2; exit 1 ;;" \
  bash "$verify" --standalone
check "deploy query captures the real kubectl error" 1 "deployments is forbidden"
check_count "deploy query does not replace captured stderr with generic fallback" "kubectl error" 0

# The primary aiwb-api readiness signal must fail when ready replicas are below desired.
run_with_stub "
  *\"cluster-info\") exit 0 ;;
  *\"get deploy\"*\"{.spec.replicas}\"*) echo 2 ;;
  *\"get deploy\"*\"{.status.readyReplicas}\"*) echo 1 ;;
  *\"get deploy\"*) exit 0 ;;
  ${HEALTHY_CNPG}${HEALTHY_ROUTES}" \
  bash "$verify" --standalone
check "under-replicated Deployment fails the primary readiness signal" 1 "aiwb-api not ready (1/2)"

run_with_stub '  *"cluster-info") exit 1 ;;' bash "$verify"
check "unreachable cluster is distinct from unhealthy" 2 "cluster unreachable"

# Missing kubectl is a tooling error (exit 1), not "cluster unreachable" (exit 2).
if run_without_kubectl bash "$verify"; then
  check "verify: missing kubectl is a tooling error, not unreachable" 1 "kubectl not found"
else
  printf 'SKIP  verify: missing kubectl (cannot isolate kubectl without dropping coreutils)\n'
fi

echo
echo "== preflight-check.sh =="

PREFLIGHT_BASE='
  *"cluster-info") exit 0 ;;
  *"config current-context") echo ctx ;;
  *"auth can-i create namespaces") echo yes ;;
  *"get storageclass"*) echo default ;;
  *"get crd aimclusterruntimeconfigs.aim.eai.amd.com") exit 0 ;;
  *"get gateway"*) exit 0 ;;
  *"get clustersecretstores.external-secrets.io"*"Ready"*) echo True ;;
  *"get deploy -A"*"API_KEY_AUTH_ANNOTATION"*) echo "envoy-ai-gateway-system/ai-gateway-discovery=True|API_KEY_AUTH_ANNOTATION|WORKLOADS_AUTHZ_PORT" ;;
  *"get ns"*) echo "Error from server (NotFound): namespace not found" >&2; exit 1 ;;'

run_with_stub "${PREFLIGHT_BASE}
  *\"get crd\"*) exit 1 ;;
  *\"get clustersecretstores\"*) exit 1 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;" \
  bash "$preflight"
check "unmet prerequisites report INCOMPLETE, not readiness" 1 "RESULT: INCOMPLETE"

run_with_stub "${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/0\nn2=true/True/0\nn3=/False/0\n' ;;" \
  bash "$preflight"
check "cordoned and NotReady nodes are not counted as schedulable" 1 "only 1 schedulable node(s) of 3"

run_with_stub "${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "fully provisioned cluster reports READY" 0 "RESULT: READY"

# Namespace creation permission is a hard blocker. `kubectl auth can-i` prints "no" AND exits 1
# for a legitimate denial, so the stub must reproduce both or the denial path is never covered.
run_with_stub "  *\"auth can-i create namespaces\") echo no; exit 1 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "missing namespace-create permission is a blocker" 1 "insufficient permissions — cannot create namespaces"
check_count "a denial is not misreported as an API error" "cannot check namespace-create permission" 0

# A failed permission query prints no answer. That is not proof that RBAC denied namespace
# creation, so surface the API query failure rather than claiming a definitive denial.
run_with_stub "  *\"auth can-i create namespaces\") exit 1 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "namespace permission query failure is distinct from denied" 1 "cannot check namespace-create permission"
check_count "a query failure does not claim insufficient permissions" "insufficient permissions" 0

# AIMClusterRuntimeConfig is an unconditional chart dependency. Put this override before
# PREFLIGHT_BASE because bash case uses the first matching arm.
run_with_stub "  *\"get crd aimclusterruntimeconfigs.aim.eai.amd.com\"*) echo 'Error from server (NotFound): customresourcedefinition not found' >&2; exit 1 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "missing AIM Engine CRD is a blocker" 1 "AIM Engine CRDs missing"

# A failed AIM CRD query is not proof that the CRD is missing. It remains a hard blocker, but
# the diagnostic must name RBAC/API access instead of directing the operator to reinstall AIM.
run_with_stub "  *\"get crd aimclusterruntimeconfigs.aim.eai.amd.com\"*) echo 'Error from server (Forbidden): customresourcedefinitions is forbidden' >&2; exit 1 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "AIM CRD query failure is distinct from absent" 1 "cannot query AIM Engine CRD"

# Shared optional-CRD checks must distinguish a genuine NotFound from query failure. The
# failure remains an INCOMPLETE WARN, but should not direct the operator to install a CRD that
# may already exist.
run_with_stub "  *\"get crd clusters.postgresql.cnpg.io\"*) echo 'Error from server (Forbidden): customresourcedefinitions is forbidden' >&2; exit 1 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "optional CRD query failure is distinct from absent" 1 "cannot query CRD 'clusters.postgresql.cnpg.io'"
check_count "optional CRD query failure does not claim CNPG is missing" "CNPG operator CRDs missing" 0

# A Gateway fallback list query failure must be distinct from an empty list. The first
# all-Gateways existence check succeeds, the default Gateway lookup fails, then --no-headers
# fails: report query/RBAC error, not "no Gateway resource found".
run_with_stub "  *\"get gateway.gateway.networking.k8s.io -A --no-headers\"*) exit 1 ;;
  *\"get gateway.gateway.networking.k8s.io https -n envoy-gateway-system\"*) exit 1 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "Gateway list-query failure is distinct from none present" 1 "cannot list Gateway resources"

# Two non-default Gateways must render as a ", "-joined list (not paste's alternating
# single-char delimiters, which would produce "gw1,gw2 gw3").
run_with_stub "  *\"get gateway.gateway.networking.k8s.io https -n envoy-gateway-system\"*) exit 1 ;;
  *\"get gateway.gateway.networking.k8s.io -A --no-headers\"*) printf 'ns1 gwA x\nns2 gwB y\n' ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "non-default Gateways render comma-space joined" 1 "gwA (ns ns1), gwB (ns ns2)"

# A successful list that returns NO Gateways is a hard blocker, not the softer "present but not
# the default" WARN. The join pipeline ends in `END{print ""}`, so it emits a lone newline for an
# empty list; command substitution strips that trailing newline, leaving gw empty and selecting
# the BLOCK. Pin it, because a regression to a WARN here would be waivable.
run_with_stub "  *\"get gateway.gateway.networking.k8s.io https -n envoy-gateway-system\"*) exit 1 ;;
  *\"get gateway.gateway.networking.k8s.io -A --no-headers\"*) exit 0 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "empty Gateway list is a BLOCK, not a WARN" 1 "BLOCK no Gateway resource found"
check_count "empty Gateway list is not reported as a present-but-non-default WARN" "Gateway present but not the default" 0

# A StorageClass list query that errors (RBAC/transient) must not be reported as the specific
# "no StorageClass found" blocker — distinguish "cannot list" from a genuinely empty list.
run_with_stub "  *\"get storageclass\"*) exit 1 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "StorageClass query failure is distinct from absent" 1 "cannot list StorageClasses"

# StorageClass display names are derived from the already-successful `-o name` result; a
# second jsonpath query is unnecessary and must not be called (it could fail after the first
# query succeeded and previously produced a false empty OK).
run_with_stub "  *\"get storageclass\"*\"jsonpath\"*) exit 1 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "StorageClass names reuse the successful list query" 0 "StorageClass(es) present: default"

# A node-list query failure must surface "cannot list nodes", not just a vague count warning.
run_with_stub "${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) exit 1 ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "node-list query failure surfaces the real cause" 1 "cannot list nodes"
check_count "node-list failure does not add the generic count warning" "could not determine node count" 0
check_count "node-list failure emits exactly one warning for one root cause" "WARN  " 1
check_count "node-list failure is not misreported as zero GPUs" "0 allocatable amd.com/gpu" 0

# GPU capacity must count only nodes a workload can actually land on. A cordoned or NotReady
# node keeps its last reported allocatable value, so summing every node would advertise
# capacity nothing can schedule onto — a false-healthy claim.
run_with_stub "${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=true/True/8\nn2=/False/8\nn3=/True/0\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "GPUs on cordoned or NotReady nodes are not counted as available" 1 "0 allocatable amd.com/gpu on schedulable nodes"
check "stranded GPU capacity is reported separately" 1 "16 allocatable amd.com/gpu sit on cordoned or NotReady nodes"

# The healthy case still sums correctly across usable nodes only.
run_with_stub "${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=false/True/8\nn3=true/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "usable-node GPUs are summed and cordoned capacity excluded" 1 "allocatable AMD GPUs on schedulable nodes: 16"

# A re-install (target namespaces already exist) is expected and must NOT gate READY: the
# pre-existing namespace is an informational NOTE, not a scoring WARN. Overrides the base's
# 'get ns -> absent' arm (case matches top-down, so this wins).
run_with_stub "  *\"get ns\"*) exit 0 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "re-install with existing namespaces still reports READY" 0 "RESULT: READY"
check "existing namespace is an informational NOTE, not a WARN" 0 "NOTE  default namespace 'aiwb' already exists"

# A failed namespace query is not proof that the namespace is absent. It must keep preflight
# INCOMPLETE and avoid an affirmative OK claim.
run_with_stub "  *\"get ns airm\"*) echo 'Error from server (Forbidden): namespaces is forbidden' >&2; exit 1 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "namespace query failure is distinct from absent" 1 "cannot query default namespace 'airm'"
check_count "namespace query failure does not claim absence" "OK    default namespace 'airm' absent" 0

# An uncordoned node has .spec.unschedulable=false (not absent); it is still schedulable
# and must be counted, or preflight would flag a healthy cluster as INCOMPLETE.
run_with_stub "${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=false/True/8\nn2=false/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "uncordoned nodes (unschedulable=false) count as schedulable" 0 "schedulable nodes: 3/3"

# API-key store: existence is not health. Missing/unreadable, Ready=False,
# blank Ready, and a failed status query must all remain INCOMPLETE.
run_with_stub "  *\"get clustersecretstores.external-secrets.io\"*\"Ready\"*) exit 1 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "missing or unreadable OpenBao API-key store reports INCOMPLETE" 1 "cannot query OpenBao API-key store"

run_with_stub "  *\"get clustersecretstores.external-secrets.io\"*\"Ready\"*) echo False ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "existing ClusterSecretStore with Ready=False reports INCOMPLETE" 1 "not Ready (status=False)"

run_with_stub "  *\"get clustersecretstores.external-secrets.io\"*\"Ready\"*) echo \"\" ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "ClusterSecretStore with blank Ready status reports INCOMPLETE" 1 "not Ready (status=unset)"

# Security prerequisite: without the Envoy Gateway SecurityPolicy CRD the workloads gateway
# cannot enforce per-model apiKeyAuth, so AIM serving routes would be unauthenticated.
run_with_stub "${PREFLIGHT_BASE}
  *\"get crd securitypolicies.gateway.envoyproxy.io\"*) echo 'Error from server (NotFound): customresourcedefinition not found' >&2; exit 1 ;;
  *\"get crd\"*) exit 0 ;;
  *\"get clustersecretstores\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "missing SecurityPolicy CRD reports INCOMPLETE (unauthenticated serving routes)" 1 "SecurityPolicy CRD missing"

# CRD presence alone proves no enforcement: with no ai-gateway-discovery controller, no one
# creates per-model SecurityPolicies, so preflight must remain INCOMPLETE.
run_with_stub "  *\"get deploy -A\"*\"API_KEY_AUTH_ANNOTATION\"*) echo \"\" ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get clustersecretstores\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "SecurityPolicy CRD without controller cannot report READY" 1 "ai-gateway-discovery controller not ready or missing the WORKLOADS_AUTHZ_PORT contract"

# An older controller carries API_KEY_AUTH_ANNOTATION but not
# WORKLOADS_AUTHZ_PORT, so the workloads serving route is unguarded. Must stay INCOMPLETE.
run_with_stub "  *\"get deploy -A\"*\"API_KEY_AUTH_ANNOTATION\"*) echo \"envoy-ai-gateway-system/ai-gateway-discovery=True|API_KEY_AUTH_ANNOTATION|\" ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get clustersecretstores\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "controller missing WORKLOADS_AUTHZ_PORT reports INCOMPLETE" 1 "missing the WORKLOADS_AUTHZ_PORT contract"

# The OpenBao ClusterSecretStore name is configurable. A valid custom store must pass when
# selected with --secret-store rather than being falsely rejected for not using the default.
run_with_stub "  *\"get clustersecretstores.external-secrets.io custom-store\"*\"Ready\"*) echo True ;;
  *\"get clustersecretstores.external-secrets.io openbao-secret-store\"*) exit 1 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight" --secret-store custom-store
check "custom configured ClusterSecretStore reports READY" 0 "ClusterSecretStore 'custom-store'"

# The ClusterSecretStore has two roles: API-key store (dropped by --no-openbao) and
# bootstrap-secret sync (dropped by --manual-bootstrap-secrets). The verdict must follow the
# two flags exactly, so no reader has to interpret a soft finding.

# --no-openbao alone: Option A still syncs the bootstrap Secrets through the store, so an
# absent store is still an unmet prerequisite and the message must name the remaining role.
run_with_stub "  *\"get clustersecretstores.external-secrets.io\"*\"Ready\"*) exit 1 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight" --no-openbao
check "--no-openbao alone with absent store reports INCOMPLETE" 1 "cannot query bootstrap-secret sync store"
check_count "--no-openbao alone emits no API-key store finding" "OpenBao API-key store" 0

run_with_stub "${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight" --no-openbao
check "--no-openbao alone with a Ready store reports READY" 0 "bootstrap-secret sync store (--no-openbao: no API-key store expected) ready"

# Both flags: the store has no role left. An absent store must not gate READY, and the
# omission must be visible as a SKIP line rather than silent.
run_with_stub "  *\"get clustersecretstores.external-secrets.io\"*\"Ready\"*) exit 1 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight" --no-openbao --manual-bootstrap-secrets
check "both flags with absent store report READY" 0 "RESULT: READY"
check "both flags print the store SKIP line" 0 "SKIP  ClusterSecretStore 'openbao-secret-store' check"
check_count "both flags emit no store WARN" "WARN  cannot query" 0

# --manual-bootstrap-secrets alone: OpenBao is still the API-key store, so the check runs
# unchanged and an absent store stays an unmet prerequisite with the API-key wording.
run_with_stub "  *\"get clustersecretstores.external-secrets.io\"*\"Ready\"*) exit 1 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight" --manual-bootstrap-secrets
check "--manual-bootstrap-secrets alone still requires the API-key store" 1 "cannot query OpenBao API-key store"

# With both flags AIWB itself has no ExternalSecrets, so a missing ESO must not be explained
# with the API-key wording; the CRD stays a WARN because combined mode still needs it.
run_with_stub "  *\"get crd externalsecrets.external-secrets.io\"*) echo 'Error from server (NotFound): customresourcedefinition not found' >&2; exit 1 ;;
${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight" --no-openbao --manual-bootstrap-secrets
check "missing ESO with both flags warns without the API-key wording" 1 "no ExternalSecrets left"
check_count "missing ESO with both flags does not cite API-key ExternalSecrets" "AIWB API-key ExternalSecrets" 0

run_with_stub '  *"cluster-info") exit 1 ;;' bash "$preflight"
check "unreachable cluster stops preflight" 2 "cluster unreachable"

# Every cluster call must be bounded by --request-timeout. A reachable-but-degraded API server
# is the one failure these probes cannot report if a call hangs, and an agent cannot interrupt
# one. The first arm only matches a bounded call; an unwrapped call falls to the second and is
# reported, so dropping the kc() wrapper fails here instead of silently regressing.
run_with_stub '  *"--request-timeout=30s cluster-info") exit 0 ;;
  *"cluster-info") echo "UNBOUNDED-CALL" >&2; exit 1 ;;' bash "$preflight"
check "cluster calls carry the default --request-timeout" 1 "cluster reachable"
check_count "no cluster call is issued unbounded" "UNBOUNDED-CALL" 0

# The timeout is tunable for a large or distant cluster, and the override must actually reach
# kubectl rather than being parsed and dropped.
run_with_stub '  *"--request-timeout=5s cluster-info") exit 0 ;;
  *"cluster-info") echo "WRONG-TIMEOUT" >&2; exit 1 ;;' \
  bash "$preflight" --request-timeout 5s
check "a custom --request-timeout reaches kubectl" 1 "cluster reachable"
check_count "the default is not used when overridden" "WRONG-TIMEOUT" 0

# A malformed duration is an invocation error (64), not a cluster verdict: passing it through
# would surface kubectl's own parse failure as a bogus "cluster unreachable".
run_with_stub '' bash "$preflight" --request-timeout 30
check "preflight: non-duration request timeout is a usage error" 64 "requires a duration like 30s"

run_with_stub '' bash "$verify" --request-timeout abc
check "verify: non-duration request timeout is a usage error" 64 "requires a duration like 30s"

# A unit with no quantity ("sm", "h") is not a Go duration. A character-class glob accepts these
# because every character is individually legal, so validation must require a digit before each
# unit — otherwise kubectl's own parse failure surfaces as a bogus cluster/install verdict.
for bad_duration in sm h ss; do
  run_with_stub '' bash "$preflight" --request-timeout "$bad_duration"
  check "preflight: unit without a quantity ('$bad_duration') is a usage error" 64 "requires a duration like 30s"
  run_with_stub '' bash "$verify" --request-timeout "$bad_duration"
  check "verify: unit without a quantity ('$bad_duration') is a usage error" 64 "requires a duration like 30s"
done

# kubectl reads a zero timeout as "no timeout", which silently restores the unbounded call this
# option exists to prevent — so it is rejected rather than quietly accepted as a valid duration.
run_with_stub '' bash "$preflight" --request-timeout 0s
check "preflight: a zero timeout is rejected, not silently unbounded" 64 "must be greater than zero"

run_with_stub '' bash "$verify" --request-timeout 0s
check "verify: a zero timeout is rejected, not silently unbounded" 64 "must be greater than zero"

# A multi-part duration is valid Go and must not be rejected by an over-strict pattern.
run_with_stub '  *"--request-timeout=1h30m cluster-info") exit 0 ;;
  *"cluster-info") echo "WRONG-TIMEOUT" >&2; exit 1 ;;' \
  bash "$preflight" --request-timeout 1h30m
check "preflight: a multi-part duration is accepted and forwarded" 1 "cluster reachable"
check_count "the multi-part duration reaches kubectl unaltered" "WRONG-TIMEOUT" 0

run_with_stub '  *"--request-timeout=30s cluster-info") exit 0 ;;
  *"cluster-info") echo "UNBOUNDED-CALL" >&2; exit 1 ;;' bash "$verify" --standalone
check_count "verify issues no unbounded cluster call" "UNBOUNDED-CALL" 0

# Helm presence is not enough: a Helm 2 binary, or one whose `helm version` fails, must BLOCK —
# every later step depends on Helm 3.x/4.x behavior. (The whole probe exits non-zero on any BLOCK.)
STUB_HELM_BODY='echo "v2.17.0"' run_with_stub '  *"cluster-info") exit 0 ;;' bash "$preflight"
check "Helm 2 is a blocker, not accepted as present" 1 "Helm 3.x or 4.x required; found 'v2.17.0'"

STUB_HELM_BODY='echo "boom" >&2; exit 1' run_with_stub '  *"cluster-info") exit 0 ;;' bash "$preflight"
check "broken helm version is a blocker" 1 "helm present but 'helm version --short' failed"

# Helm 4 is supported (INSTALL.md advertises 3.x or 4.x), so a v4 binary must be accepted.
STUB_HELM_BODY='echo "v4.2.2+g1a2b3c4"' \
  run_with_stub "${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "Helm 4 is accepted as present" 0 "Helm 3.x or 4.x present"

# A wrapper/plugin banner line before the real version must NOT flip a healthy Helm 3 to a
# block: the v3 token is matched anywhere in the output, not just as the whole-blob prefix.
# Mirror the fully-provisioned READY stub so only the helm banner is under test.
STUB_HELM_BODY='echo "WARNING: plugin hook noise"; echo "v3.16.4+g7877b45"' \
  run_with_stub "${PREFLIGHT_BASE}
  *\"get crd\"*) exit 0 ;;
  *\"get nodes\"*unschedulable*) printf 'n1=/True/8\nn2=/True/8\nn3=/True/8\n' ;;
  *\"get nodes\"*) echo 8 ;;" \
  bash "$preflight"
check "helm banner line before v3 version is still accepted" 0 "Helm 3.x or 4.x present"
check "healthy cluster with a helm banner still reports READY" 0 "RESULT: READY"

# A missing option value is an invocation error, not a cluster-readiness failure. It must use
# sysexits-style 64 rather than bash's ${2:?} exit 1 (which means INCOMPLETE/NOT READY here).
run_with_stub '' bash "$preflight" --secret-store
check "preflight: missing option value is usage error" 64 "--secret-store requires a value"

run_with_stub '' bash "$verify" --namespace
check "verify: missing option value is usage error" 64 "--namespace requires a value"

# A following flag is not a value — do not silently consume it as a namespace/store name.
run_with_stub '' bash "$preflight" --secret-store --bogus
check "preflight: next option is not consumed as a value" 64 "--secret-store requires a value"

run_with_stub '' bash "$verify" --namespace --standalone
check "verify: next option is not consumed as a value" 64 "--namespace requires a value"

# Missing kubectl is a tooling error (exit 1 NOT READY), not "cluster unreachable" (exit 2).
if run_without_kubectl bash "$preflight"; then
  check "preflight: missing kubectl is NOT READY, not unreachable" 1 "kubectl not found"
else
  printf 'SKIP  preflight: missing kubectl (cannot isolate kubectl without dropping coreutils)\n'
fi

echo
echo "== harness self-checks =="

# The assertion helpers are themselves somewhere a bad change can hide, and one hid here —
# see `check` above for the failure they must not be allowed to reacquire. Drive that exact
# shape: a marker matching on the first line, followed by far more output than a pipe will
# hold, so any reader that stops early strands a tail. At the sizes this suite really
# produces the old helper lost the race only rarely, which is what made it read as flaky
# tests; oversizing the payload makes losing it a certainty, so this guard fails every time
# against a piped helper rather than occasionally. Built with shell string operations
# only — no head/tr/fold, which this suite does not otherwise require — and by doubling
# rather than appending, so building it stays cheap.
#
# Only the `check` case is a real guard: it fails against the old piped helper and passes
# against the current one. The `check_count` case is a smoke test over the same input, for
# the reason given at that helper: `grep -c` as invoked there reads to EOF, so it cannot
# exit early today — but `-m` would change that, so do not read this as immunity.
printf -v selfcheck_block '%*s' 79 ''
selfcheck_block="${selfcheck_block// /x}"
while [ "${#selfcheck_block}" -lt 200000 ]; do
  selfcheck_block="${selfcheck_block}"$'\n'"${selfcheck_block}"
done
out="HARNESS-SELFCHECK-MARKER"$'\n'"${selfcheck_block}"
code=0
check "harness: a marker matching early in a large output is not reported missing" 0 "HARNESS-SELFCHECK-MARKER"
check_count "harness: a marker is counted across a large output" "HARNESS-SELFCHECK-MARKER" 1
# These two cases are the only ones in the file that feed the assertion helpers state not
# produced by `run_with_stub`/`run_without_kubectl`. Clear all of it, including the larger
# copy in `out`, so a case appended after this section cannot silently assert against it.
unset selfcheck_block
out=""
code=0

echo
echo "== evals =="

# The evaluation dataset is part of this skill's contract, so validate it here rather than in a
# skill-specific CI step: the repo's agent-skill-tests job discovers only tests/run-tests.sh, so
# anything not invoked from here is never checked in CI.
if eval_out=$(python3 "${here}/validate-evals.py" 2>&1); then
  printf 'PASS  %s\n' "$eval_out"
  pass=$((pass+1))
else
  printf 'FAIL  evaluation dataset invalid\n      %s\n' "$eval_out"
  fail=$((fail+1))
fi

echo
echo "passed=${pass} failed=${fail}"
[ "$fail" -eq 0 ]
