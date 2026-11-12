<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Verification: confirming the workbench is healthy

This is an AI instruction file. Use it to interpret `scripts/verify-install.sh` and decide
whether to report success, "inconclusive / what to fix", or "unreachable". The cardinal
rule: **never report success you did not verify.** A partial install must surface as
inconclusive with the specific failing component named — not as a pass.

## The primary signal

**`aiwb-api` readiness is the primary signal that the workbench is up.** It is the
Deployment named `aiwb-api` (Helm release `aiwb`, so `{release}-api`) in the Helm release
namespace `aiwb` — the namespace you `helm install -n <ns>` into, the same in standalone
and combined mode. Do **not** verify in the standalone `defaultNamespace` (default
`workbench`): that is a separate user-workload namespace the chart creates, not where the
`aiwb-api`/`aiwb-ui` Deployments or HTTPRoutes live — those always render into
`.Release.Namespace`. Its readiness probe is HTTP `GET /v1/health` on port 8080. When its
pods are `Ready`, the API is serving.

If you only check one thing, check that `aiwb-api` has its desired replicas `Ready`. Note
the chart runs a single API replica by design (poller/syncer), so "1/1 ready" is the
healthy state, not a scaling problem.

## Full checklist

- **`aiwb-api`** — desired replicas Ready (primary signal, above).
- **`aiwb-ui`** — Deployment `aiwb-ui`, readiness `GET /api/health`. The user-facing UI.
- **CNPG PostgreSQL clusters** — `clusters.postgresql.cnpg.io` in `aiwb` (and `airm` for
  combined) report a healthy state. The API cannot serve without its database. **If the
  install brings its own PostgreSQL** (INSTALL.md "Bringing Your Own Infrastructure" — the
  CNPG operator and `*-cnpg` charts are intentionally skipped, secrets point at an external
  DB), run verify with `--external-db` so the absent CNPG cluster is skipped, not failed.
  The external DB is still covered indirectly: `aiwb-api` won't reach Ready without it.
- **RabbitMQ** (combined only) — `rabbitmqclusters` in `airm` report `AllReplicasReady`.
  Not present in standalone AIWB.
- **HTTPRoutes** — `httproutes` in `aiwb` (and `airm`) exist and report `Accepted=True`
  from their parent Gateway, so the services are reachable at their external URLs.
- **AIRM** (combined only) — `airm-api` and `airm-ui` Deployments Ready and, if
  `includeDemoSetup` is on, the `airm-configure` Job completed (cluster onboarding, Step 5
  of INSTALL.md).
- **Workloads serving-route auth** — once an AIM model namespace exists,
  ai-gateway-discovery stamps a namespace-wide `workloads-extauth` `SecurityPolicy`. It must be
  fail-closed (`spec.extAuth.failOpen=false`) and route to the dedicated workloads handler port
  (8084). Before the first serving route exists there is no such policy, so its absence is a
  SKIP, not a failure. This guards the workloads route (`workloads.<domain>`) that has no
  gateway default-deny backstop; a fail-open or missing policy leaves inference unauthenticated.
  The ext_authz handler backing that port is a separate HA Deployment — the sole
  fail-closed authenticator on the inference path — reached through the policy's `extAuth`
  `backendRef` Service. When a `workloads-extauth` policy exists, that **exact** backend Service
  (the `backendRef` namespace/name, not any authz Deployment elsewhere) must both expose the
  `backendRef` port (8084) and have ready endpoints. A Service missing that port cannot be
  resolved by Envoy even with ready pods; no ready endpoints means requests fail closed
  (denied). Either way an unavailable authorizer is a real outage, not mere absence.

## Existence is not health — check the condition, not the object

Every check above asserts an observable _state_, never mere presence. This distinction is
what keeps a broken install from reading as a pass:

- A CNPG **CRD** installed with **zero Cluster resources** means the infra chart was never
  installed — an empty list must fail, not vacuously pass a phase scan.
- An HTTPRoute that **exists** but is not `Accepted` by its Gateway leaves the service
  silently unreachable (usually a `gateway.gatewayName` / `gateway.namespace` mismatch).
- A RabbitMQ cluster object that **exists** but has no ready replicas breaks AIRM↔Agent
  messaging.

When you extend these checks, assert the status condition. "The object is there" is not
evidence the component works.

## Reporting: pick the honest outcome

- **Healthy** — `aiwb-api` Ready and every checked component up. Report success and print
  the access URLs (below).
- **Inconclusive / what to fix** — any checked component not ready, or the API never
  becomes Ready within a reasonable wait. Report the specific component that failed and
  the concrete next step (route the user to `references/troubleshooting.md`). The verify
  script exits non-zero in this case so the state is unambiguous.
- **Unreachable** — cannot query the cluster at all. Report that; make no claim about
  install state.

Do not average these into a vague "mostly working." If `aiwb-api` is not Ready, the
workbench is not installed, full stop — say so.

## Access URLs to report on success

With `appDomain=<your-domain>` (from the app-chart install), the services resolve to:

- **AIWB UI**: `https://aiwbui.<your-domain>`
- **AIWB API**: `https://aiwbapi.<your-domain>`
- **AIRM UI** (combined): `https://airmui.<your-domain>`
- **AIRM API** (combined): `https://airmapi.<your-domain>`
- **Keycloak**: `https://kc.<your-domain>`

The subdomain prefixes come from the charts' gateway values (`aiwbapi`, `aiwbui`, `airmui`,
`kc`, …). On a local dev cluster without real DNS/ingress, tell the user to
`kubectl port-forward` to the API/UI services instead of using these URLs.
