<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Preflight: what a "ready" cluster means

This is an AI instruction file. Use it to interpret `scripts/preflight-check.sh` output
and decide whether the cluster is ready to install AIWB. The authoritative list of
components and versions is `helm/INSTALL.md` (Prerequisites + Step 1 Tiers 1–3); this file
tells you which findings block the install and which are conditional on the chosen mode.

## Hard blockers — never install until these pass

- **Cluster reachable.** `kubectl` can talk to the API server and you have permission to
  create namespaces. If not, nothing else matters — stop and fix access
  first (`references/troubleshooting.md` → "cluster unreachable").
- **Helm 3.x or 4.x and kubectl present** on the machine running the install.
- **A usable StorageClass.** The infra charts default to one named `default`. If the
  cluster's is named differently (e.g. `standard` on kind), that is NOT a blocker by
  itself — but you MUST pass the correct per-chart override in Step 3 (see the StorageClass
  gotcha in SKILL.md). No StorageClass at all IS a blocker: CNPG/RabbitMQ PVCs will never
  bind.
- **AIM Engine CRDs**, specifically `AIMClusterRuntimeConfig` (`aim.eai.amd.com/v1alpha1`).
  The AIWB app chart creates this resource, so the CRD must exist before `helm install aiwb`
  or the release fails. This is a blocker for every AIWB install regardless of mode.
- **Gateway.** A Gateway API implementation with a `Gateway` the charts can attach
  HTTPRoutes to. Defaults: a Gateway named `https` in namespace `envoy-gateway-system`.
  If the user's gateway differs, it's not a blocker but they must override
  `gateway.namespace` / `gateway.gatewayName` for AIWB and
  `airm-api.gateway.namespace` / `airm-api.gateway.gatewayName` for the AIRM umbrella.
  A bare AIRM `--set gateway.*` is silently ignored because the umbrella has no top-level
  `gateway` key.

## Mode-dependent checks — blocker only for the relevant mode

- **Kyverno** — required ONLY for standalone AIWB (`standAloneMode=true`), which relies on
  a Kyverno `ClusterPolicy` to auto-label workloads. Not needed for combined AIRM+AIWB
  (AIRM handles labeling). Absence is a blocker only in standalone mode.
- **GPU tier** — required for actual model/workload features on a real GPU cluster. On a
  local dev cluster (kind/minikube/Docker Desktop) with no GPUs, these can be skipped: the
  AIRM and AIWB APIs/UIs still come up, but model deployment won't work. Treat a missing GPU
  tier as a blocker on a GPU cluster and as expected/waivable on local dev.

  Be precise about what the probe establishes, because reporting unverified coverage is the
  failure this skill exists to prevent:
  - **Probed directly**: the Kaiwo and KServe CRDs.
  - **Probed indirectly**: the AMD GPU Operator, only as far as nodes advertising
    `amd.com/gpu` allocatable capacity. Capacity is a proxy for a rolled-out driver, not a
    check of the operator itself.
  - **Not probed — verify by hand against `helm/INSTALL.md`**: the AMD Device Metrics
    Exporter and its required `gpu-config` ConfigMap in `kube-amd-gpu`. INSTALL.md requires
    the exporter on every GPU node, and AIRM's GPU dashboards and workload monitoring depend
    on it. Nothing in the probe looks at it, so never report it as confirmed.

  Note: the AIM Engine _CRDs_ are a separate unconditional hard blocker, checked on their own
  and required even when the GPU tier is waived (see hard blockers).

- **RabbitMQ operator + cert-manager** — required for combined mode (AIRM↔Agent messaging).
  Not required for standalone AIWB.
- **CNPG operator** — required unless the user is bringing their own PostgreSQL (INSTALL.md
  "Bringing Your Own Infrastructure"). If BYO, the `*-cnpg` infra charts are skipped and
  the corresponding secrets must point at the external DB instead.
- **External Secrets Operator** — Option B can replace ESO only for the bootstrap Secrets
  listed in INSTALL.md; it does **not** remove the runtime ESO/OpenBao requirement for API
  keys. AIWB API-key creation creates `ExternalSecret` objects, and ESO materializes the
  per-key gateway credential Secrets. A Ready configured `ClusterSecretStore` is therefore
  required in both bootstrap-secret options, unless the install disables the API-key store
  (`openBao.enabled: false`, next bullet). Tell preflight about Option B with
  `--manual-bootstrap-secrets`; alone it changes nothing, because the store is still the
  API-key store.
- **Keycloak** — required for every real deployment and **not substitutable**: AIRM calls
  the Keycloak Admin API at runtime for user/group/role management, so a generic OIDC
  provider will not do. Also needs its realm configured (INSTALL.md "Keycloak Realm Setup").
- **MinIO / S3-compatible storage** — required for AIWB dataset and model storage. Waivable
  only when substituting your own S3 (AWS S3, GCS, Azure Blob) and pointing the chart at it.
- **OpenBao/ESO API-key store** — the configured `ClusterSecretStore` is the sole API-key
  path; cluster-auth was removed. The chart default is `openbao-secret-store`,
  but both `openBao.secretStoreRef` (AIWB app chart) and `externalSecretStore`
  (external-secrets chart) are configurable. Pass the selected name to preflight with
  `--secret-store <name>`; otherwise a valid custom store would be falsely reported absent.
  The store is optional: an installation without OpenBao sets `openBao.enabled: false`
  (AIWB app chart) and `openbaoToken.enabled: false` (external-secrets chart), and runs
  preflight with `--no-openbao`. The API then refuses to create, rebind or delete API keys
  and the UI hides the page. The store check follows the two Step 1 choices: with
  `--no-openbao` alone the store is still required for bootstrap-secret sync (Option A), so
  an absent or not-Ready store WARNs with that wording; with `--no-openbao` and
  `--manual-bootstrap-secrets` the store has no role left and the check is skipped with a
  visible `SKIP` line.
- **Inference-route auth prerequisites** — the Envoy Gateway `SecurityPolicy` CRD and a ready
  ai-gateway-discovery controller must exist so both auth paths can be reconciled: per-model
  `apiKeyAuth` on AI-gateway routes, and the namespace-wide fail-closed `workloads-extauth`
  policy on workloads serving routes. Preflight requires the controller to expose
  the `WORKLOADS_AUTHZ_PORT` env contract — an older controller can create per-model policies
  but leaves the workloads route unguarded. This proves capability, not actual traffic denial:
  before a model namespace/serving route exists there is no policy and nothing meaningful to
  request. After the first AIM route is deployed, verify (`scripts/verify-install.sh`) that the
  `workloads-extauth` policy is fail-closed and that an unauthenticated request is rejected; do
  not report serving-route security from preflight alone.
- **OpenTelemetry Operator, KEDA, Prometheus Operator CRDs** — Tier 1 dependencies: the
  vLLM metrics collector, KServe autoscaling, and the ServiceMonitor/PodMonitor CRDs the
  observability stack needs. Skippable on a local dev cluster that isn't serving models.
- **Kaiwo and KServe** — Tier 3 workload orchestration and model serving. Same rule as the
  rest of the GPU tier: required on a real GPU cluster, waivable on local dev.

## Only "READY" is a green light

The probe exits 0 only when every checked prerequisite is present. A WARN means a
prerequisite is **unconfirmed**, and the probe exits non-zero with `INCOMPLETE` — treat
that as "not cleared to install," not as a pass with advisory notes. Resolve each WARN, or
explicitly waive it as not-needed for the chosen mode and say which ones you waived and
why. Installing past an unmet prerequisite is what produces a partial install.

The probe covers what is machine-checkable from the cluster. It does not verify Keycloak
realm contents, secret-backend keys, or GPU driver rollout depth — confirm those against
`helm/INSTALL.md` before declaring the cluster ready.

## GPU capacity

On a GPU cluster, confirm nodes actually advertise `amd.com/gpu` allocatable capacity, not
just that the operator is installed — an operator that hasn't rolled out drivers yet shows
zero allocatable GPUs, which will strand GPU workloads later. Report the per-node
allocatable GPU count so the user can sanity-check it against what they expect.

## Namespaces

The install uses the Helm release namespace `aiwb` (and `airm` for combined) — the
namespace you `helm install -n <ns>` into, the same in standalone and combined mode. In
standalone mode the chart additionally creates a separate user-workload namespace named by
`defaultNamespace` (default `workbench`), but the app components (Deployments, HTTPRoutes)
stay in the release namespace, so that is where preflight probes. Pre-existing namespaces
are fine (`--create-namespace` is idempotent) — the probe reports them as an informational
NOTE (not a WARN, so a re-install is never gated), so you know whether this is a fresh
install or a re-install.

## Deciding to proceed

Proceed to Step 3 only when: cluster reachable, tooling present, a StorageClass is usable
(with the right override planned), AIM Engine CRDs present, a Gateway available, and every
mode-relevant dependency either present or explicitly waived as not-needed for this mode.
If anything hard is missing, list each gap with the specific thing to install (point at the
matching INSTALL.md row) and wait — do not start installing.
