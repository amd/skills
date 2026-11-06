---
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
name: guided-workbench-install
description: Guides a user through installing the AI Workbench (AIWB) onto a Kubernetes cluster end-to-end — preflight cluster readiness (access, StorageClass, GPU operator, required operators/CRDs), install the Helm charts in the correct order, and verify the deployment is healthy with plain-language failure triage. Use when a user wants to install, deploy, or stand up AIWB / the AI Workbench / the enterprise AI stack on a K8s cluster, or asks why their install failed partway. Use PROACTIVELY when a user is following helm/INSTALL.md by hand. Do NOT activate for onboarding models to AIM, general kubectl/helm usage unrelated to installing AIWB, or authoring/running the repo's Robot Framework E2E tests. For an established cluster that is not mid-install — hung model downloads, missing GPU metrics, or a pre-install or pre-E2E health sweep — use cluster-environment-check instead; this skill owns the install itself, including verifying it and triaging a component that never becomes ready during it.
allowed-tools: Bash, Read, Grep, Glob
---

# Guided AI Workbench (AIWB) Install

This is an AI instruction file. Follow it to guide a user from a bare Kubernetes cluster
to a working, verified AI Workbench install — without assuming they have platform
expertise.

Your job is orchestration and interpretation, not re-documentation. The authoritative
command-by-command runbook is **`helm/INSTALL.md`** in this repo. You add the layer it
lacks: gate the install on preflight, verify health after each layer, and translate
failures into plain language and a concrete fix. **Never report success you did not
verify, and never leave a silent partial install** — when you cannot confirm health, say
so explicitly and name what to fix.

The sibling `cluster-environment-check` skill diagnoses an already-installed cluster and is
read-only; this skill owns installing. They assert some of the same facts (StorageClass, the
OpenBao `ClusterSecretStore`, `amd.com/gpu` capacity), so when a chart or default changes,
check whether both need the correction.

## Golden rule: preflight → install in order → verify → triage

Do these in order. Do not start installing before preflight passes. Do not declare the
install done before verification passes.

## Step 0 — Locate the charts and the runbook

The charts and runbook live in this repo under `helm/`. Confirm you can see them before
anything else:

```bash
ls helm/INSTALL.md helm/aiwb/Chart.yaml helm/airm/Chart.yaml
```

If `helm/` is not present, you are not in the repository root — ask the user for the path
to their checkout of `github.com/amd-enterprise-ai/amd-eai-apps` and `cd` there. Two distinct working
directories: run this skill's probe scripts (`.agents/skills/guided-workbench-install/...`)
from the repo root, but run the Helm install commands from inside `helm/`, because the
chart paths `./aiwb`, `./airm`, `./infrastructure/...` (and INSTALL.md's commands) are
relative to `helm/`. `cd helm` before the first `helm install`.

Read `helm/INSTALL.md` now — it is the source of truth for exact chart names, values, and
secret keys. This skill deliberately does not copy those command strings, because they would
drift from the charts.

**Legacy cluster-auth chart leftover.** The AIWB **app** chart (`helm/aiwb/`) no longer
consumes cluster-auth and no longer applies its own access control to AIM serving routes. The
API-key store is the OpenBao/ESO `ClusterSecretStore` (default `openbao-secret-store`, set via
`openBao.secretStoreRef`). Gateway-side enforcement now has two separate paths:
ai-gateway-discovery creates per-model `apiKeyAuth` policies for AI-gateway routes, and a
namespace-wide, fail-closed `workloads-extauth` `SecurityPolicy` for workloads serving routes
(which have no default-deny backstop). See `references/preflight.md` and
`references/verification.md`. Do not treat cluster-auth as an app-level prerequisite or expect
the app chart to use it.

One live leftover remains — do not tell the operator it is unused: the ESO chart
`helm/infrastructure/aiwb-external-secrets` still declares an **ungated** `ExternalSecret` named
`cluster-auth-admin-token` (it renders on the Option A / synced-secrets path, outside the
`standAloneMode` guard). On Option A, either keep the `cluster-auth-admin-token` key provisioned
in the backend, or expect that single `ExternalSecret` to stay un-synced — a known chart
leftover, not a failed install. Flag this so the operator is not blocked at INSTALL.md's "wait
until all ExternalSecrets show `SecretSynced`" step. On Option B (manual secrets) the
`cluster-auth-admin-token` secret is not needed. Follow the charts where they and INSTALL.md
disagree.

## Step 1 — Establish what the user is installing

Ask (or infer from what the user already said) four things, because they change every
later step:

1. **Deployment mode** — one of:
   - **Combined (AIRM + AIWB)** — full platform. The default.
   - **Standalone AIWB** — single-namespace workbench, no AIRM. Sets `standAloneMode=true`
     and additionally requires **Kyverno**.
   - **Standalone AIRM** — resource manager only; not a workbench install, so redirect if
     the user actually wants the workbench.
2. **Cluster class** — real GPU cluster vs. local dev cluster (kind/minikube/Docker
   Desktop). On local dev clusters the GPU tier (AMD GPU Operator, Device Metrics
   Exporter, Kaiwo, AIM Engine) can be skipped and the StorageClass is usually `standard`,
   not `default`. See "Local Development Notes" in `helm/INSTALL.md`.
3. **Database mode** — one of:
   - **Chart-managed CNPG** — install the CNPG operator and `*-cnpg` infrastructure charts.
     The default.
   - **External PostgreSQL (BYO)** — skip CNPG and its infrastructure charts; create the
     required DBs/secrets and set application-chart database host/port values per
     INSTALL.md "Bringing Your Own Infrastructure." Remember this choice through Step 4:
     verification must use `--external-db` or the intentional lack of CNPG will look like
     a failed install.
4. **Bootstrap-secrets method + API-key store** — Option A syncs bootstrap secrets through
   ESO; Option B creates those bootstrap Secrets manually. **Option B does not remove the
   runtime ESO/OpenBao requirement for API keys:** AIWB API-key creation makes
   `ExternalSecret` objects, and ESO materializes the gateway credential Secrets.
   In both options require a Ready `ClusterSecretStore`, configured as
   `openBao.secretStoreRef` (AIWB app chart; default `openbao-secret-store`). When using
   the external-secrets chart, set its `externalSecretStore` to the same name. Carry a
   custom name through preflight (`--secret-store`) and those chart values. An installation
   that does not need API keys can set `openBao.enabled: false` (AIWB app chart) and
   `openbaoToken.enabled: false` (aiwb-external-secrets chart) instead. The UI then hides
   the API-key page, the API refuses to create, rebind or delete keys, and no
   `aiwb-openbao-token` Secret is needed. A Ready `ClusterSecretStore` is then required only
   for Option A, which still syncs the other bootstrap Secrets through it; Option B with
   OpenBao disabled needs no `ClusterSecretStore` at all. Carry both choices into preflight:
   `--no-openbao` for the disabled store, and `--manual-bootstrap-secrets` for Option B. The
   store check follows the two flags, so it never asks you to interpret a soft finding.

State back to the user which deployment mode + cluster class + database mode + bootstrap-
secrets method/API-key store you're proceeding with before you start.

## Step 2 — Preflight the cluster (gate)

**Do not install anything until preflight passes.** Run the read-only probe:

```bash
# Default ClusterSecretStore (openbao-secret-store)
bash .agents/skills/guided-workbench-install/scripts/preflight-check.sh

# Custom ClusterSecretStore selected in Step 1
bash .agents/skills/guided-workbench-install/scripts/preflight-check.sh --secret-store <name>

# openBao.enabled: false selected in Step 1 (no API-key store; Option A still syncs
# the bootstrap Secrets through the ClusterSecretStore)
bash .agents/skills/guided-workbench-install/scripts/preflight-check.sh --no-openbao

# openBao.enabled: false and Option B (no ClusterSecretStore role left, check skipped)
bash .agents/skills/guided-workbench-install/scripts/preflight-check.sh --no-openbao --manual-bootstrap-secrets
```

It checks cluster reachability, StorageClass presence, the Tier 1–3 operators/CRDs from
`helm/INSTALL.md`, GPU capacity, schedulable node count against the CNPG instance defaults,
the configured API-key store, the SecurityPolicy CRD and ai-gateway-discovery controller,
and the target namespaces. It never mutates the cluster.

**Scope boundary:** preflight proves the components needed to reconcile both inference-auth
paths — per-model `apiKeyAuth` (AI-gateway routes) and the namespace-wide fail-closed
`workloads-extauth` policy (workloads serving routes) — exist; it cannot prove
traffic denial before a model/serving route exists. A 404 on an arbitrary pre-install URL
proves nothing. Defer the traffic assertion until the first AIM route exists; then Step 4's
probe checks the `workloads-extauth` policy is fail-closed, and you should also verify an
unauthenticated request is rejected before reporting that serving-route security works.

Read `references/preflight.md` to interpret the output, understand what each check means,
and decide which findings are hard blockers vs. mode-dependent (e.g. GPU tier is optional
on local dev clusters; Kyverno only matters for standalone).

Gate — the probe's exit code is the contract, and only `READY` is a green light. The gate is
enforced by you, not by the harness: nothing stops a later `helm install` from running after a
non-`READY` probe, and nothing audits a recorded waiver, so treat these as rules you are
responsible for keeping rather than guardrails that will catch you.

- **`READY` (exit 0)** — every checked prerequisite present. Proceed to Step 3.
- **`INCOMPLETE` (exit 1, WARN items)** — no hard blocker, but prerequisites are
  unconfirmed. **This is not permission to install.** Name each WARN, say what installs it
  (point at the matching `helm/INSTALL.md` Tier 1–3 row), and continue only once each is
  either resolved or explicitly waived as not-needed for the chosen mode — record which
  ones you waived and why.
- **`NOT READY` (exit 1, BLOCK items)** — a hard blocker. Resolve it before installing.
- **Unreachable (exit 2)** — **stop.** Report it and help fix access (see
  `references/troubleshooting.md`); do not run install commands.
- **Usage error (exit 64)** — the probe invocation is invalid (unknown option or missing
  option value). Fix the command; do not diagnose this as a cluster-readiness failure.

Installing past an unmet prerequisite is exactly what produces the partial install this
skill exists to prevent — so never wave a WARN through silently.

## Step 3 — Install in the correct order

Follow `helm/INSTALL.md` for the exact commands. The required order is:

1. **Cluster dependencies** (Step 1 of INSTALL.md) — operators/CRDs must exist first.
   AIWB creates an `AIMClusterRuntimeConfig` (`aim.eai.amd.com/v1alpha1`), so **AIM Engine
   CRDs must be installed before the AIWB app chart** or the install fails.
2. **Bootstrap secrets** (Step 2) — sync through ESO (Option A) or create manually
   (Option B). With OpenBao enabled (the default), ESO/OpenBao remains required at runtime
   for AIWB API-key creation in either option; Option B only replaces bootstrap-secret
   synchronization, not the API-key store. When Step 1 chose `openBao.enabled: false`,
   there is no API-key store to require, and Option B then needs no `ClusterSecretStore`.
3. **Infrastructure charts** (Step 3) — install CNPG PostgreSQL only for chart-managed
   CNPG mode; skip the operator and `*-cnpg` charts for external PostgreSQL and create the
   external DB secrets/values instead. Install RabbitMQ where required. Wait for the chosen
   infrastructure to be healthy before the app charts.
4. **Application charts** (Step 4) — `airm` (combined mode) then `aiwb`.
5. **Onboard the cluster** (Step 5) — combined mode only.

**Custom secret-store gotcha — configure every installed chart consistently.** If Step 1
selected a store other than `openbao-secret-store`, always pass it as
`openBao.secretStoreRef` to the AIWB app chart; when installing the external-secrets chart,
also pass it as `externalSecretStore`. Using the custom name only in preflight does not
configure the installation.

**StorageClass gotcha — the override key differs per chart.** A single
`--set storage.storageClass=` does NOT cover everything:

- `airm-cnpg` / `aiwb-cnpg`: `--set storage.storageClass=<NAME> --set walStorage.storageClass=<NAME>`
- `airm-rabbitmq`: `--set persistence.storageClassName=<NAME>`

If the cluster's StorageClass is not named `default`, pass the right key for each infra
chart or the PVCs stay `Pending`.

**Single-node / local dev gotcha — `aiwb-cnpg` defaults to 3 PostgreSQL instances**
(chart-managed CNPG mode only). The CNPG charts spread instances across nodes (`airm-cnpg`
defaults to 1, `aiwb-cnpg` to 3). On a one-node cluster all three co-schedule: three times
the resources for no added resilience, and a `PodDisruptionBudget` is created once there
are two or more instances, which blocks `kubectl drain` on that node. Install with
`--set instances=1` on single-node clusters. See "Node Requirements for CNPG Clusters" in
`helm/INSTALL.md` for the full set of overrides (`enablePDB`, `podAntiAffinityType`,
`topologyKey`).

**After each layer, verify before moving on.** Don't stack the app chart on top of an
infra chart whose database never came up — that produces exactly the silent partial
install this skill exists to prevent.

## Step 4 — Verify the deployment is healthy (gate)

Step 3 left you inside `helm/`, but these paths are repo-root-relative — `cd` back to the repo
root first or the probe is "No such file or directory".

Run the verification probe with the same deployment and database modes chosen in Step 1:

```bash
# Combined + chart-managed CNPG (defaults)
bash .agents/skills/guided-workbench-install/scripts/verify-install.sh

# Standalone AIWB + chart-managed CNPG
bash .agents/skills/guided-workbench-install/scripts/verify-install.sh --standalone

# External PostgreSQL (add --standalone too when both choices apply)
bash .agents/skills/guided-workbench-install/scripts/verify-install.sh --external-db
bash .agents/skills/guided-workbench-install/scripts/verify-install.sh --standalone --external-db
```

Never omit `--external-db` for the BYO-PostgreSQL path: that deployment intentionally has
no CNPG objects, and default verification correctly treats a missing chart-managed DB as a
failure. The readiness of **`aiwb-api`** (Deployment `aiwb-api`, readiness endpoint
`/v1/health` on port 8080) is the primary signal that the workbench is up. The script also
checks `aiwb-ui`, a healthy CNPG cluster in chart-managed mode (explicitly skipped for
external DB), and HTTPRoutes `Accepted` by their Gateway. In combined mode it additionally
checks `airm-api` / `airm-ui` readiness, RabbitMQ all replicas ready, and the
`airm-configure` onboarding Job completed.

If the install overrode the ai-gateway-discovery chart's `controller.workloadsAuthzPort`
(default 8084), pass the same value to verify with `--workloads-authz-port <port>`; otherwise a
healthy `workloads-extauth` policy on the custom port is falsely reported not-fail-closed.

Read `references/verification.md` for what each check confirms and the service URLs to
report.

Gate — three outcomes, and you must pick the honest one:

- **Healthy** — `aiwb-api` ready and all checked components up. Report success and print
  the access URLs (`https://aiwbapi.<domain>`, `https://aiwbui.<domain>`, etc.).
- **Inconclusive / what to fix** — some component is not ready. **Do not report success.**
  Name the specific failing component and the concrete next check/fix (route the user
  through `references/troubleshooting.md`). The verify script exits non-zero here.
- **Unreachable** — cannot query the cluster. Report that; do not claim anything about the
  install state.

## Step 5 — Triage failures in plain language

When preflight or verification surfaces a problem, use `references/troubleshooting.md`: it
maps common symptoms (PVC Pending, image pull errors, aiwb-api CrashLoopBackOff, missing
CRD, HTTPRoute not attaching) to likely cause and the specific fix. Explain the _why_ to
the user, not just the command — the goal is that they end up with a working install and
understand what was wrong.

## References

- `references/preflight.md` — What a "ready" cluster means and how to check each
  prerequisite; which checks are hard blockers vs. mode-dependent. **Load when**
  interpreting preflight output or deciding whether to proceed.
- `references/verification.md` — How to confirm each layer is healthy, the health
  endpoints, and the access URLs to report. **Load when** running Step 4.
- `references/troubleshooting.md` — Symptom → likely cause → fix, in the gotcha pattern.
  **Load when** preflight or verification reports a problem.

## Scripts

- `scripts/preflight-check.sh` — read-only cluster readiness probe. **Run in** Step 2.
- `scripts/verify-install.sh` — post-install health probe; exits non-zero and prints an
  "inconclusive / what to fix" summary on partial installs. **Run in** Step 4.

## Tests

Run the regression suite after changing either probe. Use the full path so it works from the
repo root (the working directory the skill's other commands use); it also runs from the
`tests/` directory if you prefer:

```bash
bash .agents/skills/guided-workbench-install/tests/run-tests.sh
```

The suite stubs `kubectl`, requires no cluster, and pins the false-health regressions these
probes must prevent. It also validates the evaluation dataset and self-checks its own assertion
helpers, so it is the single entry point the repo's agent-skill-tests CI job discovers for this
skill.
