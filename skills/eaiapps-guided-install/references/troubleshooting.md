<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Troubleshooting: symptom → cause → fix

This is an AI instruction file. Use it when preflight or verification surfaces a problem.
Each entry is symptom, likely cause, and the concrete fix. Explain the _why_ to the user,
not just the command — the goal is a working install they understand.

## Cluster unreachable

`kubectl` times out, `connection refused`, or `Unauthorized`.

- **Wrong or expired kubeconfig / context.** Check `kubectl config current-context` and
  `kubectl cluster-info`. Point at the right context or re-fetch credentials for the
  cluster. Do not proceed with any install command until `kubectl get ns` works.
- **No permission to create namespaces/charts.** `kubectl auth can-i create namespaces`
  should say `yes`. If not, the user needs cluster-admin-equivalent rights for the install.

## PVC stuck Pending (CNPG / RabbitMQ never come up)

`kubectl get pvc -A` shows `Pending`; CNPG cluster never reaches healthy.

- **StorageClass name mismatch.** The infra charts default to StorageClass `default`, but
  the cluster's is named something else (e.g. `standard` on kind). Fix with the correct
  per-chart override — and note the key DIFFERS per chart:
  - `airm-cnpg` / `aiwb-cnpg`: `--set storage.storageClass=<NAME> --set walStorage.storageClass=<NAME>`
  - `airm-rabbitmq`: `--set persistence.storageClassName=<NAME>`

  A single `storage.storageClass` does NOT cover RabbitMQ — that's the most common miss.

- **No StorageClass / no provisioner.** `kubectl get storageclass` is empty or the
  provisioner isn't running. Install a StorageClass with a working dynamic provisioner
  before retrying.
- **Not enough nodes for the CNPG instance count.** `aiwb-cnpg` defaults to 3 PostgreSQL
  instances spread across nodes. The spread is a preference by default, so surplus
  instances co-schedule rather than hang — but with `podAntiAffinityType: required` they
  stay `Pending` until enough nodes exist. On a single-node cluster, install with
  `--set instances=1`.

## `kubectl drain` blocked on a node running PostgreSQL

Draining hangs and cites a `PodDisruptionBudget`.

- **The CNPG PDB is doing its job.** A PDB is created once a cluster has two or more
  instances, to stop the primary being evicted while a replica is still catching up. With
  instances co-scheduled on one node it will block that node's drain. Either scale to a
  single instance (`--set instances=1`, which switches the PDB off) or override with
  `--set enablePDB=false` when you need maintenance to proceed.

## `helm install aiwb` fails on a missing CRD

Error references `AIMClusterRuntimeConfig` or `aim.eai.amd.com/v1alpha1` as unknown.

- **AIM Engine CRDs not installed.** The AIWB chart creates an `AIMClusterRuntimeConfig`,
  so the AIM Engine CRDs must exist first (INSTALL.md Tier 3). Install AIM Engine (or at
  least its CRDs) before the AIWB app chart, then re-run the install.

## `aiwb-api` in CrashLoopBackOff or never Ready

Pods restart repeatedly, or readiness `/v1/health` never passes.

- **Database not reachable.** Check the CNPG `aiwb` cluster is healthy and the
  `aiwb-cnpg-user` secret exists with the expected keys. The API can't start without its
  DB. `kubectl logs deploy/aiwb-api -n aiwb` usually shows a DB connection error.
- **Missing secret.** A required secret (Keycloak, NextAuth, MinIO, or the `aiwb-openbao-token`
  the API uses to write API keys into OpenBao, not needed with `openBao.enabled: false`) is
  absent or has the wrong key name.
  INSTALL.md Step 2 lists exact secret names + keys; several use a specific key like `value`.
  Recreate the missing/misnamed secret and let the pod restart.
- **OpenBao disabled on an installation that has API keys.** The pod starts, then exits
  with `OpenBao store is disabled but the database holds N API keys` in
  `kubectl logs deploy/aiwb-api -n aiwb`. OpenBao is an install-time choice; the API refuses
  to run with keys that no store can revoke. Restore `openBao.enabled: true` (AIWB app chart)
  and `openbaoToken.enabled: true` (aiwb-external-secrets chart), delete every API key, and
  only then disable OpenBao again (INSTALL.md "Option A").
- **Image pull failure.** `ImagePullBackOff` → the registry needs a pull secret or the
  tag/repo is wrong. Check `backend.image` values and any `imagePullSecrets`.

## HTTPRoute not attaching / services unreachable at their URLs

`kubectl get httproutes -A` shows routes not accepted, or the URLs 404.

- **Gateway name/namespace mismatch.** The charts attach to a Gateway named `https` in
  `envoy-gateway-system` by default. If the cluster's Gateway differs, re-install/upgrade
  with `--set gateway.namespace=<NS> --set gateway.gatewayName=<NAME>` (AIWB) and, for AIRM,
  the `airm-api`-prefixed keys `--set airm-api.gateway.namespace=<NS> --set airm-api.gateway.gatewayName=<NAME>`
  (the AIRM umbrella has no top-level `gateway` key, so a bare `--set gateway.*` is silently
  ignored by Helm). Confirm the referenced Gateway actually exists.
- **No DNS / local dev cluster.** On kind/minikube there's no real ingress DNS. Use
  `kubectl port-forward` to the `aiwb-api` / `aiwb-ui` services instead of the public URLs.

## Standalone AIWB: workloads not getting labeled

Standalone mode installed but workload/project labels are missing.

- **Kyverno not installed.** Standalone AIWB (`standAloneMode=true`) relies on a Kyverno
  `ClusterPolicy` to auto-label workloads. Install Kyverno (INSTALL.md Optional
  Dependencies) — it's only needed in standalone mode; combined mode uses AIRM instead.

## GPU workloads pending / no GPUs allocatable

Model deployments never schedule; nodes show zero `amd.com/gpu`.

- **GPU operator not rolled out.** The AMD GPU Operator is installed but drivers/device
  plugin haven't finished rolling out, so nodes advertise zero allocatable GPUs. Wait for
  the operator's daemonsets to be Ready and re-check `kubectl describe node` allocatable.
- **Local dev cluster with no GPUs.** Expected — model deployment won't work there. The
  APIs/UIs still function; tell the user this is a limitation of the environment, not a
  failed install.

## When you genuinely can't tell

If a component's state is ambiguous and you can't confirm health, report **inconclusive /
what to fix** with the specific component and the next diagnostic command — never round up
to success. A clear "I couldn't verify X; run Y to check" is more useful than a false pass.
