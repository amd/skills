<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# What each check means

This is an AI instruction file. Use it to interpret `scripts/environment-check.sh` output — what
each check reads, the known platform bug it detects, and what OK / WARN / INCONCLUSIVE mean for
that check. Each check reports one of three states, and the distinction is the point of this skill:

- **OK** — the checker read the state it needed and found no problem.
- **WARN** — the checker read the state and found a real problem. The line names the fix.
- **INCONCLUSIVE** — the checker could not read the state it needed (the component's CRD is absent,
  RBAC denied the read, or the API errored). This is never treated as healthy: a silent bug must
  never pass by omission. An inconclusive check is often expected before install (the component
  isn't there yet) — say so, but never call the overall run healthy while one stands.

The overall result is **HEALTHY only when every gating check is OK** (every check gates, except the informational ones shown in `--mode e2e`, which never affect the result). Any WARN makes it PROBLEMS FOUND;
otherwise any INCONCLUSIVE makes it INCONCLUSIVE (not proven healthy).

## Contents

- [cilium-mtu — model downloads hang (scenario 4)](#cilium-mtu--model-downloads-hang-scenario-4)
- [longhorn-health — shared storage unhealthy (scenario 5)](#longhorn-health--shared-storage-unhealthy-scenario-5)
- [storage-provisioner — provisioner not live (scenario 5)](#storage-provisioner--provisioner-not-live-scenario-5)
- [stuck-storage-pods — workloads stuck on storage (scenario 6)](#stuck-storage-pods--workloads-stuck-on-storage-scenario-6)
- [gpu-metrics — broken metrics pipeline & silent failure (scenarios 7 and 8)](#gpu-metrics--broken-metrics-pipeline--silent-failure-scenarios-7-and-8)
- [aim-profiles — GPUs with no matching AIM profile (scenario 9)](#aim-profiles--gpus-with-no-matching-aim-profile-scenario-9)
- [required-secrets — unsynced ExternalSecrets](#required-secrets--unsynced-externalsecrets)
- [object-storage — object storage present](#object-storage--object-storage-present)
- [object-store-fill — SeaweedFS disks full](#object-store-fill--seaweedfs-disks-full)
- [aimartifact-rbac — permission to create AIMArtifacts](#aimartifact-rbac--permission-to-create-aimartifacts)
- [aim-route-hostnames-ready — catalog AIM deploys can get an HTTPRoute](#aim-route-hostnames-ready--catalog-aim-deploys-can-get-an-httproute)
- [eai-e2e-rbac — the E2E test identity has the permissions the suites use](#eai-e2e-rbac--the-e2e-test-identity-has-the-permissions-the-suites-use)
- [airm-quota-health — projects are provisioned and none is stuck deleting](#airm-quota-health--projects-are-provisioned-and-none-is-stuck-deleting)
- [kaiwo-queue-config — the Kaiwo controller can reconcile its queues](#kaiwo-queue-config--the-kaiwo-controller-can-reconcile-its-queues)
- [gateway-auth — the inference gateway is serving auth decisions](#gateway-auth--the-inference-gateway-is-serving-auth-decisions)
- [app-pods-stable — core app pods running and not flapping](#app-pods-stable--core-app-pods-running-and-not-flapping)
- [aiwb-api-healthy / aiwb-ui-healthy — workbench deployments up](#aiwb-api-healthy--aiwb-ui-healthy--workbench-deployments-up)
- [ui-auth-endpoints — people can sign in to the UIs](#ui-auth-endpoints--people-can-sign-in-to-the-uis)
- [oidc-issuer-consistency — each UI/API enforces the issuer Keycloak publishes](#oidc-issuer-consistency--each-uiapi-enforces-the-issuer-keycloak-publishes)
- [app-startup-errors — product logs free of sign-in and recurring errors](#app-startup-errors--product-logs-free-of-sign-in-and-recurring-errors)
- [aim-ready — a model can be deployed when there are GPUs to serve it](#aim-ready--a-model-can-be-deployed-when-there-are-gpus-to-serve-it)
- [aim-model-health — no AIM model is Degraded, Failed or Error](#aim-model-health--no-aim-model-is-degraded-failed-or-error)
- [argocd-apps — every Application Synced and Healthy, and what each tracks](#argocd-apps--every-application-synced-and-healthy-and-what-each-tracks)
- [deployed-versions — the images the run was tested against](#deployed-versions--the-images-the-run-was-tested-against)
- [gpu-schedulable — GPUs allocatable and admittable](#gpu-schedulable--gpus-allocatable-and-admittable)
- [release-spec-skew — the deployed products contain the specs about to run (opt-in)](#release-spec-skew--the-deployed-products-contain-the-specs-about-to-run-opt-in)
- [image-exists — a container image reference resolves](#image-exists--a-container-image-reference-resolves)
- [Before vs. after install](#before-vs-after-install)
- [finetuning-hf-token — gated fine-tuning needs a token that works (opt-in: `--require-hf-token`)](#finetuning-hf-token--gated-fine-tuning-needs-a-token-that-works-opt-in---require-hf-token)
- [E2E prerequisites (opt-in: `--e2e-prerequisites`)](#e2e-prerequisites-opt-in---e2e-prerequisites)

## cilium-mtu — model downloads hang (scenario 4)

The bug: Cilium auto-detects the node MTU independently per node. On a cluster with mixed NICs the
effective MTUs diverge (observed 9100 on one node vs 1280 on another), and large model-weight
downloads then hang partway with no error. A cluster-wide pinned MTU in the `cilium-config`
ConfigMap removes the per-node auto-detect entirely.

Scope limit: the per-node _effective_ MTU lives on the node OS and is **not** exposed on the
`ciliumnodes` API objects (verified against a live cluster), so the checker cannot compare per-node
MTU from the API. It reports on whether MTU is pinned cluster-wide, and — because divergence cannot
be excluded from the API when MTU is left to auto-detect — treats an unpinned config as unconfirmed
rather than healthy. A single-node cluster is the exception: with one node there is nothing for the
MTU to diverge from, so it is OK whether or not the MTU is pinned.

- **OK (pinned)** — `cilium-config` pins `mtu` to a concrete value cluster-wide. No auto-detect
  divergence is possible.
- **OK (not applicable)** — the `ciliumnodes.cilium.io` CRD is _confirmed_ absent; not a Cilium
  cluster, so this Cilium-specific bug does not apply. Reported OK (not inconclusive) so it never
  blocks HEALTHY on a non-Cilium cluster.
- **INCONCLUSIVE (CRD unreadable)** — the CRD read failed for a reason other than a confirmed
  not-found (RBAC forbidden, timeout, API error). A failed read is never treated as "not
  applicable"; the state is simply unconfirmed.
- **INCONCLUSIVE (auto-detect)** — `cilium-config` leaves `mtu` unset (the Cilium default). The
  per-node effective MTU is not API-observable, so divergence and the resulting model-download hang
  cannot be ruled out. The check does not certify the cluster healthy: compare node MTUs and pin
  `cilium-config` mtu to reach a proven-healthy result.
- **INCONCLUSIVE** — `cilium-config` itself is unreadable (RBAC/API error).
- **`--mode e2e`** — still reported, but informational: it never gates the result (its JSON entry
  carries `"gating": false`). It describes node-level state that does not decide whether the
  suites can run.

## longhorn-health — shared storage unhealthy (scenario 5)

The bug: Longhorn RWX volumes backed by nfs-ganesha can hang (an `flock()` stall on Longhorn RWX),
leaving workloads unable to mount storage. The checker reads each Longhorn volume's `robustness`
and `state`.

- **OK** — every volume is `healthy`, or `unknown` while `detached` (Longhorn cannot judge a
  detached volume, so every idle PVC reads `unknown/detached`), or Longhorn is present with no
  volumes provisioned yet.
- **WARN** — one or more volumes are `degraded`, `faulted`, blank, or `unknown` while `attached`.
  Workloads may fail to start or mount. Check the Longhorn UI and the nfs-ganesha pods and recover
  the volumes before deploying workloads.
- **Settle re-read** — a freshly attached volume reports `unknown` until its engine settles. When
  every unhealthy volume is exactly `unknown` while `attached`, the check waits about 60s and reads
  the volumes once more; that second read is authoritative. A volume clears only by turning healthy
  (or disappearing); any volume unhealthy in the second read warns, in whatever bad state it is in
  then, including one that was healthy in the first read. `degraded`, `faulted` and blank volumes
  never wait. If the second read fails, the check warns (it does not retry again or stay silent).
  `ENVIRONMENT_CHECK_LONGHORN_RECHECK_SECONDS` shortens the wait for tests; it is clamped to 0..60
  and an invalid value falls back to 60.
- **OK** (also) — no `volumes.longhorn.io` CRD and every StorageClass is node-local
  (exactly `rancher.io/local-path` or `kubernetes.io/host-path`; `kubernetes.io/nfs` and
  `kubernetes.io/no-provisioner` do not qualify): Longhorn is not in use, so its health is not applicable.
- **INCONCLUSIVE** — no `volumes.longhorn.io` CRD while some StorageClass is not node-local (another
  shared-storage layer may be in use), or the volume or StorageClass query failed. Shared-storage
  health is unconfirmed via Longhorn.
- **`--mode e2e`** — an unhealthy volume whose PVC namespace is the test namespace, an `e2e-*`
  namespace or a platform namespace (`aiwb`, `airm`, `keycloak`, `seaweedfs-instance`, `minio-tenant-default`) can WARN, and so can one with no bound PVC namespace (its scope is unconfirmed, never
  assumed to be elsewhere); an unhealthy volume confirmed in any other namespace is named in an
  informational note on an OK result.

## storage-provisioner — provisioner not live (scenario 5)

The bug: a StorageClass whose backing CSI/provisioner pods are not Ready leaves every new PVC
`Pending` indefinitely. The checker lists StorageClasses and the CSI driver pods.

Each StorageClass is correlated to its own `.provisioner`, per provisioner — not a global pod
sweep — through three positive signals for that exact driver: (a) it is a registered `CSIDriver`;
(b) its node plugin is registered on at least one node (`CSINode.spec.drivers[].name`); and (c) its
controller — the `external-provisioner` sidecar that actually handles `CreateVolume` — is alive,
proven by a leader-election `Lease` named after the driver (e.g. `external-provisioner-leader-<driver>`)
whose `renewTime` is fresh (renewed within ~2 min). This ties liveness to the StorageClass's own
driver end to end, so an unrelated Ready CSI pod cannot vouch for it.

- **OK** — every StorageClass provisioner is a registered CSI driver with a live node plugin and a
  live controller (fresh leader lease) — or is an in-tree/non-CSI provisioner (`kubernetes.io/*`),
  which is exempt.
- **WARN** — a provisioner is not a registered CSIDriver, its node plugin is on no CSINode, or its
  controller's leader lease is stale (not renewing); new PVCs will stay Pending.
- **INCONCLUSIVE** — no StorageClass at all; the CSIDrivers / CSINodes / Leases can't be read; or no
  leader lease was found to establish controller liveness. Do not assume healthy.

## stuck-storage-pods — workloads stuck on storage (scenario 6)

An in-flight signal: pods stuck `Pending`, emitting `FailedMount` events, or Jobs/events ending in
`DeadlineExceeded` are workloads that cannot get storage or whose model download stopped making
progress. Pre-install there is usually nothing running, which is a clean pass (the checker read the
state and found none stuck) — not a false failure.

A Pending pod is attributed to storage **only** when its scheduling condition points at storage (an
unbound PVC / volume) — a pod Pending for CPU, affinity, or taint reasons is not counted. A
`FailedMount` event is always definitive storage evidence. `DeadlineExceeded` is reported
separately as a possible stalled model download, because the known Cilium MTU divergence manifests
as a download Job timing out rather than a FailedMount.

- **OK** — no workloads stuck waiting on storage or model downloads.
- **WARN** — a workload has a `FailedMount` event, is Pending with a scheduling condition naming an
  unbound PVC / volume, or a Job/event reports `DeadlineExceeded`. Resolve the underlying storage
  problem; for timed-out model downloads also check `cilium-mtu`, then retry.
- **INCONCLUSIVE** — pods, events, or Jobs could not be listed (RBAC/API error).
- **`--mode e2e`** — only workloads in the test namespace, an `e2e-*` namespace or a platform namespace (`aiwb`, `airm`, `keycloak`, `seaweedfs-instance`, `minio-tenant-default`) can WARN; a stuck
  workload elsewhere is named in an informational note on an OK result.

## gpu-metrics — broken metrics pipeline & silent failure (scenarios 7 and 8)

The pipeline is `amdgpu driver -> metrics exporter -> OTel Collector scrape -> Mimir -> UI`. The exporter pods are found by the `app.kubernetes.io/name` label `metrics-exporter` (current AMD GPU operator) or `device-metrics-exporter` (older charts).
The exporter and collector pods can show `Running`/`Ready` while silently failing, so the checker
never trusts Ready — it reads the function signal in the logs of **every** exporter pod (it is a
per-node DaemonSet) and the collector. This is the anti-silence core: a green pod that isn't doing
its job is still reported.

After the log-signature scan finds nothing, the check **positively** confirms delivery: it queries
the metrics backend (Prometheus/Mimir) via the API-server service proxy — a read-only `kubectl get
--raw` call, no port-forward, within cluster-reader RBAC — for a positive count of current GPU
series carrying a **non-blank `workload_id`** (`count(gpu_gfx_activity{workload_id!=""})`). That
directly encodes the incident signature: the exporter losing kube-apiserver access blanks the
labels, so series with a real `workload_id` disappear even though the pods are Ready. Prometheus
staleness means a returned value is current. The backend is auto-discovered (a `:9090` service in a
metrics namespace) or set with `--metrics-backend <ns>/<svc>:<port>`.

- **OK (delivering)** — the count of current GPU series with a non-blank `workload_id` is > 0:
  delivery is positively proven end to end.
- **OK (idle)** — the exporter/collector/backend are readable, the count is 0, and no Running pod
  requests or limits `amd.com/gpu`; zero workload-labelled series are expected when there is no GPU
  workload to report.
- **OK (no GPUs)** — no exporter is present and the node list is readable with a total
  `amd.com/gpu` capacity of 0 — the same fact `gpu-schedulable` reads. There is no GPU pipeline to
  prove on a CPU-only cluster. A failed node read or `--require-gpu` keeps it INCONCLUSIVE.
- **WARN** — an exporter/collector log shows a failure signature (broken hop named), OR an active GPU
  workload exists but the count is 0 / empty (no series, or blank labels — not delivering).
- **INCONCLUSIVE** — the exporter isn't present on a cluster that has (or may have) GPUs, logs are
  unreadable, the collector is absent, the metrics backend can't be found/queried, or workload state
  is unreadable when the count is zero.
  The check never asserts a pass it did not prove.
- **WARN (exporter -> kube-apiserver)** — an exporter pod logs an `i/o timeout` or a repeated
  `Failed to watch` reaching the kube-apiserver, so it cannot map pods to nodes and metric labels go
  blank; the UI shows workloads with 0 GPU/VRAM even though the pod is Ready. Restore node ->
  control-plane network reachability for the exporter pod (this often traces back to the same
  Cilium/network-policy problems as cilium-mtu).
- **WARN (OTel collector -> exporter)** — a collector (every pod labelled
  `app.kubernetes.io/component=opentelemetry-collector` in `otel-lgtm-stack`, whatever the operator
  named it) logs a scrape failure against an exporter pod's address (or its Service's address or DNS name) on
  the exporter's metrics port, so metrics never reach the backend. A failure on another port of the same
  host, such as a node-level target sharing a hostNetwork exporter's IP, is not this hop. Scrape failures of other targets (e.g. the AIRM custom-metrics job) are not this hop. Check the exporter Service/endpoints and the collector's
  scrape config.
- **INCONCLUSIVE** — no exporter found (expected before install or on a cluster without GPUs), any
  exporter/collector logs are unreadable, or the exporter is present but no collector exists (the
  scrape hop is missing). Do not assume `Running` means healthy, and never pass with a hop unread.

## aim-profiles — GPUs with no matching AIM profile (scenario 9)

The bug: a GPU or partition with no matching AIM profile leaves the `aimclustermodels` at
`ManagedProfilesReady=False, reason=NoSupportedProfiles` (and therefore `Ready=False`), so models
silently never deploy — the single most repeated confusion in the support channel.

- **OK** — every defined cluster model is affirmatively `ManagedProfilesReady=True`, or AIM is
  present with no models defined.
- **WARN** — one or more models report `ManagedProfilesReady=False, reason=NoSupportedProfiles`;
  they cannot deploy on this hardware. The line names the affected models and GPU device-id(s).
- **INCONCLUSIVE** — AIM is absent / unreadable, or any model lacks the `ManagedProfilesReady`
  condition or has it `False` for another reason. The checker never claims every model is supported
  unless every model's condition actively proves it.
- **`--mode e2e --model NAME`** (exact model resource name, repeatable) — only the named models
  can WARN; other models lacking a profile are noted, not gating. A named model the cluster does
  not have is a WARN. Without `--model` every model gates, except that in `--mode e2e` the WARN
  for a model lacking a profile is informational (`"gating": false`, with the reason in the
  verdict): a catalog model without a profile for the cluster's GPU id is a hardware-coverage gap,
  not a cluster fault, as long as at least one catalog model is both Ready and has a supported
  profile; a catalog where none does still gates, and so does an unresolved profile. General mode
  always gates.

## required-secrets — unsynced ExternalSecrets

The bug: a required ExternalSecret in ns `aiwb` (e.g. `aiwb-openbao-token`, which the chart
omits by design when it runs with `openbaoToken.enabled: false`) that fails to sync
(`SecretSyncedError`) blocks `aiwb-api` from starting, even though every other secret on the store
syncs fine. The check reads each ExternalSecret's `Ready` condition.

- **OK** — every ExternalSecret in ns `aiwb` is `Ready=True`.
- **WARN** — one or more are not `Ready`; the line lists them with their reason. The secret backend
  (OpenBao/ESO) must mint those keys before the workbench (or E2E) can come up.
- **INCONCLUSIVE** — the list is unreadable, or there are no ExternalSecrets yet (cluster not
  installed / not fully synced).

## object-storage — object storage present

Replaces `minio-ready`. Object storage is SeaweedFS (namespace `seaweedfs-instance`: master,
volume, filer and S3 pods); a cluster that has no SeaweedFS may still run the MinIO tenant secret
`default-minio-tenant-env-configuration` in `minio-tenant-default`. Either way the per-project
`minio-credentials` secret is minted **at test time** (AIRM's `minio-credentials-fetcher`
ExternalSecret), so it is not a pre-run gate; the pre-run signal is that the OpenBao
`ClusterSecretStore` `openbao-secret-store` is Ready, so the fetcher _can_ mint it from its source
keys (`minio-api-access-key` / `minio-api-secret-key`).

- **OK** — SeaweedFS has all its components Ready and the `filer-s3` Service has a ready endpoint (or, on a cluster without SeaweedFS, the MinIO tenant secret is present and its pods are Ready) and
  `openbao-secret-store` is Ready.
- **WARN** — a SeaweedFS component or MinIO tenant pod is missing or not Ready, or the `filer-s3` Service is absent or has no ready endpoint; neither SeaweedFS nor a MinIO tenant is
  found; or `openbao-secret-store` is absent or not Ready.
- **INCONCLUSIVE** — a read failed for a reason other than a confirmed not-found. The verdict names the identity (forbidden)
  as the cause only when the SeaweedFS namespace itself cannot be read; other unreadable states
  carry no such attribution.

## object-store-fill — SeaweedFS disks full

A full SeaweedFS fails every S3 PUT (`failed to find writable volumes for collection:<bucket>`)
while all pods stay Ready. The check runs `df` on every PVC-backed data mount of each volume server and
`weed shell volume.list` on a Ready SeaweedFS master through `kubectl exec`, so it needs `pods/exec` in
`seaweedfs-instance`.

- **OK** — every volume server is below 85% and the bucket's collection has a writable volume (or
  there is no SeaweedFS to measure).
- **WARN** — a volume server is at or above 85% (the verdict names it, and 95% and above is the
  critical level), or no volume of the bucket's collection is writable (none is both below `volumeSizeLimit` and not read-only) **and** `volume.list` reports no free volume slot to allocate a new one. An empty collection with no free slot is the same outage. Full volumes with free slots are normal rollover and stay OK; with no slot figure reported the result is INCONCLUSIVE, never an assumed outage.
- **WARN wins over INCONCLUSIVE** — a threshold breach already measured is reported as WARN even when another probe fails; the failed probe is noted in the verdict.
- **INCONCLUSIVE** — nothing breached but a probe could not confirm: `volume.list` shows no size limit or a volume without a size (never counted as writable); the exec is forbidden (a read-only login lacks the right to exec; the runner service
  account has it), fails, or its output cannot be parsed.

## aimartifact-rbac — permission to create AIMArtifacts

The test identity must be able to create AIMArtifact CRs in the test namespace; without it, adapter
tests fail with `Failed to create AIMArtifact CR <name>: Forbidden`. Checked with
`kubectl auth can-i create aimartifacts.aim.eai.amd.com -n <test-namespace>`. Without
`--test-namespace` the namespace is an existing `e2e-*` one (the E2E runs create their own; one that
already holds a LocalQueue is preferred); with none, the check is OK with nothing to check.

- **OK** — the current kubectl identity can create AIMArtifacts. The verdict is explicit that this
  rules out the RBAC-_config_ variant only; a green result does **not** rule out an app-side
  Forbidden (the check tests the kubectl identity, not the app's runtime code path).
- **WARN** — it cannot (Forbidden); grant `create` on `aimartifacts` to the test identity there.
- **INCONCLUSIVE** — the AIMArtifact CRD is absent, or the access review could not be evaluated
  (never treat an unevaluable permission check as allowed).

## aim-route-hostnames-ready — catalog AIM deploys can get an HTTPRoute

The bug: on a **multi-listener** AIM gateway, aim-engine requires `spec.routing.hostnames` on the
effective AIMClusterRuntimeConfig. Without it the AIMService gets no HTTPRoute
(`ConfigValid=False` / `RouteHostnameRequired`) and catalog AIM deploys fail. AIWB's own default
runtimeconfig sets hostnames correctly; the failure mode is aim-engine's competing default
overriding it with none. The check resolves the gateway + runtimeconfig names from `aiwb-api`'s env
(`AIM_GATEWAY_NAMESPACE` / `AIM_GATEWAY_NAME` / `AIM_CLUSTER_RUNTIME_CONFIG_NAME`, defaulting to
`kgateway-system` / `https` / `default`).

- **OK** — the gateway has a single listener (per-route hostnames aren't required), or it is
  multi-listener and the effective AIMClusterRuntimeConfig sets routing hostnames, or routing is
  explicitly disabled (empty hostnames then cannot break anything).
- **WARN** — the gateway is multi-listener but the effective AIMClusterRuntimeConfig has routing
  enabled with no hostnames; catalog AIM deploys will fail with `RouteHostnameRequired`. The fix is
  usually to set `clusterRuntimeConfig.enable=false` on aim-engine so AIWB's hostname-bearing
  default is effective.
- **INCONCLUSIVE** — the gateway or the AIMClusterRuntimeConfig can't be read (CRD absent /
  RBAC / API error); or the config omits `routing.enabled` with no hostnames, where the CRD default
  (`enabled=true`) may still break deploys but can't be confirmed from here. Routing hostnames are
  unconfirmed either way; unconfirmed state is never counted healthy.

## eai-e2e-rbac — the E2E test identity has the permissions the suites use

The bug: the `e2e-aim-test-author` ClusterRole was absent from a bloomed release, and roughly twenty
AIWB tests failed deep into the suite with a `Forbidden` that read like a product bug. The suites
author AIM CRs with `kubectl` as the test identity to stage cases the API cannot express, so a
missing verb only surfaces mid-run.

The check reads the ClusterRole's rules and diffs them against what the suites actually do, then
confirms a ClusterRoleBinding of the same name binds that role to the OIDC test user. Required:
`aimartifacts` get/create/patch; `aimprofiles` get/list; `aimmodels` get/list/create/patch/delete;
`aimmodels/status` get/patch; `aimservices` get/list/create/patch/delete. Wildcard rules (`*`) satisfy any
verb; a `*/<subresource>` resource wildcard covers that subresource (the RBAC authorizer has no
`<parent>/*` form), and a `resourceNames`-restricted rule does not count (it grants only named
objects, while the suites create arbitrary ones).

Note `aimmodels/status` is a distinct resource from `aimmodels` — granting the parent does not grant
the subresource, and the fine-tuning cases need the status write to stage a model that reports Ready.

- **OK** — every required verb is granted and the role is bound to an OIDC user.
- **WARN** — the role is absent, a required verb is missing (the line names the resource and verbs),
  the binding is absent, it references a different role, or it binds no OIDC user.
- **INCONCLUSIVE** — the role or binding read failed for a reason other than a confirmed not-found.

## airm-quota-health — projects are provisioned and none is stuck deleting

An AIRM project is a database entity that materialises a namespace labelled
`airm.silogen.ai/project-id` plus a Kueue ClusterQueue named after the project. Two failures strand a
whole run, and both are visible without creating anything:

- **Stuck deleting (the finalizer race)** — a project namespace has been deleting for at least five
  minutes and an `airm.silogen.ai/*` finalizer is still held. Holding it during the five-minute grace
  is normal teardown and is not warned; after the grace, a rerun that reuses the name fails and the
  leftover keeps consuming quota.
- **No active quota object** — a live project's same-named ClusterQueue is missing or not `Active`.
  Do **not** require positive local `nominalQuota`: every AIRM project queue joins the `kaiwo` cohort
  with borrowing enabled, and zero-quota projects deliberately borrow free resources (covered by the
  quota-preemption E2E specification). Local zero is no guarantee, not an inability to admit.

- **OK** — every live project namespace has an Active same-named ClusterQueue; deleting namespaces
  are within the five-minute grace or no longer held by an AIRM finalizer;
  or there are no project namespaces at all.
- **WARN** — one or more lifecycle/quota problems exist; they are reported together rather than one
  masking another.
- **INCONCLUSIVE** — the namespace list is unreadable; deletion age cannot be parsed; or live projects
  exist but Kueue's ClusterQueues are absent/unreadable (quota unconfirmed — never assumed fine).

## kaiwo-queue-config — the Kaiwo controller can reconcile its queues

A `KaiwoQueueConfig` whose `status.status` is `FAILED` means the controller errors on every
reconcile (typically a stale queue entry for a namespace that was deleted), so quota operations fail
across the next run. The verdict names the object and quotes its newest event when readable.

A `clusterQueues` entry listing a namespace that no longer exists is flagged too, since it is what
makes the controller fail.

- **OK** — no `KaiwoQueueConfig` is `FAILED` and every listed namespace exists, or Kaiwo is not installed.
- **WARN** — one is `FAILED`, or an entry lists a missing namespace.
- **INCONCLUSIVE** — the CRD, the objects, or the namespace list are unreadable.

## gateway-auth — the inference gateway is serving auth decisions

The bug: a stale certificate left the gateway answering 5xx instead of making an auth decision,
which surfaces later as unexplained inference failures. The check probes `GET /v1/models`.

The invalid-key half needs **no credentials**: a synthesized garbage key must be refused with `401`.
That single probe distinguishes a healthy gateway (refuses cleanly) from a broken one (errors) and
from a misconfigured one (admits anything). The gateway URL comes from the `aiwb-ui` deployment's
`AI_GATEWAY_URL`, or `--gateway-url`.

Supplying `--gateway-key <key>` adds the authorised half: a valid key must return `200` (or `203`,
the ext_authz short-circuit for an authorised models call).

A probe that gets no HTTP answer at all (a timeout, a dropped connection) is tried up to three
times, 2s and then 5s apart, before the gateway is called unreachable: a runner can miss one request
to an endpoint that answers promptly from elsewhere, and one miss must not refuse a healthy cluster.
Any HTTP status is the gateway's real answer and is never retried. Neither is a failure decided
before a request leaves, such as a malformed URL (a non-numeric port) or an untrusted certificate,
because it fails the same way every time. When every attempt goes unanswered the verdict names each
attempt, and repeated identical errors share one label ("attempts 1-3: ..."). With `--gateway-key`
the invalid-key and valid-key probes retry independently, so a gateway that never answers can add
about 47s for each. The values live in `_NETWORK_BACKOFF_SECONDS` in `scripts/environment_check.py`.

When every attempt from the operator's host was pure silence (each one timed out, connecting or
resolving the name), the same probe is sent from inside the cluster: `kubectl run` starts a
short-lived pod in `e2e-test-runner` (the namespace default account, no token, no privileges) that
runs it and is removed when it ends, under the same retry policy (an
untrusted certificate or a malformed URL stops it at once, so a later answer from another backend
cannot hide it). A refused or reset connection, a TLS failure, or a name that does not resolve on
any attempt is the public route answering badly: it is reported as is, never replaced by an answer
from elsewhere. Every address of the host is tried and each one's failure kept, so a refusal on one
address is not lost behind a timeout on the last. The name lookup has its own deadline: a resolver
that never answers is silence, one that answers "no such name" is not. Whether a failure was silence
is recorded when the exception is caught and decided in one place for every probe path, so
truncating or combining error text cannot change it. An endpoint that sent any byte, even part of
a status line before stalling, has answered, so that is never silence either. When a
discovery fetch is abandoned at its deadline, what it had reached is read before its connections
are closed, so the closing cannot add a failure the endpoint never caused. The E2E lanes reach the
gateway from inside the cluster, so a runner without public egress must not refuse a cluster whose
gateway answers there. That answer is judged exactly like a direct one, the verdict says it came
from inside the cluster, and with `--gateway-key` the valid-key call is sent from there too, once
(the probe retries inside the workload). The probe pod trusts only public CAs, so when it rejects the
gateway's certificate, on either call, the verdict says so and stays INCONCLUSIVE, pointing at the
gateway certificate and the CA its in-cluster callers trust, not the operator's. The probe needs
`create pods`, `create pods/attach` and `delete pods` in `e2e-test-runner`: `kubectl run --rm` removes
the pod, a delete after every probe removes it if kubectl timed out first, and its own deadline stops
it otherwise. When it cannot run, or gets no answer either, the check stays INCONCLUSIVE and names
both failures.

- **OK** — an invalid key is refused with 401 (and, when a key was supplied, a valid one succeeds);
  or the aiwb-ui deployment is readable but declares no `AI_GATEWAY_URL`, a confirmed "no inference
  gateway is wired to this workbench" — there is nothing to probe, so it is not-applicable (the same
  shape `cilium-mtu` uses on a non-Cilium cluster), not unproven health.
- **OK (no routes published)** — the invalid key gets a `404`, the Envoy AI Gateway route CRD
  (`aigatewayroutes.aigateway.envoyproxy.io`) is present, there are zero `AIGatewayRoute`s, and no
  HTTPRoute or GRPCRoute attaches to a Gateway that an `ext_authz` `SecurityPolicy` targets. The
  protected gateways come from the policies' real `targetRefs`, not from a gateway name. Envoy
  answers 404 when no route matches, **before** `ext_authz` runs, so no auth decision happens while no
  model is deployed. This is the normal state of a CPU-only cluster. If any of those reads fail, the
  CRD is absent, no ext_authz policy targets a gateway, or any such route exists, a 404 still WARNs.
  When `--gateway-key` was supplied, the same no-routes 404 is INCONCLUSIVE instead: the key cannot
  be validated.
- **WARN** — any 5xx (a certificate/upstream failure, not an auth decision), an invalid key that is
  not refused, or a supplied valid key that is rejected.
- **INCONCLUSIVE (in-cluster certificate)** — this host got only silence, and the probe pod, which
  trusts only public CAs, rejected the gateway's certificate on the invalid-key or the valid-key call.
  The cluster's inference calls fail the same way unless their callers trust the gateway's CA, so check
  the gateway certificate and, for a private CA, that its callers trust it.
- **INCONCLUSIVE** — the aiwb-ui deployment cannot be read (address unconfirmed, distinct from a
  confirmed no-gateway); a `--gateway-key` was supplied but no gateway address could be discovered;
  no HTTP response was obtained from here and the in-cluster probe could not run or got no answer
  either (a silent direct probe answered from inside the cluster is judged by that answer instead);
  a direct failure that is not silence (refused, reset, no such name); or its certificate could not
  be verified from this host. The certificate case is
  called out separately because it is ambiguous: the gateway's certificate may be stale (a real
  problem) or this host may simply not trust the cluster CA. Settle it by re-running where the CA is
  trusted, or comparing with `curl -v` from the E2E runner — a host whose trust store differs will
  disagree.

## app-pods-stable — core app pods running and not flapping

Catches the silent case a plain deployment-Available check misses: a pod that is Ready **now** but
has been crash-looping is not stable, and intermittent restarts of a backend cause user-facing
failures like the UI's `/api/projects` "fetch failed". Scans pods in the `aiwb` and `airm`
namespaces.

- **OK** — every app pod is Running/Ready with no fatal waiting state and restarts below the
  threshold.
- **WARN** — a pod never got scheduled (no `containerStatuses` at all, e.g. `Unschedulable`) and is
  past the startup grace; a container (init or main) is waiting in a fatal reason (`CrashLoopBackOff`,
  `CreateContainerConfigError`, `ImagePullBackOff`, `ErrImagePull` — the first is how the `aiwb-api`
  openbao-token block manifests), a pod has stayed NotReady past a startup grace (~5 min, so a
  rollout in progress is not flagged), a pod is Ready but has restarted at least
  `--restart-threshold` times (default 3; the flapping signal) and last died within the past 6
  hours, or a container last restarted within
  `--recent-minutes` (default 30), init containers included. A termination time more than 5 minutes
  in the future is a bad timestamp and is ignored. The recent-restart rule catches the API that
  died several times during a release's startup and then settled: below the threshold, but still a
  sign the release is not healthy. A pod whose last death is older than 6 hours is not flagged for
  its count: a node outage restarts every pod on the node, the count keeps those restarts for the
  pod's lifetime, and a pod that has run cleanly since is stable. An unreadable termination time
  keeps the count rule. The line names each offending pod; pods owned by a completed Job
  are ignored.
- **INCONCLUSIVE** — neither app namespace is present (before install), or the pod list is
  unreadable.

## aiwb-api-healthy / aiwb-ui-healthy — workbench deployments up

- **OK** — the deployment is `Available=True` with `readyReplicas > 0`.
- **WARN** — it exists but is not Available; a `CreateContainerConfigError` here usually means a
  required secret/config is missing (see required-secrets).
- **INCONCLUSIVE** — the deployment is not found (expected before install) or the read failed.

## ui-auth-endpoints — people can sign in to the UIs

The bug: on a release whose UI enforced the wrong OIDC issuer, every `/api/auth/*` call answered
500, so nobody could log in — while ArgoCD showed every Application Synced/Healthy, every pod was
Running, the homepage answered 200/307, and E2E (which logs in through the API) passed. The check
GETs `/api/auth/providers` and `/api/auth/session` of the `airm-ui` and `aiwb-ui` Services through
the API-server service proxy (`kubectl get --raw`), so it needs only the operator's kubeconfig — no
public DNS, TLS trust, or gateway.

- **OK** — every installed UI answers both routes with 200 and a JSON object (NextAuth's answer to
  an anonymous caller).
- **WARN** — a route answered 4xx or 5xx (kubectl reports a 502 as `InternalError`, like any 5xx it
  has no name for, and any other unnamed code as `the status code N`), or answered success without a JSON object (kubectl hides which 2xx, so a 204 or
  an HTML page counts), or the providers route lists no sign-in provider (an entry with an `id` and a `signinUrl`, as NextAuth sends), or a UI Deployment is
  installed but its Service is missing.
  Nobody can sign in; read `app-startup-errors` and `oidc-issuer-consistency` for the cause.
  A Service whose port is named (`web`) answers "no endpoints available" to a proxy request that
  names no port, so that answer is retried once against `<service>:<port name>`.
- **INCONCLUSIVE** — the API server could not deliver the request (no ready endpoint, a failed dial
  to the pod, or no `get services/proxy` in `airm`/`aiwb` — a plain cluster-reader lacks it), a UI
  Service is missing and its Deployment cannot be read, or neither UI is installed (expected before
  install). An
  undelivered request says nothing about how the UI answers, so it is never read as a pass or a fail.

## oidc-issuer-consistency — each UI/API enforces the issuer Keycloak publishes

The UIs and the AIWB API check the token issuer strictly; the AIRM API validates through its own
Keycloak client and hands its issuer to clients (Swagger login, generated kubeconfigs). The check
computes each deployment's issuer the way the product code does — UIs: `OIDC_ISSUER`, else
`KEYCLOAK_ISSUER`; aiwb-api: `OIDC_ISSUER`, else `KEYCLOAK_PUBLIC_URL/realms/KEYCLOAK_REALM`,
against the realm of its signing keys (`OIDC_JWKS_URL`, else `KEYCLOAK_INTERNAL_URL`, realm default
`platform`; a non-Keycloak JWKS URL leaves it unproven);
airm-api: `KEYCLOAK_PUBLIC_URL/realms/KEYCLOAK_REALM` against the realm at `KEYCLOAK_INTERNAL_URL`,
ignoring the `OIDC_*` values it never reads — and compares it with the `issuer`
of Keycloak's `.well-known/openid-configuration` for that realm, fetched from the same URL the
workload uses (in-cluster Keycloak URLs — `<service>.<namespace>.svc`, optionally followed by the cluster domain (`--cluster-domain`, default
`cluster.local`),
over http(s) — through the service proxy, public ones directly). When the proxy read cannot be
delivered (a plain cluster-reader lacks `get services/proxy` in the Keycloak namespace), the
workload's public issuer URL is read instead. Keycloak may derive the issuer from the address it was
asked on, so a pass there means "no contradiction found", not "consistent", and says nothing about
which Keycloak the workload reaches; when neither can be read the issuer is unproven (INCONCLUSIVE), and
both failures are named. The public read applies only when the public URL has the same realm path as the
URL the workload reads. When Swagger's URL is the issuer's own and the direct read is refused because its host resolves to a private-unicast address, the workload's in-cluster discovery URL for the same realm is read through the service proxy instead (no `--trusted-private-host` needed). The same in-cluster read replaces a direct read that was pure silence on every address and attempt (a connection or a name lookup that timed out, as when the operator's host has no route to the public Keycloak), since the workloads read the realm in-cluster too. A lookup that answers "no such name", a TLS failure, or a refused or reset connection is not silence and gets no substitute, even when another attempt or address timed out. Either way the verdict says the public route Swagger uses was not checked. A proxy read Keycloak answered is never retried. A direct
fetch leaves from the operator's host, so it is refused when the host resolves to a non-public
address (loopback, link-local such as cloud metadata, private) or cannot be resolved, and the
connection goes only to the addresses that were vetted (each in turn), so re-resolving the name
cannot redirect it; the issuer is then unproven. A direct fetch that gets no answer (a timeout, a
dropped connection, a failed lookup) is tried up to three times, 2s and then 5s apart, as for
`gateway-auth`; a refused non-public address or a malformed URL is not retried, and the verdict names
every attempt. The workload's URL and the public fallback each retry, so a Keycloak that never
answers on either can add about 47s for each read. The one exception is a host the operator
names with `--trusted-private-host HOST` (repeatable; typically the Keycloak host, often published on
a private load-balancer address): for it a private-range address (RFC 1918, ULA) is accepted, while
loopback, link-local, multicast and reserved ranges stay refused. The cluster's own configuration
never grants this, because it is what the check is judging. A
discovery URL on `localhost` is a WARN without any fetch: inside the pod, localhost is the pod. Apart from the
public-issuer read above, it never falls back to the configured issuer: that document names that
issuer by construction and proves nothing about the Keycloak the workload reaches.

The Deployment shows only what the chart sets. A UI image can carry its own `OIDC_ISSUER`, which
wins over the chart's `KEYCLOAK_ISSUER` — the observed failure, where the image's development
default `http://localhost:8080/realms/airm` replaced a correct `KEYCLOAK_ISSUER`. No Deployment read
reveals that, so a UI's own log line (the last `--recent-minutes`, read after `ui-auth-endpoints`
has made each UI fetch its discovery document) reporting a discovery document that `names the issuer X,
but OIDC_ISSUER is Y` takes precedence: `Y` is the issuer it really enforces. The absence of that
line proves nothing (an image pointing at another, self-consistent server logs no mismatch), so a
UI on the fallback is never confirmed healthy.

- **OK** — every installed UI/API enforces the published issuer. For a UI a trailing slash is not a
  difference (the UI trims both sides); an API compares its issuer with the token's `iss` exactly,
  so for it a trailing slash is another issuer. Only the product container's configuration is read
  (`airm-ui`, `aiwb-ui`, `airm`, `aiwb`); a sidecar's env never fills in or overrides it. Two OK
  confirmations rest on weaker evidence, and the verdict text says which: a document read only from the
  workload's public issuer URL (the proxy was denied) means "no contradiction found", not which Keycloak
  the workload reaches; and a Swagger discovery URL whose host resolves to a private address from the
  operator's host is read in-cluster instead, so the public route Swagger uses was not checked. A gate
  that must prove sign-in works reads the verdict text, not only `status`, or runs with the proxy grant.
- **WARN** — an issuer differs from the published one; an API hands Swagger an
  `OPENID_CONFIGURATION_URL` (default: `KEYCLOAK_PUBLIC_URL/realms/KEYCLOAK_REALM/.well-known/openid-configuration`,
  with the product's `http://localhost:8080` when `KEYCLOAK_PUBLIC_URL` is unset) whose document
  names another issuer, or that is on localhost. That document is read even when its URL is the
  issuer's own, because Swagger reaches it through whatever serves that URL (often a public proxy
  the token check never uses); if it cannot be read, the issuer is unproven; a UI logs that it enforces another issuer;
  a UI sets neither `OIDC_ISSUER` nor `KEYCLOAK_ISSUER`; aiwb-api sets neither `OIDC_ISSUER` nor
  `KEYCLOAK_PUBLIC_URL` (it then checks against `http://localhost:8080/realms/platform`), or neither
  `OIDC_JWKS_URL` nor `KEYCLOAK_INTERNAL_URL`; airm-api leaves `KEYCLOAK_PUBLIC_URL` or
  `KEYCLOAK_INTERNAL_URL` unset (it then uses `http://localhost:8080` with realm `airm`); or an API
  sets a Keycloak value it reads to an empty string, which it uses as is. Issuer problems and Swagger
  problems get separate sentences and separate advice: the Swagger fix is `OPENID_CONFIGURATION_URL`,
  which the issuer settings do not change.
- **INCONCLUSIVE** — a UI on the `KEYCLOAK_ISSUER` fallback that logs no mismatch (the image may
  override its issuer unseen; set `OIDC_ISSUER` on the Deployment to make it checkable), a Deployment with
  no container named as the product's (even a lone one), a deployment cannot be read, or the discovery document cannot be read both in-cluster and from the public issuer URL, an issuer variable is set through
  `valueFrom`, or any issuer input is left unset while the container has `envFrom` (not
  resolved here; a literal entry wins over `envFrom`), or no UI/API is installed.

## app-startup-errors — product logs free of sign-in and recurring errors

Reads the last `--recent-minutes` (default 30) of logs of every pod of `airm-ui`, `aiwb-ui`,
`airm-api`, and `aiwb-api` (all containers).

- **OK** — no sign-in error and fewer than 3 unhandled-error lines per deployment. The UIs' other
  discovery messages — a retried fetch, an endpoint outside the issuer, logout ending a session
  locally — are not sign-in errors on their own. A deployment whose pods logged nothing in the window
  but has a ready replica also passes; the verdict names it and says that only its readiness, not its
  log, was checked.
- **WARN** — a line containing `OIDC discovery document from` (a discovery document naming another
  issuer or lacking the login endpoints, e.g. `… names the issuer https://kc.<your-domain>/realms/airm,
but OIDC_ISSUER is http://localhost:8080/realms/airm`) or `Invalid issuer` (an API rejecting
  tokens; one can be a client presenting another realm's token, so it counts from 3 lines), or 3 or
  more lines with `unhandled` / a Python traceback. The latest matching line is
  quoted.
- **INCONCLUSIVE** — a deployment or its logs cannot be read, no pod matches the Deployment, or
  its pods wrote nothing in the window and none is ready, a container logged 10000 or more lines
  in the window (the earliest were not scanned; a finding in what was read still WARNs), or no
  UI/API is installed.

## aim-ready — a model can be deployed when there are GPUs to serve it

Inference suites draw their models from the AIM catalog filtered to `Ready` — the AIMClusterModels
the AIWB API lists, without base-only models — and fail late with "No Ready AIMs available in
system" when it is empty. Running AIMServices are not the precondition: suites deploy their own.

- **OK** — at least one catalog AIMClusterModel has status `Ready`, or the cluster has no
  allocatable GPUs (no AIM is expected).
- **WARN** — GPUs are allocatable but no catalog AIM is `Ready`. The line counts them by status, or
  says the catalog is empty.
- **INCONCLUSIVE** — nodes or AIMClusterModels cannot be read, or the CRD is absent on a GPU
  cluster.
- **`--mode e2e --model NAME`** — instead of "any model is Ready", each named model must be in
  the catalog and Ready, including on a cluster with no GPUs; one that is not is a WARN.

## aim-model-health — no AIM model is Degraded, Failed or Error

- **OK** — no AIMModel or AIMClusterModel has status `Degraded`, `Failed` or `Error`.
- **WARN** — models are `Degraded`, `Failed` or `Error`; each is named with its status and the reason
  of its first false condition.
- **INCONCLUSIVE** — a model CRD or list cannot be read, a model reports no status or one the engine
  does not define (health not established), or neither CRD exists (AIM not installed).
- **`--mode e2e`** — a namespaced model is examined only in the test namespace or a platform
  namespace (`aiwb`, `airm`, `keycloak`, `seaweedfs-instance`, `minio-tenant-default`); one in another
  `e2e-*` namespace is a leftover of an earlier (possibly cancelled) run and gates only when
  `--model` names it; a cluster-scoped
  model is examined only when named by `--model` (every one when none is given). The rest are
  noted as informational.

## argocd-apps — every Application Synced and Healthy, and what each tracks

Reports the `targetRevision` mix across all Applications (a source without one follows `HEAD`) and
names every Application not pinned to a release or commit, so one that follows a branch is visible
in the record of the run. Pinned means an exact SemVer-shaped version (`2.0.4-rc.1`, `v2.0.4`,
`0.0.0-dev`) or a commit SHA; everything else — `HEAD`, `main`, `release/v2.4.x`, a chart
version range — deploys whatever lands next. Telling a branch from a same-named tag needs the source
repository, which this offline check does not read, so a non-version tag is named too. Following a
branch is reported, not warned: some environments do it on purpose.

- **OK** — every Application is `Synced` and `Healthy`, or ArgoCD is not installed. A `Synced` Application that is `Progressing` only because its PVCs are `Pending` on a `WaitForFirstConsumer` StorageClass is also OK: the claim binds once a pod uses it, so that is steady state. An `ExternalSecret` in such an Application has no Argo health; it is accepted only when its own sync status is `Synced` (or it is a hook the last sync ran to `Succeeded`). An `ExternalSecret` that is `OutOfSync` or of unknown state, or any other health-less kind not already excused stays WARN (kinds Argo never assesses, such as a `ConfigMap`, remain excused). A `Healthy` Application whose status condition says `unable to resolve '<rev>' to a commit SHA` does not fail the check: the verdict starts `WARNING` and names each with that reason, since the workload runs and only the comparison with the declared revision is impossible. When it is not `Healthy`, it stays a WARN problem.
- **WARN** — an Application is not `Synced`/`Healthy`; each is named with its sync/health state.
- **INCONCLUSIVE** — the Application CRD or list cannot be read, or an otherwise Synced and Healthy
  Application has a source without a `repoURL` (or none at all), so its revision is unknown.

## deployed-versions — the images the run was tested against

Report-only: for every Deployment container whose image repository is `airm-*`, `aiwb-*`, or
`aim-engine*`, records the image and digest its Ready pods are running, as
`<namespace>/<deployment>[<container>]=<image> (<digest>)`, so the output of a pre-E2E run records
exactly which build was tested. The template image is only the desired state and can differ
mid-rollout, so it is never recorded as serving, and until a rollout completes (every desired
replica updated, Ready and available, with no old pod left) the Ready pods may all be old ones, so nothing is recorded. It never warns, but an unconfirmed record is
INCONCLUSIVE, so a run during a rollout exits 3.

- **OK** — each product Deployment's Ready pods run one build, listed.
- **INCONCLUSIVE** — Deployments or running pods cannot be listed, none matches (expected before
  install), or a Deployment is still rolling out, has a pod selector with `matchExpressions` (its
  pods cannot be matched by labels alone), has a Ready pod that does not show exactly one
  product container, has a number of Ready, non-terminating pods other
  than its replicas (its status lags the pods), has no Ready pod, a Ready pod that exposes no running image digest, or
  Ready pods running different builds (a rollout in progress); the confirmed builds are still listed.

## gpu-schedulable — GPUs allocatable and admittable

Two conditions must both hold for GPU workloads to run: nodes advertise allocatable `amd.com/gpu`,
and the `--test-namespace` LocalQueue resolves to a ClusterQueue with `amd.com/gpu` quota. Quota on
an unrelated ClusterQueue cannot admit workloads from the test namespace and never counts as proof.
Zero local quota is only a failure outside a cohort: an AIRM project queue joins the `kaiwo` cohort
with borrowing enabled and admits GPU jobs by borrowing, so a zero-quota queue with a cohort is OK —
matching airm-quota-health, which treats the same shape as healthy.

- **OK** — GPUs are allocatable and the test namespace's LocalQueue is bound to a ClusterQueue with
  positive `amd.com/gpu` quota, or with zero local quota but a cohort to borrow from (or, with no GPU
  nodes and without `--require-gpu`, GPUs are simply not applicable on this cluster).
- **WARN** — GPU capacity exists but 0 is allocatable (the GPU operator has not rolled out drivers),
  the test namespace has no usable LocalQueue/ClusterQueue binding, its referenced ClusterQueue is
  missing or has zero GPU quota and no cohort, or `--require-gpu` was set and there are no GPUs.
- **INCONCLUSIVE** — nodes, the namespace's LocalQueues, or its referenced ClusterQueue cannot be
  read (admission unverified).

## release-spec-skew — the deployed products contain the specs about to run (opt-in)

The problem: deployment-triggered E2E runs specs from the branch under test against whatever
release image is already deployed. When the deployed AIRM/AIWB build predates a spec's behavior, that
spec fails permanently on the environment — surfacing as a plain assertion failure that reads like a
product defect (e.g. a `202` where the newer code returns `400`) rather than a version gap.

This check is **opt-in**: it runs only when the caller passes `--spec-revision REV`, because only the
caller knows which core commit's specs will run. It needs `--core-repo PATH` (a core Git checkout
that contains that revision plus the deployed release tags/commits) to resolve markers to commits.

For AIRM and AIWB independently, it reads the image actually serving a **completed** Deployment
rollout: it requires the rollout to be complete (observed generation current, and total = updated =
ready = available = desired replicas with none unavailable, so no old pod is left), then confirms every selected Ready pod runs the
chart's product container (`aiwb` / `airm`, never a sidecar) on one identical image tag **and** one
identical running `imageID` manifest digest. The digest check catches a partial retarget — some pods
still serving an older manifest under the same tag text — that the requested tag alone would hide. It
resolves that image tag to a commit using only **immutable** markers — an annotated/lightweight
release tag under `refs/tags` whose name is either SemVer (e.g. `2.0.2-rc.3`) or one of the
repository's established calendar-release forms (`v2025.08-rc.21`, `v2025.08.002`), else a Git
short/long SHA embedded at the end of the tag (`main-fdb389c`, `<branch>-product-de73d50`) — and
compares by ancestry against the spec revision. A mutable branch image tag (`main`), a digest-pinned
reference (`…@sha256:…`, whose tag is not manifest provenance), a missing running `imageID`, mixed
running digests across Ready pods, an in-progress or ambiguous rollout, and a shallow core checkout
are all INCONCLUSIVE, never OK/WARN.

**Convention assumption.** The check pins each product to its conventional Deployment name and
namespace (`aiwb-api` in `aiwb`, `airm-api` in `airm`) and its chart product container (`aiwb` /
`airm`). These follow from the standard release name/namespace; a non-default release name or
namespace makes the Deployment read "not found", so the check reports INCONCLUSIVE and never fires —
fail-safe, but it will not catch skew on an off-convention install. It is intended for the standard
int-test/E2E deployment.

**Provenance limit (known residual gap).** This checker is offline (local checkout, no registry or
network), so it cannot bind the running `imageID` digest to a source commit through build
attestations. It resolves the deployed _tag text_ to a commit, which trusts the GitOps (Argo)
tag→digest binding. It therefore does catch a _partial_ retarget (mixed digests across pods, above),
but a registry tag _uniformly_ retargeted to a different build of the same name would still resolve by
tag text and pass. Closing that fully needs digest→commit resolution via the SLSA provenance the image
workflows publish, which requires registry/attestation access — a known follow-up outside this
checker. Until then, treat an OK from this check as "the deployed tag, as
bound by GitOps, contains the specs", not a cryptographic guarantee of the running manifest.

- **OK** — every deployed product's commit contains the spec revision (the spec commit is an ancestor
  of, or equal to, each deployed commit). The specs' behavior is present in what is running.
- **WARN** — a deployed product predates the spec revision (deployed is an ancestor of the spec
  commit), or the two have diverged (neither contains the other). The line names the product and both
  revisions. Deploy a build that contains the spec revision, or run the specs from the deployed
  release's branch, before the suite.
- **INCONCLUSIVE** — `--core-repo` is not a readable repo or has shallow history, the spec revision
  does not resolve in it, a product is not deployed or its rollout is incomplete/ambiguous, its image
  is digest-pinned or its tag resolves to no immutable commit, or a git comparison errors (including a
  negative ancestry answer that a shallow clone cannot prove). An unresolved marker is never treated
  as compatible.

Because both sides are reduced to commits and compared by `git merge-base --is-ancestor`, the verdict
is exact even when the deployed stack is not one coherent SemVer (mixed release + `main-<sha>` tags
are handled). Comparing release numbers, image timestamps, or Argo sync times would give false
results and is deliberately not done.

## image-exists — a container image reference resolves

Run as the `image-exists <ref>` subcommand or via `--image REF`. Used to catch an install/E2E that
references an image tag that was never published.

- **OK** — the manifest is present (via skopeo/crane/docker, or the public Docker Hub tag API).
- **WARN** — a definitive manifest-unknown / not-found from a registry that answered.
- **INCONCLUSIVE** — auth is required / the registry is unreachable / no inspection tool is
  available. A **private** repo commonly 404s on the public Hub API, which is NOT proof the tag is
  missing — so it is inconclusive, never a false WARN.

## Before vs. after install

The same checks run regardless of install state; interpret inconclusive results against it:

- **Before install**, a bare cluster legitimately lacks Longhorn, AIM, the GPU exporter, and the
  AIRM/AIWB UIs and APIs, so those checks (including the deployed-release checks: ui-auth-endpoints,
  oidc-issuer-consistency, app-startup-errors, deployed-versions) are inconclusive — that is readiness
  information ("these components aren't here yet"), not a failure. cilium-mtu and
  storage-provisioner are the readiness-relevant ones.
- **After install**, an inconclusive result for a component that _should_ be present is itself worth
  investigating — it usually means an RBAC or API problem reading it, not that the component is
  absent.

## finetuning-hf-token — gated fine-tuning needs a token that works (opt-in: `--require-hf-token`)

The bug: a gated base model (`hfTokenRequired: true` in `GET /v1/fine-tuning/models`) only downloads
when the job receives a token. A run with no token, a rejected token, or a token without access to
that model dies with `GatedRepoError 401` after the job is already scheduled.

The token is `HF_TOKEN` from the environment, else `e2e-apiserver-testing/e2e-hf-token` (key `token`).
It is never printed.

- **WARN `no token`** — neither source holds a token.
- **WARN `token rejected`** — Hugging Face `whoami-v2` answers 401/403.
- **WARN `no access to <model>`** — the model `auth-check` endpoint denies the token (the model-info endpoint answers 200 for gated repos, so it proves nothing).
- **INCONCLUSIVE** — Hugging Face, the model list, or the runner secret could not be read.
- **OK** — the token is accepted and reads every gated model.

## E2E prerequisites (opt-in: `--e2e-prerequisites`)

Four checks for what the E2E suites need and a cluster easily lacks. Each WARN names the item.

- **e2e-local-queues** — the `--test-namespace` has a Kueue LocalQueue. Without one every Job waits
  on admission until the test's check times out.
- **fine-tuning-models** — `GET /v1/fine-tuning/models` is non-empty. The OK line lists which entries
  are `hfTokenRequired`, which is what the `run_gated_recipe_tests=auto` selection reads.
- **optional-capability-prerequisites** — a LoRA-capable recipe (`trainingType: lora`), a
  speech-to-text AIM and a diffusion-only (text-to-video, not also chat or transcription) AIM exist. A WARN names the tests that are not
  runnable (adapter fine-tuning and serving, the transcription suite, the diffusion suite) and the
  `run_*` input to leave off, so they are excluded and counted rather than failing as skips.
- **team-member-sign-in** — the RBAC suites' second account can sign in. It reuses
  `tooling/e2e/setup_e2e_prerequisites.py` (resolved under `--core-repo`), so identity, password
  and grant are the ones the suites use; no secret is printed.
