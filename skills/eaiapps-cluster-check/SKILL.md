---
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
name: eaiapps-cluster-check
description: "Checks a Kubernetes cluster for silent AI Workbench platform problems and reports a plain-language or JSON verdict before install, before E2E, and after a deploy. Flags what looks healthy but is broken: UI sign-in errors, an OIDC issuer that disagrees with Keycloak, crash-looping or restarting pods, log errors, no Ready AIM on a GPU cluster, broken AIM models, ArgoCD apps out of sync, unpinned Cilium MTU, unhealthy storage, missing GPU metrics, unsynced secrets, unschedulable GPUs, missing E2E permissions, stuck AIRM projects, a failing inference gateway, or builds older than the specs; records deployed image versions. Use when checking AI Workbench cluster health or asked why login fails, downloads hang, storage won't mount, a model won't deploy, or an E2E suite failed on permissions, version skew, or no Ready AIMs. Use PROACTIVELY before an AIWB install, before an E2E run, and after a deploy. Do NOT activate for installing AIWB, onboarding AIM models, or kubectl/helm use unrelated to platform bugs."
allowed-tools: Bash, Read, Grep, Glob
---

# Cluster Environment Check

This is an AI instruction file. Follow it to check a Kubernetes cluster for known AI Workbench
platform problems and report a plain-language verdict an operator can act on.

The problems this skill finds are **silent**: a pod shows `Running`, metrics read zero, no error
surfaces — and the same multi-person confusion recurs each time. Your job is to run the read-only
checker, then translate its findings into what is wrong, why it matters, and the concrete fix.
**Never report a cluster healthy unless the checker actively proved every check passed** — an
unread check is not a pass.

## What this runs against

The checker uses `kubectl` with **the operator's own kubeconfig** — whatever cluster the current
context points at. It works both **before** a workbench install (readiness: is this cluster fit to
install onto?) and **after** (in-flight: is the running environment healthy?). Checks that are only
meaningful post-install (stuck workloads, GPU metrics, secrets, AIWB API/UI) report inconclusive
when the component isn't there yet, never a false failure. The checker reads the cluster and changes
nothing in it, with one exception: the in-cluster gateway probe below starts a short-lived pod of its own
and removes it again.
To keep it that way, the team-member sign-in check does not use the provisioning script's in-cluster pod
fallback, which creates a pod: when the machine running the checker cannot reach Keycloak, the check
reports the account as unproven.

The sign-in checks (`ui-auth-endpoints`, and the Keycloak half of `oidc-issuer-consistency`) reach
the UIs and Keycloak through the API server's service proxy, so the identity needs `get` on
`services/proxy` in the `airm` and `aiwb` namespaces and in the namespace Keycloak runs in (taken
from the in-cluster discovery URL, `keycloak` in the charts' default). That is still read-only, but a
plain cluster-reader account lacks it. `ui-auth-endpoints` then reports INCONCLUSIVE, naming the
missing permission, never OK. `oidc-issuer-consistency` instead falls back to the workload's public
issuer URL: if that document is readable it is judged from there. Keycloak may derive the issuer from
the address it was asked on, so a pass there means "no contradiction found", not "consistent", and says
nothing about which Keycloak the workload reaches. If the public document is not readable either, the check reports
INCONCLUSIVE. A pre-E2E or post-deploy gate that must prove sign-in works has to run as an identity
with the proxy grant.

A gateway or Keycloak address the checker's own machine gets no answer from (a runner without public
egress) is probed from inside the cluster instead, where the E2E lanes reach it: Swagger's discovery
document through the Keycloak service proxy, and the gateway from a short-lived unprivileged pod in `e2e-test-runner`, which
needs `create pods`, `create pods/attach` and `delete pods` there. The probe only sends a GET. `kubectl run --rm`
removes the pod when it ends, a `kubectl delete` after every probe removes it if kubectl timed out first, and
the pod's own deadline stops it if both fail. The pod trusts only public CAs, so a gateway certificate it rejects
is reported INCONCLUSIVE, not as a fault. The verdict names the vantage point that answered.

The implementation is `scripts/environment_check.py` (the `.sh` entrypoint is a thin wrapper). It is
importable, so other lanes reuse its checks — e.g. `from environment_check import image_exists` — and
it exposes a stable `--json` verdict for automated gates.

Confirm a cluster is selected before running:

```bash
kubectl config current-context
```

If that fails or points at the wrong cluster, fix the kubeconfig/context first — the checker
cannot diagnose a cluster it cannot reach.

## Step 1 — Run the check

Run it from this skill's own directory, the folder holding this `SKILL.md`, so the path below
resolves wherever the skill is installed:

```bash
bash scripts/environment-check.sh
```

It runs every check, prints one `OK` / `WARN` / `INCONCLUSIVE` line per finding, and ends with a
`RESULT:` line. The exit code is the contract:

- **`0` HEALTHY** — every gating check actively passed (in `--mode e2e` the informational checks do not count). No known platform problems.
- **`1` PROBLEMS FOUND** — at least one `WARN`. Each problem line names the cause and the fix.
- **`2` CANNOT REACH CLUSTER** — kubectl cannot talk to the API server. Not a healthy result.
- **`3` INCONCLUSIVE** — no problems found, but a check could not read the state it needed. **Not
  a healthy result** — it means "not proven healthy," not "healthy with notes."

Useful flags:

- **`--json`** — emit a machine-readable verdict instead of human lines, for an automated gate to
  consume: `{"overall": "ok"|"problems"|"inconclusive"|"unreachable", "checks": [{name, status,
verdict, recommendation}]}` (an informational e2e-mode check also carries `"gating": false`). Per-check `status` is `ok`/`warn`/`inconclusive`; the exit code is
  unchanged. In the `unreachable` case `checks` is a single synthetic `connectivity` entry with
  `status: "inconclusive"` — a gate should key off `overall`, not assume a fixed check list.
- **`--mode general|e2e`** (default `general`) — `general` answers "is this cluster in good shape?":
  every problem anywhere is reported. `e2e` answers "can the E2E suites run here?": the storage
  checks (`longhorn-health`, `longhorn-components`, `stuck-storage-pods`) and `aim-model-health` look only at the test
  namespace, the `e2e-*` namespaces and the platform namespaces (`aiwb`, `airm`, `keycloak`, `seaweedfs-instance`, `minio-tenant-default`), `cilium-mtu` is shown but never gates (its JSON entry
  carries `"gating": false`), and problems elsewhere are printed as informational without setting
  the result. An
  automated E2E dispatch gate should require `--mode e2e` to report HEALTHY (exit 0); INCONCLUSIVE
  is not healthy.
- **`--model NAME`** (repeatable, e2e mode) — a model the run uses, matched by its exact resource name (as a deployment resolves it).
  `aim-profiles` gates only on those models, `aim-model-health` narrows models to them (a model in an `e2e-*` namespace other than the selected `--test-namespace` is a leftover of an earlier run and gates only when named; other namespaced models are scoped by namespace alone), and `aim-ready` requires each of them to be Ready. Without `--model`, every model except leftovers in `e2e-*` namespaces gates in `aim-model-health` (and in `aim-profiles` in general mode, which always gates; in e2e mode `aim-profiles` treats a model lacking a profile as informational only while another catalog model is Ready and has a supported profile), and `aim-ready` passes when any one catalog model is Ready.
- **`--s3-bucket BUCKET`** (default `default-bucket`) — the bucket whose SeaweedFS collection
  `object-store-fill` checks for a writable volume.
- **`--image REF`** (repeatable) — also verify a container image reference resolves as part of the
  run (e.g. an image an install/E2E depends on).
- **`--require-gpu`** — treat zero allocatable GPUs as a problem even on a cluster with no GPU nodes
  (for GPU E2E lanes that must have GPUs).
- **`--test-namespace NS`** — the namespace the E2E tests run in, checked for permission to create
  AIMArtifacts and for a LocalQueue with GPU quota. Default: an existing `e2e-*` namespace
  (preferring one with a LocalQueue); with none, those checks have nothing to check. Point it at the
  lane's real test namespace.
- **`--restart-threshold N`** (default `3`) — `app-pods-stable` warns when a Ready app pod has
  restarted at least N times (the crash-loop/flapping signal) and last died within the past 6 hours;
  restarts older than that (say, from a node outage the pod has since run through) are ignored.
- **`--recent-minutes N`** (default `30`) — the window for "recent": `app-pods-stable` warns on any
  app-pod restart within it, and `app-startup-errors` and `oidc-issuer-consistency` scan that much
  of the product logs (the log is the only evidence of an image overriding a UI's issuer, so
  narrowing the window narrows that detection too). Both flags must be at least 1; the check refuses
  0 or a negative value (exit 2) rather than silently disabling what it tunes.
- **`--trusted-private-host HOST`** (repeatable) — an issuer host you vouch for, so its discovery
  document may be read from a private-range address; without it a host resolving to a non-public
  address is refused.
- **`--cluster-domain DOMAIN`** (default `cluster.local`) — the cluster's DNS domain, so issuer
  and discovery URLs of the form `<service>.<namespace>.svc.DOMAIN` are read through the API-server
  service proxy rather than fetched from the operator's host.
- **`--metrics-backend NS/SVC:PORT`** — the Prometheus/Mimir query service `gpu-metrics` queries to
  positively confirm GPU series arrive (e.g. `otel-lgtm-stack/lgtm-stack:9090`). Auto-discovered
  from a `:9090` service in a metrics namespace if omitted.
- **`--gateway-url URL`** — the inference gateway `gateway-auth` probes. Auto-discovered from the
  `aiwb-ui` deployment's `AI_GATEWAY_URL` if omitted.
- **`--gateway-key KEY`** — a valid inference API key. Optional: the invalid-key probe (a healthy
  gateway answers 401) needs no credentials, so `gateway-auth` runs without this. Supplying it adds
  the authorised half — a valid key must return 200/203.
- **`--e2e-oidc-user USER`** — the OIDC username the E2E runner authenticates as. When given,
  `eai-e2e-rbac` requires the role binding to name exactly this user; omit it to accept any OIDC user.
- **`--spec-revision REV --core-repo PATH`** — opt in to `release-spec-skew`: compare the deployed
  AIRM/AIWB service-image commits with the core revision whose specs will run. `PATH` must be a core
  Git checkout containing `REV` and the deployed release tags/commits (default: current directory).
  A deployed ancestor/diverged commit is a `WARN`; a tag or revision that cannot resolve is
  `INCONCLUSIVE`, never a false pass. Callers should pass the exact checked-out SHA, not a moving ref.

To run one focused check without the full cluster suite:

```bash
# One image reference:
bash scripts/environment-check.sh image-exists <ref> [--json]

# Deployed products against one spec revision:
bash scripts/environment-check.sh \
  release-spec-skew <spec-revision> --core-repo <path> [--json]
```

A definitive image "manifest not found" is a `WARN`; an auth-required/unreadable registry is
`INCONCLUSIVE` (a private repo is not proof the tag is missing), never a false failure. The focused
skew command returns the same per-check JSON and exit-code contract as its opt-in full-run form.

## Step 2 — Report the verdict honestly

Pick the honest outcome — the anti-silence rule is the whole point of this skill:

- **HEALTHY (0):** report that no known problems were found, and name what was checked so the
  operator knows the scope (it does not prove the cluster is perfect, only that these known bugs
  are absent).
- **PROBLEMS FOUND (1):** for each `WARN`, relay the plain-English problem, why it matters, and the
  fix the line names. Report **every** problem, not just the first — several can be present at once.
- **CANNOT REACH CLUSTER (2):** say the cluster is unreachable and help fix access. Never soften
  this into a healthy or empty result.
- **INCONCLUSIVE (3):** say plainly that the cluster is **not proven healthy**, then list every
  check that could not confirm its state. One line each, naming the check, why it could not read
  (usually a CRD is absent because the component isn't installed, or RBAC/an API error blocked a
  read) and what would let it run, such as the missing permission. Do not present an inconclusive
  run as a pass.

  **A count is not a report.** "6 of 28 checks were inconclusive" tells the operator nothing they
  can act on, and listing the checks that passed is not a substitute for listing the ones that did
  not. The checks that could not run are the finding here: an operator who cannot see which ones
  they were, and why, has no way to tell an expected gap from a missing permission that is hiding a
  real fault. Name them even when there are many, and even when each looks individually harmless.

To understand what each check reads, what a finding means, and which findings are expected before
vs. after install, read `references/checks.md`.

- `references/checks.md` — What each check inspects, the known bug behind it, and how to interpret
  OK / WARN / INCONCLUSIVE per check. **Load when** interpreting a finding or deciding whether an
  inconclusive result is expected for the cluster's install state.

## Intended consumers (other lanes)

This skill is meant to be the single home for these check implementations, so other work can consume
it rather than duplicate the logic. These consumers are being built separately (they do not live in
this repo yet):

- An E2E-readiness gate is intended to run the full check with `--json` (plus `--image` /
  `--require-gpu` as needed) and read the single `overall` verdict.
- A post-deploy (bloom) gate is intended to run the full check with `--json` right after a release
  lands. The deployed-release checks (`ui-auth-endpoints`, `oidc-issuer-consistency`,
  `app-startup-errors`, `aim-ready`, `aim-model-health`, `argocd-apps`) catch a release that is
  Synced/Healthy with every pod Running but cannot sign anyone in, and `deployed-versions` records
  which images the run tested.
- A pre-bloom precheck is intended to reuse `image-exists` (subcommand or `image_exists()` import)
  instead of keeping its own registry-inspection copy.
- A deployment-triggered E2E gate can pass its checked-out core SHA as `--spec-revision` and stage a
  core checkout at `--core-repo`. That opts into a pre-assertion check that the deployed AIRM/AIWB
  builds contain the behavior described by those specs.

Keep the `--json` shape, `image-exists` interface, and `release-spec-skew` check name stable — they
are the contract those consumers will depend on.

## Tests

The stable test command runs two hermetic layers with no cluster or registry access:

```bash
bash .agents/tests/cluster-environment-check/run-tests.sh
```

- Python regression tests stub the checker’s one side-effecting primitive (`run()`) and pin every
  false-pass edge.
- Robot Framework executable specifications in `specs/environment_check.robot` invoke the shipped
  shell/Python CLI as a black box. A fixture-backed `kubectl` executable supplies complete cluster
  transcripts, so the tests verify real JSON, exit-code, and operator-visible verdict behavior
  without importing the implementation. The `release-spec-skew` scenarios additionally build a
  throwaway Git repository so the deployed-vs-spec ancestry is exercised end to end.

Run both layers after any change to the checker. The generic `agent-skill-tests` CI job discovers
this command automatically.
