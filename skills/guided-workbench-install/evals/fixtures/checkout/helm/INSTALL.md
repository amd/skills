<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Installing AI Workbench

All commands run from inside `helm/`.

## Prerequisites

- A reachable Kubernetes cluster with a default StorageClass
- `kubectl` and `helm` v3 on PATH
- For a GPU cluster: the AMD GPU Operator, the Device Metrics Exporter, and
  nodes advertising `amd.com/gpu` capacity

## Deployment modes

Decide these before installing anything:

- **Combined** installs both AIRM and AIWB. **Standalone** installs AIWB alone.
- **Cluster class**: a real GPU cluster, or a local development cluster where
  the GPU tier is skipped and the StorageClass is usually `standard`.
- **Database**: chart-managed CNPG (the default), or an external PostgreSQL.
- **Secrets**: OpenBao with External Secrets, or secrets created by hand.

## Step 1: Install cluster dependencies

Envoy Gateway, which the serving routes depend on:

```bash
helm upgrade --install eg oci://docker.io/envoyproxy/gateway-helm \
  --version v1.9.1 \
  -n envoy-gateway-system --create-namespace \
  --set crds.enabled=false
```

A GPU cluster also needs the AMD GPU Operator, the Device Metrics Exporter,
Kaiwo and the AIM Engine CRDs. A local development cluster can skip that tier.

## Step 2: Create secrets

Create the image pull secret and the application secrets in both namespaces
before the charts that consume them are installed. With OpenBao and External
Secrets this is the `ClusterSecretStore` the charts read through.

## Step 3: Install infrastructure charts

External Secrets first:

```bash
helm install airm-infra-external-secrets ./infrastructure/airm-external-secrets \
  -n airm --create-namespace
```

```bash
helm install aiwb-infra-external-secrets ./infrastructure/aiwb-external-secrets \
  -n aiwb --create-namespace
```

Then the databases. Skip these two for an external PostgreSQL:

```bash
helm install airm-infra-cnpg ./infrastructure/airm-cnpg \
  -n airm --create-namespace
```

```bash
helm install aiwb-infra-cnpg ./infrastructure/aiwb-cnpg \
  -n aiwb --create-namespace
```

Then the message broker:

```bash
helm install airm-infra-rabbitmq ./infrastructure/airm-rabbitmq \
  -n airm --create-namespace
```

## Step 4: Install application charts

AIRM first, then AIWB. For a standalone install, skip the AIRM chart:

```bash
helm install airm ./airm \
  -n airm --create-namespace \
  --set airm-api.airm.appDomain=<YOUR-DOMAIN> \
  --set airm-api.airm.backend.image.tag=${APP_IMAGE_TAG} \
  --set airm-api.airm.frontend.image.tag=${APP_IMAGE_TAG} \
  --set airm-agent.airm.image.tag=${APP_IMAGE_TAG}
```

```bash
helm install aiwb ./aiwb \
  -n aiwb --create-namespace \
  --set appDomain=<YOUR-DOMAIN> \
  --set backend.image.tag=${APP_IMAGE_TAG} \
  --set frontend.image.tag=${APP_IMAGE_TAG}
```

Then the AI Gateway and the discovery controller that reconciles serving
routes:

```bash
helm upgrade -i aieg-crd oci://docker.io/envoyproxy/ai-gateway-crds-helm \
  --version v1.1.0 \
  -n envoy-ai-gateway-system --create-namespace
```

```bash
helm upgrade -i aieg oci://docker.io/envoyproxy/ai-gateway-helm \
  --version v1.1.0 \
  -n envoy-ai-gateway-system --create-namespace
```

```bash
helm install ai-gateway-discovery ./ai-gateway-discovery \
  -n envoy-ai-gateway-system --create-namespace \
  --set controller.gateway.routeHostname=ai.<YOUR-DOMAIN>
```

## Step 5: Onboard a cluster

Run the onboarding Job that registers the cluster with AIRM, and wait for it to
succeed. A combined install is not finished until it has.

## Verification

Confirm every Deployment has its replicas ready, the CNPG clusters report a
healthy phase, the RabbitMQ cluster reports all replicas ready, the HTTPRoutes
are Accepted, and the onboarding Job succeeded. Health that cannot be confirmed
is not health: report it as inconclusive rather than as success.
