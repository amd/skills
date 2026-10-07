# OrchestrAI CI security and rollout

Privileged evaluation jobs require reviewed controller code, protected GitHub
Environments and administrator setup. The offline security preview is usable
before privileged CI is enabled. It does not allocate a machine, call a model,
or prove that a skill passes.

## What changed

| Concern | Workflow protection | Remaining boundary |
| --- | --- | --- |
| PR code could run with Portal credentials | Privileged jobs check out controller code from the exact base commit, or an administrator-selected reviewed commit. Candidate skills are separate test input. Routing and OrchestrAI require the `skills-ci` Environment. | An administrator must protect the Environment and move credentials into it. Workflow YAML can itself be changed by a PR; these checks are not a sandbox. |
| Linux installation path could contain shell syntax or traversal | Script paths have a strict character allowlist and reject empty, `.` and `..` segments. The graphics installer must request a reboot. | Only reviewed controller configuration may choose provisioning scripts. |
| Dataset-controlled matrix values could reach shell commands or choose other infrastructure | Only valid skill names and Linux/Windows entries enter the default matrix. Verdict parameters use quoted environment variables. Instinct runner labels and Environment are fixed. | Dataset contents are still untrusted input to an agent and require review before privileged execution. |
| Encoded secrets and private GitHub paths could appear in published logs | Known-value masking covers case changes, JSON escaping, common percent encoding and Base64 forms. GitHub URL exceptions are restricted to explicitly listed public projects. | No finite redactor can prevent arbitrary deliberate exfiltration. Isolation and short-lived, scoped credentials remain necessary. |
| Fork tests could execute on Instinct with OIDC privileges | Privileged jobs reject fork PRs. Only the protected Instinct job gets `id-token: write`; all other jobs have no OIDC permission. | Administrators must protect `behavioral-instinct`, its federation rules and its runner pool. |

Reports and verdict processing run on GitHub-hosted runners, without an internal
network runner or model/Portal credentials. Routing still needs a network path to
the model gateway. Approval covers **both the workflow and the exact candidate
commit**, including skill instructions, datasets, validators and dependencies.
The agent intentionally executes candidate content: a trusted controller does
not make that content safe.

## Try it without credentials

The **OrchestrAI security preview** workflow runs on pull requests touching these
files. It runs the regression tests and publishes a synthetic Linux/Windows mock
report plus redacted fixture output. It uses `ubuntu-latest`, a read-only GitHub
token, no OIDC permission, no privileged Environment and no service secrets.

Locally:

```bash
python3 -m unittest discover -s .github/scripts -p 'test_orchestrai*.py'
python3 .github/scripts/orchestrai_security_preview.py --output /tmp/skills-security-preview
```

The synthetic plan uses an example `amd/skills` source revision; it does not
clone or validate that commit. Plans stay in a temporary directory and are not
uploaded. A passing preview is **not** a passing live evaluation or an assertion
that repository/organization security settings have been configured.

Live behavioral plans target public `amd/skills` at the exact candidate commit.
The preview uses synthetic source metadata instead and is not a live checkout
test. Fork PRs cannot access the privileged jobs; maintainers must review the
candidate and use a same-repository branch for privileged testing.

## Administrator setup before live CI

1. Keep `ORCHESTRAI_PRIVILEGED_CI_ENABLED` unset or `false` during review. Live
   privileged jobs are skipped, and the final evaluation gate fails explicitly
   when graded work was requested. Structural/reference checks and the offline
   preview remain available. This switch prevents accidental activation; it is
   not an access-control boundary against modified workflow YAML.
2. Establish a trusted controller revision. By default, a PR uses its exact base
   SHA and a default-branch dispatch uses its triggering SHA. If that revision
   lacks the controller, the run fails rather than executing PR controller code.
   For bootstrap, an administrator may set `ORCHESTRAI_TRUSTED_CONTROLLER_SHA`
   to a **fully reviewed 40-character commit SHA** containing these changes.
   Clear the override after the reviewed controller is on the protected default
   branch. Do not set it automatically to an incoming PR head.
3. Configure **`skills-ci`** and **`behavioral-instinct`** Environments with named
   required reviewers, prevention of self-review, administrator bypass disabled,
   and restricted deployment branches. Choose reviewer identities deliberately.
   Confirm that the policy permits intended default-branch and same-repository
   PR deployment refs without allowing arbitrary branches. The hosted policy job
   reads these settings and fails closed if it cannot verify them. GitHub plan
   and repository visibility can affect which protection features are available.
4. Re-enter the relevant credential values in `skills-ci` Environment secrets:
   `ORCHESTRAI_PORTAL_URL`, `ORCHESTRAI_USER`, `ORCHESTRAI_PASSWORD`,
   `ORCHESTRAI_SPACE`, `ORCHESTRAI_DEVICE_TAGS`,
   `ORCHESTRAI_LINUX_DRIVER_SOURCES_JSON`, `ORCHESTRAI_WINDOWS_DRIVER_SOURCE`,
   and the routing key `ORCHESTR_API_KEY`. Remove repository-level duplicates
   and organization-secret access paths that bypass approval. GitHub does not
   let automation read existing secret values back for migration. Do not paste
   credential values into PRs, workflow inputs or logs. Node-side LLM credential
   binding remains in the shared pipeline; this change does not modify it.
5. Have runner administrators verify isolation, ephemeral replacement/cleaning,
   network reachability and runner-group repository/workflow restrictions for
   control and Instinct pools. Restrict Instinct federation to the intended
   repository, protected Environment and approved workflow claims. A label or
   an Environment gate alone does not secure a self-hosted machine. Record this
   operational review separately from the workflow's unit-test results.
6. After those controls are verified, explicitly
   set `ORCHESTRAI_PRIVILEGED_CI_ENABLED=true` and approve a narrowly scoped live
   smoke. Do not treat the offline preview as a substitute for this review.

No Environment, approver, secret value, federation rule or runner-group setting
is created or migrated by this implementation. GitHub's server-side protections,
secret scoping and runner policies enforce access; the YAML checks only provide
additional validation and clear diagnostics.

The Environment and trusted-controller checks run on a GitHub-hosted runner.
Any organization network-access policy must permit that runner to read the
repository and its Environment settings. Do not add a shared self-hosted fallback
for untrusted PR checks without reviewing isolation and runner access first.

## Unchanged provisioning and testing

The Skillscope release pin, prompts, expectations and verdict semantics remain
unchanged. Windows still enables test signing, reboots, installs the driver and
reboots again. The deferred Windows GPU-readiness check remains absent. A green
fallback-behavior test still does not prove that a GPU workload ran.

The stdout/stderr publication path still retains useful ordinary diagnostics
and applies redaction before upload and again before display. It does not expose
raw control-plane logs. Tests use synthetic values only; they demonstrate the
specific supported encodings, not a guarantee against novel encodings or a
malicious agent intentionally leaking data.
