# OrchestrAI behavioral integration

Selected Strix Linux/Windows behavioral tests execute through OrchestrAI.
Skillscope remains the selector and grader; the skills and evaluation datasets
are unchanged by this integration.

## Workflow contract

| Area | Behavior |
| --- | --- |
| Pull requests | Existing opened, synchronize, reopened and labeled events, changed-file selection and concurrency. Any label event can restart a run, as before. |
| Manual runs | Existing mode, skill and case selection; explicit requests retain Skillscope's hardware-gate bypass. |
| Schedule | Monday at 12:00 UTC, references-only; no agents or hardware. |
| Strix tests | One OrchestrAI controller per selected OS; one verdict per selected skill/OS. No additional behavioral label. |
| Instinct | Existing selected matrix, `enable_mi_ci` PR gate, Environment and OIDC credentials. Independent of OrchestrAI. |
| Routing | Existing routing room, grader and model credentials. Independent of OrchestrAI. |
| Final check | `evals / results` requires every requested graded leg to pass. Held-back Instinct cases warn. External references remain advisory. |

The workflow expands the Skillscope release's reusable jobs locally because it
does not expose an OrchestrAI execution backend. Skillscope remains pinned to
release `v0.1.3`.

## Exact candidate, no mirror

Discovery checks out `github.event.pull_request.head.sha` for a pull request,
or `github.sha` for other events. The plan passes the same SHA to the adapter,
which clones public `amd/skills` and verifies that exact commit before testing.
The branch name locates the source; it is not allowed to substitute a newer tip
for the triggering commit. A missing commit fails instead of testing other code.
There is no mirror-comparison job, public-source mapping file or fixed skills
snapshot.

Fork pull requests do not receive repository secrets. This workflow does not
change GitHub's credential withholding or introduce a privileged fork trigger.

## Provisioning and reports

Linux driver-source mapping and Windows driver installation/reboots use the
existing provisioning contract. Windows enables test signing, reboots, installs
the driver and reboots before testing. See [the provisioning and reporting
details](evals.md#windows-provisioning).

Received test stdout/stderr is redacted before upload and revalidated before
display in GitHub. Grader summaries distinguish behavioral failures from known
grading errors. Raw control-plane consoles are not published. Missing results
and incomplete grading do not pass the final check.

Installer completion does not prove GPU workload coverage. The additional
GPU-readiness check remains deferred; coverage depends on each skill's case.

## Configuration and scope

Keep `ORCHESTRAI_CONTROL_RUNNER` set to the runner pool with access to the Portal
and model gateway, such as `arc-lnx-pub`. Discovery and reporting use
GitHub-hosted runners. No separate wait-runner setting is used.

Routing and the OrchestrAI controller use the existing Actions secrets available
to those jobs. They do not declare a new `skills-ci` Environment. Secrets stored
only in that Environment will not be available. This change does not move
credentials or modify repository settings, runner groups, branch rules or
federation rules. Instinct retains its selected Environment.

Additional authorization hardening is a separate change: reviewed-controller
checkout, new Environment approval checks and additional PR/branch restrictions
are not included here. Candidate controller code runs under the existing Actions
authorization model. Runner isolation and credential scope still require
administrator review. Redaction reduces accidental disclosure; it does not
prevent deliberate exfiltration by arbitrary candidate code.

## Credential-free validation

Run the same OrchestrAI regressions used by CI:

```bash
python3 -m unittest discover -s .github/scripts -p 'test_orchestrai*.py'
python3 .github/scripts/orchestrai_preview.py --output /tmp/skills-integration-preview
```

The [integration preview workflow](../.github/workflows/orchestrai-preview.yml)
uses synthetic inputs and no service credentials, machines or model calls.
A green preview is not a live skill-quality result. Likewise, manual mock or
one-OS runs cannot claim complete Linux/Windows hardware coverage.
