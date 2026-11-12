<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Evaluation fixtures

`checkout/` is staged into the workspace of a behavioral case, so what is in it
is what the agent sees. Nothing in here is shipped to a user and nothing in
here is a document about this skill.

`checkout/helm/INSTALL.md` is a shortened stand-in for the product's own
runbook: the same helm commands, in the same order, with the prose, the values
tables and the troubleshooting sections left out. `tests/check-fixture-runbook.py`
fails if those commands stop matching `helm/INSTALL.md`.

The note explaining that lives here rather than in the staged file itself. A
case is graded on following the runbook it is given, so a staged document that
tells the agent not to install from it grades the skill against an instruction
the evaluation then penalises it for obeying.

The chart directories beside it carry the real `Chart.yaml` files, because the
skill checks for them before it starts and a stub would not survive a rename.

The cluster the charts install onto is not here. It cannot be a file, so
`evals/hooks.py` builds it.
