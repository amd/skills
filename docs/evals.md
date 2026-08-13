# Evals

Structural validation proves a skill is *well-formed*; evals prove it *works*.
They answer two questions, in order:

1. **Routing** — with the published bundle installed side by side, does the agent pick yours? This is where most skills fail, and you cannot test it alone: a skill tested by itself will happily answer prompts that belong to its neighbour.
2. **Behavior** — once yours is picked, does it do the job?

You write one file, `skills/<your-skill>/evals/evals.json`, and both questions
are graded from it. Copy [`eval/TEMPLATE.json`](../eval/TEMPLATE.json) to start.

## The file

One `evaluations` array. Each entry is a prompt plus `skill_should_trigger`:
yes if your skill should wake up for it, no if nothing should.

```json
{
  "evaluations": [
    {
      "id": "epyc-vllm-zentorch",
      "skill_should_trigger": true,
      "prompt": "Serve Llama 3.1 8B on this box with vLLM and zentorch."
    },
    {
      "id": "vllm-on-nvidia",
      "skill_should_trigger": false,
      "prompt": "Serve Llama 3.1 70B with vLLM on my NVIDIA H100 cluster."
    }
  ]
}
```

No evaluation names a skill: the folder says which skill this is, and
`skill_should_trigger` refers to it. The flag is required rather than defaulted,
so the routing expectation is always written down.

A `false` evaluation is exactly what you see above — an id, a prompt, and the
flag, plus an optional `note`. No skill loads for it, so there is no behavior
phase; writing `unexpected_behavior` there is an error, not a no-op. If a prompt
should trigger *someone else's* skill, put it in that skill's dataset as `true`
rather than in yours as `false`. Routing installs the bundle all at once, so it
is the same assertion either way.

## Grading behavior

Only a `true` evaluation can grade behavior. All four fields are optional, and
any one of them promotes the case from routing-only to behavior-graded.

| Field | Graded by | Use it for |
| --- | --- | --- |
| `logs_contain` | exact substring | a literal that must appear: a script name, a flag, a pinned image tag |
| `files_exist` | the filesystem | an artifact the run must produce |
| `expected_behavior` | an LLM judge | a step the agent must take, in plain language |
| `unexpected_behavior` | an LLM judge | the mistake this skill exists to prevent |

Prefer the deterministic two: they are instant and free, where a judged
expectation costs a second agent call.

Two things you never write. **Do not assert your own skill's name** in
`logs_contain` — routing grades that properly, and a substring match only proves
the skill was staged. **Do not label a prompt** as positive, near-miss, or
unrelated: that is derived from the flag and the file it lives in.

The full field reference is
[`eval/schema/evals.schema.json`](../eval/schema/evals.schema.json), enforced by
`python eval/run_evals.py --validate`.

## How much is enough

| Tier | Required when | What it costs you |
| --- | --- | --- |
| **0** | always | 3 evaluations with `skill_should_trigger: true`, 2 with `false`. CI rejects a skill without them. |
| **1** | your skill can be exercised on a generic runner | one evaluation with an `expected_behavior` / `unexpected_behavior` / `logs_contain` / `files_exist` expectation |
| **2** | your skill needs hardware or a live service | an `evals/machine.yml` naming the kind of machine |

Tier 0 buys more than it looks like. Routing pools every published skill's cases
into one run, so your five prompts become negative cases for every other skill,
and theirs become negatives for yours — a routing matrix nobody had to
coordinate.

Spend your effort on the near misses. Positive prompts mostly pass; the prompts
that find real problems sit just outside your scope — the wrong vendor's
hardware, the adjacent skill's job, your vocabulary used to mean something else.

**Before your skill is published** it is not in the bundle, so it cannot win a
prompt and its positives are not scored. Your dataset is still validated, your
behavior cases still run, and your `false` prompts still run as near misses
against the published skills. Positives start being scored the day the skill
joins the bundle, with no edit to them. Routing is measured against the set of
skills a user actually installs, so an unpublished skill would otherwise change
everyone else's score while shipping to nobody.

## When JSON isn't enough

Two optional files sit beside the dataset.

**`evals/machine.yml`** declares where the behavior cases run. Skip it unless you
need something other than the default runners on Linux and Windows. Both keys
are optional:

```yaml
runner_type: instinct    # `default` (assumed) or `instinct`
os: [Linux]              # defaults to every platform that runner type has
```

Most skills that need this file need only `os: [Linux]`, to drop a Windows leg
that would just exercise the failure path of Linux-only tooling.

Name the kind of machine, not its consequences. `runner_type: instinct` is the
whole declaration: the runner labels, the Linux-only constraint, the
`enable_mi_ci` pull-request label that rations that scarce pool, and the scoped
credentials all follow from it. A missing gate label is a warning rather than a
failure, so a PR is never blocked by a test it did not request. Adding a new
class of hardware means one entry in `RUNNER_TYPES` in
[`eval/datasets.py`](../eval/datasets.py), not teaching every skill the same
set of labels.

**`evals/hooks.py`** is the escape hatch for setup JSON cannot express — cloning
a repo, tearing down a container, running an external scoring script. Every
function is optional:

```python
def setup_session(cache_dir): ...     # once per run; returns {name: value} for {placeholders} in prompts
def setup(workspace, case, ctx): ...  # before each case; may return more placeholders
def teardown(workspace, case, ctx): ...
def check(run, case, ctx): ...        # after each case; raise AssertionError to fail it
```

Keep prompts and expectations in the dataset even when you use hooks, so what is
being asserted stays readable without opening Python. See
[`skills/tracelens-analysis-orchestrator/evals/hooks.py`](../skills/tracelens-analysis-orchestrator/evals/hooks.py)
for the involved case and
[`skills/serving-llms-on-instinct/evals/hooks.py`](../skills/serving-llms-on-instinct/evals/hooks.py)
for the simple one.

## Running them

```bash
python eval/run_evals.py --validate              # structure only: no agent, no tokens, instant
python eval/run_evals.py --skill <your-skill>    # routing and behavior for your skill
python eval/run_evals.py --mode routing          # the published bundle
python eval/run_evals.py --only <case-id> --keep-logs logs   # one case, keeping the transcript
```

Everything but `--validate` needs the `claude` CLI authenticated, plus whatever
your own cases need. No `pip install`: the runner is standard library only.

In CI, the `evals` workflow runs routing when a change can move a routing
decision (a published description, any dataset, or the bundle itself), and runs
behavior for the skills a change touches.
