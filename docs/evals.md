# Skill Evaluation

## Testing Pipeline Overview

A skill reaches the catalog after passing three review stages: an eligibility and compliance check, structural screening, and multi-stage agentic testing.

* **Stage 1: Eligibility and compliance** (*maintainer review, on first submission*)
  * Is the skill eligible, and does the submission follow the contribution guide and its recommendations? An AMD-owned source repo, registered in `sources.yml`, the skill authored upstream rather than hand-edited here, and the writing guidance in [best-practices.md](best-practices.md) applied. See [CONTRIBUTING.md](../CONTRIBUTING.md).
* **Stage 2: Structural screening** (*CI, on every pull request*)
  * Are the files well-formed? Required files, frontmatter, skill-card sections, eval schema, unique case ids, internal links, and manifests in sync. See [skill-requirements.md](skill-requirements.md).
* **Stage 3: Agentic testing** (*CI, on every pull request*)
  * **Routing Testing**: Does the skill trigger when it should, and stay quiet when it shouldn't? Prompts run with the published bundle installed side by side, so a skill only wins the ones it owns. You cannot test this alone: a skill tested by itself will happily answer prompts that belong to its neighbour.
  * **Behavioral Testing**: Once the skill has triggered, does it do the job? The prompt runs to completion with the skill loaded, and what the agent actually did is graded against the expectations in the dataset.

The rest of this document is the dataset that structural screening and agentic testing read. You write one file, `evals/evals.json`, inside your skill folder. It ships with the skill from your repo and is vendored into the catalog along with everything else. Copy [`eval/TEMPLATE.json`](../eval/TEMPLATE.json) to start.

## What skill owners write

One file, `skills/<your-skill>/evals/evals.json`, holding an `evaluations` array:

```json
{
  "evaluations": [
    {
      "id": "images-cost",
      "skill_should_trigger": true,
      "prompt": "I'm burning too much money on image generation APIs. Generate images on my own machine instead."
    },
    {
      "id": "generate-cat-image",
      "skill_should_trigger": true,
      "prompt": "Learn how to generate images locally, then save an image of a cat to out.png.",
      "expected_behavior": ["Install Lemonade Server if it is not already installed"],
      "unexpected_behavior": ["Reach for a cloud image path instead of local Lemonade"],
      "files_exist": ["AGENTS.md", "out.png"]
    },
    {
      "id": "finetune-on-laptop",
      "skill_should_trigger": false,
      "note": "Local, on-device, and model-shaped, but training is nobody's job here.",
      "prompt": "Fine-tune a small language model on my own dataset using my laptop GPU."
    }
  ]
}
```

Every evaluation is a prompt plus `skill_should_trigger`: `true` if your skill should fire for it, `false` if it shoudn't.

**When a prompt is all you provide, the evaluation only checks whether the skill was triggered.** Understanding whether your skill is being correctly triggered (both in isolation as well as when other skills are present) is essential and cheap to check for.

**When you add expectations, the prompt also runs end to end** and what the agent did is grated (pass fail) based on the generated logs and workspace.

### Requirements

- At least **3** evaluations with `skill_should_trigger: true`
- At least **2** evaluations with `skill_should_trigger: false`
- At least **1** of the `true` evaluations carries `expected_behavior` or
  `unexpected_behavior`, so something beyond triggering is graded

### Extended validation

`evals/evals.json` is the only dataset we require, and the only one this repo runs. A skill may ship a second file beside it, `evals/extended_evals.json`, which runs as part of the product repo for extended validation.

### Evaluation criteria and optional fields

Four optional fields, all arrays, all valid only on a `true` evaluation:

| Field | Graded by | Use it for |
| --- | --- | --- |
| `expected_behavior` | an LLM judge | a step the agent must take, in plain language |
| `unexpected_behavior` | an LLM judge | the mistake this skill exists to prevent |
| `logs_contain` | substring match | a literal that must appear: a script name, a flag, a pinned image tag |
| `files_exist` | the filesystem | an artifact the run must produce |

The bottom two are instant and free where a judged expectation costs a second agent call, so reach for them when the thing you want is literal. Never assert your own skill's name in `logs_contain`; triggering is already graded properly.

A `files_exist` entry matches whole path segments anywhere in the workspace, so `plan.md` is satisfied by `examples/plan.md` and `out/report.md` by `run-1/out/report.md`. Name the artifact rather than the directory you hope the agent picks: where a file lands is usually the agent's call, and a plan written beside the fixture it describes should not fail the run. If the location matters, ask for it in the prompt and grade it with `expected_behavior`.

The full field reference is
[`eval/schema/evals.schema.json`](../eval/schema/evals.schema.json), enforced by
`python eval/run_evals.py --validate`.

### Enabling more complex tests

Two optional files sit beside the dataset when JSON is not enough.

**`evals/machine.yml`** — needed only if the default Linux and Windows runners
are wrong for your skill. Both keys are optional:

```yaml
runner_type: instinct    # `default` (assumed) or `instinct`
os: [Linux]              # defaults to every platform that runner type has
```

Name the kind of machine and the rest follows: `runner_type: instinct` implies the runner labels, the Linux-only constraint, the `enable_mi_ci` pull-request label that rations that scarce pool, and the scoped credentials. Most skills that need this file need only `os: [Linux]`, to drop a Windows leg that would just exercise the failure path of Linux-only tooling.

**`evals/hooks.py`** — setup a dataset cannot express: cloning a repo, tearing down a container, running an external scoring script. Every function is optional:

```python
def setup_session(cache_dir): ...     # once per run; returns {name: value} for {placeholders} in prompts
def setup(workspace, case, ctx): ...  # before each case; may return more placeholders
def teardown(workspace, case, ctx): ...
def check(run, case, ctx): ...        # after each case; raise AssertionError to fail it
```

Keep prompts and expectations in the dataset even when you use hooks, so what is being asserted stays readable without opening Python. See [`skills/serving-llms-on-instinct/evals/hooks.py`](../skills/serving-llms-on-instinct/evals/hooks.py) for a simple example and [`skills/tracelens-analysis-orchestrator/evals/hooks.py`](../skills/tracelens-analysis-orchestrator/evals/hooks.py) for an involved one.

### Running tests locally

```bash
python eval/run_evals.py --validate              # structure only: no agent, no tokens, instant
python eval/run_evals.py --skill <your-skill>    # routing and behavior for your skill
python eval/run_evals.py --mode routing          # the published bundle
python eval/run_evals.py --only <case-id> --keep-logs logs   # one case, keeping the transcript
```

Everything but `--validate` needs the `claude` CLI authenticated, plus whatever your own cases need. No `pip install`: the runner is standard library only.

In CI, the `evals` workflow runs routing when a change can move a routing decision (a published description, any dataset, or the bundle itself), and runs behavior for the skills a change touches.
