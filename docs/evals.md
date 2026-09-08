# Skill Evaluation

## Testing Pipeline Overview

A skill reaches the catalog after passing three review stages: an eligibility and compliance check, structural screening, and multi-stage agentic testing.

* **Stage 1: Eligibility and compliance** (*maintainer review, on first submission*)
  * Is the skill eligible, and does the submission follow the contribution guide and its recommendations? An AMD-owned source repo, registered in `.github/federation.json`, the skill authored upstream rather than hand-edited here, and the writing guidance in [best-practices.md](best-practices.md) applied. See [CONTRIBUTING.md](../CONTRIBUTING.md).
* **Stage 2: Structural screening** (*CI, on every pull request*)
  * Are the files well-formed? Required files, frontmatter, skill-card sections, eval schema, unique case ids, internal links, and manifests in sync. See [skill-requirements.md](skill-requirements.md).
* **Stage 3: Agentic testing** (*CI, on every pull request*)
  * **Routing Testing**: Does the skill trigger when it should, and stay quiet when it shouldn't? Prompts run with the published skills installed side by side, so a skill only wins the ones it owns. You cannot test this alone: a skill tested by itself will happily answer prompts that belong to its neighbour.
  * **Behavioral Testing**: Once the skill has triggered, does it do the job? The prompt runs to completion with the skill loaded, and what the agent actually did is graded against the expectations in the dataset.

Stages 2 and 3 are run by [skillscope](https://github.com/amd/skillscope), the
harness AMD uses to grade skills wherever they live. This catalog configures it
in [`.github/workflows/evals.yml`](../.github/workflows/evals.yml) — which
skills are published and so compete for a prompt, which runners we own, and
which key pays for a run. The graders themselves are not in this repo, so the
same prompts score the same way in your product repo as they do here.

The rest of this document is the dataset those stages read. You write one file, `evals/evals.json`, inside your skill folder in this catalog. For now `evals/` is the one folder federation does not carry, so unlike the rest of a federated skill the dataset is authored and edited here rather than imported from your repo, and a re-import never overwrites it. Run `skillscope template` for a file to start from.

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
[skillscope's authoring guide](https://github.com/amd/skillscope/blob/main/docs/authoring-evals.md),
enforced by `skillscope structural`.

### Enabling more complex tests

Two optional files sit beside the dataset when JSON is not enough.

**`evals/machine.yml`** — needed only if the default Linux and Windows runners
are wrong for your skill. Both keys are optional:

```yaml
os: [Linux]              # defaults to both platforms this catalog runs
labels: [mi300x, gpu, rocm]   # extra runs-on labels your cases need
```

Name the hardware you need, not the pool that has it. Asking for any label at
all is what makes your behavioral run *scoped*: this catalog sends those legs
to the Instinct pool, holds them behind the `enable_mi_ci` pull-request label
because that hardware is scarce, and pays for them from a separate environment
with its own key. Which pool, which label, and whose key are the repo's
business and live in
[`.github/workflows/evals.yml`](../.github/workflows/evals.yml), so a new skill
that needs a GPU never means editing CI. Most skills that need this file need
only `os: [Linux]`, to drop a Windows leg that would just exercise the failure
path of Linux-only tooling.

**`evals/hooks.py`** — setup a dataset cannot express: cloning a repo, tearing down a container, running an external scoring script. Every function is optional:

```python
def setup_session(cache_dir): ...     # once per run; returns {name: value} for {placeholders} in prompts
def setup(workspace, case, ctx): ...  # before each case; may return more placeholders
def teardown(workspace, case, ctx): ...
def check(run, case, ctx): ...        # after each case; raise AssertionError to fail it
```

Keep prompts and expectations in the dataset even when you use hooks, so what is being asserted stays readable without opening Python. See [`skills/serving-llms-on-instinct/evals/hooks.py`](../skills/serving-llms-on-instinct/evals/hooks.py) for an example.

### Running tests locally

Install the harness once, at the version CI grades this repo with — the `uses:`
ref in [`.github/workflows/evals.yml`](../.github/workflows/evals.yml):

```bash
uv tool install --system-certs git+https://github.com/amd/skillscope@v0.1.1
```

Then, from the repo root:

```bash
./.github/scripts/check.sh                        # structure only: no agent, no tokens, instant
skillscope behavioral --skill <your-skill>         # your skill, end to end
skillscope routing --routing-room all              # every skill in the room together
skillscope routing --routing-room all --only <case-id> --keep-logs logs   # one case, keeping the transcript
```

`check.sh` wraps `skillscope structural` with this catalog's own settings and
adds the manifest checks, so it is the same bar CI holds you to. Everything
else needs the `claude` CLI authenticated, plus whatever your own cases need.
`skillscope --help` is the reference for the rest of the flags.

In CI, routing runs when a change can move a routing decision (a published description or any dataset), and behavioral cases run for the skills a change touches. `skillscope select` is what makes that call, so you can ask it what a change will cost before you push:

```bash
git diff --name-only main HEAD | skillscope select --changed --skills-dir 'skills/*'
```
