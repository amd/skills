# AMD Skills Walkthroughs: `rocm-doctor`

The goal of this skill is to teach your AI agent to diagnose why ROCm, the HIP
SDK, PyTorch, or llama.cpp is broken on an AMD GPU, then apply a low-risk fix
(with your consent) or hand you the exact next step. It drives the `rocm` CLI
(`rocm examine` / `rocm diagnose` / `rocm fix`) and installs that CLI for you if
it's missing.


## Step 1 - Understanding which skills are available

* Run `claude "Which skills can you see?" --model sonnet`. You should see a list
  of skills that does **not** include anything about diagnosing ROCm / AMD GPU
  failures.
* Make sure there is no `AGENTS.md` file in your local folder.


## Step 2 - Enabling claude to see `rocm-doctor`

* Install the skill with the [`skills` CLI](https://github.com/vercel-labs/skills):

```bash
npx skills add amd/skills --skill rocm-doctor --agent claude-code
```

* Run `claude "Which skills can you see?" --model sonnet`. You should see a list
  of skills that now includes `rocm-doctor`.


## Step 3 - Running the skill

Open Claude and paste a real symptom, for example:

```
torch.cuda.is_available() returns False on my AMD GPU and I get
"hipErrorNoBinaryForGpu" when I run my script
```

Claude should:

1. **Ensure the `rocm` CLI is present** — run `rocm --version`, and if it's
   missing, ask for your consent to install it (Phase 0), then re-check.
2. **Diagnose** — run `rocm diagnose --symptom "<your error>" --json` and read
   the ranked matches from the closed catalog (each with a score, evidence, and a
   fix).
3. **Propose the fix** — show you the top match's title, the evidence, the plan,
   and the `verify` command *before* changing anything.
4. **Apply with consent** — for an auto-applicable fix, run `rocm fix <id>`
   (or `--dry-run` first). Risky fixes are printed for you to run yourself.

You stay in control: nothing is mutated without an explicit OK, and if the
symptom doesn't match a known failure mode, Claude routes you to the right
upstream tracker instead of guessing.


## Step 4 - (Optional) Just inspect the host

If you only want the machine state (GPU, driver, ROCm install, groups, framework)
without a diagnosis, ask:

```
Examine my ROCm setup and tell me if anything looks wrong
```

Claude runs `rocm examine` and reads the `status` verdict
(`ok` / `no-amd-gpu` / `wsl` / `unsupported-os` / `degraded`).


## Step 5 - (Optional) Try to get things done without AMD Skills

Remove the added skill from `.claude/skills/` and rerun the experiment above.
Without the skill you should see high variance:

* The model inventing fixes (e.g. blindly setting `HSA_OVERRIDE_GFX_VERSION`)
  that don't match your actual GPU.
* The model producing a generic knowledge article instead of probing your host.
* The model suggesting risky, unverified `sudo` commands with no consent step.
