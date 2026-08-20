# Hyperloom Workspace Bootstrap

Use this file from [SKILL.md](SKILL.md) Phase 0. The optimizer runtime ships in the
Hyperloom Python wheel; this skill does not bundle it.

Steps 0–2 only prepare the workspace. Return to [SKILL.md](SKILL.md) and continue
with Phase 1 (environment prep), Phase 2 (workload intake), then Phase 3
(install, launch, monitor).

## Step 0 — Confirm the install directory

The wheel installs into a target directory (`pip install --target <dir>`) that
also holds `.env` and runtime artifacts. Do not silently use the current
directory. Show the resolved `pwd` and confirm it with the user, or let them
pick another dedicated path. `cd` into the chosen directory before installing.

## Step 1 — Install the Hyperloom wheel

Skip when `hyperloom/` already exists in the directory (wheel layout) or
`src/hyperloom/` exists (source checkout).

The runtime is published to PyPI as `hyperloom-inference-optimizer`. List the
releases, tell the user the newest one, and ask whether to install it or a
version they name.

List with `--pre` so prereleases are visible, and install an exact `==` version
so a later bootstrap installs the same runtime.

```bash
cd "$INSTALL_DIR"   # the directory confirmed in Step 0
pip index versions hyperloom-inference-optimizer --pre
pip install hyperloom-inference-optimizer==<version the user approved> --target .
```

After install, confirm:

- `hyperloom/inference_optimizer/assets/install.sh` exists
- `.cursor/skills/hyperloom-custom-advanced/SKILL.md` exists (or `.claude/` /
  `.agents/` equivalent)

Restart the agent if the new skills are not visible.

## Step 2 — Credentials and run mode

**Preferred:** run the bundled setup skill installed by the wheel:

- Cursor / Codex: `$hyperloom-setup` or load `hyperloom-setup` from
  `.cursor/skills/hyperloom-setup/SKILL.md`
- Claude Code: `/hyperloom-setup`

That skill owns the credential questions, the run-mode question, and the bare-metal
host setup backend (`install_baremetal.sh`). It is the source of truth for which
LLM providers are accepted and which variable names they use, so do not
reimplement it here or guess variable names from memory.

If it is not available, restart the agent and re-check that
`.cursor/skills/hyperloom-setup/SKILL.md` (or the `.claude/` / `.agents/`
equivalent) exists. If it is still missing, the wheel install is incomplete —
stop and report that, rather than hand-writing `.env`.

When it returns, verify `.env` before continuing. It must define
`HYPERLOOM_RUN_MODE` (`baremetal` or `docker`), `USER_DATA_PATH`,
`HYPERLOOM_SKILL_PATH`, and one complete LLM credential set. Check the values are
real, not placeholders. `HYPERLOOM_SKILL_PATH` must point at the packaged
optimizer skill that exists on disk — `hyperloom/inference_optimizer/SKILL.md`
for a wheel install, `src/hyperloom/inference_optimizer/SKILL.md` for a source
checkout.

Inspect `.env` with `grep` rather than echoing it, and never print a key value
into chat. Phase 3 loads `.env` with a shell `source`, so any value containing a
space must stay double-quoted — `hyperloom-setup` writes
`ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: ${ANTHROPIC_API_KEY}"` for
exactly that reason. A hand-edited `.env` that drops those quotes makes the launch
shell fail with exit 127.

Do not write your own live probe of the LLM endpoint. Hyperloom's `optimize`
preflight already probes `<base_url>/models` with the operator's auth and custom
headers, and refuses to start on an auth failure, so a wrong key fails there in
the first seconds rather than mid-run. Two of its outcomes are only warnings, and
both must be surfaced to the user instead of scrolled past:

- `gateway has no /models route (HTTP 404/405) ... Proceeding` — the key is
  unverified, not verified. Expected when `ANTHROPIC_BASE_URL` is native
  `https://api.anthropic.com`, which has no `/models` at that path.
- `gateway catalog unreachable ... Proceeding with custom orchestration model
  support enabled` — this is the auth failure downgraded to a warning because a
  custom model id was allowed. A bad key looks like this instead of an error.

## Docker run mode — the Phase 1 contract

When `HYPERLOOM_RUN_MODE=docker`, `hyperloom-custom-advanced` owns the container
steps (image, `docker run` flags, setup inside the container). Follow it rather
than inventing flags. Regardless of how it gets there, all of the following must
hold before Phase 3 launches anything, and every later command in this skill runs
*inside* the container:

- the container is running and you have a shell in it
- `/dev/kfd` and `/dev/dri` are mapped, and `amd-smi` or `rocm-smi` works inside
- the install directory from Step 0 and `USER_DATA_PATH` are mounted at the same
  absolute paths inside the container, so `.env` and run state resolve identically
- `MODEL_PATH` resolves inside the container
- the setup backend has been run inside the container

If any of these is unmet, fix it before Phase 3. Launching with a half-prepared
container produces failures that look like optimizer bugs.

## Step 3 — Return to SKILL.md

Bootstrap is complete. Go back to [SKILL.md](SKILL.md) and continue with Phase 1
(environment prep), Phase 2 (workload intake), then Phase 3 (install, preflight,
launch, monitor).

Read `@${HYPERLOOM_SKILL_PATH}` only for edge cases the catalog skill does not
cover: multi-node, `--framework atom`, critic/robustness backend selection,
aiter cache topology, and the full failure matrix.
