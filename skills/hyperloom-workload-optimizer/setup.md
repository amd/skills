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

Find the latest `hyperloom_inference_optimizer-*-py3-none-any.whl` from
[AMD-AGI/Hyperloom releases](https://github.com/AMD-AGI/Hyperloom/releases).
Do not pin a version in this skill — resolve the newest published wheel at
install time. The repo is public, so the default path needs only `curl` and
`python3` (no `gh` / `jq`):

```bash
cd "$INSTALL_DIR"   # the directory confirmed in Step 0
wheel_url="$(curl -fsSL "https://api.github.com/repos/AMD-AGI/Hyperloom/releases?per_page=20" \
  | python3 -c 'import sys, json
for rel in json.load(sys.stdin):
    for asset in rel.get("assets", []):
        if asset["name"].startswith("hyperloom_inference_optimizer-"):
            print(asset["browser_download_url"]); sys.exit(0)
sys.exit("no matching wheel asset found")')"
python3 -m pip install "$wheel_url" --target .
```

Optional shortcut when `gh` and `jq` are installed:

```bash
wheel_url="$(gh api repos/AMD-AGI/Hyperloom/releases?per_page=20 \
  | jq -r '[.[] | .assets[]? | select(.name | startswith("hyperloom_inference_optimizer-")) | .browser_download_url][0]')"
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

That skill writes `.env`, collects LLM settings, sets `USER_DATA_PATH`, records
`HYPERLOOM_RUN_MODE` (`baremetal` or `docker`), sets `HYPERLOOM_SKILL_PATH`,
and on bare metal runs the host setup backend (`install_baremetal.sh`).

**Fallback (inline):** when `/hyperloom-setup` is unavailable, write `.env` with
placeholders and ask the user to edit secrets locally (never paste API keys into
chat):

`<INSTALL_DIR>` below is the directory confirmed in Step 0 (the wheel and `.env`
location). `USER_DATA_PATH` may live under `<INSTALL_DIR>` or point elsewhere
(e.g. shared storage) for run state.

```bash
cat > .env <<'EOF'
ANTHROPIC_API_KEY=<PLEASE_FILL_IN>
ANTHROPIC_BASE_URL=https://api.anthropic.com
CLAUDE_MODEL=claude-opus-4-8
USER_DATA_PATH=<INSTALL_DIR>/session
HYPERLOOM_RUN_MODE=docker
HYPERLOOM_SKILL_PATH=<INSTALL_DIR>/hyperloom/inference_optimizer/SKILL.md
EOF
```

Set `HYPERLOOM_SKILL_PATH` to the packaged optimizer skill:

- wheel: `<INSTALL_DIR>/hyperloom/inference_optimizer/SKILL.md`
- source: `<INSTALL_DIR>/src/hyperloom/inference_optimizer/SKILL.md`

Stop if any required secret still equals `<PLEASE_FILL_IN>`.

## Step 3 — Return to SKILL.md

Bootstrap is complete. Go back to [SKILL.md](SKILL.md) and continue with Phase 1
(environment prep), Phase 2 (workload intake), then Phase 3 (install, preflight,
launch, monitor).

Read `@${HYPERLOOM_SKILL_PATH}` only for edge cases the catalog skill does not
cover: multi-node, `--framework atom`, critic/robustness backend selection,
aiter cache topology, and the full failure matrix.
