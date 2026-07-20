# Examples

## What the user says

**Full replay + analysis**

> Replay and analyze my HRR archive at `/data/crash/capture.hrr/pid-1842`

**Archive summary only**

> Summarize this HRR capture: `capture.hrr/pid-1842`

**Existing log**

> Analyze this HRR replay log: `replay.log` (archive is `capture.hrr/pid-1842`)

The user does not mention scripts, `HRR_PLAYBACK`, or GPU numbers.

## What the agent does

1. Finds `hrr-playback` on `PATH` or `/opt/rocm/bin/hrr-playback`
2. If missing, asks: *"Where is hrr-playback installed?"*
3. Runs `run_hrr_replay.sh --archive ... --analyze`
4. Presents the finding

## If hrr-playback is not in a standard location

User answers: *"It's in `/opt/amd-hrr/bin/hrr-playback`"*

Agent sets `HRR_PLAYBACK=/opt/amd-hrr/bin/hrr-playback` for that session only and re-runs.

If a `lib/` directory sits beside `bin/`, the runner picks it up automatically.
