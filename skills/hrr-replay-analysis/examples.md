# Examples

## What the user says

**Full replay + analysis**

> Replay and analyze my HRR archive at `/data/crash/capture.hrr/pid-1842`

**Archive summary only**

> Summarize this HRR capture: `capture.hrr`

**Existing log**

> Analyze this HRR replay log: `replay.log` (archive is `capture.hrr/pid-1842`)

The user does not mention scripts, `HRR_PLAYBACK`, or GPU numbers.

## What the agent does

1. Finds `hrr-playback` on `PATH` or at `/opt/rocm/bin/hrr-playback`
2. If missing, asks: *"Where is hrr-playback installed?"*
3. Runs `run_hrr_replay.sh --archive ... --info` to check the archive opens
4. Runs `run_hrr_replay.sh --archive ... --analyze`
5. Presents the finding

## Multi-process capture

User provides the root, not a process directory:

> Here's the capture: `/data/crash/capture.hrr`

The agent runs `--info` on the root, reads the process table, and reports which
process it chose before replaying:

> The capture holds two processes: `pid-138` (4,211 events, complete) and
> `pid-680` (13,118,764 events, `Complete: NO`). `pid-680` is the workload and
> the crash-truncated one, so I'm replaying that.

## Clean replay of a crashing workload

Replay returns `replay_pass` but the user's original run faulted. That is a
finding, not a dead end: report that the recording does not reproduce the fault
on this machine and this playback build, then ask for the original fault line to
confirm which failure was expected.

## Archive will not open

```
[HRR] Version mismatch: file=4 reader=3
```

Report `archive_version_mismatch` and stop. The archive is newer than the
installed `hrr-playback`, so a newer playback build is needed before any replay
result means anything.

## If hrr-playback is not in a standard location

User answers: *"It's in `/opt/amd-hrr/bin/hrr-playback`"*

The agent sets `HRR_PLAYBACK=/opt/amd-hrr/bin/hrr-playback` for that run only and
re-runs. If a `lib/` directory sits beside `bin/`, the runner picks it up
automatically.
