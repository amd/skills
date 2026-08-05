---
name: hrr-replay-analysis
description: >-
  Replays a HIP Record and Replay (HRR) capture of a GPU workload and reports
  what went wrong as a structured finding: fault class (write to a read-only
  page, illegal memory access, NaN/Inf output divergence, hang, out of memory,
  clean pass), fault address, failing event index, and the implicated kernel.
  Also records a new capture when the user has none yet. Use when the user
  supplies an HRR archive, a `capture.hrr` directory or a `pid-*/events.bin`
  tree, asks to replay, summarize or triage a GPU recording, mentions
  hrr-playback or HIP Record and Replay, or brings a "Memory access fault by
  GPU" or "read-only page" failure from a vLLM, PyTorch or hipBLASLt run. Do not
  use to patch or rebuild a kernel, to drive rocgdb, or to verify a fix with
  `--replace-kernel`; this skill stops at the finding and hands off.
allowed-tools: Bash, Read
---

# HRR Replay Analysis

Replay an HRR archive on the host GPU, then produce a structured finding. An HRR
archive is a deterministic record of a real HIP workload, so the same fault can
be reproduced on a different machine from the recording alone.

## What the user should say

The user only needs to point at the recording. Examples:

- *"Replay and analyze this HRR archive: `/data/capture.hrr/pid-1842`"*
- *"What's in this capture? `capture.hrr`"*
- *"Analyze this replay log from an HRR run"* (log-only path)

The user should **not** need to name scripts, set env vars, pick a GPU, or know
where ROCm is installed.

## What to ask the user (only if missing)

| Missing | Ask once |
|---------|----------|
| Archive path | *"Which `capture.hrr` directory should I use?"* |
| `hrr-playback` not found after discovery | *"Where is `hrr-playback` installed on this machine?"* |

Do **not** ask for: GPU index, Docker, source trees, HIP library paths, ROCm
install path (assume `/opt/rocm`).

Ask for the original failure signature (the user's `Memory access fault` line or
`serve.log`) only when the replay itself comes back clean. A clean replay of a
crashing workload is a finding in its own right, and the user's log is what
tells you which fault you were supposed to see.

## Agent workflow

```
1. Resolve archive — see below; a bare capture.hrr root needs a pid-* pick
2. Discover hrr-playback (see below); ask the user only if not found
3. Read metadata first: run_hrr_replay.sh --archive <dir> --info
4. Replay and analyze: run_hrr_replay.sh --archive <dir> --analyze
5. Read the generated .finding.md and explain it in plain language
```

**Execute in the same turn** — do not narrate planning steps.

Step 3 is cheap and needs no GPU, so always do it before a full replay: it
catches an unreadable archive, a wrong `pid-*`, or a reader/format mismatch
before spending GPU time.

Step 4 replays with `--sync-after-launch`, which `run_hrr_replay.sh` adds for
you. Without it the GPU is serialized only once at the end, so a fault is
reported but not attributed and the finding has no failing event and no kernel.
Pass `--no-sync` only when throughput matters more than attribution.

### Discover `hrr-playback` (in order)

1. `command -v hrr-playback`
2. `$ROCM_PATH/bin/hrr-playback` (default `ROCM_PATH=/opt/rocm`)
3. `/opt/rocm/bin/hrr-playback`
4. User-provided path → set `HRR_PLAYBACK` for that run only

`run_hrr_replay.sh` adds `/opt/rocm/lib` and a sibling `lib/` next to the
playback binary to `LD_LIBRARY_PATH` automatically.

### Resolve the archive

A capture is either a single process directory or a multi-process tree:

```
capture.hrr/
  pid-138/          # often the init/parent process — small
  pid-680/
    events.bin      # the workload stream
    blobs/
    code_objects/
```

Run `--info` on the **archive root** first when the path has `pid-*` children:
that prints the process table (PID, parent PID, complete, events, blobs) so you
can name the workload process instead of guessing. Then pick the `pid-*`
directory with the largest `events.bin` and use it for the replay:

```bash
find <capture.hrr> -name events.bin -printf '%s %h\n' 2>/dev/null | sort -rn | head -1
```

If the user already gave a `pid-*` path, use it directly.

### Commands (the agent runs these, not the user)

```bash
SKILL=<path-to>/hrr-replay-analysis   # installed skill directory

# Metadata only (seconds, no GPU needed):
"$SKILL/scripts/run_hrr_replay.sh" --archive <archive-or-pid-dir> --info

# Full replay + structured finding:
"$SKILL/scripts/run_hrr_replay.sh" --archive <pid-dir> --analyze
```

Log-only (no replay):

```bash
python3 "$SKILL/scripts/analyze_replay_finding.py" \
  --log <replay.log> --archive <pid-dir> --format markdown -o finding.md
```

## System assumptions

| Assumption | Default |
|------------|---------|
| ROCm install | `/opt/rocm` (`$ROCM_PATH` override) |
| GPU | Auto-pick the device with the most free VRAM (`GPU=<index>` override) |
| Replay mode | Native host (`/dev/kfd`); no Docker |

## Fault taxonomy

| `fault_class` | Meaning |
|---------------|---------|
| `replay_pass` | Clean replay, all device-to-host checks passed |
| `read_only_page_fault` | Write to a read-only page |
| `illegal_memory_access` | Other GPU memory fault |
| `nan_inf_divergence` | Device-to-host numerical mismatch against the recording |
| `hang` | Device or queue hang |
| `replay_oom` | Out of VRAM on the replay machine — an environment issue, not the recorded bug |
| `replay_fatal_api` | A HIP API returned an error and stopped replay |
| `archive_version_mismatch` | The archive format and this `hrr-playback` disagree; nothing was replayed |

## A faulting ATen kernel needs the original failure signature

PyTorch and vLLM reach the GPU through `<<<>>>` (`hipLaunchByPtr`), and those
kernels pass device pointers inside by-value structs: `vectorized_elementwise_kernel`
takes a `std::array<char*,N>`, `reduce_kernel` a config struct with pointers at
arbitrary offsets. Capture records those offsets and replay translates them, with
a defensive rescan for any the capture-time heuristic missed, so on a current
build these kernels replay faithfully and a fault on one is a real finding.

Two things still make such a fault ambiguous: the detector is a value-based
heuristic, and an archive recorded before that support landed carries no offsets
at all, in which case replay launches with a capture-time address and faults on a
workload that is fine. Both look identical to a workload defect.

So the fault class stays what the evidence says (`illegal_memory_access`), the
finding carries a note, and you **ask the user for their original failure
signature** before calling it their bug. If their run never faulted here, suspect
the recording. Faults on `hipModuleLaunchKernel`-launched kernels (hipBLASLt
GEMMs, custom HIP kernels) do not carry this ambiguity.

## Reading `--info`

`Complete: NO` is the **expected** signature of a capture whose workload
crashed: the trailer is written on clean shutdown only. The reader recovers
every complete record and the archive is still replayable, so do not report
`Complete: NO` as corruption. `Recovered: N events` tells you how much survived.

## When the archive will not open

`[HRR] Version mismatch: file=N reader=M` means this `hrr-playback` cannot read
this archive. Stop and report `archive_version_mismatch`: replaying is
impossible until the versions line up. `file` newer than `reader` needs a newer
playback build; `reader` newer than `file` needs the older matching build or a
fresh capture. Never present a mismatch as a workload result.

## Recording a new capture

When the user has a failing workload but no archive, capture is an
environment-variable change to their existing run — no code edit, no rebuild:

```bash
HIP_HRR_CAPTURE_OUTPUT=./capture.hrr <their normal command>
```

Capture survives a crash, so let the workload fail as it did before. The archive
is what they then replay or hand over.

## Guardrails

- **Report only what the evidence supports.** The archive does not record the
  GPU SKU, library commits, or the full software stack. If the faulting kernel
  is not in the log or the archive, say it is unknown rather than naming a
  plausible one.
- **Distinguish the recorded bug from the replay environment.** `replay_oom` and
  `archive_version_mismatch` are properties of the replay machine.
- **Do not pick a GPU on a shared host without confirming.** Replay takes a
  whole device; `--info` does not.
- **Note the playback build.** The same archive can pass on one `hrr-playback`
  build and fault on another, so record which binary produced the result.
- **Privacy.** Archives can contain prompts and payload data. Say so before the
  user forwards one outside their organization.

## Out of scope — hand off after the finding

Kernel patching, rebuilds, `--replace-kernel` verification, rocgdb and core-dump
inspection, and bisecting a divergent kernel all belong to deeper crash
analysis. Produce the finding, name the next action, and stop.

See [reference.md](reference.md) for archive layout, log-line formats and flags,
and [examples.md](examples.md) for worked prompts.
