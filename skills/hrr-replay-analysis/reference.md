# HRR capture and replay reference

- [Directory layout](#directory-layout)
- [Archive summary output](#archive-summary-output)
- [Completeness markers](#completeness-markers)
- [Version mismatch](#version-mismatch)
- [events.bin record model](#eventsbin-record-model-conceptual)
- [Replay log lines](#replay-log-lines)
- [Playback flags](#playback-flags)
- [Capture environment variables](#capture-environment-variables)
- [Tensile / hipBLASLt kernel names](#tensile--hipblaslt-kernel-name-cheat-sheet)
- [Playback build fidelity](#playback-build-fidelity)
- [ROCm layout](#rocm-layout-assumed)
- [Parser script](#parser-script)

## Directory layout

```
capture.hrr/
  pid-<capture_pid>/
    events.bin          # primary event stream (may be GB-scale)
    blobs/              # code objects, graphs, memcpy payloads
    code_objects/
    (optional metadata files)
```

A capture of a multi-process workload holds one `pid-*` directory per process.
Run `--info` on the archive **root** for the process table, then pick the
`pid-*` directory with the **largest `events.bin`** for the faulting workload —
the smaller ones are usually short-lived parents or init processes.

## Archive summary output

`hrr-playback <archive> --info` on the root of a multi-process archive:

```
HRR Archive Root: /data/capture.hrr
========================================
Capture Mode: in-tree
Owner PID:    680
Processes:    2

  PID          Parent PID   Complete   Events       Blobs      Path
  ---          ----------   --------   ------       -----      ----
  138          1            yes        4211         12         /data/capture.hrr/pid-138
  680          138          NO         13118764     8034       /data/capture.hrr/pid-680
```

A single process directory:

```
HRR Archive: /data/capture.hrr/pid-680
========================================
Complete:     NO (no shutdown trailer; capture likely crashed)
Recovered:    13118764 events
Events:       13118764
Kernels:      797227
Blobs:        8034
Code Objects: 41
```

`--info` needs no GPU. Add `--events` for the full event log (very large output).

## Completeness markers

| Signal | Meaning |
|--------|---------|
| `Complete:     yes (clean shutdown)` | Shutdown trailer present |
| `Complete:     NO (crash-truncated; trailing torn record discarded)` | Crash capture; last record was partial and was dropped |
| `Complete:     NO (no shutdown trailer; capture likely crashed)` | Crash capture; every complete record kept |
| `recovered N events` (stderr) | Reader repaired a torn tail and kept N events |

The value is lowercase `yes` or upper-case `NO`, each followed by an
explanation, so match it case-insensitively.

Crash captures are **expected** to lack a trailer and still replay. `Complete:
NO` is not corruption.

## Version mismatch

```
[HRR] Version mismatch: file=4 reader=3
```

Printed to stderr by the archive reader when the on-disk format version and the
`hrr-playback` build disagree. Nothing is replayed. `file` > `reader` needs a
newer playback binary; `reader` > `file` needs the matching older binary or a
fresh capture.

## events.bin record model (conceptual)

Each event has:

| Field | Role |
|-------|------|
| Thread id | Capturing host thread |
| Sequence / event index | Monotonic call index in replay |
| API id | HIP API (malloc, launch, memcpy, sync, …) |
| Payload | API-specific bytes (variable-length for kernel launches) |

**Kernel launch payload** includes: stream, kernel name, code-object hash, grid,
block, shared memory, **kernarg blob** (pointer table + struct args), optional
device-to-host snapshot descriptors.

## Replay log lines

### Progress

```
[HRR progress] elapsed_s=612.4 seq=13118764 kernels=797227 d2h_pass=4303 d2h_fail=0 d2h_attempted=4315 last="Cijk_..."
```

- `seq` — last replayed event sequence number (use as **failing_call_index**
  proxy when a fault follows)
- `kernels` — kernel launch count so far
- `d2h_*` — device-to-host validation counters

### Final summary

```
[HRR]   D2H checks     : 4303 pass (4300 exact, 3 within tol), 0 fail, 12 skipped
[HRR] PASS
```

`[HRR] PASS` / `[HRR] FAIL` is the replay verdict; the process exit code is 0
only when no device-to-host check failed.

### GPU memory fault (ROCr)

```
Memory access fault by GPU node-4 (Agent handle: 0x...) on address 0x7f2c0a800000. Reason: Write access to a read-only page.
:0:rocdevice.cpp:NNNN: Memory Fault Error [..., faulting addr: 0x7f2c0a800000, kernel: Cijk_...]
```

The bracket's leading fields vary by ROCm build. Both of these appear in the
wild and both are parsed:

```
Memory Fault Error [GPU index: 0, faulting addr: 0x..., kernel: hrr_fault_kernel]
Memory Fault Error [host: h1, GPU index: 2, faulting addr: 0x..., kernel: Cijk_...]
```

Extract: **fault_address**, **kernel_name**, **gpu_node**, **fault_reason**.

A GPU fault usually also trips an HRR abort line (below). The memory fault is
the finding; the abort is its consequence.

### Hang analysis block

```
Dispatch Header = 0x..., grid=[...], workgroup=[...], kernarg_address=0x..., kernel_obj=0x...
```

Extract: **kernarg_address**, **grid**, **workgroup** — ties the fault to the
launch packet.

### Per-event progress (`--sync-after-launch`)

```
[HRR] Event 1304: hipModuleLaunchKernel   -> Kernel '..._FillFunctor...' OK
[HRR] Event 1324: hipModuleLaunchKernel
```

The last of these before a fault is the failing dispatch. A GPU fault aborts the
process before HRR writes its own `Fatal` line, so on a hard memory fault this
is often the only record of **failing_call_index** and the implicated kernel.

### Fatal abort

```
[HRR] Fatal: T146 Event 9268 (hipMalloc) returned 2 (out of memory) — aborting replay
[HRR] Fatal: GPU error after T146 Event 9268 (hipLaunchKernel): 1 (invalid argument) — aborting
```

Extract: **failing_thread**, **failing_call_index**, **failing_api**. An `out of
memory` here is a replay-environment limit, not the recorded defect.

### Sub-allocation fidelity (optional playback feature)

```
[HRR] SUBALLOC OOB: kernel arg[10] rec 0x... resolves inside a captured segment but in no active tensor block
```

A high count on one `arg[N]` with `d2h_fail=0` and a later memory fault points
at a **stale or out-of-bounds device pointer** in the kernel arguments rather
than host numerics.

## Playback flags

| Flag | Use |
|------|-----|
| `--info` | Archive summary; no GPU required |
| `--info --events` | Full event log (very large) |
| `--sync-after-launch` | Synchronize after every launch → attribute a fault to a kernel. `run_hrr_replay.sh` adds this by default; `--no-sync` opts out, and `--timing` opts out on its own |
| `--sync-after-event` | Synchronize after every event → attribute a fault to any event (slowest) |
| `--single-thread` | Replay on one thread; deterministic ordering |
| `--repair` | Rewrite a torn archive tail in place |

`--replace-kernel` exists but belongs to fix verification, which is out of scope
for this skill.

## Capture environment variables

Capture is enabled on the user's unmodified workload:

```bash
HIP_HRR_CAPTURE_OUTPUT=./capture.hrr <their normal command>
```

Capture is crash-durable: let the workload fail as it did originally.

## Tensile / hipBLASLt kernel name cheat sheet

Example:

```
Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x192x128_..._SK3_..._WS64_WG16_16_1
```

| Token | Meaning |
|-------|---------|
| `Cijk_*` | Contraction GEMM family |
| `MT128x192x128` | Macro-tile dimensions |
| `SK3` | StreamK variant |
| `WS64` | Workspace-related sizing hint |
| `Bias_HA` | Bias + HPA layout flags |

A `read_only_page_fault` on a StreamK GEMM points at the edge tile, the
workspace pointer (`AddressWS`) or the output store (`AddressD`).

## Playback build fidelity

The same archive can **PASS** on one `hrr-playback` build and fault on another.
Record which playback binary produced each result.

## ROCm layout (assumed)

| Path | Role |
|------|------|
| `/opt/rocm/bin/hrr-playback` | Default playback tool location |
| `/opt/rocm/lib` | HIP/HSA and ROCm runtime libraries |
| `$ROCM_PATH` | Override prefix if ROCm is not under `/opt/rocm` |

`run_hrr_replay.sh` prepends these to `LD_LIBRARY_PATH`. If `hrr-playback` lives
in `<prefix>/bin/`, `<prefix>/lib` is added automatically.

## Parser script

```bash
python3 scripts/analyze_replay_finding.py --help
```

Outputs a JSON or Markdown `Finding` with fields:

`outcome`, `fault_class`, `fault_address`, `failing_event_seq`,
`failing_call_index`, `kernel_name`, `kernarg_address`, `d2h_fail`,
`archive_events`, `archive_complete`, …
