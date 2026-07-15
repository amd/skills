# HRR capture reference

## Directory layout

```
capture.hrr/
  pid-<capture_pid>/
    events.bin          # primary event stream (may be GB-scale)
    blobs/              # code objects, graphs, memcpy payloads
    (optional metadata files)
```

Pick the `pid-*` directory with the **largest `events.bin`** for the faulting process.

## events.bin record model (conceptual)

Each event has:

| Field | Role |
|-------|------|
| Thread id | Capturing host thread |
| Sequence / event index | Monotonic call index in replay |
| API id | HIP API (malloc, launch, memcpy, sync, …) |
| Payload | API-specific bytes (variable-length for kernel launches) |

**Kernel launch payload** includes: stream, kernel name, code-object hash, grid, block, shared memory, **kernarg blob** (pointer table + struct args), optional D2H snapshot descriptors.

## Completeness markers

| Signal | Meaning |
|--------|---------|
| `Complete: YES` (`--info`) | Clean shutdown trailer present |
| `recovered N events` | Crash capture; trailer missing; reader kept all complete records |
| `Torn trailing record` | Last record partial; preceding events valid |

On recent ROCm builds with crash-resilient HRR capture, crash captures are **expected** to lack a trailer and still be replayable (reader reports `recovered N events`).

## Replay log lines

### Progress

```
[HRR progress] elapsed_s=... seq=13118764 kernels=797227 d2h_pass=4303 d2h_fail=0 ...
```

- `seq` — last replayed event sequence number (use as **failing_call_index** proxy when fault follows)
- `kernels` — kernel launch count so far
- `d2h_*` — device-to-host validation counters

### GPU memory fault (ROCr)

```
Memory access fault by GPU node-N (Agent handle: 0x...) on address 0xADDR. Reason: ...
:0:rocdevice.cpp:NNNN: Memory Fault Error [..., faulting addr: 0xADDR, kernel: Cijk_...]
```

Extract: **fault_address**, **kernel_name**, **gpu_node**, **fault_reason**.

### Hang analysis block

```
Dispatch Header = 0x..., grid=[...], workgroup=[...], kernarg_address=0x..., kernel_obj=0x...
```

Extract: **kernarg_address**, **grid**, **workgroup** — ties fault to launch packet.

### Fatal API abort

```
[HRR] Fatal: T146 Event 9268 (hipMalloc) returned 2 (out of memory) — aborting replay
```

Extract: **failing_thread**, **failing_call_index**, **failing_api**.

### Suballoc fidelity (optional playback feature)

```
[HRR] SUBALLOC OOB: kernel arg[10] rec 0x... resolves inside a captured segment but in no active tensor block
```

High count on `arg[10]` with `d2h_fail=0` and later MAF → likely **stale/OOB device pointer** in kernarg, not host numerics.

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

**read_only_page_fault** on StreamK GEMM → investigate edge tile / workspace (`AddressWS`) / output (`AddressD`) stores.

## Playback build fidelity

The same archive can **PASS** on one `hrr-playback` build and fault on another. Record which playback binary produced each result.

## ROCm layout (assumed)

| Path | Role |
|------|------|
| `/opt/rocm/bin/hrr-playback` | Default playback tool location |
| `/opt/rocm/lib` | HIP/HSA and ROCm runtime libraries |
| `$ROCM_PATH` | Override prefix if ROCm is not under `/opt/rocm` |

`run_hrr_replay.sh` prepends these to `LD_LIBRARY_PATH`. If `hrr-playback` lives in `<prefix>/bin/`, `<prefix>/lib` is added automatically.

## Parser script

```bash
python3 skills/hrr-replay-findings/scripts/analyze_replay_finding.py --help
```

Outputs JSON or Markdown `Finding` with fields:

`outcome`, `fault_class`, `fault_address`, `failing_event_seq`, `failing_call_index`, `kernel_name`, `kernarg_address`, `d2h_fail`, `archive_events`, …
