# ATT and PMC collection

- [Step 1: Kernel Discovery](#step-1-kernel-discovery)
- [Step 2: Configure input.yaml](#step-2-configure-inputyaml)
- [Step 3: Collect ATT Trace](#step-3-collect-att-trace)
- [PMC collection for cache and HBM metrics](#pmc-collection-for-cache-and-hbm-metrics)

## Step 1: Kernel Discovery

```bash
touch /tmp/trace_ts
rocprofv3 --stats --kernel-trace -f csv -- <CMD> 2>&1
find . -maxdepth 3 -name "*stats*" -newer /tmp/trace_ts -type f 2>/dev/null
```

Parse the stats CSV and present a kernel table:

| Rank | Kernel Name | Calls | Total (us) | Avg (us) | % GPU Time |
|------|-------------|-------|------------|----------|------------|

Ask the user which kernel to trace if not obvious.

**Prefer `results.db`** if available — use sqlite3 for structured queries:
```bash
sqlite3 results.db "
SELECT ks.KernelName, COUNT(*) calls,
       ROUND(AVG(kd.end-kd.start)/1000.0,1) avg_us
FROM rocpd_kernel_dispatch kd
JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id=ks.id
GROUP BY ks.KernelName ORDER BY avg_us DESC LIMIT 20;"
```

## Step 2: Configure input.yaml

Write `/tmp/trace_input.yaml` with the job below, setting
`kernel_include_regex` to the kernel found in Step 1:

```yaml
jobs:
   -
       kernel_include_regex: <KERNEL_NAME_PATTERN>
       kernel_iteration_range: "[3-4]"
       output_file: out
       output_directory: kernel_trace_output
       output_format: [csv]
       truncate_kernels: true
       sys_trace: true
       advanced_thread_trace: true
       att_target_cu: 1
       att_shader_engine_mask: "0xf"
       att_simd_select: "0xf"
       att_buffer_size: "0x6000000"
```

Key notes:
- `kernel_iteration_range`: `"[1, [3-4]]"` selects iterations 1, 3, and 4; use `"[3-4]"` to omit earlier warmup iterations
- `att_buffer_size`: 96MB per SE; increase to `"0xC000000"` if truncated
- `att_target_cu: 1`: single CU keeps output manageable

## Step 3: Collect ATT Trace

```bash
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 rocprofv3 -i /tmp/trace_input.yaml -- <CMD> 2>&1
find . -type d -name "ui_output_agent_*" -newer /tmp/trace_ts 2>/dev/null
```

If the `rocprof-trace-decoder` library is missing, install the release matching
the ROCm version in `/opt/rocm/.info/version`. Set `RTD_VERSION` to that release
before running the block below. Choose it using the decoder's release notes for
the ROCm version on the machine being profiled:
```bash
: "${RTD_VERSION:?Set the decoder release matching the installed ROCm version}"
RTD_INSTALLER="rocprof-trace-decoder-manylinux-2.28-${RTD_VERSION}-Linux.sh"
wget -q "https://github.com/ROCm/rocprof-trace-decoder/releases/download/${RTD_VERSION}/${RTD_INSTALLER}"
chmod +x "${RTD_INSTALLER}"
"./${RTD_INSTALLER}" --skip-license --prefix=/tmp/rtd-install
find /tmp/rtd-install -name '*.so*' -exec cp -a {} /opt/rocm/lib/ \;
ldconfig
```

**Output structure:**
```
ui_output_agent_<PID>_dispatch_<N>/
├── code.json          ← PRIMARY: per-instruction stall/cycle data
├── snapshots.json     ← source file path mapping (virtual → local filename)
├── source_0_*.py      ← embedded source files
├── filenames.json     ← wave file index
├── occupancy.json     ← occupancy timeline
└── se*_sm*_sl*_wv*.json  ← per-wave raw traces
```

---

## PMC collection for cache and HBM metrics

ATT contains no cache counters. Query the installed profiler's counter list
(`rocprofv3 --list-avail`) and collect separate single-pass jobs for each group
available on the target GPU. For the CDNA counter names used by this analyzer:

```yaml
jobs:
  - kernel_include_regex: <KERNEL_NAME_PATTERN>
    pmc: [TCC_HIT_sum, TCC_MISS_sum, TCC_REQ_sum]
    output_directory: pmc_l2
    output_format: [csv]
  - kernel_include_regex: <KERNEL_NAME_PATTERN>
    pmc: [TCC_EA0_RDREQ_sum, TCC_EA0_RDREQ_32B_sum, TCC_EA0_RDREQ_DRAM_sum]
    output_directory: pmc_ea
    output_format: [csv]
```

Run `rocprofv3 -i <pmc-input.yaml> -- <CMD>` on the same workload for each group.
Counter availability and simultaneous-counter limits vary with architecture and
ROCm. Split a rejected group further; multi-pass TCC collection has caused GPU
hangs on gfx942. Retain the CSVs and record dispatch counts for each collection.
