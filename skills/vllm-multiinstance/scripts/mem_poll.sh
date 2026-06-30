#!/bin/bash
# Poll podman stats for all vllm-instance-* containers and record the peak
# aggregate memory usage. Writes a CSV trace and a final PEAK line.
#   usage: mem_poll.sh <label> <out_csv> [interval_sec]
# Stop by deleting the .run flag file: <out_csv>.run
set -uo pipefail
LABEL="${1:?label}"
OUT="${2:?out csv}"
INTERVAL="${3:-2}"
FLAG="${OUT}.run"
touch "$FLAG"
echo "ts_unix,n_instances,agg_mem_bytes,agg_mem_human,per_container" > "$OUT"

# Convert podman's human MEM (e.g. "12.3GB", "512MB", "1.2kB") to bytes.
to_bytes() {
    local v="$1" num unit
    num="${v//[^0-9.]/}"; unit="${v//[0-9.]/}"
    case "$unit" in
        B)           awk -v n="$num" 'BEGIN{printf "%.0f", n}';;
        kB|KB|KiB)   awk -v n="$num" 'BEGIN{printf "%.0f", n*1024}';;
        MB|MiB)      awk -v n="$num" 'BEGIN{printf "%.0f", n*1024*1024}';;
        GB|GiB)      awk -v n="$num" 'BEGIN{printf "%.0f", n*1024*1024*1024}';;
        TB|TiB)      awk -v n="$num" 'BEGIN{printf "%.0f", n*1024*1024*1024*1024}';;
        *)           echo 0;;
    esac
}

peak=0
peak_human="0B"
peak_n=0
# tick counter only used to vary nothing; loop until flag removed.
while [[ -f "$FLAG" ]]; do
    # One stats snapshot of all running containers; filter to vllm-instance-*.
    mapfile -t lines < <(podman stats --no-stream --format '{{.Name}} {{.MemUsage}}' 2>/dev/null | grep -E '^(bench-vllm-instance|vllm-instance)-' || true)
    agg=0; n=0; per=""
    for ln in "${lines[@]}"; do
        name="${ln%% *}"
        rest="${ln#* }"
        used="${rest%% /*}"        # take left of " / " (usage, not limit)
        used="${used// /}"
        b=$(to_bytes "$used")
        agg=$(( agg + b ))
        n=$(( n + 1 ))
        per="${per}${name}=${used};"
    done
    if (( n > 0 )); then
        human=$(awk -v b="$agg" 'BEGIN{ split("B KB MB GB TB",u," "); i=1; while(b>=1024 && i<5){b/=1024;i++} printf "%.2f%s", b, u[i] }')
        # epoch via date is unavailable in some sandboxes; use /proc/uptime delta-free stamp.
        ts=$(cut -d' ' -f1 /proc/uptime 2>/dev/null || echo 0)
        echo "${ts},${n},${agg},${human},${per}" >> "$OUT"
        if (( agg > peak )); then peak=$agg; peak_human=$human; peak_n=$n; fi
    fi
    sleep "$INTERVAL"
done

human=$(awk -v b="$peak" 'BEGIN{ split("B KB MB GB TB",u," "); i=1; while(b>=1024 && i<5){b/=1024;i++} printf "%.2f%s", b, u[i] }')
echo "PEAK label=${LABEL} instances=${peak_n} agg_mem_bytes=${peak} agg_mem_human=${human}" >> "$OUT"
echo "PEAK label=${LABEL} instances=${peak_n} agg_mem_human=${human}"
