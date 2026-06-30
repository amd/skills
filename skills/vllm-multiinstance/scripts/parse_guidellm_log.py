#!/usr/bin/env python3
"""Parse scores from a guidellm.log file (the ASCII summary tables).
usage: parse_guidellm_log.py <guidellm.log>
Pulls per-strategy rows from:
  - Server Throughput Statistics (concurrency, req/s, in tok/s, out tok/s, total tok/s)
  - Request Latency Statistics  (latency sec, TTFT ms, ITL ms, TPOT ms; medians)
Strategies appear in benchmark order (concurrency 32 then 64).
"""
import re, sys

def clean(line):
    # split an ASCII table row "| a | b | c |" -> ['a','b','c']
    return [c.strip() for c in line.strip().strip('|').split('|')]

def find_section(lines, title):
    for i, ln in enumerate(lines):
        if title in ln:
            return i
    return None

def data_rows(lines, start):
    """Yield cleaned data rows of the table beginning after `start`, i.e. rows
    that start with '| concurrent' (the strategy name)."""
    rows = []
    for ln in lines[start:]:
        s = ln.strip()
        if s.startswith('|') and re.match(r'\|\s*(concurrent|synchronous|throughput|constant|poisson)', s):
            rows.append(clean(ln))
        elif s.startswith('ℹ') and rows:
            break
        elif s.startswith('✔') and rows:
            break
    return rows

f = sys.argv[1]
lines = open(f, encoding='utf-8', errors='replace').read().splitlines()

# --- Server Throughput Statistics ---
# Columns: Strategy | Conc Mdn | Conc Mean | Req/s Mean | In tok/s | Out tok/s | Total tok/s
ti = find_section(lines, "Server Throughput Statistics")
thr = data_rows(lines, ti) if ti is not None else []

# --- Request Latency Statistics ---
# Columns: Strategy | Lat Mdn | Lat p95 | TTFT Mdn | TTFT p95 | ITL Mdn | ITL p95 | TPOT Mdn | TPOT p95
li = find_section(lines, "Request Latency Statistics")
lat = data_rows(lines, li) if li is not None else []

print(f"{'conc':>5} {'req/s':>7} {'in_tok/s':>9} {'out_tok/s':>10} {'tot_tok/s':>10} {'lat_s':>7} {'TTFT_ms':>9} {'ITL_ms':>7} {'TPOT_ms':>8}")
n = max(len(thr), len(lat))
for i in range(n):
    t = thr[i] if i < len(thr) else []
    l = lat[i] if i < len(lat) else []
    # throughput row: [strategy, conc_mdn, conc_mean, reqps_mean, in_s, out_s, tot_s]
    conc   = t[1] if len(t) > 1 else '?'
    reqps  = t[3] if len(t) > 3 else '?'
    in_s   = t[4] if len(t) > 4 else '?'
    out_s  = t[5] if len(t) > 5 else '?'
    tot_s  = t[6] if len(t) > 6 else '?'
    # latency row: [strategy, lat_mdn, lat_p95, ttft_mdn, ttft_p95, itl_mdn, itl_p95, tpot_mdn, tpot_p95]
    lat_s  = l[1] if len(l) > 1 else '?'
    ttft   = l[3] if len(l) > 3 else '?'
    itl    = l[5] if len(l) > 5 else '?'
    tpot   = l[7] if len(l) > 7 else '?'
    print(f"{conc:>5} {reqps:>7} {in_s:>9} {out_s:>10} {tot_s:>10} {lat_s:>7} {ttft:>9} {itl:>7} {tpot:>8}")
