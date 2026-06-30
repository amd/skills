#!/usr/bin/env python3
"""Extract headline guidellm metrics from a benchmarks.json.
usage: extract_perf.py <benchmarks.json>
Prints one line per benchmark (rate): rate, completed, req/s, out_tok/s, TTFT_ms(median), ITL_ms(median).
"""
import json, sys

def stat(metric, field="median"):
    """metric is a dict like {'successful': {'median':..}, ...} or {'median':..}."""
    if metric is None:
        return None
    if isinstance(metric, dict):
        # guidellm nests under 'successful' / 'total' sometimes
        for key in ("successful", "total", "all"):
            if key in metric and isinstance(metric[key], dict):
                if field in metric[key]:
                    return metric[key][field]
        if field in metric:
            return metric[field]
    return None

def g(d, *path):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur

f = sys.argv[1]
d = json.load(open(f))
bs = d["benchmarks"]
print(f"{'rate':>6} {'completed':>10} {'req/s':>8} {'out_tok/s':>10} {'TTFT_ms':>9} {'ITL_ms':>8} {'TPOT_ms':>8}")
for b in bs:
    cfg = b.get("config", {}) or {}
    # requested rate lives in config.strategy or args
    strat = cfg.get("strategy") or {}
    rate = strat.get("streams") or strat.get("max_concurrency") or strat.get("type_") if isinstance(strat, dict) else strat
    m = b.get("metrics", {}) or {}
    # request throughput
    reqps = stat(m.get("requests_per_second"))
    outtps = stat(m.get("output_tokens_per_second"))
    ttft = stat(m.get("time_to_first_token_ms"))
    itl = stat(m.get("inter_token_latency_ms"))
    tpot = stat(m.get("time_per_output_token_ms"))
    # request counts
    rs = b.get("requests", {}) or {}
    completed = None
    for key in ("successful", "completed", "total"):
        v = rs.get(key)
        if isinstance(v, list):
            completed = len(v); break
        if isinstance(v, (int, float)):
            completed = v; break
    def fmt(x, p=2):
        return f"{x:.{p}f}" if isinstance(x, (int, float)) else "n/a"
    print(f"{str(rate):>6} {str(completed):>10} {fmt(reqps):>8} {fmt(outtps):>10} {fmt(ttft,1):>9} {fmt(itl,2):>8} {fmt(tpot,2):>8}")
