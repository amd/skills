"""
PMC L2 / HBM efficiency analyzer.

Parses rocprofv3 PMC counter-collection CSV(s) and reports L2 cache behaviour
and HBM read efficiency for a kernel.  Complements hotspot_analyzer.py, which
reads ATT instruction timing (no cache counters).

Counters expected (collect via capture-kernel-trace "PMC mode"):
    L2 hit rate:        TCC_HIT_sum, TCC_MISS_sum, TCC_REQ_sum
    line utilization:   TCC_EA0_RDREQ_sum, TCC_EA0_RDREQ_32B_sum
    HBM traffic:        TCC_EA0_RDREQ_DRAM_sum
    L1->L2:             TCP_TCC_READ_REQ_sum

Usage:
    python pmc_l2_analyzer.py <pmc_csv> [<pmc_csv> ...] \
        [--kernel pa_decode_ps_kernel_0] [--ideal-gb 8.59] [--ea-channels 2]

Interpretation:
    L2 hit rate    : HIT/(HIT+MISS).  For decode with independent per-sequence
                     paged KV there is no inter-CTA reuse, so ~1-3% is EXPECTED
                     and correct (streaming).  A high value only appears when
                     the workload has real reuse (e.g. shared-prefix serving).
    32B fraction   : TCC_EA0_RDREQ_32B / TCC_EA0_RDREQ.  Fraction of HBM reads
                     that are partial 32B lines.  High % => scattered access /
                     poor spatial locality => wasted bandwidth.  ~0% => full
                     64B lines, no line-level waste.
    over-fetch     : measured HBM read bytes / ideal bytes.  ~1.0 => the kernel
                     reads exactly what it needs; >>1.0 => redundant fetches.
"""

import argparse
import csv
from collections import defaultdict


def load_counters(paths, kernel):
    agg = defaultdict(float)
    dispatches = set()
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                kn = r.get("Kernel_Name", "")
                if kernel and kernel not in kn:
                    continue
                name = r.get("Counter_Name")
                val = r.get("Counter_Value")
                if name is None or val in (None, ""):
                    continue
                agg[name] += float(val)
                dispatches.add(r.get("Dispatch_Id"))
    return agg, len(dispatches)


def main():
    ap = argparse.ArgumentParser(description="PMC L2/HBM efficiency analyzer")
    ap.add_argument("csv", nargs="+", help="pmc *_counter_collection.csv file(s)")
    ap.add_argument("--kernel", default="", help="substring filter on Kernel_Name")
    ap.add_argument("--ideal-gb", type=float, default=0.0,
                    help="ideal HBM read bytes per dispatch in GB (for over-fetch ratio)")
    ap.add_argument("--ea-channels", type=int, default=2,
                    help="EA interfaces to scale single-channel EA0 counters by (default 2)")
    args = ap.parse_args()

    agg, ndisp = load_counters(args.csv, args.kernel)
    if not agg:
        print("No matching counter rows found.")
        return 1

    print(f"  Dispatches matched: {ndisp}")
    hit = agg.get("TCC_HIT_sum", 0)
    miss = agg.get("TCC_MISS_sum", 0)
    req = agg.get("TCC_REQ_sum", 0)
    ea = agg.get("TCC_EA0_RDREQ_sum", 0)
    ea32 = agg.get("TCC_EA0_RDREQ_32B_sum", 0)
    dram = agg.get("TCC_EA0_RDREQ_DRAM_sum", 0)
    tcp = agg.get("TCP_TCC_READ_REQ_sum", 0)

    print("\n  L2 cache")
    print("  --------")
    if hit + miss > 0:
        print(f"  TCC_HIT_sum  = {hit:,.0f}")
        print(f"  TCC_MISS_sum = {miss:,.0f}")
        print(f"  L2 hit rate  = {100*hit/(hit+miss):.1f}%   (streaming decode: ~1-3% expected)")
    if tcp:
        print(f"  TCP->TCC read req (L1->L2) = {tcp:,.0f}")

    if ea > 0:
        ea64 = ea - ea32
        bytes_ea = (ea64 * 64 + ea32 * 32) * args.ea_channels
        print("\n  HBM read efficiency")
        print("  -------------------")
        print(f"  TCC_EA0_RDREQ (L2->HBM) = {ea:,.0f}")
        print(f"  32B partial fraction    = {100*ea32/ea:.1f}%   (~0% = full 64B lines, no waste)")
        print(f"  DRAM reads              = {dram:,.0f}")
        print(f"  est HBM read bytes      = {bytes_ea/1e9:.1f} GB  (EA0 x{args.ea_channels} channels)")
        if args.ideal_gb > 0 and ndisp:
            ideal = args.ideal_gb * ndisp * 1e9
            print(f"  ideal bytes             = {ideal/1e9:.1f} GB  ({args.ideal_gb} GB x {ndisp} disp)")
            print(f"  over-fetch ratio        = {bytes_ea/ideal:.2f}x   (~1.0 = no redundant fetch)")

    print("\n  Verdict")
    print("  -------")
    if hit + miss > 0:
        hr = 100 * hit / (hit + miss)
        if hr < 5:
            print("  L2 hit rate is near-zero => pure streaming, no reuse to exploit.")
            print("  Improving 'L2 hit rate' is a non-goal here; only real KV reuse")
            print("  (shared-prefix serving) would change it. ")
    if ea > 0 and ea32 / ea < 0.05:
        print("  Line utilization is full (>=95% 64B) => no spatial-locality waste.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
