#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Compare LCA (lowest-common-ancestor) partitions between the "with capture"
(gold standard) and "no capture" comparative analyses of the same traces, and
score the no-capture partition against semantics-free baselines.

Both analyses emit a ``diff_stats.csv`` in which every row is a single GPU kernel
(identified by ``source`` = trace1/trace2 and ``gpu_op_uid``) tagged with a
``lowest_common_ancestor_id`` (LCA). LCA ids are opaque cluster labels: their
numeric value is only meaningful within a single analysis, so we never compare
ids across analyses directly. Instead we measure how well the two LCA
*partitions* of the shared kernel set agree, using bidirectional cluster purity.

Forward purity (gold -> no-capture):
    For every gold LCA group, take the majority no-capture LCA among its kernels
    and count how many kernels in the group carry that majority label. Sum across
    groups and divide by the number of matched kernels.

Reverse purity (no-capture -> gold): the same metric with the roles swapped.

Reporting both catches degenerate collapses: if the no-capture analysis dumped
every kernel into one LCA, forward purity stays high (each gold group trivially
agrees with the single label) but reverse purity collapses.

Strict consistency
------------------
An all-or-nothing variant of purity: a kernel is *consistent* iff its ENTIRE
group maps to a single group on the other side (no partial credit for the
majority). Strict forward = fraction of kernels whose whole gold group lands in
one no-capture LCA; strict reverse swaps the roles. Always <= the matching
purity, with equality only when every multi-kernel group is perfectly pure.

Random baselines
----------------
To judge whether the real no-capture purity reflects genuine semantic signal or
is merely an artifact of the bucket-size distribution and the metric's structure,
we compare against semantics-free baselines. All are evaluated on the SAME
matched kernels and PRESERVE the no-capture bucket sizes as observed on the
matched set -- only the kernel->bucket assignment changes.

Baseline 1 (random): randomly shuffle which matched kernels fall into each
    bucket, preserving bucket sizes. Reported as mean +/- std over many seeds.
Baseline 2 (sequential blocks, key string order): order buckets from most to
    least popular, then walk the matched kernels ordered by the lexicographic key
    string ("source:uid"), assigning the first k1 to bucket 1 (largest), the next
    k2 to bucket 2, and so on.
Baseline 3 (sequential blocks, integer uid order): identical block assignment,
    but kernels are ordered by (source, integer gpu_op_uid).

Usage:
    compare_lca_partitions.py <with_capture_diff_stats.csv> <no_capture_diff_stats.csv>
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLS = ["source", "gpu_op_uid"]
LCA_COL = "lowest_common_ancestor_id"
LCA_NAME = "lowest_common_ancestor_name"

# Number of random-baseline seeds (Baseline 1). Kept modest by default; the
# baseline loop is O(trials * kernels), so for large kernel sets it is capped to
# REDUCED_TRIALS (see the guard in main()).
DEFAULT_TRIALS = 200
LARGE_KERNEL_THRESHOLD = 5000
REDUCED_TRIALS = 2
SEED = 0


def load(path: Path) -> pd.DataFrame:
    """Load a diff_stats.csv, validate identity columns, and add a unique key."""
    df = pd.read_csv(path)
    missing = [c for c in KEY_COLS + [LCA_COL, LCA_NAME, "name"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df.copy()
    df["key"] = df["source"].astype(str) + ":" + df["gpu_op_uid"].astype(str)
    if df["key"].duplicated().any():
        n = int(df["key"].duplicated().sum())
        raise ValueError(
            f"{path}: {n} duplicate (source, gpu_op_uid) keys; not a unique kernel id"
        )
    return df


def purity(df: pd.DataFrame, group_col: str, label_col: str):
    """Sum over each group of the majority-label count within that group.

    Returns (matched_count, total, fraction, per_group_records)."""
    total = len(df)
    matched = 0
    records = []
    for gid, grp in df.groupby(group_col, sort=True):
        counts = grp[label_col].value_counts()  # sorted desc by count
        majority_label = counts.index[0]
        majority_count = int(counts.iloc[0])
        matched += majority_count
        records.append(
            {
                "group_id": gid,
                "group_size": len(grp),
                "majority_label": majority_label,
                "majority_count": majority_count,
                "group_purity": majority_count / len(grp),
            }
        )
    frac = matched / total if total else float("nan")
    return matched, total, frac, records


def purity_frac(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Fraction of items whose A-group carries the A-group's majority B-label.

    Sum over groups of A of the modal B-count, divided by the total item count.
    """
    df = pd.DataFrame({"a": labels_a, "b": labels_b})
    matched = 0
    for _, grp in df.groupby("a", sort=False):
        matched += int(grp["b"].value_counts().iloc[0])
    return matched / len(df) if len(df) else float("nan")


def both_purities(gold: np.ndarray, nc: np.ndarray):
    """Return (forward, reverse).

    forward = gold group -> majority nc label
    reverse = nc   group -> majority gold label
    """
    return purity_frac(gold, nc), purity_frac(nc, gold)


def strict_consistency_frac(
    group_labels: np.ndarray, other_labels: np.ndarray
) -> float:
    """Fraction of items whose entire group is homogeneous in ``other_labels``.

    Group items by ``group_labels``; an item is *consistent* iff every item that
    shares its group carries the same ``other_labels`` value (so the whole group
    maps to a single other-bin). Report the fraction of consistent items.

    This is an all-or-nothing variant of ``purity_frac``: a group contributes its
    full size when perfectly pure, else 0 (no partial credit for the majority).
    Hence ``strict_consistency_frac <= purity_frac`` always, with equality only
    when every multi-item group is perfectly pure.
    """
    df = pd.DataFrame({"g": group_labels, "o": other_labels})
    consistent = 0
    for _, grp in df.groupby("g", sort=False):
        if grp["o"].nunique() == 1:
            consistent += len(grp)
    return consistent / len(df)


def both_strict(gold: np.ndarray, nc: np.ndarray):
    """Return (strict_forward, strict_reverse).

    strict_forward = fraction of items whose gold group is fully pure in nc
    strict_reverse = fraction of items whose nc   group is fully pure in gold
    """
    return strict_consistency_frac(gold, nc), strict_consistency_frac(nc, gold)


def agreement_report(gold: pd.DataFrame, ncap: pd.DataFrame) -> None:
    """Print the LCA-partition agreement (forward/reverse purity) section."""
    print("=" * 78)
    print("LCA assignment agreement: with-capture (GOLD) vs no-capture")
    print("=" * 78)
    print(
        f"with-capture kernels : {len(gold)}  | distinct gold LCAs: {gold[LCA_COL].nunique()}"
    )
    print(
        f"no-capture   kernels : {len(ncap)}  | distinct nc   LCAs: {ncap[LCA_COL].nunique()}"
    )

    g = gold[["key", "name", LCA_COL]].rename(
        columns={LCA_COL: "lca_gold", "name": "name_gold"}
    )
    n = ncap[["key", "name", LCA_COL]].rename(
        columns={LCA_COL: "lca_nc", "name": "name_nc"}
    )
    merged = g.merge(n, on="key")
    name_agree = (
        (merged["name_gold"] == merged["name_nc"]).mean()
        if len(merged)
        else float("nan")
    )

    gold_only = len(gold) - len(merged)
    nc_only = len(ncap) - len(merged)
    print("-" * 78)
    print(f"matched kernels (in both)   : {len(merged)}")
    print(f"gold-only (unmatched)       : {gold_only}")
    print(f"no-capture-only (unmatched) : {nc_only}")
    print(
        f"kernel-name agreement on matched keys: {name_agree:.4f}  (sanity: should be 1.0)"
    )

    df = merged[["key", "lca_gold", "lca_nc"]]
    print("-" * 78)
    print(f"matched-set distinct gold LCAs: {df['lca_gold'].nunique()}")
    print(f"matched-set distinct nc   LCAs: {df['lca_nc'].nunique()}")

    # Forward purity: gold groups -> majority no-capture LCA.
    f_matched, f_total, f_frac, _ = purity(df, "lca_gold", "lca_nc")
    # Reverse purity: no-capture groups -> majority gold LCA.
    r_matched, r_total, r_frac, r_recs = purity(df, "lca_nc", "lca_gold")

    # Strict consistency (all-or-nothing per group; no partial credit).
    gold_lbl = df["lca_gold"].to_numpy()
    nc_lbl = df["lca_nc"].to_numpy()
    sf = strict_consistency_frac(gold_lbl, nc_lbl)  # gold groups fully pure in nc
    sr = strict_consistency_frac(nc_lbl, gold_lbl)  # nc groups fully pure in gold
    n_tot = len(df)

    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(
        f"Forward purity (gold group -> majority no-capture LCA): "
        f"{f_matched}/{f_total} = {f_frac:.4f}"
    )
    print(
        f"Reverse purity (no-capture group -> majority gold LCA): "
        f"{r_matched}/{r_total} = {r_frac:.4f}"
    )
    print(
        f"Strict forward consistency (whole gold group -> one nc LCA): "
        f"{int(round(sf * n_tot))}/{n_tot} = {sf:.4f}"
    )
    print(
        f"Strict reverse consistency (whole nc group -> one gold LCA): "
        f"{int(round(sr * n_tot))}/{n_tot} = {sr:.4f}"
    )
    print("=" * 78)

    # Extra context: the largest no-capture groups (where collapse would show up).
    top_nc = pd.DataFrame(r_recs).sort_values("group_size", ascending=False).head(5)
    print("Largest no-capture LCA groups (matched set):")
    print(top_nc.to_string(index=False))


def baseline_report(merged: pd.DataFrame, title: str, n_trials: int) -> None:
    """Compute real + baseline purities for a given matched-kernel frame.

    ``merged`` must have columns: lca_gold, lca_nc, nc_name, source, gpu_op_uid,
    key. Bucket sizes are recomputed from ``merged`` so the "total matched
    preserved" property holds for whatever subset is passed in.
    """
    merged = merged.sort_values("key").reset_index(drop=True)
    n = len(merged)

    gold = merged["lca_gold"].to_numpy()
    nc_real = merged["lca_nc"].to_numpy()

    # No-capture bucket sizes on this (sub)set (sum == n).
    sizes = merged["lca_nc"].value_counts()  # sorted desc by count
    bucket_ids = sizes.index.to_numpy()
    bucket_sizes = sizes.to_numpy()
    names = merged.drop_duplicates("lca_nc").set_index("lca_nc")["nc_name"].to_dict()
    assert bucket_sizes.sum() == n, (bucket_sizes.sum(), n)

    # Label pool: bucket id repeated by its size (length == n).
    pool = np.repeat(bucket_ids, bucket_sizes)
    assert len(pool) == n

    # ---- Real no-capture ----
    real_fwd, real_rev = both_purities(gold, nc_real)
    real_sfwd, real_srev = both_strict(gold, nc_real)

    # ---- Baseline 1: random shuffle of the pool over matched kernels ----
    rng = np.random.default_rng(SEED)
    fwds = np.empty(n_trials)
    revs = np.empty(n_trials)
    sfwds = np.empty(n_trials)
    srevs = np.empty(n_trials)
    for t in range(n_trials):
        shuffled = pool.copy()
        rng.shuffle(shuffled)
        fwds[t], revs[t] = both_purities(gold, shuffled)
        sfwds[t], srevs[t] = both_strict(gold, shuffled)
    b1_fwd_m, b1_fwd_s = fwds.mean(), fwds.std()
    b1_rev_m, b1_rev_s = revs.mean(), revs.std()
    b1_sfwd_m, b1_sfwd_s = sfwds.mean(), sfwds.std()
    b1_srev_m, b1_srev_s = srevs.mean(), srevs.std()

    # ---- Baseline 2: sequential blocks, buckets ordered most->least popular ----
    # Kernels ordered by key (lexicographic string: source then string uid).
    seq = pool.copy()
    b2_fwd, b2_rev = both_purities(gold, seq)
    b2_sfwd, b2_srev = both_strict(gold, seq)

    # ---- Baseline 3: sequential blocks, kernels ordered by INT gpu_op_uid ----
    order3 = np.lexsort((merged["gpu_op_uid"].to_numpy(), merged["source"].to_numpy()))
    gold3 = gold[order3]
    b3_fwd, b3_rev = both_purities(gold3, pool)
    b3_sfwd, b3_srev = both_strict(gold3, pool)

    # ---------------- report ----------------
    print("=" * 74)
    print(f"{title}")
    print("Matched kernels:", n, "| no-capture buckets:", len(bucket_ids))
    print("Distinct gold LCAs:", merged["lca_gold"].nunique())
    print("=" * 74)
    print("No-capture bucket sizes (preserved by all baselines):")
    for bid, sz in zip(bucket_ids, bucket_sizes):
        print(f"  LCA {bid:>3}  {names.get(bid, '?'):<18} {sz:>4}")
    print("-" * 74)
    hdr = f"{'partition':<26}{'forward (gold->nc)':>24}{'reverse (nc->gold)':>24}"
    print(hdr)
    print("-" * 74)
    print(f"{'Real no-capture':<26}{real_fwd:>24.4f}{real_rev:>24.4f}")
    print(
        f"{'Baseline 1 (random)':<26}"
        f"{f'{b1_fwd_m:.4f} +/- {b1_fwd_s:.4f}':>24}"
        f"{f'{b1_rev_m:.4f} +/- {b1_rev_s:.4f}':>24}"
    )
    print(f"{'Baseline 2 (seq, key str)':<26}{b2_fwd:>24.4f}{b2_rev:>24.4f}")
    print(f"{'Baseline 3 (seq, int uid)':<26}{b3_fwd:>24.4f}{b3_rev:>24.4f}")
    print("-" * 74)
    print("Strict consistency (fraction of items in a FULLY pure group):")
    sh = f"{'partition':<26}{'strict fwd (gold pure)':>24}{'strict rev (nc pure)':>24}"
    print(sh)
    print("-" * 74)
    print(f"{'Real no-capture':<26}{real_sfwd:>24.4f}{real_srev:>24.4f}")
    print(
        f"{'Baseline 1 (random)':<26}"
        f"{f'{b1_sfwd_m:.4f} +/- {b1_sfwd_s:.4f}':>24}"
        f"{f'{b1_srev_m:.4f} +/- {b1_srev_s:.4f}':>24}"
    )
    print(f"{'Baseline 2 (seq, key str)':<26}{b2_sfwd:>24.4f}{b2_srev:>24.4f}")
    print(f"{'Baseline 3 (seq, int uid)':<26}{b3_sfwd:>24.4f}{b3_srev:>24.4f}")
    print("=" * 74)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "with_capture", type=Path, help="path to with-capture (GOLD) diff_stats.csv"
    )
    ap.add_argument("no_capture", type=Path, help="path to no-capture diff_stats.csv")
    ap.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help=f"random-baseline seeds (default: {DEFAULT_TRIALS}; auto-capped to "
        f"{REDUCED_TRIALS} when matched kernels > {LARGE_KERNEL_THRESHOLD})",
    )
    args = ap.parse_args()

    gold = load(args.with_capture)
    ncap = load(args.no_capture)

    # Guard: with no shared identity keys the matched set is empty and every
    # downstream report divides by zero / sorts an empty frame. Fail cleanly.
    if not (set(gold["key"]) & set(ncap["key"])):
        print(
            f"No matched keys between the two files ({len(gold)} gold rows, "
            f"{len(ncap)} no-capture rows, 0 shared keys). Nothing to compare."
        )
        return

    # ---- Part 1: LCA-partition agreement ----
    agreement_report(gold, ncap)
    print()

    # Matched set for the baselines: identity present in both. Deterministic by key.
    merged = (
        gold[["key", LCA_COL]]
        .rename(columns={LCA_COL: "lca_gold"})
        .merge(
            ncap[["key", "source", "gpu_op_uid", LCA_COL, LCA_NAME]].rename(
                columns={LCA_COL: "lca_nc", LCA_NAME: "nc_name"}
            ),
            on="key",
        )
    )

    # Guard: the random baseline is O(trials * kernels); cap trials on large sets.
    n_trials = args.trials
    if len(merged) > LARGE_KERNEL_THRESHOLD:
        n_trials = REDUCED_TRIALS
        print(
            f"[guard] matched kernels = {len(merged)} > {LARGE_KERNEL_THRESHOLD}; "
            f"reducing random-baseline trials {args.trials} -> {n_trials}"
        )
        print()

    # ---- Part 2: random / sequential baselines ----
    baseline_report(merged, "ALL MATCHED KERNELS", n_trials)

    # Exclude singleton gold groups: gold LCAs with exactly one matched kernel.
    gsz = merged.groupby("lca_gold")["key"].transform("size")
    merged_ns = merged[gsz > 1].copy()
    n_dropped = len(merged) - len(merged_ns)
    print()
    baseline_report(
        merged_ns,
        f"EXCLUDING SINGLETON GOLD GROUPS (dropped {n_dropped} kernels)",
        n_trials,
    )
    print(f"(Baseline 1 over {n_trials} random seeds; Baselines 2 & 3 deterministic.)")


if __name__ == "__main__":
    main()
