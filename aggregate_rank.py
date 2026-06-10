#!/usr/bin/env python3
# aggregate_rank.py
"""
Combine the per-window CSVs written by score_rendition.py (one per nation) into
one ranked table, with WITHIN-CLIP bootstrap confidence intervals.

Why within-clip bootstrap: the project uses exactly one official clip per nation
(this year's FIFA WC anthem), so uncertainty cannot come from averaging multiple
clips. Instead we resample each clip's ~30-40 analysis WINDOWS with replacement
to measure how stable a nation's score is given its own clip's variability. This
does NOT model between-broadcast variance (out of scope by the one-clip rule);
it is the honest uncertainty available, and overlapping intervals are reported
as "cannot be distinguished".

Usage:  python aggregate_rank.py "path/to/*_windows.csv"
"""
import glob, sys, argparse, os
import numpy as np, pandas as pd

# Components used for ranking. intonation_R (threshold-free concentration) and
# participation carry the between-nation signal; stability is near-ceiling so it
# is down-weighted; activity is a minor richness cue.
WEIGHTS = dict(intonation_R=0.40, participation=0.30, activity=0.15, stability=0.15)
COMPONENTS = list(WEIGHTS)
N_BOOT = 2000


def bootstrap_medians(win_df, rng):
    n = len(win_df)
    arr = {c: win_df[c].to_numpy(dtype=float) for c in COMPONENTS}
    point = {c: float(np.nanmedian(arr[c])) for c in COMPONENTS}
    boot = {c: np.empty(N_BOOT) for c in COMPONENTS}
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        for c in COMPONENTS:
            boot[c][b] = np.nanmedian(arr[c][idx])
    return point, boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glob_pattern")
    args = ap.parse_args()
    files = sorted(glob.glob(args.glob_pattern))
    if not files:
        sys.exit(f"No files match {args.glob_pattern}")

    rng = np.random.default_rng(0)
    nations = {}
    for f in files:
        df = pd.read_csv(f)
        name = df["nation"].iloc[0] if "nation" in df else os.path.basename(f).split("_windows")[0]
        nations[name] = bootstrap_medians(df, rng)

    # cross-nation min-max normalisation per component (from the point estimates)
    pts = {c: np.array([nations[n][0][c] for n in nations]) for c in COMPONENTS}
    lo = {c: np.nanmin(pts[c]) for c in COMPONENTS}
    hi = {c: np.nanmax(pts[c]) for c in COMPONENTS}

    def norm(c, v):
        span = hi[c] - lo[c]
        return np.clip((v - lo[c]) / span, 0, 1) if span > 1e-9 else 0.5 * np.ones_like(v)

    rows = []
    for name, (point, boot) in nations.items():
        comp_pt = sum(WEIGHTS[c] * float(norm(c, point[c])) for c in COMPONENTS)
        comp_boot = sum(WEIGHTS[c] * norm(c, boot[c]) for c in COMPONENTS)
        lo95, hi95 = np.percentile(comp_boot, [2.5, 97.5])
        rows.append(dict(nation=name, score=comp_pt, lo=lo95, hi=hi95, **point))

    rows.sort(key=lambda r: -r["score"])
    print(f"\n{'rank':<5}{'nation':<16}{'score':>7}{'95% CI':>16}   {'R':>5}{'part':>6}")
    for i, r in enumerate(rows, 1):
        print(f"{i:<5}{r['nation']:<16}{r['score']:7.3f}   [{r['lo']:.3f}, {r['hi']:.3f}]"
              f"   {r['intonation_R']:5.2f}{r['participation']:6.2f}")

    print("\nPairs that CANNOT be distinguished (overlapping 95% CIs):")
    found = False
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            if rows[b]["hi"] >= rows[a]["lo"] and rows[a]["hi"] >= rows[b]["lo"]:
                print(f"  {rows[a]['nation']} ~ {rows[b]['nation']}")
                found = True
    if not found:
        print("  (none -- every neighbour is separated)")


if __name__ == "__main__":
    main()