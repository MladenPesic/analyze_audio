#!/usr/bin/env python3
# score_rendition.py
"""
Reference-free "rendition score" for a crowd anthem clip.

No MIDI, no per-nation reference. Everything is measured from the clip itself,
along FOUR components that together describe how well a nation sang:

    intonation    how tightly the pitch sits on stable musical notes  (in tune)
    activity      how much melodic variety there is                   (not droning)
    stability     how clean / sustained the notes are                 (not smeared)
    participation how much of the clip is actually confident singing  (commitment)

Why a composite and not one number: intonation alone can't tell a good singer
from a monotone drone (both sit on a stable note); activity alone can't tell a
singer from random noise (both move around). Requiring several together is what
isolates real singing.

NON-STATIONARITY DEFENCE
------------------------
A broadcast clip drifts between a lone miked player and the massed crowd as the
camera moves. To stop a few soloist-heavy moments from dominating, the clip is
scored in short overlapping WINDOWS and aggregated with the MEDIAN (robust to a
minority of odd windows). The per-window values are also plotted over time, so
you can SEE the source change and judge whether a clip is usable.

USAGE
-----
    python score_rendition.py path/to/vocals.wav   --label "Wales"
    python score_rendition.py path/to/pitch.csv     --label "Wales"

Input may be audio (best: the separated vocals.wav) or a pitch CSV produced by
analyze.py (needs a 'time_s' column and a 'midi' or 'f0_hz' column).

Outputs: prints the four components (raw + random-chance baseline) and a
provisional composite, and saves <label>_rendition.png showing each component
over time.

NOTE: the composite blend here is PER-CLIP and provisional. The final ranking
normalises each component ACROSS all nations (that step lives in the comparison
stage), so treat the single composite number as indicative until then.
"""
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- frame grid (matches analyze.py) ----
SR, HOP = 22050, 512
FPS = SR / HOP

# ---- tunables ----
GRID_TOL_CENTS = 20.0     # "in tune" = within this many cents of a real note
WINDOW_S = 3.0            # analysis window length
HOP_S = 1.5              # window hop (50% overlap)
MIN_VOICED_PER_WINDOW = 8 # skip windows with too little singing to judge
ACTIVITY_FULL = 2.5       # entropy (bits) treated as "fully varied" when normalising
WEIGHTS = dict(intonation=0.35, activity=0.25, stability=0.20, participation=0.20)


# ----------------------------------------------------------------------
# 1. Get a pitch track (midi per frame, NaN where unvoiced) from CSV or audio
# ----------------------------------------------------------------------
def load_pitch(path):
    if str(path).lower().endswith(".csv"):
        df = pd.read_csv(path)
        if "time_s" not in df:
            raise ValueError("CSV must have a 'time_s' column")
        if "midi" in df:
            midi = df["midi"].to_numpy(dtype=float)
        elif "f0_hz" in df:
            import librosa
            midi = librosa.hz_to_midi(df["f0_hz"].to_numpy(dtype=float))
        else:
            raise ValueError("CSV must have a 'midi' or 'f0_hz' column")
        return df["time_s"].to_numpy(dtype=float), midi

    # audio path -> run pyin
    import librosa
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    f0, voiced_flag, _ = librosa.pyin(y, fmin=80, fmax=600, sr=SR, frame_length=2048)
    midi = librosa.hz_to_midi(np.where(voiced_flag, f0, np.nan))
    times = librosa.times_like(f0, sr=SR)
    return times, midi


# ----------------------------------------------------------------------
# 2. Component measures on one window's worth of pitch
# ----------------------------------------------------------------------
def _cents_to_grid(midi):
    return (midi - np.round(midi)) * 100.0

def estimate_tuning_offset(midi_v):
    """The clip's own tuning offset from the A440 grid, in cents.

    Crowds and stadium bands are routinely sharp/flat of absolute A440 tuning.
    Measuring against A440 made a tight-but-offset crowd score BELOW random
    noise (the Wales finding). Deviations from the nearest semitone live on a
    100-cent cycle, so the offset is found with a circular mean."""
    c = _cents_to_grid(midi_v)
    ang = 2.0 * np.pi * c / 100.0
    return 100.0 / (2.0 * np.pi) * np.angle(np.mean(np.exp(1j * ang)))


def intonation(midi_v, tuning_offset=0.0):
    """% of voiced frames within GRID_TOL_CENTS of a note on the clip's OWN
    tuning grid (tuning-invariant; 0..100). Noise floor is ~40% by geometry
    (a 40-cent tolerance band on a 100-cent cycle)."""
    if len(midi_v) == 0:
        return np.nan
    d = (_cents_to_grid(midi_v) - tuning_offset + 50.0) % 100.0 - 50.0
    return 100.0 * np.mean(np.abs(d) < GRID_TOL_CENTS)

def intonation_R(midi_v):
    """Circular concentration of pitch around its grid: 0 = uniform (noise),
    ~1 = perfectly on one tuning. Threshold-free and tuning-invariant by
    construction (it measures peakedness, not where the peak sits)."""
    if len(midi_v) < 2:
        return np.nan
    return float(np.abs(np.mean(np.exp(1j * 2 * np.pi * _cents_to_grid(midi_v) / 100.0))))


def activity(midi_v):
    """Entropy (bits) of the sung-note distribution. 0 = one note (drone)."""
    if len(midi_v) < 2:
        return np.nan
    _, counts = np.unique(np.round(midi_v), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

def stability(midi_v):
    """Fraction of consecutive voiced frames that hold pitch (|delta| < 0.3 semitone).
    High for clean sustained notes, low for smeared / jumpy noise."""
    if len(midi_v) < 2:
        return np.nan
    return float(np.mean(np.abs(np.diff(midi_v)) < 0.3))

def participation(voiced_mask):
    """Fraction of frames in the window that carry a confident pitch (0..1)."""
    return float(np.mean(voiced_mask)) if len(voiced_mask) else np.nan


# ----------------------------------------------------------------------
# 3. Window the whole clip and aggregate robustly (median over windows)
# ----------------------------------------------------------------------
def windowed_components(times, midi):
    win = int(round(WINDOW_S * FPS))
    hop = int(round(HOP_S * FPS))
    rows, offsets = [], []

    for start in range(0, max(len(midi) - 1, 1), hop):
        sl = slice(start, start + win)
        seg = midi[sl]
        if len(seg) < win // 2:
            continue
        voiced_mask = ~np.isnan(seg)
        midi_v = seg[voiced_mask]
        if len(midi_v) < MIN_VOICED_PER_WINDOW:
            continue
        tw = estimate_tuning_offset(midi_v)        # each window finds its OWN tuning
        offsets.append(tw)
        rows.append(dict(
            t=times[start] if start < len(times) else times[-1],
            intonation=intonation(midi_v, tw),
            intonation_R=intonation_R(midi_v),
            activity=activity(midi_v),
            stability=stability(midi_v),
            participation=participation(voiced_mask),
        ))
    df = pd.DataFrame(rows)
    df.attrs["tuning_offset_cents"] = float(np.median(offsets)) if offsets else 0.0
    return df


def soloist_warning(win_df, r_high=0.75, clip_median_max=0.50, min_run=3):
    """QC: flag possible soloist/PA-dominated segments.

    Signature: a run of consecutive windows with very concentrated pitch
    (intonation_R > r_high) inside a clip whose overall median R is much lower.
    A lone miked voice tracks far cleaner than a massed crowd, so such a run
    usually means the broadcast cut to a soloist or the PA. With one clip per
    nation this cannot be corrected afterwards -- review flagged clips manually.
    Returns a list of (t_start, t_end) segments, empty if clean.
    """
    if win_df["intonation_R"].median() >= clip_median_max:
        return []
    hot = (win_df["intonation_R"] > r_high).to_numpy()
    t = win_df["t"].to_numpy()
    segs, run_start = [], None
    for i, h in enumerate(hot):
        if h and run_start is None:
            run_start = i
        elif not h and run_start is not None:
            if i - run_start >= min_run:
                segs.append((float(t[run_start]), float(t[i - 1])))
            run_start = None
    if run_start is not None and len(hot) - run_start >= min_run:
        segs.append((float(t[run_start]), float(t[-1])))
    return segs


def aggregate(win_df):
    """Robust (median) summary across windows -> one value per component."""
    return {c: float(np.nanmedian(win_df[c])) for c in
            ["intonation", "intonation_R", "activity", "stability", "participation"]}


# ----------------------------------------------------------------------
# 4. Random-noise baseline (what each component scores on pure noise)
# ----------------------------------------------------------------------
def chance_baseline(times, midi, n_trials=8, seed=0):
    rng = np.random.default_rng(seed)
    voiced = midi[~np.isnan(midi)]
    if len(voiced) < 10:
        return {c: np.nan for c in ["intonation","intonation_R","activity","stability","participation"]}
    lo, hi = np.nanpercentile(voiced, 2), np.nanpercentile(voiced, 98)
    voiced_pattern = ~np.isnan(midi)
    cols = ["intonation", "intonation_R", "activity", "stability", "participation"]
    out = {c: [] for c in cols}
    for _ in range(n_trials):
        fake = np.full(len(midi), np.nan)
        fake[voiced_pattern] = rng.uniform(lo, hi, voiced_pattern.sum())
        agg = aggregate(windowed_components(times, fake))
        for c in cols:
            out[c].append(agg[c])
    return {c: float(np.nanmedian(v)) for c, v in out.items()}


# ----------------------------------------------------------------------
# 5. Provisional per-clip composite (final version normalises across nations)
# ----------------------------------------------------------------------
def composite(agg):
    norm = dict(
        intonation=agg["intonation"] / 100.0,
        activity=min(agg["activity"] / ACTIVITY_FULL, 1.0),
        stability=agg["stability"],
        participation=agg["participation"],
    )
    return sum(WEIGHTS[c] * norm[c] for c in WEIGHTS), norm


def plot(win_df, label, out_png):
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[1, 1])
    ax[0].set_title(f"{label}: components over time (watch for soloist vs crowd shifts)")
    for c, col in [("intonation", "#185FA5"), ("stability", "#1D9E75")]:
        ax[0].plot(win_df["t"], win_df[c], marker="o", ms=3, label=c, color=col)
    ax[0].set_ylabel("%  /  fraction x100"); ax[0].legend(loc="lower left"); ax[0].grid(alpha=0.3)
    ax[1].plot(win_df["t"], win_df["activity"], marker="o", ms=3, label="activity (bits)", color="#BA7517")
    ax[1].plot(win_df["t"], win_df["participation"] * 100, marker="o", ms=3,
               label="participation x100", color="#993556")
    ax[1].set_xlabel("time (s)"); ax[1].legend(loc="lower left"); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()


def score_clip(input_path, label, outdir="data/results"):
    """Score one clip. Returns (aggregates, chance, composite, paths).
    Callable from run_anthem.py; the CLI below is a thin wrapper."""
    import os
    os.makedirs(outdir, exist_ok=True)

    times, midi = load_pitch(input_path)
    win_df = windowed_components(times, midi)
    if win_df.empty:
        raise RuntimeError(
            "No scorable windows (too little confident singing). "
            "Check the clip / separation.")

    agg = aggregate(win_df)
    chance = chance_baseline(times, midi)
    score, norm = composite(agg)
    out_png = os.path.join(outdir, f"{label}_rendition.png")
    out_csv = os.path.join(outdir, f"{label}_windows.csv")
    plot(win_df, label, out_png)
    win_df.assign(nation=label).to_csv(out_csv, index=False)

    print(f"\n=== Rendition score: {label} ===")
    print(f"  ({len(win_df)} scored windows; "
          f"clip tuning offset {win_df.attrs.get('tuning_offset_cents', 0):+.0f} cents vs A440)\n")
    print(f"  {'component':<14}{'value':>10}{'noise baseline':>16}")
    for c, unit in [("intonation", "%"), ("intonation_R", ""), ("activity", "bits"),
                    ("stability", ""), ("participation", "")]:  # participation baseline = value by construction
        print(f"  {c:<14}{agg[c]:>9.2f}{unit:<1}{chance[c]:>14.2f}")
    print(f"\n  COMPOSITE (provisional): {score:.3f}")
    print(f"  saved plot: {out_png}  and per-window data: {out_csv}")

    segs = soloist_warning(win_df)
    if segs:
        pretty = ", ".join(f"{a:.0f}-{b:.0f}s" for a, b in segs)
        print(f"\n  [QC WARNING] possible soloist/PA-dominated segment(s): {pretty}")
        print("  A lone miked voice scores far cleaner than a crowd. Review this")
        print("  clip manually before including it in the ranking.")

    return agg, chance, score, dict(png=out_png, windows_csv=out_csv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="audio file (wav) or pitch CSV")
    ap.add_argument("--label", default=None)
    ap.add_argument("--outdir", default="data/results",
                    help="directory for the PNG and windows CSV (default: data/results)")
    args = ap.parse_args()
    label = args.label or args.input.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
    try:
        score_clip(args.input, label, args.outdir)
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()