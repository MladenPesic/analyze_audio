# compare.py
"""Generate a comparison table and chart across all processed anthems."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import librosa
from anthems import ANTHEMS
from extract_melody import extract_melody_track, melody_to_pitch_series
from align_score import (
    dtw_align, search_best_transposition, score_per_reference_note, wrap_to_octave
)


def score_anthem(key: str) -> dict | None:
    """Run alignment + scoring for one anthem. Returns metrics dict or None."""
    cfg = ANTHEMS[key]
    vid = cfg["video_id"]
    pitch_csv = Path(f"data/results/{vid}_pitch.csv")
    if not pitch_csv.exists():
        print(f"  {key}: pitch CSV missing — run pipeline first")
        return None

    # Crowd
    df = pd.read_csv(pitch_csv)
    crowd_full = df["midi"].values
    crowd_times = df["time_s"].values
    fps = 1.0 / np.median(np.diff(crowd_times))
    crowd = crowd_full[~np.isnan(crowd_full)]

    # Reference
    notes = extract_melody_track(Path(cfg["midi_path"]), cfg["melody_track"])
    _, ref_full = melody_to_pitch_series(notes, sr_frames_per_sec=fps)
    ref = ref_full[~np.isnan(ref_full)]

    # Align
    best_shift, _, path = search_best_transposition(crowd, ref, range_semitones=12)
    ref_shifted = ref + best_shift

    # Score
    note_devs = wrap_to_octave(score_per_reference_note(crowd, ref_shifted, path))

    return {
        "key": key,
        "country": cfg["country"],
        "anthem": cfg["anthem_name"],
        "transposition": best_shift,
        "n_notes": len(note_devs),
        "median_abs_cents": float(np.median(np.abs(note_devs))),
        "pct_within_50": float(np.mean(np.abs(note_devs) < 50) * 100),
        "pct_within_100": float(np.mean(np.abs(note_devs) < 100) * 100),
        "mean_offset": float(np.mean(note_devs)),
        "deviations": note_devs,  # for plotting
    }


def build_comparison_report():
    print("Scoring all anthems...\n")
    results = []
    for key in ANTHEMS:
        r = score_anthem(key)
        if r:
            print(f"  {r['country']}: {r['pct_within_50']:.1f}% within ±50¢")
            results.append(r)

    if len(results) < 1:
        print("No results to compare.")
        return

    # --- Summary table ---
    table = pd.DataFrame([{
        "Country": r["country"],
        "Anthem": r["anthem"],
        "Notes within ±50¢ (%)": round(r["pct_within_50"], 1),
        "Notes within ±100¢ (%)": round(r["pct_within_100"], 1),
        "Median |deviation| (¢)": round(r["median_abs_cents"], 1),
        "Mean offset (¢, +sharp/-flat)": round(r["mean_offset"], 1),
    } for r in results])
    table = table.sort_values("Notes within ±50¢ (%)", ascending=False).reset_index(drop=True)
    table.index = table.index + 1
    table.index.name = "Rank"

    print("\n" + "="*80)
    print("  ANTHEM SCOREBOARD — Most In-Tune Football Fans")
    print("="*80)
    print(table.to_string())

    out_csv = Path("data/results/comparison.csv")
    table.to_csv(out_csv)
    print(f"\nSaved table: {out_csv}")

    # --- Comparison chart ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    countries = [r["country"] for r in results]
    pct_50 = [r["pct_within_50"] for r in results]
    offsets = [r["mean_offset"] for r in results]

    order = np.argsort(pct_50)[::-1]
    countries_sorted = [countries[i] for i in order]
    pct_sorted = [pct_50[i] for i in order]
    offsets_sorted = [offsets[i] for i in order]

    bars = axes[0].barh(countries_sorted, pct_sorted, color="#2ca02c", alpha=0.8)
    axes[0].set_xlabel("% of notes within ±50¢ of correct pitch")
    axes[0].set_title("In-tune-ness ranking — higher is better")
    axes[0].grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, pct_sorted):
        axes[0].text(v + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{v:.1f}%", va="center")

    colors = ["#1f77b4" if o < 0 else "#d62728" for o in offsets_sorted]
    axes[1].barh(countries_sorted, offsets_sorted, color=colors, alpha=0.8)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Mean pitch offset (cents) — negative = flat, positive = sharp")
    axes[1].set_title("Sharp vs. flat tendency")
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    out_png = Path("data/results/comparison.png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved chart: {out_png}")
    plt.show()


if __name__ == "__main__":
    build_comparison_report()