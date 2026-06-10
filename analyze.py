# analyze.py
import numpy as np
import librosa
import pandas as pd
from pathlib import Path

def analyze_pitch(vocals_path: Path, sr: int = 22050,
                  fmin_hz: float = 80.0, fmax_hz: float = 600.0):
    """Extract F0 per frame and cent deviations from the nearest A440 semitone.

    Note: the cent deviations here are measured against the absolute A440
    equal-tempered grid. They are DESCRIPTIVE diagnostics only -- crowds sing
    at their own tuning, so grid-based numbers must not be used to compare
    nations. The ranking metrics (tuning-invariant) live in score_rendition.py.
    """
    y, sr = librosa.load(str(vocals_path), sr=sr, mono=True)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=sr,
        frame_length=2048,
        fill_na=np.nan,
    )
    times = librosa.times_like(f0, sr=sr)

    f0_clean = np.where(voiced_flag, f0, np.nan)

    midi = librosa.hz_to_midi(f0_clean)
    nearest = np.round(midi)
    cents = (midi - nearest) * 100.0

    return {
        "times": times,
        "f0_hz": f0_clean,
        "midi": midi,
        "nearest_midi": nearest,
        "cents": cents,
        "voiced_prob": voiced_prob,
    }

def summarize(result: dict) -> dict:
    """Descriptive stats of the pitch track.

    pct_grid_within_* measure proximity to the absolute A440 semitone grid
    ("is the pitch near SOME standard note"), NOT singing quality and NOT the
    ranking metric. Named with a grid_ prefix to avoid confusion with the
    tuning-invariant intonation measures in score_rendition.py."""
    cents = result["cents"]
    cents = cents[~np.isnan(cents)]
    if len(cents) == 0:
        return {"error": "no voiced frames detected"}

    total_frames = len(result["cents"])
    return {
        "total_frames": total_frames,
        "voiced_frames": int(len(cents)),
        "voiced_pct": round(100 * len(cents) / total_frames, 1),
        "mean_cents_offset": round(float(np.mean(cents)), 2),
        "median_cents_offset": round(float(np.median(cents)), 2),
        "abs_mean_cents": round(float(np.mean(np.abs(cents))), 2),
        "std_cents": round(float(np.std(cents)), 2),
        "pct_grid_within_25c": round(float(np.mean(np.abs(cents) < 25) * 100), 1),
        "pct_grid_within_50c": round(float(np.mean(np.abs(cents) < 50) * 100), 1),
    }

if __name__ == "__main__":
    video_id = "ZwdZOHm8r-Y"
    vocals_path = Path(f"data/stems/htdemucs_ft/{video_id}/vocals.wav")

    print(f"Analyzing: {vocals_path}")
    res = analyze_pitch(vocals_path)
    stats = summarize(res)

    print("\n=== Pitch Analysis Results ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "time_s": res["times"],
        "f0_hz": res["f0_hz"],
        "midi": res["midi"],
        "nearest_midi": res["nearest_midi"],
        "cents_deviation": res["cents"],
        "voiced_prob": res["voiced_prob"],
    })
    csv_path = out_dir / f"{video_id}_pitch.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull pitch data saved to: {csv_path}")