# run_anthem.py
"""Run the full pipeline for one anthem from the registry."""
import sys
import subprocess
from pathlib import Path
from anthems import ANTHEMS
from download import download_audio
from separate import separate_vocals
from analyze import analyze_pitch, summarize
from align_score import main as align_main
import pandas as pd
import numpy as np


def run_pipeline(key: str):
    if key not in ANTHEMS:
        print(f"Unknown anthem: {key}. Available: {list(ANTHEMS.keys())}")
        sys.exit(1)
    cfg = ANTHEMS[key]
    vid = cfg["video_id"]
    raw_wav = Path(f"data/raw/{vid}.wav")
    vocals_wav = Path(f"data/stems/htdemucs_ft/{vid}/vocals.wav")
    pitch_csv = Path(f"data/results/{vid}_pitch.csv")

    print(f"\n{'='*60}")
    print(f"  {cfg['country']} — {cfg['anthem_name']}")
    print(f"  {cfg.get('context', '')}")
    print(f"{'='*60}\n")

    # Stage 1: Download
    if not raw_wav.exists():
        print("[1/4] Downloading...")
        download_audio(cfg["youtube_url"], "data/raw",
                       start=cfg["trim_start"], end=cfg["trim_end"])
    else:
        print(f"[1/4] Using existing {raw_wav}")

    # Stage 2: Separate vocals
    if not vocals_wav.exists():
        print("[2/4] Separating vocals (this takes a few minutes)...")
        separate_vocals(raw_wav)
    else:
        print(f"[2/4] Using existing {vocals_wav}")

    # Stage 3: Extract pitch
    if not pitch_csv.exists():
        print("[3/4] Extracting pitch...")
        res = analyze_pitch(vocals_wav)
        stats = summarize(res)
        print(f"   {stats['voiced_frames']} voiced frames")
        Path("data/results").mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "time_s": res["times"], "f0_hz": res["f0_hz"], "midi": res["midi"],
            "nearest_midi": res["nearest_midi"], "cents_deviation": res["cents"],
            "voiced_prob": res["voiced_prob"],
        }).to_csv(pitch_csv, index=False)
    else:
        print(f"[3/4] Using existing {pitch_csv}")

    # Stage 4: Align and score
    print("[4/4] Aligning to reference and scoring...")
    align_main(
        video_id=vid,
        midi_path=Path(cfg["midi_path"]),
        melody_track=cfg["melody_track"],
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python run_anthem.py <anthem_key>")
        print(f"Available: {list(ANTHEMS.keys())}")
        sys.exit(1)
    run_pipeline(sys.argv[1])