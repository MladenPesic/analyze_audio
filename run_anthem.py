# run_anthem.py
"""Run the full pipeline for one anthem clip (or all of them).

Stages (each cached -- re-run with --force to recompute):
    [1/4] download   YouTube clip -> data/raw/{video_id}.wav
    [2/4] separate   Demucs vocal isolation -> data/stems/htdemucs_ft/{id}/vocals.wav
    [3/4] pitch      pyin F0 extraction -> data/results/{video_id}_pitch.csv
    [4/4] score      reference-free rendition score -> data/results/{Country}_*.{png,csv}

Usage:
    python run_anthem.py wales
    python run_anthem.py wales --force        # recompute every stage
    python run_anthem.py --all                # process every registry entry

After all tournament clips are scored, build the ranking with:
    python aggregate_rank.py "data/results/*_windows.csv"
(remember to exclude calibration clips from the glob / folder first)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from urllib.parse import urlparse, parse_qs

from anthems import ANTHEMS


def _video_id(url: str) -> str:
    u = urlparse(url)
    if u.hostname == "youtu.be":
        return u.path.lstrip("/")
    qs = parse_qs(u.query)
    if "v" in qs:
        return qs["v"][0]
    raise ValueError(f"Cannot parse a YouTube video id from: {url}")
from download import download_audio
from separate import separate_vocals
from analyze import analyze_pitch, summarize
from score_rendition import score_clip


def run_pipeline(key: str, force: bool = False):
    if key not in ANTHEMS:
        print(f"Unknown anthem: {key}. Available: {list(ANTHEMS.keys())}")
        sys.exit(1)
    cfg = ANTHEMS[key]
    vid = _video_id(cfg["url"])
    raw_wav = Path(f"data/raw/{vid}.wav")
    vocals_wav = Path(f"data/stems/htdemucs_ft/{vid}/vocals.wav")
    pitch_csv = Path(f"data/results/{vid}_pitch.csv")

    print(f"\n{'=' * 60}")
    print(f"  {cfg['country']}   [dataset: {cfg['dataset']}]")
    print(f"{'=' * 60}\n")

    # [1/4] Download
    if force or not raw_wav.exists():
        print("[1/4] Downloading...")
        download_audio(cfg["url"], "data/raw",
                       start=cfg["start"], end=cfg["end"])
    else:
        print(f"[1/4] Using cached {raw_wav}")

    # [2/4] Separate vocals
    if force or not vocals_wav.exists():
        print("[2/4] Separating vocals (slow on CPU -- minutes per clip)...")
        separate_vocals(raw_wav)
    else:
        print(f"[2/4] Using cached {vocals_wav}")

    # [3/4] Extract pitch
    if force or not pitch_csv.exists():
        print("[3/4] Extracting pitch...")
        res = analyze_pitch(vocals_wav)
        stats = summarize(res)
        print(f"      {stats['voiced_frames']} voiced frames "
              f"({stats['voiced_pct']}% of clip)")
        pitch_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "time_s": res["times"], "f0_hz": res["f0_hz"], "midi": res["midi"],
            "nearest_midi": res["nearest_midi"], "cents_deviation": res["cents"],
            "voiced_prob": res["voiced_prob"],
        }).to_csv(pitch_csv, index=False)
    else:
        print(f"[3/4] Using cached {pitch_csv}")

    # [4/4] Rendition score (reference-free)
    print("[4/4] Scoring rendition...")
    score_clip(str(pitch_csv), label=cfg["country"], outdir="data/results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key", nargs="?", help="anthem key from anthems.py")
    ap.add_argument("--all", action="store_true", help="process every registry entry")
    ap.add_argument("--force", action="store_true", help="recompute all cached stages")
    args = ap.parse_args()

    if args.all:
        for key in ANTHEMS:
            run_pipeline(key, force=args.force)
    elif args.key:
        run_pipeline(args.key, force=args.force)
    else:
        print("Usage: python run_anthem.py <anthem_key> [--force]   or   --all")
        print(f"Available: {list(ANTHEMS.keys())}")
        sys.exit(1)


if __name__ == "__main__":
    main()