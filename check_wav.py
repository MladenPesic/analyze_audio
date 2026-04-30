# check_wav.py
import soundfile as sf
import numpy as np
from pathlib import Path
import sys

def check(video_id: str):
    files = [
        Path(f"data/raw/{video_id}.wav"),
        Path(f"data/stems/htdemucs_ft/{video_id}/vocals.wav"),
        Path(f"data/stems/htdemucs_ft/{video_id}/no_vocals.wav"),
    ]
    for f in files:
        if not f.exists():
            print(f"{f.name}: MISSING\n"); continue
        data, sr = sf.read(str(f))
        rms = np.sqrt(np.mean(data**2))
        peak = np.max(np.abs(data))
        diag = ("SILENT" if peak < 1e-6
                else "nearly silent" if rms < 0.001
                else "looks like real audio")
        print(f"{f.name}\n  shape={data.shape}, sr={sr}, dtype={data.dtype}")
        print(f"  peak={peak:.4f}, rms={rms:.4f}\n  diagnosis: {diag}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_wav.py <video_id>")
        sys.exit(1)
    check(sys.argv[1])


