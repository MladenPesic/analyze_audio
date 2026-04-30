# 02_separate.py
import subprocess
from pathlib import Path

def separate_vocals(wav_path: Path, out_dir: str = "data/stems",
                    model: str = "htdemucs_ft") -> Path:
    """Run Demucs and return path to the vocals stem."""
    cmd = [
        "demucs",
        "-n", model,
        "--two-stems", "vocals",   # we only need vocals vs everything else
        "-o", out_dir,
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)
    return Path(out_dir) / model / wav_path.stem / "vocals.wav"

if __name__ == "__main__":
    vocals = separate_vocals(Path("data/raw/YOUR_ID.wav"))
    print(f"Vocals: {vocals}")