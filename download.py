# 01_download.py
import yt_dlp
from pathlib import Path

def download_audio(url: str, out_dir: str, start: str = None, end: str = None) -> Path:
    """Download audio from YouTube, optionally trimmed to [start, end] (HH:MM:SS)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{out_dir}/%(id)s.%(ext)s",
        "ffmpeg_location": r"C:\ffmpeg\bin\ffmpeg.exe",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "0",
        }],
        "postprocessor_args": [],
    }
    if start and end:
        ydl_opts["download_ranges"] = yt_dlp.utils.download_range_func(
            None, [(_to_seconds(start), _to_seconds(end))]
        )
        ydl_opts["force_keyframes_at_cuts"] = True

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(out_dir) / f"{info['id']}.wav"

def _to_seconds(t: str) -> int:
    parts = [int(p) for p in t.split(":")]
    while len(parts) < 3: parts.insert(0, 0)
    h, m, s = parts
    return h*3600 + m*60 + s

if __name__ == "__main__":
    wav = download_audio(
        "https://www.youtube.com/watch?v=ZwdZOHm8r-Y",
        "data/raw",
        start="00:00:10", end="00:01:20",   # trim to anthem only
    )
    print(f"Saved: {wav}")