# download.py
import os
from pathlib import Path

import yt_dlp


def download_audio(url: str, out_dir: str, start: str = None, end: str = None) -> Path:
    """Download audio from YouTube, optionally trimmed to [start, end] (HH:MM:SS).

    ffmpeg resolution: if the FFMPEG_PATH environment variable is set, it is
    passed to yt-dlp; otherwise yt-dlp finds ffmpeg on the system PATH. No
    machine-specific paths are hardcoded.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{out_dir}/%(id)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "0",
        }],
        "postprocessor_args": [],
    }
    ffmpeg_path = os.environ.get("FFMPEG_PATH")
    if ffmpeg_path:
        ydl_opts["ffmpeg_location"] = ffmpeg_path

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
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return h * 3600 + m * 60 + s