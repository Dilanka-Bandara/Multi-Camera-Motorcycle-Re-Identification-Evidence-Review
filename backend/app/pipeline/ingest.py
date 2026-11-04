"""Offline video probing and browser-safe transcoding."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path

from app.schemas import VideoMeta


def probe_video(path: Path, start_time_s: float) -> VideoMeta:
    """Read video metadata using ffprobe, including a frame-count fallback."""
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout)["streams"]
    if not streams:
        raise ValueError(f"No video stream in {path}")
    stream = streams[0]
    rate = Fraction(stream["avg_frame_rate"])
    if rate <= 0:
        raise ValueError("Video has no positive frame rate")
    count = stream.get("nb_frames") or stream.get("nb_read_frames")
    if count in (None, "N/A"):
        count = stream.get("nb_read_frames")
    return VideoMeta(
        path=str(path),
        fps=float(rate),
        frame_count=int(count),
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration_s=float(stream.get("duration") or int(count) / rate),
        start_time_s=start_time_s,
    )


def ingest_video(
    source: Path, destination: Path, start_time_s: float, max_height: int
) -> VideoMeta:
    """Transcode without replacing existing media, retaining frame timestamps."""
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            "-vf",
            f"scale=-2:min({max_height}\\,ih)",
            "-fps_mode",
            "passthrough",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return probe_video(destination, start_time_s)
