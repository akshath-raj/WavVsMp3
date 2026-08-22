"""Uniform audio decoding.

Feature extraction must not be confounded by *which decoder* read a file, so
every stimulus — WAV, MP3 or MP4/AAC — is decoded through the same ffmpeg
pipeline to float32 PCM at a fixed sample rate. soundfile/audioread would
silently take different code paths per container.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


class DecodeError(RuntimeError):
    pass


def decode(path: str | Path, sr: int = 16000, mono: bool = True) -> np.ndarray:
    """Decode `path` to a 1-D float32 array in [-1, 1] at `sr` Hz."""
    cmd = [
        FFMPEG, "-v", "error", "-nostdin",
        "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(sr),
        "-ac", "1" if mono else "2",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise DecodeError(f"ffmpeg failed on {path}: {proc.stderr.decode()[:400]}")
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if y.size == 0:
        raise DecodeError(f"empty decode for {path}")
    return np.ascontiguousarray(y)


def probe_duration(path: str | Path) -> float:
    """Container-reported duration in seconds (0.0 if unavailable)."""
    cmd = [
        FFPROBE, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    try:
        return float(proc.stdout.decode().strip())
    except (ValueError, AttributeError):
        return 0.0
