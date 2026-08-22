"""Canonical project paths and small filesystem helpers.

Everything downstream resolves paths through here so that scripts can be run
from any working directory.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
RAW = DATA / "raw"
META = DATA / "meta"
STIMULI = DATA / "stimuli"
FEATURES = DATA / "features"
MANIFEST = DATA / "manifest.parquet"

OUTPUTS = ROOT / "outputs"
EDA_DIR = OUTPUTS / "eda"
MODELS_DIR = OUTPUTS / "models"
XAI_DIR = OUTPUTS / "xai"
REPORTS = ROOT / "reports"

_ALL = [DATA, RAW, META, STIMULI, FEATURES, OUTPUTS, EDA_DIR, MODELS_DIR, XAI_DIR, REPORTS]


def ensure_dirs() -> None:
    for p in _ALL:
        p.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    """UTC timestamp used to version generated artefacts."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def latest(directory: Path, pattern: str) -> Path:
    """Newest file matching `pattern`; raises if there is none."""
    hits = sorted(directory.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no file matching {pattern!r} in {directory}")
    return hits[-1]
