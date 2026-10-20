"""Generate the image tiles embedded in the Fig. 1 flowchart (figs/fw_*.png).

The flowchart itself is drawn in TikZ inside main.tex; this script only renders
the data-bearing thumbnails it places: the waveform of one CREMA-D clip, that
clip's spectrogram in each of the four conditions, and a slice of the feature
table. Everything is computed from the committed stimuli and feature table.

Run with the xai_ser environment (needs librosa):
    ../xai_ser/.venv/bin/python make_fig_framework.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import librosa
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "xai_ser" / "data"
OUT = HERE / "figs"
CLIP = "1001_DFA_ANG_XX"
COND_FILES = {"ref": "ref.wav", "mp3_64": "mp3_64.mp3",
              "mp4_aac64": "mp4_aac64.mp4", "roundtrip_wav": "roundtrip_wav.wav"}
SR = 16000


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", pad_inches=0,
                transparent=True)
    plt.close(fig)


def waveform() -> None:
    y, _ = librosa.load(DATA / "stimuli" / CLIP / "ref.wav", sr=SR)
    fig, ax = plt.subplots(figsize=(1.6, 0.45))
    ax.plot(np.arange(len(y)) / SR, y, lw=0.35, color="#1f3b57")
    ax.set_xlim(0, len(y) / SR)
    ax.axis("off")
    save(fig, "fw_wave.png")


def spectrograms() -> None:
    # One shared dB scale, so the band MP3 removes reads as removed and not as
    # a per-panel normalisation artefact.
    specs = {}
    for cond, fn in COND_FILES.items():
        y, _ = librosa.load(DATA / "stimuli" / CLIP / fn, sr=SR)
        S = np.abs(librosa.stft(y, n_fft=512, hop_length=128)) ** 2
        specs[cond] = librosa.power_to_db(S, ref=1.0)
    vmax = max(s.max() for s in specs.values())
    for cond, S in specs.items():
        fig, ax = plt.subplots(figsize=(1.25, 0.8))
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma",
                  vmin=vmax - 80, vmax=vmax, interpolation="bilinear")
        ax.axis("off")
        save(fig, f"fw_spec_{cond}.png")


def feature_slice() -> None:
    f = sorted(glob.glob(str(DATA / "features" / "*.parquet")))[-1]
    d = pd.read_parquet(f)
    d = d[(d["condition"] == "ref") & d["extract_ok"].astype(bool)]
    fams = ("spectral_", "mfcc_", "prosody_", "chroma_")
    cols = [c for fam in fams for c in d.columns if c.startswith(fam)]
    rows = d.sample(36, random_state=42)[cols].to_numpy(float)
    full = d[cols].to_numpy(float)
    z = (rows - np.nanmean(full, 0)) / (np.nanstd(full, 0) + 1e-9)
    z = np.clip(np.nan_to_num(z), -2.5, 2.5)
    fig, ax = plt.subplots(figsize=(1.9, 0.75))
    ax.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
              interpolation="nearest")
    ax.axis("off")
    save(fig, "fw_features.png")
    counts = [sum(c.startswith(fam) for c in cols) for fam in fams]
    print("family column counts in tile:", dict(zip(fams, counts)))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    waveform()
    spectrograms()
    feature_slice()
    print("wrote", sorted(p.name for p in OUT.glob("fw_*.png")))
