"""Arousal arm: binary high/low, with a bias-free DV and the missing null control.

Why this replaces the six-way arm as primary:

  six-way    accuracy 0.20 vs 0.167 chance; 87.7% `neutral`; 15/50 items unpinned
  arousal    accuracy 0.68 vs 0.500 chance; AUC 0.750 (p=.0025); 25/50 unpinned

Removing `neutral` from the option set removes the model's escape hatch, and the
25/25 class split means the majority-class baseline is 0.5 rather than 0.81 (the
problem that made a valence split uninterpretable).

Two further improvements over the earlier design:

  * The DV is **P(high)**, a FIXED label, not P(gold). A model that always
    answers one label scores AUC 0.5 on it no matter how confident it is, so
    response bias cannot masquerade as signal. It also means the prompt no
    longer names the correct answer.

  * A **1-LSB dither condition** supplies the control the earlier study lacked:
    the smallest waveform change representable at 16 bits (~-96 dBFS, inaudible,
    ~10^4x smaller than the codec's own noise). If attribution maps diverge as
    much under 1-LSB dither as under MP3, then the maps are simply unstable and
    the explanation finding collapses. This is the decisive test.
"""
from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp.client import call, label_mass, pmap, print_stats
from exp.run_xai import (BANDS, N_TIME, build_masks, read_wav, to_mp3_and_back,
                         wav_bytes)

ROOT = Path(__file__).resolve().parent.parent
STIM = ROOT / "data" / "stimuli"
OUT = ROOT / "exp" / "out"
SR = 16000

AROUSAL = {"ANG": "high", "FEA": "high", "HAP": "high",
           "SAD": "low", "NEU": "low", "DIS": "low"}
LABELS = ["high", "low"]

PROMPT = ("Listen to this audio. Is the speaker's vocal energy high or low? "
          "The audio is present; do not ask for it. "
          "Answer with exactly one word: high or low.")

GRID_CONDITIONS = [
    ("ref", "ref.wav", "wav"), ("mp3_32", "mp3_32.mp3", "mp3"),
    ("mp3_64", "mp3_64.mp3", "mp3"), ("mp3_128", "mp3_128.mp3", "mp3"),
    ("rt_mp3_32", "rt_mp3_32.wav", "wav"), ("rt_mp3_64", "rt_mp3_64.wav", "wav"),
    ("rt_mp3_128", "rt_mp3_128.wav", "wav"),
]


def dither_1lsb(x: np.ndarray, seed: int) -> np.ndarray:
    """Add +-1 least-significant-bit of 16-bit noise. Inaudible by construction."""
    rng = np.random.default_rng(seed)
    step = 1.0 / 32768.0
    return x + rng.choice([-step, step], size=len(x)).astype(np.float32)


def p_high(resp) -> float | None:
    m = label_mass(resp, LABELS)
    return m.get("high")


def main() -> int:
    items = sorted(d.name for d in STIM.iterdir() if d.is_dir())

    # ---------------- part 1: format grid --------------------------------
    print("PART 1  format grid (7 conditions x 50 items)")
    jobs = [{"item_id": it, "condition": c, "file": f, "fmt": fm}
            for it in items for c, f, fm in GRID_CONDITIONS]

    def w1(j):
        r = call(STIM / j["item_id"] / j["file"], PROMPT, max_tok=12)
        said = ""
        if "__error__" not in r:
            said = (r["choices"][0]["message"]["content"] or "").strip().lower().strip(".,!'\" ")
        aro = AROUSAL[j["item_id"].split("_")[2]]
        return {**{k: v for k, v in j.items() if k != "file"},
                "arousal": aro, "said": said, "correct": said == aro,
                "p_high": p_high(r), "error": r.get("__error__")}

    grid = pd.DataFrame(pmap(w1, jobs, workers=12, desc="grid"))
    grid.to_parquet(OUT / "arousal_grid.parquet", index=False)

    # ---------------- part 2: occlusion, incl. the dither control --------
    n_masks = 1 + N_TIME + len(BANDS) + 1
    print(f"\nPART 2  occlusion ({n_masks} masks x 4 conditions x 50 items)")
    print("  building masked stimuli...", flush=True)
    xjobs = []
    for n, it in enumerate(items, 1):
        x = read_wav(STIM / it / "ref.wav")
        xd = dither_1lsb(x, seed=hash(it) % (2 ** 31))
        for mid, kind, y in build_masks(x):
            w = wav_bytes(y)
            mp3, rt = to_mp3_and_back(w, 64)
            # the same mask, applied to the dithered reference
            yd = y + (xd - x)
            wd = wav_bytes(yd)
            for cond, payload, fmt in (("ref", w, "wav"), ("mp3_64", mp3, "mp3"),
                                       ("rt_mp3_64", rt, "wav"),
                                       ("ref_dither", wd, "wav")):
                xjobs.append({"item_id": it, "mask_id": mid, "mask_kind": kind,
                              "condition": cond, "bytes": payload, "fmt": fmt})
        if n % 10 == 0 or n == len(items):
            print(f"    {n}/{len(items)} items prepared", flush=True)

    def w2(j):
        sha = hashlib.sha256(j["bytes"]).hexdigest()
        r = call(Path("/dev/null"), PROMPT, max_tok=12,
                 audio_bytes=j["bytes"], audio_sha=sha, fmt=j["fmt"])
        return {k: v for k, v in j.items() if k != "bytes"} | {
            "p_high": p_high(r), "error": r.get("__error__")}

    xai = pd.DataFrame(pmap(w2, xjobs, workers=12, desc="xai"))
    base = (xai[xai.mask_id == "unmasked"]
            .set_index(["item_id", "condition"])["p_high"].rename("p_base"))
    xai = xai.join(base, on=["item_id", "condition"])
    xai["attribution"] = xai.p_base - xai.p_high
    xai.to_parquet(OUT / "arousal_xai.parquet", index=False)
    print_stats()

    # ---------------- quick readout ---------------------------------------
    print("\n" + "=" * 72)
    print("AROUSAL GRID — accuracy and P(high) by condition")
    print("=" * 72)
    order = ["ref", "rt_mp3_32", "rt_mp3_64", "rt_mp3_128",
             "mp3_32", "mp3_64", "mp3_128"]
    t = grid.groupby("condition").agg(acc=("correct", "mean"),
                                      p_high=("p_high", "mean"),
                                      n=("correct", "size"))
    print(t.reindex([c for c in order if c in t.index]).round(4).to_string())
    print(f"\n  refusals / unparsed: "
          f"{int((~grid.said.isin(LABELS)).sum())}/{len(grid)}")
    print(f"  overall said: {grid.said.value_counts().head(4).to_dict()}")

    print("\n  sanity — the dither must be inaudible AND must change the samples:")
    x = read_wav(STIM / items[0] / "ref.wav")
    xd = dither_1lsb(x, 1)
    diff = np.abs(xd - x)
    print(f"    samples changed : {int((diff > 0).sum())}/{len(x)}")
    print(f"    max change      : {diff.max()*32768:.2f} LSB "
          f"({20*np.log10(diff.max()+1e-20):.1f} dBFS)")
    print(f"    codec noise @64k: ~25 dB SNR, i.e. ~{10**((0-(-96))/20):.0f}x larger")
    return 0


if __name__ == "__main__":
    sys.exit(main())
