"""Occlusion attribution across formats.

Method. For each item we mask one region of the CANONICAL reference waveform,
then regenerate every format condition FROM the masked signal. That preserves
the single-ancestor rule: the masked stimulus in condition X is exactly what
condition X would be if that region were absent, rather than a masked version of
an already-degraded signal.

Attribution uses the continuous DV, which is what makes this tractable at all:

    a_j = P(gold | unmasked) - P(gold | mask_j)

Positive a_j means masking region j removed evidence for the true emotion, i.e.
the model's output depended on that region. Because the backend is
bit-deterministic, a_j carries zero measurement noise -- no repeats needed, and
the within-condition stability floor is exactly 0.

Masks: 10 equal time windows + 6 frequency bands + 1 null control (a low-energy
region, which should produce near-zero attribution if the method is sound).

Conditions: ref (lossless WAV), mp3_64 (MP3 container), rt_mp3_64 (WAV container
carrying the identical mp3-decoded signal). Comparing attribution maps between
ref and rt_mp3_64 isolates the CODEC's effect on the evidence base; between
mp3_64 and rt_mp3_64 isolates the CONTAINER's.
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp.client import STATS, call, gold_of, label_mass, pmap, print_stats
from exp.run_grid import anchor_for, binary_prompt

ROOT = Path(__file__).resolve().parent.parent
STIM = ROOT / "data" / "stimuli"
OUT = ROOT / "exp" / "out"
SR = 16000
N_TIME = 10
BANDS = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
XAI_CONDITIONS = ["ref", "mp3_64", "rt_mp3_64"]


# --------------------------------------------------------------------------
# audio helpers
# --------------------------------------------------------------------------
def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def wav_bytes(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.clip(x, -1, 1 - 1e-6).__mul__(32768).astype("<i2").tobytes())
    return buf.getvalue()


def to_mp3_and_back(wav: bytes, kbps: int = 64) -> tuple[bytes, bytes]:
    """Return (mp3 bytes, wav bytes of decode(mp3)) -- the container pair."""
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        (t / "in.wav").write_bytes(wav)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(t / "in.wav"),
                        "-c:a", "libmp3lame", "-b:a", f"{kbps}k", str(t / "a.mp3")],
                       check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(t / "a.mp3"),
                        "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le",
                        str(t / "rt.wav")], check=True, capture_output=True)
        return (t / "a.mp3").read_bytes(), (t / "rt.wav").read_bytes()


def shaped_noise(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Noise carrying the clip's long-term average spectrum.

    Flat white noise would be an obvious out-of-distribution artefact; matching
    the clip's own spectral envelope keeps the mask acoustically plausible.
    """
    spec = np.abs(np.fft.rfft(x))
    env = spec / (spec.max() + 1e-12)
    w = rng.standard_normal(n).astype(np.float32)
    W = np.fft.rfft(w)
    e = np.interp(np.linspace(0, 1, len(W)), np.linspace(0, 1, len(env)), env)
    return np.fft.irfft(W * e, n).astype(np.float32)


def mask_time(x: np.ndarray, k: int, n_win: int = N_TIME) -> np.ndarray:
    """Replace window k with loudness-matched shaped noise, cosine-ramped."""
    y = x.copy()
    edges = np.linspace(0, len(x), n_win + 1).astype(int)
    s, e = edges[k], edges[k + 1]
    seg = x[s:e]
    if e - s < 8:
        return y
    rng = np.random.default_rng(1000 + k)
    filler = shaped_noise(x, e - s, rng)
    rms_seg = np.sqrt((seg ** 2).mean()) + 1e-12
    rms_fil = np.sqrt((filler ** 2).mean()) + 1e-12
    filler *= rms_seg / rms_fil
    ramp = max(int(0.010 * SR), 1)
    ramp = min(ramp, (e - s) // 2)
    if ramp > 0:
        w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp)))
        filler[:ramp] = filler[:ramp] * w + seg[:ramp] * (1 - w)
        filler[-ramp:] = filler[-ramp:] * w[::-1] + seg[-ramp:] * (1 - w[::-1])
    y[s:e] = filler
    return y


def mask_band(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Band-stop with raised-cosine transitions (no brick-wall ringing)."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    g = np.ones_like(f)
    trans = 50.0
    stop = (f >= lo) & (f <= hi)
    g[stop] = 0.0
    for edge, inward in ((lo, +1), (hi, -1)):
        sel = (f >= edge - trans) & (f <= edge + trans)
        if sel.any():
            t = (f[sel] - (edge - trans)) / (2 * trans)
            ramp = 0.5 * (1 + np.cos(np.pi * t)) if inward > 0 else 0.5 * (1 - np.cos(np.pi * t))
            g[sel] = np.minimum(g[sel], ramp)
    return np.fft.irfft(X * g, len(x)).astype(np.float32)


def build_masks(x: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    """(mask_id, mask_kind, waveform). Includes unmasked and a null control."""
    out = [("unmasked", "none", x)]
    for k in range(N_TIME):
        out.append((f"t{k}", "temporal", mask_time(x, k)))
    for lo, hi in BANDS:
        out.append((f"f{lo}_{hi}", "spectral", mask_band(x, lo, hi)))
    # null control: mask the lowest-energy 10% window. If the method is sound,
    # attribution here should sit near zero.
    edges = np.linspace(0, len(x), N_TIME + 1).astype(int)
    energies = [(x[edges[i]:edges[i + 1]] ** 2).sum() for i in range(N_TIME)]
    out.append(("null_lowenergy", "null", mask_time(x, int(np.argmin(energies)))))
    return out


# --------------------------------------------------------------------------
def main() -> int:
    items = sorted(d.name for d in STIM.iterdir() if d.is_dir())
    n_masks = 1 + N_TIME + len(BANDS) + 1
    print(f"items {len(items)} x masks {n_masks} x conditions "
          f"{len(XAI_CONDITIONS)} = {len(items)*n_masks*len(XAI_CONDITIONS)} calls\n")

    jobs = []
    print("  building masked stimuli (ffmpeg re-encode per mask)...", flush=True)
    for n, it in enumerate(items, 1):
        x = read_wav(STIM / it / "ref.wav")
        gold = gold_of(it)
        prompt = binary_prompt(gold, anchor_for(gold))
        for mid, kind, y in build_masks(x):
            w = wav_bytes(y)
            mp3, rt = to_mp3_and_back(w, 64)
            for cond, payload, fmt in (("ref", w, "wav"),
                                       ("mp3_64", mp3, "mp3"),
                                       ("rt_mp3_64", rt, "wav")):
                jobs.append({"item_id": it, "gold": gold, "mask_id": mid,
                             "mask_kind": kind, "condition": cond,
                             "prompt": prompt, "bytes": payload, "fmt": fmt})
        if n % 10 == 0 or n == len(items):
            print(f"    {n}/{len(items)} items prepared", flush=True)

    def work(j):
        sha = hashlib.sha256(j["bytes"]).hexdigest()
        r = call(Path("/dev/null"), j["prompt"], max_tok=12,
                 audio_bytes=j["bytes"], audio_sha=sha, fmt=j["fmt"])
        m = label_mass(r, [j["gold"], anchor_for(j["gold"])])
        return {k: v for k, v in j.items() if k != "bytes"} | {
            "p_gold": m.get(j["gold"]), "error": r.get("__error__")}

    rows = pmap(work, jobs, workers=12, desc="xai")
    df = pd.DataFrame(rows)

    # attribution = drop in P(gold) caused by the mask
    base = (df[df.mask_id == "unmasked"]
            .set_index(["item_id", "condition"])["p_gold"].rename("p_base"))
    df = df.join(base, on=["item_id", "condition"])
    df["attribution"] = df.p_base - df.p_gold
    df.to_parquet(OUT / "xai.parquet", index=False)
    print_stats()

    print("\n" + "=" * 70)
    print("NULL-MASK CONTROL (should sit near zero if the method is sound)")
    print("=" * 70)
    nl = df[df.mask_kind == "null"]
    real = df[df.mask_kind.isin(["temporal", "spectral"])]
    print(f"  null-mask   mean |attribution| = {nl.attribution.abs().mean():.4f}")
    print(f"  real masks  mean |attribution| = {real.attribution.abs().mean():.4f}")
    ratio = real.attribution.abs().mean() / max(nl.attribution.abs().mean(), 1e-9)
    print(f"  ratio real/null = {ratio:.2f}x")

    print("\n" + "=" * 70)
    print("MEAN |ATTRIBUTION| BY CONDITION AND MASK KIND")
    print("=" * 70)
    print(real.groupby(["condition", "mask_kind"]).attribution.agg(
        mean_abs=lambda s: s.abs().mean(), sd="std", n="size").round(4).to_string())

    print(f"\n  saved: exp/out/xai.parquet  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
