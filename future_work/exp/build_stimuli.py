"""Build the WAV-vs-MP3 stimulus ladder from the canonical reference.

Conditions, all descended from one canonical `ref` (16 kHz mono PCM16, EBU R128):

    ref          lossless WAV                                    [baseline]
    mp3_32       MP3 container @  32 kbps
    mp3_64       MP3 container @  64 kbps
    mp3_128      MP3 container @ 128 kbps
    rt_mp3_32    decode(mp3_32)  -> WAV container                [control]
    rt_mp3_64    decode(mp3_64)  -> WAV container                [control]
    rt_mp3_128   decode(mp3_128) -> WAV container                [control]

The three contrasts this affords:

    rt_mp3_X  vs ref         signal effect  (codec loss; container held = WAV)
    mp3_X     vs rt_mp3_X    container effect (decoded signal held IDENTICAL)
    mp3_X     vs ref         total format effect

The container contrast is only valid if decode(mp3_X) is bit-identical to
rt_mp3_X. That is asserted per item, not assumed; a mismatch aborts the build.

Fidelity is reported after cross-correlation alignment, because MP3 encoding
inserts encoder delay -- an unaligned SNR would measure the time shift, not the
coding loss.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STIM = ROOT / "data" / "stimuli"
OUT = ROOT / "exp" / "out"
OUT.mkdir(parents=True, exist_ok=True)

BITRATES = [32, 64, 128]


def run(cmd: list[str]) -> bytes:
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{p.stderr.decode('utf-8','replace')[:500]}")
    return p.stdout


def decoded_pcm(path: Path) -> np.ndarray:
    """Decode any container through one fixed path -> 16k mono float32 in [-1,1).

    Fixed path is what makes the hash a property of the SIGNAL, not the container.
    """
    raw = run(["ffmpeg", "-v", "error", "-i", str(path),
               "-ar", "16000", "-ac", "1", "-f", "s16le", "-"])
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def pcm_sha(x: np.ndarray) -> str:
    return hashlib.sha256((x * 32768.0).astype("<i2").tobytes()).hexdigest()


def align_lag(ref: np.ndarray, other: np.ndarray, max_lag: int = 4000) -> int:
    """Lag (samples) maximising cross-correlation; positive => `other` is delayed."""
    n = min(len(ref), len(other))
    a = ref[:n] - ref[:n].mean()
    b = other[:n] - other[:n].mean()
    if a.std() == 0 or b.std() == 0:
        return 0
    corr = np.correlate(b, a, mode="full")
    lags = np.arange(-n + 1, n)
    keep = np.abs(lags) <= max_lag
    return int(lags[keep][np.argmax(corr[keep])])


def fidelity(ref: np.ndarray, other: np.ndarray) -> dict:
    """SNR and log-spectral distance AFTER removing encoder delay."""
    lag = align_lag(ref, other)
    o = other[lag:] if lag > 0 else np.concatenate([np.zeros(-lag, np.float32), other])
    n = min(len(ref), len(o))
    r, o = ref[:n], o[:n]
    noise = o - r
    snr = 10 * np.log10((r ** 2).sum() / max((noise ** 2).sum(), 1e-20))

    # log-spectral distance over 32 ms frames / 16 ms hop
    win, hop = 512, 256
    frames = max((n - win) // hop, 1)
    w = np.hanning(win).astype(np.float32)
    d = []
    for i in range(frames):
        s = i * hop
        R = np.abs(np.fft.rfft(r[s:s + win] * w)) + 1e-10
        O = np.abs(np.fft.rfft(o[s:s + win] * w)) + 1e-10
        d.append(np.sqrt(np.mean((20 * np.log10(R) - 20 * np.log10(O)) ** 2)))
    return {"lag_samples": lag, "lag_ms": 1000.0 * lag / 16000.0,
            "snr_db": float(snr), "lsd_db": float(np.mean(d)),
            "len_ref": int(len(ref)), "len_other": int(len(other))}


def main() -> int:
    items = sorted(d.name for d in STIM.iterdir() if d.is_dir())
    print(f"items: {len(items)}\n")
    rows, bad = [], []

    for k, item in enumerate(items, 1):
        d = STIM / item
        ref_path = d / "ref.wav"
        ref = decoded_pcm(ref_path)

        for br in BITRATES:
            mp3 = d / f"mp3_{br}.mp3"
            rt = d / f"rt_mp3_{br}.wav"

            if not mp3.exists():
                run(["ffmpeg", "-v", "error", "-y", "-i", str(ref_path),
                     "-c:a", "libmp3lame", "-b:a", f"{br}k", str(mp3)])
            if not rt.exists():
                run(["ffmpeg", "-v", "error", "-y", "-i", str(mp3),
                     "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(rt)])

            pcm_mp3, pcm_rt = decoded_pcm(mp3), decoded_pcm(rt)
            h_mp3, h_rt = pcm_sha(pcm_mp3), pcm_sha(pcm_rt)
            identical = h_mp3 == h_rt
            if not identical:
                bad.append((item, br, len(pcm_mp3), len(pcm_rt)))

            f = fidelity(ref, pcm_rt)
            rows.append({"item_id": item, "bitrate": br,
                         "container_pcm_identical": identical,
                         "pcm_sha_mp3": h_mp3, "pcm_sha_rt": h_rt,
                         "mp3_bytes": mp3.stat().st_size,
                         "rt_bytes": rt.stat().st_size,
                         "ref_bytes": ref_path.stat().st_size, **f})
        if k % 10 == 0 or k == len(items):
            print(f"  {k}/{len(items)} items")

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "fidelity.parquet", index=False)

    print("\n" + "=" * 68)
    print("CONTAINER-CONTROL VALIDITY  (decode(mp3_X) == rt_mp3_X, sample-exact)")
    print("=" * 68)
    ok = int(df.container_pcm_identical.sum())
    print(f"  {ok}/{len(df)} stimulus pairs bit-identical")
    if bad:
        print(f"  FAILURES: {bad[:5]}")
        print("  -> container contrast INVALID. Stopping.")
        return 1
    print("  -> container contrast is valid: identical signal, different container.")

    print("\n" + "=" * 68)
    print("FIDELITY vs ref  (after cross-correlation alignment)")
    print("=" * 68)
    agg = df.groupby("bitrate").agg(
        snr_db=("snr_db", "mean"), lsd_db=("lsd_db", "mean"),
        lag_ms=("lag_ms", "mean"),
        mp3_kb=("mp3_bytes", lambda s: s.mean() / 1024),
        ref_kb=("ref_bytes", lambda s: s.mean() / 1024)).round(3)
    agg["compression_x"] = (agg.ref_kb / agg.mp3_kb).round(1)
    print(agg.to_string())

    print(f"\n  encoder delay: mean {df.lag_ms.mean():.1f} ms "
          f"(min {df.lag_ms.min():.1f}, max {df.lag_ms.max():.1f})")
    print("  NOTE: delay is identical for mp3_X and rt_mp3_X, so it cannot")
    print("        confound the container contrast; it is a property of the")
    print("        ref-vs-roundtrip (signal) contrast only, and is reported.")

    json.dump({"n_items": len(items), "bitrates": BITRATES,
               "container_pairs_identical": ok, "container_pairs_total": len(df),
               "by_bitrate": json.loads(agg.to_json(orient="index"))},
              open(OUT / "fidelity_summary.json", "w"), indent=1)
    print(f"\n  saved: exp/out/fidelity.parquet, exp/out/fidelity_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
