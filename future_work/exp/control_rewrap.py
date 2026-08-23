"""Sham-container control: does ANY byte difference move the output?

The container result claims that submitting `mp3_64` vs `rt_mp3_64` -- which
carry bit-identical decoded PCM -- changes P(gold) on every item. Two very
different mechanisms could produce that:

  (a) the CONTAINER selects a different decoder inside the serving stack, and
      that decoder's output differs from our ffmpeg decode; or
  (b) the service is sensitive to raw file bytes in some way that has nothing
      to do with audio, in which case the "container effect" is trivial.

This control distinguishes them. `ref_rewrap.wav` carries **the same PCM samples
as `ref.wav`** but a different byte layout (rewritten header, chunks dropped).
Same container format, same decoder, same signal -- only the bytes differ.

  P(gold) identical   -> (a): byte layout is irrelevant; the mp3-vs-roundtrip
                         difference is a genuine decoder/container effect.
  P(gold) differs     -> (b): the effect is byte-level, and the container
                         interpretation does not hold.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp.client import call, gold_of, label_mass, pmap, print_stats
from exp.run_grid import anchor_for, binary_prompt
from exp.run_xai import read_wav, wav_bytes

ROOT = Path(__file__).resolve().parent.parent
STIM = ROOT / "data" / "stimuli"
OUT = ROOT / "exp" / "out"

items = sorted(d.name for d in STIM.iterdir() if d.is_dir())
print(f"items: {len(items)}\n")

jobs = []
n_same_pcm = n_diff_bytes = 0
for it in items:
    orig_path = STIM / it / "ref.wav"
    orig_bytes = orig_path.read_bytes()
    x = read_wav(orig_path)
    rewrap = wav_bytes(x)                     # same samples, our own header

    pcm_o = hashlib.sha256((np.clip(x, -1, 1 - 1e-6) * 32768)
                           .astype("<i2").tobytes()).hexdigest()
    x2 = np.frombuffer(rewrap[44:], dtype="<i2").astype(np.float32) / 32768.0
    pcm_r = hashlib.sha256((np.clip(x2, -1, 1 - 1e-6) * 32768)
                           .astype("<i2").tobytes()).hexdigest()
    n_same_pcm += int(pcm_o == pcm_r)
    n_diff_bytes += int(orig_bytes != rewrap)

    gold = gold_of(it)
    prompt = binary_prompt(gold, anchor_for(gold))
    jobs.append({"item_id": it, "gold": gold, "variant": "orig",
                 "prompt": prompt, "bytes": orig_bytes})
    jobs.append({"item_id": it, "gold": gold, "variant": "rewrap",
                 "prompt": prompt, "bytes": rewrap})

print(f"  PCM identical after rewrap : {n_same_pcm}/{len(items)}")
print(f"  file bytes differ          : {n_diff_bytes}/{len(items)}")
print("  -> the two variants are the same audio in different byte layouts\n")


def work(j):
    sha = hashlib.sha256(j["bytes"]).hexdigest()
    r = call(Path("/dev/null"), j["prompt"], max_tok=12,
             audio_bytes=j["bytes"], audio_sha=sha, fmt="wav")
    m = label_mass(r, [j["gold"], anchor_for(j["gold"])])
    return {k: v for k, v in j.items() if k != "bytes"} | {
        "p_gold": m.get(j["gold"]), "error": r.get("__error__")}


rows = pmap(work, jobs, workers=12, desc="rewrap")
df = pd.DataFrame(rows)
df.to_parquet(OUT / "control_rewrap.parquet", index=False)
print_stats()

piv = df.pivot_table(index="item_id", columns="variant", values="p_gold").dropna()
d = np.abs(piv["rewrap"] - piv["orig"])

print("\n" + "=" * 70)
print("SHAM-CONTAINER CONTROL RESULT")
print("=" * 70)
print(f"  items compared      : {len(d)}")
print(f"  items with |Δ| > 0  : {int((d > 1e-12).sum())}/{len(d)}")
print(f"  mean |Δ|            : {d.mean():.6f}")
print(f"  max  |Δ|            : {d.max():.6f}")
print()
print("  reference points from the main grid:")
print("    container @64k  mean|Δ| = 0.0202   max|Δ| = 0.1592")
print("    codec     @64k  mean|Δ| = 0.0344   max|Δ| = 0.3071")
print()
if d.max() < 1e-9:
    print("  -> BYTE LAYOUT IS IRRELEVANT. Identical PCM in a different byte layout")
    print("     gives identical output. The mp3-vs-roundtrip difference therefore")
    print("     reflects the CONTAINER/DECODER, not raw bytes. Container claim HOLDS.")
elif d.mean() < 0.002:
    print("  -> Byte layout has a negligible effect, far below the container effect.")
    print("     Container claim holds, with this residual reported.")
else:
    print("  -> BYTE LAYOUT MATTERS. The container interpretation does NOT hold;")
    print("     the effect is byte-level sensitivity. This must be reported as such.")
