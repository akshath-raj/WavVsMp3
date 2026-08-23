"""Main grid: WAV vs MP3 on gpt-audio-1.5, four readouts, seven conditions.

Conditions (all descended from one canonical `ref`):
    ref, mp3_32, mp3_64, mp3_128, rt_mp3_32, rt_mp3_64, rt_mp3_128

Readouts:
    six_way      6-way forced choice        -> label accuracy + collapse evidence
    binary_gold  gold vs foil-anchor        -> P(gold), the continuous DV
    binary_foil  wrong-label vs anchor      -> construct-validity control
    transcribe   verbatim transcription     -> WER, intelligibility control

Why `binary_gold` is the primary DV: the 6-way readout collapses to `neutral`
(gate G0 failed at 8.3%), so label accuracy has no variance to explain. The
binary readout puts two labels in direct competition at a single token position,
where renormalised probability mass becomes a continuous, noise-free measure of
how much evidence the model finds for the true emotion.

`binary_foil` exists because a graded measure that cannot tell a true label from
a false one measures nothing. It assigns each item a deterministic wrong emotion
and runs the identical comparison; P(gold) must exceed P(foil) or the DV is void.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp.client import (EMOTIONS, STATS, call, content_of, gold_of, label_mass,
                        parse_emotion, pmap, print_stats)

ROOT = Path(__file__).resolve().parent.parent
STIM = ROOT / "data" / "stimuli"
OUT = ROOT / "exp" / "out"
OUT.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    ("ref",        "ref.wav",         "wav", "lossless",  None),
    ("mp3_32",     "mp3_32.mp3",      "mp3", "mp3",         32),
    ("mp3_64",     "mp3_64.mp3",      "mp3", "mp3",         64),
    ("mp3_128",    "mp3_128.mp3",     "mp3", "mp3",        128),
    ("rt_mp3_32",  "rt_mp3_32.wav",   "wav", "roundtrip",   32),
    ("rt_mp3_64",  "rt_mp3_64.wav",   "wav", "roundtrip",   64),
    ("rt_mp3_128", "rt_mp3_128.wav",  "wav", "roundtrip",  128),
]

# CREMA-D's 12 fixed sentences, keyed by the code embedded in each filename.
SENTENCES = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

P_SIX = ("Listen to this audio. The speaker's emotion is one of: angry, disgusted, "
         "fearful, happy, neutral, sad. Respond with exactly one word from that "
         "list and nothing else.")
P_TRANSCRIBE = "Transcribe this audio verbatim. Output only the transcription."


def binary_prompt(a: str, b: str) -> str:
    return (f"Listen to this audio. Is the speaker {a} or {b}? "
            f"Answer with exactly one word: {a} or {b}.")


def anchor_for(gold: str) -> str:
    """The label the gold competes against.

    `neutral` is the model's default attractor, so gold-vs-neutral is the most
    informative axis. Neutral-gold items would make that degenerate, so they
    compete against `angry` instead.
    """
    return "angry" if gold == "neutral" else "neutral"


def foil_for(item_id: str, gold: str, anchor: str) -> str:
    """A deterministic wrong emotion, stable across runs."""
    pool = [e for e in EMOTIONS if e not in (gold, anchor)]
    h = int(hashlib.sha256(item_id.encode()).hexdigest(), 16)
    return pool[h % len(pool)]


def main() -> int:
    items = sorted(d.name for d in STIM.iterdir() if d.is_dir())
    print(f"items {len(items)} x conditions {len(CONDITIONS)} x 4 readouts "
          f"= {len(items)*len(CONDITIONS)*4} calls\n")

    jobs = []
    for it in items:
        gold = gold_of(it)
        anchor = anchor_for(gold)
        foil = foil_for(it, gold, anchor)
        sent_code = it.split("_")[1]
        for cond, fname, fmt, family, br in CONDITIONS:
            p = STIM / it / fname
            base = dict(item_id=it, condition=cond, container=fmt, family=family,
                        bitrate=br, gold=gold, anchor=anchor, foil=foil,
                        sentence_code=sent_code,
                        reference_text=SENTENCES.get(sent_code))
            jobs += [
                {**base, "readout": "six_way", "prompt": P_SIX, "max_tok": 12},
                {**base, "readout": "binary_gold",
                 "prompt": binary_prompt(gold, anchor), "max_tok": 12},
                {**base, "readout": "binary_foil",
                 "prompt": binary_prompt(foil, anchor), "max_tok": 12},
                {**base, "readout": "transcribe",
                 "prompt": P_TRANSCRIBE, "max_tok": 64},
            ]

    def work(j):
        p = STIM / j["item_id"] / dict(
            (c[0], c[1]) for c in CONDITIONS)[j["condition"]]
        r = call(p, j["prompt"], max_tok=j["max_tok"])
        out = dict(j)
        out["error"] = r.get("__error__")
        out["raw"] = content_of(r)
        if j["readout"] == "six_way":
            out["pred"] = parse_emotion(r)
            out["correct"] = (out["pred"] == j["gold"])
            m = label_mass(r, EMOTIONS)
            out["p_gold_6way"] = m.get(j["gold"])
            out["n_labels_seen_6way"] = len(m)
        elif j["readout"] == "binary_gold":
            m = label_mass(r, [j["gold"], j["anchor"]])
            out["p_gold"] = m.get(j["gold"])
            out["binary_said"] = (out["raw"] or "").strip().lower().strip(".,!'\" ")
            out["binary_said_gold"] = out["binary_said"] == j["gold"]
        elif j["readout"] == "binary_foil":
            m = label_mass(r, [j["foil"], j["anchor"]])
            out["p_foil"] = m.get(j["foil"])
        return out

    rows = pmap(work, jobs, workers=12, desc="grid")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "grid.parquet", index=False)
    print_stats()

    # ---- immediate sanity readout -----------------------------------------
    print("\n" + "=" * 70)
    print("SIX-WAY ACCURACY BY CONDITION (documents the collapse)")
    print("=" * 70)
    s = df[df.readout == "six_way"]
    print(s.groupby("condition").agg(
        acc=("correct", "mean"), n=("correct", "size"),
        distinct_preds=("pred", "nunique")).round(3).to_string())
    print(f"\n  overall prediction distribution: "
          f"{s.pred.value_counts().to_dict()}")

    print("\n" + "=" * 70)
    print("PRIMARY DV — mean P(gold) BY CONDITION")
    print("=" * 70)
    b = df[df.readout == "binary_gold"]
    print(b.groupby("condition").agg(
        p_gold=("p_gold", "mean"), sd=("p_gold", "std"),
        n=("p_gold", "count")).round(4).to_string())

    print("\n" + "=" * 70)
    print("CONSTRUCT VALIDITY — P(gold) vs P(foil)")
    print("=" * 70)
    f = df[df.readout == "binary_foil"]
    print(f"  mean P(gold) = {b.p_gold.mean():.4f}   "
          f"mean P(foil) = {f.p_foil.mean():.4f}")
    print(f"  -> {'DISCRIMINATES' if b.p_gold.mean() > f.p_foil.mean() else 'DOES NOT DISCRIMINATE'}")

    print(f"\n  saved: exp/out/grid.parquet  ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
