"""Gate test for binary label schemes.

The six-way readout collapsed: 87.7% `neutral`, accuracy at chance, and only
15/50 items with P(gold) away from floor/ceiling. Removing `neutral` from the
option set removes the model's escape hatch. This tests candidate binary schemes
on the clean reference condition before committing the full grid to one.

A scheme passes if it (a) beats 50% chance, (b) does not collapse onto one
label, and (c) leaves a decent number of items unpinned, since a probability
already at 0 or 1 cannot move when the format changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp.client import call, label_mass, pmap, print_stats

ROOT = Path(__file__).resolve().parent.parent
STIM = ROOT / "data" / "stimuli"
OUT = ROOT / "exp" / "out"

CODE = lambda i: i.split("_")[2]

SCHEMES = {
    "valence": {
        "labels": ("positive", "negative"),
        "gold": {"HAP": "positive", "ANG": "negative", "DIS": "negative",
                 "FEA": "negative", "SAD": "negative"},          # NEU excluded
        "prompt": ("Listen to this audio. Is the emotion in the speaker's voice "
                   "positive or negative? Answer with exactly one word: "
                   "positive or negative."),
    },
    "arousal": {
        "labels": ("high", "low"),
        "gold": {"ANG": "high", "FEA": "high", "HAP": "high",
                 "SAD": "low", "NEU": "low", "DIS": "low"},
        "prompt": ("Listen to this audio. Is the speaker's vocal energy high or "
                   "low? Answer with exactly one word: high or low."),
    },
    "calm_agitated": {
        "labels": ("agitated", "calm"),
        "gold": {"ANG": "agitated", "FEA": "agitated", "HAP": "agitated",
                 "SAD": "calm", "NEU": "calm", "DIS": "calm"},
        "prompt": ("Listen to this audio. Does the speaker sound agitated or "
                   "calm? Answer with exactly one word: agitated or calm."),
    },
}


def main() -> int:
    items = sorted(d.name for d in STIM.iterdir() if d.is_dir())
    jobs = []
    for name, sc in SCHEMES.items():
        for it in items:
            g = sc["gold"].get(CODE(it))
            if g is None:
                continue                       # scheme has no gold for this item
            jobs.append({"scheme": name, "item_id": it, "gold": g,
                         "labels": sc["labels"], "prompt": sc["prompt"]})
    print(f"{len(jobs)} calls across {len(SCHEMES)} schemes\n")

    def work(j):
        r = call(STIM / j["item_id"] / "ref.wav", j["prompt"], max_tok=12)
        m = label_mass(r, list(j["labels"]))
        said = ((r.get("choices", [{}])[0].get("message", {}) or {}).get("content")
                or "").strip().lower().strip(".,!'\" ")
        return {"scheme": j["scheme"], "item_id": j["item_id"], "gold": j["gold"],
                "p_gold": m.get(j["gold"]), "said": said,
                "correct": said == j["gold"], "n_labels": len(m)}

    df = pd.DataFrame(pmap(work, jobs, workers=12, desc="gate"))
    df.to_parquet(OUT / "gate_binary.parquet", index=False)
    print_stats()

    print("\n" + "=" * 78)
    print("BINARY SCHEME GATE  (reference condition only)")
    print("=" * 78)
    print(f"  {'scheme':<15}{'n':>4}{'acc':>7}{'chance':>8}{'said-1 share':>14}"
          f"{'meanP':>8}{'medP':>8}{'unpinned':>10}")
    best = None
    for name, sc in SCHEMES.items():
        s = df[df.scheme == name]
        if not len(s):
            continue
        acc = s.correct.mean()
        share = s.said.value_counts(normalize=True).max()
        pg = s.p_gold.dropna()
        unp = int(((pg > .01) & (pg < .99)).sum())
        print(f"  {name:<15}{len(s):>4}{acc:>7.3f}{0.5:>8.2f}{share:>14.2f}"
              f"{pg.mean():>8.3f}{pg.median():>8.3f}{unp:>7}/{len(pg):<3}")
        score = (acc - .5) + 0.5 * (unp / max(len(pg), 1)) - max(0, share - .8)
        if best is None or score > best[1]:
            best = (name, score)

    print("\n  per-scheme response distribution:")
    for name in SCHEMES:
        s = df[df.scheme == name]
        if len(s):
            print(f"    {name:<15} {s.said.value_counts().to_dict()}")

    print("\n  six-way baseline for comparison: acc 0.20, 87.7% one label, "
          "15/50 unpinned")
    if best:
        print(f"\n  -> best candidate: {best[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
