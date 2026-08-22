"""Fix the refusal problem before committing the grid to the arousal scheme.

The arousal prompt produced 28% responses of the form "please provide the audio
you'd like me to listen to" -- the model asserting no audio was attached, when
it was. That is a prompt-formatting failure, not a perception failure, and it
would silently delete a quarter of the data.

Tests four variants (including audio-before-text ordering) on all 50 items and
scores them on refusal rate, bias-free discriminability (AUC of P(high)), and
how many items land away from floor/ceiling.
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp.client import BASE, CACHE, KEY, MODEL, SEED, label_mass, pmap

ROOT = Path(__file__).resolve().parent.parent
STIM = ROOT / "data" / "stimuli"
OUT = ROOT / "exp" / "out"

AROUSAL = {"ANG": "high", "FEA": "high", "HAP": "high",
           "SAD": "low", "NEU": "low", "DIS": "low"}

VARIANTS = {
    "v1_original": dict(audio_first=False, text=(
        "Listen to this audio. Is the speaker's vocal energy high or low? "
        "Answer with exactly one word: high or low.")),
    "v2_audio_first": dict(audio_first=True, text=(
        "Is the speaker's vocal energy high or low? "
        "Answer with exactly one word: high or low.")),
    "v3_explicit": dict(audio_first=True, text=(
        "The audio clip above contains a person speaking. Judging only from how "
        "they sound, is their vocal energy high or low? "
        "Reply with exactly one word: high or low.")),
    "v4_energetic": dict(audio_first=True, text=(
        "Does the speaker in the attached clip sound energetic or subdued? "
        "Reply with exactly one word: energetic or subdued.")),
}
LABELS = {"v1_original": ("high", "low"), "v2_audio_first": ("high", "low"),
          "v3_explicit": ("high", "low"), "v4_energetic": ("energetic", "subdued")}
MAPPING = {"v4_energetic": {"high": "energetic", "low": "subdued"}}


def refusal(s: str) -> bool:
    s = (s or "").lower()
    return any(k in s for k in ("provide the audio", "help with that",
                                "no audio", "i don't have", "unable to"))


def ask(path: Path, text: str, audio_first: bool, top: int = 20) -> dict:
    data = base64.b64encode(path.read_bytes()).decode()
    a = {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}
    t = {"type": "text", "text": text}
    content = [a, t] if audio_first else [t, a]
    payload = {"model": MODEL, "modalities": ["text"],
               "messages": [{"role": "user", "content": content}],
               "max_completion_tokens": 12, "temperature": 0, "seed": SEED,
               "logprobs": True, "top_logprobs": top}
    ck = hashlib.sha256(json.dumps(
        [MODEL, hashlib.sha256(path.read_bytes()).hexdigest(), text,
         audio_first, top, SEED], sort_keys=True).encode()).hexdigest()
    cf = CACHE / f"{ck}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"}, method="POST")
    for _ in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.loads(r.read().decode())
            cf.write_text(json.dumps(resp))
            return resp
        except Exception:
            pass
    return {"__error__": "failed"}


def main() -> int:
    items = sorted(d.name for d in STIM.iterdir() if d.is_dir())
    jobs = [{"variant": v, "item_id": it} for v in VARIANTS for it in items]
    print(f"{len(jobs)} calls ({len(VARIANTS)} variants x {len(items)} items)\n")

    def work(j):
        v = VARIANTS[j["variant"]]
        aro = AROUSAL[j["item_id"].split("_")[2]]
        gold = MAPPING.get(j["variant"], {}).get(aro, aro)
        labs = list(LABELS[j["variant"]])
        r = ask(STIM / j["item_id"] / "ref.wav", v["text"], v["audio_first"])
        said = ""
        if "__error__" not in r:
            said = (r["choices"][0]["message"]["content"] or "").strip().lower().strip(".,!'\" ")
        m = label_mass(r, labs)
        pos = labs[0]
        p_pos = m.get(pos)
        return {"variant": j["variant"], "item_id": j["item_id"],
                "arousal": aro, "gold": gold, "said": said,
                "refusal": refusal(said), "correct": said == gold,
                "p_pos": p_pos}

    df = pd.DataFrame(pmap(work, jobs, workers=12, desc="prompt"))
    df.to_parquet(OUT / "gate_prompt.parquet", index=False)

    print("\n" + "=" * 80)
    print("PROMPT VARIANT COMPARISON  (arousal scheme, reference condition)")
    print("=" * 80)
    print(f"  {'variant':<16}{'refusal':>9}{'acc':>7}{'base':>7}{'AUC':>7}"
          f"{'p':>10}{'unpinned':>11}")
    rows = []
    for v in VARIANTS:
        s = df[df.variant == v]
        ref_rate = s.refusal.mean()
        u = s[(~s.refusal) & s.p_pos.notna()]
        if len(u) < 8:
            print(f"  {v:<16}{ref_rate:>9.0%}   too few usable")
            continue
        a = u[u.arousal == "high"].p_pos.values
        b = u[u.arousal == "low"].p_pos.values
        if len(a) < 3 or len(b) < 3:
            print(f"  {v:<16}{ref_rate:>9.0%}   class too small")
            continue
        U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        auc = U / (len(a) * len(b))
        base = u.gold.value_counts(normalize=True).max()
        unp = int(((u.p_pos > .01) & (u.p_pos < .99)).sum())
        print(f"  {v:<16}{ref_rate:>9.0%}{u.correct.mean():>7.3f}{base:>7.3f}"
              f"{auc:>7.3f}{p:>10.3g}{unp:>8}/{len(u):<3}")
        rows.append({"variant": v, "refusal": ref_rate, "auc": auc, "p": p,
                     "unpinned": unp / len(u), "n": len(u),
                     "acc": u.correct.mean(), "base": base})

    if rows:
        r = pd.DataFrame(rows)
        r["score"] = (1 - r.refusal) * (r.auc - .5) * 2 + .3 * r.unpinned
        best = r.sort_values("score", ascending=False).iloc[0]
        print(f"\n  -> best: {best.variant}  "
              f"(refusal {best.refusal:.0%}, AUC {best.auc:.3f}, "
              f"unpinned {best.unpinned:.0%}, n={int(best.n)})")
        json.dump(r.to_dict("records"), open(OUT / "gate_prompt.json", "w"),
                  indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
