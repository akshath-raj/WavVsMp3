"""Full determinism check, cache bypassed.

The container claim rests on the floor for |dP(gold)| being exactly zero for
identical bytes. That was verified on 3 items. The byte-rewrap control then
found 3/50 items where identical PCM in a different wrapper moved the output,
which is either genuine byte-sensitivity or residual service non-determinism.
This distinguishes them by sending byte-identical requests repeatedly, with the
cache bypassed, across a much larger item set.
"""
from __future__ import annotations
import json, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp.client import BASE, KEY, MODEL, SEED, gold_of, label_mass, pmap
from exp.run_grid import anchor_for, binary_prompt
import base64

ROOT = Path(__file__).resolve().parent.parent
STIM = ROOT / "data" / "stimuli"; OUT = ROOT / "exp" / "out"
REPS = 3

def raw_call(payload):
    req = urllib.request.Request(f"{BASE}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())

items = sorted(d.name for d in STIM.iterdir() if d.is_dir())
jobs = []
for it in items:
    gold = gold_of(it)
    p = STIM / it / "ref.wav"
    data = base64.b64encode(p.read_bytes()).decode()
    payload = {"model": MODEL, "modalities": ["text"],
        "messages":[{"role":"user","content":[
            {"type":"text","text":binary_prompt(gold, anchor_for(gold))},
            {"type":"input_audio","input_audio":{"data":data,"format":"wav"}}]}],
        "max_completion_tokens":12,"temperature":0,"seed":SEED,
        "logprobs":True,"top_logprobs":20}
    for r in range(REPS):
        jobs.append({"item_id": it, "gold": gold, "rep": r, "payload": payload})

print(f"{len(items)} items x {REPS} byte-identical repeats = {len(jobs)} UNCACHED calls\n")

def work(j):
    try:
        resp = raw_call(j["payload"])
    except Exception as e:
        return {"item_id": j["item_id"], "rep": j["rep"], "p_gold": np.nan, "err": str(e)}
    m = label_mass(resp, [j["gold"], anchor_for(j["gold"])])
    return {"item_id": j["item_id"], "rep": j["rep"],
            "p_gold": m.get(j["gold"]),
            "content": (resp["choices"][0]["message"]["content"] or "").strip()}

rows = pmap(work, jobs, workers=12, desc="determinism")
df = pd.DataFrame(rows); df.to_parquet(OUT / "control_determinism.parquet", index=False)

piv = df.pivot_table(index="item_id", columns="rep", values="p_gold")
rng = (piv.max(axis=1) - piv.min(axis=1)).dropna()
print("\n" + "="*70); print("DETERMINISM RESULT (byte-identical requests, cache bypassed)"); print("="*70)
print(f"  items                       : {len(rng)}")
print(f"  items with ANY variation    : {int((rng > 1e-12).sum())}/{len(rng)}")
print(f"  mean within-item range      : {rng.mean():.6f}")
print(f"  max  within-item range      : {rng.max():.6f}")
print(f"  95th pct within-item range  : {np.percentile(rng, 95):.6f}")
print()
print("  compare against the measured effects:")
print("    container @64k  mean|d| = 0.0202   max|d| = 0.1592")
print("    codec     @64k  mean|d| = 0.0344   max|d| = 0.3071")
print()
noise = rng.mean()
print(f"  -> noise floor = {noise:.6f}; container effect is {0.0202/max(noise,1e-9):.1f}x the floor")
json.dump({"n_items": int(len(rng)), "reps": REPS,
           "items_varying": int((rng > 1e-12).sum()),
           "mean_range": float(rng.mean()), "max_range": float(rng.max()),
           "p95_range": float(np.percentile(rng, 95))},
          open(OUT / "determinism.json", "w"), indent=1)
