"""Determinism check on the arousal arm (limitation #10 in the paper).

The six-way arm found 3/50 clips whose output varied across byte-identical
requests. The paper states the arousal arm never repeated that check. This
repeats it, with the cache bypassed, on the DV actually used for the headline
results. If clips are unstable, the headline contrasts are re-run without them.
"""
from __future__ import annotations
import base64, json, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp.client import BASE, KEY, MODEL, SEED, label_mass, pmap
from exp.run_arousal import PROMPT, AROUSAL, LABELS

ROOT = Path(__file__).resolve().parent.parent
STIM, OUT = ROOT/"data"/"stimuli", ROOT/"exp"/"out"
REPS = 3

def raw(payload):
    req = urllib.request.Request(f"{BASE}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())

items = sorted(d.name for d in STIM.iterdir() if d.is_dir())
jobs = []
for it in items:
    data = base64.b64encode((STIM/it/"ref.wav").read_bytes()).decode()
    payload = {"model":MODEL,"modalities":["text"],
        "messages":[{"role":"user","content":[
            {"type":"text","text":PROMPT},
            {"type":"input_audio","input_audio":{"data":data,"format":"wav"}}]}],
        "max_completion_tokens":12,"temperature":0,"seed":SEED,
        "logprobs":True,"top_logprobs":20}
    for r_ in range(REPS):
        jobs.append({"item_id":it,"rep":r_,"payload":payload})
print(f"{len(items)} clips x {REPS} byte-identical repeats = {len(jobs)} uncached calls\n")

def work(j):
    try: resp = raw(j["payload"])
    except Exception as e:
        return {"item_id":j["item_id"],"rep":j["rep"],"p_high":np.nan,"err":str(e)}
    m = label_mass(resp, LABELS)
    return {"item_id":j["item_id"],"rep":j["rep"],"p_high":m.get("high"),
            "said":(resp["choices"][0]["message"]["content"] or "").strip().lower()}

df = pd.DataFrame(pmap(work, jobs, workers=12, desc="det"))
df.to_parquet(OUT/"arousal_determinism.parquet", index=False)
piv = df.pivot_table(index="item_id", columns="rep", values="p_high")
rng = (piv.max(axis=1)-piv.min(axis=1)).dropna()
unstable = sorted(rng[rng>1e-12].index)

print("\n"+"="*72); print("AROUSAL-ARM DETERMINISM"); print("="*72)
print(f"  clips                    : {len(rng)}")
print(f"  clips with ANY variation : {len(unstable)}/{len(rng)}")
print(f"  mean within-clip range   : {rng.mean():.6f}")
print(f"  max  within-clip range   : {rng.max():.6f}")
if unstable: print(f"  unstable clips           : {unstable}")

old = ['1005_MTI_ANG_XX','1008_TAI_DIS_XX','1022_TIE_HAP_XX']
print(f"\n  six-way arm unstable clips: {old}")
print(f"  overlap with those        : {sorted(set(unstable)&set(old))}")

if unstable:
    print("\n"+"="*72); print("HEADLINE CONTRASTS RE-RUN WITHOUT UNSTABLE CLIPS"); print("="*72)
    g = pd.read_parquet(OUT/"arousal_grid.parquet")
    gp = g.pivot_table(index="item_id", columns="condition", values="p_high").drop(index=unstable, errors="ignore")
    x = pd.read_parquet(OUT/"arousal_xai.parquet")
    un = x[x.mask_id=="unmasked"].pivot_table(index="item_id",columns="condition",values="p_high").drop(index=unstable, errors="ignore")
    dith = np.abs(un["ref_dither"]-un["ref"]).dropna()
    print(f"  n = {len(gp)} clips")
    print(f"  dither floor      mean|d| {dith.mean():.4f}")
    for kind,a,c in (("codec","rt_mp3_64","ref"),("container","mp3_64","rt_mp3_64"),("total","mp3_64","ref")):
        d = np.abs(gp[a]-gp[c]).dropna()
        print(f"  {kind:<17} mean|d| {d.mean():.4f}   = {d.mean()/dith.mean():.2f}x floor")
    from scipy import stats
    sm = pd.read_parquet(OUT/"arousal_similarity.parquet")
    sm = sm[~sm.item_id.isin(unstable)]
    nullc = "rho_dither (NULL)"
    print(f"\n  map similarity (n={len(sm)}):  null {sm[nullc].mean():.3f}", end="")
    for nm in ("container","codec","total"):
        col=f"rho_{nm}"
        if col in sm:
            cc=sm[[col,nullc]].dropna(); dd=cc[nullc]-cc[col]
            try: W,p=stats.wilcoxon(dd)
            except ValueError: p=1.0
            print(f" | {nm} {sm[col].mean():.3f} (gap {dd.mean():+.3f}, p={p:.3g})", end="")
    print()
else:
    print("\n  -> fully deterministic; limitation #10 can be closed as measured.")
json.dump({"n":int(len(rng)),"reps":REPS,"unstable":unstable,
           "mean_range":float(rng.mean()),"max_range":float(rng.max())},
          open(OUT/"arousal_determinism.json","w"), indent=1)
