"""Round 2: keep text-first (clean single-word answers) and kill the refusals."""
from __future__ import annotations
import base64, hashlib, json, sys, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exp.client import BASE, CACHE, KEY, MODEL, SEED, label_mass, pmap

ROOT=Path(__file__).resolve().parent.parent; STIM=ROOT/"data"/"stimuli"; OUT=ROOT/"exp"/"out"
AROUSAL={"ANG":"high","FEA":"high","HAP":"high","SAD":"low","NEU":"low","DIS":"low"}

VARIANTS={
 "v5_attached": dict(system=None, schema=False, text=(
   "The audio is attached to this message. Listen to it. Is the speaker's vocal "
   "energy high or low? Answer with exactly one word: high or low.")),
 "v6_system": dict(system="You always answer with exactly one word and never ask for clarification.",
   schema=False, text=(
   "Listen to this audio. Is the speaker's vocal energy high or low? "
   "Answer with exactly one word: high or low.")),
 "v7_norefuse": dict(system=None, schema=False, text=(
   "Listen to this audio. Is the speaker's vocal energy high or low? "
   "The audio is present; do not ask for it. Answer with exactly one word: high or low.")),
 "v8_schema": dict(system=None, schema=True, text=(
   "Listen to this audio. Is the speaker's vocal energy high or low?")),
}
def refusal(s):
    s=(s or "").lower()
    return any(k in s for k in ("provide the audio","help with that","no audio","i don't have","unable to"))

def ask(path,cfg):
    data=base64.b64encode(path.read_bytes()).decode()
    msgs=[]
    if cfg["system"]: msgs.append({"role":"system","content":cfg["system"]})
    msgs.append({"role":"user","content":[{"type":"text","text":cfg["text"]},
        {"type":"input_audio","input_audio":{"data":data,"format":"wav"}}]})
    payload={"model":MODEL,"modalities":["text"],"messages":msgs,
      "max_completion_tokens":12,"temperature":0,"seed":SEED,
      "logprobs":True,"top_logprobs":20}
    if cfg["schema"]:
        payload["response_format"]={"type":"json_schema","json_schema":{"name":"arousal",
          "strict":True,"schema":{"type":"object","properties":{
            "energy":{"type":"string","enum":["high","low"]}},
            "required":["energy"],"additionalProperties":False}}}
    ck=hashlib.sha256(json.dumps([MODEL,hashlib.sha256(path.read_bytes()).hexdigest(),
        cfg["text"],cfg["system"],cfg["schema"],SEED],sort_keys=True).encode()).hexdigest()
    cf=CACHE/f"{ck}.json"
    if cf.exists(): return json.loads(cf.read_text())
    req=urllib.request.Request(f"{BASE}/chat/completions",data=json.dumps(payload).encode(),
      headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"},method="POST")
    for _ in range(5):
        try:
            with urllib.request.urlopen(req,timeout=180) as r: resp=json.loads(r.read().decode())
            cf.write_text(json.dumps(resp)); return resp
        except urllib.error.HTTPError as e:
            body=e.read().decode()[:200]
            return {"__error__":f"HTTP {e.code}: {body}"}
        except Exception as e: last=str(e)
    return {"__error__":"failed"}

items=sorted(d.name for d in STIM.iterdir() if d.is_dir())
jobs=[{"variant":v,"item_id":it} for v in VARIANTS for it in items]
print(f"{len(jobs)} calls\n")
def work(j):
    cfg=VARIANTS[j["variant"]]; aro=AROUSAL[j["item_id"].split("_")[2]]
    r=ask(STIM/j["item_id"]/"ref.wav",cfg)
    if "__error__" in r:
        return {"variant":j["variant"],"item_id":j["item_id"],"arousal":aro,
                "said":"","refusal":False,"correct":False,"p_pos":np.nan,"err":r["__error__"]}
    said=(r["choices"][0]["message"]["content"] or "").strip().lower().strip(".,!'\" ")
    m=label_mass(r,["high","low"])
    return {"variant":j["variant"],"item_id":j["item_id"],"arousal":aro,"said":said,
            "refusal":refusal(said),"correct":(aro in said[:12]),"p_pos":m.get("high"),"err":None}
df=pd.DataFrame(pmap(work,jobs,workers=12,desc="prompt2"))
df.to_parquet(OUT/"gate_prompt2.parquet",index=False)
errs=df[df.err.notna()]
if len(errs): print("\nerrors:",errs.err.value_counts().head(3).to_dict())
print("\n"+"="*80)
print(f"  {'variant':<16}{'refusal':>9}{'NaN p':>7}{'acc':>7}{'base':>7}{'AUC':>7}{'p':>10}{'unpinned':>11}")
for v in VARIANTS:
    s=df[df.variant==v]; u=s[(~s.refusal)&s.p_pos.notna()]
    if len(u)<8: print(f"  {v:<16}{s.refusal.mean():>9.0%}{int(s.p_pos.isna().sum()):>7}   too few usable ({len(u)})"); continue
    a=u[u.arousal=="high"].p_pos.values; b=u[u.arousal=="low"].p_pos.values
    if len(a)<3 or len(b)<3: print(f"  {v:<16} class too small"); continue
    U,p=stats.mannwhitneyu(a,b,alternative="two-sided"); auc=U/(len(a)*len(b))
    base=u.arousal.value_counts(normalize=True).max()
    unp=int(((u.p_pos>.01)&(u.p_pos<.99)).sum())
    print(f"  {v:<16}{s.refusal.mean():>9.0%}{int(s.p_pos.isna().sum()):>7}{u.correct.mean():>7.3f}"
          f"{base:>7.3f}{auc:>7.3f}{p:>10.3g}{unp:>8}/{len(u):<3}")
print("\n  v1_original for reference: refusal 28%, AUC 0.737, p=.016, unpinned 24/36")
for v in VARIANTS:
    s=df[df.variant==v]; print(f"\n  {v} said:", s.said.value_counts().head(4).to_dict())
