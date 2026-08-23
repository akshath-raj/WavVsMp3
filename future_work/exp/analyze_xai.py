"""Refined XAI analysis: restrict to items where the measurement is trustworthy."""
import json
import numpy as np, pandas as pd
from scipy import stats
from pathlib import Path
OUT = Path('exp/out')
x = pd.read_parquet(OUT/'xai.parquet')
grid = pd.read_parquet(OUT/'grid.parquet')
stab = json.load(open(OUT/'stable_analysis.json'))
UNSTABLE = set(stab['unstable_items'])

b = grid[grid.readout=='binary_gold']
piv = b.pivot_table(index='item_id', columns='condition', values='p_gold')
unpinned = set(piv.index[(piv['ref']>0.01)&(piv['ref']<0.99)])

real = x[x.mask_kind.isin(['temporal','spectral'])]
null = x[x.mask_kind=='null']

print("="*74); print("A  NULL-MASK FLOOR — how much of the attribution is the mask itself?"); print("="*74)
for cond in ['ref','rt_mp3_64','mp3_64']:
    n = null[null.condition==cond].attribution.abs().mean()
    r = real[real.condition==cond].attribution.abs().mean()
    print(f"  {cond:<12} null {n:.4f}   real {r:.4f}   ratio {r/max(n,1e-9):.2f}x   net {r-n:+.4f}")
print("\n  The null mask inserts loudness-matched noise into the QUIETEST window,")
print("  so it is a real perturbation, not a no-op. It bounds how much of the")
print("  attribution signal is attributable to the masking operation itself.")

def sims(items, tag):
    sub = real[real.item_id.isin(items)]
    maps = sub.pivot_table(index=['item_id','mask_id'], columns='condition', values='attribution')
    rows=[]
    for it,g in maps.groupby(level=0):
        row={'item_id':it}
        for a,c,name in (('ref','rt_mp3_64','codec'),('mp3_64','rt_mp3_64','container'),('ref','mp3_64','total')):
            if a in g.columns and c in g.columns:
                u,v=g[a].values,g[c].values
                ok=np.isfinite(u)&np.isfinite(v)
                if ok.sum()>=4 and np.std(u[ok])>0 and np.std(v[ok])>0:
                    row['rho_'+name]=stats.spearmanr(u[ok],v[ok]).statistic
                    row['top3_'+name]=len(set(np.argsort(-u[ok])[:3])&set(np.argsort(-v[ok])[:3]))/3
        rows.append(row)
    sm=pd.DataFrame(rows)
    cols=[c for c in sm.columns if c.startswith(('rho_','top3_'))]
    print(f"\n  {tag}  (n={len(sm)})")
    print(sm[cols].agg(['mean','std','count']).round(4).to_string())
    return sm

print("\n"+"="*74); print("B  ATTRIBUTION-MAP SIMILARITY ACROSS SUBSETS"); print("="*74)
allitems=set(real.item_id.unique())
s_all=sims(allitems,"all items")
s_stab=sims(allitems-UNSTABLE,"deterministic items only (noise floor = exactly 0)")
s_key=sims((allitems-UNSTABLE)&unpinned,"deterministic AND unpinned (P(gold) not at 0 or 1)")

for nm,sm in (("deterministic",s_stab),("deterministic+unpinned",s_key)):
    if 'rho_codec' in sm and 'rho_container' in sm:
        cc=sm[['rho_codec','rho_container']].dropna()
        if len(cc)>3:
            d=cc.rho_container-cc.rho_codec
            try: W,p=stats.wilcoxon(d)
            except ValueError: W,p=np.nan,1.0
            print(f"\n  [{nm}] container - codec similarity: {d.mean():+.4f}  p={p:.3g}  n={len(cc)}")

print("\n"+"="*74); print("C  IS ANY MAP IDENTICAL? (rho = 1.0 would mean the evidence base survived)"); print("="*74)
for col in ['rho_codec','rho_container','rho_total']:
    if col in s_stab:
        v=s_stab[col].dropna()
        print(f"  {col:<16} mean {v.mean():.3f}   items with rho>0.95: {int((v>0.95).sum())}/{len(v)}"
              f"   items with rho<0.5: {int((v<0.5).sum())}/{len(v)}")

print("\n"+"="*74); print("D  WHICH REGIONS CARRY THE EVIDENCE (mean attribution, deterministic items)"); print("="*74)
sub=real[real.item_id.isin(allitems-UNSTABLE)]
prof=sub.pivot_table(index='mask_id',columns='condition',values='attribution',aggfunc='mean')
t=[f"t{i}" for i in range(10)]
f=sorted([m for m in prof.index if m.startswith('f')],key=lambda m:int(m.split('_')[0][1:]))
prof=prof.reindex([m for m in t+f if m in prof.index])[['ref','rt_mp3_64','mp3_64']]
print(prof.round(4).to_string())
print("\n  rank correlation of the region PROFILE between formats:")
for a,c,n in (('ref','rt_mp3_64','codec'),('mp3_64','rt_mp3_64','container'),('ref','mp3_64','total')):
    r=stats.spearmanr(prof[a],prof[c]).statistic
    print(f"    {n:<11} rho = {r:.3f}")

json.dump({"null_floor":{c: float(null[null.condition==c].attribution.abs().mean()) for c in ['ref','rt_mp3_64','mp3_64']},
 "real_mean":{c: float(real[real.condition==c].attribution.abs().mean()) for c in ['ref','rt_mp3_64','mp3_64']},
 "similarity_all":json.loads(s_all[[c for c in s_all.columns if c.startswith(('rho_','top3_'))]].mean().to_json()),
 "similarity_deterministic":json.loads(s_stab[[c for c in s_stab.columns if c.startswith(('rho_','top3_'))]].mean().to_json()),
 "similarity_det_unpinned":json.loads(s_key[[c for c in s_key.columns if c.startswith(('rho_','top3_'))]].mean().to_json()),
 "n_deterministic":len(s_stab),"n_det_unpinned":len(s_key),
 "region_profile":json.loads(prof.to_json(orient='index'))},
 open(OUT/'xai_refined.json','w'),indent=1)
s_stab.to_parquet(OUT/'xai_similarity.parquet',index=False)
print("\n  saved: exp/out/xai_refined.json")
