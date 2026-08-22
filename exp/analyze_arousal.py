"""Analysis of the arousal arm, with the 1-LSB dither condition as the null baseline.

The dither condition is the control the first study lacked. It changes the
waveform by the smallest amount representable at 16 bits (~-96 dBFS), which is
inaudible and roughly four orders of magnitude below the codec's own noise.
It therefore answers the question that decides both headline claims:

  * Performance claim. If a 1-LSB change moves P(high) as much as the container
    does, then "the container perturbs the evidence" reduces to "any waveform
    change perturbs the evidence", which is a much weaker statement.

  * Explanation claim. If attribution maps diverge as much under 1-LSB dither as
    under MP3, then occlusion maps on this model are simply unstable and the
    format finding is an artefact of a noisy measurement.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent.parent / "exp" / "out"
R = {}


def hdr(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) < 2:
        return (np.nan, np.nan)
    s = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(s, [2.5, 97.5]))


def paired(a, b, label):
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    if len(d) < 3 or np.allclose(d, 0):
        return {"label": label, "n": int(len(d)),
                "mean_diff": float(np.mean(d)) if len(d) else np.nan,
                "p": 1.0, "ci95": (np.nan, np.nan)}
    try:
        W, p = stats.wilcoxon(d)
    except ValueError:
        p = 1.0
    lo, hi = boot_ci(d)
    return {"label": label, "n": int(len(d)), "mean_diff": float(np.mean(d)),
            "median_diff": float(np.median(d)), "p": float(p),
            "ci95": (float(lo), float(hi))}


grid = pd.read_parquet(OUT / "arousal_grid.parquet")
xai = pd.read_parquet(OUT / "arousal_xai.parquet")

# ---------------- 1. does the task work at all? --------------------------
hdr("1  DISCRIMINABILITY BY CONDITION  (AUC of P(high); 0.5 = no signal)")
print("  P(high) is a FIXED label, so a model that always answers one way")
print("  scores 0.5 here however confident it is. Response bias cannot inflate it.\n")
order = ["ref", "rt_mp3_32", "rt_mp3_64", "rt_mp3_128", "mp3_32", "mp3_64", "mp3_128"]
rows = []
for c in order:
    s = grid[(grid.condition == c) & grid.p_high.notna()]
    a = s[s.arousal == "high"].p_high.values
    b = s[s.arousal == "low"].p_high.values
    if len(a) < 3 or len(b) < 3:
        continue
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rows.append({"condition": c, "auc": U / (len(a) * len(b)), "p": p,
                 "acc": s.correct.mean(), "mean_p_high": s.p_high.mean(),
                 "n": len(s)})
d = pd.DataFrame(rows).set_index("condition")
print(d.round(4).to_string())
R["discriminability"] = json.loads(d.to_json(orient="index"))
print(f"\n  six-way arm for comparison: accuracy 0.20 vs 0.167 chance, 87.7% one label")

# ---------------- 2. format contrasts on P(high) -------------------------
hdr("2  FORMAT CONTRASTS ON P(high) — signed, then magnitude")
piv = grid.pivot_table(index="item_id", columns="condition", values="p_high")
pairs = []
for br in (32, 64, 128):
    pairs += [("codec", f"rt_mp3_{br}", "ref", br),
              ("container", f"mp3_{br}", f"rt_mp3_{br}", br),
              ("total", f"mp3_{br}", "ref", br)]
res = {}
print(f"  {'contrast':<24}{'mean Δ':>10}{'mean|Δ|':>10}{'med|Δ|':>9}"
      f"{'max|Δ|':>9}{'Δ≠0':>9}{'p':>9}")
for kind, a, c, br in pairs:
    if a not in piv.columns or c not in piv.columns:
        continue
    sub = piv[[a, c]].dropna()
    dd = sub[a].values - sub[c].values
    r = paired(sub[a].values, sub[c].values, f"{kind}@{br}k")
    ad = np.abs(dd)
    r |= {"mean_abs": float(ad.mean()), "median_abs": float(np.median(ad)),
          "max_abs": float(ad.max()), "n_nonzero": int((ad > 1e-12).sum())}
    res[f"{kind}_{br}"] = r
    print(f"  {kind+'@'+str(br)+'k':<24}{r['mean_diff']:>+10.4f}{r['mean_abs']:>10.4f}"
          f"{r['median_abs']:>9.4f}{r['max_abs']:>9.4f}"
          f"{r['n_nonzero']:>5}/{len(dd):<3}{r['p']:>9.3g}")
R["contrasts"] = res

# ---------------- 3. THE NULL BASELINE -----------------------------------
hdr("3  NULL BASELINE — what does a 1-LSB (inaudible) change do?")
un = xai[xai.mask_id == "unmasked"].pivot_table(
    index="item_id", columns="condition", values="p_high")
if "ref_dither" in un.columns and "ref" in un.columns:
    sub = un[["ref_dither", "ref"]].dropna()
    dither = np.abs(sub.ref_dither.values - sub.ref.values)
    print(f"  dither (1 LSB, ~-96 dBFS, inaudible):")
    print(f"    mean|Δ| {dither.mean():.4f}   median {np.median(dither):.4f}   "
          f"max {dither.max():.4f}   items Δ≠0 {int((dither>1e-12).sum())}/{len(dither)}")
    for k in ("container_64", "codec_64", "total_64"):
        if k in res:
            m = res[k]
            ratio = m["mean_abs"] / max(dither.mean(), 1e-12)
            print(f"  {k:<14} mean|Δ| {m['mean_abs']:.4f}   "
                  f"= {ratio:.1f}x the dither floor")
    R["dither_floor"] = {"mean_abs": float(dither.mean()),
                         "median_abs": float(np.median(dither)),
                         "max_abs": float(dither.max()),
                         "n_nonzero": int((dither > 1e-12).sum()),
                         "n": int(len(dither))}
    print()
    if "container_64" in res:
        ratio = res["container_64"]["mean_abs"] / max(dither.mean(), 1e-12)
        if ratio > 3:
            print(f"  -> Container effect is {ratio:.1f}x an inaudible change. The")
            print("     container claim SURVIVES: it is not merely 'any change moves it'.")
        else:
            print(f"  -> Container effect is only {ratio:.1f}x an inaudible change.")
            print("     The container claim DOES NOT SURVIVE: a 1-LSB perturbation")
            print("     moves the output comparably, so the effect is generic")
            print("     sensitivity to any waveform change, not the container.")

# ---------------- 4. attribution map similarity --------------------------
hdr("4  ATTRIBUTION MAPS — does the dither disturb them as much as MP3 does?")
real = xai[xai.mask_kind.isin(["temporal", "spectral"])]
maps = real.pivot_table(index=["item_id", "mask_id"], columns="condition",
                        values="attribution")
COMPS = [("ref", "ref_dither", "dither (NULL)"),
         ("mp3_64", "rt_mp3_64", "container"),
         ("ref", "rt_mp3_64", "codec"),
         ("ref", "mp3_64", "total")]
sims = []
for it, g in maps.groupby(level=0):
    row = {"item_id": it}
    for a, c, name in COMPS:
        if a in g.columns and c in g.columns:
            u, v = g[a].values, g[c].values
            ok = np.isfinite(u) & np.isfinite(v)
            if ok.sum() >= 4 and np.std(u[ok]) > 0 and np.std(v[ok]) > 0:
                row[f"rho_{name}"] = stats.spearmanr(u[ok], v[ok]).statistic
                row[f"top3_{name}"] = len(set(np.argsort(-u[ok])[:3]) &
                                          set(np.argsort(-v[ok])[:3])) / 3
    sims.append(row)
sm = pd.DataFrame(sims)
cols = [c for c in sm.columns if c.startswith("rho_")]
print(sm[cols + [c for c in sm.columns if c.startswith("top3_")]]
      .agg(["mean", "std", "count"]).round(4).to_string())
sm.to_parquet(OUT / "arousal_similarity.parquet", index=False)
R["map_similarity"] = json.loads(sm[[c for c in sm.columns
                                     if c.startswith(("rho_", "top3_"))]].mean().to_json())

if "rho_dither (NULL)" in sm.columns:
    nullc = "rho_dither (NULL)"
    print(f"\n  null (inaudible) baseline: rho = {sm[nullc].mean():.3f}")
    for name in ("container", "codec", "total"):
        c = f"rho_{name}"
        if c in sm.columns:
            cc = sm[[c, nullc]].dropna()
            r = paired(cc[nullc].values, cc[c].values, f"null - {name}")
            print(f"    {name:<12} rho = {sm[c].mean():.3f}   "
                  f"null−{name} = {r['mean_diff']:+.3f}   p = {r['p']:.3g}")
            R.setdefault("null_vs", {})[name] = r
    print()
    tot = sm["rho_total"].mean() if "rho_total" in sm else np.nan
    nl = sm[nullc].mean()
    if np.isfinite(tot):
        if nl - tot > 0.05 and R.get("null_vs", {}).get("total", {}).get("p", 1) < .05:
            print("  -> Maps are RELIABLY more stable under an inaudible change than")
            print("     under MP3. The explanation finding SURVIVES the control.")
        else:
            print("  -> Maps are about as unstable under an inaudible change as under")
            print("     MP3. The explanation finding DOES NOT SURVIVE: occlusion maps")
            print("     on this model are simply unstable to any perturbation.")

json.dump(R, open(OUT / "arousal_results.json", "w"), indent=1, default=str)
print(f"\n  saved: exp/out/arousal_results.json")
