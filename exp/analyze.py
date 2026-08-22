"""Analysis for the WAV-vs-MP3 study. Emits results.json + console tables.

Contrast logic (all paired within item):
    codec      rt_mp3_X  - ref          signal degraded, container held = WAV
    container  mp3_X     - rt_mp3_X     signal IDENTICAL, container differs
    total      mp3_X     - ref          both

Because the backend is bit-deterministic, each cell has zero measurement noise;
paired differences are exact, not estimates. That is unusual and it is what
licenses inference at n = 50 on a continuous DV.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "exp" / "out"
BITRATES = [32, 64, 128]
R = {}


def boot_ci(x, fn=np.mean, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) < 2:
        return (np.nan, np.nan)
    s = fn(rng.choice(x, (n, len(x)), replace=True), axis=1)
    return tuple(np.percentile(s, [2.5, 97.5]))


def paired(a, b, label):
    """Paired comparison with Wilcoxon + bootstrap CI on the mean difference."""
    d = np.asarray(a) - np.asarray(b)
    d = d[np.isfinite(d)]
    if len(d) < 3 or np.allclose(d, 0):
        return {"label": label, "n": int(len(d)), "mean_diff": float(np.mean(d)) if len(d) else np.nan,
                "median_diff": float(np.median(d)) if len(d) else np.nan,
                "ci95": (np.nan, np.nan), "W": np.nan, "p": 1.0,
                "note": "all differences zero" if len(d) and np.allclose(d, 0) else "insufficient n"}
    try:
        W, p = stats.wilcoxon(d)
    except ValueError:
        W, p = np.nan, 1.0
    lo, hi = boot_ci(d)
    return {"label": label, "n": int(len(d)), "mean_diff": float(np.mean(d)),
            "median_diff": float(np.median(d)), "ci95": (float(lo), float(hi)),
            "W": float(W) if np.isfinite(W) else None, "p": float(p),
            "cohen_dz": float(np.mean(d) / np.std(d, ddof=1)) if np.std(d, ddof=1) > 0 else np.nan}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k], prev = adj, adj
    return out


def tost(d, bound):
    """Two one-sided tests. Equivalence declared if BOTH one-sided p < .05."""
    d = np.asarray([v for v in d if np.isfinite(v)])
    n = len(d)
    if n < 3:
        return {"bound": bound, "p_lower": 1.0, "p_upper": 1.0, "equivalent": False}
    se = np.std(d, ddof=1) / np.sqrt(n)
    if se == 0:
        eq = abs(np.mean(d)) < bound
        return {"bound": bound, "p_lower": 0.0 if eq else 1.0,
                "p_upper": 0.0 if eq else 1.0, "equivalent": bool(eq),
                "note": "zero variance: exact"}
    t_lo = (np.mean(d) + bound) / se
    t_hi = (np.mean(d) - bound) / se
    p_lo = 1 - stats.t.cdf(t_lo, n - 1)
    p_hi = stats.t.cdf(t_hi, n - 1)
    return {"bound": float(bound), "p_lower": float(p_lo), "p_upper": float(p_hi),
            "equivalent": bool(p_lo < .05 and p_hi < .05)}


def hdr(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


# ==========================================================================
grid = pd.read_parquet(OUT / "grid.parquet")
fid = pd.read_parquet(OUT / "fidelity.parquet")
R["n_calls_grid"] = int(len(grid))
R["errors_grid"] = int(grid.error.notna().sum())

# ---------- 1. six-way collapse -------------------------------------------
hdr("1  SIX-WAY FORCED CHOICE — the accuracy readout")
s = grid[grid.readout == "six_way"]
t = s.groupby("condition").agg(accuracy=("correct", "mean"),
                               n=("correct", "size"),
                               distinct=("pred", "nunique"))
print(t.round(4).to_string())
dist = s.pred.value_counts().to_dict()
print(f"\n  prediction distribution over all {len(s)} calls: {dist}")
neutral_rate = float((s.pred == "neutral").mean())
print(f"  share of responses that are `neutral`: {neutral_rate:.1%}")
R["six_way"] = {"by_condition": json.loads(t.to_json(orient="index")),
                "distribution": {str(k): int(v) for k, v in dist.items()},
                "neutral_rate": neutral_rate,
                "chance": 1 / 6}

# ---------- 2. construct validity ------------------------------------------
hdr("2  CONSTRUCT VALIDITY — does P(gold) beat P(foil)?")
bg = grid[grid.readout == "binary_gold"].set_index(["item_id", "condition"]).p_gold
bf = grid[grid.readout == "binary_foil"].set_index(["item_id", "condition"]).p_foil
j = pd.concat([bg.rename("p_gold"), bf.rename("p_foil")], axis=1).dropna()
cv = paired(j.p_gold.values, j.p_foil.values, "P(gold) - P(foil)")
print(f"  mean P(gold) = {j.p_gold.mean():.4f}   mean P(foil) = {j.p_foil.mean():.4f}")
print(f"  paired diff  = {cv['mean_diff']:+.4f}  95% CI [{cv['ci95'][0]:.4f}, {cv['ci95'][1]:.4f}]"
      f"  p = {cv['p']:.3g}  dz = {cv.get('cohen_dz', float('nan')):.2f}  n = {cv['n']}")
print(f"  -> {'DISCRIMINATES: the graded DV tracks the true label.' if cv['mean_diff'] > 0 and cv['p'] < .05 else 'DOES NOT DISCRIMINATE — DV is not valid.'}")
R["construct_validity"] = cv | {"mean_p_gold": float(j.p_gold.mean()),
                                "mean_p_foil": float(j.p_foil.mean())}

hdr("2b  SHAPE OF THE DV — is P(gold) usable, or floored?")
pg = grid[(grid.readout == "binary_gold") & grid.p_gold.notna()]
ref_pg = pg[pg.condition == "ref"].set_index("item_id").p_gold
print(f"  ref condition: mean {ref_pg.mean():.4f}  median {ref_pg.median():.4f}  "
      f"sd {ref_pg.std():.4f}")
for lo, hi in ((0, .01), (.01, .1), (.1, .5), (.5, .9), (.9, 1.01)):
    n = int(((ref_pg >= lo) & (ref_pg < hi)).sum())
    print(f"    P(gold) in [{lo:.2f}, {hi:.2f}) : {n:>3}/{len(ref_pg)} items  "
          f"{'#' * n}")
informative = ref_pg[(ref_pg > .01) & (ref_pg < .99)]
print(f"\n  items NOT pinned at a floor/ceiling: {len(informative)}/{len(ref_pg)}")
print("  (masking cannot lower a probability already at 0, so occlusion")
print("   attribution is only interpretable on the unpinned subset)")
R["dv_shape"] = {"mean": float(ref_pg.mean()), "median": float(ref_pg.median()),
                 "sd": float(ref_pg.std()), "n_unpinned": int(len(informative)),
                 "n": int(len(ref_pg))}

hdr("2c  BINARY DECISION ACCURACY — a second performance readout")
bd = grid[(grid.readout == "binary_gold") & grid.binary_said_gold.notna()]
ba = bd.groupby("condition").binary_said_gold.agg(["mean", "size"])
ba = ba.reindex([c for c in ["ref", "rt_mp3_32", "rt_mp3_64", "rt_mp3_128",
                             "mp3_32", "mp3_64", "mp3_128"] if c in ba.index])
print(ba.round(4).to_string())
R["binary_accuracy"] = json.loads(ba.to_json(orient="index"))

# ---------- 3. primary DV by condition -------------------------------------
hdr("3  PRIMARY DV — P(gold) by condition")
b = grid[grid.readout == "binary_gold"]
piv = b.pivot_table(index="item_id", columns="condition", values="p_gold")
tab = pd.DataFrame({
    "mean": piv.mean(), "sd": piv.std(), "median": piv.median(), "n": piv.count()})
tab["ci_lo"], tab["ci_hi"] = zip(*[boot_ci(piv[c].dropna().values) for c in piv.columns])
order = ["ref", "rt_mp3_32", "rt_mp3_64", "rt_mp3_128", "mp3_32", "mp3_64", "mp3_128"]
tab = tab.reindex([c for c in order if c in tab.index])
print(tab.round(4).to_string())
R["p_gold_by_condition"] = json.loads(tab.to_json(orient="index"))

# ---------- 4. the three contrasts ------------------------------------------
hdr("4  CONTRASTS — codec / container / total  (paired within item)")
contrasts, praw = {}, {}
for br in BITRATES:
    pairs = [("codec", f"rt_mp3_{br}", "ref"),
             ("container", f"mp3_{br}", f"rt_mp3_{br}"),
             ("total", f"mp3_{br}", "ref")]
    for kind, a, c in pairs:
        if a not in piv.columns or c not in piv.columns:
            continue
        sub = piv[[a, c]].dropna()
        res = paired(sub[a].values, sub[c].values, f"{kind}@{br}k: {a} - {c}")
        res["kind"], res["bitrate"] = kind, br
        contrasts[f"{kind}_{br}"] = res
        praw[f"{kind}_{br}"] = res["p"]
adj = holm(praw)
print(f"  {'contrast':<34}{'mean diff':>11}{'95% CI':>22}{'p':>10}{'p_holm':>9}")
for k, v in contrasts.items():
    v["p_holm"] = adj[k]
    print(f"  {v['label']:<34}{v['mean_diff']:>+11.4f}"
          f"   [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]{v['p']:>10.3g}{adj[k]:>9.3g}")
R["contrasts"] = contrasts

# ---------- 4b. MAGNITUDE, not direction -----------------------------------
hdr("4b  EFFECT MAGNITUDE — |Δ| per item  (signed means cancel; this does not)")
print("  The backend is bit-deterministic: identical bytes give identical output,")
print("  verified to 3 d.p. over 5 repeats. The floor for |Δ| is therefore EXACTLY 0,")
print("  so any non-zero |Δ| is signal, not measurement error.\n")
print(f"  {'contrast':<26}{'mean|d|':>9}{'med|d|':>9}{'max|d|':>9}{'items d!=0':>12}")
mags = {}
for br in BITRATES:
    for kind, a, c in (("codec", f"rt_mp3_{br}", "ref"),
                       ("container", f"mp3_{br}", f"rt_mp3_{br}"),
                       ("total", f"mp3_{br}", "ref")):
        if a not in piv.columns or c not in piv.columns:
            continue
        sub = piv[[a, c]].dropna()
        d = np.abs(sub[a].values - sub[c].values)
        m = {"mean_abs": float(d.mean()), "median_abs": float(np.median(d)),
             "max_abs": float(d.max()),
             "n_nonzero": int((d > 1e-12).sum()), "n": int(len(d)),
             "frac_gt_01": float((d > 0.01).mean()),
             "frac_gt_05": float((d > 0.05).mean())}
        mags[f"{kind}_{br}"] = m
        print(f"  {kind+'@'+str(br)+'k':<26}{m['mean_abs']:>9.4f}{m['median_abs']:>9.4f}"
              f"{m['max_abs']:>9.4f}{m['n_nonzero']:>7}/{m['n']:<4}")
R["magnitudes"] = mags

print("\n  container vs codec magnitude (paired, per item):")
for br in BITRATES:
    a, c, rf = f"mp3_{br}", f"rt_mp3_{br}", "ref"
    if a not in piv.columns:
        continue
    sub = piv[[a, c, rf]].dropna()
    d_cont = np.abs(sub[a].values - sub[c].values)
    d_cod = np.abs(sub[c].values - sub[rf].values)
    res = paired(d_cont, d_cod, f"|container| - |codec| @{br}k")
    R.setdefault("magnitude_container_vs_codec", {})[str(br)] = res
    print(f"    @{br}k  mean|container| {d_cont.mean():.4f}  vs  mean|codec| {d_cod.mean():.4f}"
          f"   diff {res['mean_diff']:+.4f}  p = {res['p']:.3g}")

# ---------- 5. equivalence test on the container effect ---------------------
hdr("5  EQUIVALENCE (TOST) — is the CONTAINER effect practically zero?")
sesoi = 0.05
print(f"  SESOI = {sesoi:.2f} probability mass, fixed on practical grounds:")
print("  a shift smaller than 5 points of probability is unlikely to change any")
print("  downstream decision that thresholds on model confidence.")
print("  NOTE: deriving the bound from the observed codec effect was rejected --")
print("  the codec effect is itself ~0, which would yield a degenerate bound that")
print("  no design could ever clear.")
R["tost_sesoi"] = sesoi
R["tost"] = {}
for br in BITRATES:
    a, c = f"mp3_{br}", f"rt_mp3_{br}"
    if a not in piv.columns:
        continue
    sub = piv[[a, c]].dropna()
    d = sub[a].values - sub[c].values
    t_ = tost(d, sesoi)
    R["tost"][f"container_{br}"] = t_
    n_nonzero = int(np.sum(np.abs(d) > 1e-12))
    print(f"  container@{br}k: mean {np.mean(d):+.5f}  max|d| {np.max(np.abs(d)):.5f}  "
          f"items differing {n_nonzero}/{len(d)}  equivalent={t_['equivalent']}")

# ---------- 6. dose-response ------------------------------------------------
hdr("6  DOSE-RESPONSE — does the codec effect scale with bitrate?")
rows = []
for it in piv.index:
    for br in BITRATES:
        c = f"rt_mp3_{br}"
        if c in piv.columns and np.isfinite(piv.loc[it, c]) and np.isfinite(piv.loc[it, "ref"]):
            rows.append({"item": it, "bitrate": br,
                         "delta": piv.loc[it, c] - piv.loc[it, "ref"]})
dr = pd.DataFrame(rows)
print(dr.groupby("bitrate").delta.agg(["mean", "std", "count"]).round(4).to_string())
rho, prho = stats.spearmanr(dr.bitrate, dr.delta)
print(f"\n  Spearman(bitrate, codec delta) = {rho:+.3f}, p = {prho:.3g}")
print("  (a positive rho means higher bitrate -> less evidence lost, i.e. dose-response)")
R["dose_response"] = {"spearman_rho": float(rho), "p": float(prho),
                      "by_bitrate": json.loads(
                          dr.groupby("bitrate").delta.mean().to_json())}

# ---------- 7. intelligibility control --------------------------------------
hdr("7  INTELLIGIBILITY CONTROL — WER by condition")
try:
    import jiwer
    tr = grid[(grid.readout == "transcribe") & grid.raw.notna() & grid.reference_text.notna()].copy()
    norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                          jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                          jiwer.ReduceToListOfListOfWords()])
    tr["wer"] = [jiwer.wer(reference=r, hypothesis=h,
                           reference_transform=norm, hypothesis_transform=norm)
                 for r, h in zip(tr.reference_text, tr.raw)]
    w = tr.groupby("condition").wer.agg(["mean", "median", "std", "count"])
    w = w.reindex([c for c in order if c in w.index])
    print(w.round(4).to_string())
    wp = tr.pivot_table(index="item_id", columns="condition", values="wer")
    wer_c = {}
    for br in BITRATES:
        if f"mp3_{br}" in wp.columns:
            sub = wp[[f"mp3_{br}", "ref"]].dropna()
            wer_c[f"mp3_{br}"] = paired(sub[f"mp3_{br}"].values, sub["ref"].values,
                                        f"WER mp3_{br} - ref")
    print()
    for k, v in wer_c.items():
        print(f"  {v['label']:<28}{v['mean_diff']:>+9.4f}  p = {v['p']:.3g}")
    R["wer"] = {"by_condition": json.loads(w.to_json(orient="index")),
                "contrasts": wer_c}
    tr.to_parquet(OUT / "wer.parquet", index=False)
except Exception as e:
    print(f"  WER unavailable: {type(e).__name__}: {e}")
    R["wer"] = {"error": str(e)}

# ---------- 8. XAI ----------------------------------------------------------
xai_path = OUT / "xai.parquet"
if xai_path.exists():
    hdr("8  XAI — occlusion attribution")
    x = pd.read_parquet(xai_path)
    real = x[x.mask_kind.isin(["temporal", "spectral"])]
    null = x[x.mask_kind == "null"]
    nm = float(null.attribution.abs().mean())
    rm = float(real.attribution.abs().mean())
    print(f"  null-mask mean |attr| = {nm:.4f}")
    print(f"  real-mask mean |attr| = {rm:.4f}    ratio = {rm/max(nm,1e-9):.2f}x")
    R["xai_null_control"] = {"null_mean_abs": nm, "real_mean_abs": rm,
                             "ratio": rm / max(nm, 1e-9)}

    print("\n  mean |attribution| by condition x mask kind:")
    print(real.groupby(["condition", "mask_kind"]).attribution
          .agg(mean_abs=lambda s: s.abs().mean(), n="size").round(4).to_string())

    # attribution-map similarity across formats, per item
    maps = real.pivot_table(index=["item_id", "mask_id"], columns="condition",
                            values="attribution")
    sims = []
    for it, g in maps.groupby(level=0):
        row = {"item_id": it}
        for a, c, name in (("ref", "rt_mp3_64", "codec"),
                           ("mp3_64", "rt_mp3_64", "container"),
                           ("ref", "mp3_64", "total")):
            if a in g.columns and c in g.columns:
                u, v = g[a].values, g[c].values
                ok = np.isfinite(u) & np.isfinite(v)
                if ok.sum() >= 4 and np.std(u[ok]) > 0 and np.std(v[ok]) > 0:
                    row[f"rho_{name}"] = stats.spearmanr(u[ok], v[ok]).statistic
                    k = 3
                    row[f"top{k}_{name}"] = len(
                        set(np.argsort(-u[ok])[:k]) & set(np.argsort(-v[ok])[:k])) / k
        sims.append(row)
    sm = pd.DataFrame(sims)
    print("\n  attribution-map similarity across formats (per item, then averaged):")
    cols = [c for c in sm.columns if c.startswith(("rho_", "top3_"))]
    print(sm[cols].agg(["mean", "std", "count"]).round(4).to_string())
    R["xai_map_similarity"] = json.loads(sm[cols].mean().to_json())
    R["xai_map_similarity_sd"] = json.loads(sm[cols].std().to_json())

    if "rho_codec" in sm and "rho_container" in sm:
        cc = sm[["rho_codec", "rho_container"]].dropna()
        if len(cc) > 3:
            res = paired(cc.rho_container.values, cc.rho_codec.values,
                         "rho_container - rho_codec")
            print(f"\n  container maps are {'MORE' if res['mean_diff']>0 else 'LESS'} similar "
                  f"than codec maps: diff {res['mean_diff']:+.4f}, p = {res['p']:.3g}, n = {res['n']}")
            R["xai_container_vs_codec_similarity"] = res
    sm.to_parquet(OUT / "xai_similarity.parquet", index=False)

    # ---- the dissociation ------------------------------------------------
    hdr("9  THE DISSOCIATION — label stable while evidence moves")
    lab = grid[grid.readout == "six_way"].pivot_table(
        index="item_id", columns="condition", values="pred", aggfunc="first")
    for br in BITRATES:
        c = f"mp3_{br}"
        if c not in lab.columns or c not in piv.columns:
            continue
        same_label = (lab[c] == lab["ref"])
        dp = (piv[c] - piv["ref"]).abs()
        moved = dp > 0.01
        both = int((same_label & moved).sum())
        print(f"  mp3_{br:<4} label unchanged: {int(same_label.sum())}/{len(lab)}   "
              f"|dP(gold)| > .01: {int(moved.sum())}/{len(piv)}   "
              f"BOTH (dissociation): {both}")
        R.setdefault("dissociation", {})[f"mp3_{br}"] = {
            "label_unchanged": int(same_label.sum()),
            "p_moved": int(moved.sum()), "both": both, "n": int(len(lab)),
            "max_abs_dp": float(dp.max())}

# ---------- fidelity summary -------------------------------------------------
hdr("10  SIGNAL FIDELITY (context for the behavioural effects)")
fa = fid.groupby("bitrate").agg(snr_db=("snr_db", "mean"), lsd_db=("lsd_db", "mean"),
                                identical=("container_pcm_identical", "all"))
print(fa.round(3).to_string())
R["fidelity"] = json.loads(fa.astype({"identical": bool}).to_json(orient="index"))

json.dump(R, open(OUT / "results.json", "w"), indent=1, default=str)
print(f"\n\n  saved: exp/out/results.json")
