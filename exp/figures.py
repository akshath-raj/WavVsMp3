"""Publication figures for the WAV-vs-MP3 study."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "exp" / "out"
FIG = ROOT / "exp" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "grid.linewidth": .6,
    "axes.axisbelow": True, "figure.facecolor": "white",
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
})

C_REF, C_RT, C_MP3 = "#2b6cb0", "#dd6b20", "#38a169"
BITRATES = [32, 64, 128]
ORDER = ["ref", "rt_mp3_32", "rt_mp3_64", "rt_mp3_128", "mp3_32", "mp3_64", "mp3_128"]

grid = pd.read_parquet(OUT / "grid.parquet")
fid = pd.read_parquet(OUT / "fidelity.parquet")
R = json.load(open(OUT / "results.json"))
b = grid[grid.readout == "binary_gold"]
piv = b.pivot_table(index="item_id", columns="condition", values="p_gold")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


# --- F1: P(gold) by condition ----------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.6))
cols = [c for c in ORDER if c in piv.columns]
xs = np.arange(len(cols))
means = [piv[c].mean() for c in cols]
cis = [R["p_gold_by_condition"].get(c, {}) for c in cols]
lo = [m - c.get("ci_lo", m) for m, c in zip(means, cis)]
hi = [c.get("ci_hi", m) - m for m, c in zip(means, cis)]
colors = [C_REF if c == "ref" else (C_RT if c.startswith("rt_") else C_MP3) for c in cols]
for i, c in enumerate(cols):
    y = piv[c].dropna().values
    ax.scatter(np.full(len(y), i) + np.random.default_rng(i).normal(0, .06, len(y)),
               y, s=7, alpha=.28, color=colors[i], linewidths=0)
ax.errorbar(xs, means, yerr=[lo, hi], fmt="o", ms=7, capsize=4,
            color="black", zorder=5, lw=1.4)
ax.set_xticks(xs)
ax.set_xticklabels([c.replace("rt_mp3_", "WAV←mp3\n").replace("mp3_", "MP3\n")
                    .replace("ref", "WAV\nlossless") for c in cols])
ax.set_ylabel("P(gold emotion)")
ax.set_title("Evidence for the true emotion, by delivery format")
h = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in
     ((C_REF, "lossless WAV"), (C_RT, "WAV carrying mp3-decoded signal"), (C_MP3, "MP3 container"))]
ax.legend(handles=h, loc="upper right", fontsize=7.5)
save(fig, "f1_pgold_by_condition.png")

# --- F2: contrast forest plot ----------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.8))
labels, mids, los, his, cols_ = [], [], [], [], []
palette = {"codec": C_RT, "container": "#805ad5", "total": C_MP3}
for kind in ("codec", "container", "total"):
    for br in BITRATES:
        k = f"{kind}_{br}"
        if k not in R["contrasts"]:
            continue
        v = R["contrasts"][k]
        labels.append(f"{kind} @ {br}k")
        mids.append(v["mean_diff"])
        ci = v["ci95"]
        los.append(v["mean_diff"] - (ci[0] if np.isfinite(ci[0]) else v["mean_diff"]))
        his.append((ci[1] if np.isfinite(ci[1]) else v["mean_diff"]) - v["mean_diff"])
        cols_.append(palette[kind])
y = np.arange(len(labels))[::-1]
ax.errorbar(mids, y, xerr=[los, his], fmt="o", ms=6, capsize=3, lw=1.4,
            ecolor="#555", ls="none")
ax.scatter(mids, y, c=cols_, s=48, zorder=5)
ax.axvline(0, color="black", lw=1, ls="--", alpha=.7)
s = R.get("tost_sesoi")
if s:
    ax.axvspan(-s, s, color="grey", alpha=.13, lw=0)
    ax.text(0, len(labels) - .3, f"±SESOI ({s:.3f})", ha="center", fontsize=7, color="#555")
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_xlabel("Δ P(gold)  (positive = more evidence than the comparison)")
ax.set_title("Codec, container, and total format effects")
save(fig, "f2_contrasts.png")

# --- F3: dose-response + fidelity ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
ax = axes[0]
rows = []
for it in piv.index:
    for br in BITRATES:
        c = f"rt_mp3_{br}"
        if c in piv.columns and np.isfinite(piv.loc[it, c]) and np.isfinite(piv.loc[it, "ref"]):
            rows.append({"bitrate": br, "delta": piv.loc[it, c] - piv.loc[it, "ref"]})
dr = pd.DataFrame(rows)
g = dr.groupby("bitrate").delta
ax.axhline(0, color="black", lw=1, ls="--", alpha=.6)
for br in BITRATES:
    v = dr[dr.bitrate == br].delta.values
    ax.scatter(np.full(len(v), br) * (1 + np.random.default_rng(br).normal(0, .012, len(v))),
               v, s=8, alpha=.3, color=C_RT, linewidths=0)
ax.errorbar(BITRATES, g.mean(), yerr=g.std() / np.sqrt(g.count()),
            fmt="o-", color="black", ms=6, capsize=4, lw=1.4)
ax.set_xscale("log")
ax.set_xticks(BITRATES)
ax.set_xticklabels([f"{b}k" for b in BITRATES])
ax.set_xlabel("MP3 bitrate")
ax.set_ylabel("Δ P(gold) vs lossless")
ax.set_title("Codec effect vs bitrate")

ax = axes[1]
fa = fid.groupby("bitrate").agg(snr=("snr_db", "mean"), lsd=("lsd_db", "mean"))
ax.plot(fa.index, fa.snr, "o-", color=C_REF, label="SNR (dB)")
ax2 = ax.twinx()
ax2.plot(fa.index, fa.lsd, "s--", color=C_MP3, label="LSD (dB)")
ax2.grid(False)
ax.set_xscale("log")
ax.set_xticks(BITRATES)
ax.set_xticklabels([f"{b}k" for b in BITRATES])
ax.set_xlabel("MP3 bitrate")
ax.set_ylabel("SNR (dB)", color=C_REF)
ax2.set_ylabel("LSD (dB)", color=C_MP3)
ax.set_title("Signal fidelity vs bitrate")
save(fig, "f3_dose_response_fidelity.png")

# --- F4/F5: XAI -------------------------------------------------------------
xp = OUT / "xai.parquet"
if xp.exists():
    x = pd.read_parquet(xp)
    real = x[x.mask_kind.isin(["temporal", "spectral"])]
    mm = real.pivot_table(index="mask_id", columns="condition",
                          values="attribution", aggfunc="mean")
    t_order = [f"t{i}" for i in range(10)]
    f_order = [m for m in mm.index if m.startswith("f")]

    def band_key(m):
        return int(m.split("_")[0][1:])
    f_order = sorted(f_order, key=band_key)
    mm = mm.reindex([m for m in t_order + f_order if m in mm.index])
    conds = [c for c in ["ref", "rt_mp3_64", "mp3_64"] if c in mm.columns]

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    v = np.abs(mm[conds].values).max()
    im = ax.imshow(mm[conds].T.values, aspect="auto", cmap="RdBu_r",
                   vmin=-v, vmax=v)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([{"ref": "WAV lossless", "rt_mp3_64": "WAV←mp3 64k",
                         "mp3_64": "MP3 64k"}[c] for c in conds])
    ax.set_xticks(range(len(mm.index)))
    ax.set_xticklabels(
        [m.replace("t", "T") if m.startswith("t") else
         m.replace("f", "").replace("_", "–") + "Hz" for m in mm.index],
        rotation=60, ha="right", fontsize=7)
    ax.axvline(9.5, color="black", lw=1.2)
    ax.text(4.5, -.72, "temporal windows", ha="center", fontsize=8, style="italic")
    ax.text(len(mm.index) - 3.5, -.72, "frequency bands", ha="center", fontsize=8,
            style="italic")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="mean attribution\nP(gold) unmasked − masked\n(negative = masking RAISED P(gold))",
                 fraction=.03, pad=.02)
    ax.set_title("Where masking most changes the model's output, by delivery format",
                 pad=22)
    save(fig, "f4_attribution_maps.png")

    sp = OUT / "xai_similarity.parquet"
    if sp.exists():
        sm = pd.read_parquet(sp)
        pairs = [("rho_container", "container\n(same signal, different container)", "#805ad5"),
                 ("rho_codec", "codec\n(mp3 loss, same container)", C_RT),
                 ("rho_total", "total\n(both)", C_MP3)]
        pairs = [p for p in pairs if p[0] in sm.columns]
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        data = [sm[p[0]].dropna().values for p in pairs]
        parts = ax.violinplot(data, showmeans=True, widths=.75)
        for pc, p in zip(parts["bodies"], pairs):
            pc.set_facecolor(p[2])
            pc.set_alpha(.45)
        for key in ("cmeans", "cmaxes", "cmins", "cbars"):
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(1.1)
        for i, d in enumerate(data, 1):
            ax.scatter(np.full(len(d), i) + np.random.default_rng(i).normal(0, .045, len(d)),
                       d, s=9, alpha=.35, color="black", linewidths=0)
        ax.axhline(1.0, color="black", ls="--", lw=1, alpha=.6)
        ax.text(len(pairs) + .45, 1.0, "identical\nmaps", va="center", fontsize=7)
        ax.set_xticks(range(1, len(pairs) + 1))
        ax.set_xticklabels([p[1] for p in pairs], fontsize=8)
        ax.set_ylabel("Spearman ρ between attribution maps")
        ax.set_title("Does the evidence base survive the format change?")
        save(fig, "f5_attribution_similarity.png")

# --- F6: dissociation --------------------------------------------------------
lab = grid[grid.readout == "six_way"].pivot_table(
    index="item_id", columns="condition", values="pred", aggfunc="first")
if "mp3_64" in piv.columns and "mp3_64" in lab.columns:
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    d = (piv["mp3_64"] - piv["ref"]).dropna()
    same = (lab["mp3_64"] == lab["ref"]).reindex(d.index).fillna(False)
    ax.scatter(piv["ref"].reindex(d.index), piv["mp3_64"].reindex(d.index),
               c=np.where(same, C_MP3, "#e53e3e"), s=26, alpha=.8, linewidths=0)
    lim = [0, max(1e-3, float(np.nanmax([piv["ref"].max(), piv["mp3_64"].max()])) * 1.08)]
    ax.plot(lim, lim, ls="--", color="black", lw=1, alpha=.6)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("P(gold) — lossless WAV")
    ax.set_ylabel("P(gold) — MP3 64k")
    n_same = int(same.sum())
    ax.set_title(f"Same label on {n_same}/{len(d)} items,\nyet the evidence moves off the diagonal")
    save(fig, "f6_dissociation.png")

print(f"\nfigures written to exp/figures/")
