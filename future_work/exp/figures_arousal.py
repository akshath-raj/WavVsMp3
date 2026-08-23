"""Figures for the arousal arm. The headline panel is the null-baseline comparison."""
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

C_NULL, C_CONT, C_CODEC, C_TOT = "#718096", "#805ad5", "#dd6b20", "#38a169"
grid = pd.read_parquet(OUT / "arousal_grid.parquet")
xai = pd.read_parquet(OUT / "arousal_xai.parquet")
sm = pd.read_parquet(OUT / "arousal_similarity.parquet")
R = json.load(open(OUT / "arousal_results.json"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


# --- A1: discriminability + P(high) separation ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
ax = axes[0]
s = grid[(grid.condition == "ref") & grid.p_high.notna()]
for i, (lab, col) in enumerate((("high", "#c53030"), ("low", "#2b6cb0"))):
    v = s[s.arousal == lab].p_high.values
    ax.scatter(np.full(len(v), i) + np.random.default_rng(i).normal(0, .07, len(v)),
               v, s=22, alpha=.65, color=col, linewidths=0)
    ax.hlines(np.mean(v), i - .28, i + .28, color="black", lw=2, zorder=5)
ax.set_xticks([0, 1])
ax.set_xticklabels(["true high\narousal", "true low\narousal"])
ax.set_ylabel("P(high)")
auc = R["discriminability"]["ref"]["auc"]
p = R["discriminability"]["ref"]["p"]
ax.set_title(f"The model does discriminate arousal\nAUC = {auc:.3f}, p = {p:.3g}")

ax = axes[1]
order = ["ref", "rt_mp3_32", "rt_mp3_64", "rt_mp3_128", "mp3_32", "mp3_64", "mp3_128"]
dd = pd.DataFrame(R["discriminability"]).T.reindex([c for c in order if c in R["discriminability"]])
xs = np.arange(len(dd))
ax.bar(xs, dd.auc.astype(float), color=["#2b6cb0" if c == "ref" else
                                        ("#dd6b20" if c.startswith("rt_") else "#38a169")
                                        for c in dd.index])
ax.axhline(.5, color="black", ls="--", lw=1)
ax.text(len(dd) - .5, .51, "chance", fontsize=7, ha="right")
ax.set_xticks(xs)
ax.set_xticklabels([c.replace("rt_mp3_", "WAV←\n") .replace("mp3_", "MP3\n")
                    .replace("ref", "WAV\nlossless") for c in dd.index], fontsize=7)
ax.set_ylabel("AUC of P(high)")
ax.set_ylim(0, 1)
ax.set_title("Discriminability survives every format")
save(fig, "a1_discriminability.png")

# --- A2: THE headline — effects against the inaudible null ------------------
fig, ax = plt.subplots(figsize=(7.2, 3.6))
floor = R["dither_floor"]["mean_abs"]
names, vals, cols = ["1-LSB dither\n(inaudible, NULL)"], [floor], [C_NULL]
for k, lab, c in (("container_64", "container @64k\n(identical PCM)", C_CONT),
                  ("codec_64", "codec @64k\n(mp3 loss)", C_CODEC),
                  ("total_64", "total @64k", C_TOT)):
    if k in R["contrasts"]:
        names.append(lab)
        vals.append(R["contrasts"][k]["mean_abs"])
        cols.append(c)
xs = np.arange(len(names))
ax.bar(xs, vals, color=cols, width=.62)
ax.axhline(floor, color=C_NULL, ls="--", lw=1.3)
ax.text(len(names) - .4, floor * 1.06, "inaudible-change floor", fontsize=7,
        ha="right", color="#444")
for i, v in enumerate(vals):
    ax.text(i, v * 1.03, f"{v:.4f}" + (f"\n{v/max(floor,1e-12):.1f}×" if i else ""),
            ha="center", fontsize=7.5)
ax.set_xticks(xs)
ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("mean |Δ P(high)| per item")
ax.set_title("Is the format effect bigger than an inaudible perturbation?")
save(fig, "a2_null_baseline.png")

# --- A3: attribution map similarity vs the null -----------------------------
pairs = [("rho_dither (NULL)", "1-LSB dither\n(inaudible, NULL)", C_NULL),
         ("rho_container", "container\n(identical PCM)", C_CONT),
         ("rho_codec", "codec\n(mp3 loss)", C_CODEC),
         ("rho_total", "total\n(WAV vs MP3)", C_TOT)]
pairs = [p for p in pairs if p[0] in sm.columns]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
data = [sm[p[0]].dropna().values for p in pairs]
parts = ax.violinplot(data, showmeans=True, widths=.75)
for pc, p in zip(parts["bodies"], pairs):
    pc.set_facecolor(p[2])
    pc.set_alpha(.45)
for key in ("cmeans", "cmaxes", "cmins", "cbars"):
    if key in parts:
        parts[key].set_color("black")
        parts[key].set_linewidth(1.1)
for i, dv in enumerate(data, 1):
    ax.scatter(np.full(len(dv), i) + np.random.default_rng(i).normal(0, .045, len(dv)),
               dv, s=9, alpha=.35, color="black", linewidths=0)
    ax.text(i, 1.06, f"{np.mean(dv):.2f}", ha="center", fontsize=8, weight="bold")
ax.axhline(1.0, color="black", ls="--", lw=1, alpha=.6)
if data:
    ax.axhline(np.mean(data[0]), color=C_NULL, ls=":", lw=1.3)
ax.set_xticks(range(1, len(pairs) + 1))
ax.set_xticklabels([p[1] for p in pairs], fontsize=8)
ax.set_ylabel("Spearman ρ between attribution maps")
ax.set_ylim(-.4, 1.14)
ax.set_title("Do format changes disturb explanations more than an inaudible change?")
save(fig, "a3_map_similarity_vs_null.png")

print("\narousal figures written")
