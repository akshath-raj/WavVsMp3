#!/usr/bin/env python
"""Presentation figures focused on the WAV-vs-MP3 contrast and its XAI consequences.

The EDA figures in `outputs/eda/` cover all four conditions and every analysis.
These five are cut specifically for the talk: what MP3 does to the signal, what
it does to the *explanations*, and how that differs by model family.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from xai_ser.datasets import family_of  # noqa: E402
from xai_ser.paths import EDA_DIR, OUTPUTS, XAI_DIR  # noqa: E402

sns.set_theme(style="whitegrid", context="talk")
OUT = OUTPUTS / "presentation"
OUT.mkdir(parents=True, exist_ok=True)

WAV = "#2E6F9E"
MP3 = "#C4462E"
FAMILY_COLORS = {"spectral": "#C4462E", "mfcc": "#2E6F9E",
                 "prosody": "#3F8F5B", "chroma": "#8A6BBE", "other": "#8C8C8C"}


def save(fig, name):
    fig.tight_layout()
    path = OUT / name
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def fig_signal_drift():
    """What MP3 @64k does to the acoustic features, relative to the WAV reference."""
    d = pd.read_csv(EDA_DIR / "format_drift.csv")
    mp3 = d[d.condition == "mp3_64"].copy()
    top = mp3.reindex(mp3.abs_smd.sort_values(ascending=False).index).head(14)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
    colors = [FAMILY_COLORS[f] for f in top.family]
    axes[0].barh(range(len(top)), top.smd, color=colors)
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels(top.feature, fontsize=11)
    axes[0].invert_yaxis()
    axes[0].axvline(0, c="k", lw=1)
    axes[0].set_xlabel("standardised mean difference vs WAV (training SDs)")
    axes[0].set_title("MP3 @ 64 kbps moves a few features enormously", fontsize=15, pad=12)
    axes[0].set_xscale("symlog", linthresh=1)

    # The distribution: almost everything is untouched, a thin tail is destroyed.
    for cond, color, label in [("mp3_64", MP3, "MP3 @ 64 kbps"),
                               ("mp4_aac64", "#B08A3E", "MP4/AAC @ 64 kbps")]:
        vals = d[d.condition == cond].abs_smd.clip(1e-5, None)
        axes[1].hist(np.log10(vals), bins=60, alpha=0.55, color=color, label=label)
    axes[1].axvline(np.log10(0.2), ls="--", c="k", lw=1.2)
    axes[1].text(np.log10(0.2), axes[1].get_ylim()[1] * 0.92, "  |SMD| = 0.2", fontsize=11)
    axes[1].set_xlabel("log10 |standardised mean difference|")
    axes[1].set_ylabel("features")
    axes[1].set_title("Median feature barely moves;\nthe damage is all in the tail", fontsize=15, pad=12)
    axes[1].legend(fontsize=11)
    save(fig, "P1_signal_drift_mp3.png")


def fig_attribution_shift():
    """The core XAI result: where the ANN's attention goes under MP3."""
    deep = XAI_DIR / "deep" / (XAI_DIR / "deep" / "LATEST").read_text().strip()
    ref = pd.read_csv(deep / "ig_global_ref.csv").set_index("feature").importance
    mp3 = pd.read_csv(deep / "ig_global_mp3_64.csv").set_index("feature").importance.reindex(ref.index)
    rep = json.load(open(deep / "deep_xai_summary.json"))
    st = rep["attribution_stability_ig"]["mp3_64"]

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))

    entered, left = set(st["entered_topk"]), set(st["left_topk"])
    kept = set(ref.nlargest(25).index) & set(mp3.nlargest(25).index)

    def cat(f):
        if f in entered:
            return "entered top-25 (codec artefact)"
        if f in left:
            return "left top-25 (voice evidence)"
        return "unchanged"

    # Log-log: attribution magnitudes span four orders of magnitude, so a linear
    # axis buries every feature that matters in the bottom-left corner.
    cats = pd.Series({f: cat(f) for f in ref.index})
    style = {"unchanged": ("#B9C4CC", 12, 0.45),
             "entered top-25 (codec artefact)": (MP3, 80, 1.0),
             "left top-25 (voice evidence)": ("#3F8F5B", 80, 1.0)}
    floor = 1e-4
    for label, (c, s, a) in style.items():
        m = (cats == label).to_numpy()
        axes[0].scatter(ref[m].clip(floor), mp3[m].clip(floor), c=c, s=s, alpha=a,
                        label=label, edgecolor="white", linewidth=0.4,
                        zorder=3 if s > 20 else 1)
    lo, hi = floor, float(max(ref.max(), mp3.max())) * 1.4
    axes[0].plot([lo, hi], [lo, hi], ls="--", c="k", lw=1, zorder=0)
    axes[0].set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
                xlabel="attribution on WAV (mean |IG|)",
                ylabel="attribution on MP3 (mean |IG|)")
    axes[0].set_title(f"ANN attributions relocate under MP3\nSpearman ρ = {st['spearman_rho']:.2f}, "
                      f"{st['top25_overlap']}/25 top features retained", fontsize=15, pad=12)
    axes[0].legend(fontsize=10, loc="upper left", framealpha=0.95)

    # Rank migration: where each displaced feature sat on WAV, and where it lands
    # on MP3. Ranks say more than in/out membership.
    r_wav = ref.rank(ascending=False)
    r_mp3 = mp3.rank(ascending=False)
    # Each feature is named at the end where it ranks high — displaced ones on
    # the left, promoted ones on the right — so the labels never collide.
    picks = (sorted(left, key=lambda f: r_wav[f])[:8]
             + sorted(entered, key=lambda f: r_mp3[f])[:8])
    for f in picks:
        promoted = f in entered
        color = MP3 if promoted else "#3F8F5B"
        axes[1].plot([0, 1], [r_wav[f], r_mp3[f]], "-o", color=color, ms=7, lw=2.2, alpha=0.9)
        if promoted:
            axes[1].text(1.05, r_mp3[f], f"  {f}", ha="left", va="center", fontsize=9.5)
            axes[1].text(-0.05, r_wav[f], f"#{int(r_wav[f])}", ha="right", va="center",
                         fontsize=9, color="#7A7A7A")
        else:
            axes[1].text(-0.05, r_wav[f], f, ha="right", va="center", fontsize=9.5)
            axes[1].text(1.05, r_mp3[f], f"#{int(r_mp3[f])}", ha="left", va="center",
                         fontsize=9, color="#7A7A7A")
    axes[1].set_yscale("log")
    axes[1].invert_yaxis()
    axes[1].set_xlim(-1.05, 2.15)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["rank on WAV", "rank on MP3"], fontsize=12)
    axes[1].set_ylabel("attribution rank (1 = most important)")
    axes[1].grid(axis="x", visible=False)
    axes[1].set_title("Voice evidence falls, codec artefacts rise", fontsize=15, pad=12)
    save(fig, "P2_attribution_shift_mp3.png")


def fig_stability_by_family():
    """Explanation stability under MP3, by model family."""
    xai = XAI_DIR / (XAI_DIR / "LATEST").read_text().strip()
    rep = json.load(open(xai / "xai_summary.json"))
    deep = XAI_DIR / "deep" / (XAI_DIR / "deep" / "LATEST").read_text().strip()
    drep = json.load(open(deep / "deep_xai_summary.json"))

    rows = []
    for name, e in rep["models"].items():
        st = e.get("attribution_stability", {}).get("mp3_64")
        if st:
            rows.append({"model": name, "rho": st["spearman_rho"],
                         "overlap": st["top25_overlap"], "kind": "SHAP"})
    st = drep["attribution_stability_ig"]["mp3_64"]
    rows.append({"model": "ann_pytorch", "rho": st["spearman_rho"],
                 "overlap": st["top25_overlap"], "kind": "Integrated Gradients"})
    df = pd.DataFrame(rows).sort_values("rho", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6))
    colors = ["#2E6F9E" if k == "SHAP" else MP3 for k in df.kind]
    axes[0].barh(df.model, df.rho, color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Spearman ρ, attribution ranking MP3 vs WAV")
    axes[0].set_title("Explanation stability under MP3", fontsize=15, pad=12)
    for i, (r, o) in enumerate(zip(df.rho, df.overlap)):
        axes[0].text(r + 0.015, i, f"{r:.3f}  ({o}/25)", va="center", fontsize=11)

    axes[1].barh(df.model, df.overlap, color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 26)
    axes[1].axvline(25, ls="--", c="k", lw=1)
    axes[1].set_xlabel("top-25 features retained")
    axes[1].set_title("How many of the top 25 survive", fontsize=15, pad=12)
    save(fig, "P3_stability_by_family.png")


def fig_neutralisation():
    """Masking the codec-shifted features restores the models."""
    rb = OUTPUTS / "robustness" / (OUTPUTS / "robustness" / "LATEST").read_text().strip()
    r = pd.read_csv(rb / "neutralisation_curves.csv")
    g = r[r.condition == "mp3_64"]

    fig, ax = plt.subplots(figsize=(11, 6.2))
    order = g[g.n_neutralised == 0].sort_values("uar_compressed").model
    palette = sns.color_palette("colorblind", len(order))
    for c, m in zip(palette, order):
        gm = g[g.model == m].sort_values("n_neutralised")
        ax.plot(gm.n_neutralised, gm.uar_compressed, marker="o", ms=5, label=m, color=c)
    clean = g.groupby("n_neutralised").uar_clean_control.mean()
    ax.plot(clean.index, clean.values, ls="--", c="k", lw=1.4, label="clean-WAV control (mean)")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 2, 3, 5, 10, 20, 40, 80])
    ax.set_xticklabels([0, 1, 2, 3, 5, 10, 20, 40, 80])
    ax.set(xlabel="codec-shifted features neutralised to training median",
           ylabel="balanced accuracy on MP3")
    ax.set_title("Masking one feature undoes most of the MP3 damage", fontsize=15, pad=12)
    ax.legend(fontsize=10, ncol=2)
    save(fig, "P4_neutralisation_mp3.png")


def fig_method_agreement():
    """Do the XAI methods even agree with each other?"""
    xai = XAI_DIR / (XAI_DIR / "LATEST").read_text().strip()
    rep = json.load(open(xai / "xai_summary.json"))
    rows = []
    for name, e in rep["models"].items():
        for k, v in e.get("lime_vs_shap", {}).items():
            rows.append({"model": name, "comparison": k.replace("shap_vs_", ""),
                         "rho": v["spearman_rho"]})
    df = pd.DataFrame(rows).pivot(index="model", columns="comparison", values="rho")

    deep = XAI_DIR / "deep" / (XAI_DIR / "deep" / "LATEST").read_text().strip()
    agree = pd.read_csv(deep / "method_agreement.csv", index_col=0).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(17, 6),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    sns.heatmap(df, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                ax=axes[0], cbar_kws={"label": "Spearman ρ"}, linewidths=0.5)
    axes[0].set_title("Classical models: SHAP vs other methods\n(low = the methods disagree)",
                      fontsize=14, pad=12)
    axes[0].set_xlabel("")

    sns.heatmap(agree, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                ax=axes[1], cbar_kws={"label": "Spearman ρ"}, linewidths=0.5)
    axes[1].set_title("ANN: gradient methods agree with each other", fontsize=14, pad=12)
    save(fig, "P5_method_agreement.png")


if __name__ == "__main__":
    fig_signal_drift()
    fig_attribution_shift()
    fig_stability_by_family()
    fig_neutralisation()
    fig_method_agreement()
    print("done ->", OUT)
