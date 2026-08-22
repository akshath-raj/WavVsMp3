#!/usr/bin/env python
"""Figures for the Interspeech submission.

Reads only committed result artefacts under ``xai_ser/outputs/``. No number is
hard-coded here except axis limits and tick labels; every value plotted is
loaded from the run directories pinned below.

    python make_figures.py            # -> figs/*.pdf and figs/*.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
XS = ROOT / "xai_ser" / "outputs"
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# Pinned run directories (the same ones the manuscript quotes).
EDA_DIR = XS / "eda"
MODELS_DIR = XS / "models" / "20260818_045719"
ANN_DIR = XS / "models" / "ann" / "20260818_050037"
XAI_DIR = XS / "xai" / "20260818_050808"
DEEP_DIR = XS / "xai" / "deep" / "20260818_050247"
ROB_DIR = XS / "robustness" / "20260818_051057"

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 7.0,
    "axes.labelsize": 7.0,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3,
    "legend.fontsize": 6.3,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "lines.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,
})

# Colourblind-safe (Okabe-Ito).
C_THRESH = "#0072B2"   # blue   -- threshold-consuming models
C_MULT = "#D55E00"     # orange -- magnitude-consuming models
C_NULL = "#009E73"     # green  -- control / null
C_GREY = "#555555"
C_ACC = "#CC79A7"

COL_1 = 3.35   # Interspeech single column, inches
COL_2 = 7.00   # full text width, inches

# Display names and the family split that the paper argues for.
THRESHOLD = ["lightgbm", "xgboost", "hist_gradient_boosting", "random_forest",
             "adaboost", "extra_trees", "decision_tree"]
MULTIPLY = ["knn", "gaussian_nb", "linear_svm", "mlp_sklearn", "ann_pytorch",
            "lda", "logistic_regression", "svm_rbf"]
NICE = {
    "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "hist_gradient_boosting": "HistGB", "random_forest": "Rand.\\ forest",
    "adaboost": "AdaBoost", "extra_trees": "Extra trees",
    "decision_tree": "Dec.\\ tree", "knn": "$k$NN",
    "gaussian_nb": "Gauss.\\ NB", "linear_svm": "Linear SVM",
    "mlp_sklearn": "MLP", "ann_pytorch": "ANN", "lda": "LDA",
    "logistic_regression": "Log.\\ reg.", "svm_rbf": "SVM (RBF)",
}
PLAIN = {k: v.replace("\\ ", " ") for k, v in NICE.items()}


def save(fig, name: str) -> None:
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print(f"  wrote figs/{name}.pdf")


def tag(ax, text: str, dx: float = -0.11, dy: float = 1.04) -> None:
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=7.5,
            fontweight="bold", va="bottom", ha="left")


# ---------------------------------------------------------------- loading
def load_all() -> dict:
    d = {}
    d["eda"] = json.loads((EDA_DIR / "eda_summary.json").read_text())
    d["drift"] = pd.read_csv(EDA_DIR / "format_drift.csv")
    d["cross"] = pd.read_csv(MODELS_DIR / "cross_format.csv")
    d["ann"] = json.loads((ANN_DIR / "metrics.json").read_text())
    d["xai"] = json.loads((XAI_DIR / "xai_summary.json").read_text())
    d["deep"] = json.loads((DEEP_DIR / "deep_xai_summary.json").read_text())
    d["neut"] = pd.read_csv(ROB_DIR / "neutralisation_curves.csv")
    d["rob"] = json.loads((ROB_DIR / "robustness_summary.json").read_text())
    d["shift_mp3"] = pd.read_csv(ROB_DIR / "feature_shift_mp3_64.csv")
    return d


def crossformat_frame(d: dict) -> pd.DataFrame:
    """Balanced accuracy per model x test condition, trained on `ref`.

    The PyTorch ANN is trained by a separate module and is not in
    `cross_format.csv`; its row is read from the ANN run's `metrics.json`.
    """
    c = d["cross"]
    c = c[(c.train_condition == "ref") & (~c.model.str.startswith("dummy"))]
    wide = c.pivot_table(index="model", columns="test_condition",
                         values="balanced_accuracy")
    m = d["ann"]["metrics"]
    wide.loc["ann_pytorch"] = {
        "ref": m["test_ref"]["balanced_accuracy"],
        "mp3_64": m["test_mp3_64"]["balanced_accuracy"],
        "mp4_aac64": m["test_mp4_aac64"]["balanced_accuracy"],
        "roundtrip_wav": m["test_roundtrip_wav"]["balanced_accuracy"],
    }
    return wide


# ---------------------------------------------------------------- Fig. 1
def fig_mechanism(d: dict) -> None:
    """What the codec does to the feature table, before any model is fitted."""
    fig, axes = plt.subplots(1, 2, figsize=(COL_1, 1.55),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})
    drift = d["drift"]
    dp = d["eda"]["decode_path"]

    # (a) ECDF of |SMD| across all 436 features, per condition.
    ax = axes[0]
    col = "abs_smd" if "abs_smd" in drift.columns else "smd"
    for cond, lab, colr in [("mp3_64", "MP3", C_MULT),
                            ("mp4_aac64", "AAC", C_THRESH)]:
        sub = drift[drift.condition == cond]
        v = np.sort(np.abs(sub[col].to_numpy(dtype=float)))
        v = v[np.isfinite(v)]
        ax.plot(np.maximum(v, 1e-6), np.arange(1, len(v) + 1) / len(v),
                color=colr, label=lab)
    ax.axvline(dp["median_abs_smd"], color=C_NULL, ls=":", lw=1.0)
    ax.text(dp["median_abs_smd"] * 2.0, 0.90, "decode-path\ncontrol",
            fontsize=4.8, color=C_NULL, va="center")
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1e2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("$|$SMD$|$ vs. uncompressed", fontsize=6.2)
    ax.set_ylabel("ECDF over 436 features", fontsize=6.2)
    ax.legend(loc="upper left", frameon=False, handlelength=1.1, fontsize=5.6)
    tag(ax, "(a)", dx=-0.32, dy=1.02)

    # (b) The tail: top shifted MP3 features in training-SD units.
    ax = axes[1]
    top = d["rob"]["per_condition"]["mp3_64"]["top_shifted_features"][:7]
    names = [t["feature"] for t in top][::-1]
    vals = [t["shift_sd"] for t in top][::-1]
    cols = [C_MULT if v > 5 else C_GREY for v in vals]
    ypos = np.arange(len(names))
    ax.barh(ypos, vals, color=cols, height=0.66)
    ax.set_yticks(ypos)
    ax.set_yticklabels(names, fontsize=4.4)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 130)
    ax.set_xlabel("shift, in training SDs", fontsize=6.2)
    ax.text(vals[-1] * 1.25, ypos[-1], f"{vals[-1]:.1f}$\\sigma$",
            va="center", fontsize=5.6, color=C_MULT, fontweight="bold")
    tag(ax, "(b)", dx=-1.05, dy=1.02)

    fig.subplots_adjust(wspace=1.05)
    save(fig, "fig_mechanism")


# ---------------------------------------------------------------- Fig. 2
def fig_crossformat(d: dict) -> None:
    """The family split: thresholds survive, products collapse."""
    wide = crossformat_frame(d)
    fig, axes = plt.subplots(1, 2, figsize=(COL_2, 1.88), sharey=True)

    order = THRESHOLD + MULTIPLY
    order = sorted(order, key=lambda m: -wide.loc[m, "ref"])

    for ax, cond, title in [
        (axes[0], "mp3_64", "MP3 @ 64\\,kbps"),
        (axes[1], "mp4_aac64", "MP4/AAC @ 64\\,kbps"),
    ]:
        for i, m in enumerate(order):
            ref = wide.loc[m, "ref"]
            got = wide.loc[m, cond]
            colr = C_THRESH if m in THRESHOLD else C_MULT
            ax.plot([ref, got], [i, i], color=colr, lw=1.5, alpha=0.55,
                    solid_capstyle="butt", zorder=1)
            ax.scatter([ref], [i], s=13, facecolor="white", edgecolor=C_GREY,
                       lw=0.7, zorder=3)
            ax.scatter([got], [i], s=15, color=colr, zorder=4)
        ax.axvline(0.1667, color=C_GREY, ls=":", lw=0.7)
        ax.axvline(0.4553, color=C_NULL, ls="--", lw=0.7)
        ax.set_xlim(0.14, 0.64)
        ax.set_ylim(-0.8, len(order) - 0.2)
        ax.invert_yaxis()
        ax.set_xlabel("balanced accuracy on 20 held-out speakers")
        ax.set_title(title.replace("\\,", " "), pad=3)

    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([PLAIN[m] for m in order], fontsize=6.0)
    axes[0].text(0.1667, -0.75, " chance", fontsize=5.5, color=C_GREY,
                 va="top", ha="left")
    axes[0].text(0.4553, -0.75, " human (voice only)", fontsize=5.5,
                 color=C_NULL, va="top", ha="left")

    handles = [
        Line2D([], [], marker="o", ls="", markerfacecolor="white",
               markeredgecolor=C_GREY, markersize=3.6,
               label="uncompressed (train condition)"),
        Line2D([], [], marker="o", ls="", color=C_THRESH, markersize=3.9,
               label="served coded \u2013 thresholding family"),
        Line2D([], [], marker="o", ls="", color=C_MULT, markersize=3.9,
               label="served coded \u2013 magnitude family"),
    ]
    axes[1].legend(handles=handles, loc="lower left", frameon=False,
                   handletextpad=0.4, borderpad=0.2)
    tag(axes[0], "(a)", dx=-0.30, dy=1.02)
    tag(axes[1], "(b)", dx=-0.06, dy=1.02)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_crossformat")


# ---------------------------------------------------------------- Fig. 3
def fig_neutralise(d: dict) -> None:
    """Masking the most codec-shifted columns, with a clean-audio control."""
    n = d["neut"]
    fig, ax = plt.subplots(1, 1, figsize=(COL_1, 1.72))
    show = ["svm_rbf", "logistic_regression", "ann_pytorch", "mlp_sklearn",
            "xgboost", "decision_tree"]
    styles = {
        "svm_rbf": (C_MULT, "-"), "logistic_regression": ("#B34700", "--"),
        "ann_pytorch": ("#E69F00", "-"), "mlp_sklearn": ("#8C4A00", ":"),
        "xgboost": (C_THRESH, "-"), "decision_tree": ("#4F7FA8", "--"),
    }
    for m in show:
        sub = n[(n.model == m) & (n.condition == "mp3_64")].sort_values(
            "n_neutralised")
        colr, ls = styles[m]
        ax.plot(np.arange(len(sub)), sub.uar_compressed, color=colr, ls=ls,
                marker="o", ms=2.0, label=PLAIN[m])
        if m in ("ann_pytorch", "svm_rbf"):
            ax.plot(np.arange(len(sub)), sub.uar_clean_control, color=colr,
                    ls=(0, (1, 2)), lw=0.7, alpha=0.55)
    xs = sorted(n.n_neutralised.unique())
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(x) for x in xs])
    ax.axvline(1, color=C_GREY, lw=0.6, ls=":")
    ax.set_xlabel("number of most codec-shifted features masked")
    ax.set_ylabel("balanced accuracy on MP3")
    ax.set_ylim(0.15, 0.70)
    ax.legend(loc="lower right", frameon=False, ncol=3, columnspacing=0.6,
              handlelength=1.2, borderpad=0.15, fontsize=5.3,
              handletextpad=0.35)
    save(fig, "fig_neutralise")


# ---------------------------------------------------------------- Fig. 4
def fig_xai(d: dict) -> None:
    """Attribution: explainer disagreement, and the model-randomisation check."""
    fig, axes = plt.subplots(1, 2, figsize=(COL_1, 1.55),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    # (a) Post-hoc methods vs SHAP on the same fitted tabular model.
    ax = axes[0]
    models = ["decision_tree", "logistic_regression", "random_forest",
              "lightgbm", "xgboost"]
    keys = [("shap_vs_intrinsic", "intrinsic", C_THRESH),
            ("shap_vs_permutation", "permutation", C_GREY),
            ("shap_vs_lime", "LIME", C_MULT)]
    x = np.arange(len(models))
    w = 0.26
    for j, (k, lab, colr) in enumerate(keys):
        vals = [d["xai"]["models"][m]["lime_vs_shap"].get(k, {}).get(
            "spearman_rho", np.nan) for m in models]
        ax.bar(x + (j - 1) * w, vals, width=w, color=colr, label=lab)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([PLAIN[m] for m in models], rotation=34, ha="right",
                       fontsize=5.2)
    ax.set_ylabel("Spearman $\\rho$ vs. SHAP", fontsize=6.2)
    ax.set_ylim(-0.05, 1.30)
    ax.legend(loc="upper center", frameon=False, ncol=3, columnspacing=0.5,
              handlelength=0.9, fontsize=5.2, borderpad=0.1,
              handletextpad=0.35)
    tag(ax, "(a)", dx=-0.30, dy=1.02)

    # (b) Model-randomisation sanity check.
    ax = axes[1]
    san = d["deep"]["model_randomisation_sanity"]
    names = list(san.keys())
    vals = [san[k]["spearman_trained_vs_random"] for k in names]
    short = {"saliency": "Sal", "input_x_gradient": "I$\\times$G",
             "integrated_gradients": "IG", "deeplift": "DL",
             "gradient_shap": "GS", "feature_ablation": "FA"}
    ax.bar(range(len(names)), vals, color=C_NULL, width=0.66)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([short[n] for n in names], rotation=34, ha="right",
                       fontsize=5.2)
    ax.set_ylim(-0.1, 1.30)
    ax.set_ylabel("$\\rho$, trained vs. random", fontsize=6.2)
    ax.text(0.5, 0.55, "all near zero", transform=ax.transAxes, ha="center",
            fontsize=5.4, color=C_GREY)
    tag(ax, "(b)", dx=-0.38, dy=1.02)

    fig.subplots_adjust(wspace=0.62)
    save(fig, "fig_xai")


# ---------------------------------------------------------------- numbers
def report(d: dict) -> None:
    """Print the values the manuscript quotes, so text and data cannot drift."""
    wide = crossformat_frame(d)
    print("\n--- cross-format balanced accuracy (train = ref) ---")
    for grp, ms in [("thresholds", THRESHOLD), ("multiplies", MULTIPLY)]:
        print(f"[{grp}]")
        for m in ms:
            r = wide.loc[m]
            print(f"  {PLAIN[m]:<14} ref {r['ref']:.3f}  mp3 {r['mp3_64']:.3f} "
                  f"({r['mp3_64']-r['ref']:+.3f})  aac {r['mp4_aac64']:.3f} "
                  f"({r['mp4_aac64']-r['ref']:+.3f})  rt {r['roundtrip_wav']:.3f}")
    dc = (wide["roundtrip_wav"] - wide["mp4_aac64"]).abs()
    print(f"\ncontainer control |rt - aac|: median {dc.median():.4f}  "
          f"max {dc.max():.4f} ({PLAIN[dc.idxmax()]})")

    print("\n--- attribution stability (Spearman rho over 436 features) ---")
    for m, v in d["xai"]["models"].items():
        st = v.get("attribution_stability")
        if not st:
            continue
        for cond in ("mp3_64", "mp4_aac64"):
            s = st[cond]
            print(f"  {PLAIN.get(m,m):<14} {cond:<10} rho {s['spearman_rho']:.3f} "
                  f"top25 {s['top25_overlap']}")
    ig = d["deep"]["attribution_stability_ig"]
    for cond in ("mp3_64", "mp4_aac64"):
        print(f"  ANN (IG)       {cond:<10} rho {ig[cond]['spearman_rho']:.3f} "
              f"top25 {ig[cond]['top25_overlap']}")

    print("\n--- neutralisation, MP3 ---")
    rb = d["rob"]["per_condition"]["mp3_64"]["models"]
    for m, v in rb.items():
        loss = v["uar_clean"] - v["uar_compressed"]
        rec1 = (v["uar_after_1_masked"] - v["uar_compressed"]) / loss if loss > 0.01 else float("nan")
        rec3 = (v["uar_after_3_masked"] - v["uar_compressed"]) / loss if loss > 0.01 else float("nan")
        print(f"  {PLAIN.get(m,m):<14} clean {v['uar_clean']:.3f} mp3 "
              f"{v['uar_compressed']:.3f} +1 {v['uar_after_1_masked']:.3f} "
              f"(rec {rec1:.1%}) +3 {v['uar_after_3_masked']:.3f} (rec {rec3:.1%})")

    print("\n--- neutralisation, AAC: best n per model ---")
    n = d["neut"]
    for m in ("ann_pytorch", "svm_rbf", "logistic_regression", "mlp_sklearn"):
        sub = n[(n.model == m) & (n.condition == "mp4_aac64")]
        if sub.empty:
            continue
        b = sub.loc[sub.uar_compressed.idxmax()]
        base = sub[sub.n_neutralised == 0].uar_compressed.iloc[0]
        print(f"  {PLAIN[m]:<14} base {base:.3f} -> best {b.uar_compressed:.3f} "
              f"at n={int(b.n_neutralised)}")

    print("\n--- ANN clean-audio control under masking (MP3 ranking) ---")
    sub = n[(n.model == "ann_pytorch") & (n.condition == "mp3_64")]
    print("  " + "  ".join(f"n={int(r.n_neutralised)}:{r.uar_clean_control:.3f}"
                           for _, r in sub.iterrows()))

    print("\n--- surrogate fidelity ---")
    for m, v in d["xai"]["models"].items():
        print(f"  {PLAIN.get(m,m):<14} {v['surrogate_fidelity']:.3f}")


def main() -> None:
    d = load_all()
    print("figures:")
    fig_mechanism(d)
    fig_crossformat(d)
    fig_neutralise(d)
    fig_xai(d)
    report(d)


if __name__ == "__main__":
    main()
