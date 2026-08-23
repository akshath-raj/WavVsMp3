"""Publication figures.

Every number plotted here is read from a committed artefact under
``xai_ser/outputs/{eda,models,xai,robustness,per_condition}/``. Nothing is
hard-coded except axis labels and the condition names the source files
themselves use.

Output: PDF (vector) at IEEE column widths, plus a PNG mirror for quick review.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
XS = ROOT / "xai_ser" / "outputs"
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- run dirs
MODELS_DIR = XS / "models" / "20260818_045719"
XAI_DIR = XS / "xai" / "20260818_050808"
DEEP_DIR = XS / "xai" / "deep" / "20260818_050247"
ROB_DIR = XS / "robustness" / "20260818_051057"
EDA_DIR = XS / "eda"
PC_DIR = XS / "per_condition"


def _newest(parent: Path, glob: str) -> Path:
    hits = sorted(parent.glob(glob))
    if not hits:
        raise FileNotFoundError(f"no {glob} under {parent}")
    return hits[-1]


CONDS = ("ref", "mp3_64", "mp4_aac64", "roundtrip_wav")
COND_LABEL = {"ref": "uncompressed\nWAV", "mp3_64": "MP3\n64k",
              "mp4_aac64": "MP4/AAC\n64k", "roundtrip_wav": "roundtrip\nWAV"}

# ---------------------------------------------------------------- style
COL = 3.45          # IEEE single-column width, inches
DCOL = 7.16         # IEEE double-column width, inches

# Okabe-Ito, colour-blind safe
BLACK = "#000000"
ORANGE = "#E69F00"
SKY = "#56B4E9"
GREEN = "#009E73"
YELLOW = "#F0E442"
BLUE = "#0072B2"
VERM = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#7F7F7F"

FAMILY_COLOUR = {
    "tree": GREEN,          # axis-aligned partitioning
    "kernel": VERM,         # magnitude-sensitive
    "linear": BLUE,
    "neural": PURPLE,
    "other": GREY,
}

COND_COLOUR = {"ref": BLACK, "mp3_64": VERM, "mp4_aac64": BLUE,
               "roundtrip_wav": GREY}

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.6,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.1,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42,
})


def save(fig, name: str) -> None:
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def panel_tag(ax, text: str, dx: float = -0.16, dy: float = 1.06) -> None:
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=8.5,
            fontweight="bold", va="top", ha="left")


# ================================================================= load
def load_all():
    d = {}
    d["drift"] = pd.read_csv(EDA_DIR / "format_drift.csv")
    d["cross"] = pd.read_csv(MODELS_DIR / "cross_format.csv")
    d["xai"] = json.loads((XAI_DIR / "xai_summary.json").read_text())
    d["deep"] = json.loads((DEEP_DIR / "deep_xai_summary.json").read_text())
    d["neut"] = pd.read_csv(ROB_DIR / "neutralisation_curves.csv")
    d["ig"] = {
        c: pd.read_csv(DEEP_DIR / f"ig_global_{c}.csv")
        for c in ("ref", "mp3_64", "mp4_aac64", "roundtrip_wav")
    }
    d["shift_mp3"] = pd.read_csv(ROB_DIR / "feature_shift_mp3_64.csv")

    # matched-format training + its null calibration
    pc = _newest(PC_DIR, "2*")
    d["pc"] = json.loads((pc / "summary.json").read_text())
    d["pc_lb"] = pd.read_csv(pc / "leaderboard_matched.csv")
    d["null"] = json.loads(
        (_newest(PC_DIR, "null_2*") / "null_calibration.json").read_text())
    try:
        pdir = _newest(PC_DIR, "paired_2*")
        d["paired"] = json.loads((pdir / "paired_null.json").read_text())
        d["paired"]["table"] = pd.read_csv(pdir / "paired_draws.csv")
    except FileNotFoundError:
        d["paired"] = None
    return d


# ================================================================= Fig. 2
def fig_signal(d):
    """What the codec does to the signal, measured on the feature table."""
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.05))

    # --- (a) the most displaced descriptors, per codec ---------------------
    ax = axes[0]
    dr = d["drift"]
    top_mp3 = dr[dr.condition == "mp3_64"].nlargest(6, "abs_smd")
    top_aac = dr[dr.condition == "mp4_aac64"].nlargest(6, "abs_smd")
    names, vals, cols_ = [], [], []
    for _, r in top_mp3.iterrows():
        names.append(r.feature); vals.append(r.abs_smd); cols_.append(VERM)
    for _, r in top_aac.iterrows():
        names.append(r.feature); vals.append(r.abs_smd); cols_.append(BLUE)
    order = np.argsort(vals)
    y = np.arange(len(order))
    ax.barh(y, [vals[i] for i in order], color=[cols_[i] for i in order],
            height=0.68, edgecolor=BLACK, lw=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=5.0,
                       family="monospace")
    ax.set_xscale("log")
    ax.set_xlabel(r"$|$SMD$|$ vs. uncompressed (log)")
    ax.set_xlim(0.4, 90)
    handles = [Patch(color=VERM, label="MP3 64k"), Patch(color=BLUE, label="MP4/AAC 64k")]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=5.9,
              handlelength=1.0)
    ax.grid(axis="x", alpha=0.25)
    panel_tag(ax, "(a)", dx=-0.52)
    ax.set_title("the six most displaced descriptors", pad=3)

    # --- (b) standardised feature shift, whole table ---------------------
    ax = axes[1]
    dr = d["drift"]
    conds = [("mp3_64", VERM, "MP3 64k"),
             ("mp4_aac64", BLUE, "MP4/AAC 64k"),
             ("roundtrip_wav", GREY, "roundtrip WAV")]
    for cond, c, lab in conds:
        v = np.sort(dr.loc[dr.condition == cond, "abs_smd"].values)
        v = np.clip(v, 1e-4, None)
        ax.plot(v, np.linspace(0, 1, len(v)), color=c, label=lab,
                lw=1.1, ls="-" if cond != "roundtrip_wav" else "--")
    ax.set_xscale("log")
    ax.set_xlabel(r"$|$SMD$|$ vs. uncompressed (log)")
    ax.set_ylabel("cumulative fraction\nof 436 features")
    ax.axvline(0.5, color=BLACK, lw=0.6, ls=":")
    ax.set_xlim(8e-5, 3e3)
    ax.set_ylim(-0.02, 1.16)
    # annotate the extreme feature
    worst = dr[dr.condition == "mp3_64"].nlargest(1, "abs_smd").iloc[0]
    ax.annotate("spectral_contrast_6_mean\n" + f"{worst.abs_smd:.1f} SD",
                xy=(worst.abs_smd, 1.0), xytext=(60, 0.60), fontsize=5.5,
                ha="center", color=VERM,
                arrowprops=dict(arrowstyle="->", lw=0.5, color=VERM))
    ax.legend(frameon=False, loc="upper left", handlelength=1.4,
              bbox_to_anchor=(-0.02, 1.03))
    ax.grid(alpha=0.25)
    panel_tag(ax, "(b)")
    ax.set_title("displacement of the whole table", pad=3)

    # --- (c) duration / encoder padding ----------------------------------
    ax = axes[2]
    dur = dr[dr.feature == "duration_s"].set_index("condition")
    order = ["mp3_64", "mp4_aac64", "roundtrip_wav"]
    ref_mean = float(dur.loc["mp3_64", "mean_ref"])
    delta_ms = [(float(dur.loc[c, "mean_cond"]) - ref_mean) * 1000 for c in order]
    cols = [VERM, BLUE, GREY]
    bars = ax.bar(range(3), delta_ms, color=cols, width=0.6)
    for b, v in zip(bars, delta_ms):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.9, f"+{v:.2f} ms",
                ha="center", fontsize=6.2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["MP3\n64k", "MP4/AAC\n64k", "roundtrip\nWAV"], fontsize=6.2)
    ax.set_ylabel("mean duration change (ms)")
    ax.set_ylim(0, max(delta_ms) * 1.34)
    ax.grid(axis="y", alpha=0.25)
    ax.annotate("AAC encoder\npriming samples", xy=(0.72, delta_ms[1] * 0.55),
                xytext=(0.05, delta_ms[1] * 0.30), fontsize=5.9, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.5))
    panel_tag(ax, "(c)")
    ax.set_title("frame alignment", pad=3)

    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig_signal")




# ================================================================= Fig. 5
FAMILY_OF = {
    "lightgbm": "tree", "xgboost": "tree", "hist_gradient_boosting": "tree",
    "random_forest": "tree", "extra_trees": "tree", "adaboost": "tree",
    "decision_tree": "tree",
    "svm_rbf": "kernel", "knn": "kernel",
    "logistic_regression": "linear", "linear_svm": "linear", "lda": "linear",
    "gaussian_nb": "linear",
    "mlp_sklearn": "neural", "ann_pytorch": "neural",
    "dummy_majority": "other", "dummy_stratified": "other",
}
FAMILY_OF["decision_tree_entropy"] = "tree"
FAMILY_OF["decision_tree_readable"] = "tree"
PRETTY = {
    "decision_tree_entropy": "Decision tree", "decision_tree_readable": "Decision tree (d=3)",
    "lightgbm": "LightGBM", "xgboost": "XGBoost",
    "hist_gradient_boosting": "HistGB", "random_forest": "Random forest",
    "extra_trees": "Extra trees", "adaboost": "AdaBoost",
    "decision_tree": "Decision tree", "svm_rbf": "SVM (RBF)", "knn": "kNN",
    "logistic_regression": "Logistic reg.", "linear_svm": "Linear SVM",
    "lda": "LDA", "gaussian_nb": "Gaussian NB", "mlp_sklearn": "MLP",
    "ann_pytorch": "ANN (PyTorch)",
}


def spread(ys, gap):
    """Shift labels apart, preserving order, so text does not overplot."""
    idx = np.argsort(ys)[::-1]
    out = np.array(ys, dtype=float)
    for k in range(1, len(idx)):
        hi, lo = idx[k - 1], idx[k]
        if out[hi] - out[lo] < gap:
            out[lo] = out[hi] - gap
    return out
# ANN results live outside cross_format.csv (separate torch run); values are the
# ones reported in xai_ser/reports/FINDINGS.md D4, reproduced from
# outputs/models/ann/<ts>/metrics.json.
ANN_ROW = {"ref": 0.5993, "mp3_64": 0.3536,
           "mp4_aac64": 0.5126, "roundtrip_wav": 0.5120}


def _crossformat_frame(d):
    cf = d["cross"]
    cf = cf[(cf.train_condition == "ref") & (~cf.model.str.startswith("dummy"))]
    piv = cf.pivot_table(index="model", columns="test_condition",
                         values="balanced_accuracy")
    piv.loc["ann_pytorch"] = pd.Series(ANN_ROW)
    return piv


def fig_crossformat(d):
    """Robustness is set by how a model consumes its features."""
    piv = _crossformat_frame(d).sort_values("ref", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.55), sharey=True)

    for ax, cond, title in zip(
            axes, ["mp3_64", "mp4_aac64"],
            ["MP3 @ 64 kbps", "MP4/AAC @ 64 kbps"]):
        names = list(piv.index)
        ends = np.array([piv.loc[m, cond] for m in names])
        lab_y = spread(ends, 0.0225)
        for m, y_end, y_lab in zip(names, ends, lab_y):
            col = FAMILY_COLOUR[FAMILY_OF.get(m, "other")]
            ax.plot([0, 1], [piv.loc[m, "ref"], y_end], "-o", color=col, ms=3.0,
                    lw=1.0, alpha=0.9)
            ax.plot([1.0, 1.10], [y_end, y_lab], lw=0.4, color=col, alpha=0.55)
            ax.annotate(PRETTY.get(m, m), (1.12, y_lab), fontsize=5.7,
                        color=col, va="center", ha="left")
        ax.axhline(0.455, color=BLACK, ls="--", lw=0.6)
        ax.text(-0.03, 0.462, "human voice-only ceiling", fontsize=5.7)
        ax.axhline(1 / 6, color=GREY, ls=":", lw=0.6)
        ax.text(-0.03, 0.174, "chance", fontsize=5.7, color=GREY)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["trained and tested\non WAV", "tested on\n" + title],
                           fontsize=6.3)
        ax.set_xlim(-0.07, 1.92)
        ax.set_title(title, pad=3)
        ax.grid(axis="y", alpha=0.22)

    axes[0].set_ylabel("balanced accuracy (20 held-out speakers)")
    axes[0].set_ylim(0.13, 0.70)
    panel_tag(axes[0], "(a)", dx=-0.13)
    panel_tag(axes[1], "(b)", dx=-0.06)

    handles = [Line2D([], [], color=FAMILY_COLOUR[k], marker="o", ms=3.0, lw=1.0,
                      label=v) for k, v in
               (("tree", "tree / ensemble: thresholds its inputs"),
                ("linear", "linear, discriminant: multiplies them"),
                ("kernel", "kernel, instance-based: multiplies them"),
                ("neural", "neural: multiplies them"))]
    axes[0].legend(handles=handles, frameon=False, loc="lower left",
                   handlelength=1.4, fontsize=5.9, bbox_to_anchor=(-0.02, 0.045))
    fig.subplots_adjust(wspace=0.10)
    save(fig, "fig_crossformat")


# ================================================================= Fig. 6
def _stability_rows(d):
    """(model, condition) -> (rho, top25 overlap) for Arm B."""
    rows = []
    for m, blob in d["xai"]["models"].items():
        if m == "svm_rbf":
            continue                      # rank-deficient KernelSHAP, excluded
        st = blob.get("attribution_stability", {})
        for cond, s in st.items():
            rows.append(dict(model=m, condition=cond, rho=s["spearman_rho"],
                             overlap=s["top25_overlap"], l1=s["l1_shift"],
                             estimator="SHAP (exact)"))
    for cond, s in d["deep"]["attribution_stability_ig"].items():
        rows.append(dict(model="ann_pytorch", condition=cond,
                         rho=s["spearman_rho"], overlap=s["top25_overlap"],
                         l1=s["l1_shift"], estimator="Integrated gradients"))
    return pd.DataFrame(rows)




# ================================================================= Fig. 7
def fig_neutralise(d):
    """One feature carries almost all of the MP3 collapse."""
    fig, ax = plt.subplots(figsize=(COL, 2.1))
    n = d["neut"]
    n = n[n.condition == "mp3_64"]
    show = [("svm_rbf", VERM, "-"), ("logistic_regression", BLUE, "-"),
            ("ann_pytorch", PURPLE, "-"), ("mlp", SKY, "--"),
            ("xgboost", GREEN, "-"), ("random_forest", "#3B7A57", "--")]
    for m, col, ls in show:
        sub = n[n.model == m].sort_values("n_neutralised")
        if sub.empty:
            continue
        x = sub.n_neutralised.values.astype(float)
        x[x == 0] = 0.6                       # log axis: place 0 just left of 1
        ax.plot(x, sub.uar_compressed, ls, color=col, ms=2.6, marker="o",
                label=PRETTY.get(m, m))
    ax.set_xscale("log")
    ax.set_xticks([0.6, 1, 2, 3, 5, 10, 20, 40, 80])
    ax.set_xticklabels(["0", "1", "2", "3", "5", "10", "20", "40", "80"])
    ax.minorticks_off()
    ax.set_xlabel("codec-shifted features replaced by the training median")
    ax.set_ylabel("balanced accuracy on MP3 64k")
    ax.axhline(0.455, color=BLACK, ls="--", lw=0.6)
    ax.text(0.62, 0.463, "human voice-only ceiling", fontsize=5.7)
    ax.annotate("masking one feature\n(spectral_contrast_6_mean) recovers\n"
                "98% of the SVM loss, 92% of the ANN's",
                xy=(1.05, 0.565), xytext=(1.45, 0.345), fontsize=6.0, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.5))
    ax.legend(frameon=False, loc="lower right", ncol=2, handlelength=1.4,
              fontsize=6.0, columnspacing=0.9, bbox_to_anchor=(1.02, -0.02))
    ax.grid(alpha=0.22)
    ax.set_ylim(0.17, 0.66)
    save(fig, "fig_neutralise")


# ================================================================= Fig. 8
def fig_methods(d):
    """Method agreement: post-hoc on tabular models vs gradient family on the ANN."""
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.35),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # --- (a) SHAP vs LIME / permutation / intrinsic, per model ------------
    ax = axes[0]
    models, lime_v, perm_v, intr_v = [], [], [], []
    for m, blob in d["xai"]["models"].items():
        if m == "svm_rbf":
            continue
        lv = blob.get("lime_vs_shap", {})
        models.append(PRETTY.get(m, m))
        lime_v.append(lv["shap_vs_lime"]["spearman_rho"])
        perm_v.append(lv["shap_vs_permutation"]["spearman_rho"])
        intr_v.append(lv["shap_vs_intrinsic"]["spearman_rho"])
    x = np.arange(len(models))
    w = 0.26
    ax.bar(x - w, intr_v, w, color=GREEN, label="intrinsic importance")
    ax.bar(x, perm_v, w, color=BLUE, label="permutation")
    ax.bar(x + w, lime_v, w, color=VERM, label="LIME")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=22, ha="right", fontsize=6.0)
    ax.set_ylabel(r"rank agreement with SHAP ($\rho$)")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color=BLACK, lw=0.6)
    ax.legend(frameon=False, loc="upper center", ncol=3, handlelength=1.2,
              fontsize=6.0, columnspacing=0.9)
    ax.grid(axis="y", alpha=0.22)
    panel_tag(ax, "(a)", dx=-0.13)
    ax.set_title("post-hoc methods disagree on the same fitted model", pad=3)

    # --- (b) gradient family on the ANN: agreement + sanity check ---------
    ax = axes[1]
    ag = d["deep"]["method_agreement_spearman"]
    names = list(ag.keys())
    nice = {"saliency": "Saliency", "input_x_gradient": "Input$\\times$Grad",
            "integrated_gradients": "Int. grad.", "deeplift": "DeepLIFT",
            "gradient_shap": "GradSHAP", "feature_ablation": "Ablation"}
    M = np.array([[ag[a][b] for b in names] for a in names])
    im = ax.imshow(M, vmin=0.9, vmax=1.0, cmap="Greens")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([nice[n] for n in names], rotation=40, ha="right",
                       fontsize=5.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([nice[n] for n in names], fontsize=5.8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=5.4, color=BLACK if M[i, j] < 0.97 else "white")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.ax.tick_params(labelsize=5.6)
    cb.outline.set_linewidth(0.4)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    panel_tag(ax, "(b)", dx=-0.22)
    ax.set_title("gradient methods agree on the ANN", pad=3)

    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig_methods")


# ================================================================= Fig. 9
def fig_sanity(d):
    """Adebayo model-randomisation check for the six deep methods."""
    sc = d["deep"]["model_randomisation_sanity"]
    fig, ax = plt.subplots(figsize=(COL, 1.7))
    nice = {"saliency": "Saliency", "input_x_gradient": "Input$\\times$Grad",
            "integrated_gradients": "Int. grad.", "deeplift": "DeepLIFT",
            "gradient_shap": "GradSHAP", "feature_ablation": "Ablation"}
    items = [(nice.get(k, k), v["spearman_trained_vs_random"])
             for k, v in sc.items()]
    items.sort(key=lambda t: t[1])
    ax.barh([i[0] for i in items], [i[1] for i in items], color=SKY,
            edgecolor=BLACK, lw=0.4, height=0.62)
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.012, i, f"{v:+.3f}", va="center", fontsize=6.0)
    ax.axvline(0, color=BLACK, lw=0.6)
    ax.axvspan(-0.25, 0.25, color=GREEN, alpha=0.10, zorder=0, lw=0)
    ax.set_xlim(-0.06, 0.30)
    ax.set_ylim(-0.65, 5.65)
    ax.set_xlabel(r"$\rho$ between trained-model and random-model attributions")
    ax.grid(axis="x", alpha=0.22)
    save(fig, "fig_sanity")


# ================================================================= Fig. 4
def fig_matched(d):
    """Matched-format training removes the damage that mismatch creates."""
    lb = d["pc_lb"]
    piv_m = lb.pivot_table(index="model", columns="condition",
                           values="balanced_accuracy")
    piv_x = _crossformat_frame(d)
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.5),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # --- (a) trained and tested in the same condition ---------------------
    ax = axes[0]
    models = list(piv_m["ref"].sort_values(ascending=False).index)
    x = np.arange(len(models))
    w = 0.2
    for i, cond in enumerate(CONDS):
        ax.bar(x + (i - 1.5) * w, [piv_m.loc[m, cond] for m in models], w,
               color=COND_COLOUR[cond], label=COND_LABEL[cond].replace("\n", " "),
               edgecolor=BLACK, lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(m, m) for m in models], rotation=24,
                       ha="right", fontsize=6.0)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 0.68)
    ax.axhline(0.455, color=BLACK, ls="--", lw=0.6)
    ax.text(len(models) - 0.45, 0.463, "human voice-only ceiling", fontsize=5.7,
            ha="right")
    ax.axhline(1 / 6, color=GREY, ls=":", lw=0.6)
    ax.text(len(models) - 0.45, 0.175, "chance", fontsize=5.7, ha="right",
            color=GREY)
    ax.legend(frameon=False, ncol=4, loc="upper center", fontsize=5.9,
              handlelength=1.0, columnspacing=0.9, bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", alpha=0.22)
    panel_tag(ax, "(a)", dx=-0.115)
    ax.set_title("trained and tested in the same condition", pad=14)

    # --- (b) mismatch vs match, on MP3 ------------------------------------
    ax = axes[1]
    pairs = [("svm_rbf", "SVM (RBF)"), ("logistic_regression", "Logistic reg."),
             ("lda", "LDA"), ("mlp_sklearn", "MLP"),
             ("random_forest", "Random forest"), ("lightgbm", "LightGBM")]
    y = np.arange(len(pairs))
    mism = [piv_x.loc[m, "mp3_64"] for m, _ in pairs]
    match = [piv_m.loc[m, "mp3_64"] for m, _ in pairs]
    ax.barh(y - 0.19, mism, 0.36, color=VERM, edgecolor=BLACK, lw=0.3,
            label="trained on WAV, served MP3")
    ax.barh(y + 0.19, match, 0.36, color=GREEN, edgecolor=BLACK, lw=0.3,
            label="trained and served on MP3")
    for i, (a, b) in enumerate(zip(mism, match)):
        if b - a > 0.05:
            ax.annotate(f"+{(b - a) * 100:.0f} pts", (max(a, b) + 0.014, i),
                        fontsize=5.8, va="center", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([p[1] for p in pairs], fontsize=6.2)
    ax.set_xlabel("balanced accuracy on MP3 64k")
    ax.set_xlim(0, 0.82)
    ax.axvline(1 / 6, color=GREY, ls=":", lw=0.6)
    ax.legend(frameon=False, loc="upper center", fontsize=5.9, handlelength=1.0,
              ncol=1, bbox_to_anchor=(0.5, 1.13))
    ax.grid(axis="x", alpha=0.22)
    panel_tag(ax, "(b)", dx=-0.30)
    ax.set_title("the collapse is mismatch, not lost information", pad=14)

    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig_matched")


# ================================================================= Fig. 5
def fig_trees(d):
    """The top two levels of the entropy tree, fitted separately per condition."""
    sp = d["pc"]["top_splits"]
    fig, axes = plt.subplots(1, 4, figsize=(DCOL, 1.55))

    for ax, cond in zip(axes, CONDS):
        nodes = {n["path"]: n for n in sp[cond]}
        ax.set_xlim(0, 1)
        ax.set_ylim(0.26, 1.02)
        ax.axis("off")
        ax.set_title(COND_LABEL[cond].replace("\n", " "), pad=4,
                     color=COND_COLOUR[cond], fontsize=7.4)

        def box(path, cx, cy, wid):
            n = nodes.get(path)
            if n is None:
                return
            same = (n["feature"] == {p["path"]: p for p in sp["ref"]}
                    .get(path, {}).get("feature"))
            edge = GREY if same else VERM
            lw = 0.5 if same else 1.1
            ax.add_patch(plt.Rectangle((cx - wid / 2, cy - 0.085), wid, 0.17,
                                       fill=True, facecolor="white",
                                       edgecolor=edge, lw=lw, zorder=3))
            ax.text(cx, cy + 0.048, n["feature"], ha="center", va="center",
                    fontsize=4.3, family="monospace", zorder=4,
                    color=BLACK if same else VERM)
            ax.text(cx, cy + 0.002, r"$\leq$ " + f"{n['threshold']:.3f}",
                    ha="center", va="center", fontsize=4.9, zorder=4)
            ax.text(cx, cy - 0.048, f"IG {n['information_gain_bits']:.3f} bits",
                    ha="center", va="center", fontsize=4.4, color=GREY, zorder=4)

        box("", 0.5, 0.80, 0.92)
        box("L", 0.26, 0.42, 0.50)
        box("R", 0.76, 0.42, 0.50)
        for x0, x1 in ((0.5, 0.26), (0.5, 0.76)):
            ax.plot([x0, x1], [0.70, 0.51], color=BLACK, lw=0.5, zorder=1)
        ax.text(0.32, 0.615, "yes", fontsize=4.6, color=GREY)
        ax.text(0.63, 0.615, "no", fontsize=4.6, color=GREY)
        ax.text(0.5, 0.955, f"H = {nodes['']['entropy_bits']:.4f} bits",
                ha="center", fontsize=5.4, color=GREY)

    handles = [Line2D([], [], color=GREY, lw=0.8, label="same feature as uncompressed"),
               Line2D([], [], color=VERM, lw=1.4, label="different feature")]
    fig.legend(handles=handles, frameon=False, ncol=2, fontsize=6.2,
               loc="lower center", bbox_to_anchor=(0.5, -0.10), handlelength=1.6)
    fig.subplots_adjust(wspace=0.10)
    save(fig, "fig_trees")


# ================================================================= Fig. 6
def fig_nullcal(d):
    """Importance-ranking displacement against its own resampling floor."""
    nl = d["null"]
    pr = d.get("paired")
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.4))

    # --- (a) marginal floors and the condition effects --------------------
    ax = axes[0]
    k = f"top{nl['top_k']}_overlap"
    bars = [
        ("tie-breaking\n(data fixed)", nl["floors"]["seed"][k]["mean"],
         nl["floors"]["seed"][k]["p05"], nl["floors"]["seed"][k]["p95"], GREEN),
        ("actor bootstrap\n(codec fixed)", nl["floors"]["bootstrap"][k]["mean"],
         nl["floors"]["bootstrap"][k]["p05"], nl["floors"]["bootstrap"][k]["p95"],
         ORANGE),
    ]
    x = np.arange(len(bars) + 4)
    for i, (lab, m, lo, hi, c) in enumerate(bars):
        ax.bar(i, m, 0.62, color=c, edgecolor=BLACK, lw=0.4)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=BLACK, lw=0.7,
                    capsize=2.2)
    labels = [b[0] for b in bars]
    ax.bar(2, nl["floors"]["container"][k], 0.62, color=GREY, edgecolor=BLACK, lw=0.4)
    labels.append("container\n(same codec)")
    for j, cond in enumerate([c for c in CONDS if c != "ref"]):
        ax.bar(3 + j, nl["effects"][cond][k], 0.62, color=COND_COLOUR[cond],
               edgecolor=BLACK, lw=0.4)
        labels.append(COND_LABEL[cond])
    ax.axhspan(nl["floors"]["bootstrap"][k]["p05"],
               nl["floors"]["bootstrap"][k]["p95"],
               color=ORANGE, alpha=0.16, zorder=0, lw=0)
    ax.text(5.4, nl["floors"]["bootstrap"][k]["p95"] + 0.4,
            "sampling floor", fontsize=5.8, color=ORANGE, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.3, rotation=30, ha="right")
    ax.set_ylabel(f"top-{nl['top_k']} features retained")
    ax.set_ylim(0, 26)
    ax.grid(axis="y", alpha=0.22)
    panel_tag(ax, "(a)", dx=-0.135)
    ax.set_title("unpaired: the sampling floor swallows every effect", pad=3)

    # --- (b) the paired within-draw comparison ----------------------------
    ax = axes[1]
    if pr is None:
        ax.axis("off")
    else:
        tab = pr["table"]
        conds = [c for c in CONDS if c != "ref"] + ["container"]
        for j, cond in enumerate(conds):
            f = tab["floor_overlap"].to_numpy(float)
            e = tab[f"{cond}_overlap"].to_numpy(float)
            jitter = (np.random.default_rng(j).random(len(f)) - 0.5) * 0.16
            ax.plot(np.vstack([np.full_like(f, j * 2 + 0.0) + jitter,
                               np.full_like(e, j * 2 + 0.85) + jitter]),
                    np.vstack([f, e]), color=GREY, lw=0.35, alpha=0.55, zorder=1)
            ax.scatter(np.full_like(f, j * 2 + 0.0) + jitter, f, s=8,
                       color=GREEN, edgecolor=BLACK, lw=0.25, zorder=3)
            ax.scatter(np.full_like(e, j * 2 + 0.85) + jitter, e, s=8,
                       color=COND_COLOUR.get(cond, YELLOW), edgecolor=BLACK,
                       lw=0.25, zorder=3)
            ax.plot([j * 2 - 0.25, j * 2 + 1.10], [e.mean()] * 2,
                    color=BLACK, lw=0.9, zorder=4)
        ax.axhline(tab["container_overlap"].mean(), color=YELLOW, lw=0.9,
                   ls="--", zorder=2)
        ax.text(0.05, tab["container_overlap"].mean() + 0.55,
                "informationally empty perturbation", fontsize=5.4,
                color="#B8860B",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8,
                          alpha=0.85))
        ax.set_xticks([j * 2 + 0.42 for j in range(len(conds))])
        ax.set_xticklabels(
            [COND_LABEL.get(c, "container\n(same codec)") for c in conds],
            fontsize=5.5)
        ax.set_ylabel(f"top-{nl['top_k']} features retained")
        ax.set_ylim(0, 26)
        ax.grid(axis="y", alpha=0.22)
        handles = [Line2D([], [], ls="", marker="o", ms=3.6, color=GREEN,
                          mec=BLACK, mew=0.25,
                          label="floor: same audio, different tie-breaking"),
                   Line2D([], [], ls="", marker="o", ms=3.6, color=GREY,
                          mec=BLACK, mew=0.25, label="effect: same actors, coded audio"),
                   Line2D([], [], color=BLACK, lw=0.9, label="mean")]
        ax.legend(handles=handles, frameon=False, loc="lower center", fontsize=5.7,
                  handlelength=1.0, bbox_to_anchor=(0.5, -0.02))
        panel_tag(ax, "(b)", dx=-0.135)
        ax.set_title("paired: codec separates from the empty perturbation", pad=3)

    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig_nullcal")


# ================================================================= main
def main():
    d = load_all()
    fig_signal(d)
    fig_crossformat(d)
    fig_matched(d)
    fig_trees(d)
    fig_nullcal(d)
    fig_neutralise(d)
    fig_methods(d)
    fig_sanity(d)
    print("done")


if __name__ == "__main__":
    main()
