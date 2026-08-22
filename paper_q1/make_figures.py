"""Publication figures for the combined manuscript.

Every number plotted here is read from a committed artefact of one of the two
study arms:

  Arm A (LALM)   exp/out/*.json, exp/out/*.parquet
  Arm B (feature) xai_ser/outputs/{eda,models,xai,robustness}/...

Nothing is hard-coded except axis labels and the band/condition names that the
source files themselves use.

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
EXP = ROOT / "exp" / "out"
XS = ROOT / "xai_ser" / "outputs"
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- run dirs
MODELS_DIR = XS / "models" / "20260818_045719"
XAI_DIR = XS / "xai" / "20260818_050808"
DEEP_DIR = XS / "xai" / "deep" / "20260818_050247"
ROB_DIR = XS / "robustness" / "20260818_051057"
EDA_DIR = XS / "eda"

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
    "tree": GREEN,          # threshold-based
    "kernel": VERM,         # magnitude-based
    "linear": BLUE,
    "neural": PURPLE,
    "other": GREY,
}

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
    d["band"] = json.loads((EXP / "band_damage.json").read_text())
    d["fid"] = json.loads((EXP / "fidelity_summary.json").read_text())
    d["arousal"] = json.loads((EXP / "arousal_results.json").read_text())
    d["auc"] = json.loads((EXP / "arousal_auc.json").read_text())
    d["grid"] = pd.read_parquet(EXP / "arousal_grid.parquet")

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
    return d


# ================================================================= Fig. 2
def fig_signal(d):
    """What the codec does to the signal, from both arms' instrumentation."""
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.05))

    # --- (a) Arm A: band-wise noise-to-signal ratio -----------------------
    ax = axes[0]
    nsr = d["band"]["nsr_db"]
    bands = list(nsr["64"].keys())
    labels = [b.replace("Hz", "").replace("–", "-") for b in bands]
    x = np.arange(len(bands))
    for br, c, m in (("32", VERM, "o"), ("64", ORANGE, "s"), ("128", BLUE, "^")):
        ax.plot(x, [nsr[br][b] for b in bands], marker=m, ms=3.2, color=c,
                label=f"{br} kbps")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=5.8)
    ax.set_ylabel("band NSR (dB)")
    ax.set_xlabel("frequency band (Hz)")
    ax.legend(frameon=False, loc="upper left", handlelength=1.3,
              bbox_to_anchor=(-0.02, 1.04), labelspacing=0.3)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(-29, -2.0)
    ax.axvspan(4.5, 5.5, color=YELLOW, alpha=0.30, zorder=0, lw=0)
    ax.annotate("top band is\nprotected least", xy=(4.80, -14.6),
                xytext=(2.35, -21.5), fontsize=5.9, ha="center", color=BLACK,
                arrowprops=dict(arrowstyle="->", lw=0.5, color=BLACK))
    panel_tag(ax, "(a)")
    ax.set_title("Arm A: spectral damage", pad=3)

    # --- (b) Arm B: standardised feature shift ---------------------------
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
    ax.set_title("Arm B: feature shift", pad=3)

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
    ax.set_title("Arm B: frame alignment", pad=3)

    fig.subplots_adjust(wspace=0.42)
    save(fig, "fig_signal")


# ================================================================= Fig. 3
def fig_armA(d):
    """Sensitivity invariant, criterion displaced, attribution reordered."""
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.15))
    auc = d["auc"]["auc_by_condition"]

    # --- (a) discriminability with bootstrap CI ---------------------------
    ax = axes[0]
    order = ["ref", "rt_mp3_32", "rt_mp3_64", "rt_mp3_128",
             "mp3_32", "mp3_64", "mp3_128"]
    nice = ["WAV\nref", "rt\n32k", "rt\n64k", "rt\n128k",
            "MP3\n32k", "MP3\n64k", "MP3\n128k"]
    y = np.arange(len(order))[::-1]
    for i, c in enumerate(order):
        a = auc[c]
        col = BLACK if c == "ref" else (GREY if c.startswith("rt") else VERM)
        ax.plot([a["lo"], a["hi"]], [y[i], y[i]], color=col, lw=1.3,
                solid_capstyle="butt", alpha=0.75)
        ax.plot(a["auc"], y[i], "o", ms=3.4, color=col)
    ax.axvline(auc["ref"]["auc"], color=BLACK, lw=0.6, ls="--", zorder=0)
    ax.axvline(0.5, color=GREY, lw=0.6, ls=":", zorder=0)
    ax.text(0.508, -0.42, "chance", fontsize=5.9, ha="left", color=GREY)
    ax.set_ylim(-0.85, len(order) - 0.35)
    ax.set_yticks(y)
    ax.set_yticklabels(nice, fontsize=5.8)
    ax.set_xlabel("AUC (fixed-label posterior)")
    ax.set_xlim(0.44, 0.94)
    ax.grid(axis="x", alpha=0.25)
    panel_tag(ax, "(a)", dx=-0.30)
    ax.set_title("sensitivity: invariant", pad=3)

    # --- (b) criterion shift: class-conditional translation ---------------
    ax = axes[1]
    for lab, key, col in (("high arousal", "meanP_high", VERM),
                          ("low arousal", "meanP_low", BLUE)):
        vals = [auc["ref"][key], auc["mp3_64"][key]]
        ax.plot([0, 1], vals, "-o", color=col, ms=4, label=lab)
        for xx, vv in zip((0, 1), vals):
            ax.annotate(f"{vv:.3f}", (xx, vv), textcoords="offset points",
                        xytext=(0, 6 if col == VERM else -11), ha="center",
                        fontsize=6.0, color=col)
    mp = [auc["ref"]["meanP"], auc["mp3_64"]["meanP"]]
    ax.plot([0, 1], mp, "--s", color=BLACK, ms=3.2, label="marginal")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["WAV ref", "MP3 64k"])
    ax.set_xlim(-0.32, 1.32)
    ax.set_ylim(-0.02, 0.47)
    ax.set_ylabel(r"mean $P(\mathrm{high})$")
    ax.legend(frameon=False, loc="upper right", handlelength=1.4,
              bbox_to_anchor=(1.04, 1.06))
    ax.grid(axis="y", alpha=0.25)
    ax.annotate("both class-conditional means\ntranslate together",
                xy=(0.5, 0.105), fontsize=5.8, ha="center", style="italic")
    panel_tag(ax, "(b)")
    ax.set_title("criterion: displaced", pad=3)

    # --- (c) attribution similarity against the measured null -------------
    ax = axes[2]
    ms = d["arousal"]["map_similarity"]
    names = ["1-LSB dither\n(null floor)", "container", "codec", "total"]
    keys = ["rho_dither (NULL)", "rho_container", "rho_codec", "rho_total"]
    cols = [YELLOW, GREY, VERM, ORANGE]
    vals = [ms[k] for k in keys]
    bars = ax.bar(range(4), vals, color=cols, edgecolor=BLACK, lw=0.5, width=0.62)
    ax.axhline(vals[0], color=BLACK, ls="--", lw=0.7, zorder=3)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.018, f"{v:.3f}",
                ha="center", fontsize=6.2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=5.8)
    ax.set_ylabel(r"attribution rank $\rho$ vs. ref")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    # significance annotations, placed above the dashed floor line
    nv = d["arousal"]["null_vs"]
    for i, k in ((1, "container"), (2, "codec")):
        p = nv[k]["p"]
        txt = "n.s." if p > 0.05 else r"$p{=}4{\times}10^{-8}$"
        ax.text(i, 0.755, txt, ha="center", fontsize=5.9,
                color=BLACK if p > 0.05 else VERM)
    ax.text(0.5, 0.985, "estimator null floor, not 1.0", transform=ax.transAxes,
            fontsize=5.9, va="top", ha="center", style="italic")
    panel_tag(ax, "(c)")
    ax.set_title("attribution: reordered", pad=3)

    fig.subplots_adjust(wspace=0.40)
    save(fig, "fig_armA")


# ================================================================= Fig. 4
def fig_bands(d):
    """Codec damage and model reliance occupy different bands (Arm A)."""
    fig, ax = plt.subplots(figsize=(COL, 1.95))
    band = d["band"]
    bands = list(band["attribution_ref"].keys())
    labels = [b.replace("Hz", "").replace("–", "-") for b in bands]
    x = np.arange(len(bands))
    attr = [band["attribution_ref"][b] for b in bands]
    nsr64 = [band["nsr_db"]["64"][b] for b in bands]

    ax.bar(x, attr, color=BLUE, width=0.6, label="model reliance (attribution)")
    ax.set_ylabel("mean occlusion attribution", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax.set_ylim(0, 0.128)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(x, nsr64, "-o", color=VERM, ms=3.4, label="codec damage (NSR, 64 kbps)")
    ax2.set_ylabel("band NSR at 64 kbps (dB)", color=VERM)
    ax2.tick_params(axis="y", colors=VERM)
    ax2.set_ylim(-28.5, -9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6.0)
    ax.set_xlabel("frequency band (Hz)")
    ax.axvspan(1.5, 3.5, color=SKY, alpha=0.16, zorder=0, lw=0)
    ax.set_ylim(0, 0.140)
    ax.text(2.5, 0.104, "model reliance\npeaks here", ha="center", fontsize=5.9,
            color=BLUE)
    ax.text(4.75, 0.0715, "codec damage\npeaks here", ha="right", fontsize=5.9,
            color=VERM)
    rho = band["spearman_attr_vs_damage_64k"]
    ax.text(0.5, 0.99,
            rf"attribution vs. damage: $\rho={rho['rho']:.2f}$, $p={rho['p']:.2f}$",
            transform=ax.transAxes, fontsize=6.2, va="top", ha="center")
    ax.grid(axis="y", alpha=0.22)
    save(fig, "fig_bands")


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
PRETTY = {
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


def fig_synthesis(d):
    """The cross-arm plane: accuracy displacement vs explanation displacement."""
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.7))

    # --- (a) the plane ----------------------------------------------------
    ax = axes[0]
    stab = _stability_rows(d)
    piv = _crossformat_frame(d)

    pts = []
    for _, r in stab.iterrows():
        if r.condition not in ("mp3_64", "mp4_aac64") or r.model not in piv.index:
            continue
        pts.append((abs(piv.loc[r.model, r.condition] - piv.loc[r.model, "ref"]),
                    1.0 - r.rho, r.model, r.condition))
    for dacc, disp, m, cond in pts:
        fam = FAMILY_OF.get(m, "other")
        ax.scatter(dacc, disp, s=27, marker="o" if cond == "mp3_64" else "^",
                   color=FAMILY_COLOUR[fam], edgecolor=BLACK, linewidth=0.4,
                   zorder=3)
    for dacc, disp, m, cond in pts:                   # label sparingly
        if cond != "mp3_64":
            continue
        if m == "ann_pytorch":
            off = (-6, 8); ha = "right"
        elif m == "logistic_regression":
            off = (-6, 7); ha = "right"
        else:
            continue
        ax.annotate(PRETTY.get(m, m), (dacc, disp), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=6.2,
                    color=FAMILY_COLOUR[FAMILY_OF[m]])

    # Arm A point: |Δ acc| = 0.68 − 0.58 (n.s.), attribution ρ = 0.491
    dacc_A = abs(d["arousal"]["discriminability"]["mp3_64"]["acc"]
                 - d["arousal"]["discriminability"]["ref"]["acc"])
    ms = d["arousal"]["map_similarity"]
    ax.scatter(dacc_A, 1 - ms["rho_codec"], s=80, marker="*", color=ORANGE,
               edgecolor=BLACK, linewidth=0.5, zorder=4)
    ax.annotate("LALM, occlusion\nattribution (Arm A)",
                (dacc_A, 1 - ms["rho_codec"]), textcoords="offset points",
                xytext=(9, -1), fontsize=6.2, color=BLACK, va="center")

    floor = 1 - ms["rho_dither (NULL)"]
    ax.axhspan(-0.02, floor, color=YELLOW, alpha=0.20, zorder=0, lw=0)
    ax.axhline(floor, color=ORANGE, ls="--", lw=0.7, zorder=1)
    ax.text(0.004, floor + 0.012, "Arm A null floor (1-LSB dither)",
            fontsize=5.9, color=ORANGE, ha="left")
    ax.text(0.398, 0.135, "Arm B attribution is computed exactly:\n"
                          "its null floor is 0",
            fontsize=5.9, color=GREY, ha="right", va="top")

    ax.set_xlabel(r"$|\Delta|$ accuracy vs. uncompressed")
    ax.set_ylabel(r"explanation displacement $\,1-\rho$")
    ax.set_xlim(-0.015, 0.40)
    ax.set_ylim(-0.02, 0.60)
    ax.grid(alpha=0.22)
    handles = [
        Line2D([], [], ls="", marker="o", ms=4, color=GREY, mec=BLACK,
               mew=0.4, label="MP3 64k"),
        Line2D([], [], ls="", marker="^", ms=4, color=GREY, mec=BLACK,
               mew=0.4, label="MP4/AAC 64k"),
    ] + [Line2D([], [], ls="", marker="s", ms=4, color=FAMILY_COLOUR[k],
                label=v) for k, v in (("tree", "tree"), ("linear", "linear"),
                                      ("neural", "neural"))]
    ax.legend(handles=handles, frameon=False, loc="upper left",
              bbox_to_anchor=(-0.01, 1.005), handlelength=1.0, fontsize=6.0,
              ncol=1, labelspacing=0.35)
    panel_tag(ax, "(a)", dx=-0.155)
    ax.set_title("accuracy change does not index explanation change", pad=3)

    # --- (b) what enters and what leaves ---------------------------------
    ax = axes[1]
    ref = d["ig"]["ref"].set_index("feature")["importance"]
    mp3 = d["ig"]["mp3_64"].set_index("feature")["importance"]
    r_ref = ref.rank(ascending=False)
    r_mp3 = mp3.rank(ascending=False)
    top_ref = set(r_ref.nsmallest(25).index)
    top_mp3 = set(r_mp3.nsmallest(25).index)
    left = sorted(top_ref - top_mp3, key=lambda f: r_ref[f])
    entered = sorted(top_mp3 - top_ref, key=lambda f: r_mp3[f])

    def kind(f):
        if f.startswith("spectral_contrast") or f.startswith("spectral_flatness"):
            return VERM                      # codec fingerprint
        if f.startswith("prosody"):
            return GREEN                     # voice
        return GREY

    n = max(len(left), len(entered))
    ax.axvline(0.5, color=BLACK, lw=0.6, ymin=0.03, ymax=0.90)
    for i, f in enumerate(left):
        ax.text(0.475, n - i, f, ha="right", va="center", fontsize=5.5,
                color=kind(f), family="monospace")
    for i, f in enumerate(entered):
        ax.text(0.525, n - i, f, ha="left", va="center", fontsize=5.5,
                color=kind(f), family="monospace")
    ax.text(0.475, n + 1.5, "left the top 25", ha="right", fontsize=6.8,
            fontweight="bold")
    ax.text(0.525, n + 1.5, "entered the top 25", ha="left", fontsize=6.8,
            fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.9, n + 2.6)
    ax.axis("off")
    handles = [Patch(color=GREEN, label="voice quality / prosody"),
               Patch(color=VERM, label="encoder fingerprint"),
               Patch(color=GREY, label="cepstral, other")]
    ax.legend(handles=handles, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.005), ncol=3, fontsize=5.9,
              handlelength=1.0, columnspacing=1.2)
    panel_tag(ax, "(b)", dx=-0.02)
    ax.set_title("under MP3 the ANN stops listening to the voice", pad=3)

    fig.subplots_adjust(wspace=0.30)
    save(fig, "fig_synthesis")


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


# ================================================================= main
def main():
    d = load_all()
    fig_signal(d)
    fig_armA(d)
    fig_bands(d)
    fig_crossformat(d)
    fig_synthesis(d)
    fig_neutralise(d)
    fig_methods(d)
    fig_sanity(d)
    print("done")


if __name__ == "__main__":
    main()
