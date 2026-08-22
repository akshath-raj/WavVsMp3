"""Deep-learning explainability for the ANN (captum).

Gradient-based attribution needs a differentiable model, which the trees do not
give us — so this is where the deep and classical arms genuinely differ in
method rather than just in wording.

Methods run here:

  Saliency            |d logit / d x| — the crudest gradient signal.
  InputXGradient      gradient scaled by the input, a first-order attribution.
  IntegratedGradients path integral from a baseline; satisfies completeness.
  DeepLift            rescale-rule backprop against a reference input.
  GradientShap        expectation of gradients over noisy baselines (SHAP's
                      deep-model estimator).
  FeatureAblation     perturbation-based, model-agnostic control for the above.

Two checks accompany them:

  * a **model-randomisation sanity check** (Adebayo et al.): attributions from
    the trained network are compared against attributions from a randomly
    re-initialised one. A method whose output barely changes is describing the
    input, not the model, and should not be believed.
  * the same **cross-format stability** analysis applied to the tree models, so
    the deep and classical arms can be compared on identical axes.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from captum.attr import (  # noqa: E402
    DeepLift,
    FeatureAblation,
    GradientShap,
    InputXGradient,
    IntegratedGradients,
    Saliency,
)
from scipy import stats as sps  # noqa: E402

from .ann import EmotionMLP, AnnConfig, load_ann  # noqa: E402
from .datasets import CONDITIONS, family_of  # noqa: E402
from .metadata import EMOTIONS  # noqa: E402
from .models import _matrices, prepare  # noqa: E402
from .paths import XAI_DIR, ensure_dirs, timestamp  # noqa: E402
from .xai_tabular import attribution_stability  # noqa: E402

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

SEED = 42


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def attribute(model, X: torch.Tensor, target: torch.Tensor, baseline: torch.Tensor) -> dict[str, np.ndarray]:
    """Every method's attributions for one batch, keyed by method name."""
    model.eval()
    X = X.clone().requires_grad_(True)
    out: dict[str, np.ndarray] = {}

    out["saliency"] = Saliency(model).attribute(X, target=target, abs=True).detach().numpy()
    out["input_x_gradient"] = InputXGradient(model).attribute(X, target=target).detach().numpy()
    out["integrated_gradients"] = IntegratedGradients(model).attribute(
        X, baselines=baseline, target=target, n_steps=64
    ).detach().numpy()
    out["deeplift"] = DeepLift(model).attribute(
        X, baselines=baseline.expand_as(X), target=target
    ).detach().numpy()

    torch.manual_seed(SEED)
    ref = torch.cat([baseline.expand(32, -1), torch.randn(32, X.shape[1]) * 0.5])
    out["gradient_shap"] = GradientShap(model).attribute(
        X, baselines=ref, target=target, n_samples=32, stdevs=0.09
    ).detach().numpy()

    out["feature_ablation"] = FeatureAblation(model).attribute(
        X, baselines=baseline, target=target
    ).detach().numpy()
    return out


def global_ranking(attr: np.ndarray, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": cols,
        "importance": np.abs(attr).mean(axis=0),
        "family": [family_of(c) for c in cols],
    }).sort_values("importance", ascending=False)


def run(shap_n: int = 600, ann_dir: Path | None = None) -> dict:
    ensure_dirs()
    ts = timestamp()
    out_dir = XAI_DIR / "deep" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    model, wrapper, blob = load_ann(ann_dir)
    cols = blob["feature_names"]
    df, all_cols, split = prepare()
    assert list(all_cols) == list(cols), "feature table changed since the ANN was trained"

    rng = np.random.default_rng(SEED)
    data = {}
    for cond in CONDITIONS:
        sub, masks, X, y = _matrices(df, cols, split, cond)
        data[cond] = (X[masks["test"]], y[masks["test"]])
    n_test = len(data["ref"][0])
    idx = rng.choice(n_test, size=min(shap_n, n_test), replace=False)

    # Baseline = the all-zeros vector, which in standardised space is the
    # training-set mean speaker: "an average clip", not silence.
    baseline = torch.zeros(1, len(cols))

    report: dict = {"created_utc": ts, "ann_dir": str(blob.get("train_condition")),
                    "n_explained": int(len(idx)), "figures": {}, "methods": {}}

    # --- attributions on ref, all methods ---
    Xref = torch.from_numpy(wrapper.transform(data["ref"][0].iloc[idx]))
    target = torch.from_numpy(wrapper.predict(data["ref"][0].iloc[idx])).long()
    attrs = attribute(model, Xref, target, baseline)

    rankings = {}
    for method, a in attrs.items():
        gr = global_ranking(a, cols)
        gr.to_csv(out_dir / f"global_{method}.csv", index=False)
        rankings[method] = gr
        report["methods"][method] = {
            "top15": gr.head(15).round(6).to_dict("records"),
            "family_share": (gr.groupby("family")["importance"].sum()
                             / gr["importance"].sum()).round(4).to_dict(),
        }

    # --- do the deep methods agree with each other? ---
    names = list(rankings)
    agree = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            ra = rankings[a].set_index("feature")["importance"]
            rb = rankings[b].set_index("feature")["importance"].reindex(ra.index)
            agree.loc[a, b] = sps.spearmanr(ra, rb).statistic
    agree.to_csv(out_dir / "method_agreement.csv")
    report["method_agreement_spearman"] = agree.round(3).to_dict()

    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(agree.astype(float), annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_title("Rank agreement between deep attribution methods")
    report["figures"]["method_agreement"] = _save(fig, out_dir / "method_agreement.png")

    # --- per-class attribution (integrated gradients) ---
    rows = []
    for k, emo in enumerate(EMOTIONS):
        tgt = torch.full((len(idx),), k, dtype=torch.long)
        a = IntegratedGradients(model).attribute(
            Xref.clone().requires_grad_(True), baselines=baseline, target=tgt, n_steps=48
        ).detach().numpy()
        mag = np.abs(a).mean(axis=0)
        for c, v in zip(cols, mag):
            rows.append({"emotion": emo, "feature": c, "importance": float(v),
                         "family": family_of(c)})
    per_class = pd.DataFrame(rows)
    per_class.to_csv(out_dir / "ig_per_class.csv", index=False)
    report["ig_top5_per_class"] = {
        e: g.nlargest(5, "importance")[["feature", "importance"]].round(5).to_dict("records")
        for e, g in per_class.groupby("emotion")
    }

    top = rankings["integrated_gradients"].head(20)["feature"].tolist()
    piv = per_class[per_class["feature"].isin(top)].pivot(
        index="feature", columns="emotion", values="importance").loc[top]
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(piv, cmap="magma", ax=ax, cbar_kws={"label": "mean |IG|"})
    ax.set_title("ANN: integrated-gradients attribution per emotion")
    report["figures"]["ig_per_class"] = _save(fig, out_dir / "ig_per_class.png")

    # --- cross-format stability, same axes as the tree models ---
    per_cond = {}
    for cond in CONDITIONS:
        Xc = torch.from_numpy(wrapper.transform(data[cond][0].iloc[idx]))
        tgt = torch.from_numpy(wrapper.predict(data[cond][0].iloc[idx])).long()
        a = IntegratedGradients(model).attribute(
            Xc.clone().requires_grad_(True), baselines=baseline, target=tgt, n_steps=48
        ).detach().numpy()
        per_cond[cond] = global_ranking(a, cols)
        per_cond[cond].to_csv(out_dir / f"ig_global_{cond}.csv", index=False)
    report["attribution_stability_ig"] = attribution_stability(per_cond)

    ref_imp = per_cond["ref"].set_index("feature")["importance"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    tops = ref_imp.nlargest(15).index.tolist()
    long = pd.concat([
        pd.DataFrame({"feature": ref_imp.index, "condition": c,
                      "importance": g.set_index("feature")["importance"].reindex(ref_imp.index).to_numpy()})
        for c, g in per_cond.items()
    ])
    sns.barplot(data=long[long["feature"].isin(tops)], x="importance", y="feature",
                hue="condition", order=tops, ax=axes[0])
    axes[0].set_title("ANN: top-15 IG attribution across formats")
    for cond, g in per_cond.items():
        if cond == "ref":
            continue
        axes[1].scatter(ref_imp, g.set_index("feature")["importance"].reindex(ref_imp.index),
                        s=8, alpha=0.6, label=cond)
    lim = float(ref_imp.max()) * 1.05
    axes[1].plot([0, lim], [0, lim], ls="--", c="grey")
    axes[1].set(xlabel="mean |IG| on ref", ylabel="mean |IG| on compressed",
                title="ANN per-feature attribution shift")
    axes[1].legend()
    report["figures"]["ann_stability"] = _save(fig, out_dir / "attribution_stability.png")

    # --- sanity check: does attribution track the model, or just the input? ---
    torch.manual_seed(SEED + 1)
    random_model = EmotionMLP(len(cols), len(EMOTIONS), AnnConfig(**blob["config"]))
    random_model.eval()
    rand_attrs = attribute(random_model, Xref.clone(), target, baseline)
    sanity = {}
    for method in attrs:
        a = global_ranking(attrs[method], cols).set_index("feature")["importance"]
        b = global_ranking(rand_attrs[method], cols).set_index("feature")["importance"].reindex(a.index)
        sanity[method] = {
            "spearman_trained_vs_random": float(sps.spearmanr(a, b).statistic),
            "top20_overlap": len(set(a.nlargest(20).index) & set(b.nlargest(20).index)),
        }
    report["model_randomisation_sanity"] = sanity

    fig, ax = plt.subplots(figsize=(8, 4.5))
    s = pd.Series({k: v["spearman_trained_vs_random"] for k, v in sanity.items()}).sort_values()
    sns.barplot(x=s.values, y=s.index, ax=ax, color=sns.color_palette("colorblind")[3])
    ax.axvline(0, c="k", lw=0.8)
    ax.set(xlabel="Spearman rho (trained vs randomly initialised model)",
           title="Model-randomisation sanity check\n(near zero is what a faithful method should show)")
    report["figures"]["sanity_check"] = _save(fig, out_dir / "sanity_check.png")

    with open(out_dir / "deep_xai_summary.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    with open(XAI_DIR / "deep" / "LATEST", "w") as fh:
        fh.write(ts)
    print(f"deep XAI complete -> {out_dir}")
    return report


def compare_with_trees(deep_dir: Path, tree_csv: Path, top_k: int = 25) -> dict:
    """Do a random forest and a neural net rely on the same acoustics?"""
    ig = pd.read_csv(deep_dir / "ig_global_ref.csv").set_index("feature")["importance"]
    tree = pd.read_csv(tree_csv).set_index("feature")["importance"].reindex(ig.index)
    rho = sps.spearmanr(ig, tree, nan_policy="omit").statistic
    a, b = set(ig.nlargest(top_k).index), set(tree.nlargest(top_k).index)
    return {
        "spearman_rho": float(rho),
        f"top{top_k}_overlap": len(a & b),
        f"top{top_k}_jaccard": len(a & b) / len(a | b),
        "shared_features": sorted(a & b),
        "ann_only": sorted(a - b),
        "tree_only": sorted(b - a),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run deep XAI over the trained ANN.")
    ap.add_argument("--shap-n", type=int, default=600)
    a = ap.parse_args()
    run(shap_n=a.shap_n)
