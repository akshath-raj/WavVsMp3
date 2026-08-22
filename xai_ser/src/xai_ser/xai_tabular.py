"""Explainability for the classical models.

Five complementary views, because no single method is trustworthy alone:

  intrinsic     — tree impurity importances and linear coefficients: free, but
                  impurity importance is biased toward high-cardinality features.
  permutation   — model-agnostic, measured on held-out data, so it answers
                  "what does this model actually rely on to generalise?"
  SHAP          — additive local attributions with a consistency guarantee;
                  TreeSHAP where exact, KernelSHAP elsewhere.
  LIME          — local surrogate fits; included precisely because it can
                  disagree with SHAP, and disagreement is itself a finding.
  surrogate     — a depth-limited tree trained on the black box's own
                  predictions, reported with its fidelity so the reader knows
                  how much of the model the readable rules actually capture.

The cross-format attribution comparison at the end is the study's core test:
whether compression moves *what the model looks at* even when it does not move
accuracy.
"""

from __future__ import annotations

import os

# Must precede any OpenMP-linked import. This module holds XGBoost, LightGBM and
# sklearn in one process and then forks joblib workers for permutation
# importance; on macOS that combination deadlocks in an OpenMP join barrier
# (observed reproducibly on the RBF-SVM stage). Parallelism is recovered inside
# permutation_importance, which forks its own workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import shap  # noqa: E402
from lime.lime_tabular import LimeTabularExplainer  # noqa: E402
from scipy import stats as sps  # noqa: E402
from sklearn.inspection import PartialDependenceDisplay, permutation_importance  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree  # noqa: E402

from .datasets import CONDITIONS, family_of
from .metadata import EMOTIONS
from .models import _matrices, prepare
from .paths import MODELS_DIR, XAI_DIR, ensure_dirs, timestamp

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

TREE_MODELS = {"decision_tree", "random_forest", "extra_trees", "xgboost", "lightgbm"}
SEED = 42


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def latest_run_dir() -> Path:
    return MODELS_DIR / (MODELS_DIR / "LATEST").read_text().strip()


def split_pipeline(pipe):
    """Separate the preprocessing steps from the final estimator."""
    pre = pipe[:-1]
    return pre, pipe[-1]


# ------------------------------------------------------------ intrinsic

def intrinsic_importance(pipe, cols: list[str]) -> pd.DataFrame | None:
    _, est = split_pipeline(pipe)
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float)
        kind = "impurity/gain"
    elif hasattr(est, "coef_"):
        vals = np.abs(np.asarray(est.coef_)).mean(axis=0)
        kind = "mean |coefficient|"
    else:
        return None
    return pd.DataFrame({"feature": cols, "importance": vals, "kind": kind,
                         "family": [family_of(c) for c in cols]}).sort_values(
        "importance", ascending=False)


# ---------------------------------------------------------- permutation

def permutation_importances(pipe, X, y, cols, n_repeats=8) -> pd.DataFrame:
    res = permutation_importance(
        pipe, X, y, n_repeats=n_repeats, random_state=SEED,
        scoring="balanced_accuracy", n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": cols, "importance": res.importances_mean,
        "std": res.importances_std, "family": [family_of(c) for c in cols],
    }).sort_values("importance", ascending=False)


# ----------------------------------------------------------------- SHAP

def is_exact_explainer(model_name: str, pipe) -> bool:
    """True when SHAP is cheap and exact (tree or linear), false for KernelSHAP.

    KernelSHAP costs `nsamples` model evaluations per explained row, which on an
    RBF-SVM with thousands of support vectors runs to hours. Cross-format SHAP is
    therefore computed only for exact explainers; kernel models get a single
    reduced-sample explanation on `ref` plus the model-agnostic methods
    (permutation, LIME) that cover them at reasonable cost.
    """
    return model_name in TREE_MODELS or hasattr(split_pipeline(pipe)[1], "coef_")


def shap_values(pipe, model_name: str, X: pd.DataFrame, background: pd.DataFrame,
                cols: list[str], nsamples: int = 100):
    """Return SHAP values shaped (n_samples, n_features, n_classes)."""
    pre, est = split_pipeline(pipe)
    Xt = pd.DataFrame(pre.transform(X), columns=cols)
    Bt = pd.DataFrame(pre.transform(background), columns=cols)

    if model_name in TREE_MODELS:
        explainer = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
        sv = explainer.shap_values(Xt, check_additivity=False)
    elif hasattr(est, "coef_"):
        explainer = shap.LinearExplainer(est, Bt)
        sv = explainer.shap_values(Xt)
    else:
        # KernelSHAP solves a weighted least-squares fit with one unknown per
        # feature. With fewer coalitions than features the system is
        # rank-deficient and returns numerically diverged values (observed here:
        # max mean|SHAP| of 2e12 against a median of 5e-4). Sampling enough
        # coalitions is the only fix, and it is not optional.
        if nsamples <= len(cols):
            raise ValueError(
                f"KernelSHAP needs nsamples > n_features to be identifiable; "
                f"got nsamples={nsamples} for {len(cols)} features. Either raise "
                f"nsamples above {len(cols)} or explain a reduced feature set."
            )
        summary = shap.kmeans(Bt, 25)
        explainer = shap.KernelExplainer(est.predict_proba, summary)
        sv = explainer.shap_values(Xt, nsamples=nsamples, silent=True)

    sv = np.asarray(sv) if not isinstance(sv, list) else np.stack(sv, axis=-1)
    if sv.ndim == 2:  # binary/linear collapse -> add a class axis
        sv = sv[:, :, None]
    if sv.shape[0] == len(EMOTIONS) and sv.shape[1] == len(Xt):  # (class, n, feat)
        sv = np.transpose(sv, (1, 2, 0))
    return sv, Xt


def global_shap_importance(sv: np.ndarray, cols: list[str]) -> pd.DataFrame:
    mag = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)
    return pd.DataFrame({"feature": cols, "importance": mag,
                         "family": [family_of(c) for c in cols]}).sort_values(
        "importance", ascending=False)


def per_class_shap(sv: np.ndarray, cols: list[str]) -> pd.DataFrame:
    rows = []
    for k in range(sv.shape[2]):
        mag = np.abs(sv[:, :, k]).mean(axis=0)
        for c, v in zip(cols, mag):
            rows.append({"emotion": EMOTIONS[k] if k < len(EMOTIONS) else str(k),
                         "feature": c, "importance": float(v), "family": family_of(c)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- LIME

def lime_importance(pipe, X_train: pd.DataFrame, X_explain: pd.DataFrame,
                    cols: list[str], n_instances: int = 60, n_features: int = 15) -> pd.DataFrame:
    """Aggregate local LIME weights into a global ranking."""
    explainer = LimeTabularExplainer(
        training_data=np.nan_to_num(X_train.to_numpy(), nan=np.nanmedian(X_train.to_numpy())),
        feature_names=cols, class_names=EMOTIONS, discretize_continuous=True,
        random_state=SEED, mode="classification",
    )
    med = X_train.median()
    filled = X_explain.fillna(med)
    acc: dict[str, float] = {}
    n_used = 0
    for i in range(min(n_instances, len(filled))):
        row = filled.iloc[i].to_numpy()
        try:
            exp = explainer.explain_instance(
                row, lambda z: pipe.predict_proba(pd.DataFrame(z, columns=cols)),
                num_features=n_features, num_samples=800, top_labels=1,
            )
        except Exception:
            continue
        label = list(exp.available_labels())[0]
        for idx, weight in exp.as_map()[label]:
            acc[cols[idx]] = acc.get(cols[idx], 0.0) + abs(weight)
        n_used += 1
    if not acc:
        return pd.DataFrame(columns=["feature", "importance", "family"])
    out = pd.DataFrame(
        {"feature": list(acc), "importance": [v / max(n_used, 1) for v in acc.values()]}
    )
    out["family"] = [family_of(c) for c in out["feature"]]
    return out.sort_values("importance", ascending=False)


# ------------------------------------------------------------ surrogate

def surrogate_tree(pipe, X: pd.DataFrame, cols: list[str], max_depth: int = 4):
    """A readable tree fitted to the black box's predictions, with fidelity."""
    y_hat = pipe.predict(X)
    med = X.median()
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=25, random_state=SEED)
    tree.fit(X.fillna(med), y_hat)
    fidelity = float((tree.predict(X.fillna(med)) == y_hat).mean())
    rules = export_text(tree, feature_names=list(cols), max_depth=max_depth)
    return tree, fidelity, rules


# --------------------------------------------- cross-format attribution

def attribution_stability(per_cond: dict[str, pd.DataFrame], top_k: int = 25) -> dict:
    """Do explanations survive the codec?

    Compares each condition's global SHAP ranking against `ref` with Spearman
    rank correlation (all features) and top-k Jaccard overlap (the features an
    analyst would actually read).
    """
    ref = per_cond["ref"].set_index("feature")["importance"]
    out = {}
    for cond, frame in per_cond.items():
        if cond == "ref":
            continue
        other = frame.set_index("feature")["importance"].reindex(ref.index)
        rho, p = sps.spearmanr(ref.to_numpy(), other.to_numpy(), nan_policy="omit")
        pear = float(np.corrcoef(np.nan_to_num(ref), np.nan_to_num(other))[0, 1])
        a = set(ref.sort_values(ascending=False).head(top_k).index)
        b = set(other.sort_values(ascending=False).head(top_k).index)
        denom = np.abs(ref).sum()
        out[cond] = {
            "spearman_rho": float(rho), "spearman_p": float(p),
            "pearson_r": pear,
            f"top{top_k}_jaccard": len(a & b) / len(a | b),
            f"top{top_k}_overlap": len(a & b),
            "entered_topk": sorted(b - a),
            "left_topk": sorted(a - b),
            "l1_shift": float(np.abs(other.fillna(0) - ref).sum() / denom) if denom else np.nan,
        }
    return out


# ------------------------------------------------------------------ run

def run(models: tuple[str, ...] = ("xgboost", "lightgbm", "decision_tree",
                                   "random_forest", "logistic_regression", "svm_rbf"),
        train_condition: str = "ref", shap_n: int = 300, lime_n: int = 60,
        perm_repeats: int = 8, kernel_shap_n: int = 60,
        run_dir: Path | None = None) -> dict:
    ensure_dirs()
    ts = timestamp()
    out_dir = XAI_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = run_dir or latest_run_dir()

    df, cols, split = prepare()
    rng = np.random.default_rng(SEED)

    data = {}
    for cond in CONDITIONS:
        sub, masks, X, y = _matrices(df, cols, split, cond)
        data[cond] = {
            "train": (X[masks["train"]], y[masks["train"]]),
            "test": (X[masks["test"]], y[masks["test"]], sub[masks["test"]]),
        }
    Xte_ref, yte_ref, meta_ref = data["ref"]["test"]
    idx = rng.choice(len(Xte_ref), size=min(shap_n, len(Xte_ref)), replace=False)
    bg = data["ref"]["train"][0].sample(n=min(200, len(data["ref"]["train"][0])), random_state=SEED)

    report: dict = {"created_utc": ts, "train_condition": train_condition,
                    "models": {}, "figures": {}}

    for name in models:
        model_path = run_dir / "fitted" / f"{train_condition}__{name}.joblib"
        if not model_path.exists():
            print(f"skip {name}: {model_path} not found")
            continue
        pipe = joblib.load(model_path)
        print(f"\n--- XAI: {name} ---")
        m_dir = out_dir / name
        m_dir.mkdir(exist_ok=True)
        entry: dict = {}

        intr = intrinsic_importance(pipe, cols)
        if intr is not None:
            intr.to_csv(m_dir / "intrinsic_importance.csv", index=False)
            entry["intrinsic_top15"] = intr.head(15).round(6).to_dict("records")

        print("  permutation importance…")
        perm = permutation_importances(pipe, Xte_ref, yte_ref, cols, n_repeats=perm_repeats)
        perm.to_csv(m_dir / "permutation_importance.csv", index=False)
        entry["permutation_top15"] = perm.head(15).round(6).to_dict("records")
        entry["permutation_family_share"] = (
            perm[perm["importance"] > 0].groupby("family")["importance"].sum()
            / perm[perm["importance"] > 0]["importance"].sum()
        ).round(4).to_dict()

        exact = is_exact_explainer(name, pipe)
        conds = CONDITIONS if exact else ["ref"]
        rows_for_shap = idx if exact else idx[:kernel_shap_n]
        print(f"  SHAP ({'exact' if exact else 'kernel'}, {len(rows_for_shap)} rows, "
              f"{len(conds)} condition(s))…")

        shap_per_cond = {}
        for cond in conds:
            Xte_c = data[cond]["test"][0].iloc[rows_for_shap]
            sv, Xt = shap_values(pipe, name, Xte_c, bg, cols)
            gi = global_shap_importance(sv, cols)
            shap_per_cond[cond] = gi
            gi.to_csv(m_dir / f"shap_global_{cond}.csv", index=False)
            if cond == "ref":
                entry["shap_top15"] = gi.head(15).round(6).to_dict("records")
                entry["shap_family_share"] = (
                    gi.groupby("family")["importance"].sum() / gi["importance"].sum()
                ).round(4).to_dict()
                pcs = per_class_shap(sv, cols)
                pcs.to_csv(m_dir / "shap_per_class.csv", index=False)
                entry["shap_top5_per_class"] = {
                    e: g.nlargest(5, "importance")[["feature", "importance"]].round(5).to_dict("records")
                    for e, g in pcs.groupby("emotion")
                }
                _shap_figures(sv, Xt, gi, pcs, name, m_dir, report)

        if len(shap_per_cond) > 1:
            entry["attribution_stability"] = attribution_stability(shap_per_cond)
            _stability_figure(shap_per_cond, name, m_dir, report)
        else:
            entry["attribution_stability_note"] = (
                "skipped: KernelSHAP is too costly to run per condition for this model"
            )

        print("  LIME…")
        lime_imp = lime_importance(pipe, data["ref"]["train"][0], Xte_ref, cols, n_instances=lime_n)
        lime_imp.to_csv(m_dir / "lime_importance.csv", index=False)
        entry["lime_top15"] = lime_imp.head(15).round(6).to_dict("records")
        entry["lime_vs_shap"] = _method_agreement(
            shap_per_cond["ref"], lime_imp, perm, intr
        )

        print("  partial dependence…")
        top_feats = shap_per_cond["ref"].head(4)["feature"].tolist()
        entry["pdp_features"] = top_feats
        _pdp_figure(pipe, Xte_ref, cols, top_feats, name, m_dir, report)

        print("  surrogate tree…")
        tree, fidelity, rules = surrogate_tree(pipe, Xte_ref, cols)
        (m_dir / "surrogate_rules.txt").write_text(rules)
        entry["surrogate_fidelity"] = fidelity
        entry["surrogate_features_used"] = sorted(
            {cols[i] for i in tree.tree_.feature if i >= 0}
        )
        fig, ax = plt.subplots(figsize=(20, 9))
        plot_tree(tree, feature_names=cols, class_names=EMOTIONS, filled=True,
                  max_depth=3, fontsize=7, ax=ax, impurity=False, proportion=True)
        ax.set_title(f"{name}: depth-limited surrogate (fidelity to black box = {fidelity:.3f})")
        report["figures"][f"{name}_surrogate"] = _save(fig, m_dir / "surrogate_tree.png")

        report["models"][name] = entry
        # Written after every model: TreeSHAP over a large forest is slow enough
        # that a run may be interrupted, and partial results are still useful.
        with open(out_dir / "xai_summary.json", "w") as fh:
            json.dump(report, fh, indent=2, default=str)

    with open(out_dir / "xai_summary.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    with open(XAI_DIR / "LATEST", "w") as fh:
        fh.write(ts)
    print(f"\nXAI complete -> {out_dir}")
    return report


def _method_agreement(shap_imp, lime_imp, perm_imp, intr_imp) -> dict:
    """Rank agreement between attribution methods on the same model."""
    base = shap_imp.set_index("feature")["importance"]
    res = {}
    for label, other in [("lime", lime_imp), ("permutation", perm_imp), ("intrinsic", intr_imp)]:
        if other is None or other.empty:
            continue
        o = other.set_index("feature")["importance"].reindex(base.index).fillna(0.0)
        rho, _ = sps.spearmanr(base.to_numpy(), o.to_numpy())
        a = set(base.nlargest(20).index)
        b = set(o.nlargest(20).index)
        res[f"shap_vs_{label}"] = {
            "spearman_rho": float(rho),
            "top20_overlap": len(a & b),
        }
    return res


def _shap_figures(sv, Xt, gi, pcs, name, m_dir, report) -> None:
    top = gi.head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.barplot(data=top, x="importance", y="feature", hue="family", dodge=False, ax=ax)
    ax.set(title=f"{name}: global SHAP importance (ref)", xlabel="mean |SHAP|")
    report["figures"][f"{name}_shap_global"] = _save(fig, m_dir / "shap_global.png")

    order = top["feature"].tolist()
    piv = pcs[pcs["feature"].isin(order)].pivot(index="feature", columns="emotion",
                                                values="importance").loc[order]
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(piv, cmap="magma", ax=ax, cbar_kws={"label": "mean |SHAP|"})
    ax.set_title(f"{name}: which feature drives which emotion")
    report["figures"][f"{name}_shap_per_class"] = _save(fig, m_dir / "shap_per_class.png")

    try:
        cls = 0
        fig = plt.figure()
        shap.summary_plot(sv[:, :, cls], Xt, feature_names=list(Xt.columns),
                          max_display=18, show=False)
        plt.title(f"{name}: SHAP beeswarm — class {EMOTIONS[cls]}")
        report["figures"][f"{name}_beeswarm"] = _save(plt.gcf(), m_dir / "shap_beeswarm_ANG.png")
    except Exception as exc:
        print(f"    beeswarm skipped: {exc}")


def _stability_figure(shap_per_cond, name, m_dir, report) -> None:
    ref = shap_per_cond["ref"].set_index("feature")["importance"]
    frames = []
    for cond, gi in shap_per_cond.items():
        s = gi.set_index("feature")["importance"].reindex(ref.index)
        frames.append(pd.DataFrame({"feature": ref.index, "condition": cond,
                                    "importance": s.to_numpy()}))
    long = pd.concat(frames)
    top = ref.nlargest(15).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=long[long["feature"].isin(top)], x="importance", y="feature",
                hue="condition", order=top, ax=axes[0])
    axes[0].set_title(f"{name}: top-15 SHAP importance across formats")
    for cond, gi in shap_per_cond.items():
        if cond == "ref":
            continue
        s = gi.set_index("feature")["importance"].reindex(ref.index)
        axes[1].scatter(ref, s, s=8, alpha=0.6, label=cond)
    lim = float(ref.max()) * 1.05
    axes[1].plot([0, lim], [0, lim], ls="--", c="grey")
    axes[1].set(xlabel="mean |SHAP| on ref", ylabel="mean |SHAP| on compressed",
                title="Per-feature attribution shift")
    axes[1].legend()
    report["figures"][f"{name}_stability"] = _save(fig, m_dir / "attribution_stability.png")


def _pdp_figure(pipe, X, cols, feats, name, m_dir, report) -> None:
    try:
        fig, axes = plt.subplots(1, len(feats), figsize=(4.2 * len(feats), 3.6))
        PartialDependenceDisplay.from_estimator(
            pipe, X.fillna(X.median()), features=feats, target=0,
            ax=axes, kind="both", subsample=40, random_state=SEED,
            feature_names=list(cols), n_jobs=1,
        )
        fig.suptitle(f"{name}: partial dependence / ICE for class {EMOTIONS[0]}", y=1.04)
        report["figures"][f"{name}_pdp"] = _save(fig, m_dir / "pdp_ice.png")
    except Exception as exc:
        print(f"    PDP skipped: {exc}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run tabular XAI over fitted models.")
    ap.add_argument("--models", nargs="*",
                    default=["xgboost", "lightgbm", "decision_tree",
                             "logistic_regression", "random_forest", "svm_rbf"])
    ap.add_argument("--shap-n", type=int, default=300)
    ap.add_argument("--lime-n", type=int, default=60)
    ap.add_argument("--perm-repeats", type=int, default=8)
    ap.add_argument("--kernel-shap-n", type=int, default=60)
    a = ap.parse_args()
    run(models=tuple(a.models), shap_n=a.shap_n, lime_n=a.lime_n,
        perm_repeats=a.perm_repeats, kernel_shap_n=a.kernel_shap_n)
