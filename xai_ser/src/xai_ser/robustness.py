"""Why does a codec cost accuracy — and for which model families?

The cross-format table says *that* compression hurts. This module asks *how*.
It ranks features by how far the codec moves them relative to the training
spread, then neutralises the worst offenders (replacing them with training
medians in both the compressed and the clean test set) and re-scores.

If accuracy recovers after neutralising a handful of features, the failure is
**covariate shift on a few inputs**, not a loss of emotional information from
the signal — a distinction that changes the fix entirely: drop or clamp those
features, rather than retrain on compressed audio.

The paired "clean control" column matters. Neutralising a feature also removes
whatever real signal it carried, so recovery is only meaningful relative to what
the same masking costs on uncompressed audio.
"""

from __future__ import annotations

import os

# Must precede any import that pulls in an OpenMP runtime. This module is the
# only place that loads PyTorch and XGBoost into the same process, and on macOS
# the two OpenMP runtimes deadlock in a join barrier the first time both thread
# pools are live. Prediction here is cheap, so serialising costs nothing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json  # noqa: E402

import joblib  # noqa: E402
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from .datasets import CONDITIONS
from .models import _matrices, prepare
from .paths import MODELS_DIR, OUTPUTS, ensure_dirs, timestamp
from .xai_tabular import latest_run_dir

sns.set_theme(style="whitegrid")

NEUTRALISE_STEPS = (1, 2, 3, 5, 10, 20, 40, 80)


def shift_ranking(clean: pd.DataFrame, compressed: pd.DataFrame,
                  train: pd.DataFrame) -> pd.Series:
    """Features ordered by |mean shift| in units of training standard deviation."""
    sd = train.std().replace(0, np.nan)
    return ((compressed - clean).mean() / sd).abs().sort_values(ascending=False)


def run(models: tuple[str, ...] = ("random_forest", "xgboost", "decision_tree",
                                   "logistic_regression", "svm_rbf", "mlp_sklearn"),
        include_ann: bool = True, train_condition: str = "ref") -> dict:
    ensure_dirs()
    ts = timestamp()
    out_dir = OUTPUTS / "robustness" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    df, cols, split = prepare()
    sub, masks, X, y = _matrices(df, cols, split, train_condition)
    train_X = X[masks["train"]]
    median = train_X.median()

    test = {}
    for cond in CONDITIONS:
        s, m, Xc, yc = _matrices(df, cols, split, cond)
        test[cond] = (Xc[m["test"]].reset_index(drop=True), yc[m["test"]])
    y_test = test[train_condition][1]

    predictors: dict[str, object] = {}
    run_dir = latest_run_dir()
    for name in models:
        p = run_dir / "fitted" / f"{train_condition}__{name}.joblib"
        if p.exists():
            predictors[name] = joblib.load(p)
        else:
            print(f"skip {name}: not fitted")
    if include_ann:
        from .ann import load_ann

        predictors["ann_pytorch"] = load_ann()[1]

    rows, shift_tables = [], {}
    for cond in CONDITIONS:
        if cond == train_condition:
            continue
        ranking = shift_ranking(test[train_condition][0], test[cond][0], train_X)
        shift_tables[cond] = ranking
        ranking.rename("shift_sd").to_csv(out_dir / f"feature_shift_{cond}.csv")

        for name, model in predictors.items():
            base_clean = balanced_accuracy_score(y_test, model.predict(test[train_condition][0]))
            base_comp = balanced_accuracy_score(y_test, model.predict(test[cond][0]))
            rows.append({"model": name, "condition": cond, "n_neutralised": 0,
                         "uar_compressed": base_comp, "uar_clean_control": base_clean})

            for n in NEUTRALISE_STEPS:
                drop = ranking.head(n).index
                Xc = test[cond][0].copy()
                Xr = test[train_condition][0].copy()
                Xc[drop] = median[drop].values
                Xr[drop] = median[drop].values
                rows.append({
                    "model": name, "condition": cond, "n_neutralised": n,
                    "uar_compressed": balanced_accuracy_score(y_test, model.predict(Xc)),
                    "uar_clean_control": balanced_accuracy_score(y_test, model.predict(Xr)),
                })
            print(f"  {name:<20} {cond:<14} base {base_comp:.4f} -> "
                  f"{rows[-1]['uar_compressed']:.4f} after {NEUTRALISE_STEPS[-1]} masked")

    res = pd.DataFrame(rows)
    res["recovery"] = res["uar_compressed"] - res.groupby(["model", "condition"])[
        "uar_compressed"].transform("first")
    res["gap_to_clean"] = res["uar_compressed"] - res["uar_clean_control"]
    res.to_csv(out_dir / "neutralisation_curves.csv", index=False)

    conds = [c for c in CONDITIONS if c != train_condition]
    fig, axes = plt.subplots(1, len(conds), figsize=(6 * len(conds), 4.6), squeeze=False)
    for ax, cond in zip(axes[0], conds):
        g = res[res["condition"] == cond]
        sns.lineplot(data=g, x="n_neutralised", y="uar_compressed", hue="model",
                     marker="o", ax=ax)
        clean = g.groupby("n_neutralised")["uar_clean_control"].mean()
        ax.plot(clean.index, clean.values, ls="--", c="grey", label="clean control (mean)")
        ax.set(title=f"Recovery on {cond}", xlabel="features neutralised to training median",
               ylabel="balanced accuracy")
    fig.tight_layout()
    fig.savefig(out_dir / "neutralisation_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "created_utc": ts,
        "train_condition": train_condition,
        "models": list(predictors),
        "per_condition": {},
    }
    for cond in conds:
        g = res[res["condition"] == cond]
        summary["per_condition"][cond] = {
            "top_shifted_features": [
                {"feature": f, "shift_sd": float(v)} for f, v in shift_tables[cond].head(10).items()
            ],
            "models": {
                name: {
                    "uar_clean": float(gm[gm["n_neutralised"] == 0]["uar_clean_control"].iloc[0]),
                    "uar_compressed": float(gm[gm["n_neutralised"] == 0]["uar_compressed"].iloc[0]),
                    "uar_after_1_masked": float(gm[gm["n_neutralised"] == 1]["uar_compressed"].iloc[0]),
                    "uar_after_3_masked": float(gm[gm["n_neutralised"] == 3]["uar_compressed"].iloc[0]),
                    "best_recovery_uar": float(gm["uar_compressed"].max()),
                }
                for name, gm in g.groupby("model")
            },
        }
    with open(out_dir / "robustness_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(OUTPUTS / "robustness" / "LATEST", "w") as fh:
        fh.write(ts)
    print(f"robustness analysis -> {out_dir}")
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Codec-shift neutralisation analysis.")
    ap.add_argument("--models", nargs="*", default=["random_forest", "xgboost", "decision_tree",
                                                    "logistic_regression", "svm_rbf", "mlp_sklearn"])
    ap.add_argument("--no-ann", action="store_true")
    a = ap.parse_args()
    run(models=tuple(a.models), include_ann=not a.no_ann)
