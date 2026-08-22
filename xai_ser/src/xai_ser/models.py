"""Classical (non-deep) classifiers and the cross-format evaluation grid.

Every model is wrapped in a pipeline that imputes missing values with training
medians. Scaling is applied only to the models that need it: tree ensembles are
left in native feature units so that SHAP values come out in interpretable
physical quantities (Hz, dB, energy ratios) rather than z-scores.

Two evaluations are run:

  * **in-format** — train and test on the same condition, the usual benchmark.
  * **cross-format** — train on one condition, test on all four. This is the
    deployment-realistic case (models are trained on clean archives and served
    compressed audio) and it is where the codec question is actually decided.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .datasets import CONDITIONS, aligned_items, feature_names, load_features, make_speaker_split
from .metadata import EMOTIONS
from .paths import MODELS_DIR, ensure_dirs, timestamp

warnings.filterwarnings("ignore")

SEED = 42


@dataclass
class ModelSpec:
    name: str
    estimator: object
    scale: bool = False
    param_dist: dict = field(default_factory=dict)
    note: str = ""


def build_zoo() -> list[ModelSpec]:
    """One representative of every major tabular-classification family."""
    return [
        ModelSpec("dummy_majority", DummyClassifier(strategy="most_frequent"),
                  note="floor: always predicts the largest class"),
        ModelSpec("dummy_stratified", DummyClassifier(strategy="stratified", random_state=SEED),
                  note="floor: samples the training prior"),
        ModelSpec("logistic_regression",
                  LogisticRegression(max_iter=3000, C=1.0, random_state=SEED),
                  scale=True, param_dist={"clf__C": [0.01, 0.1, 1.0, 10.0]},
                  note="linear, fully inspectable coefficients"),
        ModelSpec("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
                  scale=True, note="linear with covariance shrinkage"),
        ModelSpec("gaussian_nb", GaussianNB(), scale=True,
                  note="independence assumption baseline"),
        ModelSpec("knn", KNeighborsClassifier(n_neighbors=15, weights="distance"),
                  scale=True, param_dist={"clf__n_neighbors": [5, 15, 25, 45]},
                  note="instance-based, no global explanation"),
        ModelSpec("decision_tree",
                  DecisionTreeClassifier(max_depth=12, min_samples_leaf=10,
                                         class_weight="balanced", random_state=SEED),
                  param_dist={"clf__max_depth": [6, 8, 12, 20, None],
                              "clf__min_samples_leaf": [1, 5, 10, 25]},
                  note="the directly-readable model"),
        ModelSpec("random_forest",
                  RandomForestClassifier(n_estimators=600, min_samples_leaf=2,
                                         class_weight="balanced_subsample",
                                         n_jobs=-1, random_state=SEED),
                  param_dist={"clf__max_features": ["sqrt", 0.1, 0.3],
                              "clf__min_samples_leaf": [1, 2, 5]}),
        ModelSpec("extra_trees",
                  ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2,
                                       class_weight="balanced", n_jobs=-1, random_state=SEED)),
        ModelSpec("hist_gradient_boosting",
                  HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                                 early_stopping=True, random_state=SEED)),
        ModelSpec("adaboost",
                  AdaBoostClassifier(
                      estimator=DecisionTreeClassifier(max_depth=3, random_state=SEED),
                      n_estimators=300, learning_rate=0.5, random_state=SEED)),
        ModelSpec("svm_rbf", SVC(C=10.0, gamma="scale", probability=True, random_state=SEED),
                  scale=True, param_dist={"clf__C": [1.0, 10.0, 50.0]},
                  note="strong but opaque; needs model-agnostic XAI"),
        ModelSpec("linear_svm", LinearSVC(C=0.1, max_iter=5000, random_state=SEED), scale=True),
        ModelSpec("mlp_sklearn",
                  MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=600,
                                early_stopping=True, random_state=SEED),
                  scale=True, note="shallow net baseline for the PyTorch ANN"),
    ]


def add_boosting_models(zoo: list[ModelSpec]) -> list[ModelSpec]:
    """XGBoost and LightGBM if their native libraries load on this machine."""
    try:
        from xgboost import XGBClassifier

        zoo.append(ModelSpec("xgboost", XGBClassifier(
            n_estimators=600, max_depth=6, learning_rate=0.08, subsample=0.8,
            colsample_bytree=0.6, reg_lambda=1.0, tree_method="hist",
            objective="multi:softprob", num_class=len(EMOTIONS),
            n_jobs=-1, random_state=SEED, eval_metric="mlogloss")))
    except Exception as exc:  # pragma: no cover
        print(f"xgboost unavailable: {exc}")
    try:
        from lightgbm import LGBMClassifier

        zoo.append(ModelSpec("lightgbm", LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
            colsample_bytree=0.6, class_weight="balanced", n_jobs=-1,
            random_state=SEED, verbose=-1)))
    except Exception as exc:  # pragma: no cover
        print(f"lightgbm unavailable: {exc}")
    return zoo


def make_pipeline(spec: ModelSpec) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if spec.scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", spec.estimator))
    return Pipeline(steps)


def evaluate(model, X, y, label_names=EMOTIONS) -> dict:
    pred = model.predict(X)
    res = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(len(label_names))).tolist(),
        "per_class_f1": {
            label_names[i]: float(v)
            for i, v in enumerate(f1_score(y, pred, average=None, labels=range(len(label_names))))
        },
    }
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            res["log_loss"] = float(log_loss(y, proba, labels=range(len(label_names))))
            res["top2_accuracy"] = float(
                np.mean([y[i] in np.argsort(p)[-2:] for i, p in enumerate(proba)])
            )
        except Exception:
            pass
    return res


# --------------------------------------------------------------------------

def prepare(path: str | None = None) -> tuple[pd.DataFrame, list[str], object]:
    df = load_features(path)
    keep = aligned_items(df)
    df = df[df["item_id"].isin(keep)].copy()
    cols = feature_names(df)
    split = make_speaker_split(df, seed=SEED)
    return df, cols, split


def _matrices(df, cols, split, condition):
    sub = df[(df["condition"] == condition)].sort_values("item_id").reset_index(drop=True)
    masks = split.where(sub["speaker_id"])
    y = sub["emotion"].map({e: i for i, e in enumerate(EMOTIONS)}).to_numpy()
    return sub, masks, sub[cols], y


def run(
    path: str | None = None,
    train_conditions: tuple[str, ...] = ("ref",),
    tune: bool = False,
    save_models: bool = True,
) -> dict:
    ensure_dirs()
    ts = timestamp()
    run_dir = MODELS_DIR / ts
    (run_dir / "fitted").mkdir(parents=True, exist_ok=True)

    df, cols, split = prepare(path)
    zoo = add_boosting_models(build_zoo())

    # Test matrices for every condition, built once and shared. Row order is
    # item-aligned, so cross-format rows correspond to the same clips.
    test_sets = {}
    for cond in CONDITIONS:
        sub, masks, X, y = _matrices(df, cols, split, cond)
        test_sets[cond] = (X[masks["test"]], y[masks["test"]], sub[masks["test"]])

    results, cross_rows = {}, []
    for train_cond in train_conditions:
        sub, masks, X, y = _matrices(df, cols, split, train_cond)
        Xtr, ytr = X[masks["train"]], y[masks["train"]]
        Xva, yva = X[masks["val"]], y[masks["val"]]
        groups = sub.loc[masks["train"], "speaker_id"].to_numpy()

        print(f"\n=== train on {train_cond}: {len(Xtr)} train / {len(Xva)} val "
              f"/ {len(test_sets[train_cond][0])} test rows, {len(cols)} features ===")

        for spec in zoo:
            pipe = make_pipeline(spec)
            t0 = time.time()
            if tune and spec.param_dist:
                search = RandomizedSearchCV(
                    pipe, spec.param_dist, n_iter=min(6, np.prod([len(v) for v in spec.param_dist.values()])),
                    cv=GroupKFold(n_splits=3), scoring="balanced_accuracy",
                    n_jobs=1, random_state=SEED, refit=True,
                )
                search.fit(Xtr, ytr, groups=groups)
                fitted, best = search.best_estimator_, search.best_params_
            else:
                fitted, best = pipe.fit(Xtr, ytr), {}
            fit_s = time.time() - t0

            key = f"{train_cond}|{spec.name}"
            entry = {
                "model": spec.name, "train_condition": train_cond, "note": spec.note,
                "fit_seconds": round(fit_s, 2), "best_params": best,
                "val": evaluate(fitted, Xva, yva),
            }
            for test_cond, (Xte, yte, _) in test_sets.items():
                m = evaluate(fitted, Xte, yte)
                entry[f"test_{test_cond}"] = m
                cross_rows.append({
                    "model": spec.name, "train_condition": train_cond,
                    "test_condition": test_cond, **{
                        k: v for k, v in m.items() if not isinstance(v, (list, dict))
                    },
                })
            results[key] = entry

            in_fmt = entry[f"test_{train_cond}"]["balanced_accuracy"]
            print(f"  {spec.name:<24} val UAR {entry['val']['balanced_accuracy']:.3f} | "
                  f"test UAR {in_fmt:.3f} | {fit_s:5.1f}s")

            if save_models:
                joblib.dump(fitted, run_dir / "fitted" / f"{train_cond}__{spec.name}.joblib")

    cross = pd.DataFrame(cross_rows)
    cross.to_csv(run_dir / "cross_format.csv", index=False)

    leaderboard = (
        cross[(cross["train_condition"] == cross["test_condition"])]
        .sort_values("balanced_accuracy", ascending=False)
        .reset_index(drop=True)
    )
    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)

    meta = {
        "created_utc": ts,
        "source_table": df.attrs.get("source"),
        "n_features": len(cols),
        "feature_names": cols,
        "split": {"train": list(split.train), "val": list(split.val), "test": list(split.test)},
        "n_items": int(df["item_id"].nunique()),
        "train_conditions": list(train_conditions),
        "tuned": tune,
        "results": results,
    }
    with open(run_dir / "results.json", "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    with open(MODELS_DIR / "LATEST", "w") as fh:
        fh.write(ts)

    print(f"\nsaved -> {run_dir}")
    print(leaderboard.head(10)[["model", "test_condition", "accuracy", "balanced_accuracy", "macro_f1"]].to_string(index=False))
    return meta


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train the classical model zoo.")
    ap.add_argument("--path", default=None)
    ap.add_argument("--train-conditions", nargs="*", default=["ref"])
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--no-save-models", action="store_true")
    a = ap.parse_args()
    run(path=a.path, train_conditions=tuple(a.train_conditions), tune=a.tune,
        save_models=not a.no_save_models)
