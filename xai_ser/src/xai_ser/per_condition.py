"""Matched-format training: one model set fitted *per coding condition*.

Every other module in this project trains on uncompressed audio and serves the
compressed conditions, which measures train/serve mismatch. This module asks a
different question: when a model is allowed to *learn* from coded audio, does it
learn from the same evidence?

For each of the four conditions the same speaker-independent split is reused, so
a condition-to-condition comparison of what a tree splits on differs only in the
codec. Trees are grown with the Shannon-entropy criterion rather than Gini so
that the split rule reported in the paper is the one the paper's equations
describe.

Outputs (under ``outputs/per_condition/<UTC>/``)
    summary.json            everything below, in one file
    leaderboard_matched.csv model x condition, trained and tested in-condition
    dt_importance.csv       decision-tree importance, one column per condition
    dt_top_splits.json      the top three levels of each condition's tree,
                            with node entropies and information gains
    importance_stability.json  rank agreement of importance across conditions
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .datasets import CONDITIONS, family_of
from .metadata import EMOTIONS
from .models import _matrices, evaluate, prepare
from .paths import OUTPUTS, ensure_dirs, timestamp

SEED = 42
PER_COND_DIR = OUTPUTS / "per_condition"

# Depth of the tree whose splits are reported verbatim in the paper. Three
# levels is seven internal nodes -- enough to show what the codec changes,
# short enough to print.
READABLE_DEPTH = 3


@dataclass
class Spec:
    name: str
    estimator: object
    scale: bool = False
    note: str = ""
    extra: dict = field(default_factory=dict)


def build_zoo() -> list[Spec]:
    """A deliberately small zoo: one readable model, one ensemble, one linear,
    one kernel, one net. Enough to show that the split-vs-multiply distinction
    survives matched-format training."""
    zoo = [
        Spec("decision_tree_entropy",
             DecisionTreeClassifier(criterion="entropy", max_depth=12,
                                    min_samples_leaf=10, class_weight="balanced",
                                    random_state=SEED),
             note="readable model; Shannon-entropy splitting"),
        Spec("decision_tree_readable",
             DecisionTreeClassifier(criterion="entropy", max_depth=READABLE_DEPTH,
                                    min_samples_leaf=25, class_weight="balanced",
                                    random_state=SEED),
             note=f"depth-{READABLE_DEPTH} tree printed in the paper"),
        Spec("random_forest",
             RandomForestClassifier(n_estimators=600, criterion="entropy",
                                    min_samples_leaf=2,
                                    class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=SEED),
             note="ensemble of entropy trees"),
        Spec("logistic_regression",
             LogisticRegression(max_iter=3000, C=1.0, random_state=SEED),
             scale=True, note="linear, inspectable coefficients"),
        Spec("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
             scale=True, note="linear with covariance shrinkage"),
        Spec("svm_rbf", SVC(C=10.0, gamma="scale", probability=True, random_state=SEED),
             scale=True, note="kernel machine"),
        Spec("mlp_sklearn",
             MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=600,
                           early_stopping=True, random_state=SEED),
             scale=True, note="shallow network"),
    ]
    try:
        from lightgbm import LGBMClassifier

        zoo.insert(3, Spec("lightgbm", LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
            colsample_bytree=0.6, class_weight="balanced", n_jobs=-1,
            random_state=SEED, verbose=-1), note="boosted ensemble"))
    except Exception as exc:  # pragma: no cover
        print(f"lightgbm unavailable: {exc}")
    return zoo


def make_pipeline(spec: Spec) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if spec.scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", spec.estimator))
    return Pipeline(steps)


# ---------------------------------------------------------------- tree reading

def describe_splits(tree, cols: list[str], max_depth: int = READABLE_DEPTH) -> list[dict]:
    """Walk the top `max_depth` levels and report each internal node's split.

    Information gain is the standard weighted form,

        IG(node) = H(node) - (w_L/w) H(L) - (w_R/w) H(R),

    with H in bits (sklearn's entropy criterion uses log base 2) and w the
    weighted sample count at the node. Class weights are `balanced`, so the
    weighted counts are reported alongside the raw ones and the arithmetic in
    the paper reproduces from the weighted column.
    """
    t = tree.tree_
    out: list[dict] = []

    def walk(node: int, depth: int, path: str) -> None:
        if depth > max_depth or t.children_left[node] == -1:
            return
        left, right = int(t.children_left[node]), int(t.children_right[node])
        w, wl, wr = (float(t.weighted_n_node_samples[node]),
                     float(t.weighted_n_node_samples[left]),
                     float(t.weighted_n_node_samples[right]))
        h, hl, hr = (float(t.impurity[node]), float(t.impurity[left]),
                     float(t.impurity[right]))
        gain = h - (wl / w) * hl - (wr / w) * hr
        counts = np.asarray(t.value[node]).ravel()
        out.append({
            "node": int(node),
            "depth": int(depth),
            "path": path,
            "feature": cols[int(t.feature[node])],
            "family": family_of(cols[int(t.feature[node])]),
            "threshold": float(t.threshold[node]),
            "n_samples": int(t.n_node_samples[node]),
            "w_samples": w,
            "entropy_bits": h,
            "entropy_left": hl,
            "entropy_right": hr,
            "w_left": wl,
            "w_right": wr,
            "n_left": int(t.n_node_samples[left]),
            "n_right": int(t.n_node_samples[right]),
            "information_gain_bits": gain,
            # `value` is normalised per node in sklearn >= 1.3; renormalise to
            # weighted counts so the class distribution is readable.
            "class_distribution": {
                EMOTIONS[i]: round(float(counts[i] / counts.sum()), 4)
                for i in range(len(EMOTIONS))
            },
        })
        walk(left, depth + 1, path + "L")
        walk(right, depth + 1, path + "R")

    walk(0, 0, "")
    return out


def top_importances(model, cols: list[str], k: int = 25) -> list[dict]:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return []
    order = np.argsort(imp)[::-1][:k]
    return [{"rank": int(r + 1), "feature": cols[int(i)], "family": family_of(cols[int(i)]),
             "importance": float(imp[int(i)])}
            for r, i in enumerate(order) if imp[int(i)] > 0]


# ---------------------------------------------------------------------- driver

def run(path: str | None = None, conditions: tuple[str, ...] = tuple(CONDITIONS)) -> dict:
    ensure_dirs()
    ts = timestamp()
    run_dir = PER_COND_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    df, cols, split = prepare(path)
    print(f"feature table: {df.attrs.get('source')}")
    print(f"{len(cols)} features, split {len(split.train)}/{len(split.val)}/{len(split.test)} actors")

    rows: list[dict] = []
    splits_by_cond: dict[str, list[dict]] = {}
    importance_by_cond: dict[str, dict[str, float]] = {}
    dt_metrics: dict[str, dict] = {}
    full_importance: dict[str, np.ndarray] = {}

    for cond in conditions:
        sub, masks, X, y = _matrices(df, cols, split, cond)
        Xtr, ytr = X[masks["train"]].to_numpy(), y[masks["train"]]
        Xte, yte = X[masks["test"]].to_numpy(), y[masks["test"]]
        print(f"\n=== {cond}: train {Xtr.shape}, test {Xte.shape} ===")

        for spec in build_zoo():
            pipe = make_pipeline(spec)
            pipe.fit(Xtr, ytr)
            res = evaluate(pipe, Xte, yte)
            rows.append({"condition": cond, "model": spec.name, "note": spec.note,
                         **{k: v for k, v in res.items() if k != "confusion_matrix"}})
            print(f"  {spec.name:24s} bal.acc {res['balanced_accuracy']:.4f}")

            clf = pipe.named_steps["clf"]
            if spec.name == "decision_tree_readable":
                splits_by_cond[cond] = describe_splits(clf, cols)
            if spec.name == "decision_tree_entropy":
                dt_metrics[cond] = res
                imp = clf.feature_importances_
                full_importance[cond] = imp
                importance_by_cond[cond] = {
                    d["feature"]: d["importance"] for d in top_importances(clf, cols, k=25)
                }

    lb = pd.DataFrame(rows)
    lb.to_csv(run_dir / "leaderboard_matched.csv", index=False)

    imp_df = pd.DataFrame({c: full_importance[c] for c in conditions}, index=cols)
    imp_df.to_csv(run_dir / "dt_importance.csv")

    # cross-condition agreement of the decision tree's importance ranking
    stability: dict[str, dict] = {}
    ref = full_importance[conditions[0]]
    ref_top = set(imp_df[conditions[0]].nlargest(25).index)
    for cond in conditions[1:]:
        rho, p = spearmanr(ref, full_importance[cond])
        cur_top = set(imp_df[cond].nlargest(25).index)
        # restrict rho to features either condition actually uses; a rank
        # correlation over 400 shared zeros is not informative
        used = (ref > 0) | (full_importance[cond] > 0)
        rho_used, p_used = spearmanr(ref[used], full_importance[cond][used])
        stability[cond] = {
            "spearman_rho_all_features": float(rho),
            "spearman_p_all_features": float(p),
            "n_features_used_by_either": int(used.sum()),
            "spearman_rho_used_features": float(rho_used),
            "spearman_p_used_features": float(p_used),
            "top25_overlap": len(ref_top & cur_top),
            "entered_top25": sorted(cur_top - ref_top),
            "left_top25": sorted(ref_top - cur_top),
            "root_feature": splits_by_cond[cond][0]["feature"],
            "root_feature_ref": splits_by_cond[conditions[0]][0]["feature"],
            "root_changed": splits_by_cond[cond][0]["feature"]
                            != splits_by_cond[conditions[0]][0]["feature"],
        }

    with open(run_dir / "dt_top_splits.json", "w") as fh:
        json.dump(splits_by_cond, fh, indent=1)
    with open(run_dir / "importance_stability.json", "w") as fh:
        json.dump(stability, fh, indent=1)

    summary = {
        "created_utc": ts,
        "feature_table": str(df.attrs.get("source")),
        "n_features": len(cols),
        "conditions": list(conditions),
        "split": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "readable_depth": READABLE_DEPTH,
        "criterion": "entropy (Shannon, base 2)",
        "decision_tree_metrics": dt_metrics,
        "top_importances": importance_by_cond,
        "top_splits": splits_by_cond,
        "importance_stability": stability,
    }
    with open(run_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(PER_COND_DIR / "LATEST", "w") as fh:
        fh.write(ts)

    print(f"\nwrote {run_dir}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=None, help="feature table (default: newest)")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    a = ap.parse_args()
    run(path=a.path, conditions=tuple(a.conditions))


if __name__ == "__main__":
    main()
