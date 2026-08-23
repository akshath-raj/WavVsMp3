"""How deep does codec-invariance of the tree structure go?

``per_condition.py`` reports the top three levels of an entropy tree fitted
separately on each coding condition, and finds the root descriptor invariant.
That leaves the obvious question open: the root agreeing is one node, so does
the agreement survive further down, or does the tree reorganise once the
first-order evidence is spent?

This module grows the same tree to ``MAX_DEPTH`` and compares the conditions
node by node, aligning nodes by their *path* from the root (``L``/``R``
strings), which is the only alignment under which two trees fitted on different
data are comparable at all.

The comparison needs a null for the same reason the importance ranking did.
A deep node is fitted on a small, highly conditioned subsample, so its split is
chosen from many near-equal candidates and can flip without the data having
moved. Two floors are therefore measured on exactly the same statistic:

    tie-break floor   refit `ref` against `ref` under a different RNG seed.
                      Same audio, same actors, same hyper-parameters. Any
                      disagreement here is the estimator, not the codec.

    container floor   `mp4_aac64` against `roundtrip_wav`. The same AAC
                      bitstream through two containers, differing only by one
                      int16 requantisation. Informationally empty.

Only agreement that falls below those floors is attributable to coding.

Outputs (under ``outputs/per_condition/depth_<UTC>/``)
    depth_profile.json    per-depth agreement for every pair, plus the floors
    node_table.csv        every internal node of every tree, path-aligned
    divergence.json       where each pair first parts company, by path
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .datasets import CONDITIONS, family_of
from .models import _matrices, evaluate, prepare
from .paths import OUTPUTS, ensure_dirs, timestamp

SEED = 42
# Tie-break floor. sklearn permutes the feature order before searching, so a
# reseed changes only which of several equally-scoring candidates wins. Most
# seeds leave the tree untouched and some do not, so the floor is averaged over
# a panel rather than read off one reseed: with a single alternate seed this
# floor reads anywhere from 0 to 1 displaced nodes purely by luck of the draw.
ALT_SEEDS = (0, 7, 123, 2024, 31337, 99991, 5, 17, 808, 4242)
PER_COND_DIR = OUTPUTS / "per_condition"

# Depth 3 is what the paper printed. Six levels is 63 internal nodes at full
# occupancy, which is far past the point where min_samples_leaf starts pruning
# branches, so the profile covers the whole usable range of the estimator.
MAX_DEPTH = 6
MIN_LEAF = 25

REF = "ref"
CONTAINER_PAIR = ("mp4_aac64", "roundtrip_wav")


def fit_tree(Xtr, ytr, depth: int, seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", DecisionTreeClassifier(criterion="entropy", max_depth=depth,
                                       min_samples_leaf=MIN_LEAF,
                                       class_weight="balanced",
                                       random_state=seed)),
    ]).fit(Xtr, ytr)


def node_map(tree, cols: list[str]) -> dict[str, dict]:
    """Every internal node keyed by its path from the root.

    The path is the only stable identifier across two trees fitted on different
    data: node indices are an artefact of construction order.
    """
    t = tree.tree_
    out: dict[str, dict] = {}

    def walk(node: int, depth: int, path: str) -> None:
        if t.children_left[node] == -1:
            return
        left, right = int(t.children_left[node]), int(t.children_right[node])
        w, wl, wr = (float(t.weighted_n_node_samples[node]),
                     float(t.weighted_n_node_samples[left]),
                     float(t.weighted_n_node_samples[right]))
        h, hl, hr = (float(t.impurity[node]), float(t.impurity[left]),
                     float(t.impurity[right]))
        feat = cols[int(t.feature[node])]
        out[path] = {
            "path": path or "(root)",
            "depth": depth,
            "feature": feat,
            "family": family_of(feat),
            "threshold": float(t.threshold[node]),
            "entropy_bits": h,
            "information_gain_bits": h - (wl / w) * hl - (wr / w) * hr,
            "n_samples": int(t.n_node_samples[node]),
            "w_samples": w,
        }
        walk(left, depth + 1, path + "L")
        walk(right, depth + 1, path + "R")

    walk(0, 0, "")
    return out


def compare(a: dict[str, dict], b: dict[str, dict]) -> dict:
    """Path-aligned agreement between two trees, resolved by depth.

    Reported per depth:
      n_common     paths internal in both trees (the comparable population)
      n_agree      of those, how many select the same descriptor
      n_only_a/b   paths one tree splits and the other has already stopped on
      thr_rel      median relative threshold shift where the descriptor agrees
      ig_abs       median |IG(a) - IG(b)| where the descriptor agrees

    A path present in one tree and absent in the other is a structural
    disagreement of a different kind (the branch terminated early), so it is
    counted separately rather than folded into the agreement rate.
    """
    per_depth: dict[int, dict] = defaultdict(
        lambda: {"n_common": 0, "n_agree": 0, "n_only_a": 0, "n_only_b": 0,
                 "thr_rel": [], "ig_abs": []})

    for path in set(a) | set(b):
        na, nb = a.get(path), b.get(path)
        d = (na or nb)["depth"]
        rec = per_depth[d]
        if na is None:
            rec["n_only_b"] += 1
            continue
        if nb is None:
            rec["n_only_a"] += 1
            continue
        rec["n_common"] += 1
        if na["feature"] == nb["feature"]:
            rec["n_agree"] += 1
            denom = max(abs(na["threshold"]), 1e-12)
            rec["thr_rel"].append(abs(nb["threshold"] - na["threshold"]) / denom)
            rec["ig_abs"].append(abs(nb["information_gain_bits"]
                                    - na["information_gain_bits"]))

    out = {}
    for d in sorted(per_depth):
        r = per_depth[d]
        out[str(d)] = {
            "n_common": r["n_common"],
            "n_agree": r["n_agree"],
            "agreement": (r["n_agree"] / r["n_common"]) if r["n_common"] else None,
            "n_only_a": r["n_only_a"],
            "n_only_b": r["n_only_b"],
            "median_rel_threshold_shift": (float(np.median(r["thr_rel"]))
                                           if r["thr_rel"] else None),
            "median_abs_ig_diff": (float(np.median(r["ig_abs"]))
                                   if r["ig_abs"] else None),
        }
    return out


def first_divergence(a: dict[str, dict], b: dict[str, dict]) -> dict:
    """The shallowest node on each root-to-node path where the two trees differ.

    A path is only reached in both trees if every ancestor agreed, so this walks
    down from the root and stops a branch at its first mismatch. That gives the
    depth to which the two trees are genuinely identical, as opposed to the
    depth at which path-aligned nodes happen to coincide.
    """
    diverged: list[dict] = []
    identical_prefix: list[str] = []

    def walk(path: str) -> None:
        na, nb = a.get(path), b.get(path)
        if na is None and nb is None:
            return
        if na is None or nb is None:
            diverged.append({"path": path or "(root)",
                             "depth": (na or nb)["depth"],
                             "reason": "one tree stopped splitting",
                             "feature_a": na["feature"] if na else None,
                             "feature_b": nb["feature"] if nb else None})
            return
        if na["feature"] != nb["feature"]:
            diverged.append({"path": path or "(root)", "depth": na["depth"],
                             "reason": "different descriptor",
                             "feature_a": na["feature"],
                             "feature_b": nb["feature"]})
            return
        identical_prefix.append(path or "(root)")
        walk(path + "L")
        walk(path + "R")

    walk("")
    depths = [d["depth"] for d in diverged]
    return {
        "n_identical_nodes": len(identical_prefix),
        "identical_to_depth": (min(depths) - 1) if depths else MAX_DEPTH,
        "first_divergences": sorted(diverged, key=lambda r: (r["depth"], r["path"])),
    }


def paired_draws(df, cols, split, conditions, max_depth, n_rep, n_train, rng):
    """Resample the training actors and rebuild the whole profile inside each draw.

    The headline profile comes from one fit per condition, so a single node
    flipping would move ``identical_to_depth`` by a whole level. Repeating the
    comparison over subsampled actor sets gives that statistic a distribution.

    Every condition in a draw is fitted on the *same* actors, so sampling
    variance is differenced away by construction rather than estimated and
    subtracted, exactly as in the paired null of ``per_condition_null.py``.
    """
    train_actors = np.asarray(sorted(split.train))
    per_pair: dict[str, list[int]] = defaultdict(list)

    for r in range(n_rep):
        keep = set(rng.choice(train_actors, size=n_train, replace=False).tolist())
        nodes: dict[str, dict] = {}

        for cond in conditions:
            sub, masks, X, y = _matrices(df, cols, split, cond)
            tr = masks["train"] & sub["speaker_id"].isin(keep).to_numpy()
            Xtr, ytr = X[tr].to_numpy(), y[tr]
            nodes[cond] = node_map(
                fit_tree(Xtr, ytr, max_depth, SEED).named_steps["clf"], cols)
            if cond == REF:
                nodes["_tiebreak"] = node_map(
                    fit_tree(Xtr, ytr, max_depth,
                             int(ALT_SEEDS[r % len(ALT_SEEDS)])).named_steps["clf"],
                    cols)

        for name, (a, b) in pair_spec(conditions).items():
            per_pair[name].append(first_divergence(nodes[a], nodes[b])["identical_to_depth"])
        print(f"  draw {r + 1:2d}/{n_rep}  "
              + "  ".join(f"{k.replace('floor_', '').replace('codec_', '')}={v[-1]}"
                          for k, v in per_pair.items()))

    return {name: {"identical_to_depth_mean": float(np.mean(v)),
                   "identical_to_depth_sd": float(np.std(v, ddof=1)),
                   "identical_to_depth_min": int(np.min(v)),
                   "identical_to_depth_max": int(np.max(v)),
                   "per_draw": [int(x) for x in v]}
            for name, v in per_pair.items()}


def pair_spec(conditions) -> dict[str, tuple[str, str]]:
    return {
        "floor_tiebreak": (REF, "_tiebreak"),
        "floor_container": CONTAINER_PAIR,
        **{f"codec_{c}": (REF, c) for c in conditions if c != REF},
    }


def run(path: str | None = None,
        conditions: tuple[str, ...] = tuple(CONDITIONS),
        max_depth: int = MAX_DEPTH,
        n_rep: int = 0,
        n_train: int = 51) -> dict:
    ensure_dirs()
    ts = timestamp()
    run_dir = PER_COND_DIR / f"depth_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    df, cols, split = prepare(path)
    print(f"feature table: {df.attrs.get('source')}")
    print(f"{len(cols)} features, depth {max_depth}, min_samples_leaf {MIN_LEAF}\n")

    nodes: dict[str, dict[str, dict]] = {}
    acc: dict[str, float] = {}
    rows: list[dict] = []
    tiebreak_panel: list[dict] = []

    for cond in conditions:
        _, masks, X, y = _matrices(df, cols, split, cond)
        Xtr, ytr = X[masks["train"]].to_numpy(), y[masks["train"]]
        Xte, yte = X[masks["test"]].to_numpy(), y[masks["test"]]

        pipe = fit_tree(Xtr, ytr, max_depth, SEED)
        clf = pipe.named_steps["clf"]
        nodes[cond] = node_map(clf, cols)
        acc[cond] = evaluate(pipe, Xte, yte)["balanced_accuracy"]
        print(f"{cond:16s} bal.acc {acc[cond]:.4f}  "
              f"internal nodes {len(nodes[cond]):3d}  leaves {clf.get_n_leaves():3d}")

        for rec in nodes[cond].values():
            rows.append({"condition": cond, **rec})

        # Tie-break floor, averaged over a panel of reseeds on identical data.
        if cond == REF:
            for s in ALT_SEEDS:
                alt = node_map(fit_tree(Xtr, ytr, max_depth, s).named_steps["clf"], cols)
                div = first_divergence(nodes[REF], alt)
                tiebreak_panel.append({
                    "seed": int(s),
                    "identical_to_depth": div["identical_to_depth"],
                    "n_identical_nodes": div["n_identical_nodes"],
                    "n_divergences": len(div["first_divergences"]),
                    "by_depth": compare(nodes[REF], alt),
                })
            worst = min(tiebreak_panel, key=lambda r: r["identical_to_depth"])
            nodes["_tiebreak"] = node_map(
                fit_tree(Xtr, ytr, max_depth, worst["seed"]).named_steps["clf"], cols)
            depths = [r["identical_to_depth"] for r in tiebreak_panel]
            print(f"{'ref (reseeds)':16s} identical to depth "
                  f"median {float(np.median(depths)):.1f}, "
                  f"range {min(depths)}-{max(depths)} over {len(ALT_SEEDS)} seeds")

    pd.DataFrame(rows).to_csv(run_dir / "node_table.csv", index=False)

    pairs = pair_spec(conditions)
    profile, divergence = {}, {}
    for name, (a, b) in pairs.items():
        profile[name] = {"pair": [a, b], "by_depth": compare(nodes[a], nodes[b])}
        divergence[name] = {"pair": [a, b], **first_divergence(nodes[a], nodes[b])}

    print("\n--- descriptor agreement at path-aligned nodes ---")
    header = "  ".join(f"d{d}" for d in range(max_depth))
    print(f"{'comparison':22s} {header}   identical to depth")
    for name, rec in profile.items():
        cells = []
        for d in range(max_depth):
            r = rec["by_depth"].get(str(d))
            cells.append("  n/a" if not r or r["agreement"] is None
                         else f"{r['n_agree']:2d}/{r['n_common']:<2d}")
        print(f"{name:22s} {' '.join(cells)}   {divergence[name]['identical_to_depth']}")

    resampled = None
    if n_rep:
        print(f"\n--- paired actor resampling: {n_rep} draws of "
              f"{n_train}/{len(split.train)} training actors ---")
        resampled = paired_draws(df, cols, split, conditions, max_depth,
                                 n_rep, n_train, np.random.default_rng(SEED))
        print("\n  pair                  identical-to-depth  mean +/- sd  [min, max]")
        for name, r in resampled.items():
            print(f"  {name:22s} {r['identical_to_depth_mean']:.2f} "
                  f"+/- {r['identical_to_depth_sd']:.2f}  "
                  f"[{r['identical_to_depth_min']}, {r['identical_to_depth_max']}]")

    summary = {
        "created_utc": ts,
        "feature_table": str(df.attrs.get("source")),
        "n_features": len(cols),
        "max_depth": max_depth,
        "min_samples_leaf": MIN_LEAF,
        "criterion": "entropy (Shannon, base 2)",
        "seed": SEED,
        "alt_seeds": list(ALT_SEEDS),
        "conditions": list(conditions),
        "balanced_accuracy": acc,
        "n_internal_nodes": {k: len(v) for k, v in nodes.items()},
        "tiebreak_panel": tiebreak_panel,
        "depth_profile": profile,
        "divergence": divergence,
        "resampled": resampled,
        "n_rep": n_rep,
        "n_train_actors": n_train if n_rep else None,
    }
    with open(run_dir / "depth_profile.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(run_dir / "divergence.json", "w") as fh:
        json.dump(divergence, fh, indent=1)

    print(f"\nwrote {run_dir}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=None, help="feature table (default: newest)")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    ap.add_argument("--n-rep", type=int, default=0,
                    help="paired actor-resampling draws (0 = full-data profile only)")
    ap.add_argument("--n-train", type=int, default=51,
                    help="training actors sampled per draw, without replacement")
    a = ap.parse_args()
    run(path=a.path, conditions=tuple(a.conditions), max_depth=a.max_depth,
        n_rep=a.n_rep, n_train=a.n_train)


if __name__ == "__main__":
    main()
