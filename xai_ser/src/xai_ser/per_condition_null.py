"""Null calibration for the per-condition importance comparison.

A decision tree's importance ranking is not a stable object. Two features that
split almost equally well can swap, and one swap near the root reroutes every
descendant, so a ranking can move a long way without the data having moved at
all. Any claim of the form "the codec changed which features the model uses"
therefore has to clear a floor measured under manipulations that carry no codec
information.

Three floors are measured, all on the uncompressed condition:

``seed``
    The same training matrix refitted under different random tie-breaking.
    Isolates estimator variance with the data held exactly fixed.

``bootstrap``
    Training actors resampled with replacement, tie-breaking held fixed.
    Isolates sampling variance with the estimator held fixed.

``container``
    ``mp4_aac64`` against ``roundtrip_wav`` -- the same codec output routed
    through two containers. Not a floor for estimator variance but the
    informationally-empty manipulation in the coding domain, and the closest
    analogue of the codec contrast that carries no codec difference.

The codec effect is reported against all three.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier

from .datasets import CONDITIONS
from .models import _matrices, prepare
from .paths import OUTPUTS, ensure_dirs, timestamp

SEED = 42
TOPK = 25
PER_COND_DIR = OUTPUTS / "per_condition"


def dt(random_state: int = SEED) -> DecisionTreeClassifier:
    """The configuration used in `per_condition.run`, restated so the null is
    measured on the same estimator the effect is measured on."""
    return DecisionTreeClassifier(criterion="entropy", max_depth=12,
                                  min_samples_leaf=10, class_weight="balanced",
                                  random_state=random_state)


def fit_importance(X: np.ndarray, y: np.ndarray, random_state: int = SEED) -> np.ndarray:
    imp = SimpleImputer(strategy="median")
    return dt(random_state).fit(imp.fit_transform(X), y).feature_importances_


def agree(a: np.ndarray, b: np.ndarray, cols: list[str], k: int = TOPK) -> dict:
    """Rank agreement between two importance vectors.

    Spearman over all 436 columns is dominated by the ~130 both trees leave at
    exactly zero, which inflates it. The reported correlation is restricted to
    columns at least one of the two trees actually used; the all-column value is
    kept alongside it so the inflation is visible rather than hidden.
    """
    used = (a > 0) | (b > 0)
    rho_used, p_used = spearmanr(a[used], b[used])
    rho_all, _ = spearmanr(a, b)
    ta = set(np.asarray(cols)[np.argsort(a)[::-1][:k]])
    tb = set(np.asarray(cols)[np.argsort(b)[::-1][:k]])
    return {
        "rho_used": float(rho_used),
        "p_used": float(p_used),
        "rho_all": float(rho_all),
        "n_used": int(used.sum()),
        f"top{k}_overlap": len(ta & tb),
    }


def summarise(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "min": float(a.min()), "max": float(a.max()),
            "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
            "n": int(a.size)}


def run(path: str | None = None, n_rep: int = 20) -> dict:
    ensure_dirs()
    ts = timestamp()
    run_dir = PER_COND_DIR / f"null_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    df, cols, split = prepare(path)
    cols_arr = list(cols)

    mats = {}
    for cond in CONDITIONS:
        sub, masks, X, y = _matrices(df, cols, split, cond)
        mats[cond] = (X[masks["train"]].to_numpy(), y[masks["train"]],
                      sub.loc[masks["train"], "speaker_id"].to_numpy())
    Xr, yr, spk = mats["ref"]
    print(f"ref train matrix {Xr.shape}, {len(np.unique(spk))} actors")

    # ---------------------------------------------------------- floor 1: seed
    print(f"seed floor: {n_rep} refits, data fixed")
    seed_imps = [fit_importance(Xr, yr, random_state=1000 + i) for i in range(n_rep)]
    seed_pairs = [agree(seed_imps[i], seed_imps[j], cols_arr)
                  for i in range(n_rep) for j in range(i + 1, n_rep)]

    # ----------------------------------------------------- floor 2: bootstrap
    print(f"bootstrap floor: {n_rep} actor resamples, tie-breaking fixed")
    rng = np.random.default_rng(SEED)
    actors = np.unique(spk)
    boot_imps = []
    for _ in range(n_rep):
        draw = rng.choice(actors, size=len(actors), replace=True)
        idx = np.concatenate([np.where(spk == a)[0] for a in draw])
        boot_imps.append(fit_importance(Xr[idx], yr[idx], random_state=SEED))
    boot_pairs = [agree(boot_imps[i], boot_imps[j], cols_arr)
                  for i in range(n_rep) for j in range(i + 1, n_rep)]

    # -------------------------------------------------------------- effects
    print("condition effects")
    cond_imp = {c: fit_importance(*mats[c][:2], random_state=SEED) for c in CONDITIONS}
    effects = {c: agree(cond_imp["ref"], cond_imp[c], cols_arr)
               for c in CONDITIONS if c != "ref"}
    container = agree(cond_imp["mp4_aac64"], cond_imp["roundtrip_wav"], cols_arr)

    def floor_block(pairs: list[dict]) -> dict:
        return {"rho_used": summarise([p["rho_used"] for p in pairs]),
                f"top{TOPK}_overlap": summarise([p[f"top{TOPK}_overlap"] for p in pairs]),
                "n_pairs": len(pairs)}

    floors = {"seed": floor_block(seed_pairs), "bootstrap": floor_block(boot_pairs),
              "container": container}

    # Does each codec effect sit below the seed/bootstrap floors?
    verdict = {}
    for c, e in effects.items():
        row = {}
        for fl in ("seed", "bootstrap"):
            f_rho = floors[fl]["rho_used"]
            f_ov = floors[fl][f"top{TOPK}_overlap"]
            row[fl] = {
                "rho_below_floor_p05": bool(e["rho_used"] < f_rho["p05"]),
                "rho_z": float((e["rho_used"] - f_rho["mean"]) / f_rho["sd"])
                if f_rho["sd"] > 0 else None,
                "overlap_below_floor_p05": bool(e[f"top{TOPK}_overlap"] < f_ov["p05"]),
                "overlap_z": float((e[f"top{TOPK}_overlap"] - f_ov["mean"]) / f_ov["sd"])
                if f_ov["sd"] > 0 else None,
            }
        row["vs_container"] = {
            "rho_below_container": bool(e["rho_used"] < container["rho_used"]),
            "overlap_below_container": bool(
                e[f"top{TOPK}_overlap"] < container[f"top{TOPK}_overlap"]),
        }
        verdict[c] = row

    out = {
        "created_utc": ts,
        "feature_table": str(df.attrs.get("source")),
        "estimator": "DecisionTreeClassifier(criterion=entropy, max_depth=12, "
                     "min_samples_leaf=10, class_weight=balanced)",
        "n_replicates": n_rep,
        "top_k": TOPK,
        "floors": floors,
        "effects": effects,
        "verdict": verdict,
    }
    with open(run_dir / "null_calibration.json", "w") as fh:
        json.dump(out, fh, indent=1)
    pd.DataFrame(
        [{"pair": "seed", **p} for p in seed_pairs]
        + [{"pair": "bootstrap", **p} for p in boot_pairs]
    ).to_csv(run_dir / "null_pairs.csv", index=False)

    print(f"\nseed floor      rho {floors['seed']['rho_used']['mean']:.3f} "
          f"[{floors['seed']['rho_used']['p05']:.3f}, {floors['seed']['rho_used']['p95']:.3f}]"
          f"  top{TOPK} {floors['seed'][f'top{TOPK}_overlap']['mean']:.1f}")
    print(f"bootstrap floor rho {floors['bootstrap']['rho_used']['mean']:.3f} "
          f"[{floors['bootstrap']['rho_used']['p05']:.3f}, "
          f"{floors['bootstrap']['rho_used']['p95']:.3f}]"
          f"  top{TOPK} {floors['bootstrap'][f'top{TOPK}_overlap']['mean']:.1f}")
    print(f"container null  rho {container['rho_used']:.3f}  "
          f"top{TOPK} {container[f'top{TOPK}_overlap']}")
    for c, e in effects.items():
        print(f"{c:15s} rho {e['rho_used']:+.3f}  top{TOPK} {e[f'top{TOPK}_overlap']}")
    print(f"\nwrote {run_dir}")
    return out


def paired(path: str | None = None, n_rep: int = 20, frac: float = 0.85) -> dict:
    """The comparison the marginal floors cannot make.

    Sampling variance and codec effect are confounded when the floor is measured
    on one training set and the effect on another. Here both are measured inside
    the *same* resampled training set: on draw ``r`` the tree is fitted on the
    uncompressed audio of those actors twice (different tie-breaking) and on
    their coded audio once. The within-condition pair gives the floor for that
    draw, the across-condition pair gives the effect, and the two are compared
    pairwise across draws.

    Actors are subsampled without replacement, so no actor is duplicated and the
    floor is not inflated by pseudo-replication.
    """
    from scipy.stats import wilcoxon

    ensure_dirs()
    ts = timestamp()
    run_dir = PER_COND_DIR / f"paired_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    df, cols, split = prepare(path)
    cols_arr = list(cols)
    mats = {}
    for cond in CONDITIONS:
        sub, masks, X, y = _matrices(df, cols, split, cond)
        mats[cond] = (X[masks["train"]].to_numpy(), y[masks["train"]],
                      sub.loc[masks["train"], "speaker_id"].to_numpy())
    spk = mats["ref"][2]
    actors = np.unique(spk)
    k = int(round(frac * len(actors)))
    rng = np.random.default_rng(SEED)
    print(f"{n_rep} draws of {k}/{len(actors)} actors, without replacement")

    rows = []
    for r in range(n_rep):
        keep = rng.choice(actors, size=k, replace=False)
        idx = np.where(np.isin(spk, keep))[0]
        imp = {}
        Xr, yr, _ = mats["ref"]
        imp["ref_a"] = fit_importance(Xr[idx], yr[idx], random_state=2000 + r)
        imp["ref_b"] = fit_importance(Xr[idx], yr[idx], random_state=9000 + r)
        for cond in CONDITIONS:
            if cond == "ref":
                continue
            Xc, yc, _ = mats[cond]
            imp[cond] = fit_importance(Xc[idx], yc[idx], random_state=2000 + r)
        base = agree(imp["ref_a"], imp["ref_b"], cols_arr)
        row = {"draw": r, "floor_rho": base["rho_used"],
               "floor_overlap": base[f"top{TOPK}_overlap"]}
        for cond in CONDITIONS:
            if cond == "ref":
                continue
            a = agree(imp["ref_a"], imp[cond], cols_arr)
            row[f"{cond}_rho"] = a["rho_used"]
            row[f"{cond}_overlap"] = a[f"top{TOPK}_overlap"]
        # container contrast inside the same draw
        c = agree(imp["mp4_aac64"], imp["roundtrip_wav"], cols_arr)
        row["container_rho"] = c["rho_used"]
        row["container_overlap"] = c[f"top{TOPK}_overlap"]
        rows.append(row)
        print(f"  draw {r:2d}  floor {base['rho_used']:+.3f}/"
              f"{base[f'top{TOPK}_overlap']:2d}   "
              + "  ".join(f"{cd.split('_')[0]} {row[f'{cd}_rho']:+.3f}/"
                          f"{row[f'{cd}_overlap']:2d}"
                          for cd in CONDITIONS if cd != "ref"))

    tab = pd.DataFrame(rows)
    tab.to_csv(run_dir / "paired_draws.csv", index=False)

    tests = {}
    for cond in [c for c in CONDITIONS if c != "ref"] + ["container"]:
        for metric in ("rho", "overlap"):
            a = tab[f"floor_{metric}"].to_numpy(float)
            b = tab[f"{cond}_{metric}"].to_numpy(float)
            diff = a - b
            if np.allclose(diff, 0):
                stat, p = float("nan"), 1.0
            else:
                stat, p = wilcoxon(a, b)
            tests[f"{cond}_{metric}"] = {
                "floor_mean": float(a.mean()), "effect_mean": float(b.mean()),
                "mean_drop": float(diff.mean()), "sd_drop": float(diff.std(ddof=1)),
                "wilcoxon_W": float(stat), "p": float(p),
                "n_draws": int(len(a)),
                "cohen_dz": float(diff.mean() / diff.std(ddof=1))
                if diff.std(ddof=1) > 0 else None,
            }

    out = {"created_utc": ts, "n_draws": n_rep, "actor_fraction": frac,
           "n_actors_per_draw": k, "top_k": TOPK,
           "design": "within-draw: floor = ref vs ref (different tie-breaking); "
                     "effect = ref vs condition, same actors, same items",
           "tests": tests}
    with open(run_dir / "paired_null.json", "w") as fh:
        json.dump(out, fh, indent=1)

    print("\n--- paired within-draw comparison ---")
    for kk, v in tests.items():
        print(f"{kk:28s} floor {v['floor_mean']:+.3f}  effect {v['effect_mean']:+.3f}  "
              f"drop {v['mean_drop']:+.3f}  p={v['p']:.2e}")
    print(f"\nwrote {run_dir}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", default=None)
    ap.add_argument("--n-rep", type=int, default=20)
    ap.add_argument("--paired", action="store_true",
                    help="run the within-draw paired comparison instead")
    a = ap.parse_args()
    if a.paired:
        paired(path=a.path, n_rep=a.n_rep)
    else:
        run(path=a.path, n_rep=a.n_rep)


if __name__ == "__main__":
    main()
