"""Exploratory data analysis over the extracted feature table.

Produces figures in `outputs/eda/`, machine-readable tables alongside them, and
`outputs/eda/eda_summary.json` holding every number quoted in the written
report, so the report can never drift from the data it describes.

The analysis is organised around four questions:

    A. What is in the corpus?          (labels, speakers, durations, human votes)
    B. Is the feature table healthy?   (missingness, constants, outliers)
    C. Do the features see emotion?    (separability, correlation, structure)
    D. Does the codec move the features? (ref-vs-format drift and effect sizes)
"""

from __future__ import annotations

import json
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats as sps  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.feature_selection import f_classif, mutual_info_classif  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from .datasets import CONDITIONS, condition_frame, family_of, feature_names, load_features
from .metadata import EMOTIONS, EMOTION_NAMES
from .paths import EDA_DIR, ensure_dirs

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = sns.color_palette("colorblind", 6)
EMO_ORDER = EMOTIONS


def _save(fig, name: str) -> str:
    path = EDA_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------- A. corpus

def corpus_overview(df: pd.DataFrame, out: dict) -> None:
    ref = condition_frame(df, "ref", require_ok=False)

    out["corpus"] = {
        "rows_total": int(len(df)),
        "clips": int(df["item_id"].nunique()),
        "speakers": int(df["speaker_id"].nunique()),
        "conditions": sorted(df["condition"].unique().tolist()),
        "sentences": int(df["sentence_code"].nunique()),
        "class_counts": ref["emotion"].value_counts().to_dict(),
        "intensity_counts": ref["intensity"].value_counts().to_dict(),
        "sex_counts_speakers": ref.drop_duplicates("speaker_id")["sex"].value_counts().to_dict(),
        "race_counts_speakers": ref.drop_duplicates("speaker_id")["race"].value_counts().to_dict(),
        "age_speakers": {
            "min": float(ref.drop_duplicates("speaker_id")["age"].min()),
            "max": float(ref.drop_duplicates("speaker_id")["age"].max()),
            "mean": float(ref.drop_duplicates("speaker_id")["age"].mean()),
        },
        "duration_s": {
            "mean": float(ref["duration_s"].mean()),
            "std": float(ref["duration_s"].std()),
            "min": float(ref["duration_s"].min()),
            "max": float(ref["duration_s"].max()),
            "total_hours": float(ref["duration_s"].sum() / 3600),
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sns.countplot(data=ref, x="emotion", order=EMO_ORDER, ax=axes[0, 0], palette=PALETTE, hue="emotion", legend=False)
    axes[0, 0].set_title("Clips per intended emotion")
    sns.countplot(data=ref, x="intensity", order=["LO", "MD", "HI", "XX"], ax=axes[0, 1], color=PALETTE[0])
    axes[0, 1].set_title("Clips per acted intensity\n(XX = intensity unspecified)")
    sns.histplot(data=ref, x="duration_s", bins=50, ax=axes[1, 0], color=PALETTE[2])
    axes[1, 0].set_title("Clip duration (s)")
    sns.boxplot(data=ref, x="emotion", y="duration_s", order=EMO_ORDER, ax=axes[1, 1], palette=PALETTE, hue="emotion", legend=False)
    axes[1, 1].set_title("Duration by emotion")
    out["figures"]["corpus_overview"] = _save(fig, "01_corpus_overview.png")

    # Class balance across the speaker dimension: are some actors missing classes?
    per_spk = ref.pivot_table(index="speaker_id", columns="emotion", values="item_id", aggfunc="count").fillna(0)
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.heatmap(per_spk.T, cmap="viridis", cbar_kws={"label": "clips"}, ax=ax)
    ax.set_title("Clips per speaker x emotion")
    ax.set_xticks([])
    out["figures"]["speaker_class_matrix"] = _save(fig, "02_speaker_class_matrix.png")
    out["corpus"]["clips_per_speaker"] = {
        "min": float(per_spk.sum(axis=1).min()),
        "max": float(per_spk.sum(axis=1).max()),
        "mean": float(per_spk.sum(axis=1).mean()),
    }


def human_ceiling(df: pd.DataFrame, out: dict) -> None:
    """What the crowd achieved from voice alone — the realistic upper bound."""
    ref = condition_frame(df, "ref", require_ok=False).dropna(subset=["human_vote"])
    correct = (ref["human_vote"] == ref["emotion"]).mean()

    cm = pd.crosstab(ref["emotion"], ref["human_vote"]).reindex(index=EMO_ORDER, columns=EMO_ORDER).fillna(0)
    cm_norm = cm.div(cm.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[0], vmin=0, vmax=1)
    axes[0].set_title(f"Human voice-only confusion (row-normalised)\noverall accuracy = {correct:.3f}")
    axes[0].set_xlabel("crowd modal vote")
    axes[0].set_ylabel("intended emotion")
    sns.histplot(data=ref, x="human_agreement", bins=30, ax=axes[1], color=PALETTE[3])
    axes[1].set_title("Crowd agreement per clip (voice only)")
    out["figures"]["human_ceiling"] = _save(fig, "03_human_ceiling.png")

    out["human"] = {
        "voice_only_accuracy": float(correct),
        "mean_agreement": float(ref["human_agreement"].mean()),
        "median_n_ratings": float(ref["human_n_ratings"].median()),
        "tied_votes": int(ref["human_vote_tied"].sum()),
        "per_class_recall": {
            e: float(cm_norm.loc[e, e]) if e in cm_norm.index else np.nan for e in EMO_ORDER
        },
    }


# ------------------------------------------------- B. feature table health

def table_health(df: pd.DataFrame, cols: list[str], out: dict) -> None:
    ok = df["extract_ok"].fillna(False)
    nan_rate = df.loc[ok, cols].isna().mean().sort_values(ascending=False)
    nunique = df.loc[ok, cols].nunique()
    std = df.loc[ok, cols].std()

    z = np.abs(sps.zscore(df.loc[ok, cols], nan_policy="omit"))
    outlier_rate = pd.Series((z > 5).mean(axis=0), index=cols).sort_values(ascending=False)

    health = pd.DataFrame(
        {"nan_rate": nan_rate, "n_unique": nunique, "std": std, "outlier_rate_z5": outlier_rate}
    )
    health["family"] = [family_of(c) for c in health.index]
    health.to_csv(EDA_DIR / "feature_health.csv")

    out["health"] = {
        "extraction_failures": int((~ok).sum()),
        "failure_examples": df.loc[~ok, ["item_id", "condition", "extract_error"]]
        .head(10).to_dict("records"),
        "features": len(cols),
        "features_with_any_nan": int((nan_rate > 0).sum()),
        "features_nan_over_1pct": nan_rate[nan_rate > 0.01].round(4).to_dict(),
        "constant_features": nunique[nunique <= 1].index.tolist(),
        "top_outlier_features": outlier_rate.head(10).round(4).to_dict(),
        "family_sizes": pd.Series([family_of(c) for c in cols]).value_counts().to_dict(),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    sns.barplot(x=nan_rate.head(15).values, y=nan_rate.head(15).index, ax=axes[0], color=PALETTE[1])
    axes[0].set_title("Highest missing-value rates")
    axes[0].set_xlabel("fraction NaN")
    sns.barplot(x=outlier_rate.head(15).values, y=outlier_rate.head(15).index, ax=axes[1], color=PALETTE[4])
    axes[1].set_title("Highest heavy-tail rates (|z| > 5)")
    out["figures"]["table_health"] = _save(fig, "04_table_health.png")


# ---------------------------------------------------- C. emotion structure

def separability(df: pd.DataFrame, cols: list[str], out: dict) -> pd.DataFrame:
    ref = condition_frame(df, "ref")
    X = ref[cols].fillna(ref[cols].median())
    y = ref["emotion"].to_numpy()

    f_stat, p_val = f_classif(X, y)
    mi = mutual_info_classif(X, y, random_state=0)
    rank = pd.DataFrame(
        {"feature": cols, "family": [family_of(c) for c in cols],
         "anova_F": f_stat, "anova_p": p_val, "mutual_info": mi}
    ).sort_values("anova_F", ascending=False)
    rank.to_csv(EDA_DIR / "feature_separability.csv", index=False)

    top = rank.head(20)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(data=top, x="anova_F", y="feature", hue="family", dodge=False, ax=axes[0])
    axes[0].set_title("Top 20 features by ANOVA F (emotion, ref condition)")
    fam_mean = rank.groupby("family")[["anova_F", "mutual_info"]].mean().reset_index()
    sns.barplot(data=fam_mean, x="family", y="mutual_info", ax=axes[1], color=PALETTE[5])
    axes[1].set_title("Mean mutual information with emotion, by family")
    out["figures"]["separability"] = _save(fig, "05_separability.png")

    # Distributions of the four strongest single features.
    best = rank.head(4)["feature"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, feat in zip(axes.ravel(), best):
        sns.violinplot(data=ref, x="emotion", y=feat, order=EMO_ORDER, ax=ax,
                       palette=PALETTE, hue="emotion", legend=False, cut=0)
        ax.set_title(feat)
    fig.suptitle("Strongest individually-discriminative features", y=1.01)
    out["figures"]["top_feature_distributions"] = _save(fig, "06_top_feature_distributions.png")

    out["separability"] = {
        "top20_anova": rank.head(20)[["feature", "family", "anova_F", "mutual_info"]]
        .round(4).to_dict("records"),
        "family_mean_mi": rank.groupby("family")["mutual_info"].mean().round(4).to_dict(),
        "family_mean_F": rank.groupby("family")["anova_F"].mean().round(2).to_dict(),
        "n_significant_bonferroni": int((rank["anova_p"] < 0.05 / len(cols)).sum()),
    }
    return rank


def correlation_structure(df: pd.DataFrame, cols: list[str], rank: pd.DataFrame, out: dict) -> None:
    ref = condition_frame(df, "ref")
    X = ref[cols].fillna(ref[cols].median())
    corr = X.corr().abs()

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().sort_values(ascending=False)
    redundant = pairs[pairs > 0.95]

    top = rank.head(30)["feature"].tolist()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(X[top].corr(), cmap="RdBu_r", center=0, ax=ax, square=True,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation among the 30 most discriminative features")
    out["figures"]["correlation"] = _save(fig, "07_correlation.png")

    out["correlation"] = {
        "pairs_above_0.95": int(len(redundant)),
        "pairs_above_0.99": int((pairs > 0.99).sum()),
        "most_correlated_pairs": [
            {"a": a, "b": b, "r": float(v)} for (a, b), v in pairs.head(10).items()
        ],
        "mean_abs_offdiag_r": float(upper.stack().mean()),
    }


def structure_projection(df: pd.DataFrame, cols: list[str], out: dict, tsne_n: int = 2500) -> None:
    """PCA and t-SNE, coloured by emotion and by format condition."""
    use = df[df["extract_ok"].fillna(False)]
    X_all = use[cols].fillna(use[cols].median())
    Xs = StandardScaler().fit_transform(X_all)

    pca = PCA(n_components=min(50, Xs.shape[1]), random_state=0).fit(Xs)
    ev = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(np.arange(1, len(ev) + 1), np.cumsum(ev), marker="o", ms=3)
    axes[0].axhline(0.95, ls="--", c="grey")
    axes[0].set(xlabel="component", ylabel="cumulative explained variance", title="PCA scree")

    proj = pca.transform(Xs)[:, :2]
    plot_df = pd.DataFrame({"pc1": proj[:, 0], "pc2": proj[:, 1],
                            "emotion": use["emotion"].to_numpy(),
                            "condition": use["condition"].to_numpy()})
    sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="emotion", hue_order=EMO_ORDER,
                    s=6, alpha=0.5, palette=PALETTE, ax=axes[1], linewidth=0)
    axes[1].set_title("PC1-PC2 by emotion")
    sns.scatterplot(data=plot_df, x="pc1", y="pc2", hue="condition", s=6, alpha=0.5,
                    ax=axes[2], linewidth=0)
    axes[2].set_title("PC1-PC2 by format condition")
    out["figures"]["pca"] = _save(fig, "08_pca.png")

    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xs), size=min(tsne_n, len(Xs)), replace=False)
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(
        pca.transform(Xs)[idx, :30]
    )
    t_df = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1],
                         "emotion": use["emotion"].to_numpy()[idx],
                         "condition": use["condition"].to_numpy()[idx]})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(data=t_df, x="x", y="y", hue="emotion", hue_order=EMO_ORDER, s=8,
                    alpha=0.7, palette=PALETTE, ax=axes[0], linewidth=0)
    axes[0].set_title("t-SNE by emotion")
    sns.scatterplot(data=t_df, x="x", y="y", hue="condition", s=8, alpha=0.7,
                    ax=axes[1], linewidth=0)
    axes[1].set_title("t-SNE by format condition")
    out["figures"]["tsne"] = _save(fig, "09_tsne.png")

    out["structure"] = {
        "pcs_for_95pct_variance": int(np.searchsorted(np.cumsum(ev), 0.95) + 1),
        "pc1_variance": float(ev[0]),
        "pc2_variance": float(ev[1]),
        "tsne_sample": int(len(idx)),
    }


def speaker_vs_emotion_variance(df: pd.DataFrame, cols: list[str], out: dict) -> None:
    """How much of each feature's variance is speaker identity vs emotion?

    If speaker dominates, speaker-independent splitting is not optional.
    """
    ref = condition_frame(df, "ref")
    X = ref[cols].fillna(ref[cols].median())
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=cols)

    def eta_sq(groups: pd.Series) -> pd.Series:
        grand = Xs.mean()
        between = Xs.groupby(groups.to_numpy()).apply(
            lambda g: len(g) * (g.mean() - grand) ** 2
        ).sum()
        total = ((Xs - grand) ** 2).sum()
        return between / total

    eta_spk = eta_sq(ref["speaker_id"])
    eta_emo = eta_sq(ref["emotion"])
    comp = pd.DataFrame({"speaker_eta2": eta_spk, "emotion_eta2": eta_emo})
    comp["family"] = [family_of(c) for c in comp.index]
    comp.to_csv(EDA_DIR / "variance_decomposition.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=comp, x="speaker_eta2", y="emotion_eta2", hue="family", s=22, ax=ax)
    lim = max(comp["speaker_eta2"].max(), comp["emotion_eta2"].max()) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", c="grey")
    ax.set(xlabel="variance explained by speaker (eta^2)",
           ylabel="variance explained by emotion (eta^2)",
           title="Speaker identity vs emotion as sources of feature variance")
    out["figures"]["variance_decomposition"] = _save(fig, "10_variance_decomposition.png")

    out["variance"] = {
        "mean_speaker_eta2": float(comp["speaker_eta2"].mean()),
        "mean_emotion_eta2": float(comp["emotion_eta2"].mean()),
        "features_speaker_dominant": int((comp["speaker_eta2"] > comp["emotion_eta2"]).sum()),
        "by_family": comp.groupby("family")[["speaker_eta2", "emotion_eta2"]].mean().round(4).to_dict(),
    }


# ------------------------------------------------------- D. format effects

def format_drift(df: pd.DataFrame, cols: list[str], out: dict) -> pd.DataFrame:
    """Per-feature paired comparison of every condition against `ref`.

    Paired on item id, so this isolates the codec's effect on each feature from
    all between-clip variation.
    """
    ok = df[df["extract_ok"].fillna(False)]
    wide = {c: condition_frame(ok, c).set_index("item_id")[cols] for c in CONDITIONS if c in set(ok["condition"])}
    common = None
    for frame in wide.values():
        common = frame.index if common is None else common.intersection(frame.index)

    rows = []
    for cond, frame in wide.items():
        if cond == "ref":
            continue
        a = wide["ref"].loc[common]
        b = frame.loc[common]
        diff = (b - a)
        pooled = a.std().replace(0, np.nan)
        d = diff.mean() / pooled  # standardised mean difference (paired Cohen's d)
        with np.errstate(all="ignore"):
            t, p = sps.ttest_rel(b, a, nan_policy="omit")
        rows.append(pd.DataFrame({
            "condition": cond, "feature": cols, "family": [family_of(c) for c in cols],
            "mean_ref": a.mean().to_numpy(), "mean_cond": b.mean().to_numpy(),
            "smd": d.to_numpy(), "t": np.asarray(t), "p": np.asarray(p),
            "rel_change": ((b.mean() - a.mean()) / a.mean().abs().replace(0, np.nan)).to_numpy(),
        }))
    drift = pd.concat(rows, ignore_index=True)
    drift["abs_smd"] = drift["smd"].abs()
    drift["significant"] = drift["p"] < 0.05 / len(cols)
    drift.to_csv(EDA_DIR / "format_drift.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    top = (drift[drift["condition"] == "mp4_aac64"].sort_values("abs_smd", ascending=False).head(20))
    sns.barplot(data=top, x="smd", y="feature", hue="family", dodge=False, ax=axes[0])
    axes[0].set_title("Features most shifted by MP4/AAC 64k (paired SMD vs ref)")
    axes[0].axvline(0, c="k", lw=0.8)
    sns.boxplot(data=drift, x="family", y="abs_smd", hue="condition", ax=axes[1])
    axes[1].set_yscale("log")
    axes[1].set_title("Magnitude of codec-induced drift by feature family")
    out["figures"]["format_drift"] = _save(fig, "11_format_drift.png")

    summary = {}
    for cond, grp in drift.groupby("condition"):
        summary[cond] = {
            "median_abs_smd": float(grp["abs_smd"].median()),
            "features_smd_over_0.2": int((grp["abs_smd"] > 0.2).sum()),
            "features_smd_over_0.5": int((grp["abs_smd"] > 0.5).sum()),
            "features_significant": int(grp["significant"].sum()),
            "top_shifted": grp.sort_values("abs_smd", ascending=False)
            .head(10)[["feature", "smd", "rel_change"]].round(4).to_dict("records"),
            "by_family_median_abs_smd": grp.groupby("family")["abs_smd"].median().round(4).to_dict(),
        }
    out["format_drift"] = {"n_items_paired": int(len(common)), "per_condition": summary}
    return drift


def decode_path_check(df: pd.DataFrame, cols: list[str], out: dict) -> None:
    """mp4_aac64 vs roundtrip_wav: same codec output, different container.

    In the parent LLM study this contrast separated codec damage from container
    routing. A feature pipeline decodes both to PCM itself, so any residual
    difference here is pure decode-path arithmetic (the WAV control passes
    through int16 quantisation) and bounds the noise floor of every other
    comparison in this report.
    """
    ok = df[df["extract_ok"].fillna(False)]
    if not {"mp4_aac64", "roundtrip_wav"} <= set(ok["condition"]):
        return
    a = condition_frame(ok, "mp4_aac64").set_index("item_id")
    b = condition_frame(ok, "roundtrip_wav").set_index("item_id")
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]

    pooled = a[cols].std().replace(0, np.nan)
    smd = ((b[cols] - a[cols]).mean() / pooled).abs()
    identical_pcm = float((a["sha256_decoded_pcm"] == b["sha256_decoded_pcm"]).mean())

    out["decode_path"] = {
        "n_items": int(len(common)),
        "bitwise_identical_pcm_fraction": identical_pcm,
        "median_abs_smd": float(smd.median()),
        "max_abs_smd": float(smd.max()),
        "feature_at_max": str(smd.idxmax()),
        "features_smd_over_0.05": int((smd > 0.05).sum()),
    }


# ------------------------------------------------------------------- driver

def run(path: str | None = None, tsne_n: int = 2500) -> dict:
    ensure_dirs()
    df = load_features(path)
    cols = feature_names(df)
    out: dict = {"source_table": df.attrs.get("source"), "figures": {}}

    corpus_overview(df, out)
    human_ceiling(df, out)
    table_health(df, cols, out)
    rank = separability(df, cols, out)
    correlation_structure(df, cols, rank, out)
    structure_projection(df, cols, out, tsne_n=tsne_n)
    speaker_vs_emotion_variance(df, cols, out)
    format_drift(df, cols, out)
    decode_path_check(df, cols, out)

    with open(EDA_DIR / "eda_summary.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"EDA complete -> {EDA_DIR}")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run EDA over the newest feature table.")
    ap.add_argument("--path", default=None)
    ap.add_argument("--tsne-n", type=int, default=2500)
    a = ap.parse_args()
    run(path=a.path, tsne_n=a.tsne_n)
