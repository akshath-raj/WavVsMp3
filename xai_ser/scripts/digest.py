#!/usr/bin/env python
"""Print a compact digest of every stage's results.

The stage modules each write a verbose JSON summary; this pulls out the numbers
that actually go into the findings write-up so they can be read at a glance.
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from xai_ser.paths import EDA_DIR, MODELS_DIR, XAI_DIR


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def models_digest() -> None:
    ts = (MODELS_DIR / "LATEST").read_text().strip()
    run = MODELS_DIR / ts
    cross = pd.read_csv(run / "cross_format.csv")

    section(f"MODEL LEADERBOARD (in-format, train=test condition) — {ts}")
    lb = cross[cross["train_condition"] == cross["test_condition"]]
    print(lb.sort_values("balanced_accuracy", ascending=False)[
        ["model", "train_condition", "accuracy", "balanced_accuracy", "macro_f1", "cohen_kappa"]
    ].round(4).to_string(index=False))

    section("CROSS-FORMAT: trained on ref, tested on each condition (balanced accuracy)")
    piv = cross[cross["train_condition"] == "ref"].pivot(
        index="model", columns="test_condition", values="balanced_accuracy")
    piv["drop_mp3"] = piv["mp3_64"] - piv["ref"]
    piv["drop_mp4"] = piv["mp4_aac64"] - piv["ref"]
    print(piv.round(4).sort_values("ref", ascending=False).to_string())

    if (cross["train_condition"] == "mp4_aac64").any():
        section("CROSS-FORMAT: trained on mp4_aac64 (matched-format training)")
        piv2 = cross[cross["train_condition"] == "mp4_aac64"].pivot(
            index="model", columns="test_condition", values="balanced_accuracy")
        print(piv2.round(4).sort_values("mp4_aac64", ascending=False).to_string())


def ann_digest() -> None:
    base = MODELS_DIR / "ann"
    if not (base / "LATEST").exists():
        return
    ts = (base / "LATEST").read_text().strip()
    blob = json.load(open(base / ts / "metrics.json"))
    section(f"ANN — {ts}")
    print(f"best val UAR {blob['best_val_uar']:.4f}, epochs {blob['epochs_run']}")
    for k, v in blob["metrics"].items():
        print(f"  {k:<22} acc {v['accuracy']:.4f}  UAR {v['balanced_accuracy']:.4f}  "
              f"macroF1 {v['macro_f1']:.4f}")


def xai_digest() -> None:
    if not (XAI_DIR / "LATEST").exists():
        return
    ts = (XAI_DIR / "LATEST").read_text().strip()
    rep = json.load(open(XAI_DIR / ts / "xai_summary.json"))
    section(f"TABULAR XAI — {ts}")
    for name, e in rep["models"].items():
        print(f"\n--- {name}")
        print("  SHAP top10:", [r["feature"] for r in e.get("shap_top15", [])[:10]])
        print("  perm top10:", [r["feature"] for r in e.get("permutation_top15", [])[:10]])
        print("  SHAP family share:", e.get("shap_family_share"))
        print("  method agreement:", e.get("lime_vs_shap"))
        print("  surrogate fidelity:", round(e.get("surrogate_fidelity", float('nan')), 4))
        for cond, s in e.get("attribution_stability", {}).items():
            print(f"  stability {cond}: rho={s['spearman_rho']:.3f} "
                  f"top25 overlap={s['top25_overlap']} L1 shift={s['l1_shift']:.3f}")


def deep_digest() -> None:
    base = XAI_DIR / "deep"
    if not (base / "LATEST").exists():
        return
    ts = (base / "LATEST").read_text().strip()
    rep = json.load(open(base / ts / "deep_xai_summary.json"))
    section(f"DEEP XAI — {ts}")
    for m, e in rep["methods"].items():
        print(f"  {m:<22} top5 {[r['feature'] for r in e['top15'][:5]]}")
    print("\n  family share (IG):", rep["methods"]["integrated_gradients"]["family_share"])
    print("\n  sanity (trained vs random):",
          {k: round(v["spearman_trained_vs_random"], 3)
           for k, v in rep["model_randomisation_sanity"].items()})
    print("\n  IG stability across formats:",
          {k: {"rho": round(v["spearman_rho"], 3), "top25": v["top25_overlap"]}
           for k, v in rep["attribution_stability_ig"].items()})


def eda_digest() -> None:
    p = EDA_DIR / "eda_summary.json"
    if not p.exists():
        return
    d = json.load(open(p))
    section("EDA HEADLINES")
    print(f"clips {d['corpus']['clips']}, speakers {d['corpus']['speakers']}, "
          f"rows {d['corpus']['rows_total']}, {d['corpus']['duration_s']['total_hours']:.2f} h")
    print(f"human voice-only accuracy {d['human']['voice_only_accuracy']:.4f}")
    print(f"speaker eta2 {d['variance']['mean_speaker_eta2']:.4f} vs "
          f"emotion eta2 {d['variance']['mean_emotion_eta2']:.4f} "
          f"({d['variance']['features_speaker_dominant']}/{d['health']['features']} speaker-dominant)")
    print(f"decode-path control: median |SMD| {d['decode_path']['median_abs_smd']:.2e}, "
          f"max {d['decode_path']['max_abs_smd']:.4f}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["eda", "models", "ann", "xai", "deep"]
    fns = {"eda": eda_digest, "models": models_digest, "ann": ann_digest,
           "xai": xai_digest, "deep": deep_digest}
    for w in which:
        try:
            fns[w]()
        except Exception as exc:
            print(f"[{w}] unavailable: {exc}")
