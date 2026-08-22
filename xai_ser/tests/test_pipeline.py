"""Invariants the study's conclusions depend on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xai_ser.datasets import aligned_items, make_speaker_split
from xai_ser.features import _summarise
from xai_ser.metadata import EMOTIONS, parse_filename


def test_parse_filename():
    rec = parse_filename("1001_DFA_ANG_XX.wav")
    assert rec == {
        "item_id": "1001_DFA_ANG_XX",
        "speaker_id": "1001",
        "sentence_code": "DFA",
        "emotion": "ANG",
        "intensity": "XX",
    }


def test_parse_filename_rejects_unknown_emotion():
    with pytest.raises(ValueError):
        parse_filename("1001_DFA_ZZZ_XX.wav")


def test_summarise_handles_empty_and_nan():
    out = _summarise(np.array([np.nan, np.nan]), "x")
    assert set(out) == {f"x_{s}" for s in
                        ("mean", "std", "min", "max", "median", "iqr", "skew", "kurt")}
    assert all(np.isnan(v) for v in out.values())

    out = _summarise(np.array([1.0, 2.0, 3.0, 4.0]), "x")
    assert out["x_mean"] == pytest.approx(2.5)
    assert out["x_median"] == pytest.approx(2.5)
    assert out["x_iqr"] == pytest.approx(1.5)


def _toy_frame(n_speakers=20, conditions=("ref", "mp3_64")):
    rows = []
    for s in range(n_speakers):
        for i, emo in enumerate(EMOTIONS):
            for cond in conditions:
                rows.append({
                    "item_id": f"{1000 + s}_IEO_{emo}_XX",
                    "speaker_id": str(1000 + s),
                    "sex": "Male" if s % 2 else "Female",
                    "emotion": emo,
                    "condition": cond,
                    "extract_ok": True,
                })
    return pd.DataFrame(rows)


def test_speaker_split_is_disjoint_and_covers_everyone():
    df = _toy_frame()
    split = make_speaker_split(df, seed=7)
    train, val, test = set(split.train), set(split.val), set(split.test)

    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(df["speaker_id"])


def test_speaker_split_is_deterministic():
    df = _toy_frame()
    assert make_speaker_split(df, seed=7) == make_speaker_split(df, seed=7)


def test_aligned_items_drops_partial_coverage():
    df = _toy_frame(n_speakers=3)
    conditions = ["ref", "mp3_64"]
    # Break one item in one condition only.
    broken = (df["item_id"] == df["item_id"].iloc[0]) & (df["condition"] == "mp3_64")
    df.loc[broken, "extract_ok"] = False

    keep = aligned_items(df, conditions)
    assert df["item_id"].iloc[0] not in keep
    assert len(keep) == df["item_id"].nunique() - 1


def test_ann_roundtrip_preprocessing(tmp_path, monkeypatch):
    """A saved ANN must reload into a working predictor.

    Regression test: the preprocessing was originally rebuilt from raw
    statistics arrays, which leaves sklearn's private state unset and raises on
    the first transform. The fitted objects are pickled instead.
    """
    torch = pytest.importorskip("torch")
    import joblib
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    from xai_ser.ann import AnnConfig, EmotionMLP, TorchWrapper

    rng = np.random.default_rng(0)
    cols = [f"f{i}" for i in range(12)]
    X = pd.DataFrame(rng.normal(size=(40, 12)), columns=cols)
    X.iloc[0, 0] = np.nan

    imputer = SimpleImputer(strategy="median").fit(X)
    scaler = StandardScaler().fit(imputer.transform(X))
    model = EmotionMLP(12, len(EMOTIONS), AnnConfig())
    model.eval()

    joblib.dump({"imputer": imputer, "scaler": scaler}, tmp_path / "preproc.joblib")
    pre = joblib.load(tmp_path / "preproc.joblib")

    wrapper = TorchWrapper(model, pre["imputer"], pre["scaler"])
    pred = wrapper.predict(X)
    assert pred.shape == (40,)
    assert set(np.unique(pred)) <= set(range(len(EMOTIONS)))
    assert wrapper.predict_proba(X).shape == (40, len(EMOTIONS))
