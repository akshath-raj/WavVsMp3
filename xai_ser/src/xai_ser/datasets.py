"""Loading, splitting and matrix-building for the feature table.

Two rules govern every split in this project:

1. **Speaker independence.** CREMA-D has 91 actors saying 12 fixed sentences.
   A random row split would put the same voice in train and test and inflate
   accuracy by a wide margin, so actors are partitioned, never rows.
2. **Item alignment across formats.** The codec arm compares a model trained on
   one format against the *same clips* delivered in another. Splits are
   therefore defined on speakers once and reused for every condition, so
   `train=ref / test=mp4_aac64` differs from `train=ref / test=ref` only in the
   format of the test audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .metadata import EMOTIONS
from .paths import FEATURES, latest

CONDITIONS = ["ref", "mp3_64", "mp4_aac64", "roundtrip_wav"]

META_COLS = {
    "item_id", "condition", "stim_path", "ext", "bytes", "speaker_id",
    "sentence_code", "emotion", "emotion_name", "intensity", "age", "sex",
    "race", "ethnicity", "sentence_text", "extract_ok", "extract_error",
    "sha256_decoded_pcm", "n_samples", "is_silent", "human_n_ratings",
    "human_agreement", "human_vote", "human_vote_raw", "human_vote_tied",
}

FAMILIES = ("spectral", "mfcc", "chroma", "prosody")


def load_features(path: str | Path | None = None) -> pd.DataFrame:
    """Newest feature table unless an explicit path is given."""
    p = Path(path) if path else latest(FEATURES, "features_*.parquet")
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df.attrs["source"] = str(p)
    return df


def feature_names(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def family_of(name: str) -> str:
    for fam in FAMILIES:
        if name.startswith(fam):
            return fam
    return "other"


@dataclass(frozen=True)
class SpeakerSplit:
    """A fixed partition of actors into train / validation / test."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def where(self, speaker_ids: pd.Series) -> dict[str, np.ndarray]:
        return {
            "train": speaker_ids.isin(self.train).to_numpy(),
            "val": speaker_ids.isin(self.val).to_numpy(),
            "test": speaker_ids.isin(self.test).to_numpy(),
        }


def make_speaker_split(
    df: pd.DataFrame, val_frac: float = 0.12, test_frac: float = 0.22, seed: int = 42
) -> SpeakerSplit:
    """Partition actors, keeping the sex balance of the corpus in each fold."""
    spk = (
        df[["speaker_id", "sex"]].drop_duplicates().sort_values("speaker_id").reset_index(drop=True)
    )
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for _, grp in spk.groupby("sex"):
        ids = grp["speaker_id"].to_numpy()
        ids = ids[rng.permutation(len(ids))]
        n = len(ids)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        test.extend(ids[:n_test])
        val.extend(ids[n_test : n_test + n_val])
        train.extend(ids[n_test + n_val :])
    return SpeakerSplit(tuple(sorted(train)), tuple(sorted(val)), tuple(sorted(test)))


def condition_frame(df: pd.DataFrame, condition: str, require_ok: bool = True) -> pd.DataFrame:
    sub = df[df["condition"] == condition]
    if require_ok:
        sub = sub[sub["extract_ok"].fillna(False)]
    return sub.sort_values("item_id").reset_index(drop=True)


def xy(
    df: pd.DataFrame, condition: str, cols: list[str], target: str = "emotion"
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Feature matrix, integer label vector and the metadata frame behind them."""
    sub = condition_frame(df, condition)
    X = sub[cols].copy()
    y = sub[target].map({e: i for i, e in enumerate(EMOTIONS)}).to_numpy()
    return X, y, sub


def aligned_items(df: pd.DataFrame, conditions: list[str] | None = None) -> pd.Index:
    """Item ids that extracted successfully in *every* condition.

    Cross-format comparisons are only meaningful on clips present in all arms,
    so all model tables are restricted to this intersection.
    """
    conditions = conditions or CONDITIONS
    ok = df[df["extract_ok"].fillna(False)]
    counts = ok[ok["condition"].isin(conditions)].groupby("item_id")["condition"].nunique()
    return counts[counts == len(conditions)].index
