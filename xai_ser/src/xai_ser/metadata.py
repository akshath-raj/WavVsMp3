"""CREMA-D label and speaker metadata.

Filenames encode everything the supervised task needs:

    1001_DFA_ANG_XX.wav
    ^^^^ ^^^ ^^^ ^^
    |    |   |   +-- intensity: LO / MD / HI / XX (unspecified)
    |    |   +------ intended emotion: ANG DIS FEA HAP NEU SAD
    |    +---------- sentence code (12 fixed sentences)
    +--------------- actor id (1001..1091)

Actor demographics come from the corpus' VideoDemographics.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import META

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]

EMOTION_NAMES = {
    "ANG": "angry",
    "DIS": "disgusted",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

INTENSITIES = ["LO", "MD", "HI", "XX"]

SENTENCES = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}


def parse_filename(name: str) -> dict:
    """Split a CREMA-D stem into its four label fields."""
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"unexpected CREMA-D filename: {name!r}")
    actor, sentence, emotion, intensity = parts
    if emotion not in EMOTIONS:
        raise ValueError(f"unknown emotion code {emotion!r} in {name!r}")
    return {
        "item_id": stem,
        "speaker_id": actor,
        "sentence_code": sentence,
        "emotion": emotion,
        "intensity": intensity,
    }


def load_demographics() -> pd.DataFrame:
    """Actor-level demographics, keyed by `speaker_id` (string)."""
    df = pd.read_csv(META / "VideoDemographics.csv")
    df = df.rename(
        columns={
            "ActorID": "speaker_id",
            "Age": "age",
            "Sex": "sex",
            "Race": "race",
            "Ethnicity": "ethnicity",
        }
    )
    df["speaker_id"] = df["speaker_id"].astype(str)
    return df[["speaker_id", "age", "sex", "race", "ethnicity"]]


def build_index(wav_dir: Path) -> pd.DataFrame:
    """One row per source clip: labels + demographics + source path."""
    rows = []
    for p in sorted(wav_dir.glob("*.wav")):
        rec = parse_filename(p.name)
        rec["source_path"] = str(p)
        rows.append(rec)
    idx = pd.DataFrame(rows)
    if idx.empty:
        raise RuntimeError(f"no wav files found in {wav_dir}")
    idx = idx.merge(load_demographics(), on="speaker_id", how="left")
    idx["sentence_text"] = idx["sentence_code"].map(SENTENCES)
    idx["emotion_name"] = idx["emotion"].map(EMOTION_NAMES)
    return idx


# tabulatedVotes.csv reports voice-only votes with single-letter codes.
VOTE_LETTER_TO_CODE = {"A": "ANG", "D": "DIS", "F": "FEA", "H": "HAP", "N": "NEU", "S": "SAD"}


def load_perception_ratings() -> pd.DataFrame:
    """Human voice-only crowd ratings per clip.

    CREMA-D was rated in three modalities (voice only, face only, audio-visual).
    The voice-only tabulation is the fair human reference for a model that only
    hears the signal, so it gives us a human ceiling to compare classifiers
    against. Ties in `emoVote` are colon-separated and are treated as unresolved.

    tabulatedVotes.csv stacks all three modalities; the leading digit of its
    index column selects the block (1 = voice, 2 = face, 3 = audio-visual),
    verified against the VoiceVote/FaceVote columns of summaryTable.csv.
    """
    df = pd.read_csv(META / "tabulatedVotes.csv")
    df = df[df["Unnamed: 0"] // 100_000 == 1]
    out = pd.DataFrame(
        {
            "item_id": df["fileName"].astype(str),
            "human_n_ratings": df["numResponses"],
            "human_agreement": df["agreement"],
            "human_vote_raw": df["emoVote"].astype(str),
        }
    )
    out["human_vote_tied"] = out["human_vote_raw"].str.contains(":")
    out["human_vote"] = (
        out["human_vote_raw"].str.split(":").str[0].map(VOTE_LETTER_TO_CODE)
    )
    return out
