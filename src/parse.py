"""Step 6 — map raw model text to a label with an explicit failure taxonomy.

parse_status in {ok, refusal, out_of_vocab, malformed, empty, api_error}

We never silently score a failure as "wrong": a parse-rate collapse and an
accuracy collapse are different findings and must be told apart.
"""
from __future__ import annotations

import re

import pandas as pd
from rapidfuzz import fuzz

from .config import SENTENCES, cfg

EMOTIONS = cfg.emotion_labels  # [angry, disgusted, fearful, happy, neutral, sad]
FUZZY_THRESHOLD = 80  # logged; anything matched via fuzz >= this is flagged for review

# Exact-ish synonyms mapped to the canonical lexicon.
SYNONYMS = {
    "anger": "angry", "angered": "angry", "mad": "angry", "furious": "angry",
    "irritated": "angry", "annoyed": "angry",
    "disgust": "disgusted", "disgusting": "disgusted", "revolted": "disgusted",
    "fear": "fearful", "afraid": "fearful", "scared": "fearful",
    "frightened": "fearful", "anxious": "fearful", "terrified": "fearful",
    "joy": "happy", "joyful": "happy", "glad": "happy", "cheerful": "happy",
    "pleased": "happy", "content": "happy",
    "calm": "neutral", "none": "neutral", "no emotion": "neutral",
    "neutrality": "neutral", "normal": "neutral",
    "sadness": "sad", "unhappy": "sad", "sorrowful": "sad", "depressed": "sad",
    "melancholy": "sad", "down": "sad",
}

_REFUSAL_PATTERNS = [
    r"\bi (?:can'?t|cannot|am unable|'m unable)\b",
    r"\bunable to\b",
    r"\bas an ai\b",
    r"\bi (?:do not|don'?t) have\b",
    r"\bcan'?t (?:determine|tell|identify)\b",
    r"\bnot able to\b",
    r"\bsorry\b",
]
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)

_PUNCT_RE = re.compile(r"[^\w\s']")


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_emotion(raw_text: str | None, error: str | None = None) -> dict:
    """Return {parsed_label, parse_status, fuzzy_score, needs_review}."""
    if error:
        return _res(None, "api_error")
    if raw_text is None or not raw_text.strip():
        return _res(None, "empty")

    norm = _normalize(raw_text)
    if not norm:
        return _res(None, "empty")

    # 1) exact single-word or contains an exact label token
    tokens = norm.split()
    for tok in tokens:
        if tok in EMOTIONS:
            return _res(tok, "ok")

    # 2) synonym map (single token or whole-string)
    if norm in SYNONYMS:
        return _res(SYNONYMS[norm], "ok")
    for tok in tokens:
        if tok in SYNONYMS:
            return _res(SYNONYMS[tok], "ok")

    # 3) refusal detection (before fuzzy, so "sorry, I can't" isn't fuzzed to "sad")
    if _REFUSAL_RE.search(raw_text):
        return _res(None, "refusal")

    # 4) fuzzy match to nearest label (flagged for manual review)
    best_label, best_score = None, -1.0
    for lab in EMOTIONS:
        s = fuzz.ratio(norm, lab)
        if s > best_score:
            best_label, best_score = lab, s
    if best_score >= FUZZY_THRESHOLD:
        return _res(best_label, "ok", fuzzy_score=best_score, needs_review=True)

    # 5) couldn't resolve. Single unknown token -> out_of_vocab; else malformed.
    if len(tokens) == 1:
        return _res(None, "out_of_vocab", fuzzy_score=best_score)
    return _res(None, "malformed", fuzzy_score=best_score)


def _res(label, status, fuzzy_score=None, needs_review=False) -> dict:
    return {
        "parsed_label": label,
        "parse_status": status,
        "fuzzy_score": fuzzy_score,
        "needs_review": needs_review,
    }


# ---------------------------------------------------------------------------
# Transcription WER
# ---------------------------------------------------------------------------
def _normalize_transcript(text: str) -> str:
    return _normalize(text)


def wer_against_sentence(raw_text: str | None, sentence_code: str) -> float | None:
    """WER of a transcript against the known CREMA-D sentence text."""
    if raw_text is None or not raw_text.strip():
        return None
    ref = _normalize_transcript(SENTENCES[sentence_code])
    hyp = _normalize_transcript(raw_text)
    from jiwer import wer  # local import keeps parse_emotion dependency-light
    return float(wer(ref, hyp))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def parse_status_table(results: pd.DataFrame) -> pd.DataFrame:
    """Counts of parse_status by transform_id (emotion task only)."""
    emo = results[results["task"] == "emotion"]
    tab = pd.crosstab(emo["transform_id"], emo["parse_status"])
    return tab


def print_parse_report(results: pd.DataFrame) -> None:
    print("\n=== parse_status counts by transform (emotion task) ===")
    print(parse_status_table(results).to_string())
    n_review = int(results.get("needs_review", pd.Series(dtype=bool)).sum())
    print(f"\nfuzzy threshold = {FUZZY_THRESHOLD}; rows flagged for manual review: "
          f"{n_review}")
