"""Step 1 — sample 50 CREMA-D clips into data/manifest.parquet.

Constraints:
  * stratified across the 6 emotion labels (ANG, DIS, FEA, HAP, NEU, SAD)
  * >= 10 distinct speakers, gender-balanced
  * fixed seed (42), recorded in the manifest
  * fail loudly if any emotion class ends up with < 5 items

We avoid cloning the multi-GB repo: the full AudioWAV file list comes from the
GitHub trees API, and only the 50 chosen clips are downloaded (from the Git-LFS
media host).
"""
from __future__ import annotations

import io
import sys

import numpy as np
import pandas as pd
import requests
import soundfile as sf
from tqdm import tqdm

from .config import (
    CREMAD_MEDIA_BASE,
    CREMAD_RAW_BASE,
    CREMAD_TREE_API,
    EMOTIONS,
    MANIFEST_PATH,
    RAW_DIR,
    SEED,
    cfg,
)

N_TOTAL = 50
MIN_PER_CLASS = 5
MIN_SPEAKERS = 10


def _fetch_demographics() -> pd.DataFrame:
    url = f"{CREMAD_RAW_BASE}/VideoDemographics.csv"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    demo = pd.read_csv(io.StringIO(r.text))
    demo = demo.rename(columns={"ActorID": "speaker_id", "Sex": "gender"})
    demo["speaker_id"] = demo["speaker_id"].astype(str)
    return demo[["speaker_id", "gender"]]


def _fetch_audio_index() -> pd.DataFrame:
    """All AudioWAV/*.wav filenames -> parsed item table."""
    r = requests.get(CREMAD_TREE_API, timeout=60)
    r.raise_for_status()
    tree = r.json().get("tree", [])
    if r.json().get("truncated"):
        raise RuntimeError("GitHub tree listing was truncated; cannot sample safely.")
    rows = []
    for node in tree:
        path = node.get("path", "")
        if not (path.startswith("AudioWAV/") and path.endswith(".wav")):
            continue
        stem = path[len("AudioWAV/") : -len(".wav")]
        parts = stem.split("_")
        if len(parts) != 4:
            continue  # skip anything not matching ActorID_SENT_EMO_LEVEL
        speaker_id, sentence_code, emotion, level = parts
        if emotion not in EMOTIONS:
            continue
        rows.append(
            {
                "item_id": stem,
                "wav_path": path,
                "speaker_id": speaker_id,
                "sentence_code": sentence_code,
                "emotion_gold": emotion,
                "intensity": level,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No AudioWAV entries found in CREMA-D tree listing.")
    return df


def _target_per_emotion(n_total: int, emotions: list[str]) -> dict[str, int]:
    k = len(emotions)
    base = n_total // k
    rem = n_total - base * k
    # Deterministic remainder distribution: first `rem` emotions (fixed order) get +1.
    return {e: base + (1 if i < rem else 0) for i, e in enumerate(emotions)}


def _sample(df: pd.DataFrame) -> pd.DataFrame:
    """Constrained stratified sample.

    For each emotion we split the quota across genders and draw preferring
    distinct speakers (one clip per speaker until the quota forces reuse). This
    guarantees gender balance and high speaker diversity by construction.
    """
    rng = np.random.default_rng(SEED)
    targets = _target_per_emotion(N_TOTAL, EMOTIONS)
    picked_idx: list[int] = []

    for emo in EMOTIONS:
        quota = targets[emo]
        pool = df[df["emotion_gold"] == emo]
        # Split quota female/male (extra goes to female for a deterministic tie-break).
        n_f = quota // 2 + (quota % 2)
        n_m = quota - n_f
        for gender, n in (("Female", n_f), ("Male", n_m)):
            sub = pool[pool["gender"] == gender]
            chosen = _draw_diverse(sub, n, rng)
            picked_idx.extend(chosen)

    out = df.loc[picked_idx].copy()
    # Safety: if any duplicate item_ids slipped in, drop and top up deterministically.
    out = out.drop_duplicates("item_id")
    if len(out) < N_TOTAL:
        remaining = df.drop(index=out.index)
        extra = remaining.sample(N_TOTAL - len(out), random_state=SEED)
        out = pd.concat([out, extra])
    return out.reset_index(drop=True)


def _draw_diverse(sub: pd.DataFrame, n: int, rng: np.random.Generator) -> list[int]:
    """Draw n row indices from `sub`, maximizing distinct speakers."""
    if n <= 0 or sub.empty:
        return []
    # Shuffle rows deterministically.
    order = rng.permutation(sub.index.to_numpy())
    seen_speakers: set[str] = set()
    first_pass: list[int] = []
    leftover: list[int] = []
    for idx in order:
        spk = sub.at[idx, "speaker_id"]
        if spk not in seen_speakers:
            seen_speakers.add(spk)
            first_pass.append(int(idx))
        else:
            leftover.append(int(idx))
    picks = first_pass[:n]
    if len(picks) < n:  # needed more than we had distinct speakers
        picks += leftover[: n - len(picks)]
    return picks


def _download(items: pd.DataFrame) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    durations = []
    src_paths = []
    for _, row in tqdm(list(items.iterrows()), desc="download", unit="clip"):
        dest = RAW_DIR / f"{row['item_id']}.wav"
        if not dest.exists():
            url = f"{CREMAD_MEDIA_BASE}/{row['wav_path']}"
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)
        info = sf.info(str(dest))
        durations.append(round(info.frames / info.samplerate, 4))
        src_paths.append(str(dest))
    items = items.copy()
    items["source_path"] = src_paths
    items["duration_s"] = durations
    return items


def main() -> None:
    cfg.ensure_dirs()
    print("Fetching CREMA-D demographics and audio index ...")
    demo = _fetch_demographics()
    index = _fetch_audio_index()
    index = index.merge(demo, on="speaker_id", how="left")
    missing = index["gender"].isna().sum()
    if missing:
        print(f"  warning: {missing} clips missing demographics; dropping them.")
        index = index.dropna(subset=["gender"])

    print(f"  {len(index)} usable AudioWAV clips across "
          f"{index['speaker_id'].nunique()} speakers.")

    sample = _sample(index)
    sample = _download(sample)

    manifest = sample[[
        "item_id", "source_path", "speaker_id", "gender",
        "emotion_gold", "sentence_code", "duration_s",
    ]].copy()
    manifest["seed"] = SEED
    manifest.to_parquet(MANIFEST_PATH, index=False)

    _summarize_and_check(manifest)
    print(f"\nWrote {len(manifest)} rows -> {MANIFEST_PATH}")


def _summarize_and_check(m: pd.DataFrame) -> None:
    print("\n=== Emotion x gender counts ===")
    print(pd.crosstab(m["emotion_gold"], m["gender"], margins=True))

    print("\n=== Clips per speaker ===")
    spk = m.groupby(["speaker_id", "gender"]).size().rename("n_clips")
    print(spk.to_string())
    n_speakers = m["speaker_id"].nunique()
    print(f"\nDistinct speakers: {n_speakers}")
    print(f"Gender split: {m['gender'].value_counts().to_dict()}")

    # Fail-loud checks.
    class_counts = m["emotion_gold"].value_counts()
    bad = class_counts[class_counts < MIN_PER_CLASS]
    if not bad.empty:
        sys.exit(f"FATAL: emotion classes below {MIN_PER_CLASS}: {bad.to_dict()}")
    if n_speakers < MIN_SPEAKERS:
        sys.exit(f"FATAL: only {n_speakers} distinct speakers (< {MIN_SPEAKERS}).")
    print("\nAll sampling constraints satisfied.")


if __name__ == "__main__":
    main()
