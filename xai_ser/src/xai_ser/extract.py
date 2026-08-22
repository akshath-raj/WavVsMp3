"""Build the timestamped feature table that every model downstream trains on.

Output: `data/features/features_<UTC timestamp>.csv` — one row per
(clip x format condition), carrying labels, speaker metadata, human voice-only
ratings, extraction bookkeeping, and ~400 named acoustic features. A companion
`schema_<timestamp>.json` records exactly how the table was produced.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import yaml
from tqdm import tqdm

from .features import extract_one
from .metadata import load_perception_ratings
from .paths import CONFIGS, FEATURES, MANIFEST, ensure_dirs, timestamp

LABEL_COLS = [
    "item_id", "condition", "stim_path", "ext", "bytes",
    "speaker_id", "sentence_code", "emotion", "emotion_name", "intensity",
    "age", "sex", "race", "ethnicity", "sentence_text",
]


def load_feature_config() -> dict:
    with open(CONFIGS / "features.yaml") as fh:
        return yaml.safe_load(fh)


def _job(args):
    path, fcfg = args
    return extract_one(path, fcfg)


def run(limit: int | None = None, workers: int = 8, conditions: list[str] | None = None) -> pd.DataFrame:
    ensure_dirs()
    fcfg = load_feature_config()
    manifest = pd.read_parquet(MANIFEST)

    if conditions:
        manifest = manifest[manifest["condition"].isin(conditions)]
    if limit:
        keep = manifest["item_id"].drop_duplicates().head(limit)
        manifest = manifest[manifest["item_id"].isin(keep)]

    jobs = [(p, fcfg) for p in manifest["stim_path"].tolist()]
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_job, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="features"):
            records.append(fut.result())

    feat = pd.DataFrame.from_records(records)
    df = manifest[[c for c in LABEL_COLS if c in manifest.columns]].merge(
        feat, on="stim_path", how="left"
    )
    df = df.merge(load_perception_ratings(), on="item_id", how="left")

    ts = timestamp()
    out_csv = FEATURES / f"features_{ts}.csv"
    df.to_csv(out_csv, index=False)
    df.to_parquet(FEATURES / f"features_{ts}.parquet", index=False)

    # Single source of truth for what counts as a modelling column, shared with
    # datasets.py so the schema's feature count matches what the models see.
    from .datasets import META_COLS

    feature_names = [
        c for c in df.columns
        if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    schema = {
        "created_utc": ts,
        "rows": int(len(df)),
        "items": int(df["item_id"].nunique()),
        "conditions": sorted(df["condition"].unique().tolist()),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "families": {
            fam: sum(1 for c in feature_names if c.startswith(fam))
            for fam in ("spectral_", "mfcc_", "chroma_", "prosody_")
        },
        "extraction_failures": int((~df["extract_ok"].fillna(False)).sum()),
        "config": fcfg,
    }
    with open(FEATURES / f"schema_{ts}.json", "w") as fh:
        json.dump(schema, fh, indent=2)

    print(f"wrote {out_csv} — {len(df)} rows x {len(feature_names)} features")
    print(f"extraction failures: {schema['extraction_failures']}")
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract acoustic features to a timestamped CSV.")
    ap.add_argument("--limit", type=int, default=None, help="limit to N source clips")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--conditions", nargs="*", default=None)
    a = ap.parse_args()
    run(limit=a.limit, workers=a.workers, conditions=a.conditions)
