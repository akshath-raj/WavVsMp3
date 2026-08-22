"""Generate the four format conditions for every source clip.

Each item gets its own directory under `data/stimuli/<item_id>/`:

    ref.wav            16 kHz mono PCM16, EBU R128 loudness-normalised
    mp3_64.mp3         MP3 @ 64 kbps, encoded from ref
    mp4_aac64.mp4      audio-only MP4/AAC @ 64 kbps, encoded from ref
    roundtrip_wav.wav  mp4_aac64 decoded back to PCM (decode-path control)

Conditions are chained (ref -> lossy -> decode) so every downstream difference
is attributable to the codec, not to a different starting signal.
"""

from __future__ import annotations

import hashlib
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from .audio_io import FFMPEG, probe_duration
from .metadata import build_index
from .paths import CONFIGS, MANIFEST, RAW, STIMULI, ensure_dirs


def load_config() -> dict:
    with open(CONFIGS / "transforms.yaml") as fh:
        return yaml.safe_load(fh)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd[:8])}…\n{proc.stderr.decode()[:400]}")


def build_item(args: tuple[str, str, dict, bool]) -> list[dict]:
    """Render all conditions for one source clip. Returns manifest rows."""
    item_id, source_path, cfg, overwrite = args
    ref_cfg = cfg["reference"]
    out_dir = STIMULI / item_id
    out_dir.mkdir(parents=True, exist_ok=True)

    produced: dict[str, Path] = {}
    rows: list[dict] = []

    for cond in cfg["order"]:
        spec = cfg["conditions"][cond]
        dest = out_dir / f"{cond}.{spec['ext']}"

        if overwrite or not dest.exists() or dest.stat().st_size == 0:
            if spec["kind"] == "reference":
                cmd = [
                    FFMPEG, "-v", "error", "-y", "-nostdin", "-i", source_path,
                    "-af", ref_cfg["loudnorm"],
                    "-ar", str(ref_cfg["sample_rate"]),
                    "-ac", str(ref_cfg["channels"]),
                    "-c:a", ref_cfg["codec"],
                    str(dest),
                ]
            else:
                src = produced[spec["from"]]
                cmd = [FFMPEG, "-v", "error", "-y", "-nostdin", "-i", str(src),
                       *spec["args"], str(dest)]
            _run(cmd)

        produced[cond] = dest
        rows.append(
            {
                "item_id": item_id,
                "condition": cond,
                "stim_path": str(dest),
                "ext": spec["ext"],
                "bytes": dest.stat().st_size,
                "sha256_file": sha256_file(dest),
                "duration_s": probe_duration(dest),
            }
        )
    return rows


def build_all(limit: int | None = None, overwrite: bool = False, workers: int = 8) -> pd.DataFrame:
    ensure_dirs()
    cfg = load_config()
    index = build_index(RAW)
    if limit:
        index = index.head(limit)

    jobs = [(r.item_id, r.source_path, cfg, overwrite) for r in index.itertuples()]
    rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_item, j): j[0] for j in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="transcoding"):
            try:
                rows.extend(fut.result())
            except Exception as exc:  # a single bad source must not kill the run
                failures.append((futures[fut], str(exc)[:200]))

    manifest = pd.DataFrame(rows).merge(index, on="item_id", how="left")
    manifest.to_parquet(MANIFEST, index=False)

    if failures:
        fail_path = MANIFEST.parent / "transform_failures.csv"
        pd.DataFrame(failures, columns=["item_id", "error"]).to_csv(fail_path, index=False)
        print(f"{len(failures)} items failed; see {fail_path}")

    print(f"manifest: {len(manifest)} rows -> {MANIFEST}")
    return manifest


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate format conditions for CREMA-D clips.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    build_all(limit=a.limit, overwrite=a.overwrite, workers=a.workers)
