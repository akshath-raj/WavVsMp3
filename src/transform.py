"""Step 2 — generate the four stimulus conditions per item.

ref            canonical 16k/mono/PCM16, loudness-normalized (from CREMA-D source)
mp3_64         MP3 @ 64k       (from ref)
mp4_aac64      AAC @ 64k, mp4  (from ref)
roundtrip_wav  decode mp4_aac64 back to 16k/mono/PCM16 WAV  (THE CONTROL)

roundtrip_wav must carry the *same signal* as mp4_aac64, just in a WAV container.
We assert their decoded-PCM hashes are identical; if not, the control is invalid
and we stop rather than proceed.

Reads the item-level manifest from Step 1 and rewrites data/manifest.parquet as a
long (item x transform) table with per-stimulus fidelity/provenance columns.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import MANIFEST_PATH, STIMULI_DIR, cfg


# ---------------------------------------------------------------------------
# ffmpeg/ffprobe helpers
# ---------------------------------------------------------------------------
def _run(cmd: list[str]) -> bytes:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"{proc.stderr.decode('utf-8', 'replace')}"
        )
    return proc.stdout


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _decoded_pcm_float32(path) -> np.ndarray:
    """Decode any container to canonical 16k/mono float32 via ffmpeg.

    Using a single fixed decode path (16k mono s16le) makes the hash a property
    of the *signal*, independent of the container — which is exactly what lets
    mp4_aac64 and roundtrip_wav match.
    """
    raw = _run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-ar", "16000", "-ac", "1", "-f", "s16le", "-"]
    )
    pcm16 = np.frombuffer(raw, dtype="<i2")
    return (pcm16.astype(np.float32) / 32768.0)


def _sha256_decoded_pcm(path) -> str:
    arr = _decoded_pcm_float32(path)
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _ffprobe(path) -> dict:
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate,channels,bit_rate,duration",
        "-show_entries", "format=bit_rate,duration",
        "-of", "json", str(path),
    ])
    info = json.loads(out)
    stream = (info.get("streams") or [{}])[0]
    fmt = info.get("format", {})

    def _num(*vals):
        for v in vals:
            if v not in (None, "", "N/A"):
                return v
        return None

    sr = stream.get("sample_rate")
    ch = stream.get("channels")
    dur = _num(stream.get("duration"), fmt.get("duration"))
    bitrate = _num(stream.get("bit_rate"), fmt.get("bit_rate"))
    return {
        "sr": int(sr) if sr else None,
        "channels": int(ch) if ch is not None else None,
        "duration_s": round(float(dur), 4) if dur else None,
        "achieved_bitrate": int(bitrate) if bitrate else None,
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _build_cmd(kind: str, in_path, out_path, args: list[str]) -> list[str]:
    ref_cfg = cfg.transforms["reference"]
    if kind == "reference":
        return [
            "ffmpeg", "-y", "-i", str(in_path),
            "-ar", str(ref_cfg["sample_rate"]),
            "-ac", str(ref_cfg["channels"]),
            "-c:a", ref_cfg["codec"],
            "-af", ref_cfg["loudnorm"],
            str(out_path),
        ]
    # encode / decode: caller-supplied args
    return ["ffmpeg", "-y", "-i", str(in_path), *args, str(out_path)]


def _generate_item(item_id: str, source_path: str) -> list[dict]:
    conditions = cfg.transforms["conditions"]
    order = cfg.transforms["order"]
    item_dir = STIMULI_DIR / item_id
    item_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}  # transform_id -> generated file path
    rows: list[dict] = []
    for tid in order:
        spec = conditions[tid]
        out_path = item_dir / f"{tid}.{spec['ext']}"
        in_path = source_path if spec["kind"] == "reference" else paths[spec["from"]]
        cmd = _build_cmd(spec["kind"], in_path, out_path, spec.get("args", []))
        _run(cmd)
        paths[tid] = str(out_path)

        probe = _ffprobe(out_path)
        rows.append({
            "item_id": item_id,
            "transform_id": tid,
            "stim_path": str(out_path),
            "sha256_file": _sha256_file(out_path),
            "sha256_decoded_pcm": _sha256_decoded_pcm(out_path),
            **probe,
        })
    return rows


def main() -> None:
    if not MANIFEST_PATH.exists():
        sys.exit("manifest.parquet not found — run src.sample first.")
    manifest = pd.read_parquet(MANIFEST_PATH)
    if "transform_id" in manifest.columns:
        sys.exit(
            "manifest already looks expanded (has transform_id). "
            "Re-run src.sample to regenerate the item-level manifest first."
        )

    all_rows: list[dict] = []
    for _, item in tqdm(list(manifest.iterrows()), desc="transform", unit="item"):
        all_rows.extend(_generate_item(item["item_id"], item["source_path"]))

    stim = pd.DataFrame(all_rows)

    _assert_control_valid(stim)

    # Long manifest: item-level fields joined onto per-stimulus rows.
    item_cols = manifest.rename(columns={"duration_s": "source_duration_s"})
    long = stim.merge(item_cols, on="item_id", how="left")
    long.to_parquet(MANIFEST_PATH, index=False)

    _summarize(long)
    print(f"\nWrote long manifest ({len(long)} rows) -> {MANIFEST_PATH}")


def _assert_control_valid(stim: pd.DataFrame) -> None:
    """mp4_aac64 and roundtrip_wav must share an identical decoded-PCM hash."""
    piv = stim.pivot(index="item_id", columns="transform_id",
                     values="sha256_decoded_pcm")
    mismatched = piv.index[piv["mp4_aac64"] != piv["roundtrip_wav"]].tolist()
    if mismatched:
        print("\nFATAL: decoded-PCM control assertion FAILED for these items:")
        for iid in mismatched:
            print(f"  {iid}: mp4={piv.at[iid, 'mp4_aac64'][:12]}  "
                  f"roundtrip={piv.at[iid, 'roundtrip_wav'][:12]}")
        sys.exit(
            "The roundtrip control is invalid — fix the decode path before proceeding."
        )
    print(f"\nControl OK: mp4_aac64 == roundtrip_wav decoded-PCM for all "
          f"{len(piv)} items.")


def _summarize(long: pd.DataFrame) -> None:
    print("\n=== Stimulus summary (median per transform) ===")
    summ = (long.groupby("transform_id")
            .agg(n=("item_id", "size"),
                 sr=("sr", "median"),
                 channels=("channels", "median"),
                 median_bitrate=("achieved_bitrate", "median"),
                 median_dur_s=("duration_s", "median"))
            .reindex(cfg.transforms["order"]))
    print(summ.to_string())

    # Sanity: ref must be 16k/mono.
    ref = long[long["transform_id"] == "ref"]
    assert (ref["sr"] == 16000).all(), "ref not 16k"
    assert (ref["channels"] == 1).all(), "ref not mono"


if __name__ == "__main__":
    main()
