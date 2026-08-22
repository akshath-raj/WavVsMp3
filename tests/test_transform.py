"""Tests for Step 2 stimulus generation.

Verifies against the generated long manifest (data/manifest.parquet):
  * ref is 16k / mono / PCM16
  * mp4_aac64 and roundtrip_wav share an identical decoded-PCM hash
  * every expected stimulus file exists on disk

Run Steps 1-2 before these tests (they operate on real generated output).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import soundfile as sf

from src.config import MANIFEST_PATH
from src.transform import _decoded_pcm_float32

EXPECTED_TRANSFORMS = {"ref", "mp3_64", "mp4_aac64", "roundtrip_wav"}


@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        pytest.skip("manifest.parquet missing — run src.sample then src.transform")
    m = pd.read_parquet(MANIFEST_PATH)
    if "transform_id" not in m.columns:
        pytest.skip("manifest not expanded — run src.transform")
    return m


def test_ref_is_16k_mono_pcm16(manifest):
    ref = manifest[manifest["transform_id"] == "ref"]
    assert len(ref) > 0
    for _, row in ref.iterrows():
        info = sf.info(row["stim_path"])
        assert info.samplerate == 16000, row["item_id"]
        assert info.channels == 1, row["item_id"]
        assert info.subtype == "PCM_16", (row["item_id"], info.subtype)


def test_decoded_pcm_control_assertion(manifest):
    piv = manifest.pivot(index="item_id", columns="transform_id",
                         values="sha256_decoded_pcm")
    assert (piv["mp4_aac64"] == piv["roundtrip_wav"]).all(), \
        "mp4_aac64 and roundtrip_wav decoded-PCM hashes must match"


def test_control_hash_reproducible_from_bytes(manifest):
    """Recompute one pair from the actual files, not just the recorded hashes."""
    one = manifest["item_id"].iloc[0]
    rows = manifest[manifest["item_id"] == one].set_index("transform_id")
    mp4 = _decoded_pcm_float32(rows.at["mp4_aac64", "stim_path"])
    rt = _decoded_pcm_float32(rows.at["roundtrip_wav", "stim_path"])
    assert mp4.shape == rt.shape
    assert (mp4 == rt).all()


def test_all_expected_files_exist(manifest):
    for item_id, grp in manifest.groupby("item_id"):
        got = set(grp["transform_id"])
        assert EXPECTED_TRANSFORMS <= got, (item_id, got)
        for _, row in grp.iterrows():
            assert Path(row["stim_path"]).exists(), row["stim_path"]
