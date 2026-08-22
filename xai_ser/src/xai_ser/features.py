"""Acoustic feature extraction (librosa + Praat/parselmouth).

Every stimulus is reduced to one fixed-length, *named* feature vector. Names
matter as much as values here: SHAP and LIME explanations are only interpretable
if each column maps back to something an acoustician would recognise, so the
naming scheme is `<family>_<descriptor>_<statistic>` throughout.

Four families:

  spectral/*   librosa spectral shape descriptors (centroid, rolloff, flatness,
               contrast, band-energy ratios, entropy) — the family a lossy codec
               is expected to damage first, since AAC/MP3 discard high-frequency
               and perceptually-masked content.
  mfcc/*       20 MFCCs plus deltas — the standard SER front end.
  chroma/*     pitch-class energy.
  prosody/*    Praat measures: F0, jitter, shimmer, HNR, formants, intensity,
               and voiced-segment timing — the source-filter view of emotion.

Frame-level contours are summarised with eight order statistics so that the
resulting table is a plain tabular dataset that decision trees can consume.
"""

from __future__ import annotations

import hashlib
import warnings
from typing import Any

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy import stats as sps

from .audio_io import decode

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

EPS = 1e-10

# Frequency band edges (Hz) for band-energy ratios. The top two bands straddle
# the region where 64 kbps AAC/MP3 typically start low-passing, which makes them
# the most diagnostic features for the codec arm of the study.
BAND_EDGES = [0, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000]

SUMMARY_STATS = ("mean", "std", "min", "max", "median", "iqr", "skew", "kurt")


def _summarise(x: np.ndarray, name: str, stats: tuple[str, ...] = SUMMARY_STATS) -> dict[str, float]:
    """Reduce a 1-D contour to named order statistics."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{name}_{s}": np.nan for s in stats}
    out: dict[str, float] = {}
    for s in stats:
        if s == "mean":
            out[f"{name}_mean"] = float(np.mean(x))
        elif s == "std":
            out[f"{name}_std"] = float(np.std(x))
        elif s == "min":
            out[f"{name}_min"] = float(np.min(x))
        elif s == "max":
            out[f"{name}_max"] = float(np.max(x))
        elif s == "median":
            out[f"{name}_median"] = float(np.median(x))
        elif s == "iqr":
            out[f"{name}_iqr"] = float(np.subtract(*np.percentile(x, [75, 25])))
        elif s == "skew":
            out[f"{name}_skew"] = float(sps.skew(x)) if x.size > 2 else np.nan
        elif s == "kurt":
            out[f"{name}_kurt"] = float(sps.kurtosis(x)) if x.size > 3 else np.nan
    return out


# --------------------------------------------------------------------------
# librosa families
# --------------------------------------------------------------------------

def spectral_features(y: np.ndarray, sr: int, fcfg: dict) -> dict[str, float]:
    n_fft = fcfg["frame"]["n_fft"]
    hop = fcfg["frame"]["hop_length"]
    win = fcfg["frame"]["win_length"]

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    feats: dict[str, float] = {}

    contours = {
        "spectral_centroid": librosa.feature.spectral_centroid(S=S, sr=sr, freq=freqs)[0],
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(S=S, sr=sr, freq=freqs)[0],
        "spectral_rolloff85": librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0],
        "spectral_rolloff95": librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)[0],
        "spectral_flatness": librosa.feature.spectral_flatness(S=np.sqrt(S))[0],
        "spectral_zcr": librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop)[0],
        "spectral_rms": librosa.feature.rms(S=np.sqrt(S), frame_length=n_fft, hop_length=hop)[0],
        "spectral_flux": np.sqrt(np.sum(np.diff(np.sqrt(S), axis=1) ** 2, axis=0)),
    }

    # Per-frame spectral entropy: how noise-like the frame is. Codec quantisation
    # zeroes out masked bins, which shows up here before it shows up in MFCCs.
    P = S / (S.sum(axis=0, keepdims=True) + EPS)
    contours["spectral_entropy"] = -np.sum(P * np.log(P + EPS), axis=0) / np.log(P.shape[0])

    # Spectral slope: linear regression of magnitude on frequency, per frame.
    # Done as an explicit float64 reduction rather than a matmul — the BLAS on
    # Apple silicon leaks spurious divide-by-zero FP flags out of `@` here.
    f_centered = (freqs - freqs.mean()).astype(np.float64)
    denom = float(np.sum(f_centered**2)) + EPS
    mag = np.sqrt(S).astype(np.float64)
    contours["spectral_slope"] = (
        np.sum(f_centered[:, None] * (mag - mag.mean(axis=0, keepdims=True)), axis=0) / denom
    )

    for name, contour in contours.items():
        feats.update(_summarise(contour, name))

    # Band-energy ratios. Frame energy is normalised so these describe spectral
    # *shape* rather than loudness.
    total = S.sum(axis=0) + EPS
    for lo, hi in zip(BAND_EDGES[:-1], BAND_EDGES[1:]):
        sel = (freqs >= lo) & (freqs < hi)
        ratio = S[sel].sum(axis=0) / total
        feats.update(_summarise(ratio, f"spectral_band_{lo}_{hi}", ("mean", "std")))

    # Explicit high-frequency survival measures: the single most direct probes of
    # lossy-codec low-pass behaviour.
    for cutoff in (4000, 6000):
        sel = freqs >= cutoff
        feats.update(
            _summarise(S[sel].sum(axis=0) / total, f"spectral_hf_ratio_{cutoff}", ("mean", "std"))
        )

    contrast = librosa.feature.spectral_contrast(S=np.sqrt(S), sr=sr, n_bands=6)
    for i, row in enumerate(contrast):
        feats.update(_summarise(row, f"spectral_contrast_{i}", ("mean", "std")))

    return feats


def mfcc_features(y: np.ndarray, sr: int, fcfg: dict) -> dict[str, float]:
    m = fcfg["mfcc"]
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=m["n_mfcc"], n_mels=m["n_mels"], fmin=m["fmin"], fmax=m["fmax"],
        n_fft=fcfg["frame"]["n_fft"], hop_length=fcfg["frame"]["hop_length"],
        win_length=fcfg["frame"]["win_length"],
    )
    feats: dict[str, float] = {}
    for i, row in enumerate(mfcc):
        feats.update(_summarise(row, f"mfcc_{i:02d}"))

    if mfcc.shape[1] >= 9:
        d1 = librosa.feature.delta(mfcc, order=1)
        d2 = librosa.feature.delta(mfcc, order=2)
    else:  # too short for the default 9-frame delta window
        d1 = np.zeros_like(mfcc) * np.nan
        d2 = np.zeros_like(mfcc) * np.nan
    for i in range(mfcc.shape[0]):
        feats.update(_summarise(d1[i], f"mfcc_d1_{i:02d}", ("mean", "std")))
        feats.update(_summarise(d2[i], f"mfcc_d2_{i:02d}", ("mean", "std")))
    return feats


def chroma_features(y: np.ndarray, sr: int, fcfg: dict) -> dict[str, float]:
    # tuning is pinned to 0 rather than estimated. Tuning estimation is designed
    # for music and is meaningless on speech; pinning it also keeps chroma
    # comparable across format conditions, since an *estimated* tuning could
    # itself shift under compression and confound the codec contrast.
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, tuning=0.0,
        n_fft=fcfg["frame"]["n_fft"], hop_length=fcfg["frame"]["hop_length"],
        win_length=fcfg["frame"]["win_length"],
    )
    feats: dict[str, float] = {}
    for i, row in enumerate(chroma):
        feats.update(_summarise(row, f"chroma_{i:02d}", ("mean", "std")))
    return feats


# --------------------------------------------------------------------------
# Praat family
# --------------------------------------------------------------------------

def _nan_dict(keys: list[str]) -> dict[str, float]:
    return {k: np.nan for k in keys}


PROSODY_SCALAR_KEYS = [
    "prosody_jitter_local", "prosody_jitter_rap", "prosody_jitter_ppq5", "prosody_jitter_ddp",
    "prosody_shimmer_local", "prosody_shimmer_localdb", "prosody_shimmer_apq3",
    "prosody_shimmer_apq5", "prosody_shimmer_apq11", "prosody_shimmer_dda",
    "prosody_voiced_fraction", "prosody_n_voiced_segments", "prosody_voiced_rate_per_s",
    "prosody_mean_voiced_seg_s", "prosody_pause_fraction", "prosody_f0_slope_abs_mean",
    "prosody_cpps",
]


def prosody_features(y: np.ndarray, sr: int, fcfg: dict) -> dict[str, float]:
    """Praat source-filter and timing measures via parselmouth."""
    p = fcfg["praat"]
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    feats: dict[str, float] = {}

    # --- F0 ---
    try:
        pitch = snd.to_pitch(pitch_floor=p["pitch_floor"], pitch_ceiling=p["pitch_ceiling"])
        f0 = pitch.selected_array["frequency"]
        voiced = f0[f0 > 0]
        feats.update(_summarise(voiced, "prosody_f0"))
        feats["prosody_voiced_fraction"] = float(np.mean(f0 > 0))
        if voiced.size > 1:
            dt = pitch.time_step
            feats["prosody_f0_slope_abs_mean"] = float(np.mean(np.abs(np.diff(voiced) / dt)))
        else:
            feats["prosody_f0_slope_abs_mean"] = np.nan

        # Voiced-segment timing as a speech-rate / hesitancy proxy.
        mask = (f0 > 0).astype(int)
        edges = np.diff(np.concatenate(([0], mask, [0])))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        seg_lengths = (ends - starts) * pitch.time_step
        dur = snd.get_total_duration()
        feats["prosody_n_voiced_segments"] = float(len(seg_lengths))
        feats["prosody_voiced_rate_per_s"] = float(len(seg_lengths) / dur) if dur > 0 else np.nan
        feats["prosody_mean_voiced_seg_s"] = float(np.mean(seg_lengths)) if seg_lengths.size else 0.0
        feats["prosody_pause_fraction"] = float(1.0 - np.mean(mask))
    except Exception:
        feats.update(_nan_dict([f"prosody_f0_{s}" for s in SUMMARY_STATS]))
        feats.update(_nan_dict(
            ["prosody_voiced_fraction", "prosody_f0_slope_abs_mean", "prosody_n_voiced_segments",
             "prosody_voiced_rate_per_s", "prosody_mean_voiced_seg_s", "prosody_pause_fraction"]
        ))

    # --- jitter / shimmer (point-process based) ---
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", p["pitch_floor"], p["pitch_ceiling"])
        lo, hi = p["jitter_shimmer_period_floor"], p["jitter_shimmer_period_ceiling"]
        mpf, maf = p["max_period_factor"], p["max_amplitude_factor"]
        feats["prosody_jitter_local"] = call(pp, "Get jitter (local)", 0, 0, lo, hi, mpf)
        feats["prosody_jitter_rap"] = call(pp, "Get jitter (rap)", 0, 0, lo, hi, mpf)
        feats["prosody_jitter_ppq5"] = call(pp, "Get jitter (ppq5)", 0, 0, lo, hi, mpf)
        feats["prosody_jitter_ddp"] = call(pp, "Get jitter (ddp)", 0, 0, lo, hi, mpf)
        args = [[snd, pp], "Get shimmer (local)", 0, 0, lo, hi, mpf, maf]
        feats["prosody_shimmer_local"] = call(*args)
        feats["prosody_shimmer_localdb"] = call([snd, pp], "Get shimmer (local_dB)", 0, 0, lo, hi, mpf, maf)
        feats["prosody_shimmer_apq3"] = call([snd, pp], "Get shimmer (apq3)", 0, 0, lo, hi, mpf, maf)
        feats["prosody_shimmer_apq5"] = call([snd, pp], "Get shimmer (apq5)", 0, 0, lo, hi, mpf, maf)
        feats["prosody_shimmer_apq11"] = call([snd, pp], "Get shimmer (apq11)", 0, 0, lo, hi, mpf, maf)
        feats["prosody_shimmer_dda"] = call([snd, pp], "Get shimmer (dda)", 0, 0, lo, hi, mpf, maf)
    except Exception:
        feats.update(_nan_dict([k for k in PROSODY_SCALAR_KEYS if "jitter" in k or "shimmer" in k]))

    # --- harmonicity ---
    try:
        harm = call(snd, "To Harmonicity (cc)", 0.01, p["pitch_floor"], 0.1, 1.0)
        hnr = np.array(harm.values).ravel()
        feats.update(_summarise(hnr[hnr > -200], "prosody_hnr", ("mean", "std", "min", "max", "median")))
    except Exception:
        feats.update(_nan_dict([f"prosody_hnr_{s}" for s in ("mean", "std", "min", "max", "median")]))

    # --- formants ---
    # Praat's aggregate queries ("Get mean" etc.) are used rather than a
    # per-frame Python loop: same numbers, ~50x fewer interpreter round trips.
    try:
        formant = call(snd, "To Formant (burg)", 0.0, p["n_formants"], p["max_formant_hz"], 0.025, 50.0)
        for f_i in range(1, min(int(p["n_formants"]), 5) + 1):
            feats[f"prosody_f{f_i}_mean"] = call(formant, "Get mean", f_i, 0, 0, "hertz")
            feats[f"prosody_f{f_i}_std"] = call(formant, "Get standard deviation", f_i, 0, 0, "hertz")
            feats[f"prosody_f{f_i}_median"] = call(formant, "Get quantile", f_i, 0, 0, "hertz", 0.5)
            if f_i <= 3:
                feats[f"prosody_f{f_i}_bw_median"] = call(
                    formant, "Get quantile of bandwidth", f_i, 0, 0, "hertz", 0.5
                )
    except Exception:
        for f_i in range(1, 6):
            feats.update(_nan_dict([f"prosody_f{f_i}_{s}" for s in ("mean", "std", "median")]))
            if f_i <= 3:
                feats[f"prosody_f{f_i}_bw_median"] = np.nan

    # --- intensity ---
    try:
        intensity = snd.to_intensity(minimum_pitch=p["pitch_floor"])
        iv = np.array(intensity.values).ravel()
        feats.update(_summarise(iv, "prosody_intensity"))
    except Exception:
        feats.update(_nan_dict([f"prosody_intensity_{s}" for s in SUMMARY_STATS]))

    # --- smoothed cepstral peak prominence (breathiness / voice quality) ---
    try:
        pc = call(snd, "To PowerCepstrogram", 60.0, 0.002, 5000.0, 50.0)
        feats["prosody_cpps"] = call(
            pc, "Get CPPS", False, 0.02, 0.0005, 60.0, 330.0, 0.05,
            "Parabolic", 0.001, 0.05, "Straight", "Robust",
        )
    except Exception:
        feats["prosody_cpps"] = np.nan

    return feats


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def extract_one(path: str, fcfg: dict) -> dict[str, Any]:
    """Full feature vector for one audio file, plus extraction bookkeeping."""
    sr = fcfg["audio"]["sample_rate"]
    rec: dict[str, Any] = {"stim_path": path}
    try:
        y = decode(path, sr=sr, mono=fcfg["audio"]["mono"])
    except Exception as exc:
        rec.update({"extract_ok": False, "extract_error": f"decode: {exc}"[:200]})
        return rec

    # A byte-level fingerprint of the *decoded* signal. This is what lets us
    # verify that roundtrip_wav and mp4_aac64 really are the same waveform in a
    # different wrapper — the container-vs-codec contrast the study rests on.
    rec["sha256_decoded_pcm"] = hashlib.sha256(y.tobytes()).hexdigest()
    rec["duration_s"] = float(len(y) / sr)
    rec["n_samples"] = int(len(y))
    rec["peak_amplitude"] = float(np.max(np.abs(y))) if y.size else 0.0
    rec["is_silent"] = bool(rec["peak_amplitude"] < 1e-4)

    if rec["is_silent"] or len(y) < fcfg["frame"]["n_fft"]:
        rec.update({"extract_ok": False, "extract_error": "silent or too short"})
        return rec

    # Apple's Accelerate BLAS leaks FP status flags out of matmul, so numpy
    # reports divide/overflow warnings for arithmetic that is in fact finite.
    try:
        with np.errstate(all="ignore"):
            rec.update(spectral_features(y, sr, fcfg))
            rec.update(mfcc_features(y, sr, fcfg))
            rec.update(chroma_features(y, sr, fcfg))
            rec.update(prosody_features(y, sr, fcfg))
        rec["extract_ok"] = True
        rec["extract_error"] = ""
    except Exception as exc:
        rec.update({"extract_ok": False, "extract_error": f"{type(exc).__name__}: {exc}"[:200]})
    return rec


def feature_columns(record: dict[str, Any]) -> list[str]:
    """Names of the modelling columns in an extracted record."""
    skip = {
        "stim_path", "extract_ok", "extract_error", "sha256_decoded_pcm",
        "n_samples", "is_silent",
    }
    return [k for k in record if k not in skip]
