# xai_ser — Explainable speech emotion recognition across audio formats

A ground-up rebuild of the WavVsMp4 study using **interpretable machine learning**
instead of a closed-weights multimodal LLM.

## The question

The parent study asked whether delivering speech in a lossy container (MP4/AAC,
MP3) changes what a model relies on for emotion recognition — not merely whether
it changes accuracy. That question needs a model whose evidence you can actually
inspect. So here the classifier is a decision tree, a random forest, a boosted
ensemble, an SVM, or a small neural net trained on **named acoustic features**,
and the evidence is read off with SHAP, LIME, permutation importance, partial
dependence, surrogate trees, and gradient-based attribution.

Two claims are testable in this setup that were not testable through an API —
and the answers are in `reports/FINDINGS.md`:

1. **Accuracy robustness ≠ explanation robustness.** *Partly upheld.* The strong
   form (same accuracy, different evidence) does not occur; every condition that
   moves attributions also moves accuracy. The weaker form does: the accuracy
   delta systematically understates the attribution delta.
2. **Model family shapes robustness more than the codec does.** *Upheld, and it
   is the study's main result.* On 64 kbps MP3 the tree ensembles lose ~0 points
   of balanced accuracy while an RBF-SVM loses 38.3 and an MLP loses 24.6, from
   identical inputs and splits — because one family thresholds its features and
   the other multiplies them.

## Corpus

CREMA-D — 7,442 clips, 91 actors, 12 fixed sentences, 6 acted emotions
(ANG/DIS/FEA/HAP/NEU/SAD), with crowd ratings collected separately for
voice-only presentation (the human ceiling this project compares against).

Each clip is rendered into four conditions derived from one canonical reference:

| condition | what it is |
|---|---|
| `ref` | 16 kHz mono PCM16, EBU R128 loudness-normalised WAV |
| `mp3_64` | MP3 @ 64 kbps, encoded from `ref` |
| `mp4_aac64` | audio-only MP4/AAC @ 64 kbps, encoded from `ref` |
| `roundtrip_wav` | `mp4_aac64` decoded back to WAV — the decode-path control |

29,768 audio files in total.

## Pipeline

```bash
uv sync                                   # Python 3.11 environment
bash scripts/fetch_cremad.sh              # 7,442 source WAVs (LFS media endpoint)
uv run python -m xai_ser.transforms       # render the 4 conditions + manifest
uv run python -m xai_ser.extract          # -> data/features/features_<UTC>.csv
uv run python -m xai_ser.eda              # -> outputs/eda/
uv run python -m xai_ser.models           # -> outputs/models/<UTC>/
uv run python -m xai_ser.ann              # -> outputs/models/ann/<UTC>/
uv run python -m xai_ser.xai_tabular      # -> outputs/xai/<UTC>/
uv run python -m xai_ser.xai_deep         # -> outputs/xai/deep/<UTC>/
```

`scripts/run_all.py` chains the analysis stages once the audio exists;
`scripts/digest.py` prints the headline numbers from every stage.
`python -m xai_ser.robustness` runs the codec-shift neutralisation analysis.

See `reports/FINDINGS.md` for the full results.

## Features

436 named columns per file, in four families, each frame-level contour reduced
to eight order statistics:

- `spectral_*` — centroid, bandwidth, rolloff (85/95%), flatness, flux, entropy,
  slope, ZCR, RMS, spectral contrast, 8 band-energy ratios, and explicit
  high-frequency survival ratios above 4 kHz and 6 kHz.
- `mfcc_*` — 20 MFCCs with Δ and ΔΔ.
- `chroma_*` — 12 pitch classes (tuning pinned to 0; see `features.py`).
- `prosody_*` — Praat F0 statistics and slope, jitter (4), shimmer (6), HNR,
  formants F1–F5 with bandwidths, intensity, voiced-segment timing, and CPPS.

## Methodology guardrails

- **Speaker-independent splits.** Actors are partitioned into train/val/test and
  never shared. A random row split would leak voice identity and inflate scores.
- **Item alignment across formats.** The same clips appear in every condition, so
  a train-on-`ref` / test-on-`mp4_aac64` comparison differs only in format.
- **Human ceiling reported.** CREMA-D's voice-only crowd accuracy sets the
  realistic target; acted emotion is not perfectly recoverable from audio.
- **Attribution sanity checks.** Deep attributions are compared against a
  randomly initialised network; a method that scores the same on both is not
  describing the model.

## Layout

```
configs/     transform and feature-extraction settings
scripts/     corpus download, end-to-end runner
src/xai_ser/ pipeline modules (see module docstrings)
data/        raw audio, rendered stimuli, timestamped feature tables
outputs/     eda/, models/, xai/ — figures, tables, JSON summaries
reports/     written findings
```
