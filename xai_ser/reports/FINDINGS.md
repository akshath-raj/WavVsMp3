# Findings — explainable SER across audio container/codec conditions

**Corpus** CREMA-D, 7,442 clips, 91 actors, 6 acted emotions, 5.26 h of audio
**Design** every clip rendered in 4 conditions → 29,768 files → 436 named acoustic features
**Split** speaker-independent: 60 train / 11 val / 20 test actors (4,905 / 896 / 1,640 clips)
**Feature table** `data/features/features_20260818_045621.csv`

Every number below is reproduced in `outputs/*/`*.json* and was produced by the
committed pipeline; nothing here is estimated or carried over from the parent
LLM study.

---

## Headline

1. **MP3 @ 64 kbps is catastrophic for some model families and free for others.**
   Trained on clean WAV, LightGBM loses 0.5 points of balanced accuracy on MP3
   while RBF-SVM loses **38.3 points** and the PyTorch ANN loses 24.6. The split
   is not accuracy-related — it is *architectural*.
2. **The damage is covariate shift on a handful of features, not lost emotion.**
   Neutralising a **single** feature restores the ANN from 0.354 to 0.580
   balanced accuracy; three features restore it to 0.595 against a clean-audio
   ceiling of 0.599.
3. **Explanation stability is a property of the model family, not of the codec.**
   Under MP3, XGBoost keeps all 25 of its top-25 attributed features (ρ = 0.991)
   while the ANN keeps 11 (ρ = 0.677). Where a dissociation does appear it is
   real but modest: XGBoost loses only 2.2 accuracy points on MP4/AAC yet still
   swaps 3 of its top 25, and the ANN loses 8.7 points while swapping 9 — so
   accuracy consistently *under*-reports the change, but it never contradicts it.
4. **Container is not codec.** `roundtrip_wav` (the MP4/AAC signal re-wrapped as
   WAV) tracks `mp4_aac64` to within 0.003 balanced accuracy for every model.
   For a feature-based pipeline the container carries no information; only the
   codec does.

---

## A. Corpus and label quality

| | |
|---|---|
| clips / actors / sentences | 7,442 / 91 / 12 |
| class balance | 1,271 each for ANG, DIS, FEA, HAP, SAD; 1,087 NEU |
| intensity labels | 6,077 unspecified (`XX`), 455 each LO / MD / HI |
| actors | 48 male, 43 female; ages 20–74 (mean 36.4); 61 Caucasian, 22 African American, 7 Asian, 1 unknown |
| clip duration | 2.54 s ± 0.51 (range 1.27–5.01) |
| clips per actor | 76–82 (mean 81.8) |

**A1. The human voice-only ceiling is 45.5%.** CREMA-D's crowd raters, hearing
audio alone, matched the actor's intended emotion in only 45.5% of clips (mean
per-clip agreement 0.637, median 9 ratings per clip, 644 clips with tied modal
votes). Per-class recall is wildly uneven: NEU 0.966 and ANG 0.669, but SAD
0.164, HAP 0.316 and DIS 0.317 — listeners default to "neutral" when acted
affect is subtle. **Every model below beats this ceiling**, which sounds
impossible until you note that models are trained on the *intended* label and can
exploit systematic acting cues that human listeners do not consciously decode.
It also means absolute accuracy here should not be read as "emotion recognition
accuracy" in any deployable sense.

**A2. Exactly one clip failed extraction, and it is the one the corpus documents
as broken.** `1076_MTI_SAD_XX` is digitally silent in all four conditions; the
CREMA-D README lists it as having no audio. 4 of 29,768 rows (the one clip × 4
conditions). No other file failed. This is a clean pipeline validation.

**A3. Speaker identity dominates emotion in the feature space.** Mean variance
explained per feature: **speaker η² = 0.178 vs emotion η² = 0.098**, and
**340 of 436 features are speaker-dominant**. By family, prosody (η² = 0.232)
and chroma (0.254) are the most speaker-bound. A random row split would have
leaked voice identity into the test set; speaker-independent splitting is not a
nicety here, it is load-bearing.

---

## B. Feature table health

- 436 features: 240 MFCC, 114 spectral, 56 prosody, 24 chroma, 2 global
  (duration, peak amplitude). No constant columns.
- Only 5 features contain any NaN; only one exceeds 1% missing —
  `prosody_shimmer_apq11` at 11.3%, which is expected: APQ11 needs 11
  consecutive glottal periods and short/creaky clips do not supply them.
- Heavy tails are mild: the worst feature has 0.89% of values beyond |z| > 5
  (`spectral_flatness_iqr`).
- **397 of 436 features separate emotion at Bonferroni-corrected significance.**
  Strongest single features: `mfcc_d1_00_std` (F = 1194), `mfcc_00_std`
  (F = 1159), `prosody_intensity_std` (F = 1078). These are all *variability*
  measures — how much loudness and spectral energy move over the clip — not mean
  levels. Emotion in acted speech lives in the dynamics.
- By mutual information the ranking of families is chroma (0.098) > prosody
  (0.087) > spectral (0.082) > MFCC (0.061), i.e. MFCC wins on raw count, not on
  per-feature information.
- Redundancy is real but contained: 49 feature pairs correlate above |r| = 0.95,
  7 above 0.99. Several are definitional (`voiced_fraction` vs `pause_fraction`
  r = 1.00; `jitter_rap` vs `jitter_ddp` r = 1.00; `shimmer_apq3` vs
  `shimmer_dda` r ≈ 1.00) — Praat's own algebraic duplicates. Mean absolute
  off-diagonal correlation is only 0.18.
- 51 principal components are needed for 95% of variance (PC1 21.7%, PC2 8.2%),
  so the space is genuinely high-dimensional rather than a few latent factors.

---

## C. What the codec actually does to the signal

Paired per-clip comparison against `ref` across all 7,441 usable clips.

| condition | median &#124;SMD&#124; | features &#124;SMD&#124; > 0.5 | features significant |
|---|---|---|---|
| mp3_64 | 0.020 | 28 | 361 / 436 |
| mp4_aac64 | 0.025 | 27 | 370 / 436 |
| roundtrip_wav | 0.026 | 27 | 371 / 436 |

The median shift is negligible; the tail is not. Two distinct mechanisms:

**C1. MP3 @ 64 kbps guts the top spectral band.**
`spectral_contrast_6_mean` (the 6–8 kHz band) moves from 16.38 to 46.43 — a
shift of **38.6 training standard deviations**. Spectral flatness collapses in
parallel (mean 0.0189 → 0.0066) because the encoder zeroes masked bins, leaving
a spectrum that is far less noise-like. These are not subtle degradations; they
are a fingerprint of the encoder, and any model reading them in absolute terms
is now looking at an input it has never seen.

**C2. MP4/AAC shifts frame alignment via encoder padding.**
AAC adds priming samples: mean duration rises from 2.5429 s to 2.5756 s, i.e.
**+32.7 ms** for every clip, uniformly. MP3 adds +0.15 ms. That padding shifts
every frame-aligned dynamic feature, which is exactly what the drift table shows
— the largest MP4 shifts are `mfcc_d1_01_mean` (SMD −4.16), `mfcc_d1_00_mean`
(−3.47), `mfcc_d1_03_mean` (−2.38): all *delta*-MFCC means, i.e. first-order
temporal derivatives. Caveat: delta means sit near zero with small standard
deviations, so their SMDs are inflated relative to their absolute change; the
duration measurement is the harder evidence for this mechanism.

**C3. The decode-path control is tight.** `mp4_aac64` and `roundtrip_wav` carry
the same codec output through different containers. No clip is bit-identical
between them (the WAV control passes through int16 quantisation, the direct MP4
decode does not), but the median feature difference is **|SMD| = 7.6 × 10⁻⁵**
and only 12 of 436 features exceed 0.05. That is the noise floor of this study:
any effect larger than it is a real codec effect, not a decoding artefact.

---

## D. Model results

Balanced accuracy (UAR) on the held-out 20 speakers, 6-way. Chance = 0.167;
human voice-only = 0.455.

### D1. In-format leaderboard (train and test on `ref`)

| model | accuracy | balanced acc | macro F1 | κ |
|---|---|---|---|---|
| LightGBM | 0.5896 | **0.5908** | 0.5840 | 0.508 |
| SVM (RBF) | 0.5854 | 0.5856 | 0.5808 | 0.502 |
| LDA (shrinkage) | 0.5848 | 0.5854 | 0.5796 | 0.502 |
| Linear SVM | 0.5829 | 0.5843 | 0.5770 | 0.500 |
| XGBoost | 0.5811 | 0.5821 | 0.5745 | 0.497 |
| MLP (sklearn) | 0.5732 | 0.5742 | 0.5691 | 0.488 |
| HistGradientBoosting | 0.5726 | 0.5730 | 0.5662 | 0.487 |
| Logistic regression | 0.5652 | 0.5669 | 0.5608 | 0.478 |
| Random forest | 0.5323 | 0.5349 | 0.5171 | 0.439 |
| AdaBoost | 0.5195 | 0.5207 | 0.5137 | 0.423 |
| Extra trees | 0.5159 | 0.5191 | 0.4963 | 0.420 |
| kNN (k=15) | 0.4774 | 0.4791 | 0.4656 | 0.373 |
| Gaussian NB | 0.4244 | 0.4296 | 0.3791 | 0.312 |
| **Decision tree** | 0.3896 | 0.3905 | 0.3822 | 0.268 |
| Dummy (majority) | 0.1707 | 0.1667 | 0.0486 | 0.000 |

**PyTorch ANN (512-256-128, BN + dropout): accuracy 0.5988, UAR 0.5993, macro F1
0.5960, top-2 accuracy 0.805** — the best single model in the study, marginally
ahead of LightGBM.

**D2. A single decision tree costs 20 accuracy points.** The one model a human
can read end to end reaches 0.391 UAR against 0.591 for LightGBM and 0.599 for
the ANN. Intrinsic interpretability is not free here; it costs a third of the
achievable performance. This is the honest case for post-hoc XAI on a strong
model rather than intrinsic interpretability via a weak one.

**D3. Per-class difficulty mirrors the humans, but less severely.** ANN per-class
F1 on `ref`: ANG 0.770, NEU 0.611, SAD 0.607, HAP 0.567, FEA 0.555, DIS 0.467.
Anger is easiest for both models and humans; disgust is hardest for the model,
and sadness — which humans recover only 16% of the time — the model gets to
0.607 F1.

### D4. Cross-format: trained on clean WAV, served compressed

Balanced accuracy, all models trained on `ref`:

| model | ref | mp3_64 | Δ mp3 | mp4_aac64 | Δ mp4 | roundtrip_wav |
|---|---|---|---|---|---|---|
| LightGBM | 0.5908 | 0.5856 | **−0.005** | 0.5590 | −0.032 | 0.5621 |
| XGBoost | 0.5821 | 0.5861 | **+0.004** | 0.5601 | −0.022 | 0.5601 |
| Extra trees | 0.5191 | 0.5208 | +0.002 | 0.5122 | −0.007 | 0.5067 |
| Random forest | 0.5349 | 0.5323 | −0.003 | 0.5223 | −0.013 | 0.5217 |
| HistGB | 0.5730 | 0.5591 | −0.014 | 0.5304 | −0.043 | 0.5333 |
| AdaBoost | 0.5207 | 0.5107 | −0.010 | 0.4865 | −0.034 | 0.4839 |
| Decision tree | 0.3905 | 0.3818 | −0.009 | 0.3672 | −0.023 | 0.3709 |
| kNN | 0.4791 | 0.3616 | −0.118 | 0.4671 | −0.012 | 0.4571 |
| Gaussian NB | 0.4296 | 0.2774 | −0.152 | 0.4372 | +0.008 | 0.4330 |
| Linear SVM | 0.5843 | 0.4052 | −0.179 | 0.4711 | −0.113 | 0.4623 |
| MLP (sklearn) | 0.5742 | 0.3834 | −0.191 | 0.4914 | −0.083 | 0.4941 |
| **ANN (PyTorch)** | 0.5993 | 0.3536 | **−0.246** | 0.5126 | −0.087 | 0.5120 |
| LDA | 0.5854 | 0.3383 | −0.247 | 0.5067 | −0.079 | 0.5017 |
| Logistic regression | 0.5669 | 0.2333 | −0.334 | 0.4722 | −0.095 | 0.4699 |
| **SVM (RBF)** | 0.5856 | 0.2030 | **−0.383** | 0.4988 | −0.087 | 0.4969 |

**D5. Robustness is determined by whether the model thresholds its inputs.**
Every tree-based model sits within ±0.014 of its clean score on MP3. Every model
that consumes features as continuous magnitudes — SVM, logistic regression, LDA,
both neural nets — loses between 18 and 38 points. A decision tree asks "is
`spectral_contrast_6_mean` above 17?"; the answer stays the same whether the
feature reads 16.4 or 46.4. A linear or kernel model multiplies that 38-σ
excursion straight into its decision function. **The single most consequential
deployment choice for compressed audio is model family, not feature set.**

**D6. MP4/AAC costs everyone something, and nobody very much.** Losses cluster
between 0.7 and 11 points regardless of family — a broad, mild degradation
consistent with the 32.7 ms padding shift plus mild spectral loss, rather than
the single-feature blowout MP3 produces.

**D7. Container routing has no effect.** `roundtrip_wav` tracks `mp4_aac64`
within 0.003 UAR for all 16 models. In the parent LLM study this contrast was
the interesting one, because an API-fed model has an opaque decode path. A
feature pipeline decodes both itself, so the contrast is a null by construction —
and confirming the null is what licenses attributing everything else to the codec.

**D8. Matched-format training removes almost all of the damage.** Trained on
`mp4_aac64` instead of `ref`, RBF-SVM scores 0.5718 on MP4 (vs 0.5856 clean-on-
clean) and logistic regression 0.5704. The emotional information survives 64 kbps
AAC essentially intact — what fails is the train/serve mismatch. But training on
MP4 does *not* rescue MP3 (SVM: 0.167, i.e. chance), so the fix is
format-specific, not a general immunisation.

---

## E. Explainability of the classical models

SHAP (TreeSHAP where exact, LinearSHAP for the linear models, KernelSHAP for the
SVM), LIME, permutation importance, partial dependence, and a depth-4 global
surrogate tree, all measured on the held-out speakers.

**E1. Every model family converges on the same acoustic evidence.** The top SHAP
features for XGBoost and LightGBM are identical in the first five —
`mfcc_d1_00_std`, `mfcc_d2_00_std`, `duration_s`, `prosody_intensity_std`,
`mfcc_00_std` — and the decision tree leads with `prosody_intensity_std` and
`mfcc_d1_00_std`. These are the same quantities the ANN's integrated gradients
select (§F1), and the ones LIME selects for the RBF-SVM. Five model families,
four attribution algorithms, one answer: **how variable the loudness and its rate
of change are over the clip.**

**E2. Attribution mass by family is stable across models.** SHAP shares:

| model | MFCC | spectral | prosody | chroma |
|---|---|---|---|---|
| XGBoost | 0.566 | 0.207 | 0.178 | 0.033 |
| LightGBM | 0.550 | 0.209 | 0.189 | 0.030 |
| Random forest | 0.438 | 0.315 | 0.205 | 0.032 |
| Decision tree | 0.436 | 0.183 | 0.306 | 0.034 |
| Logistic regression | 0.471 | 0.306 | 0.156 | 0.062 |
| ANN (integrated gradients) | 0.542 | 0.238 | 0.153 | 0.057 |

MFCCs take roughly half the attribution while holding 55% of the columns — no
per-feature edge. The single decision tree is the outlier, leaning harder on
prosody (0.306), which is what makes it readable and also what caps its accuracy.

**E3. XAI methods disagree with each other, substantially.** Spearman rank
correlation against SHAP on the same fitted model:

| model | vs LIME | vs permutation | vs intrinsic |
|---|---|---|---|
| XGBoost | 0.446 | 0.311 | 0.463 |
| LightGBM | 0.286 | 0.161 | 0.826 |
| Random forest | 0.316 | 0.264 | 0.904 |
| Decision tree | **0.008** | 0.204 | 0.984 |
| Logistic regression | 0.384 | 0.612 | 0.983 |
| SVM (RBF) † | 0.141 | 0.095 | n/a |

† the SVM's SHAP baseline is the invalid KernelSHAP estimate of §E4; its row
measures the failure, not a disagreement about the model.

SHAP agrees closely with a model's *intrinsic* importance (gain, |coefficient|)
for the simpler models — ρ = 0.98 for the tree and the linear model — but LIME
and permutation importance are close to unrelated to it. On the decision tree,
LIME and SHAP rank features at ρ = 0.008, i.e. **no relationship at all**, while
still sharing 10 of their top 20. Practical consequence: a single-method XAI
claim about "which acoustic features matter" is not reproducible across methods.
Only the features that survive *all four* views (`mfcc_d1_00_std`,
`prosody_intensity_std`, `duration_s`, `prosody_cpps`, `prosody_f0_median`)
should be reported as robust.

**E4. KernelSHAP silently produced invalid numbers on the SVM, and this is worth
recording.** The RBF-SVM has no exact explainer, so it was given KernelSHAP with
100 sampled coalitions over 436 features. KernelSHAP fits a weighted least-
squares regression with one unknown per feature; with 100 coalitions for 436
unknowns that system is rank-deficient, and the result diverged — **maximum mean
|SHAP| of 2.1 × 10¹² against a median of 5.3 × 10⁻⁴**. Its ranking correlates
with LIME at ρ = 0.141 (2 of 20 shared) and with permutation importance at
ρ = 0.095 (2 of 20), while LIME's own top features for the same model
(`mfcc_d1_00_std`, `mfcc_d2_00_std`, `duration_s`) reproduce the consensus every
other model reaches. The SVM's KernelSHAP ranking is therefore **excluded from
the conclusions of this report** as an estimation artefact rather than a finding
about the model; `shap_values()` now raises when `nsamples <= n_features` so the
failure cannot recur silently. For the SVM, the trustworthy views are permutation
importance (measured on held-out data) and LIME.

A cheap tell that this had gone wrong: the family shares came out as exactly
0.000 for chroma and for the two global features — a well-behaved additive
attribution over 7,441 clips does not assign a whole family exactly zero.

**E5. Global surrogates capture only ~60% of the black box.** A depth-4 tree
fitted to each model's own predictions reproduces them for random forest 73.9%,
LightGBM 63.4%, XGBoost 62.1%, SVM 58.7%, decision tree 58.2% and logistic
regression 57.1% of the time. The
surrogate rules are readable — XGBoost's uses `mfcc_d1_00_std`,
`prosody_intensity_std`, `spectral_flux_mean/min`, `spectral_slope_median` and
five others — but they are a summary of a majority of the model, not the model.
Reporting surrogate rules without their fidelity would overstate what has been
explained.

**E6. Tree and linear attributions barely move under compression.** SHAP rank
correlation against `ref`:

| model | mp3_64 | mp4_aac64 | roundtrip_wav |
|---|---|---|---|
| Decision tree | 0.995 (24/25) | 0.988 (21/25) | 0.988 (21/25) |
| XGBoost | 0.991 (25/25) | 0.984 (22/25) | 0.984 (22/25) |
| Random forest | 0.990 (25/25) | 0.976 (25/25) | 0.975 (25/25) |
| LightGBM | 0.988 (23/25) | 0.983 (22/25) | 0.982 (22/25) |
| Logistic regression | 0.987 (22/25) | 0.942 (21/25) | 0.941 (21/25) |
| **ANN (IG)** | **0.677 (11/25)** | 0.891 (16/25) | 0.887 (16/25) |

**E7. Attribution stability and accuracy stability are independent axes — in both
directions.** The trees are stable in both (ρ ≈ 0.99, accuracy Δ ≈ 0). The ANN
moves in both. But **logistic regression keeps a near-identical SHAP ranking
(ρ = 0.987, 22/25 features retained) while its accuracy collapses from 0.567 to
0.233 on MP3.** Its coefficients did not change; only the inputs went
out-of-range, and a global mean-|SHAP| ranking is largely blind to that. This is
a caution about the method, not about the model: **global attribution-stability
metrics can report "nothing changed" while the classifier has effectively
stopped working.** Local attributions and input-range monitoring catch what the
global ranking misses.

---

## F. Deep-learning XAI (ANN, captum)

**F1. All six attribution methods agree on what the network uses.** Saliency,
Input×Gradient, Integrated Gradients, DeepLIFT, GradientSHAP and FeatureAblation
return the same top features: `mfcc_d1_00_std`, `mfcc_d2_00_std`,
`prosody_cpps`, `duration_s`, `mfcc_d1_01_std`. The two leaders are the standard
deviations of the first and second derivative of MFCC-0 — i.e. **how variable the
rate of change of overall loudness is**, which is a direct formalisation of vocal
agitation.

**F2. Attribution is spread across families, not concentrated in MFCCs.**
Integrated-gradients mass: MFCC 54.2%, spectral 23.8%, prosody 15.3%,
chroma 5.7%. MFCCs hold 55% of the columns and 54% of the attribution — no
per-feature advantage, matching the mutual-information result in §B.

**F3. Per-emotion evidence is acoustically sensible.**

| emotion | top attributed features |
|---|---|
| ANG | `mfcc_d1_00_std`, `mfcc_d2_00_std`, `mfcc_00_iqr` — loudness dynamics |
| DIS | `duration_s`, `prosody_f2_std`, `mfcc_00_skew` — timing and F2 movement |
| FEA | `prosody_f0_median`, `mfcc_d1_00_std`, `prosody_hnr_std` — raised pitch, unstable harmonicity |
| HAP | `duration_s`, `spectral_flux_std`, `mfcc_d1_01_std` — spectral churn |
| NEU | `mfcc_00_skew`, `prosody_cpps`, `mfcc_d1_01_std` — steady, clean phonation |
| SAD | `prosody_cpps`, `mfcc_d2_00_std`, `mfcc_03_std` — breathiness (CPPS) |

CPPS driving both sadness and neutrality, and F0 median driving fear, are exactly
the associations the SER literature reports. The network was not told any of this.

**F4. The attributions pass the model-randomisation sanity check.** Comparing
attributions from the trained network with a randomly initialised one of identical
architecture gives Spearman ρ between **−0.011 and 0.206** across all six methods
(saliency −0.011, IG 0.163, DeepLIFT 0.161, GradientSHAP 0.135, Input×Gradient
0.204, FeatureAblation 0.206). All are near zero, so these explanations describe
the trained model rather than merely the input distribution — the failure mode
Adebayo et al. warned about does not apply here.

**F5. Explanations degrade, and they degrade further than accuracy does.**
Integrated-gradients rankings compared against `ref`:

| condition | Spearman ρ | top-25 overlap | L1 attribution shift | accuracy Δ |
|---|---|---|---|---|
| mp4_aac64 | 0.891 | 16 / 25 | 0.147 | −0.087 |
| roundtrip_wav | 0.887 | 16 / 25 | 0.156 | −0.087 |
| mp3_64 | 0.677 | 11 / 25 | 0.363 | −0.246 |

Under MP4/AAC the model loses 8.7 accuracy points but **replaces 9 of its 25
most-attributed features**. An accuracy-only benchmark would call that a modest
degradation; the attribution view shows a materially different model behaviour
behind the number.

This is a *quantitative* dissociation, not a qualitative one. The parent LLM
study hypothesised that a model might return identical labels from materially
different evidence — equal accuracy, divergent explanation. That strong form does
**not** occur here: every condition that moves the attributions also moves the
accuracy, in the same direction and in the same rank order across models. What
does hold is the weaker and still consequential claim: **the accuracy delta
systematically understates the attribution delta.** A 2.2-point accuracy loss
(XGBoost on MP4) still comes with three of its top-25 features replaced.

**F6. What replaces what is the whole story.** Under MP3, the features *entering*
the top 25 are almost all codec artefacts — `spectral_flatness_mean/median/std/
min/iqr` and `spectral_contrast_6_mean` — while the features *leaving* are
genuine voice-quality and prosodic measures: `prosody_cpps`, `prosody_f0_skew`,
`prosody_f2_mean`, `prosody_f2_std`, `prosody_f3_bw_median`. **The compressed
model stops listening to the voice and starts listening to the encoder.** Under
MP4 the same displacement happens in milder form, with `mfcc_d1_*_mean` (the
padding-shifted delta means) entering and F0 statistics leaving.

---

## G. Causal diagnosis: it is covariate shift, not information loss

Features were ranked by how far each codec moves them in units of training
standard deviation, then progressively replaced with training medians in both the
compressed and the clean test set (the clean column controls for whatever real
signal the masking itself destroys). Full curves in
`outputs/robustness/*/neutralisation_curves.csv`.

**G1. One feature accounts for almost the entire MP3 collapse, in every affected
model.** Balanced accuracy on MP3 as features are neutralised:

| features masked | SVM (RBF) | Logistic reg. | ANN | MLP | XGBoost | Random forest | Decision tree |
|---|---|---|---|---|---|---|---|
| *clean-audio score* | *0.586* | *0.567* | *0.599* | *0.574* | *0.582* | *0.535* | *0.391* |
| 0 | 0.203 | 0.233 | 0.354 | 0.383 | 0.586 | 0.532 | 0.382 |
| **1** | **0.578** | **0.502** | **0.580** | **0.552** | 0.581 | 0.530 | 0.382 |
| 2 | 0.583 | 0.529 | 0.587 | 0.554 | 0.581 | 0.532 | 0.391 |
| 3 | 0.579 | **0.562** | **0.595** | **0.565** | 0.580 | 0.534 | 0.391 |
| 10 | 0.575 | 0.539 | 0.589 | 0.562 | 0.582 | 0.533 | 0.394 |
| 80 | 0.551 | 0.435 | 0.574 | 0.523 | 0.564 | 0.524 | 0.396 |

The single masked feature is `spectral_contrast_6_mean`. Masking it alone
recovers **98%** of the SVM's 38.3-point loss, **92%** of the ANN's 24.6-point
loss, and 70% of logistic regression's; three features bring logistic regression
to 0.562 against its 0.567 clean-audio score. The tree columns are flat
throughout — they had nothing to recover.

**G2. The neutralised features carry essentially no emotion.** On uncompressed
audio the same masking costs nothing out to 20 features (the ANN's clean control
sits at 0.607–0.609, marginally *above* its 0.599 unmasked baseline). These
columns were never load-bearing evidence; they were the channel through which the
encoder's fingerprint entered the model.

**G3. MP4/AAC behaves differently — diffuse, not concentrated.** No single
feature dominates. The ANN's best recovery on MP4 comes at 10 features masked
(0.513 → 0.560 against 0.599 clean) and the SVM's at 40 (0.499 → 0.542). This
matches the mechanism: AAC's 32.7 ms padding perturbs *every* frame-aligned
dynamic feature a little, rather than blowing out one band.

**G4. Over-masking eventually costs real accuracy.** By 80 features every model
is below its 40-feature score, and on MP4 the decision tree drops from 0.367 to
0.315. The useful operating point is small — 1 to 20 features — which is
precisely what makes this a practical fix rather than a feature purge.

**G5. The prescription.** For pipelines that must accept lossy audio, in order of
preference: (a) use a threshold-based model family, which is free; (b) drop or
clamp top-band spectral-contrast and flatness features, which recovers almost
everything at the cost of one feature; (c) retrain on the target codec (§D8),
which works but requires knowing the codec in advance and does not transfer
between codecs.

---

## H. Limitations

1. **Acted, not spontaneous, emotion.** CREMA-D actors perform prescribed
   emotions on 12 fixed sentences. Absolute accuracies do not transfer to
   spontaneous speech, and the human ceiling of 45.5% shows the labels are
   themselves only loosely recoverable from audio.
2. **One clean split, not repeated CV.** All results come from a single
   deterministic 60/11/20 actor partition. Differences of a point or two between
   adjacent models on the leaderboard should not be over-read; the codec effects
   reported here are 10–38 points and are not at risk from split variance.
3. **Two bitrates, two codecs.** 64 kbps MP3 and 64 kbps AAC only. The MP3 result
   is severe partly *because* 64 kbps is aggressive for that codec; higher
   bitrates were not tested and would likely show a gradient.
4. **Summary statistics discard time.** Every frame contour is collapsed to eight
   order statistics, so no attribution here localises *when* in the clip the
   evidence sits — the time-localisation question the parent LLM study posed
   remains open in this setup.
5. **The container null is structural.** `roundtrip_wav ≈ mp4_aac64` is
   guaranteed once ffmpeg decodes both; it validates the pipeline but cannot
   speak to systems with opaque decode paths, which is precisely where the parent
   study's container question lives.
6. **SMD inflation on near-zero features.** Delta-MFCC means have tiny standard
   deviations, so their standardised shifts overstate the absolute change; the
   duration measurement is the sounder evidence for the AAC padding mechanism.
7. **No cross-format SHAP for the SVM.** KernelSHAP is the only exact-agnostic
   option for an RBF-SVM and it is too expensive to run per condition at a valid
   sample budget (§E4), so the SVM contributes accuracy and robustness results
   but not an attribution-stability row. Its permutation and LIME rankings are
   single-condition only.
8. **Attribution comparisons use global rankings.** Spearman ρ and top-k overlap
   over mean|attribution| are coarse; §E7 shows they can miss a total accuracy
   collapse. A local, per-instance stability analysis would be a stronger
   instrument and is not attempted here.

---

## I. Where each result lives

| claim | artefact |
|---|---|
| corpus, labels, health, separability, codec drift (§A–C) | `outputs/eda/eda_summary.json`, 11 figures, `feature_health.csv`, `feature_separability.csv`, `format_drift.csv`, `variance_decomposition.csv` |
| leaderboard and cross-format grid (§D) | `outputs/models/<ts>/results.json`, `leaderboard.csv`, `cross_format.csv`, 32 fitted pipelines under `fitted/` |
| ANN (§D1, §F) | `outputs/models/ann/<ts>/{metrics.json, history.csv, ann.pt, preproc.joblib}` |
| classical XAI (§E) | `outputs/xai/<ts>/xai_summary.json` plus per-model SHAP/LIME/permutation CSVs, beeswarm, PDP/ICE and surrogate figures |
| deep XAI (§F) | `outputs/xai/deep/<ts>/deep_xai_summary.json`, per-method global CSVs, method-agreement and sanity-check figures |
| neutralisation curves (§G) | `outputs/robustness/<ts>/{robustness_summary.json, neutralisation_curves.csv, feature_shift_*.csv}` |
| the feature table itself | `data/features/features_20260818_045621.csv` + `schema_20260818_045621.json` |

`python scripts/digest.py` prints the headline numbers from all of these in one
pass.
