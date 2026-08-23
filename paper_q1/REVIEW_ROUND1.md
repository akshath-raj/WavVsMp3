# Peer Review Package — Round 1

**Manuscript**: "What the Model Stops Listening To: Explanation Displacement Under Perceptual Audio Coding in Speech Emotion Recognition"
**Target venue**: IEEE Transactions on Affective Computing
**Source**: `paper_q1/main.tex` (1,504 lines, 39 citations, 8 figures, 8 tables)
**Review date**: 2026-08-23
**Panel**: ARS `academic-paper-reviewer` v1.11.1, `full` mode — 5 role-separated seats + editorial synthesis
**Calibration status**: `NOT_CALIBRATED`
**Panel provenance**: all five seats executed in one session on one model family; role separation is recorded, **independence of error processes is not claimed**. Cross-model track not active (`ARS_CROSS_MODEL` unset) — correlated-error disclosure applies.

**Verification basis**: this review was conducted with read access to the underlying artefacts, not the manuscript alone. Every numeric claim flagged below was recomputed from `exp/out/*.{json,parquet}`, `xai_ser/outputs/{eda,models,xai,robustness}/`, `xai_ser/reports/FINDINGS.md`, and `research/phase6_final/final_report.md`.

---

# Phase 0 — Field Analysis & Panel Configuration

| Attribute | Determination |
|---|---|
| Primary discipline | Affective computing / speech emotion recognition |
| Secondary disciplines | Explainable AI (attribution methods); audio signal processing & perceptual coding; ML robustness / distribution shift |
| Research paradigm | Quantitative, experimental, controlled-manipulation with null calibration |
| Methodology type | Two-arm comparative benchmark experiment + mechanistic diagnosis |
| Target journal tier | Q1 (IEEE TAFFC, IF ≈ 9-11) |
| Paper maturity | Late-stage draft; complete, self-consistent, artefact-backed |

### Panel

| Seat | Configured identity |
|---|---|
| **Journal-Fit** (`EIC`) | TAFFC associate editor; multimodal affect recognition; screens for scope fit, originality, and whether claims survive their own limitations section |
| **R1 — Methodology** | Statistical/experimental methodologist in speech ML; specialises in evaluation validity, power, control design, and null/baseline calibration |
| **R2 — Domain** | Audio XAI researcher; attribution-method validity, GeMAPS/openSMILE feature traditions, codec-robustness literature |
| **R3 — Perspective** | Deployment/audit-facing practitioner; ML-systems monitoring, regulated affective-computing deployments, reproducibility policy |
| **DA — Devil's Advocate** | Fixed seat; core-argument stress test |

---

# Phase 1 — Seat Reports

---

## Seat 1 — Journal-Fit Reviewer

### Review Focus
Scope alignment with TAFFC, originality of the contribution, and whether the headline claims as stated in the title/abstract are the claims the evidence supports.

### Recommendation
**Major Revision**

### Confidence
4 — affective computing evaluation is core expertise; occlusion-estimator statistics are adjacent.

### Summary Assessment
The manuscript asks whether perceptual audio coding changes *what a speech emotion model relies on*, as distinct from *what it concludes*, and answers it on two model classes — a commercial audio LLM probed by occlusion, and fifteen feature-based classifiers probed by SHAP/LIME/permutation/Captum — sharing one canonical reference signal and one MP3-64 manipulation. The framing is excellent and squarely in TAFFC scope: affective systems are exactly the class now required to surface rationales, and the delivery-format degree of freedom is genuinely unrecorded in this literature. The strongest material is the methodological discipline: a measured null floor in Arm A, a bit-identity-verified container contrast, a self-withdrawn prior claim, and a disclosed KernelSHAP rank-deficiency failure. That combination is rarer than the empirical result and is, in my judgement, the paper's real contribution.

The gap is between the abstract and Limitation 4. The abstract asserts "no detectable loss of discriminability" and Contribution 3 asserts the strong Ghorbani-style dissociation; Limitation 4 concedes the accuracy decline is directionally consistent across the whole bitrate ladder and merely underpowered at n=50. Those are different papers. A TAFFC readership will read the abstract. Second, the manuscript is materially narrower than the work behind it: a pre-declared intelligibility control was run and is not reported, and a positive construct-validity result on the six-way task is omitted entirely. Both bear on the paper's own argument.

### Strengths

**S1: The question is correctly posed and correctly scoped.**
The two-estimand decomposition in §I-A ("does coding degrade performance" vs. "does coding alter the evidential basis") is the right cut, and the paper never lets the second collapse into the first.
**Evidence Anchor**: `text: §I-A "Does coding alter the evidential basis of the model's output, independently of whether the output changes?"`

**S2: Null calibration in Arm A is a genuine methodological contribution.**
Measuring that an inaudible 1-LSB dither already dissipates ~30% of the attribution rank ordering (ρ = 0.699) reframes every perturbation-attribution claim in audio. To my knowledge no prior audio-XAI paper reports this quantity.
**Evidence Anchor**: `table: Table IV — null (1 LSB dither) row, ρ = 0.699, mean |Δ| = 0.0459`

**S3: Self-correction is executed, not gestured at.**
§VII-F withdraws a previously headline container effect on a stated principle ("a determinism check is not a null control"), and separately excludes the SVM's KernelSHAP ranking as an estimation artefact. Both cut against the authors' interest.
**Evidence Anchor**: `text: §VII-F "We withdraw the claim as unsupported rather than disproven."`

**S4: Full numeric traceability.**
Every value I sampled from Tables II-VIII reproduced from the committed artefacts, including the two corrections the authors applied against their own findings document (container median 0.003 with maximum 0.010; fifteen trained models, not sixteen).
**Evidence Anchor**: `dataset: exp/out/arousal_results.json and xai_ser/outputs/models/20260818_045719/cross_format.csv — all sampled table values reproduce`

### Weaknesses

**W1: The abstract states the strong dissociation as established; Limitation 4 states it is not.**
**Problem**: The abstract says the LM "shows no detectable loss of discriminability (AUC 0.734 to 0.746, all intervals overlapping)"; Contribution 3 calls this "the strong dissociation — unchanged accuracy, changed evidence." Limitation 4 says: "The accuracy decline is directionally consistent across the bitrate ladder but non-significant at every bitrate." Accuracy fell 0.68 → 0.58 at 64 kbps, with the same sign at 32 and 128.
**Evidence Anchor**: `text: Abstract "no detectable loss of discriminability (AUC 0.734 to 0.746, all intervals overlapping)"`
**Why it matters**: The dissociation is the paper's headline. It is currently carried by a non-significant test at n = 50 that the authors themselves distrust in §IX. A reader stopping at the abstract acquires a stronger belief than the evidence licenses.
**Suggestion**: Move the AUC-invariance framing into the abstract explicitly — "sensitivity, measured as AUC, is statistically indistinguishable while accuracy declines non-significantly (0.68 → 0.58, McNemar p = .125)" — and reserve "dissociation" for the sensitivity/criterion/attribution triple, which is what the data actually shows.
**Severity**: Major
**Confidence**: 5 — venue-level claim calibration is core.

**W2: The paper under-reports its own study.**
**Problem**: A transcription/WER arm was executed across all seven conditions and is absent from the manuscript; a construct-validity analysis on the six-way task (P(gold) − P(foil) = 0.170, p = 3.5×10⁻³⁹, n = 342) is likewise absent. Both are in `exp/out/`.
**Evidence Anchor**: `absence: §IV — expected the transcription/WER control and the six-way construct-validity result; checked §IV-A, §IV-B, §IV-D, §IX, Data Availability`
**Why it matters**: Both are load-bearing for the manuscript's own argument (see R1-W1, R2-W2), and their omission makes the six-way "failure" look like a dead end rather than the criterion-vs-evidence result it actually is.
**Severity**: Major
**Confidence**: 4 — verified against artefacts.

**W3: Title over-scopes relative to Arm A's evidence.**
**Problem**: "What the Model Stops Listening To" is licensed by Arm B's named enter/leave analysis (§V-E-3), where the departing features are identifiable voice-quality descriptors. Arm A supports only "the attribution rank ordering changes"; occlusion over 16 regions cannot say what the LM stopped listening to.
**Evidence Anchor**: `text: §IX-11 "Occlusion indexes functional dependence, not mechanism. It licenses no claims about internal computation."`
**Why it matters**: The title generalises an Arm B capability to both arms; §VIII-E is explicit that the capability is Arm B's alone.
**Suggestion**: Either retitle to foreground the displacement (e.g. "Explanation Displacement Under Perceptual Audio Coding"), or state in §I that the title's phrasing is licensed by Arm B and figurative for Arm A.
**Severity**: Minor
**Confidence**: 4.

### Criterion-Bound Judgements

Calibration status: `NOT_CALIBRATED`

| Dimension | Criterion source | Judgement | Evidence anchor | Rationale | Uncertainty / scope | Decision bearing? |
|---|---|---|---|---|---|---|
| Originality | `quality_rubrics.md` § Originality | EXCEEDS | `table: Table IV` | Null-calibrated attribution stability on an audio LM is new; the container/codec separation under verified bit-identity is new | Novelty rests on a targeted, non-exhaustive search | No |
| Significance & Impact | § Significance | MEETS | `text: §VIII-A` | Reportable-property recommendation is concrete and cheap | Single system, single corpus | No |
| Argument Coherence | § Argument Coherence | PARTLY_MEETS | `text: Abstract vs §IX-4` | Headline claim and limitation disagree in strength | — | **Yes** |
| Evidence Sufficiency | § Evidence Sufficiency | PARTLY_MEETS | `absence: §IV — expected WER control; checked §IV, §IX` | Executed controls omitted | Fixable by reporting existing data | **Yes** |
| Writing Quality | § Writing Quality | EXCEEDS | `text: §V-D "A decision tree asks whether spectral_contrast_6_mean exceeds a threshold near 17"` | Mechanism prose is unusually clear | — | No |
| Literature Integration | § Literature Integration | MEETS | `text: §II-C` | Fragility/sanity-check/disagreement triad correctly assembled | See R2-W1 | No |
| Methodological Rigor | § Methodological Rigor | PARTLY_MEETS | `figure: Fig. 1 — "null B: container re-wrap"` | Null discipline is asymmetric between arms (DA-C1) | — | **Yes** |

### Questions for Authors
1. Was the WER arm excluded deliberately (e.g. judged out of scope for a coding paper) or by oversight? If deliberate, on what ground, given that §II-D's lexical-dominance argument depends on intelligibility being preserved?
2. Would you accept demoting Table VIII's transparency ordering from a finding to an observation, retaining the within-Arm-B contrast as the supported claim?
3. Is there a reason the six-way construct-validity result is not reported as the paper's cleanest instance of "argmax hides available evidence"?

### Minor Issues
- **Figures and Tables**: Fig. 2 caption reports 37.7 pooled SD; §III-D, the abstract and §XI report 38.6 training SD. Both are correct against `format_drift.csv` and the robustness output respectively, but the paper never says so.
- **Layout**: Table VIII's "Null floor" column reads 1.000 for Arm B, while Fig. 8's caption says Arm B's floor is "zero". Two referents (ρ floor vs. displacement floor), one word.

---

## Seat 2 — Peer Reviewer 1 (Methodology)

### Review Focus
Control design, estimator validity, power, and whether each reported statistic is bound to the contrast it is attributed to.

### Recommendation
**Major Revision**

### Confidence
5 — evaluation design and paired nonparametric inference are core expertise.

### Summary Assessment
Methodologically this is a careful paper with two structural problems and one traceability defect. The careful parts: single-ancestry stimulus generation, item alignment across conditions, speaker-independent splitting justified by an actual variance decomposition (η² = 0.178 speaker vs 0.098 emotion), bit-identity verification of the container contrast (150/150), a fixed-label posterior readout that provably makes response bias uninformative, and a measured perturbation null. That last is the paper's best methodological idea and it is executed correctly in Arm A.

The problems are that the idea is not executed in Arm B, that a pre-declared control was run and withheld, and that one headline statistic is paired with the wrong contrast's p-value. None is fatal; all are fixable from data already committed. Separately, the "thresholds vs. multiplies" mechanism — the paper's central Arm B explanation — is asserted over a grouping that two of its eight rows do not fit.

### Strengths

**S1: The fixed-label readout is the right design and its rationale is correct.**
Scoring P(high) rather than P(correct) makes a constant responder score AUC = 0.5 exactly, so the 86.9% `low` bias is provably uninformative about the estimand. I verified the bias rate: 868/1000... in the reported grid, `said == low` on 86.86% of trials.
**Evidence Anchor**: `text: §IV-B "under a fixed-label readout a constant responder attains AUC = 0.5 exactly, so bias is provably uninformative"`

**S2: Neutralisation carries its own control.**
Masking the same features on *uncompressed* audio (ANN 0.607-0.609 vs 0.599 unmasked) is what converts §VI from a correlation into a diagnosis. Many robustness papers omit exactly this column.
**Evidence Anchor**: `table: Table VII — clean row 0.599 against the clean-control 0.607-0.609 cited in §VI text`

**S3: Speaker-independent splitting is justified empirically, not by convention.**
**Evidence Anchor**: `text: §V-B "mean η² = 0.178 for speaker against 0.098 for emotion, with 340 of 436 features speaker-dominant"`

### Weaknesses

**W1: A pre-declared positive control was executed and is not reported.**
**Problem**: `exp/out/wer.parquet` contains per-condition word error rates for all seven conditions (ref 0.085, mp3_32 0.109, mp3_64 0.114, mp3_128 0.147, rt_mp3_64 0.128). The design document (`research/phase6_final/final_report.md` §3.6) designates the transcription arm "in advance as a positive control for intelligibility preservation." The manuscript contains no WER result and no statement that the arm was dropped.
**Evidence Anchor**: `absence: §IV — expected the pre-declared WER intelligibility control; checked §IV-A, §IV-B, §IV-D, §IX Limitations, Data Availability`
**Why it matters**: Three separate claims lean on intelligibility being preserved — the lexical-dominance reading of the six-way collapse (§IV-B), the argument that coding damages paralinguistic rather than linguistic content (§II-F lineage), and the criterion-shift interpretation (§IV-D-2). The control that adjudicates all three exists and is silent. Its numbers are also non-monotone in bitrate, which is itself relevant to §IV-D-5.
**Suggestion**: Add a WER row to Table II or a short §IV-D subsection with the per-condition means and a paired test; if the arm is judged out of scope, say so explicitly in §IX.
**Severity**: Major
**Confidence**: 5 — recomputed from the committed parquet.

**W2: A headline statistic is paired with a different contrast's p-value.**
**Problem**: §IV-D-2 reads "Marginal P(high) declines from 0.208 to 0.144, a paired shift of −0.064 (p = 6.8×10⁻⁵, Wilcoxon signed-rank)." The magnitude 0.208 → 0.144 is the **total** effect (`ref` → `mp3_64`), for which `arousal_results.json` gives mean_diff = −0.0643, **p = 7.5×10⁻⁴**. The quoted p = 6.83×10⁻⁵ belongs to the **codec** contrast (`rt_mp3_64` − `ref`), whose mean_diff is −0.0627.
**Evidence Anchor**: `text: §IV-D-2 "a paired shift of −0.064 (p = 6.8×10⁻⁵, Wilcoxon signed-rank)"`
**Why it matters**: Both contrasts are significant, so the substantive conclusion survives — but the criterion-shift result is one of three headline findings and is quoted in the abstract and conclusion. A referee checking the artefacts will find the mismatch, and the fix is one decision: report the codec contrast (−0.063, p = 6.8×10⁻⁵) or the total (−0.064, p = 7.5×10⁻⁴), consistently across abstract, §IV-D-2, §VIII-C and §XI.
**Severity**: Major
**Confidence**: 5 — both values recomputed.

**W3: A declared control's value is never reported, and it is not small.**
**Problem**: §IV-C states the mask set includes "one minimum-energy null mask that bounds the attribution attributable to the masking operation itself." The bound is never given. From `exp/out/arousal_xai.parquet` it is mean attribution **0.029** on `ref` — 31% of the most-relied-on region (f1000_2000, 0.095), and **larger than two of the sixteen real regions** (t7 = 0.018, f250_500 = 0.015).
**Evidence Anchor**: `dataset: exp/out/arousal_xai.parquet — mask_id null_lowenergy, mean attribution 0.029 on condition ref`
**Why it matters**: The paper's own thesis is that an effect is uninterpretable without its floor. §IV-D-5 reports "fourteen of sixteen regions yield positive attribution, the theoretically expected direction" as evidence the operator measures something structured — but two of those fourteen sit below the masking-artefact floor. Reporting the mask floor strengthens the claim where it holds and correctly bounds it where it does not.
**Suggestion**: Add the null-mask value to Fig. 4 as a horizontal reference line and one sentence in §IV-D-5.
**Severity**: Major
**Confidence**: 5.

**W4: Arm B has no attribution-domain null, only an accuracy-domain one.**
**Problem**: Fig. 1 declares "null B: container re-wrap, codec output held fixed, decode path varied," and §III-A calls per-arm null calibration "load-bearing." But every Arm B ρ in Table VI is computed against `ref`; the re-wrap null is evaluated only as a balanced-accuracy difference (median 0.003). ρ(roundtrip_wav, mp4_aac64) — the exact analogue of Arm A's dither floor — is never computed, and §VII-A instead asserts an analytic floor of unity because SHAP is deterministic.
**Evidence Anchor**: `figure: Fig. 1 — "null B" box specifies container re-wrap, but Table VI's roundtrip column is measured against ref, not against mp4_aac64`
**Why it matters**: Determinism of the estimator is not the same as invariance of the ranking under an informationally empty input change — which is precisely the distinction §VII-F withdraws a prior claim over. As stated, the paper applies its own prescription to one arm and exempts the other on a weaker argument than the one it rejects elsewhere.
**Suggestion**: Compute ρ(roundtrip_wav, mp4_aac64) per model from the committed rankings. On the XGBoost entries the two conditions share identical entered/left sets and ρ within 0.0004 of each other against `ref`, so the empirical floor will almost certainly land near unity — which *supports* the paper. This is a cheap win, not a threat.
**Severity**: Major
**Confidence**: 5 — the missing computation was confirmed against `outputs/xai/*/xai_summary.json`.

**W5: The "thresholds vs. multiplies" grouping does not fit two of its own rows.**
**Problem**: Table V places kNN and Gaussian NB in the "multiplies" block. kNN computes distances; Gaussian NB evaluates per-feature likelihoods. Neither multiplies a feature by a learned weight into a decision function, which is the stated mechanism in §V-D.
**Evidence Anchor**: `table: Table V — kNN (−0.118) and Gaussian NB (−0.152) grouped under "multiplies"`
**Why it matters**: The dichotomy is the paper's central Arm B explanation and is restated in the abstract ("ordered by how a model consumes its inputs"). Two rows out of eight that do not instantiate the mechanism weaken a claim that is otherwise strongly supported by the tree/linear contrast.
**Suggestion**: Rename the blocks to the property that actually separates them — "axis-aligned partitioning" vs. "magnitude-sensitive" (distance, density, or weighted-sum). The empirical split is unchanged; the mechanism claim becomes correct for all fifteen rows.
**Severity**: Major
**Confidence**: 4 — classifier mechanics, core; the taxonomy call is partly a framing judgement.

**W6: Arm A's power is disclosed but its consequences are not propagated.**
**Problem**: n = 50 items, one deployment, task/prompt/readout selected after inspecting reference-condition data (Limitation 12). The non-monotone dose-response (−0.050, −0.063, −0.055 across the bitrate ladder) is interpreted in §IV-D-5 as independently reproducing Wu et al.'s fidelity/behaviour dissociation.
**Evidence Anchor**: `text: §IV-D-5 "The absence of dose-response is not a failed manipulation; it independently reproduces the central claim of Wu et al."`
**Why it matters**: At n = 50 with overlapping CIs on all three codec contrasts, "no monotone trend" and "insufficient resolution to detect a monotone trend" are not distinguishable. The Wu et al. reading is plausible but is presented as the only reading.
**Suggestion**: State both readings and note that the psychoacoustic saturation argument in Limitation 7 (0.46 dB SNR from 64 to 128 kbps) already supplies a physical reason the ladder is compressed — that is the stronger version of the same point.
**Severity**: Minor
**Confidence**: 5.

### Criterion-Bound Judgements

| Dimension | Criterion source | Judgement | Evidence anchor | Rationale | Uncertainty / scope | Decision bearing? |
|---|---|---|---|---|---|---|
| Methodological Rigor | `quality_rubrics.md` § Methodological Rigor | PARTLY_MEETS | `figure: Fig. 1 — null B declared but not measured in the attribution domain` | Null discipline asymmetric; declared controls unreported | All repairable from committed data | **Yes** |
| Evidence Sufficiency | § Evidence Sufficiency | PARTLY_MEETS | `absence: §IV — expected WER control and null-mask value; checked §IV, §IX` | Two executed controls withheld | — | **Yes** |
| Argument Coherence | § Argument Coherence | PARTLY_MEETS | `table: Table V — kNN/Gaussian NB under "multiplies"` | Mechanism does not cover its own grouping | Framing, not data | **Yes** |
| Originality | § Originality | MEETS | `table: Table IV` | Null-floor measurement is novel | — | No |
| Writing Quality | § Writing Quality | MEETS | `text: §III-C` | Identification strategy stated crisply | — | No |
| Literature Integration | § Literature Integration | NOT_ASSESSED | — | R2's remit | — | No |
| Significance & Impact | § Significance | NOT_ASSESSED | — | Journal-Fit/R3 remit | — | No |

### Questions for Authors
1. Can you supply ρ(roundtrip_wav, mp4_aac64) per model, so Arm B's null is measured on the same axis as its effect?
2. Which contrast does the abstract's criterion shift refer to — codec (−0.063) or total (−0.064)?
3. Was any correction applied across the three McNemar tests, or are they reported as three independent descriptive checks?

### Minor Issues
- **Figures and Tables**: Table II reports McNemar p-values in §IV-D-1 prose but not in the table; readers comparing conditions must cross-reference.
- **Layout**: §IV-C gives the mask count as "ten temporal + six band-stop + one minimum-energy" (17), while §IV-D-4 and Fig. 8 refer to "16 spectro-temporal regions." Correct — the null mask is excluded from the ρ computation — but the paper should say so once.

---

## Seat 3 — Peer Reviewer 2 (Domain)

### Review Focus
Whether the audio-XAI and codec-robustness literatures are correctly represented, whether the paper's own domain discoveries are fully reported, and whether the attribution claims are stated at the right strength for this field.

### Recommendation
**Major Revision**

### Confidence
4 — audio XAI and SER feature traditions are core; LALM evaluation is adjacent.

### Summary Assessment
The literature framing is accurate and unusually well-chosen. The fragility → sanity-check → disagreement triad (Ghorbani, Adebayo, Krishna) is exactly the right scaffold for an attribution-stability paper, and the paper does not merely cite it — it runs the Adebayo model-randomisation check and reports the Krishna disagreement on its own models. The GeMAPS/openSMILE lineage is correctly invoked to defend named features, and the audioLIME/Nasr critique of time-frequency interpretable bases is engaged rather than dismissed. Per-emotion attributions (CPPS → sadness and neutrality, F0 median → fear, F2 movement → disgust) reproduce known SER associations and are correctly flagged as unsupervised recovery.

Two domain-side problems. First, a substantial block of the project's own EDA and attribution-mass findings is absent, including one result that directly supports a claim the paper does make. Second, a citation carries a wrong author name, and the manuscript's verification claim is broader than any artefact in the repository supports.

### Strengths

**S1: The Adebayo check is actually run, not cited.**
Six gradient methods against a randomly initialised network of identical architecture, ρ ∈ [−0.011, 0.206]. This is the check most attribution papers cite and skip.
**Evidence Anchor**: `figure: Fig. 6 — model-randomisation sanity check, all six methods near zero`

**S2: The within-family agreement result is interpreted correctly and against interest.**
Reporting that the six Captum methods agree at ρ ≥ 0.94 and then saying this is *not* evidence of correctness because they share a mathematical lineage is the right reading and the uncommon one.
**Evidence Anchor**: `text: §V-E-2 "High agreement within a method family is therefore not evidence of correctness — these methods share a mathematical lineage"`

**S3: Named features are defended on the right ground.**
**Evidence Anchor**: `text: §V-A "An attribution over prosody_cpps is a claim about breathiness; an attribution over a learned embedding dimension is not a claim about anything."`

**S4: The KernelSHAP failure is reported with its diagnostic tell.**
Rank-deficiency at 100 coalitions over 436 features, max mean |SHAP| = 2.1×10¹², plus the family-share-exactly-zero signature. Valuable to the field independent of this paper's thesis.
**Evidence Anchor**: `text: §VII-F "maximum mean |SHAP| of 2.1×10¹² against a median of 5.3×10⁻⁴"`

### Weaknesses

**W1: The attribution-mass-by-family result is omitted, and it supports a claim the paper makes without it.**
**Problem**: `FINDINGS.md` §E2/§F2 report SHAP family shares across six models (XGBoost MFCC 0.566 / spectral 0.207 / prosody 0.178 / chroma 0.033; ANN via IG 0.542/0.238/0.153/0.057) together with the observation that MFCCs hold 55% of the columns and take ~54% of the attribution — no per-feature advantage — and that this matches the mutual-information ranking by family (chroma 0.098 > prosody 0.087 > spectral 0.082 > MFCC 0.061). None of this appears in the manuscript.
**Evidence Anchor**: `absence: §V-E — expected the SHAP family-share table and the MI-by-family comparison; checked §V-A, §V-E-1, §V-E-2, §VII, figures`
**Why it matters**: §V-E-1 claims convergent evidence across five families and four algorithms, and supports it with a top-5 list. The family-share table is the quantitative version of that claim across the whole 436-dimensional space, and the MI comparison is an *independent* (non-attribution) corroboration of it. This is the paper's strongest available evidence that the convergence is not an artefact of the attribution operators, and it is on the floor.
**Suggestion**: Add the six-row family-share table to §V-E-1 with the MI ranking as a footnote; it costs a quarter column and materially strengthens the convergence claim.
**Severity**: Major
**Confidence**: 4.

**W2: The six-way construct-validity result is omitted, and it is the paper's cleanest instance of its own thesis.**
**Problem**: `exp/out/results.json` reports, on the six-way task, P(gold) − P(foil) = 0.170 (95% CI [0.138, 0.203], Wilcoxon W = 5364, p = 3.5×10⁻³⁹, Cohen's d_z = 0.55, n = 342). The model's log-probability mass on the correct emotion reliably exceeds a matched foil even while its argmax is `neutral` on 87.7% of trials. §IV-B reports only the argmax failure.
**Evidence Anchor**: `dataset: exp/out/results.json — construct_validity, mean_diff 0.170, p 3.5e-39, n 342`
**Why it matters**: The paper's core argument is that the decision surface hides what the model is actually doing. This result *is* that argument, measured directly, at an effect size and sample the rest of Arm A cannot match. It also independently justifies the move to a log-probability readout in §IV-B, which is currently justified only by the bias argument. As written, §IV-B leaves the reader believing the six-way task produced nothing.
**Suggestion**: Report it in §IV-B as the reason the readout changed, not as a discarded pilot. It reframes the task pivot from "the task failed" to "the argmax readout failed while the underlying discrimination did not."
**Severity**: Major
**Confidence**: 5 — recomputed.

**W3: A cited author's name is wrong.**
**Problem**: `refs.bib` entry `nasr2025beyond` gives "Nasr, Sofiane". The arXiv record for 2511.11691, verified in this repository's own `reference_verification.json`, gives **Seham Nasr**, Zhao Ren, David Johnson.
**Evidence Anchor**: `dataset: paper_q1/refs.bib entry nasr2025beyond — author "Nasr, Sofiane" vs verified "Seham Nasr"`
**Why it matters**: Misattribution of a living author in a Q1 journal, in a reference the paper leans on twice (§II-B and Limitation 10).
**Severity**: Minor
**Confidence**: 5.

**W4: The verification claim in the AI-use declaration exceeds the available evidence.**
**Problem**: §"Declaration of Generative AI Use" states "every citation was verified against a Crossref DOI record or the arXiv API." The only verification artefact in the repository, `research/phase2_investigation/reference_verification.json`, contains **19 records**, keyed to the predecessor protocol report (`cao2014`, `ghorbani2019`, …), not to this manuscript's 43-entry `refs.bib` (`cao2014cremad`, `ghorbani2019fragile`, …). Approximately 24 cited works — including `guo2017calibration`, `krishna2022disagreement`, `rudin2019stop`, `lundberg2020treeshap`, `eyben2016gemaps`, `jadoul2018parselmouth` — have no verification record.
**Evidence Anchor**: `absence: Declaration of Generative AI Use — expected a verification record covering all 39 cited works; checked reference_verification.json (19 entries, predecessor keys), paper_q1/README.md, refs.bib`
**Why it matters**: This is an integrity statement, and it is the one class of claim a reader cannot check for themselves. The README's assertion that refs.bib holds "39 references, each verified against Crossref or the arXiv API" carries the same problem. Either re-run verification over the current bib and commit the trail, or narrow the sentence to what was actually done.
**Severity**: Major
**Confidence**: 5 — key sets compared directly.

**W5: Related work omits the SER-specific codec literature beyond a single citation.**
**Problem**: §II-A rests the entire "coding degrades SER" literature on Reddy & Vijayarajan. The paper's own results are then positioned as resolving a tension with that one source (§VII-C).
**Evidence Anchor**: `text: §II-A "Reddy and Vijayarajan find that standard codecs reduce SER accuracy with a magnitude conditional on both codec family and feature representation"`
**Why it matters**: A tension with one paper is thinner than the resolution in §VII-C deserves. Work on telephony/GSM-coded SER, packet-loss robustness, and bandwidth-limited affect recognition would let the "magnitude is not a property of the codec" claim be tested against several prior magnitudes rather than one.
**Suggestion**: Add 2-4 codec/channel-robustness SER references and state whether their reported magnitudes are consistent with the model-family-dependence claim.
**Severity**: Minor
**Confidence**: 3 — I am confident the section is thin; I have not exhaustively surveyed what is available.

### Criterion-Bound Judgements

| Dimension | Criterion source | Judgement | Evidence anchor | Rationale | Uncertainty / scope | Decision bearing? |
|---|---|---|---|---|---|---|
| Literature Integration | § Literature Integration | PARTLY_MEETS | `text: §II-A` | XAI lineage excellent; codec-SER base is a single source | — | No |
| Evidence Sufficiency | § Evidence Sufficiency | PARTLY_MEETS | `absence: §V-E — expected family-share table; checked §V, figures` | Own findings under-reported | Repairable from committed data | **Yes** |
| Originality | § Originality | MEETS | `text: §II-F` | Gap statement is accurate and appropriately hedged | Absence-of-evidence claim | No |
| Argument Coherence | § Argument Coherence | MEETS | `text: §V-E-2` | Method-disagreement reasoning is sound | — | No |
| Methodological Rigor | § Methodological Rigor | NOT_ASSESSED | — | R1's remit | — | No |
| Writing Quality | § Writing Quality | MEETS | `text: §V-A` | — | — | No |
| Significance & Impact | § Significance | MEETS | `text: §VIII-D` | Archive/audit framing is a real domain contribution | — | No |

### Questions for Authors
1. Will you re-run reference verification over the current 43-entry bib and commit the trail, or narrow the declaration?
2. Is the family-share table omitted for space, or was it judged redundant with the top-5 lists?
3. Does the MI-by-family ranking (chroma highest per feature, MFCC lowest) survive as a claim you would make in print? It cuts against the MFCC-dominance impression the top-5 lists create.

### Minor Issues
- **Citation Format**: four `refs.bib` entries are unused (`kapoor2024reforms`, `lakens2017equivalence`, `lakens2018tutorial`, `schuirmann1987tost`) — residue from the predecessor protocol report. Harmless to compilation, but they signal an un-pruned bibliography.
- **References**: `sotirou2024musiclime` is correct in the manuscript; note the repository's verification record files the same work under `foteinopoulou2024musiclime`, which is wrong there, not here.

---

## Seat 4 — Peer Reviewer 3 (Perspective)

### Review Focus
Cross-disciplinary transfer, deployment and audit relevance, and whether the practical recommendations are actionable by the people the paper addresses.

### Recommendation
**Minor Revision**

### Confidence
4 — ML-systems monitoring and audit practice are core; codec internals adjacent.

### Summary Assessment
This is the seat where the paper is strongest and I have least to say against it. The framing — that a rationale surfaced to a clinician or auditor has its own robustness properties, independent of the label's — is a genuine cross-disciplinary import from the ML-monitoring literature into affective computing, and the paper converts it into three concrete operational statements: report the delivery format; report attribution stability against a null; do not use global attribution stability as a proxy for model health without input-range monitoring. The last of these is derived from the paper's own negative result (logistic regression, ρ = 0.987 with accuracy collapsing 0.567 → 0.233) and is, in my view, the single most transferable sentence in the manuscript.

The criterion-shift finding has an operational reading the paper states well: a uniform 0.064 posterior displacement changes which instances cross any fixed threshold, invisibly to accuracy monitoring, and it yields a falsifiable field prediction. My concerns are about reach, not correctness: two stakeholder groups are addressed implicitly and one recommendation is harder to act on than the paper implies.

### Strengths

**S1: The blind-spot result is a deployable warning, correctly labelled.**
**Evidence Anchor**: `text: §V-E-4 "A global attribution-stability metric can report 'nothing changed' while the classifier has effectively stopped working."`

**S2: The archive/audit asymmetry is a real and previously unstated problem.**
Auditing on archived compressed material while inference runs on uncompressed input yields agreeing decisions and disagreeing explanations. This is a compliance scenario, not a hypothetical.
**Evidence Anchor**: `text: §VIII-D "An explanation is not reproducible unless the delivery format is part of the record."`

**S3: The repair is ranked by cost, not by elegance.**
Model family (free) → clamp one descriptor → retrain per codec (requires advance knowledge, does not transfer). That ordering is how a practitioner actually decides.
**Evidence Anchor**: `text: §VI "use a threshold-based model family, which is free; or clamp the top-band spectral descriptors"`

**S4: The dual-use boundary is drawn explicitly.**
**Evidence Anchor**: `text: Ethics Declaration "the findings here concern robustness and explanation validity and are, if anything, cautionary about deploying such systems"`

### Weaknesses

**W1: The central recommendation is not operationalised for the audience most affected.**
**Problem**: §VIII-A asks systems to "report attribution stability across the input representations they will actually encounter … against a null." For Arm A's regime — an API-fed model — the null the paper uses is a 1-LSB dither, which requires the practitioner to construct a matched informationally-empty perturbation and re-run the full attribution suite over it, doubling inference cost.
**Evidence Anchor**: `text: §VIII-A "should report attribution stability across the input representations they will actually encounter, alongside accuracy, and should report it against a null"`
**Why it matters**: The recommendation is correct and the paper is the reason to believe it, but it is currently a principle rather than a procedure. A short "minimum viable protocol" — which null, how many items, what statistic, what cost — would make it adoptable rather than admirable.
**Suggestion**: Add a five-line recipe box or a short subsection: null = LSB dither at the bit depth of the delivery path; n ≥ 50 items; report ρ(a_ref, a_null) as the floor and ρ(a_ref, a_condition) as the effect, with the paired test.
**Severity**: Minor
**Confidence**: 4.

**W2: The regulatory stakeholder is addressed without being named.**
**Problem**: §VIII-D describes audit reproducibility and Ethics notes dual use, but the manuscript never connects to the compliance regimes that now mandate explanation for affect-inferring systems in employment, education, and biometric categorisation contexts.
**Evidence Anchor**: `absence: §VIII-D — expected a link from explanation-reproducibility to the regimes that mandate explanation; checked §VIII-D, Ethics Declaration, §X`
**Why it matters**: The paper's most policy-relevant claim — that a rationale is not reproducible unless the delivery format is recorded — has a natural home in record-keeping obligations. Making that link explicit widens the readership without weakening any claim.
**Severity**: Minor
**Confidence**: 3 — regulatory specifics are outside my core.

**W3: The "container is inert" null is likely to be over-generalised by readers.**
**Problem**: §VII-D concludes "format conversion between containers, absent re-encoding, appears behaviourally inert" and "the operational risk localises entirely to the codec." §VII-D itself notes Arm B's null is close to structural, and Arm A's holds for one deployment.
**Evidence Anchor**: `text: §VII-D "The practical consequence is favourable: format conversion between containers, absent re-encoding, appears behaviourally inert."`
**Why it matters**: This is the sentence a practitioner will quote. The predecessor design document was careful that a container null is evidence for a *disjunction* — the container is irrelevant, **or** the ingestion path decodes to PCM before anything model-specific happens — and that the contrast cannot separate them. The manuscript drops that distinction, having earlier made the opposite point about the endpoint validating container structure.
**Suggestion**: Restore one sentence of the disjunction: the practical licence holds either way, but the mechanistic claim does not follow.
**Severity**: Major
**Confidence**: 4.

### Criterion-Bound Judgements

| Dimension | Criterion source | Judgement | Evidence anchor | Rationale | Uncertainty / scope | Decision bearing? |
|---|---|---|---|---|---|---|
| Significance & Impact | § Significance | EXCEEDS | `text: §V-E-4` | Blind-spot result is directly deployable | Single-system evidence base | No |
| Argument Coherence | § Argument Coherence | MEETS | `text: §VIII-C` | Criterion-shift → threshold-crossing reasoning is tight | — | No |
| Originality | § Originality | MEETS | `text: §VIII-D` | Archive/audit asymmetry is new framing | — | No |
| Evidence Sufficiency | § Evidence Sufficiency | PARTLY_MEETS | `text: §VII-D` | Container null stated more strongly than the design licenses | Repairable in one sentence | **Yes** |
| Writing Quality | § Writing Quality | EXCEEDS | `text: §VIII` | — | — | No |
| Methodological Rigor | § Methodological Rigor | NOT_ASSESSED | — | R1's remit | — | No |
| Literature Integration | § Literature Integration | NOT_ASSESSED | — | R2's remit | — | No |

### Questions for Authors
1. Would you add a minimum-viable null-calibration protocol so §VIII-A becomes adoptable?
2. Can §VII-D restore the container-null disjunction (inert vs. decoded-before-inference) that the design cannot separate?

### Minor Issues
- **Discussion**: §VIII-C's falsifiable prediction about confidence distributions across ingestion paths is the paper's most testable operational claim and is buried mid-paragraph; it deserves its own sentence break.

---

## Seat 5 — Devil's Advocate

### Calibration Status
`NOT_CALIBRATED`

### Criterion-Bound Judgements

| Dimension / criterion | Criterion source | Judgement | Evidence anchors | Rationale | Uncertainty / scope limit | Decision bearing? |
|---|---|---|---|---|---|---|
| 1. Core Thesis | DA Dim. 1 | PARTLY_MEETS | `text: Abstract "no detectable loss of discriminability"` | Strong-form dissociation rests on a null result the paper distrusts in §IX-4 | Underpowered, not disproven | **Yes** |
| 2. Cherry-Picking | DA Dim. 2 | PARTLY_MEETS | `absence: §IV — expected WER and construct-validity results; checked §IV, §IX` | Omissions are asymmetric: both withheld results complicate a clean narrative | Cannot establish intent; only the pattern | **Yes** |
| 3. Confirmation Bias | DA Dim. 3 | MEETS | `text: §VII-F` | Self-withdrawal and KernelSHAP exclusion cut against interest | — | No |
| 4. Logic Chain | DA Dim. 4 | PARTLY_MEETS | `figure: Fig. 1 — null B declared, measured only in the accuracy domain` | The paper's prescriptive rule is not applied to its own Arm B | — | **Yes** |
| 5. Overgeneralization | DA Dim. 5 | PARTLY_MEETS | `table: Table VIII — 4-point ordering across confounded axes` | Transparency ordering promoted to a contribution on n = 1 per class outside Arm B | Authors disclose the confounds | **Yes** |
| 6. Alternative Paths | DA Dim. 6 | PARTLY_MEETS | `text: §IV-D-5` | Estimator-noise reading of the non-monotone ladder is unconsidered | — | No |
| 7. Stakeholder Blind Spots | DA Dim. 7 | MEETS | `text: Ethics Declaration` | Dual-use named; R3 covers reach | — | No |
| 8. "So What?" | DA Dim. 8 | EXCEEDS | `text: §VIII-A` | Null-calibration prescription is durable regardless of the empirical result | — | No |
| 9. Field-Norm Calibration | DA Dim. 9 | MEETS | `dataset: xai_ser/outputs, exp/out — all severities rest on the paper's own artefacts` | No finding below rests on an ungrounded field norm | — | No |

### Strongest Counter-Argument

Here is the strongest case against this paper, made from its own data.

The paper's title, abstract, and Contribution 3 claim a dissociation: coding relocates evidence more than it relocates accuracy. But Arm B — fifteen models, 7,441 clips, exact attribution, a real sample — **does not reproduce the strong form**, and the paper says so in §VII-A. The strong form exists in exactly one place: a single commercial endpoint, n = 50 items, one binary task chosen after the six-way task failed, one prompt chosen on the third attempt, one readout chosen after inspecting reference-condition data, and an accuracy decline of ten points that is called "no detectable loss" because a McNemar test on seven discordant pairs returned p = .125. The same sign appears at 32 and 128 kbps. The honest description of Arm A's accuracy result is *underpowered*, and the paper says exactly this in Limitation 4 — then does not say it in the abstract.

Now compound that. The paper's own governing principle is that an effect is meaningless without an informationally-null floor measured on the same axis. Arm A obeys this. Arm B does not: its declared null (the container re-wrap, Fig. 1) is evaluated only as a balanced-accuracy difference, while every attribution ρ is measured against `ref` and the attribution floor is *asserted* to be unity because SHAP is deterministic. That is the same category of argument the paper withdraws a prior finding over in §VII-F — "a determinism check is not a null control." The paper convicts itself in one arm on a principle it suspends in the other.

Add the pattern of omissions. A pre-declared intelligibility control was run and is unreported. A construct-validity result showing the six-way task *did* carry gold-label information (p = 3.5×10⁻³⁹) is unreported. A declared masking-artefact floor is measured but its value withheld — and it is large enough to swallow two of the sixteen regions the paper counts as evidence the operator works.

None of this makes the paper wrong. The criterion shift is solid, the attribution reordering clears its floor by eight orders of magnitude in p, the container null replicates twice, and Arm B's covariate-shift diagnosis is excellent and fully supported. But the paper as titled is carried by its weakest arm, and its methodological sermon is preached more consistently than it is practised. Retitle around what Arm B proves and what Arm A's null-calibration *method* contributes, report the withheld controls, and the paper gets stronger by claiming less.

### Issue List

#### CRITICAL

| # | Dimension | Issue Description | Evidence Anchor | Confidence | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|---|---|---|---|---|---|
| C1 | 4. Logic Chain | The paper's governing methodological rule — an attribution effect requires a null measured on the same axis — is applied to Arm A and suspended for Arm B, where the declared container-re-wrap null is only ever evaluated as a balanced-accuracy difference and the attribution floor is asserted as unity on a determinism argument the paper elsewhere rejects (§VII-F). ρ(roundtrip_wav, mp4_aac64) is never computed despite both rankings being committed. | `figure: Fig. 1 — "null B: container re-wrap, codec output held fixed, decode path varied", against Table VI where every ρ is measured versus ref` | 5 — core: control design | — | — |
| C2 | 1. Core Thesis | The abstract and Contribution 3 assert the strong dissociation ("no detectable loss of discriminability", "the evidence moved and the decision did not") while §IX-4 concedes the accuracy decline is directionally consistent across all three bitrates and merely non-significant at n = 50 (0.68 → 0.58, McNemar p = .125). The headline claim and the limitation are not the same claim. | `text: §IX-4 "The accuracy decline is directionally consistent across the bitrate ladder but non-significant at every bitrate."` | 5 — core: claim calibration | — | — |
| C3 | 2. Cherry-Picking | Two executed analyses that complicate the narrative are absent from the manuscript: the pre-declared WER intelligibility control (all seven conditions, `exp/out/wer.parquet`) and the six-way construct-validity result (P(gold) − P(foil) = 0.170, p = 3.5×10⁻³⁹, n = 342, `exp/out/results.json`). §IV-B presents the six-way task as a bare failure. | `absence: §IV — expected the pre-declared WER control and the six-way construct-validity result; checked §IV-A, §IV-B, §IV-D, §IX, Data Availability` | 5 — recomputed from artefacts | — | — |

#### MAJOR

| # | Dimension | Issue Description | Evidence Anchor | Confidence | Field-Norm Boundary | Evidence-Crossing Rationale |
|---|---|---|---|---|---|---|
| M1 | 5. Overgeneralization | Table VIII's transparency ordering is promoted to a contribution ("striking and, to our knowledge, not previously reported") on four points with one model per class outside Arm B, across three axes the paper itself lists as confounded. The abstract states it as a finding. | `table: Table VIII — four rows, three of them single-model, with operator/representation/family covarying` | 4 — inference scope | — | — |
| M2 | 4. Logic Chain | §IV-C declares a minimum-energy null mask that "bounds the attribution attributable to the masking operation itself"; the bound is never reported. It is 0.029, above two of the sixteen regions §IV-D-5 counts as evidence the operator is measuring structure. | `dataset: exp/out/arousal_xai.parquet — mask_id null_lowenergy mean attribution 0.029 on ref, versus t7 0.018 and f250_500 0.015` | 5 — recomputed | — | — |
| M3 | 4. Logic Chain | §IV-D-2 attaches the total-effect magnitude (0.208 → 0.144, −0.064) to the codec contrast's p-value (6.8×10⁻⁵). The total contrast's p is 7.5×10⁻⁴; the codec contrast's mean_diff is −0.0627. | `text: §IV-D-2 "a paired shift of −0.064 (p = 6.8×10⁻⁵, Wilcoxon signed-rank)"` | 5 — both values recomputed | — | — |
| M4 | 1. Core Thesis | The stated mechanism for Arm B ("thresholds vs. multiplies") does not describe two of the eight models it groups as "multiplies": kNN computes distances, Gaussian NB evaluates per-feature likelihoods. The dichotomy is restated in the abstract. | `table: Table V — kNN and Gaussian NB placed in the "multiplies" block` | 4 — classifier mechanics | — | — |
| M5 | 6. Alternative Paths | §IV-D-5 reads the non-monotone bitrate ladder as independently reproducing Wu et al.'s fidelity/behaviour dissociation, without considering that at n = 50 with overlapping CIs the ladder is simply unresolved. The paper's own Limitation 7 (0.46 dB from 64 to 128 kbps) supplies a stronger physical version of the same argument. | `text: §IV-D-5 "The absence of dose-response is not a failed manipulation; it independently reproduces the central claim of Wu et al."` | 4 — inference under low power | — | — |
| M6 | 3. Confirmation Bias | The Declaration of Generative AI Use states "every citation was verified against a Crossref DOI record or the arXiv API." The repository's only verification artefact holds 19 records under the predecessor report's keys; roughly 24 of the 39 cited works have no record. | `absence: Declaration of Generative AI Use — expected a verification trail covering the current 39 citations; checked reference_verification.json, paper_q1/README.md, refs.bib` | 5 — key sets compared | — | — |

#### MINOR

| # | Dimension | Issue Description | Evidence Anchor | Confidence |
|---|---|---|---|---|
| m1 | 5. Overgeneralization | §VII-D's "the operational risk localises entirely to the codec" drops the disjunction the design cannot separate (container inert vs. decoded to PCM before anything model-specific runs). | `text: §VII-D "The operational risk localises entirely to the codec"` | 4 — inference scope |
| m2 | 4. Logic Chain | Fig. 2's caption (37.7 pooled SD) and §III-D/abstract/§XI (38.6 training SD) report the same shift on two scales without reconciling them. | `figure: Fig. 2 caption — "moves by 37.7 pooled SD" against §III-D "the shift is 38.6σ"` | 5 — both verified |
| m3 | 8. "So What?" | "Floor" denotes the ρ floor (unity) in §VII-A and Limitation 2, and the displacement floor (zero) in Fig. 8's caption. | `figure: Fig. 8 caption — "have a floor of zero" against Table VIII column "Null floor … 1.000"` | 5 — textual |

### Ignored Alternative Explanations/Paths

1. **The Arm A attribution reordering may be partly a posterior-compression artefact rather than evidence relocation.** Mean P(high) falls from 0.208 to 0.144 under coding — a 31% contraction of the quantity attributions are differences *of*. If per-region decrements scale with the base posterior, the rank ordering becomes noisier under compression for arithmetic reasons, independent of any change in reliance. The 1-LSB dither does not control for this, because the dither does not move the posterior mean. A per-utterance normalisation of `a_j` by P(high | x) would test it, and the data to do so is in `arousal_xai.parquet`. This is the alternative I would press hardest at review.

2. **Arm B's "strong form does not occur" may be a consequence of the estimator, not the models.** Global mean-|SHAP| rankings over 436 features are dominated by a stable head; the paper's own §V-E-4 shows they can miss a total accuracy collapse. A per-instance stability analysis — which Limitation 9 concedes is the stronger instrument — might well surface the strong dissociation in Arm B too. The paper's cross-arm explanation (§VII-A: representation, not sophistication) is plausible but currently unfalsified against this simpler one.

3. **The six-way task's failure may be a readout failure, not a capability failure.** The construct-validity result (C3) is the evidence, and it is the paper's own. §IV-B's lexical-dominance reading is compatible with it but not required by it.

### Missing Stakeholder Perspectives
- The codec/encoder engineering community, whose bit-allocation decisions are the paper's causal mechanism and who are addressed only as a physical process.
- Corpus and benchmark maintainers, who are the parties that would actually implement "report the delivery format."

### Unexamined Premise

The paper assumes throughout that **the top-25 attributed features constitute "the evidence"** and that turnover in that set constitutes evidence relocation. Both arms' headline stability statistics — top-k overlap, enter/leave lists, rank ρ — inherit this. But nothing establishes that rank 24 and rank 26 differ in decision relevance, and Arm B's own §V-E-4 result demonstrates that a model can retain 22 of 25 top features while its decision function has stopped functioning. The premise that ranking-turnover tracks evidential change is the load-bearing assumption of the entire paper, and it is never argued — only operationalised.

### Observations (Non-Defects)
- The self-withdrawal in §VII-F is the most credible passage in the manuscript and would be worth foregrounding in the introduction rather than deferring to §VII.
- The convergence of five model families and four attribution algorithms on MFCC-0 derivative variability is a genuinely interesting SER result that the paper treats as a warm-up.
- Every value I sampled from Tables I-VIII reproduced from the committed artefacts, including two places where the authors corrected their own findings document. The traceability standard here is above the field norm.

---

# Phase 2 — Editorial Decision

## Decision: **MAJOR REVISION**

### Panel Convergence

| Seat | Recommendation | Confidence |
|---|---|---|
| Journal-Fit | Major Revision | 4 |
| R1 — Methodology | Major Revision | 5 |
| R2 — Domain | Major Revision | 4 |
| R3 — Perspective | Minor Revision | 4 |
| DA | 3 CRITICAL, 6 MAJOR | — |

Four of five seats converge on Major Revision. R3's Minor reflects genuine strength in the deployment-relevance dimension, not disagreement about the defects — R3 did not assess methodology or literature and says so.

**Correlated-error disclosure**: all five seats ran on one model family in one session. Convergence below is corroboration under role separation, **not** evidence of independent error processes.

### DA CRITICAL Adjudication (Iron Rule #4 — each adjudicated visibly)

| # | Adjudication | Rationale |
|---|---|---|
| **C1** — asymmetric null calibration | **VALIDATED** | Independently found by R1 (W4) from a different starting point. Confirmed against `outputs/xai/*/xai_summary.json`: every `attribution_stability` entry is keyed to `ref`; no roundtrip-vs-mp4 comparison exists. The paper's own §VII-F principle applies. **Blocks Accept.** Repairable from committed data — likely *supports* the paper's conclusion. |
| **C2** — abstract vs. Limitation 4 | **VALIDATED** | Independently found by Journal-Fit (W1). The abstract, Contribution 3, and §XI state the strong dissociation; §IX-4 does not. **Blocks Accept.** Repairable by rewording, with no change to any analysis. |
| **C3** — withheld executed analyses | **VALIDATED** | Independently found by R1 (W1), R2 (W2), and Journal-Fit (W2), from three different angles. Both artefacts confirmed present and unreported. **Blocks Accept.** Repairable by reporting existing data. |

All three CRITICALs are validated and all three are repairable without new experiments. This is why the decision is Major Revision rather than Reject.

### Consensus Findings (≥ 2 seats)

| Finding | Seats | Severity |
|---|---|---|
| Executed analyses (WER control, construct validity) unreported | Journal-Fit, R1, R2, DA | Critical |
| Abstract overstates the dissociation relative to Limitation 4 | Journal-Fit, DA | Critical |
| Arm B has no attribution-domain null | R1, DA | Critical |
| Declared null-mask value withheld and non-trivial | R1, DA | Major |
| Verification claim exceeds available evidence | R2, DA | Major |
| "Floor" / SD-scale terminology collides with itself | Journal-Fit, DA | Minor |

### Divergence

R1 rates the "thresholds vs. multiplies" grouping Major (mechanism does not cover its own rows); DA rates it Major on the same ground; no seat contests the empirical split, only its stated mechanism. **Arbitration**: the empirical result is secure and unchallenged; only the causal label needs repair. Retained at Major because the label appears in the abstract.

DA's alternative explanation #1 (posterior-compression artefact in Arm A attribution) was raised by no other seat. **Arbitration**: it is testable from committed data and bears directly on the paper's second headline result. Escalated to the roadmap as a required check rather than a required reanalysis — if the normalised statistic reproduces, one sentence closes it.

### What the panel does **not** dispute

The container null (both arms), the covariate-shift diagnosis and single-feature repair, the KernelSHAP exclusion, the model-randomisation sanity check, the method-disagreement result, the logistic-regression blind spot, and the numeric traceability of every table. Arm B is, in the panel's collective assessment, publication-ready as it stands. The revision burden falls almost entirely on Arm A's framing and on reporting completeness.

---

## Revision Roadmap (immutable core — no work ordering implied)

**Priority 1 — Required before acceptance**

| ID | Item | Source | Minimum remedy |
|---|---|---|---|
| P1-1 | Compute and report Arm B's attribution-domain null: ρ(roundtrip_wav, mp4_aac64) per model, alongside the existing ρ-versus-`ref` column | DA-C1, R1-W4 | Add one column to Table VI; both rankings are already committed |
| P1-2 | Reconcile the abstract and Contribution 3 with §IX-4 on the strong dissociation | DA-C2, EIC-W1 | Reword abstract to "sensitivity invariant, accuracy non-significantly reduced (0.68 → 0.58, McNemar p = .125), attribution reordered beyond floor"; reserve "dissociation" for the sensitivity/criterion/attribution triple |
| P1-3 | Report the pre-declared WER intelligibility control across all seven conditions, or state in §IX that the arm was dropped and why | DA-C3, R1-W1 | One table row or one short subsection from `exp/out/wer.parquet` |
| P1-4 | Report the six-way construct-validity result in §IV-B as the reason the readout changed | DA-C3, R2-W2 | Two sentences: P(gold) − P(foil) = 0.170, 95% CI [0.138, 0.203], d_z = 0.55, p = 3.5×10⁻³⁹, n = 342 |
| P1-5 | Report the minimum-energy null-mask attribution value and mark it on Fig. 4 | DA-M2, R1-W3 | Value = 0.029; note that two of sixteen regions fall below it |
| P1-6 | Correct the criterion-shift statistic/p-value pairing consistently across abstract, §IV-D-2, §VIII-C, §XI | DA-M3, R1-W2 | Choose codec (−0.063, p = 6.8×10⁻⁵) or total (−0.064, p = 7.5×10⁻⁴) |
| P1-7 | Narrow the Declaration of Generative AI Use and `README.md` to the verification actually performed, or re-run it over the current 43-entry bib and commit the trail | DA-M6, R2-W4 | Either action closes it |
| P1-8 | Fix `refs.bib` `nasr2025beyond`: first author is Seham Nasr, not Sofiane Nasr | R2-W3 | One-line edit |

**Priority 2 — Strongly recommended**

| ID | Item | Source |
|---|---|---|
| P2-1 | Demote Table VIII's transparency ordering from contribution to observation; retain the within-Arm-B contrast as the supported claim | DA-M1 |
| P2-2 | Rename Table V's blocks to the property that separates all fifteen rows ("axis-aligned partitioning" vs. "magnitude-sensitive") | DA-M4, R1-W5 |
| P2-3 | Test and report DA alternative #1: normalise Arm A's `a_j` by per-utterance P(high\|x) and confirm the codec effect survives | DA-Alt-1 |
| P2-4 | Add the SHAP family-share table and the MI-by-family ranking to §V-E-1 | R2-W1 |
| P2-5 | Restore the container-null disjunction in §VII-D (inert vs. decoded-before-inference) | R3-W3, DA-m1 |
| P2-6 | State both readings of the non-monotone bitrate ladder; lead with the psychoacoustic-saturation argument from Limitation 7 | DA-M5, R1-W6 |
| P2-7 | Argue, or explicitly assume, the premise that top-k ranking turnover tracks evidential change | DA-Unexamined Premise |

**Priority 3 — Editorial**

- Reconcile 37.7 pooled SD / 38.6 training SD in one place (DA-m2)
- Disambiguate "floor" between §VII-A/Limitation 2 and Fig. 8's caption (DA-m3, EIC minor)
- State once that the null mask is excluded from the 16-region ρ computation (R1 minor)
- Add a minimum-viable null-calibration protocol to §VIII-A (R3-W1)
- Prune the four unused `refs.bib` entries (R2 minor)
- Consider retitling to foreground displacement, or note that the title's phrasing is licensed by Arm B (EIC-W3)
- Add 2-4 codec/channel-robustness SER references to §II-A (R2-W5)
- Name the regulatory record-keeping context in §VIII-D (R3-W2)

---

## Editorial Note to Author

The panel's collective judgement is that this is a strong paper whose defects are almost entirely defects of *reporting*, not of *research*. No Priority 1 item requires a new experiment; six of the eight are satisfied from data already committed to this repository, and one is a one-line bib fix.

The recurring pattern across three of the five seats is the same: the manuscript claims slightly more than its evidence in the abstract, and reports slightly less than its evidence in the body. Those two habits point in opposite directions and both are cheap to correct. Correcting them will make the paper harder to attack without making any of its supported claims smaller.

The panel notes explicitly that Arm B, the KernelSHAP disclosure, the §VII-F self-withdrawal, and the numeric traceability of the manuscript are above the norm for this venue.
