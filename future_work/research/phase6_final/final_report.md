# Separating Container from Codec: A Black-Box XAI Protocol and Feasibility Analysis for Audio Format Robustness in a Multimodal Large Language Model

**Research design and feasibility report** · *Final (Phase 6)*

Prepared with the ARS `deep-research` pipeline (v2.12.1), full mode
Date: 17 August 2026
Corresponding project: `WavVsMp4` — Audio LLM Format Robustness pilot

---

## Abstract

**This is a protocol and feasibility report; it specifies a study and reports the preparatory work
that shaped it, not results on its primary questions.** Every deployed audio pipeline transcodes,
yet evaluations of large audio-language models (LALMs) do not report the container in which audio
was submitted. The protocol determines whether lossy container delivery (MP4/AAC at 64 kbps)
changes not only a multimodal LLM's accuracy on speech emotion recognition but *which regions of
the signal its output depends on*. The design's distinguishing feature is a contrast that holds the decoded
waveform constant while varying the container — `mp4_aac64` against a `roundtrip_wav` control —
permitting codec degradation and container delivery to be separated without model internals.
Because the target model is closed-weights and API-only, explanation is measured black-box, by
temporal and spectral occlusion, with a mandatory within-format stability floor established before
any across-format divergence is interpreted. A verified 23-source review situates the work between
a mature codec-robustness literature, a young LALM-evaluation literature that omits format
entirely, and an XAI literature showing that explanations can shift while predictions hold. Three
adversarial checkpoints and an audit of existing pilot data materially changed the design: a
pre-declared equivalence bound was shown to be unreachable at n = 50; an audit found 92% of pilot
calls failed on API quota; and the primary attribution measure was shown to become mathematically
undefined under a degenerate response distribution. The protocol is presented with sequential
gates, each of whose failure is a pre-authorised reportable outcome.

**Keywords:** audio language models, format robustness, audio codecs, explainable AI, occlusion
attribution, explanation stability, speech emotion recognition, black-box interpretability

---

## 1. Introduction

### 1.1 Problem

Audio reaches a model through a pipeline. It is captured, encoded for transport, stored in a
container, often transcoded again, and finally submitted to an API. At every step the bytes change
and, at several, the waveform changes with them. Benchmarks do not work this way. They read a WAV
file from disk and submit it.

The gap between those two situations is not obviously important, and that is precisely why it has
gone unexamined. The assumption underwriting current practice is that a model's audio front end
decodes any container to PCM early, so that only the *signal* matters and the container is a
delivery detail. If true, this is a convenient and defensible assumption. It has not been tested.

There is a second gap, less obvious and more consequential. Suppose transcoding leaves accuracy
untouched. Standard evaluation would record no effect and move on. But an unchanged label does not
entail an unchanged basis for that label. In computer vision this dissociation is established:
perceptually indistinguishable inputs that receive the same prediction — sometimes with higher
confidence — can be assigned substantially different interpretations (Ghorbani et al., 2019). If
that phenomenon transfers to audio under ordinary transcoding rather than adversarial construction,
then benchmarks that report only accuracy are systematically blind to a form of format sensitivity
that exists in every production system.

### 1.2 Purpose

This report specifies a protocol to test that transfer, and reports the preparatory work —
literature synthesis, adversarial design review, and an audit of existing pilot data — that
determined the protocol's final shape. It does not report experimental results on the primary
questions; those await gate clearance described in §3.7.

**Why the protocol is the deliverable.** The decisions that determine what this study can conclude
— the equivalence bound, the interpretation grid, the degeneracy threshold, the conditional-analysis
rule — are all decisions that lose their force if published after the data. A study that predicts
its own null, as this one does from published evidence (§2.6), has no credible way to report that
null unless the rules for reading it were fixed beforehand. Three of those rules changed during
preparation, and none of the changes came from data: the equivalence bound was found unreachable by
statistical review (§4.2), the attribution measure was found to become undefined under a degenerate
response distribution by analytic review (§4.3), and the pilot audit found a quota failure rather
than a methodological one (§4.1). Publishing the design now is what makes the eventual result —
including a null, including a stopped study — reportable at all.

### 1.3 Research question

> **Does lossy container delivery (MP4/AAC at 64 kbps) alter which acoustic regions a multimodal
> LLM relies on for speech emotion recognition, beyond any change in its accuracy?**

Three sub-questions decompose it:

- **SQ1 (Performance).** How much does 6-way emotion accuracy — and transcription WER — change
  across `ref` (lossless WAV), `mp3_64`, `mp4_aac64`, and `roundtrip_wav`?
- **SQ2 (Mechanism).** Of any degradation under `mp4_aac64`, how much is signal-level codec loss
  and how much is container delivery itself?
- **SQ3 (Explanation).** On items whose predicted label is *unchanged* between `ref` and
  `mp4_aac64`, do occlusion-based attribution maps nonetheless diverge — by more than the map
  instability observed between repeated runs on `ref` alone?

SQ3 is the headline. SQ1 and SQ2 are supporting.

### 1.4 Contribution

The contribution is a composition rather than an invention, and the report is explicit about which
parts are inherited:

1. **Inherited.** The dissociation idea (Ghorbani et al., 2019), perturbation-based attribution
   (Ribeiro et al., 2016; Zeiler & Fergus, 2014), and the discipline of validating attribution
   before interpreting it (Adebayo et al., 2018).
2. **Transferred.** From adversarial perturbation in vision to routine transcoding in audio; from
   white-box gradients to black-box occlusion against a commercial API with no gradients,
   activations, or log-probabilities.
3. **New, as far as a targeted search could establish.** The container/codec decomposition — a
   contrast in which the decoded waveform is held constant and only the container varies — and the
   first measurement of run-to-run stability for black-box attribution on an audio LLM.

---

## 2. Literature Review and Theoretical Framework

Twenty-three sources were screened and verified (§3.1). Five themes organise them.

### 2.1 Lossy coding removes paralinguistic information; fidelity metrics do not track it

Compression reduces speech-emotion-recognition accuracy for task-specific systems, and the
magnitude depends on both codec and acoustic feature family (Reddy & Vijayarajan, 2020). Neural
codec benchmarking converges from the other direction: Codec-SUPERB evaluates codecs through
downstream tasks precisely because signal-level metrics fail to capture losses in semantic,
linguistic, and paralinguistic content (Wu et al., 2024).

The design consequence is direct. Reporting SNR or log-spectral distance between `ref` and
`mp4_aac64` establishes that the signal changed; it says nothing about whether the model lost
anything it was using. Fidelity and behaviour are measured separately and neither substitutes for
the other.

### 2.2 Audio LLMs may not be listening in the way the codec literature assumes

Chen et al. (2025) built a benchmark to separate lexical from acoustic contributions to emotion
judgements and evaluated six state-of-the-art LALMs. They report consistent **lexical dominance**:
models predict *neutral* when lexical cues are neutral or absent, gain little when cues align,
fail under cue conflict, and approach chance in paralinguistic contexts. Their conclusion is that
these systems "largely transcribe rather than listen".

This does not stand unopposed. Zhang et al. (2026) benchmark speech LLMs for SER across 35 corpora
in 15 languages — not the evaluation profile of a model class performing at chance — and identify
a different first-order problem: moving from closed-set classification to open text generation
introduces zero-shot stochasticity that makes results highly prompt-sensitive. Lee et al. (2025)
add that cross-model comparisons are routinely confounded by differing prompting methods and
inference parameters.

The resulting position is genuinely open. There is credible evidence of substantial lexical
reliance and credible evidence of real acoustic capability. Which dominates for the model and
corpus at hand is an empirical question, and the protocol's first gate is designed to answer it
before anything else runs.

### 2.3 Explanations can move while predictions stand still

Ghorbani et al. (2019) demonstrated for image classifiers that perceptually indistinguishable
inputs assigned the same label can receive substantially different interpretations. Adebayo et al.
(2018) supply the necessary counterweight: some widely used saliency methods are independent of
both the model and the data generating process, producing convincing maps that explain nothing,
and visual inspection cannot tell them apart from real ones.

Together these fix a non-negotiable requirement. An attribution map must be shown reproducible
under no manipulation before any change in it under manipulation can mean anything. In this design
that requirement is `S_within`, the within-format stability floor.

### 2.4 Black-box attribution for audio is feasible, and its validity is contested

LIME established that a model can be explained through input perturbation and output observation
alone (Ribeiro et al., 2016); occlusion sensitivity established the specific move of masking an
input region and reading the output change (Zeiler & Fergus, 2014). Together they make attribution
possible in the only regime available here.

But audio-specific work pushes back on exactly the representation this design must use. audioLIME
argues that interpretability for audio should mean *listenability*, and that treating spectrogram
patches as image superpixels produces components corresponding to nothing audible; its remedy is
to perturb source-separated stems (Haunschmid et al., 2020). Nasr et al. (2025) extend the critique
to SER: vision-derived saliency highlights time–frequency regions without establishing that they
correspond to meaningful acoustic markers of emotion.

Source separation does not apply to single-speaker CREMA-D clips — there are no stems to toggle.
This design therefore uses time windows and frequency bands, the representation under critique.
The response is mitigation plus disclosure, not rebuttal (§3.5, §6.2).

### 2.5 Delivery format is invisible in current practice

No source in the corpus reports the container in which audio was submitted. Model reports and
benchmarks specify corpora, prompts, and sometimes sampling rate (Chu et al., 2024; Li et al.,
2025; Lee et al., 2025). Robustness work is framed around adversarial attacks and naturally
occurring degradation such as noise and reverberation (Li et al., 2025), not around the transcoding
every production pipeline performs. A systematic absence across 23 sources is itself the finding.

### 2.6 Derived prediction

Themes 2.1 and 2.2 appear to conflict — compression should hurt emotion recognition, yet a
lexically reliant model should barely notice. The conflict dissolves on inspection: the codec
findings were established on systems that classify from prosodic and spectral features, which is
where coding loss lands. A model that reaches its judgement substantially through transcribed
lexical content has a different exposure profile — sensitive to degradation that harms
intelligibility, comparatively insensitive to degradation that harms prosody alone. At 64 kbps,
intelligibility is essentially intact.

> **Prediction, fixed before data collection.** For a lexically reliant model on a fixed-lexicon
> corpus, SQ1 should return a **small or null accuracy effect** and the transcription arm should
> sit at a **WER floor in every condition** — not because format is irrelevant, but because what
> the format destroys is not what the model was using. Conditional on the WER control confirming
> that intelligibility is in fact preserved.

The prediction is falsifiable and cuts against the study's own interest, which is the point. If it
holds, SQ1's null carries a mechanism rather than an absence. If it fails — if accuracy moves —
then an accuracy-first framing would have been the stronger organisation of this work, and the
report will say so rather than retro-fitting the explanation angle.

**Scenario ordering.** The following is a judgement call recorded in advance, not a calculated
forecast; no probability here is derived from a model or a power analysis. It exists so that the
eventual result can be compared against what was expected, rather than against what turns out to be
convenient.

| Rank | Scenario | Rough expectation |
|---|---|---|
| 1 | Gate G0 fails or is marginal; study pivots task or model before the main grid | Most likely single outcome |
| 2 | Grid runs; SQ1 null with WER at floor; SQ2 inconclusive; SQ3 explanations stable | Most likely outcome *conditional on* passing the gates |
| 3 | Grid runs; SQ1 null; **SQ3 explanations diverge** — the dissociation cell | A live possibility, not the favourite |
| 4 | SQ1 shows a real accuracy effect | Least expected; would mean the §2.6 prediction is wrong and accuracy-first was the better framing |

---

## 3. Method

### 3.1 Literature identification and verification

Twelve structured searches spanning codec robustness, LALM evaluation, audio XAI, explanation
stability, and reporting standards surfaced approximately 110 records. Tutorials, vendor pages,
patents, and content-farm summaries were excluded; 35 records were screened on title and abstract;
23 entered the corpus.

Every source was verified programmatically before citation. The Semantic Scholar Graph API
returned HTTP 429 on every unauthenticated request in this session (`[S2-API-UNAVAILABLE]`), so
verification proceeded on the two remaining indexes: Crossref DOI lookup with a Levenshtein 0.70
title cross-check and title-search fallback, and the arXiv API by identifier. **23 of 23 sources
verified.** Two Crossref records (Ribeiro et al., 2016; Lakens, 2017) initially returned
`DOI_MISMATCH` because Crossref stores them under truncated titles; direct DOI re-fetch confirmed
authors, venue, and pagination in both cases. No source entered on an "unable to confirm" basis.
The full audit trail is machine-readable in `reference_verification.json`.

Reporting follows REFORMS (Kapoor et al., 2024), adopted because no EQUATOR guideline covers
computational benchmark experiments.

### 3.2 Materials

Fifty CREMA-D clips (Cao et al., 2014), stratified by emotion across the six-way lexicon and
diversified across speakers, frozen before any model call. CREMA-D comprises 7,442 clips from 91
actors reciting 12 predefined sentences across six emotion categories, crowd-rated by 2,443
raters. The fixed lexicon holds linguistic content constant across items, which is what allows
acoustic effects to be isolated — and, per §2.2, also what exposes the design to a lexical-dominance
failure mode.

One figure from the corpus paper is load-bearing: **human recognition of intended emotion from
audio alone is 40.9%** (Cao et al., 2014). The comparison to a model is indicative rather than
like-for-like — those raters judged the full corpus without a forced-choice prompt — but it
establishes that this is a hard task for humans, and it recalibrates the floor threshold in §3.7
from a modest bar to approximately human parity.

### 3.3 Conditions

All conditions descend from a single canonical reference: 16 kHz mono PCM16, EBU R128 loudness-
normalised. No condition is generated independently from the CREMA-D source, which prevents
divergent resampling or loudness paths from masquerading as format effects.

| Condition | Construction | Container |
|---|---|---|
| `ref` | canonical reference | WAV |
| `mp3_64` | MP3, libmp3lame, 64 kbps | MP3 |
| `mp4_aac64` | AAC 64 kbps, audio-only MP4 | MP4 |
| `roundtrip_wav` | `mp4_aac64` decoded back to canonical PCM | WAV |
| `roundtrip_wav_mp3` | `mp3_64` decoded back to canonical PCM | WAV |

The two round-trip controls are the design's identifying mechanism. They yield three planned
contrasts:

| Contrast | What differs | Identifies |
|---|---|---|
| `roundtrip_wav` − `ref` | waveform only | **signal effect** (codec loss) |
| `mp4_aac64` − `roundtrip_wav` | container only; decoded waveform held constant | **container effect** |
| `mp4_aac64` − `ref` | both | **total format effect** |

### 3.4 Model and submission protocol

`gemini-flash-latest` via the Google AI Studio API, with the resolved `model_version` captured per
call. Audio is submitted as **raw base64 bytes with a MIME type only** — never a filename, path,
or URL, which would leak the condition into the prompt and destroy the manipulation. Prompts are
byte-identical across conditions. Three repeats per cell at fixed decoding settings capture
sampling variability, which matters more than it might appear: Zhang et al. (2026) identify
zero-shot stochasticity in generative SER evaluation as a first-order problem, and a
prompt-sensitivity arm (two alternative emotion prompts on `ref` and `mp4_aac64`) tests whether any
format effect survives rewording.

Call order is randomised so conditions interleave rather than run in blocks, guarding against
model drift across a run that free-tier quota may spread over days. Call sequence position is
recorded and checked as a covariate.

### 3.5 Attribution protocol

The model is closed-weights and API-only. Gradients, attention, activations, and token
log-probabilities are unavailable by construction. Attribution is therefore purely perturbational.

**Temporal occlusion.** Each clip is divided into K = 10 equal windows. Each window in turn is
replaced with speech-shaped noise at matched short-term loudness, with 10 ms raised-cosine edge
ramps to avoid click artefacts.

**Spectral occlusion.** A band-stop filter is applied to one of B = 6 bands spanning 0–8 kHz
(0–250, 250–500, 500–1k, 1–2k, 2–4k, 4–8k Hz), chosen on acoustically interpretable boundaries
rather than arbitrary bins, in partial response to Nasr et al. (2025).

**Score.** With no log-probabilities available, importance is the **label-flip rate**:
`a_j = 1 − (# repeats returning the unmasked modal label) / R`, giving a 16-dimensional attribution
vector per item per condition. R was raised from 3 to 7 after adversarial review. The direction of
the problem is what made the change necessary rather than merely desirable: at R = 3 the measure
takes only four values, ties dominate the rank statistics, and `S_within` and `S_across` are
compressed toward each other — biasing SQ3 **toward a null**, so that the study would report stable
explanations when it had merely failed to measure them.

**Mask-artefact control.** Two null masks per item — a leading-silence window and a band above the
speech energy range — estimate the flip rate attributable to the masking operation alone. This
control is doing unusual work: it is measuring an unmeasured quantity, not confirming a known
small one, and if it comes back large the attribution arm is reported as uninterpretable.

### 3.6 Analysis

**SQ1.** Binomial GLMM, `correct ~ format + task + (1 | item) + (1 | speaker)`, `ref` as reference
level; marginal accuracies with bootstrap CIs; Holm correction across the three planned contrasts.
WER analysed separately under a parallel model; the two tasks are never pooled. The transcription
arm is designated in advance as a **positive control for intelligibility preservation**, not a
second hypothesis test — floor-level WER everywhere is a reportable finding, namely that the format
effect is paralinguistic-specific.

**SQ2.** Effects decomposed across the three contrasts. Because the expected and most informative
container result is a null, a non-significant test is insufficient: equivalence is assessed by two
one-sided tests (Schuirmann, 1987; Lakens, 2017; Lakens et al., 2018) against a pre-declared
smallest effect size of interest. Only a passed equivalence test licenses "the container does not
matter"; a non-significant, non-equivalent result is reported as **inconclusive**, a pre-authorised
outcome with pre-written phrasing.

The bound itself was the subject of the first adversarial finding (§4.2). It is not fixed a priori
but computed from observed discordance as the tightest bound the design can support at 80% power,
declared before the main grid and reported as a limitation.

**SQ3.** Across-format similarity `S_across = ρ(a_ref, a_mp4)` (Spearman, plus top-3 Jaccard) is
compared against the within-format floor `S_within = ρ(a_ref, a_ref′)` from an independent second
pass on `ref`. An explanation shift exists only where `S_across` is reliably below `S_within`.
Tested by Wilcoxon signed-rank on the paired per-item difference, item as unit, with a bootstrap CI
on the median. `S_within` is reported as a **primary result in its own right** — no published
reference value exists for the run-to-run stability of black-box attribution on an audio model.

**Interpretation grid, fixed before data.** All four cells are reportable:

| Accuracy | Explanation | Reading |
|---|---|---|
| Degraded | Diverged | Ordinary degradation; XAI localises where |
| Degraded | Stable | Uniform loss; same evidence base, less of it |
| **Unchanged** | **Diverged** | **The dissociation — benchmarks under-report format sensitivity** |
| Unchanged | Stable | Format-robust at this operating point |

### 3.7 Sequential gates

Each gate's failure is a pre-authorised reportable outcome, not a prompt to relax the gate.

| Gate | Test | Failure response |
|---|---|---|
| **G0** | Reference accuracy ≥ 40% on ≥ 20 items × 3 repeats | Do not run the main grid. Pivot: stronger model; or reduced label set (binary arousal); or transcription-based task. |
| **G0b** | Response degeneracy: modal label constant across all 16 masks for ≤ 30% of items | Flip-rate attribution declared inapplicable; switch to the pre-specified **forced binary contrast** (gold vs. confusable alternative). |
| **G1** | Signal identity: `roundtrip_wav` ≡ decode(`mp4_aac64`), after cross-correlation alignment | SQ2 and SQ3 unanswerable as designed; study reduces to SQ1 and reports why. |
| **G2** | Power calibration: derive Δ from observed discordance | If no usable bound exists, SQ2 is reported as inconclusive by design. |
| **G3** | Main grid (1,350 calls) | — |
| **G4** | XAI arm at R = 7 (≈ 7,280 calls incl. floor and controls) | Reduce to 12 items at R = 5 with a wider, declared noise floor. |

### 3.8 Ethics and disclosure

CREMA-D is publicly released for research with consented actors; no new human-subject data is
collected and no re-identification is attempted. Institutional determination of human-subjects
status has not been sought and is not asserted here. Speech emotion recognition carries a
surveillance and workplace-monitoring dual-use profile; this study measures the *format
sensitivity of a model* and its findings must not be presented as validating emotion inference as
a construct. See §6.4. API keys are stored in a gitignored `.env`; both keys used during setup were
transmitted in plaintext and require rotation. This research was conducted with AI assistance
throughout — agentic literature search, drafting, and analysis scaffolding under human direction
(§7).

---

## 4. Results of the Preparatory Phases

No results are reported for SQ1–SQ3; those await gate clearance. Four preparatory findings are
reported, and three of them changed the design.

### 4.1 The pilot run failed on quota, not on method

An audit of the existing 180-call smoke run found:

| Outcome | Calls | Share |
|---|---:|---:|
| `api_error` — HTTP 429, free-tier quota exceeded | 152 | 84.4% |
| `api_error` — transport `ConnectionError` | 14 | 7.8% |
| `ok` — response received and parsed | **14** | **7.8%** |

The grid was exercised correctly; the quota was not sufficient to run it. Calls spanned
2026-08-09T10:27Z to 2026-08-10T06:33Z, meaning quota exhaustion persisted across roughly twenty
hours. Of the 14 successful calls, 8 were emotion trials, distributed across only `ref` (5) and
`mp3_64` (3). **No successful `mp4_aac64` or `roundtrip_wav` emotion trial survives**, so the
principal contrast has zero usable paired observations.

Two consequences. First, the revised call budget of roughly 9,230 calls is not feasible on the free
tier; billing must be enabled or the design must shrink. Projected spend is approximately US$3.70
against a US$25 hard ceiling enforced in the runner — this is a quota-policy constraint, not a cost
constraint. Second, the power calibration demanded by §4.2 cannot be performed on existing data.

### 4.2 The pre-declared equivalence bound was unreachable

Adversarial review of the design established that for a paired binary outcome with item as the unit
of analysis at n = 50, the minimum detectable effect at α = .05 and 80% power is on the order of
15–18 percentage points under plausible discordance (McNemar, 1947). Repeats reduce measurement
noise but do not increase the number of independent units, because item-level variance dominates.

The originally declared smallest effect size of interest — 3 accuracy points — would therefore
essentially never clear both TOST bounds. SQ2 would return "inconclusive" by arithmetic rather than
by evidence, and, worse, an inconclusive result of that kind is easy to narrate as "no container
effect found", which is exactly the inference equivalence testing exists to prevent. The bound was
withdrawn and replaced by a procedure (derive from observed discordance; declare before the grid;
report as a limitation).

### 4.3 The attribution measure can become undefined, not merely weak

Adversarial review of the analysis established a failure mode the synthesis had classified as a
power problem. If the model returns a constant label irrespective of masking, every `a_j = 0`, the
attribution vector is constant, and Spearman correlation between two constant vectors is
**undefined**. `S_within` and `S_across` both cease to exist. SQ3 — the study's primary question —
would return not a weak answer but no answer, after roughly 7,000 API calls, while producing an
all-zero results table that a careless reader would interpret as perfectly stable explanations.

Gate G0b and the pre-specified forced-binary replacement task were added in response, together with
a code-level guard that raises on constant-vector correlation rather than emitting `nan`.

### 4.4 Two weak signals point at the same risk

Chen et al. (2025) report that LALMs predict *neutral* when lexical cues are neutral or absent.
CREMA-D consists of 12 emotionally neutral fixed sentences (Cao et al., 2014) — the corpus
configuration under which that failure mode is predicted. The pilot's 8 successful emotion trials
were all incorrect, with *neutral* returned for *angry* items.

Neither signal is sufficient alone, and the pilot signal is the weaker of the two by a wide margin:
those 8 observations come from 5 items in 2 of the 4 conditions, drawn from a run in which 92% of
calls failed, and the 95% confidence interval on the underlying accuracy spans roughly 0–37%. It is
a flag, not an estimate. Chen et al. (2025), for its part, is one benchmark whose model coverage may
not include the model under study, and it stands against Zhang et al. (2026), who benchmark speech
LLMs for SER across 35 corpora and 15 languages. What justifies action is not the strength of either
signal but their independence: a published prediction of a specific failure mode, and a
pilot exhibiting that specific failure signature. That combination warrants treating gate G0 as a
live risk rather than a formality — which is what the gate order encodes. It does not warrant any
conclusion about the model's capability, which remains unmeasured.

---

## 5. Discussion

### 5.1 What the preparatory phases actually established

Three things, none of them the intended finding, all of them load-bearing.

The design's central contrast has never been attempted, as far as a targeted search can establish.
Nothing in 23 verified sources separates container from codec by holding the decoded waveform
constant. That gap is narrow enough to be closed by one contrast, and the apparatus to do so
already exists in this project.

The design's central risk is not the one it was built to manage. The protocol was engineered
against sampling noise, model drift, and mask artefacts. The threat that adversarial review and
literature synthesis jointly surfaced is different: the model may not perform the task well enough
for any manipulation to have room to act, and if it degenerates to a constant response the primary
measure does not weaken but vanishes.

The design's most certain output is a quantity it originally treated as an inconvenience. No
published value exists for how much a black-box attribution map varies between repeated runs under
no manipulation. `S_within` must be measured regardless of how anything else resolves. Given that
the format effect is predicted flat and the container effect may be inconclusive, this may be the
most durable and reusable result the study produces.

### 5.2 Implications if the gates clear

If G0 and G0b clear and the grid runs, the interpretation grid in §3.6 governs. The cell of
interest — unchanged accuracy with diverged explanation — would indicate that accuracy-only
benchmarking is blind to a form of format sensitivity present in every transcoding pipeline, and
would extend Ghorbani et al.'s (2019) fragility result from adversarially constructed perturbations
to ordinary engineering operations. That is a stronger claim in one respect than the original,
because nobody has to construct anything: the perturbation is what deployment already does.

A passed equivalence test on the container contrast would be a genuinely useful negative result,
provided two claims are kept apart.

The **practical** claim survives either reading: if behaviour is equivalent across containers
carrying the same decoded waveform, format-agnostic submission is safe for this model at this
operating point — safe because the container is inert, or safe because the ingestion path decodes
to PCM before anything model-specific happens. The operator does not need to know which.

The **mechanistic** claim does not survive. "The container does not matter" and "the container
never reached the model as a container" are different states of the world that this contrast
cannot separate (§6.3), and only the additional metadata probe — a container with intact audio
payload but altered container-level fields — would begin to distinguish them. The report must not
let the practical licence carry the mechanistic conclusion along with it.

### 5.3 Implications if the gates do not clear

The honest output is a methods note: the protocol, the stability floor, and a documented account of
why a lexically reliant model on a fixed-lexicon corpus cannot support a format experiment. That is
a smaller contribution than intended and a real one. Pre-committing to it now is what separates
this from a study that redefines success after seeing its data.

Should G0 fail, the literature indicates the pivot rather than leaving it to improvisation: reduce
the label set to a high/low-arousal binary, or move to a paralinguistic task with no lexical
shortcut. Both preserve the format manipulation while removing the escape route — and arguably
produce a better study, because a task with no lexical shortcut is one where compression has
something to destroy.

### 5.4 A reporting-practice argument that holds regardless

Across 23 verified sources spanning model reports, benchmarks, robustness evaluations, and a recent
survey, **not one states the container in which audio was submitted**. Whether or not container
turns out to matter, that is an unforced gap in reproducibility. Lee et al. (2025) already document
that cross-model comparison is confounded by unreported prompting and inference settings; input
encoding belongs on that list. REFORMS (Kapoor et al., 2024) provides the natural home for such a
requirement. This recommendation costs nothing and does not depend on any result in this study.

---

## 6. Limitations

### 6.1 Scope
Single model, single corpus, single bitrate, acted emotion, English, single-speaker, clean audio.
External validity is traded for internal validity deliberately. Zhang et al. (2026) evaluate across
35 corpora and 15 languages; this study is 50 clips. The reusable output is the protocol, not a
general claim about audio LLMs.

### 6.2 Attribution validity
The interpretable representation — time windows and frequency bands — is the one audioLIME
explicitly critiques as non-listenable (Haunschmid et al., 2020), and Nasr et al. (2025) extend the
critique to SER. Source separation, the recommended remedy, does not apply to single-speaker
speech. Mitigations (ramped edges, loudness-matched filler, null-mask controls) reduce the artefact
risk without discharging the objection. Perturbation-based attribution identifies what the output
*depends on*, not what the model *attends to*, and the report holds that distinction in its section
headings, not only in its caveats.

### 6.3 Measurement and inference
Label-flip rate at R = 7 is an eight-level measure — coarse, mitigated by item-level aggregation.
Confidence is proxied by repeat agreement because log-probabilities are unavailable. The XAI subset
is selected on label agreement, which biases SQ3 *conservatively* (toward stability) and supports no
accuracy inference. A null on SQ2 is evidence for a **disjunction** — the container is irrelevant,
*or* the ingestion path decodes to PCM immediately — and the design cannot distinguish them.

### 6.4 Evidence base
Semantic Scholar was unavailable throughout (HTTP 429), so verification rested on two indexes
rather than three. The corpus excludes paywalled and non-English SER-robustness work. The novelty
claim is an absence-of-evidence result from a targeted, non-exhaustive search: it supports "we found
no prior work doing this", not "no prior work exists".

### 6.5 Ethics
Findings concern format sensitivity of a model, not the validity of automatic emotion recognition.
Acted emotion is a controlled acoustic manipulation, not evidence about emotion in the wild. Given
the surveillance and workplace-monitoring applications of this technology, results should not be
cited as support for deploying emotion inference.

### 6.6 Responsible Use Statement

This work characterises how a machine-learning system responds to changes in audio file format. It
provides **no evidence** that automatic speech emotion recognition is accurate, fair, or
appropriate for use in consequential settings — hiring, policing, border control, insurance,
clinical assessment, workplace or classroom monitoring, or any other context where a person is
subject to an inference about their emotional state.

The emotion task is used here as an instrument: a forced-choice paralinguistic judgement sensitive
enough to register acoustic degradation. Its role is comparable to a test signal. Any result
reported by this study — including a finding that the model is *robust* to format change — speaks
to the stability of a measurement pipeline and not to the validity of the underlying construct.
Readers citing this work in support of deploying emotion recognition would be citing it for a claim
it does not make and its design cannot support.

---

## 7. Conclusion and Recommendations

This report specifies a protocol for a question current practice has not asked — whether the
container an audio file arrives in changes what a multimodal LLM's output depends on — and reports
the preparatory work that reshaped it. The interpretation of that work is given in §5 and is not
restated here.

**Recommendations, in execution order:**

1. **Enable billing** on the API key. The full design is quota-blocked, not cost-blocked
   (≈ US$3.70 projected against a US$25 ceiling).
2. **Run gate G0** — reference accuracy on ≥ 20 items × 3 repeats — before anything else, and
   decide in advance whether the 40% threshold is a deliberate benchmark against the 40.9% human
   audio-only rate reported by Cao et al. (2014) — a comparison that is indicative rather than
   like-for-like, since those raters judged the full corpus without a forced-choice prompt — or
   should instead be a lower criterion chosen purely for headroom.
3. **Run gate G0b** — response degeneracy — before committing to the XAI arm, and switch to the
   forced binary contrast if it fires.
4. **Preregister** the interpretation grid, the equivalence-bound derivation procedure, the
   degeneracy threshold, and the conditional-analysis rule. The study predicts its own null; a
   registered report is the natural venue strategy.
5. **Verify signal identity** by cross-correlation alignment before interpreting any container
   result, reporting recovered AAC encoder delay and duration difference as data.
6. **Report `S_within` as a primary result**, whatever else happens.
7. **Rotate both API keys** at pilot end.

The study may not survive its own gates. It was designed so that finding out is cheap and saying so
is pre-authorised.

---

## AI Disclosure

This research was conducted with substantial AI assistance under human direction. An agentic
pipeline performed literature search, source verification against the Crossref and arXiv APIs,
evidence synthesis, adversarial design review, and drafting of this report. The research question
framing was selected by the human researcher at a decision checkpoint. All 23 cited sources were
verified programmatically against bibliographic indexes; the audit trail is retained in
`reference_verification.json`. No source is cited that could not be confirmed to exist. The
empirical claims in §4 derive from direct inspection of this project's own data files and are
reproducible from them.

---

## References

Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). *Sanity checks
for saliency maps* (arXiv:1810.03292). arXiv. https://arxiv.org/abs/1810.03292

Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014). CREMA-D:
Crowd-sourced emotional multimodal actors dataset. *IEEE Transactions on Affective Computing,
5*(4), 377–390. https://doi.org/10.1109/TAFFC.2014.2336244

Chen, J., Guo, Z., Chun, J., Wang, P., Perrault, A., & Elsner, M. (2025). *Do audio LLMs really
LISTEN, or just transcribe? Measuring lexical vs. acoustic emotion cues reliance*
(arXiv:2510.10444). arXiv. https://arxiv.org/abs/2510.10444

Chu, Y., Xu, J., Yang, Q., Wei, H., Wei, X., Guo, Z., Leng, Y., Lv, Y., He, J., Lin, J., Zhou, C.,
& Zhou, J. (2024). *Qwen2-Audio technical report* (arXiv:2407.10759). arXiv.
https://arxiv.org/abs/2407.10759

Ghorbani, A., Abid, A., & Zou, J. (2019). *Interpretation of neural networks is fragile*
(arXiv:1710.10547). arXiv. https://arxiv.org/abs/1710.10547

Haunschmid, V., Manilow, E., & Widmer, G. (2020). *audioLIME: Listenable explanations using source
separation* (arXiv:2008.00582). arXiv. https://arxiv.org/abs/2008.00582

Hu, H., You, L., Xu, H., Wang, Q., Yu, F. R., & Ma, F. (2025). *EmoBench-M: Benchmarking emotional
intelligence for multimodal large language models* (arXiv:2502.04424). arXiv.
https://arxiv.org/abs/2502.04424

Kapoor, S., Cantrell, E. M., Peng, K., Pham, T. H., Bail, C. A., Gundersen, O. E., Hofman, J. M.,
Hullman, J., Lones, M. A., Malik, M. M., Nanayakkara, P., Poldrack, R. A., Raji, I. D., Roberts,
M., Salganik, M. J., … Narayanan, A. (2024). REFORMS: Consensus-based recommendations for
machine-learning-based science. *Science Advances, 10*(18). https://doi.org/10.1126/sciadv.adk3452

Lakens, D. (2017). Equivalence tests: A practical primer for *t* tests, correlations, and
meta-analyses. *Social Psychological and Personality Science, 8*(4), 355–362.
https://doi.org/10.1177/1948550617697177

Lakens, D., Scheel, A. M., & Isager, P. M. (2018). Equivalence testing for psychological research:
A tutorial. *Advances in Methods and Practices in Psychological Science, 1*(2), 259–269.
https://doi.org/10.1177/2515245918770963

Lee, T., Tu, H., Wong, C. H., Wang, Z., Yang, S., & Mai, Y. (2025). *AHELM: A holistic evaluation
of audio-language models* (arXiv:2508.21376). arXiv. https://arxiv.org/abs/2508.21376

Li, K., Shen, C., Liu, Y., Han, J., Zheng, K., Zou, X., Wang, L. Z., Zhang, S., Du, X., Luo, H.,
Jin, Y., Xing, X., Ma, Z., Liu, Y., Zhang, Y., Fang, J., Wang, K., Yan, Y., Deng, G., … Li, X.
(2025). *AudioTrust: Benchmarking the multifaceted trustworthiness of audio large language models*
(arXiv:2505.16211). arXiv. https://arxiv.org/abs/2505.16211

Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions*
(arXiv:1705.07874). arXiv. https://arxiv.org/abs/1705.07874

Luo, K., Zhou, Z., Wang, L., Lin, L., Shao, T., & Zhang, Y. (2026). *A survey of large audio
language models: Generalization, trustworthiness, and outlook* (arXiv:2605.20266). arXiv.
https://arxiv.org/abs/2605.20266

McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or
percentages. *Psychometrika, 12*(2), 153–157. https://doi.org/10.1007/BF02295996

Nasr, S., Ren, Z., & Johnson, D. (2025). *Beyond saliency: Enhancing explanation of speech emotion
recognition with expert-referenced acoustic cues* (arXiv:2511.11691). arXiv.
https://arxiv.org/abs/2511.11691

Reddy, A. P., & Vijayarajan, V. (2020). Audio compression with multi-algorithm fusion and its
impact in speech emotion recognition. *International Journal of Speech Technology, 23*(2),
277–285. https://doi.org/10.1007/s10772-020-09689-9

Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the
predictions of any classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining* (pp. 1135–1144). Association for Computing Machinery.
https://doi.org/10.1145/2939672.2939778

Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach
for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and
Biopharmaceutics, 15*(6), 657–680. https://doi.org/10.1007/BF01068419

Sotirou, T., Lyberatos, V., Menis Mastromichalakis, O., & Stamou, G. (2024). *MusicLIME:
Explainable multimodal music understanding* (arXiv:2409.10496). arXiv.
https://arxiv.org/abs/2409.10496

Wu, H., Chen, X., Lin, Y.-C., Chang, K., Du, J., Lu, K.-H., Liu, A. H., Chung, H.-L., Wu, Y.-K.,
Yang, D., Liu, S., Wu, Y.-C., Tan, X., Glass, J., Watanabe, S., & Lee, H. (2024). *Codec-SUPERB @
SLT 2024: A lightweight benchmark for neural audio codec models* (arXiv:2409.14085). arXiv.
https://arxiv.org/abs/2409.14085

Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. In
*Computer Vision – ECCV 2014* (Lecture Notes in Computer Science, Vol. 8689, pp. 818–833).
Springer. https://doi.org/10.1007/978-3-319-10590-1_53

Zhang, H., Chou, H.-C., Narayanan, S., & Hain, T. (2026). *VoxEmo: Benchmarking speech emotion
recognition with speech LLMs* (arXiv:2603.08936). arXiv. https://arxiv.org/abs/2603.08936

---

## Appendix A — Reference verification audit

Machine-readable trail: `research/phase2_investigation/reference_verification.json`.
23/23 verified. Method per source: Crossref DOI lookup with Levenshtein 0.70 title cross-check
(9 sources), Crossref title search after `DOI_MISMATCH` on a truncated-title record (2 sources),
arXiv identifier lookup (12 sources). Semantic Scholar unavailable throughout
(`[S2-API-UNAVAILABLE]`, HTTP 429 on all unauthenticated requests).

## Appendix B — Pipeline artefacts

| Phase | Artefact |
|---|---|
| 1 | `phase1_scoping/rq_brief.md`, `methodology_blueprint.md`, `da_checkpoint1.md`, `feasibility_addendum.md` |
| 2 | `phase2_investigation/search_strategy.md`, `annotated_bibliography.md`, `reference_verification.json` |
| 3 | `phase3_analysis/synthesis.md`, `da_checkpoint2.md` |
| 4 | `phase4_composition/report_draft.md` |
| 5 | `phase5_review/review_panel.md` — editorial verdict, integrity review, DA Checkpoint 3 |
| 6 | `phase6_final/final_report.md` (this document), `handoff_pack.md` |

## Appendix C — Revision record (Phase 6)

Revision loop 1 of a maximum 2. All Required, Major, and Advisory findings from
`phase5_review/review_panel.md` were addressed; nothing was deferred to Acknowledged Limitations.

| ID | Finding | Resolution |
|---|---|---|
| E1 | Title over-promised results | Retitled to name the genre |
| E2 | Abstract buried the genre | Genre stated in the abstract's first sentence |
| E3 | §4.4 reasoned from a weak pilot signal | Weakness stated in-paragraph with the CI; conclusion narrowed to "justifies a gate, not an estimate" |
| E4 | §7 restated §5.1 | §7 opening compressed; interpretation left to §5 |
| E5 | Modal-outcome forecast lacked uncertainty | Ranked scenario table added at §2.6, explicitly labelled a judgement call |
| E6 | Human-parity caveat not repeated at §7 | Caveat restated at recommendation 2 |
| ETH1 | Dual-use boundary distributed across caveats | §6.6 Responsible Use Statement added |
| M9 | Genre load-bearing but undefended | §1.2 now argues why the protocol is the deliverable |
| M10 | Practical licence conflated with mechanistic conclusion | §5.2 separates them explicitly |
| m1 | Tie-bias direction unstated | §3.5 states that ties bias SQ3 *toward the null* |
| m2 | Mentalist phrasing | "decision rests on" → "output depends on" |
| m3 | Appendix listed unwritten artefacts | Appendix B updated to actual artefacts |
