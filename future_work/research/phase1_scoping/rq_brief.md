# Research Question Brief
**Agent:** `research_question_agent` | **Phase:** 1 (Scoping) | **Date:** 2026-08-17

---

## Topic Area

Whether and how the *delivery format* of an audio signal — a lossless WAV container versus a
lossy MP4/AAC (or MP3) container — changes the behaviour of a closed-weights multimodal large
language model on a speech understanding task, and whether explainable-AI (XAI) attribution
methods can localise the mechanism of any such change.

The topic sits at the intersection of three literatures that have so far not been connected:

1. **Codec robustness in speech technology** — a mature literature on how lossy compression
   degrades ASR and speech emotion recognition (SER) for *task-specific* models.
2. **Multimodal / audio LLM evaluation** — a young literature that overwhelmingly evaluates on
   clean, uncompressed corpora and rarely reports the container in which audio was submitted.
3. **XAI for audio** — perturbation- and surrogate-based attribution (occlusion, audioLIME/SLIME,
   SHAP over time–frequency segments) developed for classifiers, not for API-only generative models.

---

## Primary Research Question

> **Does lossy container delivery (MP4/AAC at 64 kbps) alter which acoustic regions a multimodal
> LLM relies on for speech emotion recognition, beyond any change in its accuracy?**

The question is deliberately *explanation-first*. The accuracy question ("does MP4 hurt?") is
largely settled for task-specific models and is treated here as a necessary but subordinate
sub-question. The novel claim under test is that **performance-level robustness and
explanation-level robustness can dissociate** — the model can return the same label from a
materially different part of the signal, which a pure accuracy benchmark cannot detect.

---

## FINER Assessment

| Criterion | Score | Justification |
|-----------|-------|---------------|
| **F**easible | 4/5 | Stimuli, transform pipeline, model client, and a working 180-call smoke run already exist in this repository. Perturbation-based XAI needs only forward API calls — no weights, gradients, or attention access. Cost is bounded (~$0.73 projected for the 1,800-call main grid; the XAI arm adds a bounded number of masked variants per item). The −1 reflects API-side non-determinism and free-tier rate limits, which extend wall-clock but not method validity. |
| **I**nteresting | 4/5 | The interesting result is a *dissociation*: equal accuracy with divergent attributions would mean current audio-LLM benchmarks systematically under-report format sensitivity. A null result is also informative (it would license format-agnostic pipelines). |
| **N**ovel | 4/5 | Codec effects on ASR/SER are well studied; XAI for audio classifiers is well studied; **neither has been applied to a closed multimodal LLM with a container-vs-codec decomposition.** The `mp4_aac64` vs `roundtrip_wav` contrast (byte-identical decoded PCM, different container) is, to our knowledge, not present in the audio-LLM evaluation literature. Novelty is claimed for the *combination and the contrast*, not for any single component. |
| **E**thical | 4/5 | CREMA-D is a public, consented, acted-emotion corpus released for research. No new human data collection. The −1 acknowledges (a) the dual-use profile of emotion recognition and (b) that findings about *format* must not be laundered into claims about real-world emotion inference validity. Both are addressed explicitly in the ethics review. |
| **R**elevant | 4/5 | Every production audio pipeline transcodes. If container choice moves either accuracy or the model's evidence base, that is directly actionable for anyone deploying an audio LLM, and it prescribes a reporting norm (state your container) for benchmark authors. |
| **Average** | **4.0/5** | Threshold (≥3.0 average, no criterion <2) satisfied. |

---

## Scope Boundaries

**In scope**
- **Model**: one closed-weights multimodal LLM accessed via API — Google `gemini-flash-latest`
  (resolving to `gemini-3.6-flash`, captured per call as `model_version`).
- **Task**: forced-choice speech emotion recognition over a fixed 6-way lexicon
  (`angry, disgusted, fearful, happy, neutral, sad`); verbatim transcription (WER) as a
  secondary, contrasting task.
- **Corpus**: 50 CREMA-D clips, fixed-lexicon sentences, stratified by emotion.
- **Formats**: four conditions derived from a single canonical reference —
  `ref` (16 kHz mono PCM16, EBU R128 normalised WAV), `mp3_64` (MP3 @ 64 kbps),
  `mp4_aac64` (audio-only MP4, AAC @ 64 kbps), `roundtrip_wav` (MP4/AAC decoded back to
  canonical PCM and delivered as WAV).
- **XAI family**: black-box, perturbation-based attribution only — temporal occlusion,
  spectral band occlusion, and a segment-level surrogate (LIME-style) — plus repeat-based
  response entropy as a confidence proxy.

**Out of scope**
- White-box XAI (gradients, integrated gradients, attention rollout, probing classifiers).
  The target model is API-only; these are unavailable *by construction*, not by choice.
- Bitrate sweeps and codec ladders (e.g. AAC at 16/32/96/128 kbps). One operating point per
  codec keeps the grid tractable; bitrate is named as the primary extension.
- Open-weights audio LLMs (Qwen2-Audio, SALMONN, Phi-4-multimodal). Phi-4 was probed and
  rejected for this design: it rejects MP4/AAC input and decodes transparently via libsndfile,
  which collapses the container-routing contrast. It is retained only as a possible
  transparent-decode contrast in later work.
- Claims about human emotion perception, or about the validity of automatic emotion
  recognition as a construct. Acted emotion is a proxy for a controlled acoustic manipulation,
  nothing more.
- Multi-speaker, noisy, far-field, or non-English audio.

**Key assumptions**
1. **A1 — Opaque decode.** The target model's audio ingestion path is not a simple
   decode-to-PCM front end shared across containers; if it were, `mp4_aac64` and
   `roundtrip_wav` would be behaviourally identical by construction. This assumption is
   *testable within the design*, not merely assumed: sub-question 2 measures it.
2. **A2 — Signal identity.** `roundtrip_wav` carries the same decoded waveform as `mp4_aac64`
   to within the tolerance of the decoder. This must be **verified numerically** (sample-level
   or near-sample-level agreement) before any container claim is made. Verification is a
   blocking prerequisite, not an assumption to be waved through.
3. **A3 — Perturbation validity.** Masking a time or frequency region and observing a label
   change is evidence that the region carried decision-relevant information, *conditional on*
   the mask itself not creating out-of-distribution artefacts that independently confuse the
   model. Mask design (ramped edges, matched-loudness filler) and a mask-only control address
   this directly.
4. **A4 — Stability under sampling.** Three repeats per cell at fixed decoding settings are
   sufficient to separate genuine format effects from API sampling noise. The smoke run informs
   whether this holds; if repeat variance is large, repeats must increase before attribution is
   interpreted.

---

## Sub-questions

1. **SQ1 (Performance).** How large is the change in 6-way SER accuracy — and in transcription
   WER — when the same canonical signal is delivered as `ref`, `mp3_64`, `mp4_aac64`, and
   `roundtrip_wav`?
2. **SQ2 (Mechanism).** Of any degradation observed under `mp4_aac64`, how much is attributable
   to signal-level codec loss (recoverable from the `roundtrip_wav` vs `ref` contrast) versus to
   container delivery itself (the `mp4_aac64` vs `roundtrip_wav` contrast, where the decoded
   signal is held constant)?
3. **SQ3 (Explanation).** On items where the predicted label is *unchanged* between `ref` and
   `mp4_aac64`, do perturbation-based attribution maps nonetheless diverge — and is that
   divergence larger than the map instability observed between repeated runs on `ref` alone?

---

## Sub-Question Bindings (#547)

1. **SQ1** — inherits: model=`gemini-flash-latest`/`gemini-3.6-flash`; corpus=50 CREMA-D clips;
   formats=all four conditions; tasks=emotion + transcribe; timeframe=single pilot run window.
   deviations: none.
2. **SQ2** — inherits: all SQ1 bindings. deviations: none. Restricted to the
   `mp4_aac64` / `roundtrip_wav` / `ref` triad; `mp3_64` contributes only as a second lossy
   reference point, since no MP3 round-trip control is generated.
3. **SQ3** — inherits: model, corpus, timeframe as above. deviations: **narrowed by design** —
   formats restricted to `ref` and `mp4_aac64`; task restricted to emotion (transcription has no
   single-label decision to attribute); item set restricted to the label-agreement subset.
   This is a narrowing within the parent scope, not a broadening, and requires no approval.

---

## Candidate Questions Considered

| # | Candidate | FINER Avg | Why not selected |
|---|-----------|-----------|------------------|
| 1 | *Does lossy container delivery (MP4/AAC @64k) alter which acoustic regions a multimodal LLM relies on for SER, beyond any change in its accuracy?* | **4.0** | **Selected.** Explanation-first, single-clause, method-implying, and the only candidate whose null result is still publishable. |
| 2 | *To what extent does audio codec and container format degrade multimodal LLM speech emotion recognition accuracy?* | 3.4 | Feasible and relevant but Novel = 2: "lossy compression degrades speech tasks" is established for task-specific models, and extending it to one more model class is incremental. Retained as SQ1. |
| 3 | *Does the MP4 container itself, independent of AAC codec degradation, change model behaviour?* | 3.8 | Sharp and genuinely novel, but its answer is plausibly a flat null (most ingestion stacks decode to PCM early), and a null on a single binary contrast makes a thin study. Retained as SQ2, where it functions as a mechanism test rather than the headline. |
| 4 | *Can perturbation-based attribution on the clean reference predict which items will flip label under compression?* | 3.6 | Attractive and forward-looking, but Feasible = 3: predictive validity needs a sufficient count of flipped items, and the smoke run cannot guarantee that count exists at n=50. Recorded as a **conditional secondary analysis** — run only if the main grid yields enough flips to support it. |
| 5 | *How do open-weights and closed-weights audio LLMs differ in format robustness?* | 2.8 | Feasible = 2 under current constraints: the one available open-weights endpoint (Phi-4-multimodal) rejects MP4/AAC and decodes transparently, so the central contrast cannot be constructed. Deferred to future work. |

---

## Handoff

- Consumed by: `research_architect_agent` (Phase 1, methodology blueprint) and
  `devils_advocate_agent` (Checkpoint 1).
- Blocking prerequisite flagged for the architect: **assumption A2 must be numerically verified**
  before SQ2 or SQ3 results may be interpreted.
