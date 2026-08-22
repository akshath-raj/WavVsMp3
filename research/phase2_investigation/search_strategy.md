# Search Strategy & Screening Record
**Agents:** `bibliography_agent` + `source_verification_agent` | **Phase:** 2 (Investigation)
**Date:** 2026-08-17

---

## Search objective

Phase 1 handed Phase 2 five specific evidence targets, of which (e) was framed as a falsification
task rather than a confirmation task:

| # | Target | Outcome |
|---|--------|---------|
| (a) | Codec / bitrate effects on SER and ASR | **Found** — mature evidence base |
| (b) | Audio-LLM robustness evaluation practice | **Found** — young but active |
| (c) | Perturbation-based XAI for audio + faithfulness controls | **Found** — established methods |
| (d) | Explanation-stability metrics | **Found** — strong theoretical precedent |
| (e) | Prior container-vs-codec decomposition (novelty falsification) | **Not found** — see caveat below |

---

## Databases and indexes queried

| Index | Role | Status |
|---|---|---|
| Web search (general academic) | Discovery | Used |
| arXiv API (`export.arxiv.org`) | Preprint verification | Used — all queries resolved |
| Crossref API (`api.crossref.org`) | DOI registry of record | Used — polite pool, `mailto` in User-Agent |
| Semantic Scholar Graph API | Tier-0 verification | **`[S2-API-UNAVAILABLE]`** — every unauthenticated request returned HTTP 429 in this session. Per `semantic_scholar_api_protocol.md`, the pipeline degrades gracefully to the remaining indexes rather than blocking. Verification therefore rests on two indexes, not three. |

## Search strings (representative)

```
effect of audio codec compression bitrate on speech emotion recognition accuracy
effect of MP3 AAC compression on automatic speech recognition word error rate
emotion recognition speech arousal compressed speech AMR codec degradation
audio large language model robustness evaluation noise compression perturbation benchmark
large audio language model speech emotion recognition benchmark Gemini Qwen2-Audio CREMA-D
audioLIME SLIME interpretable explanation music audio source separation LIME
explainable AI speech emotion recognition attribution saliency time-frequency occlusion
"interpretation of neural networks is fragile" saliency map instability sanity checks
audio file format container effect on model predictions WAV vs MP3 evaluation confound
audio language model input encoding sample rate resampling sensitivity "same audio" different format
Codec-SUPERB neural audio codec benchmark downstream paralinguistic emotion information loss
Lakens equivalence testing TOST two one-sided tests primer
REFORMS reporting standards machine learning based science
```

## Inclusion / exclusion criteria

**Included** if the source (i) reports empirical evidence on how signal degradation affects a
speech task, **or** (ii) defines or evaluates a perturbation-based attribution method, **or**
(iii) establishes a methodological standard the design depends on (equivalence testing, ML
reporting, paired proportion tests), **or** (iv) documents the corpus or a model class under study.

**Excluded**: vendor blog posts and tutorial sites (encountered frequently for the "WAV vs MP3
for ML" query and uniformly non-empirical); patents; content-farm summaries; any source whose
existence could not be confirmed in Crossref or arXiv.

## Screening flow

```
Records surfaced across 12 searches                      ~110
  ├─ removed: tutorials, blogs, vendor pages, patents     ~55
  ├─ removed: duplicates across indexes                   ~20
  └─ screened on title/abstract                            35
        ├─ excluded: peripheral to all five targets        16
        └─ carried to verification                         19
              ├─ VERIFIED (Crossref DOI or arXiv ID)       19
              └─ NOT FOUND / unverifiable                    0
```

Verification audit trail: `reference_verification.json` (machine-readable, one record per source,
with match method and similarity score).

---

## Verification results

**19 / 19 sources verified.** No source entered the corpus on a "difficult to verify" basis —
per the skill's IRON RULE, the gray zone is a FAIL, and nothing landed there.

Two records required the documented fallback path and are disclosed rather than silently resolved:

| Source | Event | Resolution |
|---|---|---|
| Ribeiro et al. (2016) | `DOI_MISMATCH (0.49)` on `10.1145/2939672.2939778` | Crossref stores the KDD record under the truncated title *"Why Should I Trust You?"*. Direct DOI re-fetch confirms authors (Ribeiro, Singh, Guestrin), venue (Proc. 22nd ACM SIGKDD), pages 1135–1144, 2016-08-13. **Record is genuine**; the mismatch was a metadata-truncation artefact, not a fabrication signal. |
| Lakens (2017) | `DOI_MISMATCH (0.36)` on `10.1177/1948550617697177`; title search resolved to the PsyArXiv preprint | Crossref stores the SPPS record under the truncated title *"Equivalence Tests"*. Direct DOI re-fetch confirms *Social Psychological and Personality Science*, 8(4), 355–362, 2017-05. **The published version is cited**, not the preprint. |

Both are exactly the pattern the protocol's title cross-check is designed to surface, and in both
cases the check fired correctly on a real record. Neither is a DOI-hallucination case.

---

## Novelty falsification result (target (e))

Phase 1 scored FINER Novelty at 4/5 and instructed Phase 2 to falsify it. **The claim survived,
but weakened in one place and strengthened in another.**

**Weakened.** The proposition "the model's decision may rest on a different acoustic basis after a
perturbation that leaves its output unchanged" is *not* novel. Ghorbani et al. (2019) established
exactly this for image classifiers, and Adebayo et al. (2018) showed that attribution methods can
fail basic sanity checks entirely. The conceptual contribution of SQ3 is therefore a **transfer**,
not an invention: from adversarial perturbation in vision to deployment-realistic transcoding in
audio, and from white-box gradient attribution to black-box occlusion on a commercial API. That
framing must be used; claiming the dissociation itself as new would be an overstatement a reviewer
would catch immediately.

**Strengthened.** No source was found that separates *container* from *codec* by holding the
decoded waveform constant, for any audio model. Searches aimed directly at this returned only
practitioner guidance ("prefer WAV, MP3 loses information") and codec-degradation studies that
vary the signal and the format together. The `mp4_aac64` / `roundtrip_wav` contrast appears to be
unattempted.

**Caveat, recorded as a limitation rather than buried:** this is absence of evidence from a
targeted but non-exhaustive search, run without Semantic Scholar and without paywalled
full-text indexes. It supports "we found no prior work doing this", which is what the report will
say. It does not support "no prior work exists", which the report will not say.

**Revised FINER Novelty: 4/5 → 3.5/5.** The transfer is genuine and the container contrast is
unclaimed, but the core intuition is inherited from an existing literature that must be credited
prominently.

---

## The single most consequential finding for this study

Chen et al. (2025) report that large audio-language models exhibit **lexical dominance**: they
"predict *neutral* when lexical cues are neutral or absent", show limited gains when lexical and
acoustic cues align, and fail under cue conflict, with paralinguistic-context performance
"approaching chance levels" — concluding that current models "largely *transcribe* rather than
*listen*".

CREMA-D consists of 12 fixed, emotionally neutral sentences (Cao et al., 2014). It is therefore
**precisely the corpus configuration under which Chen et al. predict a collapse to *neutral***.

The existing smoke run (`feasibility_addendum.md`) produced 8 parsed emotion responses, all
incorrect, with `neutral` returned for `angry` items. n = 8 is far too small to be evidence on its
own — but it is the exact failure signature an independent published benchmark predicts for this
corpus. Two weak signals pointing the same way warrant treating gate G0 as a live risk, not a
formality.

This is escalated to Phase 3 as the central threat to the study's viability, and to
`da_checkpoint2.md` for adversarial handling.
