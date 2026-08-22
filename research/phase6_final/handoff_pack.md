# Handoff Pack — `deep-research` → `academic-paper`
**Phase:** 6 (Final) | **Date:** 2026-08-17 | **Pipeline:** ARS `deep-research` v2.12.1, full mode

This pack is what `academic-paper`'s `intake_agent` consumes if the project proceeds to
manuscript preparation. Trigger phrase: *"now write a paper based on this"*.

---

## 1. Materials available

| Material | Artefact | Intake effect |
|---|---|---|
| Research Question Brief | `phase1_scoping/rq_brief.md` | Skip topic scoping |
| Methodology Blueprint | `phase1_scoping/methodology_blueprint.md` (+ Amendments) | Skip design derivation |
| Feasibility audit | `phase1_scoping/feasibility_addendum.md` | Feeds Limitations + Methods |
| Annotated Bibliography | `phase2_investigation/annotated_bibliography.md` | Skip literature search |
| Verification audit trail | `phase2_investigation/reference_verification.json` | Skip citation re-verification |
| Search strategy | `phase2_investigation/search_strategy.md` | Feeds Methods reproducibility |
| Synthesis + gap analysis | `phase3_analysis/synthesis.md` | Accelerates Findings / Discussion |
| Adversarial review record | `phase1_scoping/da_checkpoint1.md`, `phase3_analysis/da_checkpoint2.md`, `phase5_review/review_panel.md` | Pre-empts reviewer objections |
| Full report | `phase6_final/final_report.md` | Base manuscript |

**Preregistration sidecar:** `not_provided`. No builder-produced
`preregistration-artifact/1.0` sidecar exists — the non-shell architect supplies only the caller
declaration, and the deterministic `build-preregistration-artifact` subcommand has not been run.
Companion handle: none. A downstream consumer must not synthesise one from prose.

---

## 2. Decisions already locked (do not relitigate at intake)

1. **RQ hierarchy: explanation-first.** Selected by the human researcher at the Phase 1 checkpoint.
   SQ3 (attribution divergence) is the headline; SQ1 (accuracy) and SQ2 (container vs. codec) are
   supporting.
2. **Corpus: CREMA-D, fixed 50-item stratified sample.** Load-bearing and frozen. Do not swap.
3. **Model: `gemini-flash-latest`.** Chosen after Foundry probing; Phi-4-multimodal rejected
   because it refuses MP4/AAC and decodes transparently, collapsing the container contrast.
4. **XAI family: black-box perturbation only.** White-box methods are unavailable by construction,
   not by preference.
5. **Reporting standard: REFORMS** (Kapoor et al., 2024).
6. **Genre: protocol + feasibility report**, defended in §1.2 of the final report.

---

## 3. Open decisions requiring the researcher

| # | Decision | Why it cannot be defaulted |
|---|---|---|
| D1 | Enable API billing, or shrink the design | The full grid (≈9,230 calls, ≈US$3.70) is quota-blocked on the free tier. This is a spending decision only the account owner can make. |
| D2 | Set the G0 threshold | Is 40% a deliberate benchmark against the 40.9% human audio-only rate (Cao et al., 2014), or should it be lower purely for headroom? Changes what "the study is viable" means. |
| D3 | Preregister, and where | The study predicts its own null. A registered report is the natural venue strategy; OSF or AsPredicted for the lighter form. |
| D4 | Pivot target if G0 fails | Stronger model, binary arousal contrast, or a paralinguistic task with no lexical shortcut. The literature favours the last; the researcher owns the call. |
| D5 | Rotate both API keys | Both were transmitted in plaintext during setup. |

---

## 4. Execution checklist (engineering, not writing)

Ordered. Each gate's failure is a reportable outcome with pre-declared phrasing — see final report
§3.7.

- [ ] Enable billing on `GEMINI_API_KEY` (D1)
- [ ] Add `roundtrip_wav_mp3` to `configs/transforms.yaml` (DA CP1, M4)
- [ ] Implement fidelity gate with **cross-correlation alignment** before any deviation metric;
      report recovered AAC encoder delay and duration delta per item (DA CP1, M1)
- [ ] **Gate G0** — reference accuracy, ≥20 items × 3 repeats
- [ ] **Gate G0b** — response degeneracy; switch to forced binary contrast if it fires (DA CP2, C2)
- [ ] Add analysis guard: constant-vector Spearman must **raise**, never return `nan`
- [ ] **Gate G2** — derive equivalence bound Δ from observed `ref`↔`mp4_aac64` discordance (DA CP1, C1)
- [ ] Preregister interpretation grid, Δ-derivation procedure, degeneracy threshold, and the
      ≥15-flip conditional-analysis rule (D3)
- [ ] **Gate G3** — main grid, 1,350 calls, randomised interleaved call order
- [ ] **Gate G4** — XAI arm at R = 7, incl. `S_within` second pass and null-mask controls
- [ ] Report `S_within` as a primary result regardless of other outcomes (DA CP2, M8)
- [ ] Rotate keys (D5)

---

## 5. Verified reference corpus

23 sources, 23 verified (Crossref DOI lookup, Crossref title search after truncated-title
`DOI_MISMATCH`, or arXiv identifier). Semantic Scholar unavailable throughout
(`[S2-API-UNAVAILABLE]`, HTTP 429 on all unauthenticated requests) — verification rested on two of
three planned indexes, which is disclosed in the final report §6.4.

Full APA 7.0 reference list: `phase6_final/final_report.md` §References.
Machine-readable trail: `phase2_investigation/reference_verification.json`.

---

## 6. Known weaknesses the manuscript must carry forward

These are not defects to fix at intake; they are disclosed limitations that any downstream draft
must preserve rather than quietly drop.

1. The interpretable representation (time windows, frequency bands) is the one audioLIME
   explicitly critiques as non-listenable. Source separation, the recommended remedy, does not
   apply to single-speaker speech.
2. A null on SQ2 supports a **disjunction** (container irrelevant *or* early decode), and the
   practical licence it grants must not be reported as a mechanistic conclusion.
3. The novelty claim is absence of evidence from a targeted, non-exhaustive search without
   Semantic Scholar or paywalled full-text indexes. "We found no prior work doing this" — never
   "no prior work exists".
4. The XAI subset is outcome-selected on label agreement; it biases SQ3 conservatively and
   supports no accuracy inference.
5. Single model, single corpus, single bitrate, acted emotion, English, clean, single-speaker.
6. Perturbation attribution shows what the output *depends on*, never what the model *attends to*.
