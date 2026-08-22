# Devil's Advocate Report — Checkpoint 2
**Agent:** `devils_advocate_agent` | **Phase:** 3 (post-analysis)
**Reviews:** `synthesis.md` + the 19-source evidence base (23 after this checkpoint's remedy)

---

## Verdict: **REVISE**

One Critical issue, three Major. All are correctable; two required changes to the evidence base
itself, which have been made and are recorded below.

**Steel-manning first.** The synthesis does something most syntheses avoid: it derives a
*falsifiable prediction of its own null result* from published evidence, and states it before data
collection. Resolving the T1/T2 tension into "the format destroys information the model was not
using" is a genuine analytic move, not a summary. And it names the single most likely outcome of
the whole study in plain language rather than leaving the reader to infer it.

---

## Critical Issues (Blocks Progression)

### C2. If the model degenerates to a constant response, the attribution measure is not weak — it is undefined

- **Type:** Method / measurement collapse
- **Location:** `synthesis.md` §4 gaps G1 and G5, which treat the floor effect as a *power* problem;
  `methodology_blueprint.md` Steps 4–5.
- **Problem:** The synthesis correctly identifies that the model may collapse to *neutral* on a
  neutral-lexicon corpus. It then treats the consequence as reduced sensitivity. That
  under-describes the failure. The attribution score is defined as
  `a_j = 1 − (# repeats returning the unmasked modal label) / R`. If the model returns *neutral*
  regardless of what is masked, then every `a_j = 0`, and the attribution vector is **constant**.
  Spearman correlation between two constant vectors is **undefined**, not merely low. Top-k
  Jaccard overlap on an all-zero vector is arbitrary. `S_within` and `S_across` both cease to
  exist, and SQ3 — the headline question, the one the user explicitly chose — does not return a
  weak answer. It returns no answer at all, after ~9,000 API calls.
- **Impact:** A degenerate-response model silently destroys the primary research question while
  producing a full results table of zeros that looks like data. This is worse than a null: it is a
  null that can be mistaken for a finding ("explanations were perfectly stable across formats!").
- **Recommendation:**
  1. Add a **response-degeneracy check as gate G0b**, run on the G0 sample before any XAI calls:
     compute the entropy of the model's label distribution across items and across masks. If the
     modal label is constant for more than 30% of items across all 16 masks, the flip-rate
     attribution measure is declared inapplicable and must be replaced before proceeding.
  2. Pre-specify the replacement now, not later. The natural one is a **forced binary contrast**:
     instead of the 6-way lexicon, ask the model to choose between two labels (the gold label and a
     confusable alternative). This guarantees a non-degenerate response distribution and makes flip
     rate meaningful. It narrows the task but preserves the manipulation.
  3. Add an explicit guard in the analysis code: any Spearman computed on a constant vector must
     raise, not return `nan` silently into a results table.

---

## Major Issues

### M6. The synthesis rests its central pivot on a single source — and the counter-evidence was mentioned without a citation

- **Type:** Evidence sufficiency / **IRON RULE breach**
- **Location:** `synthesis.md` §2 Contradiction 1, and §3 convergence table, row "Audio LLMs rely
  more on lexical than acoustic cues", diverging column: *"Reported SER accuracies for frontier
  models on CREMA-D are non-trivial"*.
- **Problem:** Two distinct faults. First, the entire T2 theme — which reshapes the study's
  predicted outcome, motivates a possible task pivot, and is called "the most consequential
  finding" — rests on **one benchmark paper** (Chen et al., 2025) evaluating six models, and there
  is no confirmation that the specific model under study was among them. That is availability bias
  dressed as synthesis. Second, and more seriously, the one piece of counter-evidence offered in
  the divergence table **carries no citation**. Every claim must have a verified citation; a
  hedging clause inserted to look balanced, with nothing behind it, is worse than omitting the row.
- **Recommendation:** Locate and verify actual counter-evidence, or delete the claim.
- **Status: REMEDIED.** Four additional sources were searched, verified via the arXiv API, and
  added to the bibliography as Theme 6 — VoxEmo (Zhang et al., 2026), EmoBench-M (Hu et al., 2025),
  AHELM (Lee et al., 2025), and the LALM trustworthiness survey (Luo et al., 2026). VoxEmo
  materially qualifies T2: speech LLMs are benchmarked on SER across 35 corpora and 15 languages,
  which is not the profile of a model class at chance. T2 must be downgraded from "the model
  probably isn't listening" to "**there is credible evidence of substantial lexical reliance, and
  credible evidence of real SER capability; which dominates for this model on this corpus is an
  open empirical question that gate G0 will answer.**" The synthesis's derived prediction survives
  as a *prediction*, not as an established expectation.

### M7. The synthesis vindicates the framing the user had just chosen — and does not notice

- **Type:** Confirmation bias / motivated reasoning
- **Location:** `synthesis.md` §2, "It also vindicates the explanation-first framing."
- **Problem:** The user selected explanation-first framing at the Phase 1 checkpoint. The Phase 3
  synthesis then derives, from the literature, that accuracy will be flat — and concludes that this
  makes the user's choice correct. The reasoning chain may well be sound, but the *order of
  events* is exactly the shape motivated reasoning takes, and the text presents the convergence as
  a satisfying confirmation rather than examining it. A hostile reviewer will notice that the
  analysis rewarded the client's preference.
- **Recommendation:** Keep the argument, drop the framing of it as vindication. State the
  contingency plainly and symmetrically: *if* SQ1 is flat, explanation-first is the only framing
  with a question left; *if* SQ1 shows a real effect, accuracy-first would have been the stronger
  choice and the report should say so. The prediction must be able to embarrass the framing, or it
  is not a prediction.

### M8. The most likely finding is a first measurement of an unknown quantity — and it is filed as a gap, not a contribution

- **Type:** Missed inference / significance
- **Location:** `synthesis.md` §4, gap G2.
- **Problem:** G2 notes that no published value exists for the run-to-run stability of an audio
  attribution map. It treats this as a problem (SQ3 cannot be powered in advance). But an
  unmeasured quantity that the design measures **as a mandatory control** is a contribution that
  is guaranteed to exist regardless of how every other question turns out. The study currently
  frames its one certain deliverable as an inconvenience.
- **Recommendation:** Promote `S_within` from control to reported result. Given that the format
  effect may be null and the container effect inconclusive, the run-to-run stability of black-box
  attribution on a commercial audio LLM may be the most durable thing this study produces —
  and it is directly useful to anyone else attempting black-box XAI on a stochastic API.

---

## Minor Issues

- The T1/T2 resolution assumes 64 kbps AAC preserves intelligibility. Plausible, but asserted
  rather than measured. The WER positive control (M3) tests it — so the prediction must be stated
  as *conditional on* that control passing, not alongside it.
- "The single most likely outcome" in §5 is a point prediction with no uncertainty attached.
  Attach rough probabilities, or state it as a ranked ordering of scenarios. As written it invites
  the reader to treat a judgement call as a forecast.
- Cao et al.'s 40.9% human audio-only figure is used to recalibrate gate G0, correctly. But human
  and model are not commensurable here: the humans were rating *intended* emotion across the full
  7,442-clip corpus, while the model faces a 50-item stratified subset under a forced-choice
  prompt. The comparison is indicative, not a like-for-like benchmark, and should be labelled as
  such wherever it appears.
- Theme 5's claim that "no source reports the container" is an absence claim over a non-exhaustive
  reading. It is correctly hedged in `search_strategy.md` and should inherit the same hedge here.

---

## Observations

- If G0b fires and the task pivots to a forced binary contrast, the study gets *better*, not
  worse: a binary forced choice removes the lexical shortcut, guarantees a measurable response
  distribution, and sharpens the attribution signal. Worth pre-framing as the planned design rather
  than as a rescue.
- VoxEmo's finding on zero-shot stochasticity means the repeat structure and prompt-sensitivity arm
  are not merely good practice — they address a documented, named failure mode. The Methods should
  cite it there rather than leaving those controls looking like generic caution.
- Nobody in this corpus reports the container. That fact is itself a citable finding for the
  Discussion — a reporting-practice gap evidenced by a systematic absence across 23 sources, which
  is a stronger form of the argument than "we found a gap".

---

## Cherry-Picking and Bias Audit

| Check | Finding |
|---|---|
| Were counter-examples sought? | **Initially no.** Searches were framed to support the emerging story; counter-evidence was added only after this checkpoint forced it (M6). |
| Are contradictions resolved or explained away? | Contradiction 1 is genuinely resolved (scope difference, testable). Contradiction 2 is explicitly *not* resolved and is disclosed — correct handling. Contradiction 3 is left as a live tension requiring a decision — also correct. |
| Would different inclusion criteria change the picture? | **Yes.** Including the paywalled SER-robustness literature and non-English work could shift T1's strength, and Semantic Scholar's absence (429 throughout) means one of three planned indexes never ran. Disclosed in `search_strategy.md`. |
| Is any theme carried by a single source? | **T2 was**, and this was the checkpoint's most consequential catch. Now carried by Chen et al. plus VoxEmo in tension. |
| Does the evidence favour the client's preference? | **It appeared to, and that was the problem** (M7). After remediation the evidence is genuinely split. |

---

## Strongest Counter-Argument

> "You have assembled a literature review that predicts your own experiment will find nothing,
> identified that your model may not perform the task at all, discovered your pilot data is 92%
> API errors, and concluded that the study should proceed — with the justification that the
> control condition you were forced to add will produce a number nobody has published before.
> A reviewer will ask why this is a study rather than a methods note."

**The response, and it must be in the paper rather than in a rebuttal letter:** because the gates
are real. G0 can stop the study before the main grid. G0b can stop it before the XAI arm. If both
pass, the design answers a question nobody has asked with a control nobody has run. If either
fails, the honest output *is* a methods note — the protocol, the stability floor, and a documented
account of why a lexically dominant model on a fixed-lexicon corpus cannot support a format
experiment. That is a smaller contribution than hoped and a real one, and pre-committing to it now
is what distinguishes this from a study that will quietly redefine success after seeing its data.

---

## What's Missing

- Any estimate of how often `gemini-flash-latest` was itself included in the benchmarks the
  synthesis relies on. If it was not evaluated in either Chen et al. or VoxEmo, both are weaker
  guides than the synthesis implies.
- The paywalled and non-English SER-robustness literature, entirely absent from this corpus.
- Semantic Scholar as a third verification index (429 throughout the session).
- A registered-report route. Given that the study predicts its own null, in-principle acceptance
  before data collection is the natural venue strategy and is not mentioned anywhere.

---

## Stress Test Results

| Test | Result |
|------|--------|
| Remove the strongest source — does the argument hold? | **Partly.** Removing Chen et al. (2025) removes the predicted-null mechanism, and the study reverts to an open empirical question — weaker motivation, intact design. |
| Flip the research question — is the opposing view credible? | **Yes.** "Format is irrelevant to a model that decodes to PCM immediately and reads the words" is highly credible and is now the modal prediction. |
| Apply to a different context — does the finding generalise? | **No**, and the corpus makes this sharper than Checkpoint 1 did: VoxEmo's 35 corpora across 15 languages is the scale at which SER claims generalise. This study is 50 clips, one model, one bitrate. |
| "So what?" — is the significance justified? | **Conditionally.** Justified via the stability-floor contribution (M8) and the reporting-practice gap even under total nulls. Not justified if the report leads with the format effect and buries the controls. |

---

## Resolution Log

`[DA-DECISION: C2 raised | ACTION: Escalate to Critical | REASON: degenerate response collapses the
attribution vector to a constant, making the primary analysis mathematically undefined rather than
merely underpowered — and producing an all-zero results table that reads as a positive finding.]`

`[DA-DECISION: M6 raised | ACTION: Escalate | REASON: uncited counter-claim is an IRON RULE breach,
and single-source dependency on the theme that reshapes the entire study.]`

**Synthesis-agent response (recorded 2026-08-17):** C2 accepted; gate G0b, the pre-specified
forced-binary replacement task, and the constant-vector guard are added to the blueprint. M6
accepted and remedied — four counter-evidence sources located, verified via arXiv, and added as
Theme 6; T2 downgraded to an open empirical question. M7 accepted; the "vindicates the framing"
sentence is replaced with a symmetric conditional. M8 accepted; `S_within` is promoted to a
reported result.

`[DA-DECISION: Response scored 5/5 on C2 and M6 | ACTION: Concede | REASON: both were remedied by
changing the artefacts — a new gate with a pre-specified replacement measure, and four new verified
sources that genuinely cut against the synthesis's own thesis — rather than by argument.]`

`[DA-CONCESSION-RATE: 2 of 4 findings conceded this checkpoint. Below the 50% pause threshold. Bar
for any further concession in this checkpoint rises to 5/5 per the anti-sycophancy rule.]`

**Post-revision verdict: PASS.** Cleared to proceed to Phase 4.
