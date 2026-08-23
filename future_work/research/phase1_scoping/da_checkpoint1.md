# Devil's Advocate Report — Checkpoint 1
**Agent:** `devils_advocate_agent` | **Phase:** 1 (post-scoping)
**Reviews:** `rq_brief.md` + `methodology_blueprint.md`

---

## Verdict: **REVISE**

One Critical issue blocks progression to Phase 2. It is correctable inside Phase 1 and does not
require abandoning the design. Resolution is recorded at the end of this document.

**Acknowledged strengths first (steel-manning).** The `mp4_aac64` / `roundtrip_wav` contrast is a
genuinely good idea: it is the rare case where a "container effect" can be isolated without model
internals, because the decoded signal is held constant by construction. The mandatory
within-format stability floor (V7) is the single best decision in the blueprint — most black-box
XAI work omits it and thereby reports noise as insight. The four-cell interpretation grid declared
in advance is real preregistration discipline, not decoration.

---

## Critical Issues (Blocks Progression)

### C1. The equivalence bound is unreachable at n = 50, which makes SQ2's expected result unclaimable

- **Type:** Method / statistical power
- **Location:** Methodology Blueprint, Analytical Framework Step 3 ("pre-declared smallest effect
  size of interest Δ = 3 accuracy points"); RQ Brief SQ2.
- **Problem:** The blueprint correctly recognises that a null container effect requires an
  equivalence test rather than a non-significant NHST — and then sets a bound that the design
  cannot support. For a paired binary outcome with the item as the unit of analysis (n = 50), the
  minimum detectable effect at α = .05, power = .80 is on the order of **15–18 percentage points**
  under plausible discordance (p₀₁+p₁₀ ≈ 0.15–0.25). Repeats shrink measurement noise but do not
  raise the effective number of independent units, because item-level random variance dominates.
  A Δ = 3 equivalence test at n = 50 would require both TOST bounds to be cleared by an interval
  far narrower than the design's own resolution: it will essentially never pass. The predictable
  outcome is that SQ2 returns "inconclusive" *by arithmetic rather than by evidence*, and the
  study's cleanest contrast produces nothing.
- **Impact:** SQ2 — the mechanism question, and the contrast the novelty claim rests on — becomes
  undecidable. Worse, an inconclusive result would be easy to narrate as "no container effect
  found", which is precisely the inference the equivalence test was introduced to prevent.
- **Recommendation:** Before the full grid runs, (a) compute the actual minimum detectable effect
  and the equivalence bound the design *can* support, from the smoke-run discordance rate;
  (b) reset Δ to that honest value and state it as a limitation rather than a target;
  (c) if a tight bound is genuinely wanted, the only real fixes are more items or more repeats —
  cost is not the constraint here, wall-clock is, so state which was chosen and why;
  (d) declare in advance that "inconclusive" is a reportable SQ2 outcome and pre-write how it will
  be phrased, so the write-up cannot drift into a null claim.

---

## Major Issues

### M1. AAC encoder delay breaks the "signal identity" premise unless explicitly handled

- **Type:** Method / hidden confound
- **Location:** Blueprint V1, assumption A2.
- **Problem:** AAC encoding inserts priming/delay samples (commonly ~1024 or 2112 samples, encoder
  dependent) and pads the final frame. A decoded MP4 is therefore typically **time-shifted and
  length-changed** relative to its input. `roundtrip_wav` inherits that shift; `ref` does not. So
  the `roundtrip_wav` − `ref` contrast silently mixes coding loss with a small onset offset and a
  duration change, and a naive sample-wise comparison in V1 will report large deviation for a
  reason that has nothing to do with codec fidelity.
- **Recommendation:** In the Step 1 fidelity gate, align by cross-correlation before computing any
  deviation metric, and **report the recovered delay and duration difference per item as data**.
  If the shift is non-trivial, add a delay-compensated variant of `ref` so the codec contrast is
  clean. Do not compute SNR on unaligned signals.

### M2. Attribution scores at R = 3 are a four-level measure; the SQ3 test may be tie-dominated

- **Type:** Method / measurement resolution
- **Location:** Blueprint Step 4 ("label-flip rate", R = 3) and Step 5 (Spearman ρ).
- **Problem:** With R = 3, each of the 16 attribution entries takes one of {0, ⅓, ⅔, 1}. Spearman
  correlation over a 16-element vector with that many ties is a low-information statistic, and
  both `S_within` and `S_across` will be compressed toward each other — biasing SQ3 toward a null
  regardless of what is true. The study would then report "explanations are stable" when it has
  merely failed to measure them.
- **Recommendation:** Raise R for the XAI arm specifically (R = 7 gives eight levels and costs
  roughly $1 more), or replace flip-rate with a graded score. Additionally, report tie fractions
  and use a tie-corrected similarity; pre-register a minimum acceptable `S_within` below which the
  attribution measurement is declared too noisy to interpret at all.

### M3. WER on a 12-sentence fixed lexicon will likely hit a ceiling and detect nothing

- **Type:** Method / measurement sensitivity
- **Location:** RQ Brief SQ1 (transcription as secondary task); Blueprint Step 2.
- **Problem:** CREMA-D's fixed lexicon is short, clean, and highly predictable. A current
  multimodal LLM will plausibly transcribe near-perfectly in *every* condition, including 64 kbps
  AAC — 64 kbps is far above the rate at which intelligibility degrades. The transcription arm
  then contributes 600 calls and zero discriminative information.
- **Recommendation:** Keep the arm (it is a useful demonstration that the pipeline works and that
  format effects are task-dependent), but reframe it in advance as a **positive control for
  intelligibility preservation**, not as a second test of the hypothesis. If WER is at floor
  everywhere, that is the finding: the format effect is specific to paralinguistic content.
  State this before the run, not after.

### M4. `mp3_64` cannot support any mechanistic claim, and its presence invites one

- **Type:** Design asymmetry
- **Location:** RQ Brief scope; Blueprint contrast table.
- **Problem:** The blueprint concedes MP3 has no round-trip control, then still carries it through
  the grid. Any MP3-vs-MP4 difference will be read mechanistically by readers even if the text
  disclaims it, because the reader sees two containers side by side.
- **Recommendation:** Generate `roundtrip_wav_mp3` as well. It is one ffmpeg invocation and 150
  extra calls, and it makes the decomposition symmetric. There is no good reason not to.

### M5. A null container effect is confounded with the assumption that created it

- **Type:** Unfalsifiable premise
- **Location:** Assumption A1 ("opaque decode").
- **Problem:** If `mp4_aac64` ≈ `roundtrip_wav`, two very different worlds are consistent with the
  data: (i) the container genuinely does not matter, or (ii) the ingestion path decodes to PCM
  immediately, making the containers identical downstream — the exact property for which Phi-4 was
  *rejected*. The design cannot distinguish them, yet the model was selected on the belief that it
  has an opaque path.
- **Recommendation:** State plainly that a null result on SQ2 is evidence for the disjunction, not
  for (i). Optionally add a cheap discriminating probe: submit a container whose payload is
  intact but whose metadata is unusual (e.g. altered container-level duration or sample-rate
  fields) and see whether behaviour changes at all. Any sensitivity to non-audio bytes would
  establish that container parsing is behaviourally live.

---

## Minor Issues

- The XAI subset is selected on label agreement, which biases toward *stable* items and therefore
  biases SQ3 **conservatively** (against finding divergence). This is the right direction to err,
  but it must be stated explicitly, or a reviewer will read outcome-dependent selection as a
  fishing expedition.
- "Explanation" is used throughout for what is strictly output-dependence under occlusion. The
  blueprint warns against sliding from "depends on" to "attends to" — the report must actually
  hold that line in its section headings and figure captions, which is where such slides happen.
- Randomised call order (V4) helps with drift but creates a within-item ordering nuisance if the
  API caches or conditions on recent traffic. Record call sequence position and check it as a
  covariate.
- FINER Novelty is scored 4/5 before the literature search has run. That score is a hypothesis.
  Phase 2 is explicitly tasked with falsifying it (handoff item (e)), and the score must be
  revised down if prior work exists.
- Cost is reported as trivial (~$2) while wall-clock is the real constraint, yet no wall-clock
  budget or stopping rule is stated. Add one.

---

## Observations

- The strongest version of this study may not be the emotion task at all. Paralinguistic tasks
  where 64 kbps coding actually removes information — speaker verification, prosodic stress,
  breathiness or voice-quality judgements — would give the manipulation more to bite on. Worth
  naming as an extension.
- If the dissociation cell (unchanged accuracy, diverged explanation) does appear, the natural
  follow-up is a bitrate ladder to find where explanation shift begins relative to accuracy loss.
  Explanation-level robustness may degrade earlier than accuracy — that would be the paper.
- The four-cell grid deserves to be in the final report as a figure. It is the clearest statement
  of what the study can conclude.

---

## Strongest Counter-Argument

> "You have shown that a stochastic commercial API returns somewhat different answers when given
> somewhat different bytes. With no access to weights, activations, or log-probabilities, you
> cannot separate an ingestion-path difference from ordinary sampling variance layered on real
> signal degradation — and your 'explanation' is a 16-cell occlusion map estimated from three
> samples per cell on twenty clips of acted emotion. The container contrast, your one clean idea,
> will almost certainly return a null that your own sample size cannot convert into a claim."

The design's answer must be structural, not rhetorical: the within-format stability floor
(V7) exists precisely to absorb the sampling-variance objection, and the honest response to the
power objection is C1's fix — report the bound the design can actually support and let the null be
inconclusive if that is what it is.

---

## What's Missing

- A stated wall-clock budget and a stopping rule for the free-tier run.
- Any human baseline or reference model on the same 50 items. Without one there is no anchor for
  whether the model's `ref` accuracy is good, and therefore no way to know if there is headroom to
  lose.
- A pilot-based power calculation. The smoke run (180 calls) already contains the discordance
  information needed for it and has not been used.
- Consideration of whether Google-side server behaviour (transcoding, caching, request routing)
  varies by MIME type independently of the model — an infrastructure confound sitting upstream of
  the model entirely.

---

## Stress Test Results

| Test | Result |
|------|--------|
| Remove the strongest source — does the argument hold? | **Yes.** Removing the container contrast (SQ2) leaves SQ1 + SQ3, which still constitute a coherent study. |
| Flip the research question — is the opposing view credible? | **Yes.** "Format is irrelevant because every ingestion path decodes to PCM early" is a highly credible prior, and the design must be able to report it. |
| Apply to a different context — does the finding generalise? | **No.** Single model, single corpus, single bitrate, acted emotion. Generalisation is explicitly disclaimed; the reusable output is the protocol. |
| "So what?" — is the significance justified? | **Partially.** Justified if the dissociation cell appears or if SQ2 yields an interpretable bound. Not justified if the study reduces to "64 kbps AAC costs a few accuracy points", which is already known for task-specific models. |

---

## Resolution Log

`[DA-DECISION: C1 raised | ACTION: Escalate to Critical | REASON: pre-declared equivalence bound is
arithmetically unreachable at n=50, converting the design's cleanest contrast into a guaranteed
non-result.]`

**Architect response (recorded 2026-08-17):** C1 accepted in full. The blueprint's Step 3 is
amended: Δ is no longer fixed at 3 points a priori; it is to be computed from the smoke-run
discordance rate as the smallest bound the design can support at 80% power, declared before the
full grid runs, and reported alongside the result as a design limitation. "Inconclusive" is added
as an explicit, pre-authorised SQ2 outcome with pre-written phrasing. M1–M5 accepted: cross-
correlation alignment added to the fidelity gate, XAI repeats raised to R = 7, the transcription
arm reframed as a positive control, `roundtrip_wav_mp3` added, and the A1 disjunction stated in
both the blueprint and the eventual discussion.

`[DA-DECISION: Architect response scored 5/5 | ACTION: Concede | REASON: response addresses the
core attack directly — it does not defend the unreachable bound, it replaces the procedure that
produced it and pre-commits to reporting the weakened claim.]`

**Post-revision verdict: PASS.** Cleared to proceed to Phase 2.
