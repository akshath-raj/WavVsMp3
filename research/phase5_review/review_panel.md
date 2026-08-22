# Phase 5 — Review Panel
**Agents:** `editor_in_chief_agent`, `ethics_review_agent`, `devils_advocate_agent` (Checkpoint 3)
**Reviews:** `phase4_composition/report_draft.md`
**Date:** 2026-08-17

---

# Part I — Editorial Review

**Agent:** `editor_in_chief_agent` (Q1 venue standard)

## Verdict: **MINOR REVISION**

The manuscript is a research design and feasibility report, and it should be judged as one. On that
standard it is unusually solid: the design contains an identifying contrast the literature has not
attempted, the analysis plan is fixed in advance including its null-result phrasing, and the
preparatory findings are reported against the authors' own interest rather than around it. What
stands between this and acceptance is positioning and a handful of overstatements, not substance.

### Assessment by criterion

| Criterion | Rating | Note |
|---|---|---|
| Originality | **Good** | The container/codec decomposition is genuinely unclaimed. Correctly framed as transfer-plus-new-contrast rather than invention, with Ghorbani et al. credited prominently. |
| Methodological rigour | **Strong** | Common-ancestry stimulus construction, condition concealment, randomised call order, equivalence testing for the expected null, and — the standout — a within-format stability floor before any across-format claim. Most black-box XAI work omits the last of these. |
| Evidence sufficiency | **Adequate with a caveat** | 23 verified sources, appropriate for a design report. One of three planned verification indexes never ran. Disclosed. |
| Argument coherence | **Strong** | §2.6's derived prediction is the manuscript's best passage: it resolves an apparent contradiction into a falsifiable claim that cuts against the study's own interest. |
| Writing quality | **Good** | Clear, unhedged where it should be, hedged where it must be. Some repetition across §5.1 and §7. |
| Reproducibility | **Strong** | Configs, gates, prompts, budgets, and a machine-readable verification trail. REFORMS adopted. |

### Required revisions

**E1 (Required). The title promises more than the manuscript delivers.**
"Does the Container Change the Evidence?" reads as a results paper. No result on that question is
reported. Retitle to signal the genre — e.g. *"Separating Container from Codec: A Black-Box XAI
Protocol and Feasibility Analysis for Audio Format Robustness in a Multimodal LLM."* Readers who
arrive expecting findings and leave with a protocol will score the paper on what it did not claim
to be.

**E2 (Required). The abstract buries the genre until its fourth sentence.**
State in sentence one that this is a protocol and feasibility report. The current opening is a
motivating claim, which is good rhetoric and poor signposting.

**E3 (Required). §4.4's two-weak-signals argument needs its own weakness stated in the same
paragraph, not two sections later.**
The 0/8 pilot result is drawn from 5 items in 2 conditions with 92% call failure around it. §4.4
says n = 8 is too small, then proceeds to reason from it. The reasoning is legitimate — converging
weak evidence justifies a gate, not a conclusion — but the sentence "two independent weak signals
pointing the same direction" should explicitly note that one of those signals is 8 observations
from a run that mostly failed.

**E4 (Recommended). §5.1 and §7 substantially restate each other.**
Three findings appear in both. Compress §7 to recommendations and let §5.1 carry the interpretation.

**E5 (Recommended). Attach uncertainty to the modal-outcome forecast.**
The synthesis names a single most likely outcome. Either give a ranked scenario ordering with rough
probabilities, or mark it explicitly as a judgement call. As written a reader may take it for a
calculated forecast.

**E6 (Recommended). The human-parity comparison recurs without its caveat.**
Cao et al.'s 40.9% is correctly caveated at §3.2 and then used bare at §7. Repeat the caveat at
each use or drop the second one.

### Not required, but the strongest available improvement

Submit as a **registered report**. A study that predicts its own null, pre-declares its
interpretation grid, and pre-authorises its failure outcomes is the paradigm case for
in-principle acceptance. §7 mentions this once. It should be the stated venue strategy.

---

# Part II — Research Integrity Review

**Agent:** `ethics_review_agent`

## Integrity verdict: **CLEARED**

No fabrication, no plagiarism, no source misrepresentation, and AI disclosure is present and
specific. Human-subjects status is reported administratively without asserting a determination.

### Checks

| Check | Result |
|---|---|
| Citation fabrication | **Pass.** 23/23 verified against Crossref or arXiv; audit trail retained. Two `DOI_MISMATCH` events disclosed in-text rather than silently resolved. |
| Vibe citing / mashup references | **Pass.** No reference assembled from partial recall; all metadata pulled from index records. |
| "Difficult to verify" treated as acceptable | **Pass.** Zero sources in the gray zone. The IRON RULE held. |
| Source misrepresentation | **Pass.** Chen et al. (2025) is the load-bearing source and is characterised accurately, including the qualification that its model coverage may not include the target model. Counter-evidence (Zhang et al., 2026) is presented as genuinely cutting against the manuscript's own thesis. |
| Attribution integrity | **Pass.** §1.4 explicitly separates inherited, transferred, and new contributions — unusual and correct. |
| AI disclosure | **Pass.** Specific about which steps were AI-performed and which decision was human. |
| Data integrity | **Pass.** §4.1's failure statistics are reproducible from the project's own result store; reporting a 92% failure rate against interest is the opposite of selective reporting. |
| Human subjects | **Reported, not determined.** Public consented corpus; no new collection. Authorization status: `not_provided`. Review pathway: institutional determination required. The manuscript does not claim exemption. |

### Dual-use screening — advisory, not blocking

Speech emotion recognition carries surveillance and workplace-monitoring applications. The
manuscript already declines to validate emotion inference as a construct and states that acted
emotion is a controlled acoustic manipulation rather than evidence about emotion in the wild. That
is the correct posture.

**Recommendation (advisory):** add a short **Responsible Use Statement** near §6.5 making the
boundary explicit rather than distributed across caveats — that the work characterises a model's
sensitivity to file format and provides no evidence that automatic emotion recognition is accurate,
fair, or appropriate for deployment in consequential settings. Subject matter alone never blocks;
this is a clarity improvement, not a condition.

### Operational security note

Both API keys were transmitted in plaintext during project setup. §3.8 and §7 both flag rotation.
Retained here so it is not lost when the report is excerpted.

---

# Part III — Devil's Advocate, Checkpoint 3

**Agent:** `devils_advocate_agent` | Final vulnerability scan

## Verdict: **PASS**

No Critical issues. Two Major, three Minor. The Critical findings from Checkpoints 1 and 2 were
remedied by changing the design, not by re-describing it, and the remedies survive inspection.

### Major Issues

**M9. The manuscript's genre is load-bearing and under-defended.**
A design-and-feasibility report is the right genre for this work, but the manuscript never argues
for the genre — it simply occupies it. A hostile reviewer will read §4 and conclude that the study
failed and was written up as a protocol to salvage the effort. The rebuttal exists in the material
(the gates were designed before the audit; the equivalence problem was found by review, not by
data; the container contrast is unclaimed regardless of outcome) but it is never assembled in one
place. **Recommendation:** one paragraph in §1.2 stating why the protocol is the deliverable —
that the design's decisions are exactly the ones that must be fixed before data, and that a study
predicting its own null forfeits its credibility if it publishes those decisions afterwards.

**M10. The container-effect null remains a disjunction, and §5.2 lets the reader forget it.**
§6.3 correctly states that a null on SQ2 is consistent with both "container is irrelevant" and
"ingestion decodes to PCM immediately". But §5.2 says a passed equivalence test "would license
format-agnostic submission for this model at this operating point" without restating the
disjunction. That inference is actually fine for the *practical* claim — if the decode is early,
format-agnostic submission is safe for that reason — but the *mechanistic* claim does not follow
and the two sit one sentence apart. **Recommendation:** separate them explicitly: the practical
licence holds under either disjunct; the mechanistic conclusion holds under neither without the
metadata probe.

### Minor Issues

- §3.5 says R was "raised from 3 to 7 after adversarial review" without saying that ties in a
  4-level measure would bias SQ3 *toward the null* — the direction is what makes the change
  necessary rather than merely nicer, and it is one clause.
- The word "explanation" carries the report. §6.2 holds the line ("depends on" vs. "attends to"),
  but §5.2's phrase "what a multimodal LLM's decision rests on" edges back toward mentalism. Prefer
  "what its output depends on".
- Appendix B lists Phase 5 and 6 artefacts as though they exist at time of writing. Mark as
  forthcoming or generate before circulation.

### Stress Test Results

| Test | Result |
|------|--------|
| Remove the strongest source — does the argument hold? | **Yes.** Removing Chen et al. (2025) removes the predicted mechanism for the null; the design, the gaps, and the gates stand unchanged. |
| Flip the research question — is the opposing view credible? | **Yes**, and the manuscript says so. "Format is irrelevant because ingestion decodes early and the model reads the words" is the modal prediction, stated as such. |
| Would a hostile reviewer find fatal flaws? | **No fatal flaw.** The available attacks — small n, single model, contested attribution representation, a pilot that mostly failed — are all disclosed by the manuscript itself, usually before the reviewer would reach them. |
| Is the "so what?" answered? | **Yes, at two levels.** The reporting-practice recommendation (§5.4) holds regardless of any result. `S_within` is a guaranteed contribution. Both survive total nulls on SQ1–SQ3. |
| Are limitations genuine or performative? | **Genuine.** §6.2 concedes that the core method is the one the audio-XAI literature criticises, and does not claim the mitigations solve it. §6.4 concedes the novelty claim is absence-of-evidence. Neither concession is decorative. |

### Strongest Counter-Argument

> "This is a protocol for an experiment that its own authors expect to return nothing, on a model
> they have not established can do the task, using an attribution representation that the audio-XAI
> literature specifically warns against, validated against a pilot in which 92% of calls failed."

**Response, which the manuscript makes and should keep making:** every clause is true and every
clause is in the manuscript. The expected null has a stated mechanism and a falsification condition.
The model's capability is the first gate, not an assumption. The attribution representation is used
because the recommended alternative is inapplicable to single-speaker speech, with the artefact
measured rather than assumed. The pilot failure is a quota fact with a US$3.70 remedy. A design
whose weaknesses are all pre-declared and gated is not the same object as a design whose weaknesses
are discovered by reviewers.

---

# Consolidated revision list for Phase 6

| ID | Source | Severity | Action |
|---|---|---|---|
| E1 | Editor | Required | Retitle to signal protocol/feasibility genre |
| E2 | Editor | Required | Abstract states genre in sentence one |
| E3 | Editor | Required | §4.4 states the pilot signal's weakness in-paragraph |
| E4 | Editor | Recommended | Compress §7 overlap with §5.1 |
| E5 | Editor | Recommended | Attach uncertainty to modal-outcome forecast |
| E6 | Editor | Recommended | Repeat or drop the human-parity caveat at §7 |
| ETH1 | Ethics | Advisory | Add Responsible Use Statement |
| M9 | DA CP3 | Major | Defend the genre explicitly in §1.2 |
| M10 | DA CP3 | Major | Separate practical licence from mechanistic conclusion in §5.2 |
| m1–m3 | DA CP3 | Minor | Tie-direction clause; "depends on" wording; mark appendix artefacts forthcoming |

**Revision loop 1 of a maximum 2.** Anything unresolved after loop 2 moves to Acknowledged
Limitations.
