# Synthesis & Gap Analysis
**Agent:** `synthesis_agent` | **Phase:** 3 (Analysis)
**Consumes:** `rq_brief.md`, `methodology_blueprint.md`, `feasibility_addendum.md`,
`annotated_bibliography.md` (19 verified sources)

---

## 1. Thematic synthesis

### T1 — Lossy coding removes paralinguistic information, and signal-level fidelity does not predict how much

Two independent lines converge. Classical SER work finds that standard speech codecs reduce
emotion-recognition accuracy, with the magnitude varying by codec and by acoustic feature family
(Reddy & Vijayarajan, 2020). Modern neural-codec benchmarking reaches the same place from the
other direction: Codec-SUPERB reports scenarios in which **signal-level metrics fail to capture
losses in semantic, linguistic, or paralinguistic content**, which is why it evaluates codecs
through downstream tasks rather than through reconstruction error alone (Wu et al., 2024).

The methodological consequence is concrete and shapes this study's Step 1: measuring SNR, log
spectral distance, or PESQ between `ref` and `mp4_aac64` tells you the signal changed; it does not
tell you whether anything the model uses was destroyed. Fidelity metrics and behavioural metrics
must both be reported, and neither may stand in for the other.

### T2 — Audio LLMs may not be listening in the sense the codec literature assumes

The strongest and most disruptive finding in the corpus. Chen et al. (2025) evaluate six
state-of-the-art large audio-language models on a benchmark built specifically to separate lexical
from acoustic contributions, and find consistent **lexical dominance**: models default to *neutral*
when lexical cues are neutral or absent, gain little when lexical and acoustic cues align, fail
under cue conflict, and approach chance in paralinguistic contexts. Their summary — these systems
"largely transcribe rather than listen" — describes a model class whose emotion judgements rest
substantially on transcribed words rather than on prosody.

This is not a peripheral caveat. It changes what the present study is most likely to find, and it
does so *before* any data is collected.

### T3 — Explanations can move while predictions stand still, and attribution can be vacuous

Ghorbani et al. (2019) establish, for image classifiers, that perceptually indistinguishable
inputs assigned the *same* label — sometimes with higher confidence — can receive substantially
different interpretations. This is the accuracy/explanation dissociation that SQ3 proposes to look
for, already demonstrated in another modality. Adebayo et al. (2018) supply the necessary
discipline: some widely used saliency methods are independent of both the model and the data
generating process, producing convincing-looking maps that explain nothing, and visual inspection
cannot distinguish these from real ones.

Together these justify the design's most important control. An attribution map must first be shown
**reproducible under no manipulation** before any change in it under manipulation can be
interpreted. That is V7, the within-format stability floor, and it is not optional.

### T4 — Black-box attribution for audio is feasible but its validity is actively contested

LIME established that a model can be explained purely through input perturbation and output
observation (Ribeiro et al., 2016), and occlusion sensitivity established the specific move of
masking an input region and reading the output change (Zeiler & Fergus, 2014). Together they make
XAI possible on a closed API with no gradients, activations, or log-probabilities — which is the
only regime available here.

But the audio-specific literature pushes back on exactly the representation this study must use.
audioLIME argues that interpretability for audio should mean **listenability**, and that treating
spectrogram patches like image superpixels yields components corresponding to nothing audible;
its answer is to perturb source-separated stems (Haunschmid et al., 2020). Nasr et al. (2025)
extend the critique to SER specifically: vision-derived saliency highlights time–frequency regions
without establishing that those regions correspond to meaningful acoustic markers of emotion.

Source separation is unavailable for single-speaker CREMA-D clips — there are no stems to toggle —
so this study is left with time windows and frequency bands, i.e. the representation under
critique. That is a genuine methodological debt, mitigated (ramped mask edges, loudness-matched
filler, null-mask controls) but not discharged.

### T5 — Delivery format is invisible in current evaluation practice

No source in the corpus reports the container in which audio was submitted to a model. Audio-LLM
reports and benchmarks specify corpora, prompts, and sometimes sampling rate, but not the file
format (Chu et al., 2024; Li et al., 2025). Robustness work in this area is framed around
adversarial attacks and naturally occurring degradation such as noise and reverberation (Li et
al., 2025) — not around the transcoding that every production pipeline performs as a matter of
course. Searches aimed directly at container-versus-codec separation returned only practitioner
guidance ("prefer WAV, MP3 loses information") and studies that vary signal and format together.

The gap is real. It is also small enough to be filled by exactly one contrast:
`mp4_aac64` versus `roundtrip_wav`.

---

## 2. Contradiction resolution

### Contradiction 1 — T1 says compression hurts emotion recognition; T2 implies it should barely matter

| | Prediction for this study |
|---|---|
| **T1 (codec literature)** | 64 kbps AAC removes paralinguistic detail → measurable SER accuracy drop under `mp4_aac64` |
| **T2 (LISTEN)** | The model leans on lexical content; 64 kbps preserves intelligibility → negligible accuracy drop |

**This is not a genuine contradiction — it is a scope difference, and resolving it produces the
study's sharpest prediction.**

The codec findings were established on task-specific SER systems that extract prosodic and
spectral features and classify from them. Such systems are sensitive to coding loss precisely
because coding loss lands on the features they use. An LLM that reaches its emotion judgement
substantially via transcribed lexical content has a different exposure profile: it is sensitive to
degradation that harms *intelligibility*, and comparatively insensitive to degradation that harms
*prosody only*. At 64 kbps, intelligibility is essentially intact.

**Derived prediction (pre-registered here, before data):**

> For a lexically dominant audio LLM on a fixed-lexicon corpus, SQ1 should return a **small or null
> accuracy effect**, and the transcription arm should sit at a **WER floor across all conditions** —
> not because format is irrelevant, but because the information the format destroys is not the
> information the model was using.

This prediction is falsifiable, follows from published evidence rather than from hope, and — if
confirmed — makes SQ1 *interesting* rather than merely negative, because a null then carries a
mechanistic explanation rather than an absence of one.

**What this means for the framing — stated symmetrically, per DA Checkpoint 2 (M7).** The RQ
hierarchy was chosen before this prediction was derived, so the convergence is not evidence that
the choice was right. The honest statement is conditional and cuts both ways: *if* SQ1 comes back
flat, explanation-first is the only framing with a live question left in it; *if* SQ1 shows a real
accuracy effect, accuracy-first would have been the stronger organisation and the report must say
so plainly rather than retro-fitting the explanation angle. A prediction that cannot embarrass the
framing it accompanies is not doing any work.

### Contradiction 2 — audioLIME says time–frequency perturbation is invalid; this design depends on it

Not resolvable by argument, and this synthesis does not attempt to explain it away. The honest
position: audioLIME's critique lands, source separation is inapplicable to single-speaker speech,
and the mitigations reduce the artefact risk without eliminating it. What makes the design
defensible rather than merely convenient is that the **null-mask control measures the artefact
directly** — if masking alone flips labels at a high rate, the study reports that its attribution
signal is contaminated, rather than reporting attributions. The limitation is disclosed with
audioLIME cited as its source.

### Contradiction 3 — The corpus is chosen for experimental control and is exposed by that same control

CREMA-D's 12 fixed sentences hold linguistic content constant across items, which is precisely
what allows acoustic effects to be isolated (Cao et al., 2014). It is also what makes the corpus
the configuration in which Chen et al. (2025) predict a collapse to *neutral*. The property that
makes the design clean is the property that may make the model unresponsive.

Not resolvable by re-framing; it is a real tension requiring a decision. See Gap G1 and DA
Checkpoint 2.

---

## 3. Evidence convergence / divergence map

| Claim | Converging sources | Diverging / limiting sources | Epistemic status |
|---|---|---|---|
| Lossy coding degrades SER for task-specific models | Reddy & Vijayarajan (2020); Wu et al. (2024) | — | **Well supported** |
| Signal-fidelity metrics do not predict downstream paralinguistic loss | Wu et al. (2024) | — | **Supported**, single strong source |
| Audio LLMs rely more on lexical than acoustic cues for emotion | Chen et al. (2025) | Zhang et al. (2026) benchmark speech LLMs for SER across 35 corpora and 15 languages — not the profile of a model class at chance; Lee et al. (2025) show cross-model comparisons are confounded by prompting and inference settings | **Open empirical question.** Credible evidence of substantial lexical reliance *and* of real SER capability. Which dominates for this model on this corpus is what gate G0 measures. (Downgraded from "supported" at DA Checkpoint 2, M6.) |
| Open-ended generative SER evaluation is highly prompt-sensitive and stochastic | Zhang et al. (2026) | — | **Supported** — and it is external validation for the repeat structure (V5) and prompt-sensitivity arm (V8) |
| Explanations can shift while predictions hold | Ghorbani et al. (2019) | Demonstrated in vision under *adversarial* perturbation; transfer to audio under *natural* perturbation is untested | **Established in-domain; the transfer is this study's hypothesis** |
| Attribution methods require validity controls | Adebayo et al. (2018); Nasr et al. (2025) | — | **Well supported** |
| Time–frequency occlusion is a valid interpretable representation for audio | Zeiler & Fergus (2014); Ribeiro et al. (2016) | Haunschmid et al. (2020); Nasr et al. (2025) | **Contested** — used with disclosed limitation |
| Container is separable from codec in its effect on a model | — | — | **No evidence either way** — the gap this study addresses |
| Human audio-only emotion recognition on CREMA-D ≈ 40.9% | Cao et al. (2014) | — | **Well supported** — and it recalibrates gate G0 to human parity |

---

## 4. Gap analysis

**G1 — No published expectation for this model on this corpus (blocking).**
Nothing in the corpus establishes what `gemini-flash-latest` achieves on CREMA-D audio-only.
Cao et al. (2014) put *human* audio-only accuracy at 40.9%, which means the Phase 1 floor of 40%
was inadvertently set at **human parity** — a far more demanding bar than intended. Combined with
Chen et al.'s (2025) predicted collapse to *neutral* on neutral-lexicon corpora and the smoke run's
8/8 incorrect responses (predominantly *neutral* for *angry* items), the probability that gate G0
fails is material. G0 must be run before anything else, and the 40% threshold should be restated
as a *deliberate* human-parity benchmark or lowered to a defensible headroom criterion — but the
choice must be made explicitly and in advance.

**G2 — No reference value exists for explanation stability in audio (and this is a contribution, not
just a gap).**
No source reports how much an audio attribution map varies between repeated runs under no
manipulation. `S_within` therefore cannot be predicted, only measured, and the study cannot power
SQ3 in advance.

Per DA Checkpoint 2 (M8), this is **promoted from control to reported result.** The design must
measure `S_within` regardless of how every other question resolves, which makes it the study's one
guaranteed empirical output: the run-to-run stability of black-box occlusion attribution on a
stochastic commercial audio LLM. Given that the format effect is predicted flat and the container
effect may be inconclusive, this may prove the most durable and reusable thing the study produces,
and it is directly actionable for anyone else attempting black-box XAI against an API.

**G3 — No prior container/codec decomposition means no prior effect size.**
There is nothing to power SQ2 against, which is a second, independent reason C1's equivalence
bound had to be derived from pilot data rather than from literature.

**G4 — The mask-artefact rate for speech LLMs is unknown.**
The null-mask control is doing more work than usual: it is not confirming a known-small quantity,
it is measuring an unmeasured one. If it turns out large, the attribution arm is uninterpretable
and must be reported as such.

**G5 — The task may be the wrong probe if the model is lexically dominant.**
Should G0 fail, the literature points to specific alternatives: reduce the label set to a
high/low-arousal binary (easier, more prosody-dependent), or move to a paralinguistic task with no
lexical shortcut (speaker or voice-quality judgements). Both keep the format manipulation intact
while removing the lexical escape route. This is a design pivot, not a fallback — and it arguably
produces a *better* study, because a task with no lexical shortcut is one where compression has
something to destroy.

---

## 5. Theoretical framework for the report

The study sits at the junction of two established results and one unexamined seam.

- From **codec robustness** (T1): lossy coding removes paralinguistic information that
  signal-fidelity metrics do not track.
- From **explanation fragility** (T3): a stable prediction does not imply a stable decision basis.
- The **unexamined seam** (T5): delivery format is not reported in audio-LLM evaluation, and
  container has never been separated from codec.

The contribution is the composition: **transfer the fragility result from adversarial vision to
routine audio transcoding, and test it on a decision basis measured black-box, in a design where
container and codec can be pulled apart.** Neither the fragility idea nor the perturbation method
is new; their combination on this seam is.

**The single most likely outcome, given the evidence assembled here, is:** flat accuracy (SQ1
null, mechanistically explained by lexical dominance), an inconclusive-or-equivalent container
effect (SQ2), and a first empirical measurement of audio attribution stability (SQ3) whose
interesting cell — divergent explanation under unchanged accuracy — is a live possibility but not
the favourite. That is a publishable study if and only if it is framed and pre-registered as such
from the outset. It is not a study that can afford to discover its null late.
