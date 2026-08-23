# Annotated Bibliography (APA 7.0)
**Agents:** `bibliography_agent` + `source_verification_agent` | **Phase:** 2
**Corpus:** 19 sources, all verified against Crossref DOI records or the arXiv API.
**Grading:** discipline-relative (`source_quality_hierarchy.md` §Field-Specific Adjustments) — in
computer science, a peer-reviewed conference paper is a gold-standard venue, and a well-cited
arXiv preprint that has since appeared at a major venue is graded on that venue.

Legend — **Tier**: 1 = peer-reviewed venue, 2 = preprint, 3 = grey literature.
**Grade**: A = directly load-bearing and methodologically sound; B = sound and supporting;
C = contextual only.

---

## Theme 1 — Codec and compression effects on speech tasks

**Reddy, A. P., & Vijayarajan, V. (2020). Audio compression with multi-algorithm fusion and its
impact in speech emotion recognition. *International Journal of Speech Technology, 23*(2),
277–285. https://doi.org/10.1007/s10772-020-09689-9**
*Tier 1 · Grade B · Verified: Crossref DOI lookup (1.00)*
Directly on-topic for SQ1: compression degrades SER, and the degradation is feature-dependent
rather than uniform. Establishes the baseline expectation that lossy coding costs accuracy for
*task-specific* models — which is exactly why SQ1 is a supporting question and not the headline.
**Limitation for our purposes:** classical hand-crafted-feature pipelines, not neural models, and
certainly not LLMs. It licenses an expectation, not a prediction.

**Wu, H., Chen, X., Lin, Y.-C., Chang, K., Du, J., Lu, K.-H., Liu, A. H., Chung, H.-L., Wu, Y.-K.,
Yang, D., Liu, S., Wu, Y.-C., Tan, X., Glass, J., Watanabe, S., & Lee, H. (2024). *Codec-SUPERB @
SLT 2024: A lightweight benchmark for neural audio codec models* (arXiv:2409.14085). arXiv.
https://arxiv.org/abs/2409.14085**
*Tier 1 (SLT 2024 workshop) · Grade A · Verified: arXiv ID (1.00)*
The most methodologically important source in this theme. Its central observation —
**signal-level metrics alone do not capture losses in semantic, linguistic, or paralinguistic
content** — is the direct justification for this study's design choice to measure fidelity (SNR,
LSD, PESQ) *and* behaviour separately rather than assuming one predicts the other. Also
establishes emotion recognition as the standard probe for paralinguistic information loss under
coding, which is what our task selection does.

**Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014). CREMA-D:
Crowd-sourced emotional multimodal actors dataset. *IEEE Transactions on Affective Computing,
5*(4), 377–390. https://doi.org/10.1109/TAFFC.2014.2336244**
*Tier 1 · Grade A · Verified: Crossref DOI lookup (1.00)*
The corpus paper. 7,442 clips, 91 actors, 12 fixed sentences, 6 emotion categories, crowd-rated by
2,443 raters. **Two facts from it are load-bearing for this study.** First, human recognition of
intended emotion from **audio alone is 40.9%** — which recalibrates gate G0's 40% floor from
"a modest bar" to "human parity", a materially different and much harder threshold. Second, the
fixed 12-sentence lexicon holds linguistic content constant across items, which is what makes the
corpus suitable for isolating acoustic effects — and simultaneously what exposes it to the
lexical-dominance failure mode below.

---

## Theme 2 — Audio-LLM capability and robustness evaluation

**Chen, J., Guo, Z., Chun, J., Wang, P., Perrault, A., & Elsner, M. (2025). *Do audio LLMs really
LISTEN, or just transcribe? Measuring lexical vs. acoustic emotion cues reliance*
(arXiv:2510.10444). arXiv. https://arxiv.org/abs/2510.10444**
*Tier 1 (EACL 2026) · Grade A · Verified: arXiv ID (1.00), submitted 2025-10-12, rev. 2025-10-17*
**The most consequential source in the corpus.** Introduces LISTEN, a benchmark that disentangles
lexical from acoustic contributions to emotion judgements in large audio-language models. Across
six state-of-the-art LALMs it finds consistent **lexical dominance**: models predict *neutral*
when lexical cues are neutral or absent, gain little when cues align, fail under cue conflict, and
approach chance in paralinguistic contexts. Conclusion: current models "largely transcribe rather
than listen".
**Implication, and it cuts both ways.** *Against* the study: CREMA-D's emotionally neutral fixed
lexicon is the exact configuration predicted to collapse to *neutral*, threatening a floor effect
that would flatten every contrast (see `search_strategy.md` and Phase 3). *For* the study: if the
model's emotion judgement rests mostly on transcribed words, then the temporal- and
spectral-occlusion arm becomes a direct measurement of that dependence — masking regions that
carry lexical content versus regions that carry prosody. SQ3 acquires a second, independent
motivation.

**Chu, Y., Xu, J., Yang, Q., Wei, H., Wei, X., Guo, Z., Leng, Y., Lv, Y., He, J., Lin, J., Zhou,
C., & Zhou, J. (2024). *Qwen2-Audio technical report* (arXiv:2407.10759). arXiv.
https://arxiv.org/abs/2407.10759**
*Tier 2 · Grade B · Verified: arXiv ID (1.00)*
Reference point for open-weights audio LLM capability and for what is reported in audio-LLM
evaluation practice. Relevant to the deferred cross-model extension (RQ Brief candidate 5).
Notably, like the rest of this literature, it does not state the container format in which audio
was submitted — the reporting gap this study argues should close.

**Li, K., Shen, C., Liu, Y., Han, J., Zheng, K., Zou, X., Wang, L. Z., Zhang, S., Du, X., Luo, H.,
… Li, X. (2025). *AudioTrust: Benchmarking the multifaceted trustworthiness of audio large
language models* (arXiv:2505.16211). arXiv. https://arxiv.org/abs/2505.16211**
*Tier 2 · Grade B · Verified: arXiv ID (1.00), rev. 2026-03-12*
Frames ALLM robustness as covering both adversarial attacks and **naturally occurring performance
degradation**. This study sits squarely in the second category and, unlike most work there, uses a
degradation that is not noise or reverberation but the transcoding every production pipeline
already performs. Useful for positioning; not a methodological dependency.

---

## Theme 3 — Perturbation-based XAI and its foundations

**Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the
predictions of any classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on
Knowledge Discovery and Data Mining* (pp. 1135–1144). ACM.
https://doi.org/10.1145/2939672.2939778**
*Tier 1 · Grade A · Verified: Crossref DOI re-fetch (truncated-title record; see search_strategy.md)*
LIME — the model-agnostic surrogate that makes black-box attribution possible at all. Its core
premise, that a model can be probed purely through input perturbation and output observation, is
the licence under which this study can do XAI on a closed API with no gradients, activations, or
log-probabilities. Our segment-level attribution is a simplified LIME in this lineage.

**Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. In
*Computer Vision – ECCV 2014* (Lecture Notes in Computer Science, pp. 818–833). Springer.
https://doi.org/10.1007/978-3-319-10590-1_53**
*Tier 1 · Grade A · Verified: Crossref DOI lookup (1.00)*
The origin of occlusion sensitivity: systematically mask a region of the input and observe the
change in output. This study's temporal- and spectral-occlusion protocol is that method
transposed to the time and time–frequency axes of audio.

**Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to interpreting model predictions*
(arXiv:1705.07874). arXiv. https://arxiv.org/abs/1705.07874**
*Tier 1 (NeurIPS 2017) · Grade B · Verified: arXiv ID (1.00)*
SHAP's additive-attribution framework and its consistency axioms. Cited as the principled
alternative to raw occlusion; not adopted here because exact Shapley values over 16 segments would
multiply the call budget beyond what the API quota supports. That trade-off is stated in the
report's limitations rather than left implicit.

**Haunschmid, V., Manilow, E., & Widmer, G. (2020). *audioLIME: Listenable explanations using
source separation* (arXiv:2008.00582). arXiv. https://arxiv.org/abs/2008.00582**
*Tier 1 (MML 2020 workshop) · Grade A · Verified: arXiv ID (1.00)*
The key adaptation of LIME to audio, and the source of this study's most important design
constraint. Its argument: interpretability for audio must mean **listenability**, and perturbing
spectrogram patches as if they were image superpixels produces components that do not correspond
to anything audible. audioLIME instead perturbs source-separated stems.
**Where we depart, and why.** Source separation is not applicable to single-speaker CREMA-D clips
— there are no stems to toggle. Our interpretable components are therefore time windows and
frequency bands, which is the representation audioLIME critiques. The honest response is not to
claim the critique doesn't apply but to mitigate it directly: ramped mask edges, loudness-matched
filler, and null-mask controls that measure how much of any observed flip is the mask's own
artefact. This is named as a limitation in the report, with audioLIME cited as its source.

**Sotirou, T., Lyberatos, V., Menis Mastromichalakis, O., & Stamou, G. (2024). *MusicLIME:
Explainable multimodal music understanding* (arXiv:2409.10496). arXiv.
https://arxiv.org/abs/2409.10496**
*Tier 2 · Grade C · Verified: arXiv ID (1.00), rev. 2025-03-17*
Extends LIME-style explanation to multimodal (audio + lyrics) models, showing how attribution can
be split across modalities. Contextual support for the idea that attribution can reveal *which*
information channel a model leans on — conceptually parallel to Chen et al.'s lexical-vs-acoustic
split, in a different domain.

**Nasr, S., Ren, Z., & Johnson, D. (2025). *Beyond saliency: Enhancing explanation of speech
emotion recognition with expert-referenced acoustic cues* (arXiv:2511.11691). arXiv.
https://arxiv.org/abs/2511.11691**
*Tier 2 · Grade B · Verified: arXiv ID (1.00)*
Argues that saliency methods imported from vision highlight spectrogram regions without
establishing that those regions correspond to meaningful acoustic markers of emotion — limiting
faithfulness. A direct critique of the class of method this study uses, and the reason our
spectral bands are defined on acoustically interpretable ranges rather than arbitrary bins. The
report cites this as an acknowledged weakness with a partial mitigation, not a solved problem.

---

## Theme 4 — Explanation stability: the theoretical core of SQ3

**Ghorbani, A., Abid, A., & Zou, J. (2019). *Interpretation of neural networks is fragile*
(arXiv:1710.10547). arXiv. https://arxiv.org/abs/1710.10547**
*Tier 1 (AAAI 2019) · Grade A · Verified: arXiv ID (1.00)*
**The paper SQ3 is built on.** Demonstrates that perceptually indistinguishable inputs receiving
the *same predicted label* — sometimes with *increased* confidence — can be assigned substantially
different interpretations. That is the accuracy/explanation dissociation, established in vision.
This study's contribution is the transfer: from adversarially constructed perturbations to
**ordinary transcoding that happens in every deployed pipeline**, and from white-box gradients to
black-box occlusion on a commercial API. Prominent citation is mandatory; without it the
dissociation would read as a claim of invention rather than of transfer.

**Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). *Sanity checks
for saliency maps* (arXiv:1810.03292). arXiv. https://arxiv.org/abs/1810.03292**
*Tier 1 (NeurIPS 2018) · Grade A · Verified: arXiv ID (1.00)*
Shows that some widely used saliency methods are independent of both the model and the data
generating process — i.e. they produce plausible-looking maps that explain nothing, and visual
assessment cannot tell the difference. This is the direct justification for **V7, the within-format
stability floor**: an attribution map must be shown to be reproducible under no manipulation
before any change in it under manipulation can mean anything. Any version of this study without
that control fails the standard Adebayo et al. set.

---

## Theme 5 — Methodological and reporting standards

**Kapoor, S., Cantrell, E. M., Peng, K., Pham, T. H., Bail, C. A., Gundersen, O. E., Hofman, J.
M., Hullman, J., Lones, M. A., Malik, M. M., Nanayakkara, P., Poldrack, R. A., Raji, I. D.,
Roberts, M., Salganik, M. J., … Narayanan, A. (2024). REFORMS: Consensus-based recommendations for
machine-learning-based science. *Science Advances, 10*(18).
https://doi.org/10.1126/sciadv.adk3452**
*Tier 1 · Grade A · Verified: Crossref DOI lookup (1.00)*
32-item, 8-module consensus checklist for ML-based science, developed by 19 researchers across
disciplines. Adopted as this study's reporting standard, resolving the Phase 1 open question
(no EQUATOR guideline covers computational benchmark experiments). Its emphasis on reproducibility
and on failures of validity and generalizability maps directly onto the report's obligations to
state the exact resolved `model_version`, the full prompt text, encoder arguments, and call
attrition at every stage.

**Lakens, D. (2017). Equivalence tests: A practical primer for *t* tests, correlations, and
meta-analyses. *Social Psychological and Personality Science, 8*(4), 355–362.
https://doi.org/10.1177/1948550617697177**
*Tier 1 · Grade A · Verified: Crossref DOI re-fetch (truncated-title record; published version cited, not the preprint)*
**Lakens, D., Scheel, A. M., & Isager, P. M. (2018). Equivalence testing for psychological
research: A tutorial. *Advances in Methods and Practices in Psychological Science, 1*(2), 259–269.
https://doi.org/10.1177/2515245918770963**
*Tier 1 · Grade A · Verified: Crossref DOI lookup (1.00)*
**Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power
approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics
and Biopharmaceutics, 15*(6), 657–680. https://doi.org/10.1007/BF01068419**
*Tier 1 · Grade A · Verified: Crossref DOI lookup (1.00)*
The TOST lineage, from Schuirmann's original procedure to Lakens' modern treatment. These sources
supply the smallest-effect-size-of-interest framework that DA Checkpoint 1 used to expose C1: the
requirement to *declare* a SESOI is precisely what revealed that Δ = 3 accuracy points is
unreachable at n = 50. They also supply the correct discipline for reporting SQ2 — a null container
effect can only be claimed via a passed equivalence test, never via a non-significant NHST.

**McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions
or percentages. *Psychometrika, 12*(2), 153–157. https://doi.org/10.1007/BF02295996**
*Tier 1 · Grade B · Verified: Crossref DOI lookup (1.00)*
The paired-proportions test underlying the power reasoning in DA Checkpoint 1: with the same items
in every format, the informative quantity is the discordant pairs, and their count at n = 50 is
what caps the design's resolution. Cited in the Methods where the minimum detectable effect is
derived.

---

## Theme 6 — Counter-evidence and evaluation practice (added after DA Checkpoint 2 flagged one-source dependency on Chen et al.)

**Zhang, H., Chou, H.-C., Narayanan, S., & Hain, T. (2026). *VoxEmo: Benchmarking speech emotion
recognition with speech LLMs* (arXiv:2603.08936). arXiv. https://arxiv.org/abs/2603.08936**
*Tier 2 · Grade A · Verified: arXiv ID, published 2026-03-09*
The most important counterweight in the corpus, and it earns Grade A on two counts. First, it
covers 35 emotion corpora across 15 languages for speech LLMs, establishing that these models are
evaluated at scale on SER and are not uniformly at chance — a direct qualification of the
lexical-dominance picture. Second, and independently valuable to this design, it identifies
**zero-shot stochasticity** as a first-order evaluation problem: moving from closed-set
classification to open text generation makes results "highly sensitive to prompts". That is
external, published support for two of this study's controls — the repeat structure (V5) and the
`emotion_v2`/`v3` prompt-sensitivity arm (V8) — which were specified on general principle and turn
out to address a documented failure mode.

**Hu, H., You, L., Xu, H., Wang, Q., Yu, F. R., & Ma, F. (2025). *EmoBench-M: Benchmarking
emotional intelligence for multimodal large language models* (arXiv:2502.04424). arXiv.
https://arxiv.org/abs/2502.04424**
*Tier 2 · Grade C · Verified: arXiv ID, published 2025-02-06*
Multimodal emotional-intelligence benchmarking; contextual evidence that MLLMs are evaluated on
emotion tasks as a matter of course. Contributes breadth, not a methodological dependency.

**Lee, T., Tu, H., Wong, C. H., Wang, Z., Yang, S., & Mai, Y. (2025). *AHELM: A holistic evaluation
of audio-language models* (arXiv:2508.21376). arXiv. https://arxiv.org/abs/2508.21376**
*Tier 2 · Grade B · Verified: arXiv ID, published 2025-08-29*
Aggregates datasets across capabilities including fairness and safety, and explicitly names the
problem that model comparisons are confounded by "different prompting methods and inference
parameters". Supports this study's insistence on byte-identical prompts across conditions (V3) —
and supports the broader argument that **unreported input-processing choices are a live source of
non-comparability** in audio-LLM evaluation, of which the container is one more instance.

**Luo, K., Zhou, Z., Wang, L., Lin, L., Shao, T., & Zhang, Y. (2026). *A survey of large audio
language models: Generalization, trustworthiness, and outlook* (arXiv:2605.20266). arXiv.
https://arxiv.org/abs/2605.20266**
*Tier 2 · Grade C · Verified: arXiv ID, published 2026-05-18*
Recent survey of LALM trustworthiness and generalization; used for positioning and to check that
the container/codec gap is not covered by an existing review. It is not.

---

## Corpus assessment

| Property | Value |
|---|---|
| Sources | 23 |
| Verified | 23 (100%) |
| Tier 1 (peer-reviewed venue) | 14 |
| Tier 2 (preprint) | 9 |
| Grade A (load-bearing) | 12 |
| Median year | 2018 |
| Published 2024 or later | 8 |
| Retraction/update flags (Crossref `updated-by`) | none detected |

**Currency.** The split is deliberate and defensible: methodological foundations (LIME, occlusion,
TOST, McNemar, sanity checks) are older because they are settled, while every source describing
the object of study — audio LLMs and their evaluation — is from 2024 or later. A recency-weighted
corpus would have been worse here, not better.

**Gaps the corpus does not cover, carried to Phase 3:**
1. No source establishes what `gemini-flash-latest` specifically achieves on CREMA-D, so gate G0
   cannot be predicted from the literature and must be measured.
2. No source measures explanation stability for an audio model under *any* perturbation — the
   within-format noise floor for audio attribution has no published reference value.
3. No source separates container from codec for any model, which is the novelty claim and also
   means there is no prior effect size to power against.
