# Backend Probe — Azure AI Foundry `gpt-audio-1.5`
**Date:** 2026-08-17 | **Trigger:** user switched the API arm from Gemini to Microsoft Foundry
**Scripts:** `probe_foundry_audio.py`, `probe_foundry_g0.py`, `probe_foundry_logprobs.py`
**Raw data:** `foundry_audio_probe.json`, `foundry_g0_probe.json`, `foundry_logprob_probe.json`,
`foundry_stability_probe.json`

---

## Connection facts

| Item | Value |
|---|---|
| Endpoint | `https://<resource>.services.ai.azure.com/openai/v1/chat/completions` |
| Deployment name | **`gpt-audio-1.5`** (not `gpt-1.5-audio` — the name was transposed) |
| Deployments on resource | exactly 1 |
| Auth header | `Authorization: Bearer <key>` (an `api-key` header also reaches the resource) |
| Text-only requests | **rejected** — "requires that either input content or output modality contain audio". Must send `modalities: ["text"]` plus an audio part. |
| Determinism | `temperature: 0` + `seed` → **bit-identical outputs and logprobs** |
| Logprobs | **supported**, `top_logprobs` up to 20 |

`/openai/v1/models` returns the 403-model regional *catalogue*, not deployments. Use
`/openai/deployments?api-version=2023-03-15-preview` to see what is actually deployed.

---

## Finding 1 — MP4/AAC is rejected. The container arm must go via MP3.

| Payload | Declared `format` | Result |
|---|---|---|
| `ref.wav` | `wav` | **accepted** |
| `roundtrip_wav.wav` | `wav` | **accepted** |
| `mp3_64.mp3` | `mp3` | **accepted** |
| `mp4_aac64.mp4` | `mp4` | rejected — *"Supported values are: 'wav' and 'mp3'"* |
| `mp4_aac64.mp4` | `m4a` | rejected — same |
| `mp4_aac64.mp4` | `aac` | rejected — same |
| `mp4_aac64.mp4` | **`wav`** (deliberate mislabel) | rejected — *"This model does not support the format you provided"* |

The mislabel case is the informative one. Declaring MP4 bytes as `wav` produces a **different**
error from declaring an unsupported format string — so the service **inspects the actual container
bytes** rather than trusting the label. Container parsing is therefore behaviourally live on this
backend at least at the validation layer, which is a precondition for the container question being
meaningful at all.

**Consequence for SQ2.** The `mp4_aac64` vs `roundtrip_wav` contrast is unavailable here, exactly as
it was for Phi-4. The equivalent contrast survives via MP3:

```
mp3_64            (MP3 container, lossy-coded signal)
roundtrip_wav_mp3 (WAV container, SAME decoded signal)   <- needs generating
```

`roundtrip_wav_mp3` was already an approved amendment (DA Checkpoint 1, M4). It is now load-bearing
rather than a symmetry nicety. Note the project name is now slightly off: the live contrast is
WAV vs **MP3**, not WAV vs MP4.

---

## Finding 2 — Gate G0 FAILS, and the failure is degenerate

12 items, reference (lossless WAV) condition, `temperature=0`, `seed=12345`:

| Metric | Value |
|---|---|
| Accuracy | **1/12 = 8.3%** (below the 16.7% chance rate for a 6-way forced choice) |
| Prediction distribution | `{neutral: 12}` |
| Distinct labels used | **1 of 6** |

The single correct answer was the one item whose gold label *is* `neutral`. The model did not
guess badly; it did not vary at all.

This is the collapse Chen et al. (2025) predict for neutral-lexicon corpora, now observed on a
**second independent backend** — the Gemini smoke run showed the same signature (`neutral` returned
for `angry` items). Two different vendors, same corpus, same failure.

**Gate G0 fails and gate G0b (response degeneracy) fires.** Under a label-only readout the study
is dead here: every occlusion mask would return `neutral`, the attribution vector would be
constant, and Spearman ρ would be undefined — precisely the Critical failure mode raised at DA
Checkpoint 2 (C2).

---

## Finding 3 — Logprobs rescue it, and turn the failure into the finding

Because `logprobs` are exposed, the readout does not have to be the label. Using a **forced binary
prompt** ("Is the speaker *angry* or *neutral*?") puts two labels in direct competition at one
token position, and the probability mass on the gold label becomes a continuous dependent variable.

P(gold) across three conditions, `temperature=0`, `seed` fixed:

| Item | Gold | `ref.wav` | `roundtrip_wav` | `mp3_64` | Argmax |
|---|---|---:|---:|---:|---|
| 1005_MTI_ANG_XX | angry | 0.000 | 0.000 | 0.000 | constant |
| **1009_IEO_ANG_MD** | angry | **0.038** | **0.338** | **0.015** | constant |
| 1022_ITH_ANG_XX | angry | 0.000 | 0.000 | 0.000 | constant |
| 1026_ITS_ANG_XX | angry | 0.004 | 0.000 | 0.004 | constant |
| 1012_IWL_SAD_XX | sad | 0.202 | 0.244 | 0.081 | constant |
| 1015_DFA_SAD_XX | sad | 0.009 | 0.022 | 0.002 | constant |

Mean spread across formats **0.085**; max **0.323**. **Argmax moved on 0 of 6 items.**

### The noise floor is exactly zero

Five *identical* repeated calls per item per condition:

| Item | Condition | P(gold) × 5 | SD | Range |
|---|---|---|---:|---:|
| 1009_IEO_ANG_MD | ref | 0.038, 0.038, 0.038, 0.038, 0.038 | 0.000 | 0.000 |
| 1009_IEO_ANG_MD | roundtrip_wav | 0.338 × 5 | 0.000 | 0.000 |
| 1012_IWL_SAD_XX | ref | 0.202 × 5 | 0.000 | 0.000 |
| 1012_IWL_SAD_XX | roundtrip_wav | 0.244 × 5 | 0.000 | 0.000 |
| 1015_DFA_SAD_XX | ref | 0.009 × 5 | 0.000 | 0.000 |
| 1015_DFA_SAD_XX | roundtrip_wav | 0.022 × 5 | 0.000 | 0.000 |

**Within-condition range: 0.0000. Across-format spread: up to 0.3229.** The signal is not API
jitter.

**This is the dissociation cell, observed directly.** The label never moves — an accuracy-only
benchmark would report this model as perfectly format-robust — while the probability mass
supporting the correct answer shifts by up to 0.32. That is the hypothesis SQ3 was built to test,
appearing in a 24-call exploratory probe.

---

## What this changes in the protocol

| Element | Before | After |
|---|---|---|
| Primary DV | 6-way label accuracy (binary, paired) | **P(gold) under forced binary prompt** (continuous, noise-free) |
| Repeats | 3 per cell (sampling noise) | **1** — backend is deterministic |
| `S_within` noise floor | unknown, had to be measured | **measured: 0.000** |
| DA CP1 **C1** (unreachable Δ = 3pp at n = 50) | blocking | **largely dissolved** — a continuous DV with zero measurement error has far more power than paired proportions with a 15–18pp MDE |
| DA CP2 **C2** (degenerate response → undefined attribution) | Critical | **dissolved** — attribution runs on P(gold), which varies even when the label does not |
| Gate G0 (≥40% accuracy) | untested | **FAILS (8.3%)** — but is no longer disqualifying, because the readout changed |
| Gate G0b (degeneracy) | untested | **FIRES** — and the pre-specified forced-binary replacement is exactly what works |
| Container contrast | `mp4_aac64` vs `roundtrip_wav` | **`mp3_64` vs `roundtrip_wav_mp3`** (MP4 rejected) |
| Call budget | ~9,230 | roughly **one third** — repeats drop from 3 to 1 |

The two gates the research phase installed both fired, and both had a pre-specified response that
worked. That is the gates doing their job rather than the study failing.

---

## Open issues before these numbers can be trusted

**O1 — Unexplained logprob anomaly (must resolve).** In the free 6-way readout, the chosen token
was `'neutral'` at logprob −6.25 while `'<|end|>'` appeared in the same position's `top_logprobs`
at −0.00, under `temperature=0`. Under greedy decoding the emitted token should be the argmax.
Either `top_logprobs` is not aligned to the chosen position, or the service post-processes the
distribution (e.g. suppressing an immediate stop). The binary readout behaved sensibly, but the
entire graded method rests on these numbers meaning what they appear to mean. Resolve by testing a
prompt with a known-forced answer and checking whether the emitted token is ever the listed argmax.

**O2 — Determinism verified over minutes, not days.** Five identical calls within one window agreed
to 3 d.p. That does not establish stability across deployment updates. Re-run the stability probe
at the start and end of the main grid and record both.

**O3 — Exploratory n.** Six items, one prompt, three conditions, no correction. This is a signal
worth designing around, not a result. The direction is also odd and unexplained: `roundtrip_wav`
(the *degraded* signal) scored **higher** P(angry) than `ref` on the one item with real movement.

**O4 — Prompt dependence.** The forced binary prompt reveals the gold label to the model as one of
two options, which is a different task from the 6-way. Whether P(gold) under binary tracks anything
the 6-way readout would call "the model's evidence" needs its own validation.

**O5 — Credential hygiene.** The Foundry key was pasted in plaintext in chat, as were the Gemini
and original Foundry keys. All require rotation.
