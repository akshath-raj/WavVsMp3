# future_work — the audio language model arm (parked)

This directory holds the black-box arm of the project: occlusion attribution
against a commercial audio language model (`gpt-audio-1.5` via an
OpenAI-compatible Azure AI Foundry endpoint). It is **not** part of the current
manuscript, which is confined to interpretable feature models and a small neural
network, because those admit exact attribution over named acoustic descriptors
and an audio LLM does not.

Nothing here has been deleted. The arm re-executes without further paid
inference calls: every API response is content-addressed in `exp/cache/` by a
digest over deployment, audio bytes, prompt and decoding parameters (8,016
unique responses, 0 errors).

## Layout

| Path | What it is |
|---|---|
| `exp/` | experiment drivers, gates, controls, analysis, and the response cache |
| `exp/out/` | result JSON/parquet: arousal grid, occlusion attributions, fidelity, band damage, WER, determinism |
| `src/`, `configs/` | the stimulus-construction and API-client pipeline |
| `data/` | Arm A manifest and results tables (bulk audio is regenerable, see `.gitignore`) |
| `research/` | the ARS protocol and feasibility report that shaped the design (phases 1-6) |
| `probe_*.py` | endpoint capability probes (container acceptance, log-probability availability) |
| `.venv/`, `pyproject.toml`, `uv.lock` | the Arm A environment |

The `.venv` was moved with the rest of the arm and its script shebangs now point
at the old location. Recreate it with `uv sync` from inside this directory
before re-running anything.

## What the arm established

Recorded here so it is not lost while the work is parked:

- **A measured null floor for perturbation attribution.** An inaudible 1-LSB
  dither already dissipates ~30% of the occlusion rank ordering
  (rho = 0.699). No published reference value for this quantity exists, and
  without it no claim of the form "manipulation M changed the explanation" is
  interpretable.
- **Sensitivity survives coding; the criterion does not.** Binary arousal AUC is
  statistically indistinguishable across seven conditions (0.710-0.773) while
  marginal P(high) falls 0.208 -> 0.144.
- **Attribution reorders beyond the floor.** Codec rho = 0.491 against the
  0.699 floor (p = 4.2e-8); the container contrast does not (p = .478).
- **A withdrawn claim.** An earlier version reported a container effect on the
  ground that byte-identical requests return bit-identical responses. That is a
  determinism check, not a null control, and the claim was withdrawn once the
  dither floor was measured. The reasoning is preserved in
  `research/phase6_final/final_report.md`.
- **Unreported controls that a resumed study should use.** `exp/out/wer.parquet`
  holds a transcription arm across all seven conditions (ref 0.085 -> mp3_128
  0.147); `exp/out/results.json` holds a six-way construct-validity result
  (P(gold) - P(foil) = 0.170, p = 3.5e-39, n = 342) showing the model carries
  gold-label information below its argmax.
