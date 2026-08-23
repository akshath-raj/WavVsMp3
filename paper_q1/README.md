# paper_q1 — manuscript

**What the Model Learns from Coded Audio: Null-Calibrated Feature Attribution
in Explainable Speech Emotion Recognition**

Formatted for **IEEE Transactions on Affective Computing** (IEEEtran journal
class, IEEE numeric citations). 19 pages including two appendices and
references.

Confined to interpretable models over named acoustic descriptors — fifteen
scikit-learn classifiers plus a PyTorch MLP, attributed with SHAP, LIME,
permutation importance, partial dependence, global surrogates and six
gradient-based methods. The audio-language-model arm of the parent project is
parked under `../future_work/` and is not part of this manuscript.

## Build

```bash
python make_figures.py          # regenerate figs/*.pdf from committed artefacts
python make_appendix.py         # regenerate appendix_features.tex + appendix_splits.tex
tectonic -X compile main.tex    # -> main.pdf  (downloads IEEEtran on first run)
```

Both generator scripts read only committed result files under
`../xai_ser/outputs/` and hard-code no numbers except axis labels and descriptor
definitions. `make_appendix.py` reads the column list from the feature table
itself, so the appendix cannot drift from the data the models were fitted on.

## Layout

```
main.tex                 manuscript
refs.bib                 28 references, all cited
make_figures.py          8 figures, PDF (vector) + PNG mirror
make_appendix.py         Appendix A (feature inventory) + Appendix B (tree splits)
appendix_features.tex    GENERATED — 436 descriptors, defined and enumerated
appendix_splits.tex      GENERATED — every internal node of all four trees
figs/                    GENERATED
REVIEW_ROUND1.md         simulated 5-seat peer-review panel on the previous draft
```

Figure 1 (the design) is drawn in TikZ inside `main.tex`; the rest are
matplotlib.

## Where the results come from

| Section | Artefact |
|---|---|
| Corpus, separability, codec drift (§III–IV) | `xai_ser/outputs/eda/` |
| In-format leaderboard, train/serve mismatch (§VI-A–B) | `xai_ser/outputs/models/20260818_045719/` |
| **Matched-format training (§VI-C)** | `xai_ser/outputs/per_condition/20260823_094134/` |
| **Per-condition tree splits (§VI-D, App. B)** | same run, `dt_top_splits.json` |
| **Null calibration (§VI-E)** | `per_condition/null_20260823_094903/`, `paired_20260823_095359/` |
| Post-hoc + deep attribution (§VI-F–G) | `xai_ser/outputs/xai/` |
| Neutralisation (§VI-H) | `xai_ser/outputs/robustness/20260818_051057/` |

The three per-condition runs are reproduced by:

```bash
cd ../xai_ser
PYTHONPATH=src .venv/bin/python -m xai_ser.per_condition
PYTHONPATH=src .venv/bin/python -m xai_ser.per_condition_null --n-rep 20
PYTHONPATH=src .venv/bin/python -m xai_ser.per_condition_null --paired --n-rep 20
```

## Notes on the numbers

1. **Container control.** `xai_ser/reports/FINDINGS.md` §D7 states that
   `roundtrip_wav` tracks `mp4_aac64` "within 0.003 UAR for all 16 models".
   0.003 is the *median*; the maximum is 0.010 (kNN). The paper reports both.
2. **Model count.** The zoo is 14 scikit-learn classifiers plus a PyTorch ANN —
   15 trained models. The "16" in the findings document counts the two dummy
   baselines.
3. **Two floors, two meanings.** §VI-G measures attribution stability with the
   model *fixed* and its input changed; §VI-E measures ranking stability with the
   model *refitted*. The first is high (ρ ≥ 0.98 for trees), the second is not.
   These are different quantities and the paper says so explicitly.
4. **The RBF-SVM's KernelSHAP estimate** is rank-deficient (100 coalitions, 436
   descriptors) and is excluded from every attribution conclusion; this is stated
   in the manuscript rather than silently dropped.
5. **Descriptor count.** 436 columns. The extraction schema JSON lists 435
   because `peak_amplitude` is emitted outside the family functions; the
   appendix is built from the table's actual columns.
