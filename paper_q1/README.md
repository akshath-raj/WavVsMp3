# paper_q1 — combined manuscript

A single journal article unifying the two studies in this repository:

| Arm | Study | Source of results |
|---|---|---|
| A | Black-box audio language model, occlusion attribution | `exp/out/*.json`, `exp/out/*.parquet` |
| B | Interpretable feature models, SHAP/LIME/captum | `xai_ser/outputs/{eda,models,xai,robustness}/` |

Formatted for **IEEE Transactions on Affective Computing** (IEEEtran journal
class, IEEE numeric citations). 16 pages including references.

## Build

```bash
python make_figures.py          # regenerate figs/*.pdf from committed artefacts
tectonic -X compile main.tex    # -> main.pdf  (downloads IEEEtran on first run)
```

`make_figures.py` reads only committed result files; it hard-codes no numbers
except axis labels. The one exception is the PyTorch ANN's cross-format row,
which lives outside `cross_format.csv` and is taken from
`outputs/models/ann/<ts>/metrics.json` (marked in the source).

## Layout

```
main.tex            manuscript
refs.bib            39 references, each verified against Crossref or the arXiv API
make_figures.py     figure generation
figs/               8 figures, PDF (vector) + PNG mirror
```

Figure 1 (the two-arm design) and the flow structure are drawn in TikZ inside
`main.tex`; the remaining figures are matplotlib.

## Notes on the numbers

Two corrections were applied while writing, relative to the source findings
documents:

1. **Container control.** `xai_ser/reports/FINDINGS.md` §D7 states that
   `roundtrip_wav` tracks `mp4_aac64` "within 0.003 UAR for all 16 models".
   0.003 is the *median*; the maximum is 0.010 (kNN). The paper reports both.
2. **Model count.** The Arm B zoo is 14 scikit-learn classifiers plus a PyTorch
   ANN — 15 trained models. The "16" in the findings document counts the two
   dummy baselines.

The RBF-SVM's KernelSHAP estimate is rank-deficient (100 coalitions, 436
features) and is excluded from every attribution conclusion; this is stated in
the manuscript rather than silently dropped.
