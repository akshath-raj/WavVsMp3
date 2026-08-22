# paper_interspeech — conference manuscript (Arm B only)

*Thresholds Survive, Products Collapse: Model Family Governs Robustness to
Perceptual Coding in Speech Emotion Recognition*

A standalone conference paper covering **only** the interpretable-models arm of
this repository (`xai_ser/`). The audio-language-model arm (`exp/`) does not
appear here; the two-arm journal version is `paper_q1/`.

## Build

```bash
python make_figures.py          # regenerate figs/*.pdf from committed artefacts
tectonic -X compile main.tex    # -> main.pdf
```

`make_figures.py` also prints every number the manuscript quotes, so text and
data cannot drift. It hard-codes no reported value — only axis limits and tick
labels.

## Layout

```
main.tex            manuscript
main.pdf            compiled output
refs.bib            32 references, each verified against Crossref or the arXiv API
make_figures.py     figure generation + a `report()` pass that dumps every quoted number
figs/               4 figures, PDF (vector) + PNG mirror
abstract_bilingual.md   English + 繁體中文 abstract and keywords
```

## Where the numbers come from

Every value is read from committed artefacts under `xai_ser/outputs/`, at these
pinned run directories:

| Source | Used for |
|---|---|
| `eda/eda_summary.json`, `eda/format_drift.csv` | corpus stats, human ceiling, variance decomposition, feature drift, decode-path control |
| `models/20260818_045719/cross_format.csv` | Table 1 (14 scikit-learn models) |
| `models/ann/20260818_050037/metrics.json` | Table 1, ANN row (trained by a separate module, so it is not in `cross_format.csv`) |
| `xai/20260818_050808/xai_summary.json` | Table 2 tabular rows, explainer disagreement, surrogate fidelity |
| `xai/deep/20260818_050247/deep_xai_summary.json` | Table 2 ANN row, gradient-method agreement, model-randomisation check |
| `robustness/20260818_051057/` | Fig. 4, the 38.6σ feature shift, neutralisation recovery |

## Template note

The official `interspeech.sty` is not redistributable and is not bundled. The
preamble reproduces the Interspeech layout (A4, two 80 mm columns, 10 mm
gutter, 9 pt Times) so the paper compiles anywhere with Tectonic. For an actual
submission, drop the official class file in and delete the geometry/style block
marked in `main.tex`.

## Page count

The current build is **6 pages** (content through p. 5, references on
pp. 5–6). Interspeech allows 4 pages of content plus a 5th page for references
only, so this is one page over. The overage is content, not formatting — the
paper carries 4 figures, 2 tables, and the full attribution story including
both negative results. Closing the gap requires dropping a result; the
cheapest candidates, in order:

1. Cut §4.4 (attribution) down to the stability table and drop Fig. 3 and the
   explainer-disagreement result — saves ≈ 1 page.
2. Drop Fig. 2 and rely on Table 1 alone for the family split — saves ≈ 0.5 page.
3. Trim the reference list from 32 to ~24 — saves ≈ 0.4 page.

Option 1 alone lands it at 5 pages. All three are content decisions, so they
are left to the author rather than applied silently.

## Scope differences from `paper_q1`

| | `paper_q1` (journal, two arms) | `paper_interspeech` (this paper) |
|---|---|---|
| Arms | A (audio LM) + B (feature models) | B only |
| Headline | explanation displacement exceeds accuracy displacement | model family governs codec robustness |
| Null control | 1-LSB dither (A) + decode-path (B) | decode-path only |
| Container result | both arms | feature arm only |

The RBF-SVM's KernelSHAP estimate is rank-deficient (100 coalitions, 436
features) and is excluded from every attribution conclusion, as in `paper_q1`.
The SVM contributes accuracy and robustness results only.
