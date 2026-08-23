#!/usr/bin/env python3
"""Generate the feature-inventory appendix (`appendix_features.tex`).

Two things are emitted, in this order:

1. **Descriptor tables**, one per family: what each base descriptor measures,
   which order statistics are applied to it, and how many columns it therefore
   contributes.
2. **The complete enumeration** of all 436 column names, in compact multi-column
   longtables, so the appendix is a checkable inventory rather than a summary.

Column names are read from the committed feature table, not hard-coded, so the
appendix cannot drift from the data the models were fitted on.
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
XAI_SER = HERE.parent / "xai_ser"
sys.path.insert(0, str(XAI_SER / "src"))

import pandas as pd  # noqa: E402

from xai_ser.datasets import feature_names, family_of  # noqa: E402
from xai_ser.paths import FEATURES, latest  # noqa: E402

STATS8 = "mean, std, min, max, median, IQR, skew, kurtosis"
STATS2 = "mean, std"
STATS5 = "mean, std, min, max, median"


def esc(s: str) -> str:
    """LaTeX-escape an identifier for use inside \\ttf{}."""
    return s.replace("_", r"\_")


# ---------------------------------------------------------------- descriptors
# (prefix-matcher, printed name, definition, statistics, family)
DESCRIPTORS: list[tuple[str, str, str, str, str]] = [
    # ---- spectral -----------------------------------------------------------
    ("spectral_centroid", "Spectral centroid",
     "First moment of the power spectrum. It is the frequency about which spectral energy is balanced. Correlates with perceived brightness.",
     STATS8, "spectral"),
    ("spectral_bandwidth", "Spectral bandwidth",
     "Power-weighted second moment about the centroid. It measures how widely energy is spread in frequency.",
     STATS8, "spectral"),
    ("spectral_rolloff85", "Spectral roll-off (85\\,\\%)",
     "Frequency below which 85\\,\\% of the frame's spectral energy lies.",
     STATS8, "spectral"),
    ("spectral_rolloff95", "Spectral roll-off (95\\,\\%)",
     "As above at 95\\,\\%. Directly sensitive to codec low-pass behaviour.",
     STATS8, "spectral"),
    ("spectral_flatness", "Spectral flatness",
     "Geometric mean of the magnitude spectrum divided by its arithmetic mean. It approaches 1 for noise-like frames and 0 for tonal ones, and it collapses when an encoder sets masked bins to zero.",
     STATS8, "spectral"),
    ("spectral_zcr", "Zero-crossing rate",
     "Fraction of adjacent sample pairs with a sign change. It is a cheap proxy for voicing and frication.",
     STATS8, "spectral"),
    ("spectral_rms", "Frame RMS energy",
     "Root-mean-square magnitude per frame, which gives the loudness contour.",
     STATS8, "spectral"),
    ("spectral_flux", "Spectral flux",
     "Euclidean norm of the frame-to-frame difference of the magnitude spectrum. It measures how fast the spectrum is changing.",
     STATS8, "spectral"),
    ("spectral_entropy", "Spectral entropy",
     "Shannon entropy of the power spectrum normalised to its own bin count (Eq.~\\ref{eq:specent}). It is a scale-free measure of how evenly energy is distributed across frequency.",
     STATS8, "spectral"),
    ("spectral_slope", "Spectral slope",
     "Least-squares slope of magnitude regressed on frequency, per frame. This is the spectral tilt.",
     STATS8, "spectral"),
    ("spectral_band_", "Band-energy ratios (8 bands)",
     "Fraction of frame energy in each of 0--500, 500--1000, 1000--2000, 2000--3000, 3000--4000, 4000--5000, 5000--6000, 6000--8000\\,Hz. Energy-normalised, so these describe spectral \\emph{shape} rather than level.",
     STATS2, "spectral"),
    ("spectral_hf_ratio_", "High-frequency survival ratios",
     "Fraction of frame energy above 4 and above 6\\,kHz. The most direct probes of lossy-codec low-pass behaviour in the inventory.",
     STATS2, "spectral"),
    ("spectral_contrast_", "Spectral contrast (7 sub-bands)",
     "Per-band difference between spectral peaks and valleys in decibels. Band 6 covers 6--8\\,kHz and is the single most codec-displaced column in the table (Section~\\ref{sec:signal}).",
     STATS2, "spectral"),
    # ---- cepstral -----------------------------------------------------------
    ("mfcc_d1_", "$\\Delta$MFCC (20)",
     "First-order temporal derivative of each MFCC, over a 9-frame window.",
     STATS2, "mfcc"),
    ("mfcc_d2_", "$\\Delta\\Delta$MFCC (20)",
     "Second-order temporal derivative of each MFCC.",
     STATS2, "mfcc"),
    ("mfcc_", "MFCC (20)",
     "Discrete cosine transform of the log mel-filterbank energies, which is the standard SER front end. MFCC-0 tracks overall log energy, so its derivatives measure how fast loudness changes.",
     STATS8, "mfcc"),
    # ---- chroma -------------------------------------------------------------
    ("chroma_", "Chroma (12 pitch classes)",
     "Energy folded onto the twelve semitone pitch classes. Tuning is pinned to 0 rather than estimated, because an estimated tuning could itself shift under compression and become part of the effect being measured.",
     STATS2, "chroma"),
    # ---- prosody ------------------------------------------------------------
    ("prosody_f0_slope_abs_mean", "$F_0$ slope (mean $|\\cdot|$)",
     "Mean absolute first difference of the voiced $F_0$ contour, per second. It is the rate of pitch movement.",
     "scalar", "prosody"),
    ("prosody_f0_", "Fundamental frequency $F_0$",
     "Praat autocorrelation pitch over voiced frames only.",
     STATS8, "prosody"),
    ("prosody_jitter_", "Jitter (4 variants)",
     "Cycle-to-cycle variation in glottal period: local, RAP, PPQ5, DDP. A perturbation measure of the voice source.",
     "scalar", "prosody"),
    ("prosody_shimmer_", "Shimmer (6 variants)",
     "Cycle-to-cycle variation in glottal amplitude: local, local\\,dB, APQ3, APQ5, APQ11, DDA. APQ11 needs eleven consecutive periods and is the one column in the table with non-trivial missingness (11.3\\,\\%).",
     "scalar", "prosody"),
    ("prosody_hnr_", "Harmonics-to-noise ratio",
     "Ratio of periodic to aperiodic energy in decibels. It correlates with hoarseness and breathiness.",
     STATS5, "prosody"),
    ("prosody_f1_bw_median", "Formant bandwidths $B_1$--$B_3$",
     "Median $-3$\\,dB bandwidth of the first three formants (Burg estimator).",
     "median", "prosody"),
    ("prosody_f", "Formants $F_1$--$F_5$",
     "Burg-estimator formant frequencies, which describe the vocal-tract filter. $F_2$ movement is the strongest single correlate of disgust in this corpus.",
     "mean, std, median", "prosody"),
    ("prosody_intensity_", "Intensity",
     "Praat short-term intensity contour in decibels. Its standard deviation is among the top-ranked features for every model family in the study.",
     STATS8, "prosody"),
    ("prosody_voiced_fraction", "Voiced fraction",
     "Proportion of frames with a defined $F_0$.",
     "scalar", "prosody"),
    ("prosody_pause_fraction", "Pause fraction",
     "Complement of the voiced fraction. By definition it equals $1 -$ voiced fraction ($r = -1.00$).",
     "scalar", "prosody"),
    ("prosody_n_voiced_segments", "Voiced-segment count",
     "Number of maximal runs of voiced frames.",
     "scalar", "prosody"),
    ("prosody_voiced_rate_per_s", "Voiced-segment rate",
     "Voiced segments per second, which is a proxy for speech rate.",
     "scalar", "prosody"),
    ("prosody_mean_voiced_seg_s", "Mean voiced-segment duration",
     "Mean length in seconds of a voiced run.",
     "scalar", "prosody"),
    ("prosody_cpps", "CPPS",
     "Smoothed cepstral peak prominence, which is the standard acoustic correlate of breathy phonation. Drives both sadness and neutrality in every attribution method applied here.",
     "scalar", "prosody"),
    # ---- global -------------------------------------------------------------
    ("duration_s", "Clip duration",
     "Decoded duration in seconds. Rises by 32.68\\,ms under AAC encoder priming, uniformly across every clip.",
     "scalar", "global"),
    ("peak_amplitude", "Peak amplitude",
     "Maximum absolute sample value after loudness normalisation.",
     "scalar", "global"),
]

FAMILY_TITLES = OrderedDict([
    ("spectral", "Spectral descriptors"),
    ("mfcc", "Cepstral descriptors"),
    ("prosody", "Prosodic and voice-quality descriptors (Praat)"),
    ("chroma", "Chroma descriptors"),
    ("global", "Global descriptors"),
])


def assign(col: str) -> str:
    """Longest-prefix match of a column onto a descriptor row."""
    best = ""
    for prefix, *_ in DESCRIPTORS:
        if col.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return best


def main() -> None:
    path = latest(FEATURES, "features_*.parquet")
    df = pd.read_parquet(path, columns=None)
    cols = feature_names(df)
    counts: dict[str, int] = {}
    for c in cols:
        counts[assign(c)] = counts.get(assign(c), 0) + 1
    unassigned = [c for c in cols if not assign(c)]
    if unassigned:
        raise SystemExit(f"columns matched no descriptor: {unassigned[:10]}")

    fam_cols: dict[str, list[str]] = {k: [] for k in FAMILY_TITLES}
    for c in cols:
        f = family_of(c)
        fam_cols["global" if f == "other" else f].append(c)

    out: list[str] = []
    w = out.append

    w("% GENERATED BY make_appendix.py -- DO NOT EDIT BY HAND")
    w(r"\appendices")
    w("")
    w(r"\section{Complete Feature Inventory}\label{app:features}")
    w("")
    w(f"Every stimulus is reduced to {len(cols)} named columns. Frame-level "
      r"contours are summarised by eight order statistics (mean, standard "
      r"deviation, minimum, maximum, median, interquartile range, skewness, "
      r"excess kurtosis). Descriptors marked with a shorter list are summarised "
      r"only by those statistics, and scalars are already clip-level. Naming is "
      r"\ttf{<family>\_<descriptor>\_<statistic>} throughout, which is what makes "
      r"an attribution over a column a claim about an acoustic quantity.")
    w("")
    w(r"Analysis frames are \SI{512}{point} FFT, \SI{400}{sample} window, "
      r"\SI{160}{sample} hop at \SI{16}{\kilo\hertz}, which gives \SI{25}{\milli\second} "
      r"windows every \SI{10}{\milli\second}. Praat measures use a "
      r"\SIrange{75}{500}{\hertz} pitch range.")
    w("")

    # ---- Part 1: descriptor tables ----
    w(r"\subsection{Descriptors and their definitions}")
    w("")
    for fam, title in FAMILY_TITLES.items():
        rows = [d for d in DESCRIPTORS if d[4] == fam]
        n = sum(counts.get(d[0], 0) for d in rows)
        w(r"\begin{table}[!ht]")
        w(rf"\caption{{{title}. {n} columns of {len(cols)}.}}")
        w(rf"\label{{tab:app-{fam}}}")
        w(r"\centering\scriptsize\setlength{\tabcolsep}{3pt}")
        w(r"\begin{tabular}{@{}p{1.9cm}p{3.5cm}p{1.5cm}r@{}}")
        w(r"\toprule")
        w(r"Descriptor & Definition & Statistics & $n$\\")
        w(r"\midrule")
        for prefix, name, definition, stats, _ in rows:
            k = counts.get(prefix, 0)
            if k == 0:
                continue
            w(rf"{name} & {definition} & {stats} & {k}\\")
        w(r"\bottomrule")
        w(r"\end{tabular}")
        w(r"\end{table}")
        w("")

    # ---- Part 2: full enumeration ----
    w(r"\subsection{Enumerated column names}")
    w("")
    w(r"The complete inventory, in the order the extractor emits it.")
    w("")
    ncol = 4
    for fam, title in FAMILY_TITLES.items():
        names = fam_cols[fam]
        if not names:
            continue
        wide = len(names) > 30
        w(r"\begin{table*}[!ht]" if wide else r"\begin{table}[!ht]")
        w(rf"\caption{{{title}: all {len(names)} column names.}}")
        w(rf"\label{{tab:app-list-{fam}}}")
        nc = ncol if wide else 3
        w(r"\centering\tiny\setlength{\tabcolsep}{3pt}")
        w(r"\begin{tabular}{@{}" + "l" * nc + r"@{}}")
        w(r"\toprule")
        rows = (len(names) + nc - 1) // nc
        grid = [names[i * rows:(i + 1) * rows] for i in range(nc)]
        for r in range(rows):
            cells = [rf"\texttt{{{esc(g[r])}}}" if r < len(g) else "" for g in grid]
            w(" & ".join(cells) + r"\\")
        w(r"\bottomrule")
        w(r"\end{tabular}")
        w(r"\end{table*}" if wide else r"\end{table}")
        w("")

    dest = HERE / "appendix_features.tex"
    dest.write_text("\n".join(out) + "\n")
    print(f"wrote {dest}  ({len(cols)} columns, "
          + ", ".join(f"{k}={len(v)}" for k, v in fam_cols.items()) + ")")
    print(f"source: {path}")

    write_splits_appendix()


# ------------------------------------------------------- Appendix B: splits
COND_NAME = {"ref": "Uncompressed WAV", "mp3_64": "MP3 @ 64 kbit/s",
             "mp4_aac64": "MP4/AAC @ 64 kbit/s",
             "roundtrip_wav": "Roundtrip WAV (AAC re-wrapped)"}


# Depth to which Appendix B enumerates. The tree is grown to depth 6 for the
# profile of Section~\ref{sec:depth}; four levels is 15 internal nodes at full
# occupancy, which covers the range over which any pair of conditions still
# shares structure and stays printable.
APPENDIX_DEPTH = 4


def write_splits_appendix() -> None:
    """Every internal node to depth APPENDIX_DEPTH, for all four conditions.

    Read from the depth-profile run rather than the depth-3 run of
    ``per_condition.py``. CART is greedy and both runs use the same
    ``min_samples_leaf``, so the shallower tree's nodes are exactly this
    table's depth 0 to 2 rows; nothing is restated differently, only extended.
    """
    import pandas as pd

    from xai_ser.paths import OUTPUTS

    run = sorted((OUTPUTS / "per_condition").glob("depth_2*"))[-1]
    nodes = pd.read_csv(run / "node_table.csv")
    nodes = nodes[nodes["depth"] <= APPENDIX_DEPTH]

    def sort_key(p: str) -> tuple:
        p = "" if p == "(root)" else p
        return (len(p), p)

    out: list[str] = []
    w = out.append
    w("% GENERATED BY make_appendix.py -- DO NOT EDIT BY HAND")
    w(r"\section{Per-Condition Tree Splits}\label{app:splits}")
    w("")
    w(rf"The tables below list every internal node down to depth "
      rf"{APPENDIX_DEPTH} of the entropy tree of Section~\ref{{sec:trees}}. The "
      r"tree was fitted independently on each condition, using the same actors, "
      r"the same items and the same hyper-parameters. The \emph{Path} column "
      r"reads from the root, and \ttf{L} denotes the branch taken when the "
      r"parent's test is satisfied. Entropies are given in bits. The value $n$ "
      r"is the raw sample count, and the information gain is computed on the "
      r"class-weighted counts (Eq.~\ref{eq:ig}).")
    w("")
    w(r"A path missing from a table is one the tree stopped splitting, because "
      r"a child would have fallen below the 25-sample leaf minimum. Comparing "
      r"the same path across the four tables is the node-level comparison "
      r"summarised in Table~\ref{tab:depth} and Fig.~\ref{fig:depth}.")
    w("")

    counts = {}
    for cond in ("ref", "mp3_64", "mp4_aac64", "roundtrip_wav"):
        sub = nodes[nodes["condition"] == cond].copy()
        sub = sub.sort_values("path", key=lambda s: s.map(sort_key))
        counts[cond] = len(sub)
        w(r"\begin{table}[!ht]")
        w(rf"\caption{{Entropy tree fitted on \textbf{{{COND_NAME[cond]}}}, "
          rf"internal nodes to depth {APPENDIX_DEPTH}.}}")
        w(rf"\label{{tab:app-splits-{cond.replace('_', '-')}}}")
        w(r"\centering\scriptsize\setlength{\tabcolsep}{3pt}")
        w(r"\begin{tabular}{@{}llrrrr@{}}")
        w(r"\toprule")
        w(r"Path & Feature & Threshold & $H$ & IG & $n$\\")
        w(r"\midrule")
        for _, n in sub.iterrows():
            path = r"\textit{root}" if n["path"] == "(root)" else f"\\texttt{{{n['path']}}}"
            w(rf"{path} & \texttt{{{esc(n['feature'])}}} & "
              rf"{n['threshold']:.4f} & {n['entropy_bits']:.4f} & "
              rf"{n['information_gain_bits']:.4f} & {int(n['n_samples'])}\\")
        w(r"\bottomrule")
        w(r"\end{tabular}")
        w(r"\end{table}")
        w("")

    dest = HERE / "appendix_splits.tex"
    dest.write_text("\n".join(out) + "\n")
    print(f"wrote {dest}  (source {run.name}, internal nodes to depth "
          f"{APPENDIX_DEPTH}: " + ", ".join(f"{k}={v}" for k, v in counts.items()) + ")")


if __name__ == "__main__":
    main()
