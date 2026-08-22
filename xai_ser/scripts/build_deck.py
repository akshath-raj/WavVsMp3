#!/usr/bin/env python
"""Assemble the presentation as a single self-contained HTML file.

Figures are embedded as data URIs because the Artifact CSP blocks every external
host. Content is generated here rather than hand-written so the numbers on the
slides come from the same files the report cites.
"""

from __future__ import annotations

import base64
from pathlib import Path

from xai_ser.paths import OUTPUTS, ROOT

WEB = OUTPUTS / "presentation" / "web"
DEST = OUTPUTS / "presentation" / "deck.html"


def img(name: str) -> str:
    data = base64.b64encode((WEB / f"{name}.jpg").read_bytes()).decode()
    return f"data:image/jpeg;base64,{data}"


def figure(name: str, caption: str, alt: str, hero: bool = False) -> str:
    """`hero` figures span the slide and share it with prose, so they are capped
    tighter to keep the slide inside one viewport."""
    return (
        f'<figure class="fig{" hero" if hero else ""}">\n'
        f'  <img src="{img(name)}" alt="{alt}" loading="lazy" />\n'
        f'  <figcaption>{caption}</figcaption>\n'
        f'</figure>'
    )


CSS = """
:root {
  --ink:      #0D1418;
  --paper:    #F6F8F8;
  --surface:  #FFFFFF;
  --muted:    #64757E;
  --rule:     #D4DCE0;
  --wav:      #2E6F9E;
  --mp3:      #C4462E;
  --voice:    #3F8F5B;
  --ghost:    #E3E8EA;

  --display: "Helvetica Neue", Helvetica, Arial, sans-serif;
  --body: Charter, "Bitstream Charter", "Sitka Text", Cambria, Georgia, serif;
  --data: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --ink:     #E7EEF1;
    --paper:   #0D1418;
    --surface: #141E24;
    --muted:   #91A3AC;
    --rule:    #27353D;
    --wav:     #63A6D2;
    --mp3:     #E2755E;
    --voice:   #63B183;
    --ghost:   #1D272D;
  }
}
:root[data-theme="dark"] {
  --ink: #E7EEF1; --paper: #0D1418; --surface: #141E24; --muted: #91A3AC;
  --rule: #27353D; --wav: #63A6D2; --mp3: #E2755E; --voice: #63B183; --ghost: #1D272D;
}
:root[data-theme="light"] {
  --ink: #0D1418; --paper: #F6F8F8; --surface: #FFFFFF; --muted: #64757E;
  --rule: #D4DCE0; --wav: #2E6F9E; --mp3: #C4462E; --voice: #3F8F5B; --ghost: #E3E8EA;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--body);
  font-size: 17px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

.deck { scroll-snap-type: y mandatory; overflow-y: auto; height: 100svh; scroll-behavior: smooth; }
@media (prefers-reduced-motion: reduce) { .deck { scroll-behavior: auto; } }

.slide {
  scroll-snap-align: start;
  min-height: 100svh;
  padding: 0 clamp(20px, 5vw, 72px) 76px;
  display: flex; flex-direction: column;
  position: relative;
  border-bottom: 1px solid var(--rule);
}
.inner { width: 100%; max-width: 1180px; margin: 0 auto; flex: 1;
         display: flex; flex-direction: column; justify-content: center; gap: 22px; padding: 46px 0 0; }

/* ---- frequency-band rail -------------------------------------------------
   Eight segments matching the study's actual band edges. On slides about MP3
   the top bands drain out, which is literally what the codec does to them. */
.rail { position: absolute; inset: 0 0 auto; display: flex; height: 6px; gap: 2px;
        padding: 0 clamp(20px, 5vw, 72px); }
.rail span { flex: 1; background: var(--wav); opacity: .30; }
.rail span:nth-child(1) { flex: 2.2; }
.rail span:nth-child(2) { flex: 1.4; }
.slide[data-band="mp3"] .rail span:nth-child(7) { background: var(--mp3); opacity: .22; }
.slide[data-band="mp3"] .rail span:nth-child(8) { background: var(--mp3); opacity: .85; }
.slide[data-band="full"] .rail span:nth-child(8) { opacity: .85; }

/* ---- type ---------------------------------------------------------------- */
.eyebrow {
  font-family: var(--data); font-size: 11.5px; letter-spacing: .16em;
  text-transform: uppercase; color: var(--muted); margin: 0;
}
h1 { font-family: var(--display); font-weight: 700; letter-spacing: -.032em;
     font-size: clamp(38px, 5.6vw, 72px); line-height: 1.02; margin: 0; text-wrap: balance; }
h2 { font-family: var(--display); font-weight: 700; letter-spacing: -.026em;
     font-size: clamp(27px, 3.5vw, 43px); line-height: 1.08; margin: 0; text-wrap: balance; }
h3 { font-family: var(--display); font-weight: 700; letter-spacing: -.012em;
     font-size: 16px; margin: 0 0 6px; text-transform: none; }
p { margin: 0; max-width: 68ch; }
.lede { font-size: clamp(18px, 1.6vw, 21px); color: var(--ink); max-width: 62ch; }
.small { font-size: 14.5px; color: var(--muted); max-width: 74ch; }
strong { font-weight: 600; }
em.term { font-style: normal; font-family: var(--data); font-size: .92em; }

.wav { color: var(--wav); }
.mp3 { color: var(--mp3); }
.voice { color: var(--voice); }

/* ---- components ---------------------------------------------------------- */
.cols { display: grid; gap: 26px; grid-template-columns: repeat(auto-fit, minmax(258px, 1fr)); }
.cols.two { grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); }
.split { display: grid; gap: 30px; grid-template-columns: minmax(0, 1.35fr) minmax(0, 1fr); align-items: center; }
@media (max-width: 900px) { .split { grid-template-columns: 1fr; } }

.card { background: var(--surface); border: 1px solid var(--rule); padding: 17px 19px;
        display: flex; flex-direction: column; gap: 7px; }
.card .k { font-family: var(--data); font-size: 11px; letter-spacing: .14em;
           text-transform: uppercase; color: var(--muted); }
.card .v { font-family: var(--display); font-weight: 700; font-size: 30px;
           letter-spacing: -.02em; font-variant-numeric: tabular-nums; }
.card .n { font-size: 14px; color: var(--muted); line-height: 1.45; }

.stat-row { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }

table { border-collapse: collapse; width: 100%; font-family: var(--data);
        font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--rule); }
th { font-size: 10.5px; letter-spacing: .13em; text-transform: uppercase; color: var(--muted); font-weight: 400; }
td.num { text-align: right; }
.tablewrap { overflow-x: auto; }

.fig { margin: 0; display: flex; flex-direction: column; gap: 9px; }
.fig img { max-width: 100%; max-height: 46vh; width: auto; height: auto;
           display: block; margin: 0 auto; background: #fff;
           border: 1px solid var(--rule); }
.fig.hero img { max-height: 38vh; }
.fig figcaption { font-size: 13.5px; color: var(--muted); max-width: 82ch; }

.chain { display: flex; flex-wrap: wrap; align-items: stretch; gap: 10px; }
.step { flex: 1 1 160px; border: 1px solid var(--rule); background: var(--surface); padding: 13px 15px; }
.step .t { font-family: var(--data); font-size: 12px; letter-spacing: .1em;
           text-transform: uppercase; color: var(--muted); }
.step .d { font-family: var(--display); font-weight: 700; font-size: 17px; margin-top: 3px; letter-spacing: -.01em; }
.step .s { font-size: 13.5px; color: var(--muted); margin-top: 5px; line-height: 1.45; }
.step.is-wav { border-left: 3px solid var(--wav); }
.step.is-mp3 { border-left: 3px solid var(--mp3); }

ul.tight { margin: 0; padding-left: 1.1em; max-width: 70ch; }
ul.tight li { margin-bottom: 7px; }
ul.plain { list-style: none; margin: 0; padding: 0; }

.ref { display: grid; grid-template-columns: auto 1fr; gap: 10px 15px; align-items: baseline;
       padding: 9px 0; border-bottom: 1px solid var(--rule); }
.ref .tag { font-family: var(--data); font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase;
            color: var(--paper); background: var(--muted); padding: 2px 7px; white-space: nowrap; }
.ref .tag.core { background: var(--mp3); }
.ref p { font-size: 14px; line-height: 1.5; max-width: none; }
.ref .why { color: var(--muted); font-size: 13.5px; }

.pull { border-left: 3px solid var(--mp3); padding: 4px 0 4px 18px; }
.pull p { font-family: var(--display); font-weight: 700; letter-spacing: -.02em;
          font-size: clamp(19px, 2.2vw, 27px); line-height: 1.22; max-width: 26ch; }

.foot { position: absolute; left: clamp(20px, 5vw, 72px); right: clamp(20px, 5vw, 72px);
        bottom: 22px; display: flex; justify-content: space-between; align-items: baseline;
        font-family: var(--data); font-size: 11px; letter-spacing: .13em;
        text-transform: uppercase; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 11px; }
.foot span:last-child { padding-right: 82px; }
@media (max-width: 760px) { .foot span:last-child { padding-right: 0; } }

.nav { position: fixed; right: 16px; bottom: 16px; z-index: 20; display: flex; gap: 6px; }
.nav button { font-family: var(--data); font-size: 15px; line-height: 1; width: 34px; height: 34px;
              border: 1px solid var(--rule); background: var(--surface); color: var(--ink); cursor: pointer; }
.nav button:hover { border-color: var(--wav); color: var(--wav); }
.nav button:focus-visible { outline: 2px solid var(--wav); outline-offset: 2px; }

@media (max-width: 760px) {
  .deck { scroll-snap-type: none; height: auto; overflow: visible; }
  .slide { min-height: 0; padding-bottom: 60px; }
  .inner { justify-content: flex-start; }
  .nav { display: none; }
}
"""

JS = """
const deck = document.querySelector('.deck');
const slides = Array.from(document.querySelectorAll('.slide'));
function go(dir) {
  const y = deck.scrollTop;
  const next = dir > 0
    ? slides.find(s => s.offsetTop > y + 8)
    : [...slides].reverse().find(s => s.offsetTop < y - 8);
  if (next) deck.scrollTo({ top: next.offsetTop });
}
document.addEventListener('keydown', e => {
  if (['ArrowRight','ArrowDown','PageDown',' '].includes(e.key)) { e.preventDefault(); go(1); }
  if (['ArrowLeft','ArrowUp','PageUp'].includes(e.key)) { e.preventDefault(); go(-1); }
});
document.getElementById('prev').addEventListener('click', () => go(-1));
document.getElementById('next').addEventListener('click', () => go(1));
"""

RAIL = "<div class='rail'>" + "".join("<span></span>" for _ in range(8)) + "</div>"


def slide(section: str, n: int, total: int, body: str, band: str = "wav") -> str:
    return f"""
<section class="slide" data-band="{band}">
  {RAIL}
  <div class="inner">
{body}
  </div>
  <div class="foot"><span>{section}</span><span>{n:02d} / {total}</span></div>
</section>"""


def build() -> str:
    S: list[tuple[str, str, str]] = []   # (section, band, body)

    # ---------------------------------------------------------------- opening
    S.append(("", "full", f"""
    <p class="eyebrow">Explainable AI · Speech Emotion Recognition · Format Robustness</p>
    <h1>The same voice,<br />a different container.</h1>
    <p class="lede">What changes inside a model when a clip arrives as
      <span class="wav">lossless WAV</span> instead of <span class="mp3">64 kbps MP3</span> —
      measured not by what the model answers, but by <strong>what it looks at</strong>.</p>
    <div class="stat-row">
      <div class="card"><span class="k">Corpus</span><span class="v">7,442</span>
        <span class="n">CREMA-D clips, 91 actors, 6 acted emotions</span></div>
      <div class="card"><span class="k">Rendered</span><span class="v">29,768</span>
        <span class="n">audio files across four format conditions</span></div>
      <div class="card"><span class="k">Features</span><span class="v">436</span>
        <span class="n">named librosa + Praat descriptors per file</span></div>
      <div class="card"><span class="k">Explainers</span><span class="v">10</span>
        <span class="n">SHAP, LIME, permutation, PDP, surrogate, 6 gradient methods</span></div>
    </div>
    <p class="small">Dataset · preprocessing · feature engineering · EDA · literature review</p>"""))

    S.append(("The question", "mp3", """
    <p class="eyebrow">Framing</p>
    <h2>Accuracy is the wrong instrument for this question.</h2>
    <div class="split">
      <div style="display:flex;flex-direction:column;gap:16px">
        <p>Compression is not a research condition — it is what happens to every clip in every
          deployed pipeline. The question is not whether a lossy codec costs a model some points.
          It is whether the model, handed the <strong>same spoken content</strong> in a different
          container, still reaches its answer <strong>from the same acoustic evidence</strong>.</p>
        <p>Two models can score identically and disagree completely about why. A benchmark that
          reports only accuracy cannot see that, and cannot warn you about it.</p>
        <p class="small">This is why the classifier here is a decision tree, a forest, a boosted
          ensemble, an SVM and a small neural net over <em class="term">named</em> acoustic
          features — not an end-to-end black box. Every unit of evidence has an acoustic name, so
          "what moved" is answerable in the vocabulary of speech science.</p>
      </div>
      <div class="pull"><p>Does MP3 change the model's mind, or just its evidence?</p></div>
    </div>"""))

    # ------------------------------------------------------------ literature
    S.append(("Literature", "wav", """
    <p class="eyebrow">Literature review · 1 of 3</p>
    <h2>Codecs degrade speech models. That part is settled.</h2>
    <div class="cols two">
      <div class="ref"><span class="tag">Corpus</span><div>
        <p>Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., &amp; Verma, R.
          (2014). CREMA-D: Crowd-sourced emotional multimodal actors dataset.
          <em>IEEE Transactions on Affective Computing, 5</em>(4), 377–390.</p>
        <p class="why">The corpus paper. 7,442 clips, 91 actors, 12 fixed sentences, crowd-rated by
          2,443 raters. Reports human recognition from <strong>audio alone at 40.9%</strong> — the
          number that keeps this task honest.</p></div></div>
      <div class="ref"><span class="tag">Codec</span><div>
        <p>Reddy, A. P., &amp; Vijayarajan, V. (2020). Audio compression with multi-algorithm fusion
          and its impact in speech emotion recognition. <em>International Journal of Speech
          Technology, 23</em>(2), 277–285.</p>
        <p class="why">Establishes the baseline expectation: compression costs SER accuracy, and the
          cost is <strong>feature-dependent rather than uniform</strong>. Licenses an expectation,
          not a prediction — and says nothing about explanations.</p></div></div>
      <div class="ref"><span class="tag">Codec</span><div>
        <p>Wu, H., Chen, X., Lin, Y.-C., … Lee, H. (2024). <em>Codec-SUPERB @ SLT 2024: A
          lightweight benchmark for neural audio codec models</em> (arXiv:2409.14085).</p>
        <p class="why">The methodological anchor: <strong>signal-level fidelity metrics do not
          capture losses in paralinguistic content</strong>. You cannot infer behavioural damage
          from SNR or PESQ — you have to measure behaviour separately.</p></div></div>
      <div class="ref"><span class="tag">Standards</span><div>
        <p>Kapoor, S., Cantrell, E. M., Peng, K., … Narayanan, A. (2024). REFORMS: Consensus-based
          recommendations for machine-learning-based science. <em>Science Advances, 10</em>(18).</p>
        <p class="why">Adopted as the reporting standard — hence the speaker-independent splits,
          the recorded encoder arguments, and the per-stage attrition counts later in this deck.</p></div></div>
    </div>
    <p class="small">All references verified against Crossref DOI records or the arXiv API.</p>"""))

    S.append(("Literature", "wav", """
    <p class="eyebrow">Literature review · 2 of 3</p>
    <h2>The toolkit: attribution by perturbation, and its known limits.</h2>
    <div class="cols two">
      <div class="ref"><span class="tag">Method</span><div>
        <p>Ribeiro, M. T., Singh, S., &amp; Guestrin, C. (2016). "Why should I trust you?":
          Explaining the predictions of any classifier. <em>KDD '16</em>, 1135–1144.</p>
        <p class="why">LIME. A model can be probed purely by perturbing inputs and watching
          outputs — the licence under which any of this is possible on an opaque model.</p></div></div>
      <div class="ref"><span class="tag">Method</span><div>
        <p>Lundberg, S. M., &amp; Lee, S.-I. (2017). <em>A unified approach to interpreting model
          predictions</em> (arXiv:1705.07874). NeurIPS 2017.</p>
        <p class="why">SHAP's additive framework and consistency axioms. Used here as the primary
          global attribution — exact via TreeSHAP for the ensembles, and, as slide 21 shows,
          <strong>badly behaved when sampled</strong>.</p></div></div>
      <div class="ref"><span class="tag">Method</span><div>
        <p>Zeiler, M. D., &amp; Fergus, R. (2014). Visualizing and understanding convolutional
          networks. <em>ECCV 2014</em>, 818–833.</p>
        <p class="why">The origin of occlusion sensitivity: mask part of the input, watch the
          output move. Our feature-neutralisation test is this method on the feature axis.</p></div></div>
      <div class="ref"><span class="tag">Audio</span><div>
        <p>Haunschmid, V., Manilow, E., &amp; Widmer, G. (2020). <em>audioLIME: Listenable
          explanations using source separation</em> (arXiv:2008.00582).</p>
        <p class="why">Interpretability for audio must mean <strong>listenability</strong>;
          spectrogram patches are not audible objects. Our answer is to make every feature an
          acoustician's quantity — F0, jitter, CPPS, band energy — rather than a pixel region.</p></div></div>
      <div class="ref"><span class="tag">Audio</span><div>
        <p>Nasr, S., Ren, Z., &amp; Johnson, D. (2025). <em>Beyond saliency: Enhancing explanation
          of speech emotion recognition with expert-referenced acoustic cues</em> (arXiv:2511.11691).</p>
        <p class="why">Saliency imported from vision highlights regions without showing they are
          meaningful acoustic markers. The direct argument for expert-referenced features.</p></div></div>
      <div class="ref"><span class="tag">Audio</span><div>
        <p>Sotirou, T., Lyberatos, V., Menis Mastromichalakis, O., &amp; Stamou, G. (2024).
          <em>MusicLIME: Explainable multimodal music understanding</em> (arXiv:2409.10496).</p>
        <p class="why">Attribution can reveal <em>which information channel</em> a model leans on —
          the same move we make between voice evidence and codec artefact.</p></div></div>
    </div>"""))

    S.append(("Literature", "mp3", """
    <p class="eyebrow">Literature review · 3 of 3 · load-bearing</p>
    <h2>Explanations are fragile, and they can lie convincingly.</h2>
    <div class="split">
      <div style="display:flex;flex-direction:column;gap:14px">
        <div class="ref"><span class="tag core">Core</span><div>
          <p>Ghorbani, A., Abid, A., &amp; Zou, J. (2019). <em>Interpretation of neural networks is
            fragile</em> (arXiv:1710.10547). AAAI 2019.</p>
          <p class="why">Perceptually indistinguishable inputs, given the <strong>same predicted
            label</strong> — sometimes with higher confidence — receive substantially different
            interpretations. The accuracy/explanation dissociation, established in vision under
            <em>adversarially constructed</em> perturbation.</p></div></div>
        <div class="ref"><span class="tag core">Core</span><div>
          <p>Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., &amp; Kim, B. (2018).
            <em>Sanity checks for saliency maps</em> (arXiv:1810.03292). NeurIPS 2018.</p>
          <p class="why">Some widely used saliency methods are independent of the model <em>and</em>
            the data — plausible pictures that explain nothing, and the eye cannot tell. Any
            claim that an explanation moved must first prove the explanation was real.</p></div></div>
      </div>
      <div class="pull"><p>Ghorbani needed an adversary. A transcode is not an adversary — it is Tuesday.</p></div>
    </div>"""))

    S.append(("Positioning", "mp3", """
    <p class="eyebrow">The gap</p>
    <h2>Three literatures that have not been introduced.</h2>
    <div class="cols">
      <div class="card"><span class="k">Established</span>
        <span class="n" style="font-size:15px;color:var(--ink)">Codec robustness is measured on
        <strong>task-specific models</strong>, and reported as an accuracy delta.</span></div>
      <div class="card"><span class="k">Established</span>
        <span class="n" style="font-size:15px;color:var(--ink)">XAI for audio is developed on
        <strong>clean corpora</strong>, where the format is never a variable.</span></div>
      <div class="card"><span class="k">Established</span>
        <span class="n" style="font-size:15px;color:var(--ink)">Explanation fragility is shown under
        <strong>adversarial</strong> perturbation, in vision.</span></div>
    </div>
    <div class="pull"><p>Nobody has asked what an ordinary MP3 encode does to a model's explanation.</p></div>
    <p>So the design is: hold the spoken content fixed, vary only the container, and measure the
      <strong>attribution</strong> — across six model families and ten explainability methods, with
      Adebayo's sanity check run first so that a moving explanation means something.</p>"""))

    # ------------------------------------------------------------------ data
    S.append(("Dataset", "wav", f"""
    <p class="eyebrow">Dataset</p>
    <h2>CREMA-D: fixed sentences, acted emotion, crowd-rated.</h2>
    <div class="split">
      <div style="display:flex;flex-direction:column;gap:15px">
        <p>Twelve fixed sentences spoken by 91 actors in six emotions. The fixed lexicon is the
          reason this corpus suits the question: <strong>linguistic content is held constant</strong>,
          so acoustic effects can be isolated from what is being said.</p>
        <div class="tablewrap"><table>
          <tr><th>Property</th><th>Value</th></tr>
          <tr><td>Clips / actors / sentences</td><td class="num">7,442 · 91 · 12</td></tr>
          <tr><td>Emotions</td><td class="num">ANG DIS FEA HAP NEU SAD</td></tr>
          <tr><td>Class balance</td><td class="num">1,271 each; NEU 1,087</td></tr>
          <tr><td>Intensity labels</td><td class="num">6,077 unspecified · 455 each LO/MD/HI</td></tr>
          <tr><td>Actor sex</td><td class="num">48 M · 43 F</td></tr>
          <tr><td>Actor age</td><td class="num">20–74 (mean 36.4)</td></tr>
          <tr><td>Clip duration</td><td class="num">2.54 s ± 0.51</td></tr>
          <tr><td>Total audio</td><td class="num">5.26 hours</td></tr>
        </table></div>
      </div>
      {figure("01_corpus_overview",
              "Class counts, acted intensity, duration distribution, and duration by emotion. "
              "Near-balanced classes; neutral is smaller because it has no intensity variants.",
              "Four panels showing CREMA-D class balance, intensity counts and duration distributions")}
    </div>"""))

    S.append(("Dataset", "wav", f"""
    <p class="eyebrow">Dataset · label quality</p>
    <h2>The ceiling is low, and that is the honest context.</h2>
    <div class="split">
      {figure("03_human_ceiling",
              "Left: crowd voice-only confusion against the actor's intended emotion, row-normalised. "
              "Right: per-clip agreement among raters.",
              "Confusion matrix of human voice-only ratings and a histogram of rater agreement")}
      <div style="display:flex;flex-direction:column;gap:14px">
        <p>Hearing audio alone, CREMA-D's crowd matched the actor's intended emotion in
          <strong>45.5%</strong> of clips in our recomputation from the voice-only vote tabulation.
          Recall is wildly uneven: neutral 0.97 and anger 0.67, but <strong>sadness 0.16</strong> —
          listeners fall back on "neutral" when acted affect is subtle.</p>
        <p class="small">The corpus paper reports 40.9% for the same modality. The gap is a
          definitional one — we score the <em>modal</em> vote per clip against the intended label,
          the paper scores individual ratings. Both are reported rather than reconciled away.</p>
        <p class="small">Consequence for this study: absolute accuracy is not a meaningful measure
          of "emotion recognition" here, which is a further reason to put the weight on attribution.</p>
      </div>
    </div>"""))

    # --------------------------------------------------------- preprocessing
    S.append(("Preprocessing", "mp3", """
    <p class="eyebrow">Preprocessing · design</p>
    <h2>One canonical reference, then a controlled chain.</h2>
    <p>Every condition descends from the same normalised source, so any downstream difference is
      attributable to the codec and nothing else — not to loudness, sample rate, or channel count.</p>
    <div class="chain">
      <div class="step is-wav"><div class="t">Source</div><div class="d">CREMA-D WAV</div>
        <div class="s">7,442 clips pulled from the corpus' git-lfs media endpoint.</div></div>
      <div class="step is-wav"><div class="t">Condition · ref</div><div class="d">16 kHz mono PCM16</div>
        <div class="s">EBU R128 loudness-normalised (<em class="term">I=-23, LRA=7, TP=-2</em>). The lossless reference.</div></div>
      <div class="step is-mp3"><div class="t">Condition · mp3_64</div><div class="d">MP3 @ 64 kbps</div>
        <div class="s"><em class="term">libmp3lame</em>, encoded from ref. The primary contrast.</div></div>
      <div class="step is-mp3"><div class="t">Condition · mp4_aac64</div><div class="d">MP4/AAC @ 64 kbps</div>
        <div class="s">Audio-only MP4, for a second codec at matched bitrate.</div></div>
      <div class="step"><div class="t">Control · roundtrip_wav</div><div class="d">AAC decoded to WAV</div>
        <div class="s">Same codec output, different container — isolates container from codec.</div></div>
    </div>
    <p class="small">Decoding is uniform: every stimulus, whatever its container, is decoded to
      float32 PCM through the <em>same</em> ffmpeg call. Letting soundfile read the WAVs and
      audioread the MP3s would confound the codec effect with the decoder.</p>"""))

    S.append(("Preprocessing", "mp3", """
    <p class="eyebrow">Preprocessing · results</p>
    <h2>29,768 files, one failure, and it was already documented.</h2>
    <div class="split">
      <div class="tablewrap"><table>
        <tr><th>Condition</th><th class="num">Files</th><th class="num">Mean size</th><th class="num">Total</th><th class="num">Mean duration</th></tr>
        <tr><td><span class="wav">ref</span> — WAV</td><td class="num">7,442</td><td class="num">81.5 kB</td><td class="num">606 MB</td><td class="num">2.5429 s</td></tr>
        <tr><td><span class="mp3">mp3_64</span></td><td class="num">7,442</td><td class="num">21.4 kB</td><td class="num">159 MB</td><td class="num">2.5430 s</td></tr>
        <tr><td>mp4_aac64</td><td class="num">7,442</td><td class="num">21.6 kB</td><td class="num">160 MB</td><td class="num">2.5756 s</td></tr>
        <tr><td>roundtrip_wav</td><td class="num">7,442</td><td class="num">82.5 kB</td><td class="num">614 MB</td><td class="num">2.5756 s</td></tr>
      </table></div>
      <div style="display:flex;flex-direction:column;gap:14px">
        <p><strong>MP3 preserves duration to 0.15 ms.</strong> AAC does not: it adds
          <strong>+32.7 ms</strong> of encoder priming to every clip, uniformly. That padding shifts
          every frame-aligned feature — a mechanism that shows up later in the drift table and has
          nothing to do with audio quality.</p>
        <p>Extraction failed on exactly <strong>one clip in four conditions</strong>
          (<em class="term">1076_MTI_SAD_XX</em>, digitally silent) — the file CREMA-D's own README
          lists as having no audio. 4 rows of 29,768. Nothing else failed.</p>
        <p class="small">A pipeline that finds precisely the known-bad file, and nothing else, is a
          pipeline you can start trusting.</p>
      </div>
    </div>"""))

    S.append(("Preprocessing", "wav", """
    <p class="eyebrow">Preprocessing · the control that licenses everything else</p>
    <h2>How small is "no effect"? Measure it, don't assume it.</h2>
    <div class="split">
      <div style="display:flex;flex-direction:column;gap:15px">
        <p>The <em class="term">roundtrip_wav</em> condition carries the identical AAC codec output
          in a WAV wrapper. Any difference between it and <em class="term">mp4_aac64</em> is pure
          decode-path arithmetic — the WAV path passes through int16 quantisation, the direct MP4
          decode does not.</p>
        <p>That difference is the study's <strong>noise floor</strong>. Effects larger than it are
          real; effects smaller than it are plumbing.</p>
      </div>
      <div class="stat-row">
        <div class="card"><span class="k">Paired clips</span><span class="v">7,441</span>
          <span class="n">every usable clip, matched by id</span></div>
        <div class="card"><span class="k">Median |SMD|</span><span class="v">7.6e-5</span>
          <span class="n">difference between the two containers</span></div>
        <div class="card"><span class="k">Features &gt; 0.05</span><span class="v">12</span>
          <span class="n">of 436 — the entire measurable decode effect</span></div>
        <div class="card"><span class="k">Bit-identical</span><span class="v">0%</span>
          <span class="n">as expected: int16 quantisation differs</span></div>
      </div>
    </div>
    <p class="small">Reported honestly: the container null here is <em>structural</em> — once ffmpeg
      decodes both, it almost has to hold. It validates the measurement chain; it does not
      generalise to systems whose decode path you cannot see.</p>"""))

    # ---------------------------------------------------- feature engineering
    S.append(("Features", "wav", """
    <p class="eyebrow">Feature engineering</p>
    <h2>436 columns, every one with an acoustic name.</h2>
    <p>Frame-level contours are reduced to eight order statistics — mean, SD, min, max, median, IQR,
      skew, kurtosis — so the result is a plain tabular dataset a decision tree can consume and a
      phonetician can read.</p>
    <div class="cols">
      <div class="card"><span class="k">spectral_* · 114</span>
        <span class="n" style="color:var(--ink)">Centroid, bandwidth, rolloff 85/95%, flatness, flux,
        entropy, slope, ZCR, RMS, 7-band contrast, 8 band-energy ratios, and explicit
        high-frequency survival ratios above 4 and 6 kHz.</span>
        <span class="n"><strong>Why:</strong> this is the family a lossy codec attacks first.</span></div>
      <div class="card"><span class="k">mfcc_* · 240</span>
        <span class="n" style="color:var(--ink)">20 MFCCs with Δ and ΔΔ; full statistics on the
        static coefficients, mean and SD on the derivatives.</span>
        <span class="n"><strong>Why:</strong> the standard SER front end — included so results are
        comparable to the literature.</span></div>
      <div class="card"><span class="k">prosody_* · 56</span>
        <span class="n" style="color:var(--ink)">Praat F0 statistics and slope, jitter ×4,
        shimmer ×6, HNR, formants F1–F5 with bandwidths, intensity, voiced-segment timing, CPPS.</span>
        <span class="n"><strong>Why:</strong> the source–filter view — where emotion actually lives,
        and what we expect a codec to leave alone.</span></div>
      <div class="card"><span class="k">chroma_* · 24 + 2 global</span>
        <span class="n" style="color:var(--ink)">12 pitch classes, plus clip duration and peak
        amplitude.</span>
        <span class="n"><strong>Note:</strong> chroma tuning is pinned to 0 — estimating it is a
        music operation, and an <em>estimated</em> tuning could itself shift under compression and
        confound the contrast.</span></div>
    </div>"""))

    S.append(("Features", "wav", """
    <p class="eyebrow">Feature engineering · settings</p>
    <h2>The parameters, stated so the numbers can be reproduced.</h2>
    <div class="split">
      <div class="tablewrap"><table>
        <tr><th>Analysis</th><th>Setting</th></tr>
        <tr><td>Sample rate / channels</td><td class="num">16 kHz · mono</td></tr>
        <tr><td>FFT size</td><td class="num">512 (32 ms)</td></tr>
        <tr><td>Window / hop</td><td class="num">400 (25 ms) / 160 (10 ms)</td></tr>
        <tr><td>MFCC</td><td class="num">20 coefficients · 40 mel bands</td></tr>
        <tr><td>Mel range</td><td class="num">20 – 8000 Hz</td></tr>
        <tr><td>Praat pitch range</td><td class="num">75 – 500 Hz</td></tr>
        <tr><td>Max formant / count</td><td class="num">5500 Hz · 5</td></tr>
        <tr><td>Band edges (Hz)</td><td class="num">0·500·1k·2k·3k·4k·5k·6k·8k</td></tr>
      </table></div>
      <div style="display:flex;flex-direction:column;gap:14px">
        <p>The band edges are not arbitrary. The top two straddle the region where 64 kbps encoders
          begin discarding content, which makes those columns the most diagnostic features in the
          table — and, as it turns out, the most dangerous.</p>
        <p class="small">Two engineering notes worth recording. Praat's aggregate formant queries
          replace a per-frame Python loop — same numbers, ~50× fewer interpreter round trips.
          And <em class="term">librosa.estimate_tuning</em> segfaulted on this machine, which is a
          second reason tuning is pinned rather than estimated.</p>
        <p class="small">Extraction: 29,768 files in 10 min 20 s on 10 workers.</p>
      </div>
    </div>"""))

    # ------------------------------------------------------------------- EDA
    S.append(("EDA", "wav", f"""
    <p class="eyebrow">EDA · 1 of 4 · table health and redundancy</p>
    <h2>Before any model: is the table sound?</h2>
    <div class="cols two">
      {figure("04_table_health",
              "Left: the only features with missing values. Right: features with the heaviest tails.",
              "Bar charts of missing-value rates and heavy-tail rates per feature")}
      {figure("07_correlation",
              "Correlation among the 30 most discriminative features. The blocks are real structure "
              "— MFCC statistics and intensity statistics each move together.",
              "Correlation heatmap of the thirty most discriminative features")}
    </div>
    <div class="cols">
      <p class="small">No constant columns. Only <strong>5 of 436</strong> features contain any
        missing value, and one exceeds 1%: <em class="term">prosody_shimmer_apq11</em> at 11.3% —
        expected, since APQ11 needs eleven consecutive glottal periods and short or creaky clips do
        not supply them.</p>
      <p class="small">Heavy tails are mild: the worst feature has 0.89% of values beyond
        |z| &gt; 5. Redundancy is contained — 49 pairs correlate above 0.95, 7 above 0.99, several
        of them Praat's own algebraic duplicates (<em class="term">jitter_rap</em> vs
        <em class="term">jitter_ddp</em>, r = 1.00). Mean absolute off-diagonal correlation: 0.18.</p>
    </div>"""))

    S.append(("EDA", "wav", f"""
    <p class="eyebrow">EDA · 2 of 4 · what carries emotion</p>
    <h2>Emotion lives in the dynamics, not the levels.</h2>
    <div class="cols two">
      {figure("05_separability",
              "Left: the 20 strongest features by ANOVA F against emotion. Right: mean mutual "
              "information with emotion, by family.",
              "Bar chart of top features by ANOVA F and mean mutual information per feature family")}
      {figure("06_top_feature_distributions",
              "The four strongest individual features, by emotion. Anger and neutral separate on "
              "loudness variability; the confusable classes overlap heavily.",
              "Violin plots of the four most discriminative features split by emotion")}
    </div>
    <div class="cols">
      <p class="small"><strong>397 of 436</strong> features separate emotion at Bonferroni-corrected
        significance. The strongest are all <em>variability</em> measures —
        <em class="term">mfcc_d1_00_std</em> (F = 1194), <em class="term">mfcc_00_std</em>
        (F = 1159), <em class="term">prosody_intensity_std</em> (F = 1078). Translated:
        <strong>how much the loudness moves</strong> carries more about acted emotion than any
        average level does.</p>
      <p class="small">Per feature, MFCCs are the <em>weakest</em> family by mutual information
        (0.061) despite holding 55% of the columns; chroma (0.098) and prosody (0.087) lead. MFCC
        dominance downstream is a matter of count, not quality — a point the attribution results
        return to.</p>
    </div>"""))

    S.append(("EDA", "wav", f"""
    <p class="eyebrow">EDA · 3 of 4 · the confound that had to be handled</p>
    <h2>Speaker identity outweighs emotion in 340 of 436 features.</h2>
    <div class="split">
      {figure("10_variance_decomposition",
              "Variance explained per feature by speaker identity (x) against emotion (y). "
              "Points below the diagonal are features that describe the voice, not the feeling.",
              "Scatter plot of speaker eta-squared against emotion eta-squared per feature")}
      <div style="display:flex;flex-direction:column;gap:14px">
        <p>Mean variance explained: <strong>speaker η² = 0.178</strong> against
          <strong>emotion η² = 0.098</strong>. Prosody (0.232) and chroma (0.254) are the most
          speaker-bound families of all.</p>
        <p>This is why every split in the study partitions <strong>actors</strong>, never rows:
          60 train / 11 validation / 20 test speakers, with no voice appearing on both sides.</p>
        <div class="pull"><p>A random row split would have measured voice recognition and called it emotion.</p></div>
      </div>
    </div>"""))

    S.append(("EDA", "mp3", f"""
    <p class="eyebrow">EDA · 4 of 4 · structure</p>
    <h2>High-dimensional, and format is not a visible cluster.</h2>
    <div class="cols two">
      {figure("08_pca",
              "PCA scree, then PC1–PC2 coloured by emotion and by format condition.",
              "PCA cumulative variance curve and two scatter plots coloured by emotion and format")}
      {figure("09_tsne",
              "t-SNE on 2,500 sampled rows, coloured by emotion and by format condition.",
              "Two t-SNE embeddings coloured by emotion and by format condition")}
    </div>
    <p><strong>51 principal components</strong> are needed for 95% of the variance (PC1 21.7%,
      PC2 8.2%) — the space is genuinely high-dimensional, not a few latent factors in disguise.</p>
    <p>Note what the right-hand panels do <em>not</em> show: the format conditions sit on top of one
      another. At the level of gross geometry, an MP3 clip looks like its WAV original. The damage is
      not visible here — it is concentrated in a handful of directions, which is exactly why it took
      a paired, per-feature test to find.</p>"""))

    # ------------------------------------------------------------------- XAI
    S.append(("MP3 vs WAV", "mp3", f"""
    <p class="eyebrow">The signal · WAV vs MP3</p>
    <h2>The median feature is untouched. The tail is destroyed.</h2>
    {figure("P1_signal_drift_mp3",
            "Paired per-clip comparison against the WAV reference across all 7,441 usable clips. "
            "Left: the 14 most-shifted features under MP3 (note the symmetric-log axis). "
            "Right: the full distribution of absolute standardised shifts for both codecs.",
            "Bar chart of most-shifted features under MP3 and histogram of shift magnitudes", hero=True)}
    <div class="split">
      <p>Across 436 features the median shift under MP3 is <strong>0.020 SD</strong> — nothing. But
        <strong>28 features move by more than half a standard deviation</strong>, and one moves by
        <strong>38.6</strong>: <em class="term">spectral_contrast_6_mean</em>, the 6–8 kHz band,
        goes from 16.4 to 46.4. Spectral flatness collapses alongside it as the encoder zeroes
        masked bins.</p>
      <div class="pull"><p>These are not degradations. They are the encoder's fingerprint.</p></div>
    </div>"""))

    S.append(("XAI", "mp3", f"""
    <p class="eyebrow">XAI · the central result</p>
    <h2>The network stops listening to the voice.</h2>
    {figure("P2_attribution_shift_mp3",
            "Integrated gradients over the held-out speakers, WAV against MP3, log–log. "
            "Right: rank migration for the displaced and promoted features.",
            "Scatter of attribution on WAV versus MP3, and a slope chart of feature rank changes", hero=True)}
    <div class="split">
      <p>Attribution rank correlation falls to <strong>ρ = 0.677</strong> and only
        <strong>11 of the top 25</strong> features survive. What <em class="mp3">enters</em> the top
        25 is almost entirely codec artefact — five <em class="term">spectral_flatness_*</em>
        statistics and <em class="term">spectral_contrast_6_mean</em>, which climbs to
        <strong>rank #1</strong>. What <em class="voice">leaves</em> is voice evidence:
        <em class="term">prosody_cpps</em> (breathiness), <em class="term">prosody_f0_skew</em>,
        <em class="term">prosody_f2_mean</em> and <em class="term">prosody_f2_std</em>.</p>
      <div class="pull"><p>Same words, same speaker, same emotion. Different evidence.</p></div>
    </div>"""))

    S.append(("XAI", "mp3", f"""
    <p class="eyebrow">XAI · by model family</p>
    <h2>Whose explanation survives MP3? Almost everyone's but the network's.</h2>
    {figure("P3_stability_by_family",
            "Attribution-ranking stability against the WAV reference under MP3. SHAP for the "
            "classical models, integrated gradients for the neural network.",
            "Bar charts of Spearman rank correlation and top-25 retention per model", hero=True)}
    <div class="split">
      <p>Every tree and linear model keeps ρ ≥ 0.987 and at least 22 of its top 25 features. The
        neural network keeps 11. The difference is architectural: a tree asks whether
        <em class="term">spectral_contrast_6_mean</em> is above a threshold, and 16.4 and 46.4 give
        the same answer. A network multiplies that 38-SD excursion straight through its first layer.</p>
      <div class="pull"><p>Explanation stability is a property of the model family, not of the codec.</p></div>
    </div>
    <p class="small">The caveat, against a tidy story: logistic regression keeps a near-identical
      SHAP ranking (ρ = 0.987) while its accuracy collapses by 33 points. A global ranking can
      report "nothing changed" while the classifier has stopped working.</p>"""))

    S.append(("XAI", "wav", f"""
    <p class="eyebrow">XAI · do the methods agree?</p>
    <h2>Mostly, no. And that is a finding about XAI, not about audio.</h2>
    {figure("P5_method_agreement",
            "Left: rank agreement between SHAP and the other attribution methods, per classical "
            "model. Right: agreement among the six gradient-based methods on the neural network.",
            "Two heatmaps of Spearman rank correlation between attribution methods", hero=True)}
    <div class="split">
      <p>SHAP tracks a model's <em>intrinsic</em> importance closely (ρ = 0.98 for the decision tree
        and the linear model) but LIME and permutation importance are nearly unrelated to it. On the
        decision tree, LIME and SHAP rank features at <strong>ρ = 0.008</strong> — no relationship
        at all — while still sharing 10 of their top 20.</p>
      <div style="display:flex;flex-direction:column;gap:12px">
        <p>The six gradient methods on the network, by contrast, agree at ρ ≥ 0.94 throughout.</p>
        <p class="small"><strong>KernelSHAP failed outright on the SVM</strong> and is excluded from
          every conclusion: 100 sampled coalitions cannot identify 436 unknowns, and the fit diverged
          to a maximum mean |SHAP| of 2.1 × 10¹² against a median of 5.3 × 10⁻⁴. The code now raises
          instead of returning it.</p>
      </div>
    </div>
    <p class="small">Consequence: only features surviving <em>all</em> methods should be called
      robust — <em class="term">mfcc_d1_00_std</em>, <em class="term">prosody_intensity_std</em>,
      <em class="term">duration_s</em>, <em class="term">prosody_cpps</em>,
      <em class="term">prosody_f0_median</em>.</p>"""))

    S.append(("XAI", "mp3", f"""
    <p class="eyebrow">XAI · from correlation to cause</p>
    <h2>If the artefact features are the mechanism, removing them should fix it.</h2>
    {figure("P4_neutralisation_mp3",
            "Features ranked by codec-induced shift, then progressively replaced with training "
            "medians. The dashed line is the clean-WAV control — what the masking itself costs.",
            "Line chart of balanced accuracy on MP3 as codec-shifted features are neutralised", hero=True)}
    <div class="split">
      <p>Neutralising a <strong>single feature</strong> — <em class="term">spectral_contrast_6_mean</em> —
        recovers 98% of the SVM's loss and 92% of the network's. Three features bring every affected
        model within a point of its clean-audio score. The tree curves are flat because they had
        nothing to recover.</p>
      <div style="display:flex;flex-direction:column;gap:12px">
        <p>Crucially, the clean-WAV control barely moves: those columns were carrying
          <strong>no emotional information</strong> in the first place.</p>
        <div class="pull"><p>Not lost information — a corrupted channel the model was reading.</p></div>
      </div>
    </div>"""))

    S.append(("XAI", "wav", """
    <p class="eyebrow">XAI · the check that comes first</p>
    <h2>Are these explanations real?</h2>
    <div class="split">
      <div style="display:flex;flex-direction:column;gap:15px">
        <p>Adebayo et al. showed that some attribution methods produce convincing maps that are
          independent of the trained model. If ours were among them, every result in this deck would
          be describing the input distribution and nothing else.</p>
        <p>So each method was run twice — once on the trained network, once on a randomly
          initialised network of identical architecture — and the two rankings compared.</p>
      </div>
      <div class="tablewrap"><table>
        <tr><th>Method</th><th class="num">ρ, trained vs random</th></tr>
        <tr><td>Saliency</td><td class="num">−0.011</td></tr>
        <tr><td>GradientSHAP</td><td class="num">0.135</td></tr>
        <tr><td>DeepLIFT</td><td class="num">0.161</td></tr>
        <tr><td>Integrated gradients</td><td class="num">0.163</td></tr>
        <tr><td>Input × gradient</td><td class="num">0.204</td></tr>
        <tr><td>Feature ablation</td><td class="num">0.206</td></tr>
      </table></div>
    </div>
    <p>All near zero. The attributions describe the trained model, not the data — so a change in
      them under MP3 is a change in the model's behaviour, which is the whole claim.</p>
    <p class="small">Independently, the per-emotion attributions land where speech science says they
      should: F0 median drives fear, CPPS drives sadness and neutrality, loudness dynamics drive
      anger. Nobody told the network any of that.</p>"""))

    # ------------------------------------------------------------- takeaways
    S.append(("Takeaways", "mp3", """
    <p class="eyebrow">Where this leaves us</p>
    <h2>Four things worth carrying out of the room.</h2>
    <div class="cols two">
      <div class="card"><span class="k">On the dissociation</span>
        <span class="n" style="color:var(--ink);font-size:15px">Ghorbani's strong form — same answer,
        different evidence — <strong>did not reproduce</strong> here. Everything that moved the
        attributions also moved the accuracy. What holds is the weaker claim: the accuracy delta
        <strong>systematically understates</strong> the attribution delta.</span></div>
      <div class="card"><span class="k">On architecture</span>
        <span class="n" style="color:var(--ink);font-size:15px">Robustness to MP3 is decided by
        whether a model <strong>thresholds</strong> its inputs or multiplies them — not by how
        accurate it is, and not by how many features it has.</span></div>
      <div class="card"><span class="k">On mechanism</span>
        <span class="n" style="color:var(--ink);font-size:15px">The failure is <strong>covariate
        shift on a handful of columns</strong>, not lost emotional content. Which means it is
        fixable at the feature level, cheaply, without retraining.</span></div>
      <div class="card"><span class="k">On method</span>
        <span class="n" style="color:var(--ink);font-size:15px">XAI methods disagreed with each
        other more than the codecs disagreed with each other. A single-method explanation of "which
        acoustic features matter" is <strong>not reproducible</strong>.</span></div>
    </div>
    <div class="pull"><p>Report the container. It is part of the method, not part of the plumbing.</p></div>
    <p class="small">Limits: acted rather than spontaneous emotion; one deterministic speaker split;
      two codecs at one bitrate; summary statistics that discard <em>when</em> in the clip evidence
      sits; and global attribution rankings that, as shown, can miss a total accuracy collapse.</p>"""))

    S.append(("References", "wav", """
    <p class="eyebrow">References · verified against Crossref DOI records or the arXiv API</p>
    <h2>Sources.</h2>
    <div class="cols two">
      <ul class="plain" style="display:flex;flex-direction:column;gap:11px;font-size:13.5px;line-height:1.5">
        <li>Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., &amp; Kim, B. (2018).
          <em>Sanity checks for saliency maps</em> (arXiv:1810.03292). arXiv.</li>
        <li>Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., &amp; Verma, R. (2014).
          CREMA-D: Crowd-sourced emotional multimodal actors dataset. <em>IEEE Transactions on
          Affective Computing, 5</em>(4), 377–390. https://doi.org/10.1109/TAFFC.2014.2336244</li>
        <li>Ghorbani, A., Abid, A., &amp; Zou, J. (2019). <em>Interpretation of neural networks is
          fragile</em> (arXiv:1710.10547). arXiv.</li>
        <li>Haunschmid, V., Manilow, E., &amp; Widmer, G. (2020). <em>audioLIME: Listenable
          explanations using source separation</em> (arXiv:2008.00582). arXiv.</li>
        <li>Kapoor, S., Cantrell, E. M., Peng, K., … Narayanan, A. (2024). REFORMS: Consensus-based
          recommendations for machine-learning-based science. <em>Science Advances, 10</em>(18).
          https://doi.org/10.1126/sciadv.adk3452</li>
        <li>Lundberg, S. M., &amp; Lee, S.-I. (2017). <em>A unified approach to interpreting model
          predictions</em> (arXiv:1705.07874). arXiv.</li>
      </ul>
      <ul class="plain" style="display:flex;flex-direction:column;gap:11px;font-size:13.5px;line-height:1.5">
        <li>Nasr, S., Ren, Z., &amp; Johnson, D. (2025). <em>Beyond saliency: Enhancing explanation of
          speech emotion recognition with expert-referenced acoustic cues</em> (arXiv:2511.11691). arXiv.</li>
        <li>Reddy, A. P., &amp; Vijayarajan, V. (2020). Audio compression with multi-algorithm fusion
          and its impact in speech emotion recognition. <em>International Journal of Speech
          Technology, 23</em>(2), 277–285. https://doi.org/10.1007/s10772-020-09689-9</li>
        <li>Ribeiro, M. T., Singh, S., &amp; Guestrin, C. (2016). "Why should I trust you?": Explaining
          the predictions of any classifier. In <em>Proceedings of the 22nd ACM SIGKDD International
          Conference on Knowledge Discovery and Data Mining</em> (pp. 1135–1144). ACM.
          https://doi.org/10.1145/2939672.2939778</li>
        <li>Sotirou, T., Lyberatos, V., Menis Mastromichalakis, O., &amp; Stamou, G. (2024).
          <em>MusicLIME: Explainable multimodal music understanding</em> (arXiv:2409.10496). arXiv.</li>
        <li>Wu, H., Chen, X., Lin, Y.-C., … Lee, H. (2024). <em>Codec-SUPERB @ SLT 2024: A lightweight
          benchmark for neural audio codec models</em> (arXiv:2409.14085). arXiv.</li>
        <li>Zeiler, M. D., &amp; Fergus, R. (2014). Visualizing and understanding convolutional
          networks. In <em>Computer Vision – ECCV 2014</em> (pp. 818–833). Springer.
          https://doi.org/10.1007/978-3-319-10590-1_53</li>
      </ul>
    </div>
    <p class="small">Every figure in this deck was produced by the committed pipeline; the underlying
      tables live in <em class="term">outputs/eda/</em>, <em class="term">outputs/xai/</em> and
      <em class="term">outputs/robustness/</em>.</p>"""))

    total = len(S)
    body = "\n".join(slide(sec, i + 1, total, b, band) for i, (sec, band, b) in enumerate(S))

    return f"""<title>The same voice, a different container</title>
<style>{CSS}</style>
<main class="deck">
{body}
</main>
<div class="nav">
  <button id="prev" type="button" aria-label="Previous slide">&#8249;</button>
  <button id="next" type="button" aria-label="Next slide">&#8250;</button>
</div>
<script>{JS}</script>
"""


if __name__ == "__main__":
    DEST.write_text(build())
    print(f"{DEST}  ({DEST.stat().st_size / 1024 / 1024:.2f} MB)")
