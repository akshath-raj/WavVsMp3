# Feasibility Addendum — Empirical Audit of the Existing Smoke Run
**Phase:** 1 (Scoping) | **Date:** 2026-08-17
**Trigger:** Devil's Advocate Checkpoint 1, "What's Missing" — *"A pilot-based power calculation.
The smoke run (180 calls) already contains the discordance information needed for it and has not
been used."*

The audit was run. The smoke run does **not** contain that information, because the smoke run
largely failed. This changes the Feasibility picture materially and is recorded here rather than
discovered later.

---

## What the audit found

Source: `data/results.parquet` (180 rows), `data/responses/*.json`, calls timestamped
2026-08-09T10:27Z → 2026-08-10T06:33Z.

| Outcome | Calls | Share |
|---|---:|---:|
| `api_error` — **HTTP 429, free-tier quota exceeded** | 152 | 84.4% |
| `api_error` — `ConnectionError` (transport) | 14 | 7.8% |
| `ok` — response received and parsed | **14** | **7.8%** |
| **Total** | 180 | 100% |

Grid coverage attempted: 5 items × 4 formats × 2 tasks × 3 repeats, plus the `emotion_v2`/`v3`
prompt-sensitivity cells on `ref` and `mp4_aac64` — i.e. the smoke run exercised the *design*
correctly. It was the *quota* that failed, not the harness.

Of the 14 successful calls, 8 were emotion trials, distributed across only the `ref` (5) and
`mp3_64` (3) conditions. Zero successful `mp4_aac64` or `roundtrip_wav` emotion trials survive.

**All 8 successful emotion trials returned an incorrect label** (observed accuracy 0/8; the model
returned `neutral` for several `angry` items). This must not be read as an accuracy estimate:
n = 8, drawn from 5 items, one condition-imbalanced sample, with a 95% CI on the true accuracy
spanning roughly 0–37%. It is recorded only as a flag that `ref`-condition performance may be low
enough to raise a floor-effect concern (see below).

---

## Consequences for the design

### F1 — The power calculation demanded by C1 cannot be performed yet (blocking)

C1's fix required deriving the achievable equivalence bound Δ from the smoke-run discordance rate
between `ref` and `mp4_aac64`. There are **zero usable paired observations** for that contrast.
The C1 remedy is therefore deferred, not satisfied: a *successful* smoke run of at least 10 items
× {`ref`, `mp4_aac64`} × 3 repeats must complete before Δ can be set, and Δ must be set before the
main grid runs.

### F2 — The full design is not feasible on the free tier

The revised call budget is ≈ 5,070 calls. The free tier exhausted quota after roughly 28
successful-or-attempted calls in the first window and stayed exhausted across a ~20-hour span.
Extrapolating even generously, the grid would take weeks of wall-clock and would be interleaved
with model-version drift across that period — which V4 was written to avoid, and which at that
timescale it cannot absorb. **Billing must be enabled on the API key, or the design must shrink.**
Projected spend at the smoke-run cost basis is ≈ $2.00, far below the $25 ceiling; this is a quota
policy constraint, not a cost constraint.

### F3 — A possible floor effect threatens the whole manipulation

If the model's accuracy on the *clean reference* condition is near chance (1/6 ≈ 16.7% for the
6-way forced choice), there is no headroom for compression to remove. Every contrast in the study
would then be a comparison between two floors, and SQ1–SQ3 would all return nulls for a reason
that has nothing to do with format. The DA's "What's Missing" item — no reference anchor for
whether `ref` accuracy is any good — becomes acute.

**Gate G0 (new, precedes all other gates):** establish `ref`-condition accuracy on ≥ 20 items × 3
repeats. If it does not clear a pre-declared floor of **40%** (well above chance, leaving ≥ 20
points of headroom), the study must not proceed to the full grid as designed. Remedies in order of
preference: (a) switch to a stronger audio model; (b) reduce the label set from 6-way to a 4-way
or binary high/low-arousal contrast, where the task is easier and headroom exists; (c) change task
to one the model demonstrably performs (the transcription arm, per M3, is the natural candidate) —
though this weakens the paralinguistic motivation.

---

## Revised FINER — Feasible

| | Before audit | After audit |
|---|---|---|
| **F**easible | 4/5 | **2/5** |

Justification for the downgrade: the harness, stimuli, and design are sound and complete, but the
study is currently **quota-blocked** and carries an unquantified floor-effect risk on its primary
outcome. Both are resolvable — enable billing, run G0 — and neither is a flaw in the design. The
score returns to 4/5 once G0 passes on a billed key. Average FINER falls to **3.6/5**, still above
the 3.0 threshold, with no criterion below 2.

---

## Revised gate order

```
G0  Reference-accuracy floor check  (≥40% on ref, ≥20 items × 3 repeats)   [NEW — blocks all]
G1  Signal-identity fidelity gate   (cross-correlation aligned; A2 / V1 / M1)
G2  Power calibration               (set Δ from real ref↔mp4 discordance; C1)
G3  Main grid                       (1,350 calls incl. roundtrip_wav_mp3)
G4  XAI arm                         (R=7; ~3,720 calls)
```

Each gate's failure is a reportable outcome with a pre-declared response, not a reason to relax
the gate.
