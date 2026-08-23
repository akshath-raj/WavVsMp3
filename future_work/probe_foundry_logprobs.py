"""Decisive probe: can we extract a GRADED emotion signal from gpt-audio-1.5
even though its argmax is stuck on `neutral`?

Gate G0 failed on this backend (12/12 -> neutral, 1 of 6 labels used). Under a
label-only readout that kills the study: every occlusion mask returns `neutral`,
the attribution vector is constant, and Spearman rho is undefined (DA CP2, C2).

But logprobs are supported. If probability mass over the emotion labels moves
with the input while the argmax does not, the study is not only rescued -- the
degenerate argmax becomes the point. That is the accuracy/explanation
dissociation measured directly.

Three readouts are tested, cheapest first:
  A. free 6-way, dump every token position's top_logprobs
  B. forced binary (gold vs neutral) -- puts two labels in direct competition
  C. does the graded signal MOVE across format conditions?
"""
import base64, json, math, os, time, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_env(path=os.path.join(ROOT, ".env")):
    env = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


ENV = load_env()
BASE, KEY, MODEL = (
    ENV["FOUNDRY_AUDIO_BASE_URL"].rstrip("/"),
    ENV["FOUNDRY_AUDIO_API_KEY"],
    ENV["FOUNDRY_AUDIO_MODEL"],
)
STIM = os.path.join(ROOT, "data", "stimuli")
GOLD = {"ANG": "angry", "DIS": "disgusted", "FEA": "fearful",
        "HAP": "happy", "NEU": "neutral", "SAD": "sad"}


def post(payload, timeout=120):
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        b = e.read().decode()[:300]
        try: b = json.loads(b)
        except Exception: pass
        return e.code, b
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def ask(path, fmt, prompt, max_tok=8, top=20):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return post({
        "model": MODEL, "modalities": ["text"],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {"data": data, "format": fmt}}]}],
        "max_completion_tokens": max_tok,
        "temperature": 0, "seed": 12345,
        "logprobs": True, "top_logprobs": top,
    })


def positions(resp):
    """Yield (index, chosen_token, [(token, logprob), ...]) per generated token."""
    lp = (resp["choices"][0].get("logprobs") or {}).get("content") or []
    for i, pos in enumerate(lp):
        alts = [(a["token"], a["logprob"]) for a in pos.get("top_logprobs", [])]
        yield i, pos.get("token"), alts


items = sorted(d for d in os.listdir(STIM) if os.path.isdir(os.path.join(STIM, d)))
ANGRY = [i for i in items if i.split("_")[2] == "ANG"]
SAD = [i for i in items if i.split("_")[2] == "SAD"]
print(f"items: {len(items)} | angry: {len(ANGRY)} | sad: {len(SAD)}\n")

SIX = ("Listen to this audio. The speaker's emotion is one of: angry, disgusted, "
       "fearful, happy, neutral, sad. Respond with exactly one word from that "
       "list and nothing else.")

# --- A. where do the emotion labels actually compete? -----------------------
print("=" * 72)
print("A  free 6-way: dump every token position (is any position label-bearing?)")
print("=" * 72)
it = ANGRY[0]
st, r = ask(os.path.join(STIM, it, "ref.wav"), "wav", SIX)
if st != 200:
    print(f"  HTTP {st}: {str(r)[:200]}")
else:
    print(f"  item={it} (gold=angry) content={r['choices'][0]['message']['content']!r}")
    for i, tok, alts in positions(r):
        shown = ", ".join(f"{t!r}:{lp:.2f}" for t, lp in alts[:8])
        print(f"   pos{i} chosen={tok!r:<12} top: {shown}")

# --- B. forced binary: put two labels in direct competition ------------------
print("\n" + "=" * 72)
print("B  forced binary (gold vs neutral): does mass separate the two labels?")
print("=" * 72)


def binary_prompt(a, b):
    return (f"Listen to this audio. Is the speaker {a} or {b}? "
            f"Answer with exactly one word: {a} or {b}.")


def label_mass(resp, labels):
    """Total probability on each label across the first label-bearing position."""
    for _, _, alts in positions(resp):
        m = {}
        for t, lp in alts:
            k = t.strip().lower()
            for lab in labels:
                if k == lab or (k and lab.startswith(k) and len(k) >= 3):
                    m[lab] = m.get(lab, 0.0) + math.exp(lp)
        if len(m) >= 2:          # a position where the labels genuinely compete
            return m
    return {}


probe_items = (ANGRY[:3] + SAD[:2])
for it in probe_items:
    gold = GOLD[it.split("_")[2]]
    st, r = ask(os.path.join(STIM, it, "ref.wav"), "wav", binary_prompt(gold, "neutral"))
    if st != 200:
        print(f"  {it:<22} HTTP {st} {str(r)[:120]}")
        continue
    out = (r["choices"][0]["message"]["content"] or "").strip()
    m = label_mass(r, [gold, "neutral"])
    if m:
        tot = sum(m.values()) or 1
        print(f"  {it:<22} gold={gold:<10} said={out!r:<12} "
              f"P({gold})={m.get(gold,0)/tot:.3f}  P(neutral)={m.get('neutral',0)/tot:.3f}")
    else:
        print(f"  {it:<22} gold={gold:<10} said={out!r:<12} (no competing position found)")
    time.sleep(0.8)

# --- C. does the graded signal MOVE across formats? -------------------------
print("\n" + "=" * 72)
print("C  does the graded signal move across format conditions?")
print("   (this is the whole study in miniature)")
print("=" * 72)
CONDS = [("ref.wav", "wav"), ("roundtrip_wav.wav", "wav"), ("mp3_64.mp3", "mp3")]
print(f"  {'item':<22} {'gold':<10} " + "  ".join(f"{c[0][:13]:>13}" for c in CONDS))
rows = []
for it in ANGRY[:4] + SAD[:2]:
    gold = GOLD[it.split("_")[2]]
    vals, labels_out = [], []
    for fname, fmt in CONDS:
        p = os.path.join(STIM, it, fname)
        st, r = ask(p, "wav" if fmt == "wav" else fmt, binary_prompt(gold, "neutral"))
        if st != 200:
            vals.append(float("nan")); labels_out.append(f"HTTP{st}"); continue
        m = label_mass(r, [gold, "neutral"])
        tot = sum(m.values()) or 1
        vals.append(m.get(gold, 0.0) / tot)
        labels_out.append((r["choices"][0]["message"]["content"] or "").strip().lower())
        time.sleep(0.8)
    rows.append({"item": it, "gold": gold, "p_gold": vals, "labels": labels_out})
    print(f"  {it:<22} {gold:<10} " + "  ".join(f"{v:>13.3f}" for v in vals)
          + ("   [argmax constant]" if len(set(labels_out)) == 1 else "   [ARGMAX MOVED]"))

fin = [r for r in rows if not any(math.isnan(v) for v in r["p_gold"])]
if fin:
    spreads = [max(r["p_gold"]) - min(r["p_gold"]) for r in fin]
    argmax_moved = sum(1 for r in fin if len(set(r["labels"])) > 1)
    print(f"\n  mean |P(gold)| spread across formats: {sum(spreads)/len(spreads):.4f}")
    print(f"  max spread on any item              : {max(spreads):.4f}")
    print(f"  items where the ARGMAX moved        : {argmax_moved}/{len(fin)}")
    print()
    if max(spreads) > 0.01:
        print("  -> GRADED SIGNAL IS LIVE. Probability mass moves with format even where")
        print("     the label does not. The degenerate argmax stops being fatal: it")
        print("     becomes the dissociation the study set out to measure.")
    else:
        print("  -> Graded signal is inert across formats. The API arm cannot")
        print("     detect format effects at this operating point.")

json.dump(rows, open(os.path.join(ROOT, "research", "phase2_investigation",
                                  "foundry_logprob_probe.json"), "w"), indent=1)
print("\n  saved: research/phase2_investigation/foundry_logprob_probe.json")
