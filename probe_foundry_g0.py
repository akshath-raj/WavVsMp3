"""Follow-up probe for gpt-audio-1.5: logprobs support, determinism, and a
mini gate-G0 accuracy check on the clean reference condition.

G0 is the blocking gate from the research protocol: if reference accuracy is at
the floor, no format manipulation has room to act. The Gemini smoke run and the
first Foundry probe both returned `neutral` for an `angry` item, which is the
failure signature Chen et al. (2025) predict for neutral-lexicon corpora. This
measures it properly instead of inferring it from n=1.
"""
import base64, json, os, time, urllib.error, urllib.request
from collections import Counter

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
LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad"]
PROMPT = (
    "Listen to this audio. The speaker's emotion is one of: angry, disgusted, "
    "fearful, happy, neutral, sad. Respond with exactly one word from that list "
    "and nothing else."
)
# CREMA-D encodes intended emotion in the filename: <id>_<sentence>_<EMO>_<intensity>
GOLD = {"ANG": "angry", "DIS": "disgusted", "FEA": "fearful",
        "HAP": "happy", "NEU": "neutral", "SAD": "sad"}


def post(payload, timeout=120):
    req = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:400]
        try:
            body = json.loads(body)
        except Exception:
            pass
        return e.code, body
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def call(path, fmt, extra=None):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    payload = {
        "model": MODEL,
        "modalities": ["text"],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "input_audio", "input_audio": {"data": data, "format": fmt}},
        ]}],
        "max_completion_tokens": 32,
    }
    payload.update(extra or {})
    return post(payload)


def parse(resp):
    try:
        txt = (resp["choices"][0]["message"]["content"] or "").strip().lower()
    except Exception:
        return None
    txt = txt.strip(".,!' \"")
    return txt if txt in LABELS else f"UNPARSED:{txt[:30]}"


items = sorted(d for d in os.listdir(STIM) if os.path.isdir(os.path.join(STIM, d)))
print(f"stimuli available: {len(items)} items\n")

# --- A. logprobs support ----------------------------------------------------
print("=" * 70)
print("A  logprobs support  (would upgrade attribution from label-flip to graded)")
print("=" * 70)
p = os.path.join(STIM, items[0], "ref.wav")
st, r = call(p, "wav", {"logprobs": True, "top_logprobs": 6})
if st == 200:
    lp = (r["choices"][0].get("logprobs") or {}).get("content")
    if lp:
        print(f"  SUPPORTED. first token top-{len(lp[0].get('top_logprobs', []))} alternatives:")
        for alt in lp[0].get("top_logprobs", [])[:6]:
            print(f"    {alt['token']!r:<16} logprob={alt['logprob']:.4f}")
    else:
        print("  HTTP 200 but no logprobs payload returned -> not available.")
else:
    err = r.get("error", r) if isinstance(r, dict) else r
    print(f"  NOT SUPPORTED (HTTP {st}): {str(err)[:180]}")

# --- B. determinism ---------------------------------------------------------
print("\n" + "=" * 70)
print("B  determinism  (temperature=0 + seed): does repeat-variance vanish?")
print("=" * 70)
outs = []
for i in range(3):
    st, r = call(p, "wav", {"temperature": 0, "seed": 12345})
    outs.append(parse(r) if st == 200 else f"HTTP{st}")
    time.sleep(0.8)
print(f"  3 identical requests -> {outs}")
print(f"  deterministic: {len(set(outs)) == 1}")

# --- C. mini gate G0 --------------------------------------------------------
N = 12
print("\n" + "=" * 70)
print(f"C  mini gate G0: reference-condition accuracy on {N} items (1 call each)")
print("=" * 70)
rows, preds = [], Counter()
for it in items[:N]:
    gold = GOLD.get(it.split("_")[2], "?")
    st, r = call(os.path.join(STIM, it, "ref.wav"), "wav", {"temperature": 0, "seed": 12345})
    pred = parse(r) if st == 200 else f"HTTP{st}"
    ok = pred == gold
    preds[pred] += 1
    rows.append({"item": it, "gold": gold, "pred": pred, "correct": ok})
    print(f"  {it:<22} gold={gold:<10} pred={str(pred):<12} {'OK' if ok else ''}")
    time.sleep(0.8)

n_ok = sum(r["correct"] for r in rows)
acc = n_ok / len(rows) if rows else 0
print(f"\n  accuracy: {n_ok}/{len(rows)} = {acc:.1%}")
print(f"  prediction distribution: {dict(preds)}")
print(f"  distinct labels used: {len({r['pred'] for r in rows})} of 6")
print()
if acc >= 0.40:
    print("  -> G0 PASSES the 40% bar (approx. human audio-only parity, Cao et al. 2014).")
elif len({r["pred"] for r in rows}) <= 2:
    print("  -> G0 FAILS and responses are near-DEGENERATE. This is the")
    print("     lexical-dominance collapse Chen et al. (2025) predict. Gate G0b would fire.")
else:
    print("  -> G0 FAILS the 40% bar, but responses are NOT degenerate.")
    print("     Headroom exists; consider a reduced label set rather than a model change.")

json.dump({"accuracy": acc, "n": len(rows), "rows": rows,
           "distribution": dict(preds)},
          open(os.path.join(ROOT, "research", "phase2_investigation",
                            "foundry_g0_probe.json"), "w"), indent=1)
print("\n  saved: research/phase2_investigation/foundry_g0_probe.json")
