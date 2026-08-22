"""Probe the Azure AI Foundry audio endpoint before committing the API arm to it.

Answers four questions, in order:
  1. Does auth work, and does the model name resolve?
  2. Does it accept audio at all?
  3. WHICH containers does it accept?  <- the one that decides whether the
     container-vs-codec contrast (SQ2) is testable on this backend.
  4. Does it return a usable one-word emotion label?

Reads credentials from .env (gitignored). Prints no secrets.
"""
import base64, json, os, sys, time, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_env(path=os.path.join(ROOT, ".env")):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


ENV = load_env()
BASE = ENV["FOUNDRY_AUDIO_BASE_URL"].rstrip("/")
KEY = ENV["FOUNDRY_AUDIO_API_KEY"]
MODEL = ENV["FOUNDRY_AUDIO_MODEL"]

ITEM = os.path.join(ROOT, "data", "stimuli", "1005_MTI_ANG_XX")
EMOTION_PROMPT = (
    "Listen to this audio. The speaker's emotion is one of: angry, disgusted, "
    "fearful, happy, neutral, sad. Respond with exactly one word from that list "
    "and nothing else."
)


def post(path, payload, auth="bearer", timeout=120):
    url = f"{BASE}{path}"
    headers = {"Content-Type": "application/json"}
    if auth == "bearer":
        headers["Authorization"] = f"Bearer {KEY}"
    else:
        headers["api-key"] = KEY
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:600]
        try:
            body = json.loads(body)
        except Exception:
            pass
        return e.code, body
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def audio_msg(prompt, data, fmt):
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {"data": data, "format": fmt}},
        ],
    }]


def text_of(resp):
    try:
        return (resp["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None


print(f"endpoint : {BASE}")
print(f"model    : {MODEL}")
print(f"key      : ...{KEY[-6:]} (len {len(KEY)})\n")

# --- 1. auth + model resolution (text only) ---------------------------------
print("=" * 70)
print("STEP 1  auth + model resolution (text-only round trip)")
print("=" * 70)
# gpt-audio-1.5 rejects text-only requests ("requires that either input content
# or output modality contain audio"), so the auth probe must itself carry audio.
auth_mode = None
_probe_audio = b64(os.path.join(ITEM, "ref.wav"))
for mode in ("bearer", "api-key"):
    status, resp = post("/chat/completions", {
        "model": MODEL,
        "modalities": ["text"],
        "messages": audio_msg("Reply with the single word: ok", _probe_audio, "wav"),
        "max_completion_tokens": 16,
    }, auth=mode)
    print(f"  auth={mode:<8} HTTP {status}  ->  {str(text_of(resp) or resp)[:220]}")
    if status == 200:
        auth_mode = mode
        break

if auth_mode is None:
    print("\nFAILED: no auth mode produced a 200. Stopping.")
    sys.exit(1)
print(f"\n  working auth header: {auth_mode}")

# --- 2/3. container acceptance ----------------------------------------------
print("\n" + "=" * 70)
print("STEP 2+3  container acceptance  (THE decisive test for SQ2)")
print("=" * 70)

CASES = [
    ("ref.wav",           "wav"),
    ("roundtrip_wav.wav", "wav"),
    ("mp3_64.mp3",        "mp3"),
    ("mp4_aac64.mp4",     "mp4"),
    ("mp4_aac64.mp4",     "m4a"),
    ("mp4_aac64.mp4",     "aac"),
    ("mp4_aac64.mp4",     "wav"),   # deliberate mislabel: is `format` a hint or authoritative?
]

results = []
for fname, fmt in CASES:
    path = os.path.join(ITEM, fname)
    if not os.path.exists(path):
        print(f"  {fname:<20} format={fmt:<5} SKIP (missing)")
        continue
    status, resp = post("/chat/completions", {
        "model": MODEL,
        "modalities": ["text"],
        "messages": audio_msg(EMOTION_PROMPT, b64(path), fmt),
        "max_completion_tokens": 32,
    }, auth=auth_mode)
    out = text_of(resp)
    if status == 200:
        verdict = "ACCEPTED"
        detail = repr(out)
    else:
        verdict = "REJECTED"
        err = resp
        if isinstance(err, dict):
            err = err.get("error", err)
            err = err.get("message", err) if isinstance(err, dict) else err
        detail = str(err)[:200]
    print(f"  {fname:<20} format={fmt:<5} HTTP {status}  {verdict:<9} {detail}")
    results.append((fname, fmt, status, verdict, out))
    time.sleep(1.0)

# --- 4. summary --------------------------------------------------------------
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
ok = {(f, fm) for f, fm, s, v, o in results if v == "ACCEPTED"}
wav_ok = any(fm == "wav" and f.endswith(".wav") for f, fm in ok)
mp3_ok = any(fm == "mp3" for f, fm in ok)
mp4_ok = any(fm in ("mp4", "m4a", "aac") for f, fm in ok)

print(f"  WAV accepted        : {wav_ok}")
print(f"  MP3 accepted        : {mp3_ok}")
print(f"  MP4/AAC accepted    : {mp4_ok}")
print()
if mp4_ok and wav_ok:
    print("  -> SQ2 testable as designed: mp4_aac64 vs roundtrip_wav.")
elif mp3_ok and wav_ok:
    print("  -> SQ2 testable via the MP3 route only: mp3_64 vs roundtrip_wav_mp3")
    print("     (roundtrip_wav_mp3 needs generating; it is already an approved amendment).")
else:
    print("  -> SQ2 NOT testable on this backend. API arm would add nothing.")

json.dump(
    [{"file": f, "format": fm, "http": s, "verdict": v, "output": o} for f, fm, s, v, o in results],
    open(os.path.join(ROOT, "research", "phase2_investigation", "foundry_audio_probe.json"), "w"),
    indent=1,
)
print("\n  saved: research/phase2_investigation/foundry_audio_probe.json")
