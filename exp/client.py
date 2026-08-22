"""Cached, resumable, concurrent client for Azure AI Foundry `gpt-audio-1.5`.

Design notes that matter for the science:

* `temperature=0` + fixed `seed`. The backend was verified bit-deterministic
  (5 identical calls -> identical logprobs to 3 d.p., range 0.0000), so one call
  per cell is sufficient and repeats would be pure waste.

* Label probability is read from `top_logprobs` with a mandatory filter:
  **only tokens with non-null `bytes` are counted.** The raw distribution puts
  most of its mass on non-emittable special tokens (15 distinct ids all
  rendering as `<|end|>`), which would otherwise deflate every probability.
  See research/phase2_investigation/o1_resolution.json.

* Cache key is a hash of everything that could change the answer: model, audio
  bytes, prompt, decode params. Re-runs are free and the grid is resumable.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "exp" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

_ENV = {}
for _line in (ROOT / ".env").read_text().splitlines():
    _line = _line.strip()
    if _line and not _line.startswith("#") and "=" in _line:
        _k, _v = _line.split("=", 1)
        _ENV[_k.strip()] = _v.strip()

BASE = _ENV["FOUNDRY_AUDIO_BASE_URL"].rstrip("/")
KEY = _ENV["FOUNDRY_AUDIO_API_KEY"]
MODEL = _ENV["FOUNDRY_AUDIO_MODEL"]
SEED = 12345

EMOTIONS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad"]
GOLD_FROM_CODE = {"ANG": "angry", "DIS": "disgusted", "FEA": "fearful",
                  "HAP": "happy", "NEU": "neutral", "SAD": "sad"}

_stats_lock = threading.Lock()
STATS = {"calls": 0, "cache_hits": 0, "errors": 0,
         "input_tokens": 0, "output_tokens": 0, "audio_tokens": 0}


def gold_of(item_id: str) -> str:
    return GOLD_FROM_CODE[item_id.split("_")[2]]


def _fmt_of(path: Path) -> str:
    return "mp3" if path.suffix.lower() == ".mp3" else "wav"


def _key(audio_sha: str, prompt: str, max_tok: int, top: int) -> str:
    blob = json.dumps([MODEL, audio_sha, prompt, max_tok, top, SEED], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


_sha_cache: dict[str, str] = {}


def _audio_sha(path: Path) -> str:
    p = str(path)
    if p not in _sha_cache:
        _sha_cache[p] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _sha_cache[p]


def _post(payload: dict, timeout: int = 180):
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY}"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def call(path: Path, prompt: str, max_tok: int = 12, top: int = 20,
         audio_bytes: bytes | None = None, audio_sha: str | None = None,
         fmt: str | None = None, retries: int = 6) -> dict:
    """One completion. Returns the raw API response dict (cached)."""
    sha = audio_sha or _audio_sha(path)
    ck = _key(sha, prompt, max_tok, top)
    cf = CACHE / f"{ck}.json"
    if cf.exists():
        with _stats_lock:
            STATS["cache_hits"] += 1
        return json.loads(cf.read_text())

    data = base64.b64encode(audio_bytes if audio_bytes is not None
                            else path.read_bytes()).decode()
    payload = {
        "model": MODEL, "modalities": ["text"],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "input_audio",
             "input_audio": {"data": data, "format": fmt or _fmt_of(path)}}]}],
        "max_completion_tokens": max_tok,
        "temperature": 0, "seed": SEED,
        "logprobs": True, "top_logprobs": top,
    }

    last = None
    for attempt in range(retries):
        try:
            resp = _post(payload)
            u = resp.get("usage") or {}
            with _stats_lock:
                STATS["calls"] += 1
                STATS["input_tokens"] += u.get("prompt_tokens") or 0
                STATS["output_tokens"] += u.get("completion_tokens") or 0
                det = (u.get("prompt_tokens_details") or {})
                STATS["audio_tokens"] += det.get("audio_tokens") or 0
            cf.write_text(json.dumps(resp))
            return resp
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            last = f"HTTP {e.code}: {body}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 30) + random.random() * 2)
                continue
            break
        except Exception as e:                              # transport
            last = f"{type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 30) + random.random() * 2)
    with _stats_lock:
        STATS["errors"] += 1
    return {"__error__": last}


# --------------------------------------------------------------------------
# readouts
# --------------------------------------------------------------------------
def content_of(resp: dict) -> str | None:
    if "__error__" in resp:
        return None
    try:
        return (resp["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None


def _emittable(alts: list[dict]) -> list[tuple[str, float]]:
    """Drop special tokens (bytes=None); they are not emittable as content."""
    return [(a["token"], a["logprob"]) for a in alts if a.get("bytes")]


def label_mass(resp: dict, labels: list[str]) -> dict:
    """Renormalised probability over `labels` at the first competing position.

    A position 'competes' when at least two of the labels appear among its
    emittable alternatives -- that is the token slot where the model is
    actually choosing between them.
    """
    if "__error__" in resp:
        return {}
    lp = (resp["choices"][0].get("logprobs") or {}).get("content") or []
    best = {}
    for pos in lp:
        m: dict[str, float] = {}
        for tok, logp in _emittable(pos.get("top_logprobs", [])):
            k = tok.strip().lower().lstrip("-_ ")
            for lab in labels:
                if k == lab or (len(k) >= 3 and lab.startswith(k)):
                    m[lab] = m.get(lab, 0.0) + math.exp(logp)
                    break
        if len(m) > len(best):
            best = m
        if len(best) == len(labels):
            break
    if not best:
        return {}
    tot = sum(best.values())
    return {k: v / tot for k, v in best.items()} if tot > 0 else {}


def parse_emotion(resp: dict) -> str | None:
    c = content_of(resp)
    if c is None:
        return None
    c = c.strip().strip(".,!'\" ").lower()
    return c if c in EMOTIONS else f"UNPARSED:{c[:24]}"


def pmap(fn, jobs, workers: int = 6, desc: str = ""):
    """Concurrent map that preserves input order and reports progress."""
    out = [None] * len(jobs)
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, j): i for i, j in enumerate(jobs)}
        for f in as_completed(futs):
            out[futs[f]] = f.result()
            done += 1
            if done % 25 == 0 or done == len(jobs):
                el = time.time() - t0
                rate = done / el if el else 0
                print(f"    {desc} {done}/{len(jobs)}  "
                      f"({rate:.1f}/s, eta {(len(jobs)-done)/max(rate,1e-9):.0f}s, "
                      f"cache {STATS['cache_hits']}, err {STATS['errors']})",
                      flush=True)
    return out


def print_stats():
    print(f"\n  API calls made : {STATS['calls']}")
    print(f"  cache hits     : {STATS['cache_hits']}")
    print(f"  errors         : {STATS['errors']}")
    print(f"  input tokens   : {STATS['input_tokens']:,} "
          f"(audio {STATS['audio_tokens']:,})")
    print(f"  output tokens  : {STATS['output_tokens']:,}")
