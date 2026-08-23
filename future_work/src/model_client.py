"""Step 4 — model API wrapper.

One concrete client for the pilot: Phi-4-multimodal-instruct on an Azure AI
Foundry OpenAI-compatible endpoint. Audio is uploaded as raw base64 bytes with a
declared container format derived from the file extension — never a filename or
URL (a filename would leak the condition and could bias the model).
"""
from __future__ import annotations

import base64
import random
import time
from pathlib import Path
from typing import Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    InternalServerError,
    RateLimitError,
)
from openai import OpenAI

from .config import cfg

# file extension -> the `format` string the API expects in input_audio
_EXT_TO_FORMAT = {".wav": "wav", ".mp3": "mp3"}

MAX_RETRIES = 5
MAX_OUTPUT_TOKENS = 256
MAX_BACKOFF_SECONDS = 30.0  # never block a single call longer than this


class ModelClient(Protocol):
    model_id: str

    def version_string(self) -> str: ...

    def call(self, audio_path: str, prompt: str) -> dict:
        """Returns {raw_text, model_version, latency_ms, usage, error}."""
        ...


def _audio_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext not in _EXT_TO_FORMAT:
        raise ValueError(f"unsupported audio extension for upload: {ext} ({path})")
    return _EXT_TO_FORMAT[ext]


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


class FoundryOpenAIClient:
    """Azure AI Foundry, OpenAI-compatible chat/completions."""

    def __init__(self) -> None:
        env = cfg.env
        self.model_id = env["FOUNDRY_MODEL"]
        self._client = OpenAI(
            base_url=env["FOUNDRY_BASE_URL"],
            api_key=env["FOUNDRY_API_KEY"],
            max_retries=0,   # we implement our own backoff
            timeout=60.0,    # this deployment hangs under load; fail fast + retry
        )

    def version_string(self) -> str:
        return self.model_id

    def call(self, audio_path: str, prompt: str) -> dict:
        fmt = _audio_format(audio_path)
        data = _b64(audio_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "input_audio",
                 "input_audio": {"data": data, "format": fmt}},
            ],
        }]

        last_err: str | None = None
        for attempt in range(MAX_RETRIES + 1):
            t0 = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=0,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
                latency_ms = int((time.perf_counter() - t0) * 1000)
                usage = resp.usage
                return {
                    "raw_text": resp.choices[0].message.content,
                    "model_version": resp.model,
                    "latency_ms": latency_ms,
                    "usage": {
                        "input_tokens": getattr(usage, "prompt_tokens", None),
                        "output_tokens": getattr(usage, "completion_tokens", None),
                    },
                    "error": None,
                }
            except (RateLimitError, InternalServerError, APIConnectionError) as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt == MAX_RETRIES:
                    break
                self._sleep_backoff(attempt, e)
            except APIStatusError as e:
                # 5xx -> retry; other 4xx (e.g. 400 bad audio) -> do not retry
                if e.status_code and 500 <= e.status_code < 600 and attempt < MAX_RETRIES:
                    last_err = f"{type(e).__name__}({e.status_code}): {e}"
                    self._sleep_backoff(attempt, e)
                    continue
                return _error_result(f"{type(e).__name__}({e.status_code}): {e}")
            except Exception as e:  # noqa: BLE001 — record anything else as api_error
                return _error_result(f"{type(e).__name__}: {e}")

        return _error_result(last_err or "exhausted retries")

    @staticmethod
    def _sleep_backoff(attempt: int, exc: Exception) -> None:
        # Honor Retry-After when the server supplies it, else exponential + jitter.
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            delay = retry_after
        else:
            delay = min(2.0 ** attempt, 30.0)
        delay += random.uniform(0, 0.5)
        time.sleep(delay)


def _retry_after_seconds(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    val = resp.headers.get("retry-after") if hasattr(resp, "headers") else None
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _error_result(msg: str) -> dict:
    return {
        "raw_text": None,
        "model_version": None,
        "latency_ms": None,
        "usage": {"input_tokens": None, "output_tokens": None},
        "error": msg,
    }


# ---------------------------------------------------------------------------
# Gemini (Google AI Studio) — REST, closed model with an opaque decode path.
# ---------------------------------------------------------------------------
import requests  # noqa: E402

# file extension -> mime type for Gemini inline_data
_EXT_TO_MIME = {".wav": "audio/wav", ".mp3": "audio/mp3", ".mp4": "audio/mp4"}
# gemini-3.x flash is a "thinking" model: hidden reasoning consumes the output
# budget, so this must be generous or the visible answer is truncated to empty.
GEMINI_MAX_OUTPUT_TOKENS = 2048


class GeminiClient:
    """Google Generative Language API (generateContent)."""

    def __init__(self) -> None:
        env = cfg.env
        self.model_id = env["GEMINI_MODEL"]
        self._api_key = env["GEMINI_API_KEY"]
        self._url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_id}:generateContent"
        )

    def version_string(self) -> str:
        return self.model_id

    def call(self, audio_path: str, prompt: str) -> dict:
        ext = Path(audio_path).suffix.lower()
        if ext not in _EXT_TO_MIME:
            return _error_result(f"unsupported audio extension: {ext}")
        payload = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": _EXT_TO_MIME[ext], "data": _b64(audio_path)}},
            ]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS,
            },
        }

        last_err = None
        for attempt in range(MAX_RETRIES + 1):
            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    self._url, params={"key": self._api_key}, json=payload, timeout=90,
                )
            except requests.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt == MAX_RETRIES:
                    break
                time.sleep(min(2.0 ** attempt, 30.0) + random.uniform(0, 0.5))
                continue

            latency_ms = int((time.perf_counter() - t0) * 1000)
            if resp.status_code == 200:
                return _parse_gemini_ok(resp.json(), latency_ms)
            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if attempt == MAX_RETRIES:
                    break
                # Cap the honored delay: Gemini's daily-quota RetryInfo can be an
                # HOUR, and blocking a single call that long is never right. If the
                # server wants longer than the cap, that's quota exhaustion — retry
                # briefly, then let it surface as api_error for the resumable run.
                want = _gemini_retry_delay(resp) or (2.0 ** attempt)
                time.sleep(min(want, MAX_BACKOFF_SECONDS) + random.uniform(0, 0.5))
                continue
            # other 4xx: do not retry
            return _error_result(f"HTTP {resp.status_code}: {resp.text[:200]}")

        return _error_result(last_err or "exhausted retries")


def _parse_gemini_ok(data: dict, latency_ms: int) -> dict:
    cands = data.get("candidates") or []
    text = None
    if cands:
        parts = (cands[0].get("content") or {}).get("parts") or []
        chunks = [p["text"] for p in parts if "text" in p]
        text = "".join(chunks) if chunks else None
    usage = data.get("usageMetadata", {})
    # Output billed = visible candidate tokens + hidden thinking tokens.
    out_tokens = (usage.get("candidatesTokenCount") or 0) + \
                 (usage.get("thoughtsTokenCount") or 0)
    return {
        "raw_text": text,
        "model_version": data.get("modelVersion"),
        "latency_ms": latency_ms,
        "usage": {
            "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": out_tokens or None,
        },
        # No text -> let the parser classify as `empty` (a model failure, distinct
        # from api_error). finish_reason is retained in the saved JSON for triage.
        "error": None,
        "finish_reason": cands[0].get("finishReason") if cands else "no_candidates",
    }


def _gemini_retry_delay(resp) -> float | None:
    ra = resp.headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    try:
        for d in resp.json().get("error", {}).get("details", []):
            if d.get("@type", "").endswith("RetryInfo") and "retryDelay" in d:
                return float(str(d["retryDelay"]).rstrip("s"))
    except Exception:  # noqa: BLE001
        pass
    return None


def get_client() -> ModelClient:
    backend = cfg.model_backend
    if backend == "gemini":
        return GeminiClient()
    if backend == "foundry_openai":
        return FoundryOpenAIClient()
    raise ValueError(f"unknown MODEL_BACKEND: {backend}")
