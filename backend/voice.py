"""Voice reports (ElevenLabs): read aloud a report the AI analyst already produced and displayed.

The endpoint takes a `report_id`, never free text: the text is rebuilt on the server from the
exact report that was shown (backend/analyst.py), so the audio can't add claims and the paid API
can't be used for arbitrary text. Audio is kept in a small in-memory cache (no files on disk).

Configuration (server side only): ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID (a voice from
your ElevenLabs library; use a stock or designed voice, not a clone of a real person without
their consent), optional ELEVENLABS_MODEL and VOICE_TIMEOUT seconds.
REST: POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128
with the key in the xi-api-key header.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from collections import OrderedDict
from typing import Callable, Dict, Optional

import requests

from . import analyst
from .ratelimit import RateLimiter

DEFAULT_MODEL = "eleven_multilingual_v2"  # the API's default model; up to 10,000 characters per request
ENDPOINT = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
OUTPUT_FORMAT = "mp3_44100_128"
MAX_CHARS = 4000  # well under every current model's per-request limit; a report is usually ~1,500
MAX_CACHED, MAX_CACHED_BYTES = 20, 40_000_000
VOICE_LIMIT = RateLimiter(10, 60.0)  # actual ElevenLabs calls per minute, for the whole server

_lock = threading.Lock()
_audio: "OrderedDict[str, bytes]" = OrderedDict()


class VoiceError(Exception):
    def __init__(self, status: str, message: str, http: int = 503, retry_after: Optional[float] = None):
        super().__init__(message)
        self.status, self.http, self.retry_after = status, http, retry_after


def configured(env: Optional[Dict[str, str]] = None) -> dict:
    env = os.environ if env is None else env
    return {"configured": bool(env.get("ELEVENLABS_API_KEY") and env.get("ELEVENLABS_VOICE_ID")),
            "model": env.get("ELEVENLABS_MODEL", DEFAULT_MODEL), "max_chars": MAX_CHARS}


def check_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        raise VoiceError("empty", "There is no report text to read.", 422)
    if len(text) > MAX_CHARS:
        raise VoiceError("too_long", f"The report is {len(text):,} characters; voice reports are limited to {MAX_CHARS:,}.", 422)
    return text


def synthesize(text: str, env: Optional[Dict[str, str]] = None, post: Optional[Callable] = None) -> bytes:
    env = os.environ if env is None else env
    key, voice = env.get("ELEVENLABS_API_KEY"), env.get("ELEVENLABS_VOICE_ID")
    if not key or not voice:
        raise VoiceError("not_configured", "Voice reports are not configured on this server (ELEVENLABS_API_KEY and "
                                           "ELEVENLABS_VOICE_ID). The written report above is complete.")
    if not re.fullmatch(r"[A-Za-z0-9]{6,64}", voice):
        raise VoiceError("not_configured", "ELEVENLABS_VOICE_ID does not look like a voice id.")
    text = check_text(text)
    post = post or requests.post  # looked up at call time
    model, timeout = env.get("ELEVENLABS_MODEL", DEFAULT_MODEL), float(env.get("VOICE_TIMEOUT", "30"))
    cache_key = hashlib.sha256(f"{voice}|{model}|{text}".encode()).hexdigest()
    with _lock:
        if cache_key in _audio:
            _audio.move_to_end(cache_key)
            return _audio[cache_key]
    allowed, retry_after = VOICE_LIMIT.acquire()  # only actual calls take a slot, cached audio doesn't
    if not allowed:
        raise VoiceError("rate_limited", f"The limit of {VOICE_LIMIT.max_calls} voice reports per minute was reached; "
                                         f"try again in {int(retry_after) + 1} seconds. The written report above is complete.",
                         429, retry_after)
    try:
        res = post(ENDPOINT.format(voice_id=voice), params={"output_format": OUTPUT_FORMAT}, timeout=timeout,
                   json={"text": text, "model_id": model},
                   headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"})
    except requests.Timeout:
        raise VoiceError("timeout", f"ElevenLabs did not answer within {timeout:g} seconds.", 504)
    except requests.RequestException as e:
        raise VoiceError("provider_error", f"Could not reach ElevenLabs ({type(e).__name__}).", 502)
    if res.status_code == 429:
        raise VoiceError("rate_limited", "ElevenLabs' rate limit or quota was reached; try again later.", 429)
    if res.status_code >= 400:
        code, why = _provider_error(res, secrets=(key, voice))
        if res.status_code in (401, 403) or code in ("invalid_api_key", "authentication_error"):
            raise VoiceError("not_authorized", f"ElevenLabs rejected the API key or voice (HTTP {res.status_code})"
                                               + (f": {why}" if why else "."), 502)
        raise VoiceError("provider_error", f"ElevenLabs returned HTTP {res.status_code}" + (f": {why}" if why else "."), 502)
    audio = res.content or b""
    ctype = (res.headers or {}).get("Content-Type", "")
    if len(audio) < 100 or not (ctype.startswith("audio/") or audio[:3] == b"ID3" or audio[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
        raise VoiceError("invalid_response", "ElevenLabs did not return audio.", 502)
    with _lock:
        _audio[cache_key] = audio
        while len(_audio) > MAX_CACHED or sum(len(a) for a in _audio.values()) > MAX_CACHED_BYTES:
            _audio.popitem(last=False)
    return audio


def _provider_error(res, secrets=()) -> tuple:
    """(error code, readable message) from an ElevenLabs error body, e.g. {"detail": {"status": ...,
    "code": ..., "type": ..., "message": ...}}. The key and voice id are redacted and the message is
    capped, so it is safe to show in the page; anything unexpected yields (None, None)."""
    try:
        body = res.json()
    except (ValueError, AttributeError):
        return None, None
    d = body.get("detail", body) if isinstance(body, dict) else None
    if isinstance(d, dict):
        code = d.get("code") or d.get("type") or d.get("status")
        msg = d.get("message") or d.get("status")
    elif isinstance(d, str):
        code, msg = None, d
    else:
        return None, None
    msg = str(msg or "")
    for s in secrets:
        if s:
            msg = msg.replace(s, "<redacted>")
    return (str(code) if code else None), (msg[:300] or None)


def for_report(report_id: str, env: Optional[Dict[str, str]] = None, post: Optional[Callable] = None) -> bytes:
    report = analyst.get_report(report_id)
    if report is None:
        raise VoiceError("unknown_report", "That report is no longer available; ask the analyst again, then listen.", 404)
    return synthesize(analyst.report_text(report), env, post)
