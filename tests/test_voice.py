"""Voice reports (backend/voice.py). A fake HTTP client stands in for ElevenLabs: no key, no
network, no paid calls."""
import json
import re
from pathlib import Path

import pytest
import requests

from backend import analyst as an
from backend import voice as vo
from backend.ratelimit import RateLimiter
from tests.test_analyst import sources

ROOT = Path(__file__).resolve().parent.parent
ENV = {"ELEVENLABS_API_KEY": "el-test-key", "ELEVENLABS_VOICE_ID": "Voice123abc"}
MP3 = b"ID3" + b"\x00" * 500


class Resp:
    def __init__(self, status=200, content=MP3, ctype="audio/mpeg"):
        self.status_code, self.content, self.headers = status, content, {"Content-Type": ctype}


class Recorder:
    def __init__(self, response=None, exc=None):
        self.calls, self.response, self.exc = [], response or Resp(), exc

    def __call__(self, url, params=None, timeout=None, json=None, headers=None):
        self.calls.append({"url": url, "params": params, "json": json, "headers": headers, "timeout": timeout})
        if self.exc:
            raise self.exc
        return self.response


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(vo, "VOICE_LIMIT", RateLimiter(10, 60.0))  # a fresh budget for each test
    vo._audio.clear(); an._cache.clear(); an._reports.clear()
    yield
    vo._audio.clear(); an._cache.clear(); an._reports.clear()


def shown_report():
    """A report exactly as the analyst returned it to the page (deterministic: no Gemini key)."""
    return an.explain("summary", sources=sources(), env={})


def test_reads_exactly_the_displayed_report():
    r = shown_report()
    post = Recorder()
    audio = vo.for_report(r["report_id"], env=ENV, post=post)
    assert audio == MP3
    sent = post.calls[0]["json"]["text"]
    for k in an.SECTIONS:  # every displayed item is read, nothing else is added
        for it in r["sections"][k]:
            assert it["text"].rstrip(".") in sent
    assert sent.startswith("Anomality report.") and sent.endswith(r["disclaimer"])
    assert not re.search(r"\bF\d+\b", sent)  # citation ids are not read aloud


def test_key_stays_in_the_header_and_uses_the_documented_request():
    r = shown_report()
    post = Recorder()
    vo.for_report(r["report_id"], env=ENV, post=post)
    c = post.calls[0]
    assert c["url"] == "https://api.elevenlabs.io/v1/text-to-speech/Voice123abc" and "el-test-key" not in c["url"]
    assert c["headers"]["xi-api-key"] == "el-test-key" and c["params"] == {"output_format": "mp3_44100_128"}
    assert c["json"]["model_id"] == vo.DEFAULT_MODEL and c["timeout"] == 30


def test_not_configured_means_no_call():
    post = Recorder(exc=AssertionError("must not call ElevenLabs without a key"))
    for env in ({}, {"ELEVENLABS_API_KEY": "k"}, {"ELEVENLABS_VOICE_ID": "Voice123abc"}):
        with pytest.raises(vo.VoiceError) as e:
            vo.synthesize("Hello.", env=env, post=post)
        assert e.value.status == "not_configured" and "written report" in str(e.value)
    assert not post.calls


def test_only_known_reports_can_be_read():
    with pytest.raises(vo.VoiceError) as e:
        vo.for_report("0123456789abcdef", env=ENV, post=Recorder())
    assert e.value.status == "unknown_report" and e.value.http == 404


@pytest.mark.parametrize("text,status", [("", "empty"), ("   \n ", "empty"), ("x" * (vo.MAX_CHARS + 1), "too_long")])
def test_empty_or_too_long_text_is_rejected_before_any_call(text, status):
    post = Recorder()
    with pytest.raises(vo.VoiceError) as e:
        vo.synthesize(text, env=ENV, post=post)
    assert e.value.status == status and e.value.http == 422 and not post.calls


@pytest.mark.parametrize("response,exc,status", [
    (Resp(500), None, "provider_error"),
    (Resp(429), None, "rate_limited"),
    (Resp(401), None, "not_authorized"),
    (None, requests.Timeout(), "timeout"),
    (None, requests.ConnectionError(), "provider_error"),
    (Resp(200, content=b'{"detail": "oops"}', ctype="application/json"), None, "invalid_response"),
    (Resp(200, content=b""), None, "invalid_response"),
])
def test_provider_failures_are_reported(response, exc, status):
    with pytest.raises(vo.VoiceError) as e:
        vo.synthesize("A short report.", env=ENV, post=Recorder(response, exc))
    assert e.value.status == status and "el-test-key" not in str(e.value)


class JsonResp(Resp):
    """An ElevenLabs error response with a JSON body."""

    def __init__(self, status, body):
        super().__init__(status, content=json.dumps(body).encode(), ctype="application/json")
        self._body = body

    def json(self):
        return self._body


# The body ElevenLabs returns when the key's *id* is pasted instead of the secret key (seen live).
KEY_ID_USED = {"detail": {"status": "api_key_id_used_as_api_key", "code": "invalid_api_key", "type": "authentication_error",
                          "message": "API key ID used as API key - only valid API keys can be used. API keys start with "
                                     "'sk_' and are shown when the key is created or rotated."}}


def test_a_key_id_instead_of_a_key_is_reported_as_an_auth_error_with_the_reason():
    with pytest.raises(vo.VoiceError) as e:
        vo.synthesize("A short report.", env=ENV, post=Recorder(JsonResp(400, KEY_ID_USED)))
    assert e.value.status == "not_authorized" and e.value.http == 502
    assert "HTTP 400" in str(e.value) and "API key ID used as API key" in str(e.value) and "sk_" in str(e.value)


@pytest.mark.parametrize("status,body,expected_status,expected_text", [
    (400, {"detail": {"status": "voice_not_found", "message": "A voice with the voice_id Voice123abc was not found."}},
     "provider_error", "A voice with the voice_id <redacted> was not found."),
    (422, {"detail": [{"loc": ["body", "text"], "msg": "field required"}]}, "provider_error", "ElevenLabs returned HTTP 422."),
    (400, {"detail": "model eleven_x does not exist"}, "provider_error", "model eleven_x does not exist"),
    (401, {"detail": {"status": "invalid_api_key", "message": "Invalid API key el-test-key"}}, "not_authorized", "Invalid API key <redacted>"),
])
def test_provider_error_details_are_shown_without_secrets(status, body, expected_status, expected_text):
    with pytest.raises(vo.VoiceError) as e:
        vo.synthesize("A short report.", env=ENV, post=Recorder(JsonResp(status, body)))
    msg = str(e.value)
    assert e.value.status == expected_status and expected_text in msg
    assert "el-test-key" not in msg and "Voice123abc" not in msg


def test_a_non_json_error_body_still_gives_a_clear_message():
    with pytest.raises(vo.VoiceError) as e:
        vo.synthesize("A short report.", env=ENV, post=Recorder(Resp(400, content=b"<html>bad</html>", ctype="text/html")))
    assert e.value.status == "provider_error" and str(e.value) == "ElevenLabs returned HTTP 400."


def test_provider_message_is_capped():
    long = {"detail": {"status": "x", "message": "y" * 5000}}
    with pytest.raises(vo.VoiceError) as e:
        vo.synthesize("A short report.", env=ENV, post=Recorder(JsonResp(400, long)))
    assert len(str(e.value)) < 400


def test_audio_is_cached_not_regenerated():
    r = shown_report()
    post = Recorder()
    vo.for_report(r["report_id"], env=ENV, post=post)
    vo.for_report(r["report_id"], env=ENV, post=post)
    assert len(post.calls) == 1


def test_no_audio_files_are_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    vo.synthesize("A short report.", env=ENV, post=Recorder())
    assert not list(tmp_path.rglob("*"))


def test_no_voice_secret_in_the_frontend():
    for f in ("static/app.js", "static/index.html"):
        src = (ROOT / f).read_text(encoding="utf-8")
        assert "ELEVENLABS" not in src and "xi-api-key" not in src and "api.elevenlabs.io" not in src


def test_api_handlers(monkeypatch):
    from fastapi import HTTPException

    from backend import app as app_mod
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert app_mod.api_voice_status()["configured"] is False
    r = shown_report()
    with pytest.raises(HTTPException) as e:
        app_mod.api_voice(app_mod.VoiceRequest(report_id=r["report_id"]))
    assert e.value.status_code == 503 and e.value.detail["status"] == "not_configured"
    monkeypatch.setenv("ELEVENLABS_API_KEY", "el-test-key")
    monkeypatch.setenv("ELEVENLABS_VOICE_ID", "Voice123abc")
    monkeypatch.setattr(vo.requests, "post", Recorder())
    out = app_mod.api_voice(app_mod.VoiceRequest(report_id=r["report_id"]))
    assert out.media_type == "audio/mpeg" and out.body == MP3


def test_voice_rate_limit_counts_real_calls_only(monkeypatch):
    t = [1000.0]
    monkeypatch.setattr(vo, "VOICE_LIMIT", RateLimiter(1, 60.0, clock=lambda: t[0]))
    post = Recorder()
    vo.synthesize("First report.", env=ENV, post=post)
    vo.synthesize("First report.", env=ENV, post=post)  # cached: free
    with pytest.raises(vo.VoiceError) as e:  # another text can't get around the budget
        vo.synthesize("Second report.", env=ENV, post=post)
    assert e.value.status == "rate_limited" and e.value.http == 429 and 0 < e.value.retry_after <= 60
    assert len(post.calls) == 1 and "written report above is complete" in str(e.value)
    t[0] += 61
    vo.synthesize("Second report.", env=ENV, post=post)
    assert len(post.calls) == 2


def test_rejected_requests_do_not_use_the_voice_budget():
    for text in ("", "x" * (vo.MAX_CHARS + 1)):
        with pytest.raises(vo.VoiceError):
            vo.synthesize(text, env=ENV, post=Recorder())
    with pytest.raises(vo.VoiceError):
        vo.synthesize("Hello.", env={}, post=Recorder())
    assert vo.VOICE_LIMIT.used() == 0


def test_voice_api_sends_retry_after(monkeypatch):
    from fastapi import HTTPException

    from backend import app as app_mod
    r = shown_report()

    def limited(*a, **k):
        raise vo.VoiceError("rate_limited", "Limit reached.", 429, 7.2)
    monkeypatch.setattr(vo, "for_report", limited)
    with pytest.raises(HTTPException) as e:
        app_mod.api_voice(app_mod.VoiceRequest(report_id=r["report_id"]))
    assert e.value.status_code == 429 and e.value.headers["Retry-After"] == "8"
