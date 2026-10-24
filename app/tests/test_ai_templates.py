from app.ai.commentary import generate_commentary
from app.ai.voice import synthesize_voice


def test_commentary_template_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = generate_commentary({"event_id": "e1", "event_type": "change_alarm", "asset": "SPY", "reason": "risk"}, cache_dir=tmp_path)
    assert out["source"] == "template"


def test_voice_no_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_VOICE_ID", raising=False)
    assert synthesize_voice("x", "e1", cache_dir=tmp_path) is None
