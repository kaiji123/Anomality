"""The frontend and the backend agree on every API route. Offline; no Gemini/ElevenLabs calls.

Regression for the "Unavailable: Not Found" errors in the Time Machine, Alerts and AI Analyst
sections: every /api URL that static/app.js requests (with its HTTP method) must be a route the
FastAPI app registers. Routes are matched the way Starlette matches a request, without a socket.
"""
import re
from pathlib import Path

import pytest
from starlette.routing import Match

from backend.app import app

ROOT = Path(__file__).resolve().parent.parent
JS = (ROOT / "static" / "app.js").read_text(encoding="utf-8")


def frontend_requests():
    """(method, path) for each /api request in app.js; template parts become a sample value."""
    out = set()
    for m in re.finditer(r"""(api|fetch)\(\s*[`"](/api/[^`"?$]*(?:\$\{[^}]*\}[^`"?$]*)*)""", JS):
        call, raw = m.group(1), m.group(2)
        path = re.sub(r"/\$\{[^}]*\}", "/X", raw)  # "/${symbol}" is a path parameter
        path = re.sub(r"\$\{.*", "", path).rstrip("/")  # any other "${...}" (an optional query string) ends the path
        tail = JS[m.end(): m.end() + 200]
        method = "POST" if call == "fetch" and re.search(r"method:\s*\"POST\"", tail.split(")")[0] + tail[:120]) else "GET"
        out.add((method, path))
    return sorted(out)


def registered(method: str, path: str) -> bool:
    scope = {"type": "http", "path": path, "method": method, "root_path": "", "query_string": b"", "headers": []}
    return any(r.matches(scope)[0] == Match.FULL for r in app.routes)


REQUESTS = frontend_requests()


def test_the_frontend_requests_the_new_sections_routes():
    paths = {p for _, p in REQUESTS}
    for expected in ("/api/timemachine", "/api/timemachine/X", "/api/alerts", "/api/alerts/X/ack",
                     "/api/analyst/status", "/api/analyst", "/api/voice/status", "/api/voice"):
        assert expected in paths, f"app.js no longer requests {expected}"
    assert ("POST", "/api/analyst") in REQUESTS and ("POST", "/api/voice") in REQUESTS
    assert ("POST", "/api/alerts/X/ack") in REQUESTS


@pytest.mark.parametrize("method,path", REQUESTS)
def test_every_frontend_request_has_a_backend_route(method, path):
    assert registered(method, path), f"{method} {path} is requested by static/app.js but not registered in backend/app.py"


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/timemachine"), ("GET", "/api/timemachine/NVDA"), ("GET", "/api/alerts"),
    ("POST", "/api/alerts/market_stress:market:2020-02-24/ack"), ("GET", "/api/analyst/status"),
    ("POST", "/api/analyst"), ("GET", "/api/voice/status"), ("POST", "/api/voice"),
])
def test_the_previously_failing_routes_are_registered_with_the_right_method(method, path):
    assert registered(method, path)
    wrong = "GET" if method == "POST" else "POST"
    assert not registered(wrong, path)  # the wrong method is a 405, not a silent match


def test_unknown_api_paths_are_not_routes():
    assert not registered("GET", "/api/time-machine") and not registered("GET", "/api/alert")


def test_previously_failing_routes_answer_offline(monkeypatch):
    """Expected responses (handlers called directly, offline data, no API keys, no external calls)."""
    monkeypatch.setenv("ANOMALITY_OFFLINE", "1")
    for k in ("GEMINI_API_KEY", "ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"):
        monkeypatch.delenv(k, raising=False)
    from backend import app as app_mod
    meta = app_mod.api_time_machine_meta()
    assert len(meta["assets"]) == 56 and meta["first"] <= meta["last"]
    assert app_mod.api_time_machine("NVDA", date="2020-03-13")["status"] == "ok"
    alerts = app_mod.api_alerts(date="2020-03-13")
    assert alerts["as_of"] == "2020-03-13" and alerts["checks"]
    assert app_mod.api_analyst_status() == {"configured": False, "model": app_mod.analyst.DEFAULT_MODEL}
    assert app_mod.api_voice_status()["configured"] is False
    r = app_mod.api_analyst(app_mod.AnalystRequest(kind="limits"))
    assert r["source"] == "deterministic" and r["status"] == "not_configured" and r["facts"]
