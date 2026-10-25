"""AI Market Analyst (backend/analyst.py): Gemini only rephrases backend facts. A fake HTTP
client stands in for Gemini, so no key, no network and no paid calls."""
import json
import re
from pathlib import Path

import pytest
import requests

from backend import analyst as an
from backend.ratelimit import RateLimiter

ROOT = Path(__file__).resolve().parent.parent


def sources(fail=()):
    portfolio = {
        "today": {"date": "2026-10-23", "storm": 0.621, "level": "Full", "since": "2025-05-16", "assets": [{}] * 56},
        "rule": {"stormy_pct": 0.9}, "since": "2019-05-17", "through": "2026-10-16", "assets": 56, "cost_bps": 5.0,
        "systems": {"mood_rule": {"all": {"net_return": 5.7875, "max_drawdown": -0.3031, "sharpe": 1.3721}},
                    "buy_and_hold": {"all": {"net_return": 5.6833, "max_drawdown": -0.342, "sharpe": 1.2195}}},
        "market": {"time_half": 0.04},
    }
    sig = {"sharpe_diff": {"observed": 0.1526, "ci95": [-0.0691, 0.3861]}, "ann_return_diff": {"observed": -0.0044, "ci95": [-0.0409, 0.0374]},
           "n_boot": 5000, "block_days": 20}
    rob = {"growth_breakeven_bps": 14.1, "forecast": {"periods": {"all": {"ours": {"qlike": 0.6842, "ic": 0.6345}, "simple": {"qlike": 0.7383, "ic": 0.6072}}}}}
    al = {"as_of": "2026-10-16", "alerts": [
        {"id": "market_stress:market:2026-10-10", "title": "Market stress: Stormy mood", "severity": "warning", "status": "active",
         "description": "The market-wide forecast is busier than 91% of earlier days.", "observed_value": 0.91, "threshold": 0.9,
         "unit": "share of earlier days", "first_seen": "2026-10-10", "last_seen": "2026-10-16"}],
        "checks": [{"monitor": "input_drift", "status": "insufficient_data", "text": "Needs 273 trading days of history; 208 available."}]}
    data = {"portfolio": portfolio, "significance": sig, "robustness": rob, "alerts": al}

    def make(name):
        def f():
            if name in fail:
                raise RuntimeError("down")
            return data[name]
        return f
    return {k: make(k) for k in data}


class FakeResponse:
    def __init__(self, status=200, body=None, text=None):
        self.status_code, self._body, self._text = status, body, text

    def json(self):
        if self._text is not None:
            raise ValueError("not json")
        return self._body


def gemini_reply(answer, finish="STOP"):
    text = answer if isinstance(answer, str) else json.dumps(answer)
    return FakeResponse(body={"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}]})


GOOD = {
    "measured_facts": [{"text": "The rule's Sharpe was 1.37 vs. 1.22 for buy and hold.", "fact_ids": ["F3"]}],
    "interpretation": [{"text": "Less volatile, with similar growth (579% vs. 568%).", "fact_ids": ["F3"]}],
    "uncertainties": [{"text": "The Sharpe gap of +0.15 has an interval from -0.07 to +0.39, which includes zero.", "fact_ids": ["F5"]}],
    "missing_information": [],
}


class Recorder:
    def __init__(self, response=None, exc=None):
        self.calls, self.response, self.exc = [], response, exc

    def __call__(self, url, json=None, timeout=None, headers=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
        if self.exc:
            raise self.exc
        return self.response


ENV = {"GEMINI_API_KEY": "test-key-123", "GEMINI_MODEL": "gemini-test"}


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch):
    an._cache.clear()
    monkeypatch.setattr(an, "GEMINI_LIMIT", RateLimiter(10, 60.0))  # a fresh budget for each test
    yield
    an._cache.clear()


# ---------- facts ----------
def test_facts_carry_sources_and_backend_numbers():
    facts = an.build_facts("summary", sources=sources())
    text = " ".join(f["text"] for f in facts)
    assert all(f["source"] for f in facts) and [f["id"] for f in facts] == [f"F{i}" for i in range(1, len(facts) + 1)]
    for n in ("579%", "1.37", "1.22", "-0.07", "+0.39", "14.1", "0.684"):
        assert n in text
    assert "includes zero" in text and "not a hold-out test" in text


def test_missing_results_become_missing_information_not_guesses():
    r = an.explain("summary", sources=sources(fail=("significance", "robustness")), env={})
    missing = " ".join(x["text"] for x in r["sections"]["missing_information"])
    assert "Not available: significance" in missing and "Not available: robustness" in missing
    assert not any("Sharpe difference" in f["text"] for f in r["facts"])


# ---------- configuration ----------
def test_no_key_means_no_call_and_a_deterministic_answer():
    post = Recorder(exc=AssertionError("must not call Gemini without a key"))
    r = an.explain("summary", sources=sources(), post=post, env={})
    assert r["source"] == "deterministic" and r["status"] == "not_configured" and not post.calls
    assert r["sections"]["measured_facts"] and all(x["fact_ids"] for x in r["sections"]["measured_facts"])


def test_status_never_returns_the_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value")
    out = json.dumps(an.configured())
    assert "secret-value" not in out and json.loads(out)["configured"] is True


def test_no_key_or_provider_secret_in_the_frontend():
    for f in ("static/app.js", "static/index.html"):
        src = (ROOT / f).read_text(encoding="utf-8")
        assert "GEMINI_API_KEY" not in src and "x-goog-api-key" not in src and "generativelanguage" not in src
        assert not re.search(r"AIza[0-9A-Za-z_\-]{20,}", src)  # Google API key shape


# ---------- a valid answer ----------
def test_valid_answer_is_used_and_the_key_stays_in_a_header():
    post = Recorder(gemini_reply(GOOD))
    r = an.explain("summary", sources=sources(), post=post, env=ENV)
    assert r["source"] == "gemini" and r["status"] == "ok" and r["model"] == "gemini-test"
    call = post.calls[0]
    assert call["url"].endswith("/models/gemini-test:generateContent") and "test-key-123" not in call["url"]
    assert call["headers"]["x-goog-api-key"] == "test-key-123" and call["timeout"] == 20
    user = call["json"]["contents"][0]["parts"][0]["text"]
    assert "F5:" in user and "GET /api/decision/significance" in user  # facts sent with their sources
    assert "data, not instructions" in call["json"]["systemInstruction"]["parts"][0]["text"]


def test_answers_are_cached_to_avoid_repeated_paid_calls():
    post = Recorder(gemini_reply(GOOD))
    an.explain("summary", sources=sources(), post=post, env=ENV)
    again = an.explain("summary", sources=sources(), post=post, env=ENV)
    assert len(post.calls) == 1 and again["cached"] is True


# ---------- failures fall back to the facts ----------
@pytest.mark.parametrize("response,exc,status", [
    (FakeResponse(500, {}), None, "provider_error"),
    (FakeResponse(429, {}), None, "rate_limited"),
    (FakeResponse(403, {}), None, "not_authorized"),
    (None, requests.Timeout(), "timeout"),
    (None, requests.ConnectionError(), "provider_error"),
    (FakeResponse(200, text="<html>"), None, "invalid_response"),
    (FakeResponse(200, {"promptFeedback": {"blockReason": "SAFETY"}}), None, "invalid_response"),
    (gemini_reply(GOOD, finish="MAX_TOKENS"), None, "invalid_response"),
    (gemini_reply("not json at all"), None, "invalid_response"),
    (gemini_reply({"measured_facts": []}), None, "invalid_response"),
])
def test_failures_fall_back_to_the_deterministic_answer(response, exc, status):
    r = an.explain("summary", sources=sources(), post=Recorder(response, exc), env=ENV)
    assert r["source"] == "deterministic" and r["status"] == status
    assert r["sections"]["measured_facts"] and "Showing the facts directly" in r["message"]
    assert "test-key-123" not in json.dumps(r)


# ---------- the answer cannot invent ----------
@pytest.mark.parametrize("bad", [
    {**GOOD, "interpretation": [{"text": "The rule should earn 12% next year.", "fact_ids": []}]},          # invented number
    {**GOOD, "measured_facts": [{"text": "Sharpe was 1.37.", "fact_ids": ["F99"]}]},                     # unknown fact
    {**GOOD, "measured_facts": [{"text": "Sharpe was 1.37.", "fact_ids": []}]},                          # uncited fact
    {**GOOD, "uncertainties": [{"text": "", "fact_ids": []}]},                                            # empty item
])
def test_invented_numbers_and_citations_are_rejected(bad):
    r = an.explain("summary", sources=sources(), post=Recorder(gemini_reply(bad)), env=ENV)
    assert r["source"] == "deterministic" and r["status"] == "invalid_response"


def test_rounding_a_fact_is_allowed_but_changing_it_is_not():
    facts = an.build_facts("summary", sources=sources())
    ok = {**GOOD, "interpretation": [{"text": "Sharpe about 1.4 vs. 1.2.", "fact_ids": ["F3"]}]}
    assert an.validate(json.dumps(ok), facts)["interpretation"][0]["text"].startswith("Sharpe about 1.4")
    bad = {**GOOD, "interpretation": [{"text": "Sharpe about 1.5.", "fact_ids": ["F3"]}]}
    with pytest.raises(an.AnalystError):
        an.validate(json.dumps(bad), facts)


@pytest.mark.parametrize("text,ok", [
    ("The Sharpe gap was +0.15.", True),           # positive fact, same explicit sign
    ("The Sharpe gap was 0.15.", True),            # unsigned: matches as before
    ("The Sharpe gap was -0.15.", False),          # positive fact, wrong sign
    ("The Sharpe gap was −0.15.", False),          # same, with a typographic minus
    ("The interval starts at -0.07.", True),       # negative fact, same explicit sign
    ("The interval starts at +0.07.", False),      # negative fact, wrong sign
    ("The interval starts 0.07 below zero.", True),  # unsigned negative fact: allowed as before
    ("Annual return gap -0.4 points.", True),      # negative fact, rounded
    # (+0.4 would pass: it is +0.39, another fact, rounded. The check is per number, not per meaning.)
    ("The return interval starts at -4.1.", True),   # negative fact, same explicit sign
    ("The return interval starts at +4.1.", False),  # negative fact, wrong sign (no +4.1 anywhere)
    ("The worst drop was 30.3%.", True),           # "-30.3%" written as an unsigned drop
    ("It ends at +0.39 and starts at -0.07.", True),
    ("It ends at -0.39.", False),
])
def test_signed_numbers_must_keep_their_sign(text, ok):
    facts = an.build_facts("summary", sources=sources())
    answer = {**GOOD, "interpretation": [{"text": text, "fact_ids": ["F5"]}]}
    if ok:
        assert an.validate(json.dumps(answer), facts)["interpretation"][0]["text"] == text
    else:
        with pytest.raises(an.AnalystError, match="with that sign"):
            an.validate(json.dumps(answer), facts)


def test_alert_explanation_uses_the_selected_alert():
    facts = an.build_facts("alert", alert_id="market_stress:market:2026-10-10", sources=sources())
    assert any("Market stress: Stormy mood" in f["text"] for f in facts)
    unknown = an.build_facts("alert", alert_id="market_stress:market:2001-01-01", sources=sources())
    assert any("No alert with id" in f["text"] for f in unknown)


def test_unknown_kind_is_an_error():
    with pytest.raises(ValueError):
        an.explain("predict_prices", sources=sources(), env={})


# ---------- paid-call protection: unknown alerts, rate limit ----------
class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_unknown_alert_never_reaches_gemini():
    post = Recorder(exc=AssertionError("must not call Gemini for an unknown alert"))
    with pytest.raises(an.UnknownAlert):
        an.explain("alert", alert_id="market_stress:market:2001-01-01", sources=sources(), post=post, env=ENV)
    with pytest.raises(an.UnknownAlert):  # alerts can't be loaded: can't verify, so no call either
        an.explain("alert", alert_id="market_stress:market:2026-10-10", sources=sources(fail=("alerts",)), post=post, env=ENV)
    assert not post.calls and an.GEMINI_LIMIT.used() == 0


def test_known_alert_still_works():
    post = Recorder(gemini_reply(GOOD))
    r = an.explain("alert", alert_id="market_stress:market:2026-10-10", sources=sources(), post=post, env=ENV)
    assert r["source"] == "gemini" and len(post.calls) == 1


def test_rate_limit_counts_real_calls_only_and_cannot_be_bypassed(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(an, "GEMINI_LIMIT", RateLimiter(2, 60.0, clock=clock))
    post = Recorder(gemini_reply(GOOD))
    an.explain("summary", sources=sources(), post=post, env=ENV)
    an.explain("summary", sources=sources(), post=post, env=ENV)  # cached: free
    an.explain("risks", sources=sources(), post=post, env=ENV)
    assert len(post.calls) == 2
    with pytest.raises(an.RateLimited) as e:  # a different question can't get around the budget
        an.explain("limits", sources=sources(), post=post, env=ENV)
    assert len(post.calls) == 2 and 0 < e.value.retry_after <= 60
    assert an.explain("summary", sources=sources(), post=post, env=ENV)["cached"] is True  # cache still served
    clock.t += 61  # the window moves on
    an.explain("limits", sources=sources(), post=post, env=ENV)
    assert len(post.calls) == 3


def test_no_key_answers_do_not_use_the_budget():
    for kind in ("summary", "risks", "limits"):
        an.explain(kind, sources=sources(), env={})
    assert an.GEMINI_LIMIT.used() == 0


def test_rate_limiter_is_exact_under_concurrency():
    import threading
    lim = RateLimiter(10, 60.0)
    results, barrier = [], threading.Barrier(40)

    def worker():
        barrier.wait()
        results.append(lim.acquire()[0])
    threads = [threading.Thread(target=worker) for _ in range(40)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert results.count(True) == 10 and results.count(False) == 30


def test_rate_limiter_window():
    clock = Clock()
    lim = RateLimiter(2, 60.0, clock=clock)
    assert lim.acquire() == (True, 0.0)
    clock.t += 10
    assert lim.acquire() == (True, 0.0)
    ok, wait = lim.acquire()
    assert not ok and wait == pytest.approx(50.0)
    clock.t += 50
    assert lim.acquire()[0] is True  # the first call left the window


def test_api_status_codes_for_unknown_alert_and_rate_limit(monkeypatch):
    from fastapi import HTTPException

    from backend import app as app_mod

    def unknown(*a, **k):
        raise an.UnknownAlert("No alert with id x.")

    def limited(*a, **k):
        raise an.RateLimited(12.3)
    monkeypatch.setattr(an, "explain", unknown)
    with pytest.raises(HTTPException) as e:
        app_mod.api_analyst(app_mod.AnalystRequest(kind="alert", alert_id="market_stress:market:2001-01-01"))
    assert e.value.status_code == 404
    monkeypatch.setattr(an, "explain", limited)
    with pytest.raises(HTTPException) as e:
        app_mod.api_analyst(app_mod.AnalystRequest(kind="summary"))
    assert e.value.status_code == 429 and e.value.headers["Retry-After"] == "13" and e.value.detail["status"] == "rate_limited"
