"""Price loading and cleaning (backend/data.py), with Yahoo's response faked offline."""
import time

import pytest

from backend import data


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise data.requests.HTTPError(f"HTTP {self.status_code}")


def _sydney_payload(session_open: bool):
    # Sydney sessions open 10:00 local = 23:00 UTC the previous day. Two closed sessions plus,
    # optionally, a still-open one that Yahoo also repeats as a "live" row.
    ts = [1792623600, 1792710000]           # 2026-10-21 23:00Z, 2026-10-22 23:00Z  -> Oct 22, Oct 23 local
    closes, adj = [50.0, 51.0], [49.0, 50.0]
    now = int(time.time())
    if session_open:
        ts += [1792796400, 1792813600]      # 2026-10-23 23:00Z (Oct 24 local) + live update same session
        closes += [52.0, 52.5]
        adj += [52.0, 52.5]
        regular = {"start": 1792796400, "end": now + 3600}
    else:
        regular = {"start": 1792710000, "end": now - 3600}
    n = len(ts)
    return {"chart": {"error": None, "result": [{
        "meta": {"symbol": "BHP.AX", "exchangeTimezoneName": "Australia/Sydney", "currency": "AUD",
                 "currentTradingPeriod": {"regular": regular}},
        "timestamp": ts,
        "indicators": {"quote": [{"open": closes, "high": closes, "low": closes, "close": closes,
                                  "volume": [1000] * n}], "adjclose": [{"adjclose": adj}]},
    }]}}


def test_dates_use_the_exchange_time_zone_and_prices_are_dividend_adjusted(monkeypatch):
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: FakeResponse(_sydney_payload(session_open=False)))
    df, info = data._fetch_yahoo("BHP.AX", "1y")
    assert [d.strftime("%Y-%m-%d") for d in df.index] == ["2026-10-22", "2026-10-23"]
    assert df["close"].tolist() == pytest.approx([49.0, 50.0])  # adjclose applied
    assert info["timezone"] == "Australia/Sydney" and not info["session_in_progress"]


def test_an_unfinished_session_and_its_live_duplicate_are_dropped(monkeypatch):
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: FakeResponse(_sydney_payload(session_open=True)))
    df, info = data._fetch_yahoo("BHP.AX", "1y")
    assert df.index.is_unique
    assert df.index[-1].strftime("%Y-%m-%d") == "2026-10-23"
    assert info["session_in_progress"]


def test_unknown_symbol_raises_instead_of_falling_back(monkeypatch):
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: FakeResponse({}, status=404))
    data._cache.clear()
    with pytest.raises(data.DataError):
        data.load_prices("NOPE123", "1y")


def test_network_failure_falls_back_to_clearly_labelled_synthetic_data(monkeypatch):
    def boom(*a, **k):
        raise data.requests.ConnectionError("offline")
    monkeypatch.setattr(data.requests, "get", boom)
    data._cache.clear()
    df, info = data.load_prices("ZZZTEST", "1y")
    assert info["synthetic"] is True and len(df) > 150


def test_a_code_bug_surfaces_instead_of_hiding_behind_synthetic_data(monkeypatch):
    def bug(*a, **k):
        raise KeyError("unexpected field")  # not a network problem
    monkeypatch.setattr(data.requests, "get", bug)
    data._cache.clear()
    with pytest.raises(KeyError):
        data.load_prices("ZZZBUG", "1y")
