"""Price loading and cleaning (backend/data.py), with Yahoo's response faked offline."""
import time

import numpy as np
import pandas as pd
import pytest

from backend import data


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Each test gets empty cache and snapshot folders and a clean Yahoo cooldown."""
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(data, "SNAPSHOT_DIR", tmp_path / "snapshot")
    monkeypatch.delenv("ANOMALITY_OFFLINE", raising=False)
    data._cache.clear()
    data._status.update(cooldown_until=0.0, last_error=None)
    yield
    data._cache.clear()
    data._status.update(cooldown_until=0.0, last_error=None)


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


# ---------------------------------------------------------------- saved prices: cache, snapshot, cooldown
def _frame(days=400, start="2025-01-01", price=100.0):
    idx = pd.bdate_range(start, periods=days)
    c = price * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, days)))
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6}, index=idx)


def _info(sym):
    return {"symbol": sym, "name": sym, "currency": "USD", "exchange": "X", "type": "EQUITY", "synthetic": False}


def test_a_good_download_is_saved_and_served_when_yahoo_rate_limits(monkeypatch):
    monkeypatch.setattr(data, "_fetch_yahoo", lambda s, p: (_frame(), _info(s)))
    df1, info1 = data.load_prices("AAA", "1y")
    assert not info1.get("cached")
    assert (data.CACHE_DIR / "AAA_1y.csv.gz").exists()

    def rate_limited(s, p):
        raise data.requests.HTTPError("429 Too Many Requests")
    monkeypatch.setattr(data, "_fetch_yahoo", rate_limited)
    monkeypatch.setattr(data, "CACHE_TTL", 0)  # the saved copy is no longer "fresh": Yahoo is tried first
    data._cache.clear()
    df2, info2 = data.load_prices("AAA", "1y")
    assert info2["cached"] is True and info2["source"] == "cache" and info2["synthetic"] is False
    assert df2.index[-1] == df1.index[-1] and (df2.index[-1] - df2.index[0]).days <= 365  # trimmed to 1y
    assert df2["close"].tolist() == pytest.approx(df1["close"].iloc[-len(df2):].tolist(), rel=1e-7)


def test_after_a_failure_yahoo_is_paused_instead_of_hit_again(monkeypatch):
    calls = []
    def boom(s, p):
        calls.append(s)
        raise data.requests.ConnectionError("offline")
    monkeypatch.setattr(data, "_fetch_yahoo", boom)
    data.load_prices("ZZA", "1y")
    data.load_prices("ZZB", "1y")
    assert calls == ["ZZA"]                      # the second asset didn't wait for another error
    assert data.status()["yahoo_paused_for"] > 0


def test_the_snapshot_serves_a_fresh_clone_and_is_cut_to_the_requested_range(monkeypatch):
    data._write(_frame(days=2600, start="2016-11-01"), _info("SNAP"), data.SNAPSHOT_DIR / "SNAP.csv.gz", {"saved_at": time.time() - 86400})
    monkeypatch.setenv("ANOMALITY_OFFLINE", "1")
    monkeypatch.setattr(data, "_fetch_yahoo", lambda s, p: pytest.fail("offline mode must not call Yahoo"))
    df, info = data.load_prices("SNAP", "2y")
    assert info["cached"] and info["source"] == "snapshot"
    assert (df.index[-1] - df.index[0]).days <= 731 and len(df) > 400


def test_the_newest_saved_copy_wins_and_a_short_copy_cannot_stand_in_for_a_long_range(monkeypatch):
    old, new = _frame(days=2600, start="2016-01-01", price=100), _frame(days=600, start="2024-06-01", price=200)
    data._write(old, _info("MIX"), data.SNAPSHOT_DIR / "MIX.csv.gz", {"saved_at": time.time() - 7 * 86400})
    data._write(new, _info("MIX"), data.CACHE_DIR / "MIX_2y.csv.gz", {"saved_at": time.time() - 3600})
    monkeypatch.setenv("ANOMALITY_OFFLINE", "1")
    _, two = data.load_prices("MIX", "2y")
    _, five = data.load_prices("MIX", "5y")
    assert two["source"] == "cache"              # newer
    assert five["source"] == "snapshot"          # the 2y cache can't cover 5 years


def test_no_saved_prices_and_no_synthetic_raises_the_original_error(monkeypatch):
    def boom(s, p):
        raise data.requests.ConnectionError("offline")
    monkeypatch.setattr(data, "_fetch_yahoo", boom)
    with pytest.raises(data.requests.ConnectionError):
        data.load_prices("NONE", "1y", allow_synthetic=False)
