"""Plain-language anomaly report logic (backend/regime_service.py). Uses the shipped models and
forecast files only, no network."""
import pandas as pd
import pytest

from backend import regime_service as rs


def test_weekly_move_converts_annualized_volatility():
    # 25.2% annualized -> 25.2 * sqrt(5/252) = 3.55% for a typical week
    assert rs.weekly(0.252) == pytest.approx(3.55, abs=0.01)


def test_mood_bands_follow_the_documented_percentiles():
    assert [rs.mood_for(p)["name"] for p in (0.10, 0.50, 0.80, 0.95)] == ["Calm", "Normal", "Choppy", "Stormy"]


def test_mood_scale_reports_band_edges_in_weekly_moves():
    hist = pd.Series([0.1 + i / 1000 for i in range(400)], index=pd.bdate_range("2019-01-01", periods=400))
    m = rs.mood_for(0.5, hist, 0.3)
    cuts = m["scale"]["cuts"]
    assert cuts == sorted(cuts) and len(cuts) == 3
    assert m["scale"]["value"] == rs.weekly(0.3)


def test_consecutive_surprise_days_merge_into_one_episode():
    days = [{"date": d, "actual": 0.5, "forecast": 0.2, "ratio": r}
            for d, r in [("2026-08-20", 2.3), ("2026-08-21", 2.6), ("2026-08-24", 2.4), ("2026-05-29", 2.5)]]
    events = rs._events(days)
    assert len(events) == 2
    assert events[0]["peak"] == "2026-08-21" and events[0]["days"] == 3  # most recent first, biggest day kept


def test_track_record_matches_the_documented_results():
    t = rs.track_record()
    assert t["periods"] == 30
    assert t["beats_simple"] == 29
    assert t["miss"] == {"ours": 1.96, "simple": 2.24, "original": 1.98}
    assert t["research"]["windows_better"] == 25
    assert t["research"]["p_value"] < 0.001
