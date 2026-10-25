"""Intelligent alerts and drift detection (backend/alerts.py). Synthetic panels with known
answers, plus the shipped data for look-ahead. Offline."""
import numpy as np
import pandas as pd
import pytest

from backend import alerts as al
from backend.time_machine import HORIZON, Data

SYMS = [f"S{i:02d}" for i in range(12)] + ["COIN-USD"]


def panel(n_days=320, extra_price_days=10, seed=0):
    """Calm, stationary panel: forecasts, rv20 and results all near 20% vol with noise."""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2020-01-06", periods=n_days + extra_price_days)
    days = cal[:n_days]
    rets = pd.DataFrame(rng.normal(0, 0.01, (len(cal), len(SYMS))), index=cal, columns=SYMS)
    rows = []
    for j, s in enumerate(SYMS):
        level = 0.15 + 0.01 * j
        pred = level * np.exp(rng.normal(0, 0.15, n_days))
        rv20 = pred * np.exp(rng.normal(0, 0.10, n_days))
        y = pred * np.exp(rng.normal(0, 0.10, n_days))
        for i, d in enumerate(days):
            rows.append({"symbol": s, "date": d, "pred": pred[i], "rv20": rv20[i], "is_crypto": s.endswith("-USD"),
                         "ret": rets.loc[d, s], "y": y[i], "pred_naive": rv20[i]})
    df = pd.DataFrame(rows).sort_values(["symbol", "date"], ignore_index=True)
    return Data(df=df, calendar=cal, rets=rets)


def check(result, monitor):
    return next(c for c in result["checks"] if c["monitor"] == monitor)


def last_day(d):
    return d.df["date"].max()


# ---------- statistics ----------
def test_psi_is_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 5000)
    assert al.psi(a, rng.normal(0, 1, 5000)) < 0.02
    assert al.psi(a, rng.normal(1.5, 1, 1000)) > al.PSI_MAJOR


# ---------- thresholds and boundaries ----------
@pytest.mark.parametrize("value,status", [(0.90, "alert"), (0.8999, "ok"), (0.97, "alert"), (0.30, "ok")])
def test_market_stress_threshold_is_inclusive(value, status):
    d = panel(n_days=80)
    t = last_day(d)
    c = al.market_checks(d, t, pd.Series({t: value}))[0]
    assert c.status == status and c.threshold == al.STORMY


@pytest.mark.parametrize("ratio,status", [(2.0, "alert"), (1.999, "ok")])
def test_surprise_threshold_is_inclusive(ratio, status):
    d = panel(n_days=80)
    t = last_day(d)
    m = (d.df["date"] == t) & (d.df["symbol"] == "S03")
    d.df.loc[m, "rv20"] = d.df.loc[m, "pred"] * ratio
    d.df.loc[(d.df["date"] == t) & (d.df["symbol"] != "S03"), "rv20"] *= 0.5  # keep the others well below
    c = al.surprise_checks(d, t)[0]
    assert c.status == status
    if status == "alert":
        assert c.evidence["assets"][0]["symbol"] == "S03"


def test_no_alerts_when_nothing_is_wrong():
    d = panel()
    r = al.evaluate(d, last_day(d), now=d.calendar.max())
    firing = [c["monitor"] for c in r["checks"] if c["status"] == "alert"]
    assert "market_stress" not in firing and "input_drift" not in firing and "prediction_drift" not in firing
    assert not [c for c in r["checks"] if c["type"] == "data_quality" and c["status"] == "alert"]


# ---------- missing values, windows, staleness ----------
def test_invalid_and_missing_values_are_data_quality_alerts():
    d = panel(n_days=80)
    t = last_day(d)
    d.df.loc[(d.df["date"] == t) & (d.df["symbol"] == "S01"), "pred"] = np.nan
    d.df.loc[(d.df["date"] == t) & (d.df["symbol"] == "S02"), "ret"] = np.nan
    d.df = d.df[~((d.df["date"] == t) & (d.df["symbol"] == "S04"))]
    cs = {c.monitor: c for c in al.data_quality_checks(d, t, now=None)}
    assert cs["invalid_values"].status == "alert" and cs["invalid_values"].evidence["assets"] == ["S01"]
    assert cs["missing_returns"].status == "alert" and cs["missing_returns"].evidence["assets"] == ["S02"]
    assert cs["missing_forecasts"].status == "alert" and cs["missing_forecasts"].evidence["assets"] == ["S04"]
    assert al.surprise_checks(d, t)[0].evidence["assets_checked"] == len(SYMS) - 2  # NaN forecast and S04 skipped


def test_a_market_holiday_is_not_a_data_problem():
    d = panel(n_days=80)
    holiday = d.df["date"].unique()[-2]
    d.df = d.df[d.df["date"] != holiday]  # every asset absent: the market was closed
    t = last_day(d)
    assert {c.monitor: c.status for c in al.data_quality_checks(d, t, None)}["missing_forecasts"] == "ok"


def test_short_history_gives_insufficient_data_not_alerts():
    d = panel(n_days=100)
    r = al.evaluate(d, last_day(d))
    for m in ("input_drift", "prediction_drift"):
        c = check(r, m)
        assert c["status"] == "insufficient_data" and "273" in c["text"]
    assert not [a for a in r["alerts"] if a["monitor"] in ("input_drift", "prediction_drift")]
    tiny = panel(n_days=20)
    assert check(al.evaluate(tiny, last_day(tiny)), "performance_drift")["status"] == "insufficient_data"


def test_stale_prices_only_against_the_clock():
    d = panel(n_days=80)
    t, last = last_day(d), d.calendar.max()
    fresh = al.evaluate(d, t, now=last + pd.Timedelta(days=1))
    stale = al.evaluate(d, t, now=last + pd.offsets.BDay(10))
    past = al.evaluate(d, t, now=None)
    assert check(fresh, "stale_prices")["status"] == "ok"
    assert check(stale, "stale_prices")["status"] == "alert" and check(stale, "stale_prices")["value"] == 10
    assert not [c for c in past["checks"] if c["monitor"] == "stale_prices"]


# ---------- drift: three different things ----------
def _shift(d, col, factor, last_n):
    days = sorted(d.df["date"].unique())[-last_n:]
    m = d.df["date"].isin(days)
    d.df.loc[m, col] = d.df.loc[m, col] * factor


def test_input_drift_without_prediction_drift():
    d = panel()
    _shift(d, "rv20", 2.5, al.RECENT_DAYS)
    r = al.evaluate(d, last_day(d))
    assert check(r, "input_drift")["status"] == "alert" and check(r, "prediction_drift")["status"] == "ok"


def test_prediction_drift_without_input_drift():
    d = panel()
    _shift(d, "pred", 2.5, al.RECENT_DAYS)
    r = al.evaluate(d, last_day(d))
    assert check(r, "prediction_drift")["status"] == "alert" and check(r, "input_drift")["status"] == "ok"


def test_performance_drift_uses_only_completed_horizons():
    d = panel()
    base = check(al.evaluate(d, last_day(d)), "performance_drift")
    assert base["status"] == "ok"
    # Results of the last HORIZON days are not known yet at t: wrecking them changes nothing.
    _shift(d, "y", 10, HORIZON)
    assert check(al.evaluate(d, last_day(d)), "performance_drift") == base
    # A wilder market alone is not a model problem: scaling every result hurts the simple
    # estimate as much as the model, so the relative score barely moves.
    days = sorted(d.df["date"].unique())
    known = d.df["date"].isin(days[-(al.RECENT_DAYS + HORIZON):-HORIZON])
    wild = Data(df=d.df.assign(y=np.where(known, d.df["y"] * 4, d.df["y"])), calendar=d.calendar, rets=d.rets)
    assert check(al.evaluate(wild, last_day(d)), "performance_drift")["status"] == "ok"
    # The model losing its skill on results that ARE known at t fires it: shuffle the forecasts
    # across assets each day (same distribution, so no prediction drift; ranking destroyed).
    rng = np.random.default_rng(5)
    for day in days[-(al.RECENT_DAYS + HORIZON):-HORIZON]:
        m = d.df["date"] == day
        d.df.loc[m, "pred"] = rng.permutation(d.df.loc[m, "pred"].to_numpy())
    r = al.evaluate(d, last_day(d))
    perf = check(r, "performance_drift")
    assert perf["status"] == "alert" and perf["value"] > al.NO_SKILL
    assert perf["evidence"]["window"][1] == str(days[-HORIZON - 1].date())
    assert check(r, "prediction_drift")["status"] == "ok"


# ---------- episodes, ids, lifecycle ----------
def _storm_panel():
    """A storm: market forecasts jump for days 200-214, then calm again (300 days in total)."""
    d = panel(n_days=300)
    days = sorted(d.df["date"].unique())
    m = d.df["date"].isin(days[200:215])
    d.df.loc[m, "pred"] *= 3
    return d, days


def test_same_episode_keeps_one_id_across_refreshes():
    d, days = _storm_panel()
    a1 = al.evaluate(d, days[205])
    a2 = al.evaluate(d, days[210])
    ids1 = {a["id"] for a in a1["alerts"] if a["monitor"] == "market_stress" and a["status"] == "active"}
    ids2 = {a["id"] for a in a2["alerts"] if a["monitor"] == "market_stress" and a["status"] == "active"}
    assert ids1 == ids2 and len(ids1) == 1
    assert al.evaluate(d, days[210])["alerts"] == a2["alerts"]  # deterministic, no duplicates on refresh


def test_an_episode_longer_than_the_lookback_keeps_its_id_and_true_start():
    """Regression: the start used to be cut at the 20-day window, so a long episode got a new id
    (and a wrong first-seen date) every day."""
    d = panel(n_days=300)
    days = sorted(d.df["date"].unique())
    # A 41-day storm (twice the lookback window) that keeps building, so every day is busier than
    # all earlier ones; a flat storm would fall below 90% once its own days fill the history.
    for k, day in enumerate(days[200:241]):
        m = d.df["date"] == day
        d.df.loc[m, "pred"] *= 3 + 0.2 * k
    seen = []
    for i in (215, 225, 232, 240):  # several consecutive evaluations, all past the window
        a = next(x for x in al.evaluate(d, days[i])["alerts"] if x["monitor"] == "market_stress" and x["status"] == "active")
        seen.append((a["id"], a["first_seen"], a["first_seen_exact"]))
    assert len(set(seen)) == 1
    assert seen[0] == (f"market_stress:market:{days[200].date()}", str(days[200].date()), True)
    ended = al.evaluate(d, days[245])  # resolved episodes keep the same id and start too
    r = next(x for x in ended["alerts"] if x["monitor"] == "market_stress")
    assert r["status"] == "resolved" and r["id"] == seen[0][0]


def test_an_episode_running_back_to_the_first_day_of_data_is_flagged():
    d = panel(n_days=60)
    days = sorted(d.df["date"].unique())
    d.df.loc[d.df["symbol"] == "S02", "ret"] = np.nan  # no price returns from the very first day
    a = next(x for x in al.evaluate(d, days[40])["alerts"] if x["monitor"] == "missing_returns")
    assert a["first_seen"] == str(days[0].date()) and a["first_seen_exact"] is False


def test_episode_resolves_after_the_condition_ends():
    d, days = _storm_panel()
    r = al.evaluate(d, days[218])
    ms = [a for a in r["alerts"] if a["monitor"] == "market_stress"]
    assert len(ms) == 1 and ms[0]["status"] == "resolved"
    assert ms[0]["first_seen"] == str(days[200].date()) and ms[0]["resolved_on"] > ms[0]["last_seen"]


def test_acknowledge_then_resolve():
    d, days = _storm_panel()
    active = al.evaluate(d, days[205])
    aid = next(a["id"] for a in active["alerts"] if a["monitor"] == "market_stress")
    al.acknowledge(aid, when="2026-10-25T10:00:00")
    try:
        assert next(a for a in al.with_acks(active)["alerts"] if a["id"] == aid)["status"] == "acknowledged"
        later = al.with_acks(al.evaluate(d, days[218]))
        assert next(a for a in later["alerts"] if a["id"] == aid)["status"] == "resolved"  # gone, whatever the ack
    finally:
        al._acks.pop(aid, None)


def test_regime_change_is_active_for_a_while_then_ages_out():
    d, days = _storm_panel()
    early = check(al.evaluate(d, days[200 + al.PERSIST_DAYS - 2]), "regime_change")
    assert early["status"] == "ok"  # changed, but not held long enough yet
    held = check(al.evaluate(d, days[200 + al.PERSIST_DAYS]), "regime_change")
    assert held["status"] == "alert" and held["evidence"]["to"] == "Stormy"


# ---------- no look-ahead ----------
def _perturb_after(data, t, seed=3):
    rng = np.random.default_rng(seed)
    df = data.df.copy()
    late = df["date"] > t
    for col in ("pred", "rv20", "y"):
        df.loc[late, col] = df.loc[late, col] * rng.uniform(0.1, 10, late.sum())
    df = df[~late | (rng.random(len(df)) < 0.5)]
    return Data(df=df.reset_index(drop=True), calendar=data.calendar, rets=data.rets * -3)


def test_alerts_on_a_past_day_ignore_later_data_synthetic():
    d, days = _storm_panel()
    t = days[260]
    assert al.evaluate(d, t) == al.evaluate(_perturb_after(d, t), t)


@pytest.fixture(scope="module")
def real():
    from backend import time_machine as tm
    mp = pytest.MonkeyPatch()
    mp.setenv("ANOMALITY_OFFLINE", "1")
    try:
        yield tm.load()
    finally:
        mp.undo()


@pytest.mark.parametrize("date", ["2020-03-13", "2022-06-16"])
def test_alerts_on_a_past_day_ignore_later_data_real(real, date):
    t = pd.Timestamp(date)
    base = al.evaluate(real, t)
    assert al.evaluate(_perturb_after(real, t), t) == base
    assert any(a["status"] == "active" for a in base["alerts"])  # these were stressed days


def test_real_covid_alerts(real):
    r = al.evaluate(real, pd.Timestamp("2020-03-13"))
    active = {a["monitor"] for a in r["alerts"] if a["status"] == "active"}
    assert {"market_stress", "regime_change"} <= active
    assert check(r, "input_drift")["status"] == "insufficient_data"  # under 273 days of forecast history then


# ---------- API ----------
def test_api_handlers(real):
    from fastapi import HTTPException

    from backend import app as app_mod
    r = app_mod.api_alerts(date="2020-03-13")
    assert r["as_of"] == "2020-03-13" and r["config"]["stormy_pct"] == al.STORMY
    for a in r["alerts"]:
        assert {"id", "type", "severity", "title", "description", "observed_value", "threshold", "source",
                "scope", "status", "evidence", "first_seen", "last_seen"} <= set(a)
    with pytest.raises(HTTPException) as e:
        app_mod.api_alerts(date="2010-01-04")
    assert e.value.status_code == 422
    aid = r["alerts"][0]["id"]
    try:
        assert app_mod.api_alert_ack(aid)["id"] == aid
    finally:
        al._acks.pop(aid, None)


def test_alert_results_cache_follows_the_data_object(monkeypatch):
    from backend import time_machine as tm
    a, b = panel(n_days=80, seed=1), panel(n_days=80, seed=2)
    current = {"data": a}
    monkeypatch.setattr(tm, "load", lambda: current["data"])
    al._results.clear()
    try:
        r1 = al.latest(str(last_day(a).date()))
        assert al.latest(str(last_day(a).date()))["checks"] == r1["checks"] and al._results["slot"][0] is a
        current["data"] = b  # new data: the old results are dropped, not reused
        al.latest(str(last_day(b).date()))
        assert al._results["slot"][0] is b and len(al._results["slot"][1]) == 1
    finally:
        al._results.clear()
