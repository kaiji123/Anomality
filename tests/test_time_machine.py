"""Historical Time Machine (backend/time_machine.py): what was known at t, kept apart from what
happened after t. Synthetic panels for the edge cases; the shipped data for invariance. Offline."""
import copy

import numpy as np
import pandas as pd
import pytest

from backend import time_machine as tm

STOCKS, CRYPTO = ["AAA", "BBB", "CCC"], ["COIN-USD"]


def synthetic(n_days=90, extra_price_days=10, seed=0):
    """U.S.-style trading calendar (weekdays), 3 stocks + 1 crypto sampled on it. Prices run
    `extra_price_days` past the last forecast, like the real data."""
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2021-01-04", periods=n_days + extra_price_days)
    fdays = cal[:n_days]
    rets = pd.DataFrame(rng.normal(0, 0.02, (len(cal), 4)), index=cal, columns=STOCKS + CRYPTO)
    rows = []
    for sym in STOCKS + CRYPTO:
        for i, d in enumerate(fdays):
            pred = 0.2 + 0.1 * np.sin(i / 9) + (0.3 if 60 <= i < 70 else 0)  # a storm late in the sample
            rows.append({"symbol": sym, "date": d, "pred": pred, "rv20": pred * (1.5 if i % 7 == 0 else 1.0),
                         "is_crypto": sym in CRYPTO, "ret": rets.loc[d, sym], "y": pred * 1.1, "pred_naive": pred})
    df = pd.DataFrame(rows).sort_values(["symbol", "date"], ignore_index=True)
    return tm.Data(df=df, calendar=cal, rets=rets)


def known_part(v):
    out = copy.deepcopy({k: v[k] for k in ("status", "timing", "known")})
    out["known"]["decision"]["mood_rule"].pop("matches_backtest", None)  # compares with the full backtest by design
    return out, v["series"]["known"]


# ---------- valid date, timing, outcome ----------
def test_valid_date_known_timing_and_outcome():
    d = synthetic()
    t = d.df["date"].unique()[70]
    v = tm.view("AAA", str(pd.Timestamp(t).date()), d)
    row = d.df[(d.df.symbol == "AAA") & (d.df.date == t)].iloc[0]
    assert v["status"] == "ok" and v["known"]["date"] == str(pd.Timestamp(t).date())
    assert v["known"]["forecast"]["vol"] == pytest.approx(row["pred"], abs=1e-4)
    assert v["known"]["rv20"] == pytest.approx(row["rv20"], abs=1e-4)
    cal = d.calendar[d.calendar > t]
    assert v["timing"]["traded"] == str(cal[0].date()) and v["timing"]["first_return"] == str(cal[1].date())
    o = v["outcome"]
    assert o["status"] == "complete" and o["start"] == str(cal[0].date()) and o["end"] == str(cal[tm.HORIZON - 1].date())
    expected_move = (1 + d.rets.loc[cal[:tm.HORIZON], "AAA"]).prod() - 1
    assert o["move_pct"] == pytest.approx(expected_move * 100, abs=0.01)
    assert o["realized_vol"] == pytest.approx(row["y"], abs=1e-4) and o["ratio"] == pytest.approx(1.1, abs=1e-3)


def test_unknown_asset_and_bad_date():
    d = synthetic()
    with pytest.raises(tm.UnknownAsset):
        tm.view("ZZZ", "2021-03-01", d)
    with pytest.raises(tm.BadDate):
        tm.view("AAA", "2021-13-45", d)


# ---------- no forecast, calendar ----------
def test_dates_without_a_forecast():
    d = synthetic()
    before = tm.view("AAA", "2020-12-01", d)
    after = tm.view("AAA", "2030-01-01", d)
    assert before["status"] == "before_history" and before["known"] is None and before["outcome"] is None
    assert after["status"] == "after_history" and after["known"] is None


def test_weekend_uses_the_previous_trading_day_never_a_later_one():
    d = synthetic()
    for sym in ("AAA", "COIN-USD"):  # crypto trades on weekends, but its forecasts follow the U.S. calendar
        v = tm.view(sym, "2021-01-09", d)  # a Saturday
        assert v["known"]["date"] == "2021-01-08" and "not a trading day" in v["note"]


# ---------- missing inputs and results ----------
def test_missing_rv20_disables_only_the_rule_that_needs_it():
    d = synthetic()
    t = d.df["date"].unique()[70]
    d.df.loc[(d.df.symbol == "AAA") & (d.df.date == t), "rv20"] = np.nan
    v = tm.view("AAA", str(pd.Timestamp(t).date()), d)
    k = v["known"]
    assert k["rv20"] is None and k["surprise_ratio"] is None
    assert k["decision"]["previous_rule"] is None and any("rv20" in f for f in k["flags"])
    assert k["decision"]["mood_rule"]["level"] in {"Full", "Half"}  # the mood rule doesn't use rv20


def test_missing_realized_result_is_reported_not_filled():
    d = synthetic()
    t = d.df["date"].unique()[50]
    d.df.loc[(d.df.symbol == "BBB") & (d.df.date == t), "y"] = np.nan
    o = tm.view("BBB", str(pd.Timestamp(t).date()), d)["outcome"]
    assert o["status"] == "unavailable" and "realized_vol" not in o


def test_incomplete_horizon_says_not_available_yet():
    d = synthetic(extra_price_days=3)  # only 3 price days after the last forecast
    last = d.df["date"].max()
    o = tm.view("CCC", str(last.date()), d)["outcome"]
    assert o["status"] == "incomplete" and o["days_available"] == 3 and "not available yet" in o["text"]


def test_warm_up_is_flagged():
    d = synthetic()
    v = tm.view("AAA", str(d.df["date"].min().date()), d)
    k = v["known"]
    assert k["decision"]["mood_rule"]["storminess"] is None and k["decision"]["mood_rule"]["level"] == "Full"
    assert any("warming up" in f for f in k["flags"])


def test_storm_in_the_known_window_cuts_exposure():
    d = synthetic()
    t = d.df["date"].unique()[62]  # inside the synthetic storm
    k = tm.view("AAA", str(pd.Timestamp(t).date()), d)["known"]
    assert k["decision"]["mood_rule"]["level"] == "Half" and k["decision"]["mood_rule"]["reason"].startswith("stormy")


# ---------- no look-ahead ----------
def _perturb_after(data, t, seed=1):
    """Change or delete everything strictly after t: forecasts, inputs, results and prices."""
    rng = np.random.default_rng(seed)
    df = data.df.copy()
    late = df["date"] > t
    df.loc[late, "pred"] = df.loc[late, "pred"] * rng.uniform(0.2, 5, late.sum())
    df.loc[late, "rv20"] = df.loc[late, "rv20"] * rng.uniform(0.2, 5, late.sum())
    df.loc[late, "y"] = df.loc[late, "y"] * 3
    df.loc[late, "ret"] = -df.loc[late, "ret"]
    df = df[~late | (rng.random(len(df)) < 0.6)]  # and drop ~40% of the later rows
    rets = data.rets.copy()
    rets.loc[rets.index > t] = rets.loc[rets.index > t] * -2
    return tm.Data(df=df.reset_index(drop=True), calendar=data.calendar, rets=rets)


def test_known_part_ignores_the_future_synthetic():
    d = synthetic()
    t = pd.Timestamp(d.df["date"].unique()[64])
    base = tm.view("AAA", str(t.date()), d)
    changed = tm.view("AAA", str(t.date()), _perturb_after(d, t))
    truncated = tm.view("AAA", str(t.date()), tm.Data(df=d.df[d.df.date <= t], calendar=d.calendar, rets=d.rets))
    assert known_part(base) == known_part(changed) == known_part(truncated)
    assert base["outcome"] != changed["outcome"]  # the perturbation did reach the future part


@pytest.fixture(scope="module")
def real():
    mp = pytest.MonkeyPatch()
    mp.setenv("ANOMALITY_OFFLINE", "1")
    tm._cache.clear()
    try:
        yield tm.load()
    finally:
        mp.undo()


@pytest.mark.parametrize("symbol,date", [("NVDA", "2020-03-13"), ("BTC-USD", "2022-06-13"), ("SPY", "2024-08-05")])
def test_known_part_ignores_the_future_on_the_real_data(real, symbol, date):
    """The real code on the shipped data: rewriting or deleting every row after t (for all 56
    assets, since the mood rule looks at the whole market) changes nothing known at t."""
    t = pd.Timestamp(date)
    base = tm.view(symbol, date, real)
    changed = tm.view(symbol, date, _perturb_after(real, t))
    assert base["status"] == "ok"
    assert known_part(base) == known_part(changed)
    assert base["outcome"] != changed["outcome"]


@pytest.mark.parametrize("symbol,date", [("NVDA", "2020-03-16"), ("AAPL", "2022-06-16"), ("ETH-USD", "2021-05-19"), ("SPY", "2026-10-16")])
def test_reconstructed_decision_is_the_backtests_decision(real, symbol, date):
    k = tm.view(symbol, date, real)["known"]
    assert k["decision"]["mood_rule"]["matches_backtest"] is True


def test_real_example_matches_the_stored_forecast(real):
    v = tm.view("NVDA", "2020-03-13", real)
    assert v["known"]["forecast"]["vol"] == pytest.approx(0.5915, abs=1e-4)
    assert v["outcome"]["status"] == "complete" and v["outcome"]["surprise"] is True  # realized ~3x the forecast
    assert v["timing"] == {**v["timing"], "traded": "2020-03-16", "first_return": "2020-03-17"}
    assert all(d <= "2020-03-13" for d in v["series"]["known"]["date"])
    assert all(d > "2020-03-13" for d in v["series"]["after"]["date"])


# ---------- caches are tied to the data object, bounded, and replaced when data changes ----------
def test_full_simulation_is_reused_then_replaced_not_accumulated(monkeypatch):
    import backend.decision_service as ds
    calls = []
    real_simulate = ds.simulate
    monkeypatch.setattr(ds, "simulate", lambda df, cfg: calls.append(len(df)) or real_simulate(df, cfg))
    tm._cache.clear()
    a, b = synthetic(seed=1), synthetic(seed=2)
    tm._full_simulation(a)
    tm._full_simulation(a)
    assert len(calls) == 1  # same data object: computed once
    tm._full_simulation(b)
    assert len(calls) == 2 and tm._cache["full"][0] is b  # new data: recomputed, the old result is gone
    assert sum(1 for k in tm._cache if k == "full" or (isinstance(k, tuple) and k[0] == "full")) == 1


def test_a_reused_object_id_cannot_return_another_datasets_result():
    """Python reuses ids of freed objects; the cache must compare the object itself."""
    tm._cache.clear()
    a = synthetic(seed=1)
    first = tm._full_simulation(a)
    b = tm.Data(df=a.df.assign(pred=a.df["pred"] * 3), calendar=a.calendar, rets=a.rets)  # different content
    tm._cache["full"] = (object(), first)  # a slot whose owner is gone (as if its id were reused)
    assert tm._full_simulation(b) is not first
    assert tm._cache["full"][0] is b


def test_view_cache_is_bounded_and_dropped_when_data_changes(monkeypatch):
    tm._cache.clear()
    monkeypatch.setattr(tm, "MAX_VIEWS", 3)
    a = synthetic(seed=1)
    days = [str(pd.Timestamp(x).date()) for x in a.df["date"].unique()[60:66]]
    v1 = tm.cached_view("AAA", days[0], a)
    assert tm.cached_view("AAA", days[0], a) is v1  # reused
    for d in days[1:]:
        tm.cached_view("AAA", d, a)
    assert len(tm._cache["views"][1]) == 3  # bounded (least recently used dropped)
    b = synthetic(seed=2)
    v2 = tm.cached_view("AAA", days[0], b)
    assert v2 is not v1 and tm._cache["views"][0] is b and len(tm._cache["views"][1]) == 1


# ---------- API ----------
def test_api_handlers(real):
    from fastapi import HTTPException

    from backend import app as app_mod
    m = app_mod.api_time_machine_meta()
    assert len(m["assets"]) == 56 and m["horizon_days"] == 5
    assert app_mod.api_time_machine("nvda", date="2020-03-13")["symbol"] == "NVDA"
    assert app_mod.api_time_machine("NVDA", date="2010-01-04")["status"] == "before_history"
    for sym, date, code in [("GME", "2020-03-13", 404), ("NVDA", "2020-02-30", 422)]:
        with pytest.raises(HTTPException) as e:
            app_mod.api_time_machine(sym, date=date)
        assert e.value.status_code == code
