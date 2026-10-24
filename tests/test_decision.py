"""Decision layer: the rules, and proof the backtest is causal and costs are exact."""
import numpy as np
import pandas as pd
import pytest

from backend.backtest import backtest, hold, portfolio_returns, stats
from backend.decision import DecisionConfig, MoodConfig, decide, mood_positions, positions, storminess
from backend.decision_service import RULE, explain, market_storminess, simulate, summarize
from backend.regime_service import MOODS


def test_exec_lag():
    """Signal flipped on day t changes position on day t+2 only."""
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    pos = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=idx)
    ret = pd.Series([0.01] * 5, index=idx)
    r = backtest(pos, ret, cost_bps=0.0, exec_lag=1)
    # pos.shift(2) = [0,0,0,1,1]; net = [0,0,0,.01,.01]; mean = .004; annualized = 1.008
    assert abs(r["ann_return"] - 1.008) < 1e-6, r["ann_return"]
    assert r["n_days"] == 5


def test_costs_exact():
    """One switch of size 1 costs exactly 5 bps once."""
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    pos = pd.Series([0.0] + [1.0] * 9, index=idx)
    ret = pd.Series([0.0] * 10, index=idx)
    r = backtest(pos, ret, cost_bps=5.0, exec_lag=1)
    assert abs(r["net_return"] + 5e-4) < 1e-6, r["net_return"]


def test_hysteresis():
    cfg = DecisionConfig(stress_surprise=2.0, min_hold_days=5, variant="tiered")
    exp, _ = decide(0.10, 0.10, 0, 1.0, cfg)
    assert exp == 1.0
    exp, reason = decide(0.10, 0.30, 1, 1.0, cfg)  # 3x surprise but just switched -> hold
    assert exp == 1.0 and "hold" in reason
    exp, reason = decide(0.10, 0.30, 5, 1.0, cfg)  # hysteresis elapsed -> cash
    assert exp == 0.0 and "stress" in reason


def test_tiered():
    cfg = DecisionConfig(min_hold_days=0, variant="tiered")
    assert decide(0.10, 0.10, 0, 1.0, cfg)[0] == 1.0   # calm
    assert decide(0.10, 0.15, 0, 1.0, cfg)[0] == 0.5   # 1.5x choppy
    assert decide(0.10, 0.25, 0, 1.0, cfg)[0] == 0.0   # 2.5x stress


def test_positions_waits_min_hold_days_between_switches():
    cfg = DecisionConfig(min_hold_days=3, variant="binary")
    f = np.full(8, 0.10)
    rv = np.array([0.30, 0.10, 0.10, 0.10, 0.10, 0.30, 0.30, 0.30])
    exposure, reasons = positions(f, rv, cfg)
    # Starts invested, so the day-0 spike is held through; the later spike switches to cash.
    assert exposure.tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    assert reasons[0].startswith("hold") and reasons[5].startswith("stress")
    cfg0 = DecisionConfig(min_hold_days=0, variant="binary")
    assert positions(f, rv, cfg0)[0].tolist() == [0, 1, 1, 1, 1, 0, 0, 0]


def _two_assets(exposure_a, exposure_b, ret_a, ret_b):
    dates = pd.date_range("2021-01-04", periods=len(ret_a), freq="B")
    return pd.DataFrame({
        "symbol": ["A"] * len(dates) + ["B"] * len(dates),
        "date": list(dates) * 2,
        "exposure": list(exposure_a) + list(exposure_b),
        "ret": list(ret_a) + list(ret_b),
    })


def test_portfolio_counts_exposure_once():
    """Half exposure earns half the return, not a quarter (the old portfolio code multiplied twice)."""
    per = _two_assets([0.5] * 6, [0.5] * 6, [0.01] * 6, [0.01] * 6)
    d = portfolio_returns(hold(per), cost_bps=0.0)
    assert d["net"].iloc[2:].tolist() == pytest.approx([0.005] * 4)


def test_portfolio_decision_never_earns_its_own_day():
    """Going to cash on day t (after a crash on day t) can't dodge day t's or t+1's return."""
    ret = [0.0, 0.0, -0.10, -0.05, 0.0, 0.0]
    exposure = [1, 1, 0, 0, 0, 0]  # decided at the close of day 2, using day 2's crash
    per = _two_assets(exposure, exposure, ret, ret)
    d = portfolio_returns(hold(per), cost_bps=0.0)
    # Held exposure is the decision delayed 2 days: [0, 0, 1, 1, 0, 0].
    assert d["net"].tolist() == pytest.approx([0.0, 0.0, -0.10, -0.05, 0.0, 0.0])


def test_portfolio_of_one_asset_matches_single_backtest():
    rng = np.random.default_rng(0)
    ret = rng.normal(0, 0.02, 40)
    exposure = (rng.random(40) > 0.3).astype(float)
    per = _two_assets(exposure, exposure, ret, ret)
    per = per[per["symbol"] == "A"]
    d = portfolio_returns(hold(per), cost_bps=5.0)
    single = backtest(pd.Series(exposure, index=d.index), pd.Series(ret, index=d.index), cost_bps=5.0)
    assert stats(d["net"], d["turnover"]) == pytest.approx(single)


def test_summarize_on_synthetic_panel():
    """End to end without network: two assets, one with a volatility spike."""
    dates = pd.date_range("2020-01-01", "2023-12-29", freq="B")
    rng = np.random.default_rng(1)
    rows = []
    for sym, spike in (("A", True), ("B", False)):
        rv = np.full(len(dates), 0.2)
        if spike:
            rv[300:340] = 0.6
        rows.append(pd.DataFrame({"symbol": sym, "date": dates, "pred": 0.2, "rv20": rv,
                                  "is_crypto": False, "ret": rng.normal(0.0004, 0.01, len(dates))}))
    df = pd.concat(rows, ignore_index=True)
    s = summarize(df)
    assert set(s["systems"]) == {"mood_rule", "buy_and_hold", "surprise_rule"}
    assert s["systems"]["buy_and_hold"]["all"]["turnover"] < 0.3  # bought once
    assert s["systems"]["surprise_rule"]["all"]["turnover"] > s["systems"]["buy_and_hold"]["all"]["turnover"]
    assert "COVID crash" in s["systems"]["surprise_rule"]
    assert set(s["periods"]) == {k for k in s["systems"]["mood_rule"] if k not in ("all", "label")}
    tiered = simulate(df, DecisionConfig())
    assert (tiered[tiered["symbol"] == "A"]["exposure"] == 0.0).any()
    assert (tiered[tiered["symbol"] == "B"]["exposure"] == 1.0).all()


def test_explain_is_plain_language():
    assert "at half" in explain("stormy (93%)", 0.93)
    assert "93%" in explain("stormy (93%)", 0.93)
    assert "fully invested" in explain("not stormy (62%)", 0.62)
    assert "history" in explain("warming up", None)


# ---------------------------------------------------------------- mood rule
def test_stormy_threshold_matches_the_apps_mood_bands():
    """The rule's Stormy is exactly the app's Stormy (where the Choppy band ends)."""
    choppy_end = [cut for cut, name, _ in MOODS if name == "Choppy"][0]
    assert RULE.stormy_pct == choppy_end


def test_storminess_uses_only_the_past():
    v = np.array([1.0, 2.0, 3.0, 2.5, 0.5, 4.0])
    pct = storminess(v, min_history=2)
    assert np.isnan(pct[:2]).all()
    assert pct[2] == 1.0          # 3.0 is above both earlier days
    assert pct[3] == pytest.approx(2 / 3)
    assert pct[4] == 0.0
    future = v.copy(); future[5] = 100.0  # changing the last day never changes earlier days
    assert np.array_equal(storminess(future, 2)[:5], pct[:5], equal_nan=True)


def test_mood_positions_half_only_when_stormy():
    pct = np.array([np.nan, 0.5, 0.89, 0.90, 0.99])
    exposure, reasons = mood_positions(pct, MoodConfig())
    assert exposure.tolist() == [1.0, 1.0, 1.0, 0.5, 0.5]
    assert reasons[0] == "warming up" and reasons[3].startswith("stormy")


def test_mood_rule_follows_the_market_not_crypto_or_one_asset():
    """Every asset gets the market's call; the market forecast is the average over stocks only."""
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    calm = np.full(len(dates), 0.2)
    spike = calm.copy(); spike[100:] = 0.6          # market storm from day 100
    rows = [pd.DataFrame({"symbol": s, "date": dates, "pred": p, "rv20": 0.2, "is_crypto": c, "ret": 0.0})
            for s, p, c in (("A", spike, False), ("B", spike, False), ("BTC-USD", calm * 5, True))]
    df = pd.concat(rows, ignore_index=True)
    storm = market_storminess(df, MoodConfig(min_history=60))
    assert storm.iloc[:60].isna().all() and storm.iloc[100] == 1.0
    sim = simulate(df, MoodConfig(min_history=60))
    for _, g in sim.groupby("symbol"):
        assert (g["exposure"].iloc[:100] == 1.0).all() and (g["exposure"].iloc[100:106] == 0.5).all()
