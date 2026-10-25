"""Backtest data integrity and significance. Offline: shipped forecasts + the committed price
snapshot (data/snapshot/), no network."""
import numpy as np
import pandas as pd
import pytest

from backend import significance as sig


@pytest.fixture(scope="module")
def panel():
    mp = pytest.MonkeyPatch()
    mp.setenv("ANOMALITY_OFFLINE", "1")
    from backend import regime_service
    from backend.decision_service import frame
    try:
        yield regime_service.recommended_forecasts().reset_index(), frame(use_cache=False)
    finally:
        mp.undo()


# ---------- the join that once dropped every Monday ----------
def test_every_forecast_gets_a_return(panel):
    preds, df = panel
    assert len(df) == len(preds)  # nothing dropped by the price join
    assert not df.duplicated(["symbol", "date"]).any()
    assert np.isfinite(df["ret"]).all()


def test_mondays_and_post_holiday_days_are_kept(panel):
    _, df = panel
    days = df.drop_duplicates("date")["date"]
    assert days.dt.dayofweek.value_counts().get(0, 0) > 300  # ~1 in 5 trading days


def test_universe_only_changes_when_an_asset_starts(panel):
    """Fewer than all assets on a date only because an asset's forecasts start later; no holes."""
    _, df = panel
    wide = df.pivot(index="date", columns="symbol", values="ret")
    for sym in wide:
        s = wide[sym]
        first = s.first_valid_index()
        assert s.loc[first:].notna().all(), f"{sym} has missing days after it starts"
    late = {s for s in wide if wide[s].first_valid_index() > wide.index[0]}
    assert late <= {"SOL-USD", "AVAX-USD"}, late


# ---------- significance ----------
def test_bootstrap_is_reproducible_and_paired():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=500)
    a = pd.Series(rng.normal(0.0005, 0.01, 500), index=idx)
    r1 = sig.paired_bootstrap(a, a * 0.5, block=10, n_boot=400, seed=3)
    assert r1 == sig.paired_bootstrap(a, a * 0.5, block=10, n_boot=400, seed=3)
    # Scaling returns leaves Sharpe unchanged; pairing makes every resample see that exactly.
    assert r1["sharpe_diff"]["observed"] == pytest.approx(0, abs=1e-9)
    assert r1["sharpe_diff"]["ci95"] == pytest.approx([0, 0], abs=1e-9)


def test_bootstrap_detects_a_real_edge_and_not_a_fake_one():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=2000)
    base = rng.normal(0.0003, 0.01, 2000)
    better = pd.Series(base + 0.0008, index=idx)  # +20%/yr every day: a real edge
    noise = pd.Series(base + rng.normal(0, 0.002, 2000), index=idx)  # same mean, extra noise
    real = sig.paired_bootstrap(better, pd.Series(base, index=idx), n_boot=1000)
    fake = sig.paired_bootstrap(noise, pd.Series(base, index=idx), n_boot=1000)
    assert real["ann_return_diff"]["ci95"][0] > 0 and real["ann_return_diff"]["p_not_better"] < 0.01
    lo, hi = fake["ann_return_diff"]["ci95"]
    assert lo < 0 < hi


def test_block_indices_are_consecutive_and_in_range():
    idx = sig.block_indices(n=50, block=7, n_boot=20, rng=np.random.default_rng(0))
    assert idx.shape == (20, 50) and idx.min() >= 0 and idx.max() < 50
    steps = np.diff(idx[:, :7], axis=1) % 50
    assert (steps == 1).all()  # first block of every resample is 7 consecutive days


def test_too_few_days_is_an_error():
    s = pd.Series([0.01, 0.02, 0.0], index=pd.bdate_range("2020-01-01", periods=3))
    with pytest.raises(ValueError):
        sig.paired_bootstrap(s, s)


def test_mood_rule_significance_on_the_real_backtest(monkeypatch):
    monkeypatch.setenv("ANOMALITY_OFFLINE", "1")
    sig._cache.clear()
    r = sig.mood_vs_buy_and_hold()
    assert r["days"] == 1865 and r["block_days"] == 20
    s = r["sharpe_diff"]
    assert s["ci95"][0] <= s["observed"] <= s["ci95"][1]
    assert s["observed"] == pytest.approx(1.3721 - 1.2195, abs=1e-3)  # matches results/decision.json
    assert "interval" in r["text"]


# ---------- robustness: costs and forecast quality ----------
@pytest.fixture(scope="module")
def robust():
    from backend import robustness
    mp = pytest.MonkeyPatch()
    mp.setenv("ANOMALITY_OFFLINE", "1")
    robustness._cache.clear()
    try:
        yield robustness.report()
    finally:
        mp.undo()


def test_cost_table_at_5_bps_is_the_published_backtest(robust):
    import json
    from regime.config import ROOT
    published = json.loads((ROOT / "results" / "decision.json").read_text())["systems"]
    i5 = robust["costs"]["bps"].index(5)
    for name, rows in robust["costs"]["systems"].items():
        for k in ("ann_return", "sharpe", "max_drawdown"):
            assert rows[i5][k] == pytest.approx(published[name]["all"][k], abs=2e-3), (name, k)


def test_higher_costs_never_help(robust):
    for rows in robust["costs"]["systems"].values():
        growth = [r["net_return"] for r in rows]
        assert growth == sorted(growth, reverse=True)


def test_growth_breakeven_brackets_the_crossing(robust):
    from backend import robustness
    from backend.backtest import portfolio_returns
    from backend.decision_service import frame, held_by_system
    be = robust["growth_breakeven_bps"]
    assert be is not None and 5 < be < 20  # out-grows at the 5 bps assumption, not at 20 (docs)
    mp = pytest.MonkeyPatch()
    mp.setenv("ANOMALITY_OFFLINE", "1")
    held = held_by_system(frame())
    mp.undo()
    g = lambda n, c: robustness._growth(portfolio_returns(held[n], cost_bps=c))
    assert g("mood_rule", be - 0.5) > g("buy_and_hold", be - 0.5)
    assert g("mood_rule", be + 0.5) < g("buy_and_hold", be + 0.5)


def test_fast_forecast_metrics_match_regime_metrics():
    from backend import robustness
    from regime.metrics import evaluate
    fc = robustness.forecast_frame()
    w = fc[(fc["date"] >= "2020-02-19") & (fc["date"] <= "2020-04-30")]
    for m in ("ours", "simple"):
        ref = evaluate(w.set_index(["date", "symbol"]), m)
        assert {k: robustness.forecast_metrics(w, m)[k] for k in ref} == ref


def test_forecast_models_are_scored_on_the_same_rows(robust):
    f = robust["forecast"]["periods"]["all"]
    assert len({v["n"] for v in f.values()}) == 1


# ---------- causality with missing days, warm-up, conventions (synthetic, exact) ----------
def _panel(days, missing=()):
    """Two assets; asset B has no row on the `missing` day indices (e.g. a market holiday)."""
    idx = pd.bdate_range("2021-01-04", periods=days)
    rows = [{"symbol": s, "date": d, "day": i} for s in ("A", "B") for i, d in enumerate(idx)
            if not (s == "B" and i in missing)]
    return pd.DataFrame(rows).sort_values(["symbol", "date"], ignore_index=True)


def test_held_exposure_always_comes_from_at_least_two_days_earlier_even_with_gaps():
    """Encode each decision as its own day number: whatever is held on day t must have been
    decided on day t-2 or earlier, including right after a missing day."""
    from backend.backtest import hold
    per = _panel(12, missing=(4, 5))
    per["exposure"] = per["day"].astype(float) + 100  # decision id = 100 + day it was made
    h = hold(per)
    decided = h["held"] - 100
    live = h["held"] > 0  # before the first decision is executed, nothing is held
    assert (decided[live] <= h.loc[live, "day"] - 2).all()
    b = h[h["symbol"] == "B"].set_index("day")
    assert b.loc[6, "held"] - 100 == 2  # after the gap, B holds an older decision: delayed, never early


def test_traded_amount_is_the_change_in_held_exposure():
    from backend.backtest import hold
    per = _panel(6)
    per["exposure"] = [1, 1, 0.5, 0.5, 1, 1] * 2
    h = hold(per)
    a = h[h["symbol"] == "A"]
    assert a["held"].tolist() == [0, 0, 1, 1, 0.5, 0.5]
    assert a["traded"].tolist() == [0, 0, 1, 0, 0.5, 0]  # entry, then one cut, each charged once


def test_surprise_rule_hold_counter_counts_observed_days():
    """The minimum hold counts trading days present in the data: a missing day neither
    advances nor resets it."""
    from backend.decision import DecisionConfig, positions
    cfg = DecisionConfig(min_hold_days=3, variant="tiered")
    rv = np.array([0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.2, 0.2, 0.2, 0.2, 0.2])  # stress on days 4-5
    expo, reasons = positions(np.full(len(rv), 0.2), rv, cfg)
    assert expo.tolist() == [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]  # cash for 3 observed days, then back
    assert reasons[6] == "hold (1d < 3)" and reasons[8] == "calm/normal"
    # Start-up quirk: the starting position counts as a fresh switch, so stress in an asset's
    # first `min_hold_days` observations is ignored (affects only the first days of each asset).
    early, why = positions(np.full(4, 0.2), np.array([0.2, 0.5, 0.5, 0.2]), cfg)
    assert early.tolist() == [1, 1, 1, 1] and why[1] == "hold (1d < 3)"


def test_mood_rule_stays_invested_while_warming_up():
    from backend.decision import MoodConfig, mood_positions, storminess
    cfg = MoodConfig(min_history=60)
    pct = storminess(np.linspace(0.1, 0.5, 40), cfg.min_history)  # fewer days than needed
    expo, reasons = mood_positions(pct, cfg)
    assert np.isnan(pct).all() and (expo == 1).all() and set(reasons) == {"warming up"}


def test_rebalance_trades_by_hand():
    from backend.robustness import rebalance_trades
    # two assets at 50/50; +10% and -10% -> 55/45 -> trade 5 points each way back to 50/50
    assert rebalance_trades(np.array([[0.10, -0.10]])) == pytest.approx([0.10])
    assert rebalance_trades(np.array([[0.05, 0.05]])) == pytest.approx([0.0])  # moved together


def test_drift_basket_by_hand():
    from backend.robustness import drift_basket
    out = drift_basket(np.array([[1.0, 0.0], [0.0, 0.3]]))
    # day 1: 0.5*100% = +50%; weights become 2/3, 1/3; day 2: 1/3 * 30% = +10%
    assert out == pytest.approx([0.5, 0.1])


def test_conventions_on_the_real_backtest(robust):
    import json
    from regime.config import ROOT
    cv = robust["conventions"]
    published = json.loads((ROOT / "results" / "decision.json").read_text())["systems"]
    for n in ("mood_rule", "buy_and_hold"):
        assert cv["A"][n]["sharpe"] == pytest.approx(published[n]["all"]["sharpe"], abs=2e-3)
    assert cv["assets"] == {"A": 56, "fixed": 54}
    for key in "ABCD":  # the ranking on reward / risk holds under every convention
        assert cv[key]["mood_rule"]["sharpe"] > cv[key]["buy_and_hold"]["sharpe"], key
    # a true buy-and-hold basket drifts into crypto: far more volatile than the daily rebalanced one
    assert cv["C"]["buy_and_hold"]["ann_vol"] > 1.5 * cv["B"]["buy_and_hold"]["ann_vol"]
