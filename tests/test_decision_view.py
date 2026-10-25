"""Decision-layer dashboard view (backend/decision_view.py). Shipped forecasts and results/decision.json
only, no network."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from backend import decision_view as dv
from backend.backtest import backtest
from backend.decision import target_for
from backend.run_decision import build_positions
from regime.metrics import daily_ic, evaluate


@pytest.fixture(scope="module")
def dec():
    return dv.load_decision()


@pytest.fixture(scope="module")
def rows(dec):
    return dv._backtest_rows(dv._forecasts(), dec)


# ---------- the tape is the backtest's rule, on the backtest's days ----------
def test_tape_reproduces_the_sample_log_in_decision_json(dec, rows):
    sample = dec["decision_log_sample"]
    g = rows[rows["symbol"] == sample["symbol"]].sort_values("date")
    expo, reasons = dv.run_rule(g["pred"].to_numpy(), g["rv20"].to_numpy(), dv.BALANCED)
    assert len(g) == len(sample["rows"])
    assert g["date"].dt.strftime("%Y-%m-%d").tolist() == [r["date"] for r in sample["rows"]]
    assert expo.tolist() == [r["exposure"] for r in sample["rows"]]
    assert reasons == [r["reason"] for r in sample["rows"]]


@pytest.mark.parametrize("mode", list(dv.RISK_MODES))
def test_fast_rule_matches_build_positions(rows, mode):
    g = rows[rows["symbol"] == "NVDA"].copy()
    ref = build_positions(g, dv.RISK_MODES[mode])
    expo, reasons = dv.run_rule(g.sort_values("date")["pred"].to_numpy(), g.sort_values("date")["rv20"].to_numpy(),
                                dv.RISK_MODES[mode])
    assert expo.tolist() == ref["exposure"].tolist() and reasons == ref["reason"].tolist()


def test_nvda_balanced_tape_matches_the_project_notes():
    t = dv.tape("NVDA", "balanced")
    assert t["counts"] == {"days": 1459, "cash": 6, "half": 343, "full": 1110}
    assert t["latest"]["date"] == "2026-10-16"
    assert t["latest"]["exposure"] == 1.0 and t["latest"]["reason"] == "calm/normal"
    assert len(t["rows"]) == 30
    assert [r["date"] for r in t["rows"]] == sorted(r["date"] for r in t["rows"])  # chronological
    assert t["rows"][-1]["date"] == t["latest"]["date"]


# ---------- risk modes ----------
def test_risk_modes_only_move_the_two_thresholds():
    b = dv.RISK_MODES["balanced"]
    assert (b.stress_surprise, b.choppy_surprise, b.min_hold_days, b.variant) == (2.0, 1.3, 5, "tiered")
    for m in dv.RISK_MODES.values():
        assert m.choppy_surprise < m.stress_surprise
        assert (m.min_hold_days, m.variant) == (b.min_hold_days, b.variant)


@pytest.mark.parametrize("surprise,expected", [
    (1.0, {"conservative": 1.0, "balanced": 1.0, "aggressive": 1.0}),
    (1.2, {"conservative": 0.5, "balanced": 1.0, "aggressive": 1.0}),
    (1.4, {"conservative": 0.5, "balanced": 0.5, "aggressive": 1.0}),
    (1.7, {"conservative": 0.0, "balanced": 0.5, "aggressive": 0.5}),
    (2.2, {"conservative": 0.0, "balanced": 0.0, "aggressive": 0.5}),
    (2.5, {"conservative": 0.0, "balanced": 0.0, "aggressive": 0.0}),
])
def test_risk_mode_mapping(surprise, expected):
    assert {m: target_for(surprise, c)[0] for m, c in dv.RISK_MODES.items()} == expected


def test_more_conservative_means_more_time_de_risked():
    t = dv.tape("NVDA", "balanced")["modes"]
    c, b, a = (t[m]["counts"] for m in ("conservative", "balanced", "aggressive"))
    assert c["cash"] >= b["cash"] >= a["cash"]
    assert c["full"] <= b["full"] <= a["full"]


def test_selector_never_changes_the_portfolio_view(dec):
    before = copy.deepcopy(dv.portfolio())
    for m in dv.RISK_MODES:
        dv.tape("NVDA", m)
    assert dv.portfolio() == before
    assert before["config"]["stress_surprise"] == 2.0 and before["config"]["choppy_surprise"] == 1.3
    for s in dv.SYSTEMS:
        assert {k: before["systems"][s][k] for k in dv.METRIC_KEYS} == {k: dec[s][k] for k in dv.METRIC_KEYS}


def test_unknown_mode_and_asset():
    with pytest.raises(ValueError):
        dv.tape("NVDA", "yolo")
    with pytest.raises(dv.DecisionUnavailable):
        dv.tape("GME")


# ---------- cost curve ----------
def test_cost_curve_is_exact_against_the_backtest():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=600)
    pos = pd.Series(rng.choice([0.0, 0.5, 1.0], size=600), index=idx)
    ret = pd.Series(rng.normal(0.0005, 0.01, size=600), index=idx)
    at5 = backtest(pos, ret, cost_bps=5.0)
    curve = dv.cost_curve({"adaptive_binary": at5, "adaptive_tiered": at5, "buy_and_hold": at5}, bps=(0, 20, 50))
    for c, v in zip((0, 20, 50), curve["ann_return"]["adaptive_tiered"]):
        assert v == pytest.approx(backtest(pos, ret, cost_bps=c)["ann_return"], abs=1e-5)


def test_cost_curve_from_decision_json(dec):
    p = dv.portfolio()
    cc = p["cost_curve"]
    i5 = cc["bps"].index(5)
    for s in dv.SYSTEMS:
        assert cc["ann_return"][s][i5] == pytest.approx(dec[s]["ann_return"], abs=1e-5)
    tz = cc["breakeven"]["adaptive_tiered"]["zero_return_bps"]
    m = dec["adaptive_tiered"]
    assert m["ann_return"] + m["turnover"] * (5 - tz) / 1e4 == pytest.approx(0, abs=1e-4)
    # buy-and-hold earns more at every cost >= 0 in this backtest, so no break-even is reported
    assert cc["breakeven"]["adaptive_tiered"]["matches_buy_and_hold_bps"] is None


# ---------- forecast metrics use the regime.metrics definitions ----------
def test_fast_metrics_match_regime_metrics():
    p = dv._forecasts()
    sub = p[(p["date"] >= "2020-02-19") & (p["date"] <= "2020-04-30")]
    ref = daily_ic(sub.set_index(["date", "symbol"]), "pred")
    fast = dv.daily_ic_fast(sub, "pred")
    assert np.allclose(fast.sort_index().to_numpy(), ref.sort_index().to_numpy(), atol=1e-10)
    ours = dv.forecast_metrics(sub, "pred")
    e = evaluate(sub.set_index(["date", "symbol"]), "pred")
    assert {k: ours[k] for k in ("ic", "rmse", "r2", "qlike", "n")} == e


# ---------- missing and malformed inputs ----------
def test_missing_and_malformed_decision_json(tmp_path, dec):
    with pytest.raises(dv.DecisionUnavailable, match="not found"):
        dv.load_decision(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(dv.DecisionUnavailable, match="could not be read"):
        dv.load_decision(bad)

    def broken(fn):
        d = copy.deepcopy(dec)
        fn(d)
        return d

    cases = [
        broken(lambda d: d.pop("buy_and_hold")),
        broken(lambda d: d["adaptive_tiered"].update(sharpe=float("nan"))),
        broken(lambda d: d["adaptive_tiered"].update(max_drawdown=0.3)),
        broken(lambda d: d["adaptive_binary"].pop("turnover")),
        broken(lambda d: d["decision_log_sample"].update(rows=[])),
        broken(lambda d: d["decision_log_sample"]["rows"][0].update(exposure=0.25)),
    ]
    for d in cases:
        f = tmp_path / "case.json"
        f.write_text(json.dumps(d))
        with pytest.raises(dv.DecisionUnavailable):
            dv.load_decision(f)


# ---------- API ----------
def test_decision_api_routes():
    # Handlers are called directly: TestClient's event loop opens a local socketpair, which the
    # offline guard in conftest.py blocks on Windows.
    from fastapi import HTTPException

    from backend import app as app_mod
    paths = {r.path for r in app_mod.app.routes}
    assert {"/api/decision/portfolio", "/api/decision/tape/{symbol}"} <= paths
    t = app_mod.api_decision_tape("nvda", mode="aggressive")
    assert t["mode"] == "aggressive" and t["symbol"] == "NVDA"
    with pytest.raises(HTTPException) as e:
        app_mod.api_decision_tape("GME", mode="balanced")
    assert e.value.status_code == 404
    assert set(app_mod.api_decision_portfolio()["systems"]) == set(dv.SYSTEMS)
    json.dumps(app_mod.api_decision_portfolio(), allow_nan=False)  # JSON-safe, no NaN
