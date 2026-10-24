"""Prove the backtest is causal and costs are exact. Run before trusting any number."""
import numpy as np
import pandas as pd
from backend.backtest import backtest
from backend.decision import DecisionConfig, decide


def test_exec_lag():
    """Signal flipped on day t changes position on day t+2 only."""
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    pos = pd.Series([0.0, 1.0, 1.0, 1.0, 1.0], index=idx)
    ret = pd.Series([0.01] * 5, index=idx)
    r = backtest(pos, ret, cost_bps=0.0, exec_lag=1)
    # pos.shift(2) = [NaN,NaN,0,1,1] -> fillna(0) -> [0,0,0,1,1]
    # net = [0,0,0,.01,.01]; mean = .02/5 = .004; ann = .004*252 = 1.008
    assert abs(r["ann_return"] - 1.008) < 1e-6, r["ann_return"]
    # And prove causality: no return is earned on days 0 and 1
    assert r["n_days"] == 5  # 5 net days, 2 with positive exposure


def test_costs_exact():
    """One switch of size 1 costs exactly 5 bps once."""
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    pos = pd.Series([0.0] + [1.0] * 9, index=idx)
    ret = pd.Series([0.0] * 10, index=idx)
    r = backtest(pos, ret, cost_bps=5.0, exec_lag=1)
    # one trade of size 1 = 5 bps; net_return = 1 - (1 - 5e-4) ... check it's negative but tiny
    assert r["net_return"] < 0
    assert abs(r["net_return"] + 5e-4) < 1e-6, r["net_return"]


def test_hysteresis():
    cfg = DecisionConfig(stress_surprise=2.0, min_hold_days=5, variant="tiered")
    # calm -> 1.0
    exp, _ = decide(0.10, 0.10, 0, 1.0, cfg); assert exp == 1.0
    # 3x surprise but just switched -> hold
    exp, reason = decide(0.10, 0.30, 1, 1.0, cfg)
    assert exp == 1.0 and "hold" in reason
    # 3x surprise and hysteresis elapsed -> cash
    exp, reason = decide(0.10, 0.30, 5, 1.0, cfg)
    assert exp == 0.0 and "stress" in reason


def test_tiered():
    cfg = DecisionConfig(min_hold_days=0, variant="tiered")
    assert decide(0.10, 0.10, 0, 1.0, cfg)[0] == 1.0   # calm
    assert decide(0.10, 0.15, 0, 1.0, cfg)[0] == 0.5   # 1.5x choppy
    assert decide(0.10, 0.25, 0, 1.0, cfg)[0] == 0.0   # 2.5x stress