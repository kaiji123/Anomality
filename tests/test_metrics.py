"""Volatility-forecast metrics (regime/metrics.py)."""
import numpy as np
import pandas as pd

from regime.metrics import daily_ic, evaluate, qlike


def test_qlike_is_zero_for_a_perfect_forecast():
    y = np.array([0.1, 0.2, 0.5])
    assert qlike(y, y) == 0.0


def test_qlike_punishes_under_forecasting_more_than_over_forecasting():
    y = np.array([0.20])
    assert qlike(y, np.array([0.10])) > qlike(y, np.array([0.40]))


def _panel(pred, actual):
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2024-01-02", "2024-01-03"]), list("ABCDEF")],
                                     names=["date", "symbol"])
    return pd.DataFrame({"pred": pred * 2, "y": actual * 2}, index=idx)


def test_daily_ic_is_one_when_the_ranking_is_perfect_and_minus_one_when_reversed():
    vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert daily_ic(_panel(vals, vals)).mean() == 1.0
    assert daily_ic(_panel(vals, vals[::-1])).mean() == -1.0


def test_evaluate_reports_rmse_in_volatility_units():
    df = _panel([0.2] * 6, [0.3] * 6)
    assert abs(evaluate(df)["rmse"] - 0.1) < 1e-9
