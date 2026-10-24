"""Chronological backtest with transaction costs.

Net return: pos_t * r_t - c * |pos_t - pos_{t-1}|
pos is the signal shifted by 1 + exec_lag (signal at close t earns r_{t+2}).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def backtest(positions: pd.Series, returns: pd.Series,
             cost_bps: float = 5.0, exec_lag: int = 1) -> dict:
    pos = positions.shift(1 + exec_lag).fillna(0.0)
    ret = pos * returns
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (cost_bps / 1e4)
    net = (ret - cost).dropna()
    if net.empty:
        return dict(net_return=0., ann_return=0., ann_vol=0., sharpe=0.,
                    max_drawdown=0., turnover=0., n_days=0)
    equity = (1 + net).cumprod()
    peak = equity.cummax()
    dd = float((equity / peak - 1).min())
    ann_ret = float(net.mean() * 252)
    ann_vol = float(net.std() * np.sqrt(252))
    return dict(
        net_return=float(equity.iloc[-1] - 1),
        ann_return=ann_ret,
        ann_vol=ann_vol,
        sharpe=ann_ret / ann_vol if ann_vol > 0 else 0.0,
        max_drawdown=dd,
        turnover=float(turnover.mean() * 252),
        n_days=int(len(net)),
    )