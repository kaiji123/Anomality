"""Chronological backtest with transaction costs.

Net return: pos_t * r_t - c * |pos_t - pos_{t-1}|
pos is the signal shifted by 1 + exec_lag (signal at close t earns r_{t+2}).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .decision import COST_BPS, EXEC_LAG


def stats(net: pd.Series, turnover: pd.Series) -> dict:
    """Summary of a daily net-return series (and the daily turnover behind it)."""
    net = net.dropna()
    if net.empty:
        return dict(net_return=0., ann_return=0., ann_vol=0., sharpe=0.,
                    max_drawdown=0., turnover=0., n_days=0)
    equity = (1 + net).cumprod()
    ann_ret = float(net.mean() * 252)
    ann_vol = float(net.std() * np.sqrt(252))
    return dict(
        net_return=float(equity.iloc[-1] - 1),
        ann_return=ann_ret,
        ann_vol=ann_vol,
        sharpe=ann_ret / ann_vol if ann_vol > 0 else 0.0,
        max_drawdown=float((equity / equity.cummax() - 1).min()),
        turnover=float(turnover.mean() * 252),
        n_days=int(len(net)),
    )


def backtest(positions: pd.Series, returns: pd.Series,
             cost_bps: float = COST_BPS, exec_lag: int = EXEC_LAG) -> dict:
    """One asset (or one already-combined position series)."""
    pos = positions.shift(1 + exec_lag).fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    net = pos * returns - turnover * (cost_bps / 1e4)
    return stats(net, turnover)


def hold(per: pd.DataFrame, exec_lag: int = EXEC_LAG) -> pd.DataFrame:
    """Add each asset's *held* exposure (its decision delayed by 1 + exec_lag days) and the
    amount traded each day. `per` has columns symbol, date, exposure, sorted by symbol, date.

    The delay must happen per asset, before averaging: averaging first would let a decision
    made with day t's data earn day t's return."""
    held = per.groupby("symbol", sort=False)["exposure"].shift(1 + exec_lag).fillna(0.0)
    traded = held.groupby(per["symbol"], sort=False).diff().abs().fillna(held)
    return per.assign(held=held, traded=traded)


def portfolio_returns(per: pd.DataFrame, cost_bps: float = COST_BPS) -> pd.DataFrame:
    """Equal weight across the assets present each day. `per` comes from `hold()` and has a
    `ret` column. Returns daily net return and turnover, indexed by date."""
    net = per["held"] * per["ret"] - per["traded"] * (cost_bps / 1e4)
    g = pd.DataFrame({"net": net, "turnover": per["traded"], "date": per["date"]}).groupby("date")
    return g.mean().sort_index()
