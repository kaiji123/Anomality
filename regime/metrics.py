"""Volatility-forecast metrics used in the paper: IC, RMSE, R², QLIKE."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def daily_ic(df: pd.DataFrame, pred: str = "pred", actual: str = "y", min_assets: int = 5) -> pd.Series:
    """Spearman rank correlation between forecast and actual across assets, for each date."""
    def one(g: pd.DataFrame) -> float:
        if len(g) < min_assets:
            return np.nan
        return g[pred].rank().corr(g[actual].rank())
    return df.groupby(level="date").apply(one).dropna()


def qlike(y: np.ndarray, yhat: np.ndarray) -> float:
    ratio = np.maximum(y, 1e-4) ** 2 / np.maximum(yhat, 1e-3) ** 2
    return float(np.mean(ratio - np.log(ratio) - 1))


def evaluate(df: pd.DataFrame, pred: str = "pred", actual: str = "y", mask: Optional[pd.Series] = None) -> Dict[str, float]:
    if mask is not None:
        df = df[mask]
    y, yhat = df[actual].to_numpy(), df[pred].to_numpy()
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    ic = daily_ic(df, pred, actual)
    return {
        "ic": round(float(ic.mean()), 4) if len(ic) else None,
        "rmse": round(float(np.sqrt(sse / len(y))), 4),
        "r2": round(1 - sse / sst, 4) if sst > 0 else None,
        "qlike": round(qlike(y, yhat), 4),
        "n": int(len(y)),
    }
