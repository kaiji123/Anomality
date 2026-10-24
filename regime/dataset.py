"""Builds the training panel: one row per (date, asset) with features x, regime variables z and target y.

All features use data up to and including day t. The target uses days t+1 … t+horizon.
"""
from __future__ import annotations

import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backend.data import load_prices

from .config import CACHE_DIR, CRYPTO, STOCKS, X_COLS, Z_COLS, TrainConfig

ANNUAL = np.sqrt(252)


def load_closes(symbols: List[str], period: str, use_cache: bool = True) -> pd.DataFrame:
    """Daily closes (split- and dividend-adjusted) for each symbol, one column per symbol."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"closes_{period}.pkl"
    if use_cache and cache.exists() and time.time() - cache.stat().st_mtime < 12 * 3600:
        closes = pickle.loads(cache.read_bytes())
        if set(symbols) <= set(closes.columns):
            return closes[symbols]

    def one(sym: str) -> Tuple[str, pd.Series]:
        try:
            df, _ = load_prices(sym, period, allow_synthetic=False)
            return sym, df["close"]
        except Exception as e:  # skip symbols that fail to download
            print(f"  skipping {sym}: {e}")
            return sym, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        series: Dict[str, pd.Series] = {s: c for s, c in pool.map(one, symbols) if c is not None}
    closes = pd.DataFrame(series)
    cache.write_bytes(pickle.dumps(closes))
    return closes


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def build_panel(cfg: TrainConfig, use_cache: bool = True) -> pd.DataFrame:
    stocks = list(STOCKS)
    crypto = list(CRYPTO) if cfg.include_crypto else []
    closes = load_closes(stocks + crypto, cfg.period, use_cache)
    stocks = [s for s in stocks if s in closes.columns]
    crypto = [s for s in crypto if s in closes.columns]

    # Use the U.S. trading calendar. Crypto trades every day, so sampling it on these dates
    # means a Monday crypto return includes the weekend.
    calendar = closes[stocks].dropna(how="all").index
    closes = closes.reindex(calendar)
    if crypto:
        closes[crypto] = closes[crypto].ffill(limit=3)

    r = np.log(closes).diff()
    r_m = r[stocks].mean(axis=1, skipna=True)  # equal-weighted market return (stocks only)

    feats: Dict[str, pd.DataFrame] = {
        "rv5": r.rolling(5).std() * ANNUAL,
        "rv20": r.rolling(20).std() * ANNUAL,
        "rv60": r.rolling(60).std() * ANNUAL,
        "cum5": r.rolling(5).sum(),
        "cum20": r.rolling(20).sum(),
        "rsi14": _rsi(closes) / 100.0,
    }
    for k in range(10):
        feats[f"lag{k}"] = r.shift(k)

    # Regime variables.
    mkt_vol = r_m.rolling(20).std() * ANNUAL
    feats["mkt_vol"] = pd.DataFrame({c: mkt_vol for c in closes.columns})
    cov = r.rolling(120).cov(r_m)
    var = r_m.rolling(120).var()
    beta = cov.div(var, axis=0).shift(1)  # beta from the preceding 120 days
    idio = r - beta.mul(r_m, axis=0)
    feats["idio_vol"] = idio.rolling(20).std() * ANNUAL

    # Target: annualized realized volatility over the next `horizon` days.
    h = cfg.horizon
    feats["y"] = r.rolling(h).std().shift(-h) * ANNUAL

    long = pd.concat({name: df.stack(future_stack=True) for name, df in feats.items()}, axis=1)
    long.index.names = ["date", "symbol"]
    long = long.replace([np.inf, -np.inf], np.nan)
    long = long.dropna(subset=X_COLS + Z_COLS)
    long["is_crypto"] = long.index.get_level_values("symbol").isin(crypto)
    return long.sort_index()
