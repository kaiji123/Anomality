"""Market data loading.

Pulls daily OHLCV candles from Yahoo Finance's public chart endpoint. Works for
stocks (AAPL), ETFs (SPY) and crypto pairs (BTC-USD). Results are cached in
memory for a few minutes. If the network is unavailable, a synthetic series is
generated so the app still runs (the response is flagged as synthetic).
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) anomality/1.0"}
CACHE_TTL = 600  # seconds

_cache: Dict[Tuple[str, str], Tuple[float, pd.DataFrame, dict]] = {}
_lock = threading.Lock()


class DataError(Exception):
    pass


def _fetch_yahoo(symbol: str, period: str) -> Tuple[pd.DataFrame, dict]:
    resp = requests.get(
        YAHOO_URL.format(symbol=symbol),
        params={"range": period, "interval": "1d", "includePrePost": "false"},
        headers=HEADERS,
        timeout=15,
    )
    if resp.status_code == 404:
        raise DataError(f"Symbol '{symbol}' not found")
    resp.raise_for_status()
    payload = resp.json().get("chart", {})
    if payload.get("error"):
        raise DataError(payload["error"].get("description", "Unknown symbol"))
    result = (payload.get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        raise DataError(f"No price data for '{symbol}'")

    meta = result.get("meta", {})
    tz = meta.get("exchangeTimezoneName") or "UTC"
    quote = result["indicators"]["quote"][0]
    # Label each candle with its trading date in the exchange's own time zone.
    # (Yahoo stamps e.g. a Sydney session at 23:00 UTC the previous day.)
    dates = (
        pd.to_datetime(result["timestamp"], unit="s", utc=True)
        .tz_convert(tz).tz_localize(None).normalize()
    )
    df = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        },
        index=dates,
    )
    # Prices are split-adjusted by Yahoo; also adjust for dividends using adjclose.
    adj = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    if adj is not None:
        factor = pd.Series(adj, index=dates, dtype=float) / df["close"].astype(float)
        factor = factor.where(factor > 0).fillna(1.0)
        for col in ("open", "high", "low", "close"):
            df[col] = df[col] * factor
    df = df.dropna(subset=["close"])
    # The live price for the current session can arrive as an extra row; keep the latest.
    df = df[~df.index.duplicated(keep="last")]
    df["volume"] = df["volume"].fillna(0)
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])

    # Drop the current session if it hasn't closed yet (partial volume/range would
    # look artificially quiet). Crypto days close at 00:00 UTC.
    dropped_partial = False
    regular = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    if regular.get("start") and regular.get("end") and time.time() < regular["end"] and len(df):
        session_day = pd.Timestamp(regular["start"], unit="s", tz="UTC").tz_convert(tz).tz_localize(None).normalize()
        if df.index[-1] == session_day:
            df = df.iloc[:-1]
            dropped_partial = True

    info = {
        "symbol": meta.get("symbol", symbol),
        "name": meta.get("longName") or meta.get("shortName") or symbol,
        "currency": meta.get("currency", "USD"),
        "exchange": meta.get("fullExchangeName") or meta.get("exchangeName", ""),
        "type": meta.get("instrumentType", ""),
        "timezone": tz,
        "session_in_progress": dropped_partial,
        "synthetic": False,
    }
    return df, info


def _synthetic(symbol: str, period: str) -> Tuple[pd.DataFrame, dict]:
    """Geometric Brownian motion with volatility regimes and jump events."""
    days = {"1y": 365, "2y": 730, "5y": 1825, "10y": 3650}.get(period, 730)
    seed = abs(hash(symbol)) % (2**32)
    rng = np.random.default_rng(seed)
    is_crypto = symbol.upper().endswith("-USD")
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days, freq="D")
    if not is_crypto:
        idx = idx[idx.dayofweek < 5]
    n = len(idx)
    base_vol = 0.035 if is_crypto else 0.015
    regime = np.exp(np.cumsum(rng.normal(0, 0.05, n)) * 0.3)
    regime = np.clip(regime / regime.mean(), 0.5, 2.5)
    rets = rng.standard_t(4, n) * base_vol * regime / np.sqrt(2)
    jumps = rng.random(n) < 0.01
    rets[jumps] += rng.normal(0, base_vol * 6, jumps.sum())
    close = 100 * np.exp(np.cumsum(rets))
    spread = np.abs(rng.normal(0, base_vol * regime, n))
    vol_base = 1e7 * (1 + 4 * np.abs(rets) / base_vol)
    df = pd.DataFrame(
        {
            "open": close * np.exp(rng.normal(0, base_vol / 3, n)),
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": vol_base * rng.lognormal(0, 0.3, n),
        },
        index=idx,
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    info = {
        "symbol": symbol,
        "name": f"{symbol} (synthetic demo data)",
        "currency": "USD",
        "exchange": "Offline",
        "type": "CRYPTOCURRENCY" if is_crypto else "EQUITY",
        "synthetic": True,
    }
    return df, info


def load_prices(symbol: str, period: str = "5y", allow_synthetic: bool = True) -> Tuple[pd.DataFrame, dict]:
    symbol = symbol.strip().upper()
    key = (symbol, period)
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_TTL:
            return hit[1].copy(), dict(hit[2])
    try:
        df, info = _fetch_yahoo(symbol, period)
    except DataError:
        raise
    except Exception:
        if not allow_synthetic:
            raise
        df, info = _synthetic(symbol, period)
    if len(df) < 150:
        raise DataError(f"Not enough history for '{symbol}' ({len(df)} days). Try a longer range.")
    with _lock:
        _cache[key] = (now, df, info)
    return df.copy(), dict(info)
