"""Market data loading.

Pulls daily OHLCV candles from Yahoo Finance's public chart endpoint. Works for
stocks (AAPL), ETFs (SPY) and crypto pairs (BTC-USD).

Where prices come from, in order:
1. Memory (10 minutes).
2. The local disk cache (data/cache/, not in git) if it was saved in the last 10 minutes,
   so a server restart doesn't re-download everything.
3. Yahoo Finance. Every good download is saved to the disk cache.
4. If Yahoo fails (rate limit, no network, bad response): the newest saved copy, from the
   disk cache or the snapshot that ships with the repo (data/snapshot/), flagged `cached`.
   After a failure Yahoo is skipped for a few minutes, so a rate limit doesn't make every
   request wait for an error.
5. Only if nothing is saved: a synthetic series, flagged `synthetic` (or an error when
   synthetic data isn't allowed).

Set ANOMALITY_OFFLINE=1 to skip Yahoo entirely and use saved prices (e.g. for a demo on
unreliable Wi-Fi). Refresh the snapshot with `python -m backend.snapshot`.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) anomality/1.0"}
CACHE_TTL = 600  # seconds: memory, and how long a disk copy counts as fresh
FALLBACK_TTL = 60  # seconds a saved-copy answer is kept in memory before Yahoo is tried again
COOLDOWN = 180  # seconds Yahoo is skipped after it fails
ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"        # local only (gitignored)
SNAPSHOT_DIR = ROOT / "data" / "snapshot"  # committed: lets a fresh clone run without Yahoo
PERIOD_DAYS = {"1y": 365, "2y": 730, "5y": 1826, "10y": 3653}

_cache: Dict[Tuple[str, str], Tuple[float, pd.DataFrame, dict]] = {}
_lock = threading.Lock()
_status = {"cooldown_until": 0.0, "last_error": None}


def offline() -> bool:
    return os.environ.get("ANOMALITY_OFFLINE", "").strip().lower() in ("1", "true", "yes")


def status() -> dict:
    """Where prices are coming from right now (shown in the app)."""
    wait = max(0.0, _status["cooldown_until"] - time.time())
    return {"offline": offline(), "yahoo_paused_for": round(wait), "last_error": _status["last_error"] if wait else None}


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


# ---------------------------------------------------------------- saved prices (disk cache + snapshot)
def _safe(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", symbol)


def _write(df: pd.DataFrame, info: dict, csv_path: Path, extra: dict) -> None:
    """Save a price frame and its info next to each other, atomically."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    df.to_csv(tmp, index_label="date", float_format="%.8g", compression="gzip")
    os.replace(tmp, csv_path)
    meta = csv_path.with_name(csv_path.name.replace(".csv.gz", ".json"))
    tmp = meta.with_name(meta.name + ".tmp")
    tmp.write_text(json.dumps({"info": info, **extra}))
    os.replace(tmp, meta)


def _read(csv_path: Path) -> Optional[Tuple[pd.DataFrame, dict]]:
    meta = csv_path.with_name(csv_path.name.replace(".csv.gz", ".json"))
    if not (csv_path.exists() and meta.exists()):
        return None
    try:
        df = pd.read_csv(csv_path, index_col="date", parse_dates=["date"], compression="gzip")
        return df, json.loads(meta.read_text())
    except (OSError, ValueError, KeyError):
        return None  # a damaged file is skipped, never fatal


def save_to_cache(symbol: str, period: str, df: pd.DataFrame, info: dict) -> None:
    try:
        _write(df, info, CACHE_DIR / f"{_safe(symbol)}_{period}.csv.gz", {"saved_at": time.time()})
    except OSError:
        pass  # a read-only disk shouldn't break the app


def _slice(df: pd.DataFrame, period: str) -> pd.DataFrame:
    days = PERIOD_DAYS.get(period)
    return df if days is None or df.empty else df[df.index > df.index[-1] - pd.Timedelta(days=days)]


def _candidates(symbol: str, period: str) -> List[Tuple[float, pd.DataFrame, dict, str]]:
    """Every saved copy that covers `period`: (saved_at, frame, info, source)."""
    want = PERIOD_DAYS.get(period, 0)
    out = []
    for p, days in PERIOD_DAYS.items():
        if days < want:
            continue  # a 1y copy can't stand in for 5y
        hit = _read(CACHE_DIR / f"{_safe(symbol)}_{p}.csv.gz")
        if hit:
            out.append((hit[1]["saved_at"], _slice(hit[0], period), hit[1]["info"], "cache"))
    snap = _read(SNAPSHOT_DIR / f"{_safe(symbol)}.csv.gz")
    if snap:
        out.append((snap[1]["saved_at"], _slice(snap[0], period), snap[1]["info"], "snapshot"))
    return out


def saved_prices(symbol: str, period: str) -> Optional[Tuple[pd.DataFrame, dict]]:
    """The newest saved copy of `symbol` covering `period`, flagged as cached."""
    found = _candidates(symbol, period)
    if not found:
        return None
    saved_at, df, info, source = max(found, key=lambda c: c[0])
    info = {**info, "cached": True, "source": source, "synthetic": False,
            "as_of": time.strftime("%Y-%m-%d %H:%M", time.localtime(saved_at))}
    return df, info


def _fresh_from_cache(symbol: str, period: str) -> Optional[Tuple[pd.DataFrame, dict]]:
    hit = _read(CACHE_DIR / f"{_safe(symbol)}_{period}.csv.gz")
    if hit and time.time() - hit[1]["saved_at"] < CACHE_TTL:
        return hit[0], hit[1]["info"]
    return None


# ---------------------------------------------------------------- the loader
def load_prices(symbol: str, period: str = "5y", allow_synthetic: bool = True) -> Tuple[pd.DataFrame, dict]:
    symbol = symbol.strip().upper()
    key = (symbol, period)
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and now < hit[0]:
            return hit[1].copy(), dict(hit[2])
    ttl = CACHE_TTL
    fresh = _fresh_from_cache(symbol, period)
    if fresh:
        df, info = fresh
    elif offline() or now < _status["cooldown_until"]:
        df, info, ttl = _fallback(symbol, period, allow_synthetic, None)
    else:
        try:
            df, info = _fetch_yahoo(symbol, period)
            save_to_cache(symbol, period, df, info)
        except DataError:
            raise
        except (requests.RequestException, ValueError) as e:
            # Network down, timeout, HTTP error (incl. rate limits) or an unreadable response.
            # Any other exception is a bug and should surface, not hide behind saved data.
            _status["cooldown_until"] = time.time() + COOLDOWN
            _status["last_error"] = f"{type(e).__name__}: {e}"[:200]
            df, info, ttl = _fallback(symbol, period, allow_synthetic, e)
    if len(df) < 150:
        raise DataError(f"Not enough history for '{symbol}' ({len(df)} days). Try a longer range.")
    with _lock:
        _cache[key] = (now + ttl, df, info)
    return df.copy(), dict(info)


def _fallback(symbol: str, period: str, allow_synthetic: bool,
              error: Optional[Exception]) -> Tuple[pd.DataFrame, dict, int]:
    saved = saved_prices(symbol, period)
    if saved:
        return saved[0], saved[1], FALLBACK_TTL
    if allow_synthetic:
        df, info = _synthetic(symbol, period)
        return df, info, FALLBACK_TTL
    if error is not None:
        raise error
    raise DataError(f"Yahoo Finance is unavailable and no saved prices exist for '{symbol}'.")
