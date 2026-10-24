"""anomality API server. Run: uvicorn backend.app:app --reload"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .analysis import FEATURES, DecisionModel, Settings, detect, features
from .data import DataError, load_prices

STATIC = Path(__file__).resolve().parent.parent / "static"

WATCHLIST = {
    "stocks": ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD", "NFLX", "JPM", "SPY", "QQQ"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD", "LINK-USD"],
}

app = FastAPI(title="anomality", version="1.0")


def _clean(obj):
    """Make results JSON-safe (NaN/inf -> null)."""
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def analyze(symbol: str, period: str, s: Settings, full: bool = True) -> dict:
    df, info = load_prices(symbol, period)
    fx = detect(features(df, s.threshold), s)
    model = DecisionModel(s).run(fx)
    last, prev = fx.iloc[-1], fx.iloc[-2]
    summary = {
        **info,
        "price": round(float(last["close"]), 6),
        "change_pct": round(100 * float(last["close"] / prev["close"] - 1), 2),
        "last_date": fx.index[-1].strftime("%Y-%m-%d"),
        "anomaly_score": float(last["anomaly_score"]),
        "anomaly_today": last["anomaly_type"],
        "anomalies_30d": int(fx["is_anomaly"].iloc[-30:].sum()),
        "risk_score": model["risk_score"],
        "verdict": model["verdict"],
        "auc": model["metrics"]["auc"],
        "spark": [round(float(v), 6) for v in fx["close"].iloc[-60:]],
    }
    if not full:
        return summary

    dates = fx.index.strftime("%Y-%m-%d").tolist()
    anomalies = fx[fx["is_anomaly"]]
    return {
        "summary": summary,
        "series": {
            "date": dates,
            "open": fx["open"].round(6).tolist(),
            "high": fx["high"].round(6).tolist(),
            "low": fx["low"].round(6).tolist(),
            "close": fx["close"].round(6).tolist(),
            "volume": fx["volume"].tolist(),
            "anomaly_score": fx["anomaly_score"].tolist(),
        },
        "anomalies": [
            {
                "date": d.strftime("%Y-%m-%d"),
                "type": r["anomaly_type"],
                "close": round(float(r["close"]), 6),
                "ret_pct": round(100 * float(r["ret"]), 2),
                "ret_z": round(float(r["ret_z"]), 2),
                "vol_z": None if math.isnan(r["vol_z"]) else round(float(r["vol_z"]), 2),
                "score": float(r["anomaly_score"]),
                "iso": bool(r["iso_flag"]),
            }
            for d, r in anomalies.iloc[::-1].iterrows()
        ],
        "model": model,
        "settings": s.__dict__,
    }


def _settings(threshold: float, horizon: int, depth: int, model: str) -> Settings:
    return Settings(threshold=threshold, horizon=horizon, depth=depth, model=model)


@app.get("/api/watchlist")
def watchlist():
    return {**WATCHLIST, "features": FEATURES}


@app.get("/api/analyze/{symbol}")
def api_analyze(
    symbol: str,
    period: str = Query("5y", pattern="^(1y|2y|5y|10y)$"),
    threshold: float = Query(2.5, ge=1.5, le=5),
    horizon: int = Query(5, ge=1, le=20),
    depth: int = Query(4, ge=2, le=8),
    model: str = Query("forest", pattern="^(forest|tree)$"),
):
    try:
        return _clean(analyze(symbol, period, _settings(threshold, horizon, depth, model)))
    except DataError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/scan")
def api_scan(
    symbols: str = Query(..., description="Comma-separated tickers"),
    period: str = Query("5y", pattern="^(1y|2y|5y|10y)$"),
    threshold: float = Query(2.5, ge=1.5, le=5),
    horizon: int = Query(5, ge=1, le=20),
    depth: int = Query(4, ge=2, le=8),
    model: str = Query("forest", pattern="^(forest|tree)$"),
):
    s = _settings(threshold, horizon, depth, model)
    tickers: List[str] = [t.strip().upper() for t in symbols.split(",") if t.strip()][:40]

    def one(sym: str) -> dict:
        try:
            return analyze(sym, period, s, full=False)
        except Exception as e:  # report per-symbol failures without failing the scan
            return {"symbol": sym, "error": str(e)}

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, tickers))
    return _clean({"results": results})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
