"""anomality API server. Run: uvicorn backend.app:app --reload"""
from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi import Path as PathParam
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .analysis import FEATURES, DecisionModel, Settings, detect, features
from .data import DataError, load_prices
from .data import status as data_status
from . import alerts, analyst, decision_service, robustness, significance, time_machine, voice
from .regime_service import RegimeUnavailable, plain_asset, plain_market

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


ANALYSIS_TTL = 600  # seconds; same freshness as the price cache
_analysis_cache: dict = {}
_analysis_lock = threading.Lock()


def analyze(symbol: str, period: str, s: Settings, full: bool = True) -> dict:
    """Cached wrapper: reloading a page or rescanning doesn't refit every model."""
    key = (symbol.strip().upper(), period, tuple(sorted(s.__dict__.items())), full)
    now = time.time()
    with _analysis_lock:
        hit = _analysis_cache.get(key)
        if hit and now - hit[0] < ANALYSIS_TTL:
            return hit[1]
    result = _analyze(symbol, period, s, full)
    with _analysis_lock:
        if len(_analysis_cache) > 500:  # bound memory: drop expired entries
            for k in [k for k, (t, _) in _analysis_cache.items() if now - t >= ANALYSIS_TTL]:
                del _analysis_cache[k]
        _analysis_cache[key] = (now, result)
    return result


def _analyze(symbol: str, period: str, s: Settings, full: bool = True) -> dict:
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
    return {**WATCHLIST, "features": FEATURES, "data": data_status()}


@app.get("/api/status")
def api_status():
    """Where prices are coming from: Yahoo, or saved copies (offline mode or Yahoo paused)."""
    return data_status()


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

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, tickers))
    return _clean({"results": results})


@app.get("/api/regime/market")
def api_regime_market():
    try:
        return _clean(plain_market())
    except RegimeUnavailable as e:
        raise HTTPException(503, str(e))


@app.get("/api/regime/{symbol}")
def api_regime(symbol: str):
    try:
        return _clean(plain_asset(symbol))
    except RegimeUnavailable as e:
        raise HTTPException(404, str(e))


@app.get("/api/decision/portfolio")
def api_decision_portfolio():
    try:
        return _clean(decision_service.plain_portfolio())
    except decision_service.DecisionUnavailable as e:
        raise HTTPException(503, str(e))


@app.get("/api/decision/significance")  # before /api/decision/{symbol}, which would capture it
def api_decision_significance():
    """Mood rule vs. buy-and-hold: paired block-bootstrap intervals for the Sharpe and return gaps."""
    try:
        return _clean(significance.mood_vs_buy_and_hold())
    except (decision_service.DecisionUnavailable, ValueError) as e:
        raise HTTPException(503, str(e))


@app.get("/api/decision/robustness")  # before /api/decision/{symbol}, which would capture it
def api_decision_robustness():
    """Backtest re-priced at other trading costs, and forecast quality (IC, RMSE, R², QLIKE)."""
    try:
        return _clean(robustness.report())
    except decision_service.DecisionUnavailable as e:
        raise HTTPException(503, str(e))


@app.get("/api/decision/{symbol}")
def api_decision(symbol: str):
    try:
        return _clean(decision_service.plain_asset(symbol))
    except decision_service.DecisionUnavailable as e:
        raise HTTPException(404, str(e))


@app.get("/api/timemachine")
def api_time_machine_meta():
    """Assets and date range the Historical Time Machine covers."""
    try:
        return _clean(time_machine.meta(time_machine.load()))
    except (time_machine.TimeMachineError, decision_service.DecisionUnavailable) as e:
        raise HTTPException(503, str(e))


@app.get("/api/timemachine/{symbol}")
def api_time_machine(symbol: str, date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$")):
    """What the system knew at the close of `date` (forecast, inputs, reconstructed decision), and,
    kept separate, what happened over the forecast horizon after it. Missing data -> 200 with a status."""
    try:
        return _clean(time_machine.cached_view(symbol, date))
    except decision_service.DecisionUnavailable as e:
        raise HTTPException(503, str(e))
    except time_machine.UnknownAsset as e:
        raise HTTPException(404, str(e))
    except time_machine.BadDate as e:
        raise HTTPException(422, str(e))
    except time_machine.TimeMachineError as e:
        raise HTTPException(503, str(e))


@app.get("/api/alerts")
def api_alerts(date: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")):
    """Alerts and every monitor's status on the newest forecast day, or on a past `date` (using only
    data available that day). Deterministic; thresholds in the response's `config`."""
    try:
        return _clean(alerts.latest(date))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except (time_machine.TimeMachineError, decision_service.DecisionUnavailable) as e:
        raise HTTPException(503, str(e))


@app.post("/api/alerts/{alert_id}/ack")
def api_alert_ack(alert_id: str = PathParam(..., pattern=r"^[a-z_]+:[A-Za-z0-9_.\-]+:\d{4}-\d{2}-\d{2}$")):
    """Mark an alert as seen (kept in memory until the server restarts)."""
    return alerts.acknowledge(alert_id)


class AnalystRequest(BaseModel):
    kind: str = Field("summary", pattern="^(summary|alert|risks|limits)$")
    alerts_date: Optional[str] = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    alert_id: Optional[str] = Field(None, max_length=120, pattern=r"^[a-z_]+:[A-Za-z0-9_.\-]+:\d{4}-\d{2}-\d{2}$")


@app.get("/api/analyst/status")
def api_analyst_status():
    """Whether Gemini is configured (never returns the key)."""
    return analyst.configured()


@app.post("/api/analyst")
def api_analyst(req: AnalystRequest):
    """Gemini explains facts the backend computed; falls back to those facts if it can't."""
    if req.kind == "alert" and not req.alert_id:
        raise HTTPException(422, "Choose an alert to explain.")
    try:
        return _clean(analyst.explain(req.kind, req.alert_id, alerts_date=req.alerts_date))
    except analyst.UnknownAlert as e:
        raise HTTPException(404, str(e))
    except analyst.RateLimited as e:
        raise HTTPException(429, {"status": "rate_limited", "message": str(e)},
                            headers={"Retry-After": str(int(e.retry_after) + 1)})


class VoiceRequest(BaseModel):
    report_id: str = Field(..., pattern="^[0-9a-f]{16}$")


@app.get("/api/voice/status")
def api_voice_status():
    """Whether voice reports are configured (never returns the key)."""
    return voice.configured()


@app.post("/api/voice")
def api_voice(req: VoiceRequest):
    """Read aloud a report the analyst already returned (by id, never free text). audio/mpeg."""
    try:
        audio = voice.for_report(req.report_id)
    except voice.VoiceError as e:
        headers = {"Retry-After": str(int(e.retry_after) + 1)} if e.retry_after is not None else None
        raise HTTPException(e.http, {"status": e.status, "message": str(e)}, headers=headers)
    return Response(content=audio, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
