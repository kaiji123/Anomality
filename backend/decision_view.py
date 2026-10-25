"""Read-only view of the decision layer for the dashboard (GET /api/decision/...).

Nothing here retrains or changes a model, and nothing changes the backtest methodology:
  - portfolio numbers come from results/decision.json (written by `python -m backend.run_decision`)
  - the per-asset tape re-runs the same rule (backend.decision.decide) on the same forecasts
    (backend.run_decision.load_predictions) and on the same days the backtest kept
  - the cost curve is exact for the arithmetic annual return: ann_return = mean(daily net) * 252
    and daily net = pos * r - turnover * c, so ann_return(c) = ann_return(5) + turnover * (5 - c) / 1e4
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from regime.metrics import qlike

from .decision import DecisionConfig, decide, target_for

ROOT = Path(__file__).resolve().parent.parent
DECISION_JSON = ROOT / "results" / "decision.json"

SYSTEMS = ("adaptive_binary", "adaptive_tiered", "buy_and_hold")
METRIC_KEYS = ("net_return", "ann_return", "ann_vol", "sharpe", "max_drawdown", "turnover", "n_days")
BACKTEST_COST_BPS = 5.0  # backend.run_decision.COST_BPS, the cost decision.json was computed at
TAPE_DAYS = 30

# Risk tolerance only moves the two surprise thresholds of the existing tiered rule.
# Choppy exposure stays 0.5: DecisionConfig has no parameter for it (target_for hardcodes 0.5).
BALANCED = DecisionConfig(variant="tiered")  # the config every portfolio number was computed with
RISK_MODES: Dict[str, DecisionConfig] = {
    "conservative": replace(BALANCED, stress_surprise=1.6, choppy_surprise=1.1),
    "balanced": BALANCED,
    "aggressive": replace(BALANCED, stress_surprise=2.4, choppy_surprise=1.5),
}

_lock = threading.RLock()
_cache: dict = {}


class DecisionUnavailable(Exception):
    pass


# ---------- inputs ----------
def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_decision(d: dict) -> dict:
    """Check results/decision.json has the schema run_decision writes. Raises DecisionUnavailable."""
    if not isinstance(d, dict):
        raise DecisionUnavailable("decision.json is not a JSON object")
    for s in SYSTEMS:
        m = d.get(s)
        if not isinstance(m, dict):
            raise DecisionUnavailable(f"decision.json has no results for '{s}'")
        bad = [k for k in METRIC_KEYS if not _finite(m.get(k))]
        if bad:
            raise DecisionUnavailable(f"decision.json '{s}' has missing or non-numeric {', '.join(bad)}")
        if m["n_days"] <= 0 or m["turnover"] < 0 or not -1 <= m["max_drawdown"] <= 0 or m["ann_vol"] < 0:
            raise DecisionUnavailable(f"decision.json '{s}' has out-of-range values")
    log = d.get("decision_log_sample")
    if not isinstance(log, dict) or not isinstance(log.get("rows"), list) or not log["rows"]:
        raise DecisionUnavailable("decision.json has no decision_log_sample rows")
    for r in log["rows"]:
        if not isinstance(r, dict) or not isinstance(r.get("date"), str) or r.get("exposure") not in (0, 0.5, 1) \
                or not _finite(r.get("forecast")) or not _finite(r.get("realized")) or not isinstance(r.get("reason"), str):
            raise DecisionUnavailable("decision.json decision_log_sample has a malformed row")
    return d


def load_decision(path: Path = DECISION_JSON) -> dict:
    if not path.exists():
        raise DecisionUnavailable("results/decision.json not found. Run `python -m backend.run_decision` first.")
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise DecisionUnavailable(f"results/decision.json could not be read: {e}")
    return validate_decision(d)


def _forecasts() -> pd.DataFrame:
    """The exact forecast frame the backtest used (floored at 3%, sorted by symbol and date)."""
    with _lock:
        if "preds" not in _cache:
            from .run_decision import load_predictions
            try:
                _cache["preds"] = load_predictions()
            except SystemExit as e:  # load_predictions exits when no forecast file exists
                raise DecisionUnavailable(str(e))
        return _cache["preds"]


def _backtest_rows(preds: pd.DataFrame, dec: dict) -> pd.DataFrame:
    """Rows the backtest kept. Its price join left stocks/ETFs without a return on Mondays and the
    day after holidays, so those rows were dropped; crypto kept every day. The sample log in
    decision.json (a stock) lists exactly the kept stock days, so it gives that calendar."""
    sample = dec["decision_log_sample"]
    sym_crypto = preds.loc[preds["symbol"] == sample.get("symbol"), "is_crypto"]
    if sym_crypto.empty or bool(sym_crypto.iloc[0]):
        return preds  # cannot infer the stock calendar; fall back to every forecast day
    kept = pd.to_datetime(pd.Series([r["date"] for r in sample["rows"]])).unique()
    return preds[preds["is_crypto"] | preds["date"].isin(kept)]


# ---------- decisions ----------
def run_rule(pred: np.ndarray, rv20: np.ndarray, cfg: DecisionConfig) -> tuple:
    """Same state machine as backend.run_decision.build_positions, for one asset in date order."""
    active, days = 1.0, 0
    expo, reasons = [], []
    for f, r in zip(pred, rv20):
        t, reason = decide(float(f), float(r), days, active, cfg)
        if t != active:
            days, active = 0, t
        else:
            days += 1
        expo.append(active)
        reasons.append(reason)
    return np.asarray(expo), reasons


def _counts(expo: np.ndarray) -> dict:
    return {"days": int(len(expo)), "cash": int((expo == 0).sum()), "half": int((expo == 0.5).sum()),
            "full": int((expo == 1).sum())}


def _warnings(surprise: float, reason: str, raw_target: float, raw_reason: str, cfg: DecisionConfig) -> list:
    out = []
    if reason.startswith("hold"):
        out.append(f"Held by the {cfg.min_hold_days}-day minimum hold. Without it the rule would say "
                   f"{raw_target:g} ({raw_reason}).")
    for name, edge in (("choppy", cfg.choppy_surprise), ("stress", cfg.stress_surprise)):
        if edge - 0.1 <= surprise < edge:
            out.append(f"Surprise {surprise:.2f}× is within 0.1 of the {name} boundary ({edge:g}×).")
    if surprise >= cfg.stress_surprise:
        out.append("Realized volatility is far above the forecast: the model is being surprised.")
    return out


def tape(symbol: str, mode: str = "balanced", days: int = TAPE_DAYS) -> dict:
    if mode not in RISK_MODES:
        raise ValueError(f"Unknown risk mode '{mode}'. Use one of: {', '.join(RISK_MODES)}")
    dec = load_decision()
    rows = _backtest_rows(_forecasts(), dec)
    g = rows[rows["symbol"] == symbol.strip().upper()]
    if g.empty:
        raise DecisionUnavailable(f"{symbol.upper()} has no out-of-sample forecasts, so no decision history.")
    g = g.sort_values("date")
    pred, rv = g["pred"].to_numpy(), g["rv20"].to_numpy()
    cfg = RISK_MODES[mode]
    expo, reasons = run_rule(pred, rv, cfg)
    surprise = rv / np.maximum(pred, 1e-6)
    dates = g["date"].dt.strftime("%Y-%m-%d").tolist()
    n = min(days, len(g))
    recent = [{"date": dates[i], "forecast": round(float(pred[i]), 4), "realized": round(float(rv[i]), 4),
               "surprise": round(float(surprise[i]), 3), "exposure": float(expo[i]), "reason": reasons[i]}
              for i in range(len(g) - n, len(g))]
    raw_t, raw_r = target_for(float(surprise[-1]), cfg)
    latest = {**recent[-1], "signal": raw_t, "signal_reason": raw_r,
              "warnings": _warnings(float(surprise[-1]), reasons[-1], raw_t, raw_r, cfg)}
    return {
        "symbol": symbol.strip().upper(),
        "mode": mode,
        "config": asdict(cfg),
        "is_crypto": bool(g["is_crypto"].iloc[0]),
        "first_date": dates[0],
        "counts": _counts(expo),
        "switches_recent": int((np.diff(expo[len(g) - n:]) != 0).sum()),
        "latest": latest,
        "rows": recent,
        "modes": {m: {"config": asdict(c), "counts": _counts(run_rule(pred, rv, c)[0])} for m, c in RISK_MODES.items()},
        "all_days": int((_forecasts()["symbol"] == symbol.strip().upper()).sum()),
    }


# ---------- forecast quality (regime.metrics definitions, vectorized IC) ----------
def daily_ic_fast(df: pd.DataFrame, pred: str, actual: str = "y", min_assets: int = 5) -> pd.Series:
    """Same as regime.metrics.daily_ic (Spearman across assets per date), without groupby.apply."""
    n = df.groupby("date")[pred].transform("size")
    d = df[n >= min_assets]
    ra = d.groupby("date")[pred].rank()
    rb = d.groupby("date")[actual].rank()
    da = ra - ra.groupby(d["date"]).transform("mean")
    db = rb - rb.groupby(d["date"]).transform("mean")
    num = (da * db).groupby(d["date"]).sum()
    den = np.sqrt((da ** 2).groupby(d["date"]).sum() * (db ** 2).groupby(d["date"]).sum())
    return (num / den.replace(0, np.nan)).dropna()


def forecast_metrics(df: pd.DataFrame, pred: str, actual: str = "y") -> Optional[dict]:
    if df.empty:
        return None
    y, f = df[actual].to_numpy(), df[pred].to_numpy()
    sse, sst = float(np.sum((y - f) ** 2)), float(np.sum((y - y.mean()) ** 2))
    ic = daily_ic_fast(df, pred, actual)
    return {"ic": round(float(ic.mean()), 4) if len(ic) else None,
            "rmse": round(float(np.sqrt(sse / len(y))), 4),
            "r2": round(1 - sse / sst, 4) if sst > 0 else None,
            "qlike": round(qlike(y, f), 4), "n": int(len(y))}


def _original_preds(ours: pd.DataFrame) -> Optional[pd.DataFrame]:
    from regime.config import MODEL_DIR, VOL_FLOOR
    p = MODEL_DIR / "serving_paper.csv.gz"
    if not p.exists():
        return None
    raw = pd.read_csv(p, parse_dates=["date"], usecols=["date", "symbol", "pred"])
    out = ours[["date", "symbol", "y"]].merge(raw, on=["date", "symbol"], how="inner")
    out["pred"] = out["pred"].clip(lower=VOL_FLOOR)
    return out


# ---------- portfolio ----------
def cost_curve(systems: dict, bps=tuple(range(0, 101, 5))) -> dict:
    """Arithmetic annual return vs one-way cost, exact given decision.json (see module docstring)."""
    c0 = BACKTEST_COST_BPS
    curve = {s: [round(m["ann_return"] + m["turnover"] * (c0 - c) / 1e4, 5) for c in bps] for s, m in systems.items()}
    be = {}
    bh = systems["buy_and_hold"]
    for s in ("adaptive_binary", "adaptive_tiered"):
        m = systems[s]
        zero = c0 + m["ann_return"] * 1e4 / m["turnover"] if m["turnover"] > 0 else None
        dt = m["turnover"] - bh["turnover"]
        vs_bh = c0 + (m["ann_return"] - bh["ann_return"]) * 1e4 / dt if dt != 0 else None
        be[s] = {"zero_return_bps": round(zero, 1) if zero is not None and zero >= 0 else None,
                 "matches_buy_and_hold_bps": round(vs_bh, 1) if vs_bh is not None and vs_bh >= 0 else None}
    return {"bps": list(bps), "ann_return": curve, "breakeven": be}


def _regimes(bt: pd.DataFrame, preds: pd.DataFrame, orig: Optional[pd.DataFrame]) -> list:
    from .regime_slices import REGIMES
    bt = bt.copy()
    for variant in ("tiered", "binary"):
        cfg = replace(BALANCED, variant=variant)
        bt[variant] = np.nan
        for _, g in bt.groupby("symbol", sort=False):
            bt.loc[g.index, variant] = run_rule(g["pred"].to_numpy(), g["rv20"].to_numpy(), cfg)[0]
    out = []
    for label, (start, end) in REGIMES.items():
        w = bt[(bt["date"] >= start) & (bt["date"] <= end)]
        fw = preds[(preds["date"] >= start) & (preds["date"] <= end)]
        if w.empty:
            out.append({"label": label, "start": start, "end": end, "days": 0})
            continue
        t = w["tiered"]
        ow = orig[(orig["date"] >= start) & (orig["date"] <= end)] if orig is not None else None
        out.append({
            "label": label, "start": start, "end": end,
            "days": int(w["date"].nunique()),
            "exposure": {"adaptive_tiered": round(float(w.groupby("date")["tiered"].mean().mean()), 3),
                         "adaptive_binary": round(float(w.groupby("date")["binary"].mean().mean()), 3),
                         "buy_and_hold": 1.0},
            "tiered_mix": {"cash": round(float((t == 0).mean()), 3), "half": round(float((t == 0.5).mean()), 3),
                           "full": round(float((t == 1).mean()), 3)},
            "mean_surprise": round(float((w["rv20"] / w["pred"]).mean()), 3),
            "forecast": {"ours": forecast_metrics(fw, "pred"),
                         "original": forecast_metrics(ow, "pred") if ow is not None else None,
                         "simple": forecast_metrics(fw, "pred_naive")},
        })
    return out


def portfolio() -> dict:
    """Everything here is fixed at the Balanced config; the risk selector never touches it."""
    dec = load_decision()
    with _lock:
        hit = _cache.get("portfolio")
        if hit and hit[0] == DECISION_JSON.stat().st_mtime:
            return hit[1]
    preds = _forecasts()
    naive = preds.assign(pred_naive=preds["pred_naive"].clip(lower=0.03))
    orig = _original_preds(preds)
    bt = _backtest_rows(preds, dec)
    systems = {s: {k: dec[s][k] for k in METRIC_KEYS} for s in SYSTEMS}
    for m in systems.values():
        m["cost_drag"] = m["turnover"] * BACKTEST_COST_BPS / 1e4
    out = {
        "config": asdict(BALANCED),
        "cost_bps": BACKTEST_COST_BPS,
        "period": {"start": str(preds["date"].min().date()), "end": str(preds["date"].max().date())},
        "assets": int(preds["symbol"].nunique()),
        "crypto_assets": int(preds.loc[preds["is_crypto"], "symbol"].nunique()),
        "rows": {"forecasts": int(len(preds)), "backtest": int(len(bt))},
        "stock_days": {"kept": int(bt.loc[~bt["is_crypto"], "date"].nunique()),
                       "all": int(preds.loc[~preds["is_crypto"], "date"].nunique())},
        "systems": systems,
        "cost_curve": cost_curve(systems),
        "forecast": {"ours": forecast_metrics(preds, "pred"),
                     "original": forecast_metrics(orig, "pred") if orig is not None else None,
                     "simple": forecast_metrics(naive, "pred_naive")},
        "regimes": _regimes(bt, naive, orig),
    }
    with _lock:
        _cache["portfolio"] = (DECISION_JSON.stat().st_mtime, out)
    return out
