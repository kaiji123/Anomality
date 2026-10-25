"""Intelligent alerts and drift detection: deterministic, traceable, and only from data that was
available on the evaluation day.

Every monitor ("check") reports a status for the evaluation day t: ok, alert, or
insufficient_data, with the measured value, the threshold and the windows it used. Alerts are
the checks that fired, grouped into episodes: an alert's id is (monitor, scope, first day of the
episode), with the first day traced back as far as the condition held (not just the lookback
window), so refreshing never creates a duplicate; an episode that ended within the lookback
window is reported as resolved.

Monitors and thresholds (see docs/ALERTS.md):
- market_stress:   market forecast busier than 90% of earlier days (the app's Stormy band).
- model_surprise:  an asset's last 20 days moved >= 2x its forecast (the app's surprise mark).
- regime_change:   the market mood band changed and held for 3 trading days.
- data_quality:    missing forecasts, invalid values, missing returns, stale prices.
- data_drift:      inputs (log rv20, all assets) recent 21 days vs. the 252 before: PSI.
- prediction_drift: forecasts (log pred) recent vs. reference: PSI.
- performance_drift: model QLIKE / simple-estimate QLIKE over the last 21 days whose 5-day
                   result is complete at t. Above 1 = no skill over the simple estimate.
PSI convention: < 0.10 stable, 0.10-0.25 moderate shift, >= 0.25 major shift (a widely used
rule of thumb, not a significance test).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from regime.metrics import qlike

from .time_machine import HORIZON, SURPRISE, Data

STORMY = 0.90
PERSIST_DAYS = 3
RECENT_DAYS, REFERENCE_DAYS = 21, 252
PSI_MODERATE, PSI_MAJOR = 0.10, 0.25
PSI_BINS = 10
NO_SKILL = 1.0
STALE_BUSINESS_DAYS = 3
LOOKBACK_DAYS = 20  # episodes that ended within this many trading days are shown as resolved

CONFIG = {
    "stormy_pct": STORMY, "surprise_ratio": SURPRISE, "regime_persist_days": PERSIST_DAYS,
    "recent_days": RECENT_DAYS, "reference_days": REFERENCE_DAYS, "psi_moderate": PSI_MODERATE,
    "psi_major": PSI_MAJOR, "psi_bins": PSI_BINS, "relative_qlike_no_skill": NO_SKILL,
    "forecast_horizon_days": HORIZON, "stale_business_days": STALE_BUSINESS_DAYS, "lookback_days": LOOKBACK_DAYS,
}

_lock = threading.Lock()
_acks: Dict[str, str] = {}  # alert id -> acknowledged at (in memory: resets when the server restarts)


@dataclass
class Check:
    monitor: str
    type: str
    scope: str
    status: str  # ok | alert | insufficient_data
    value: Optional[float]
    threshold: Optional[float]
    unit: str
    text: str
    evidence: dict
    severity: str = "info"

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------- statistics
def psi(reference: np.ndarray, recent: np.ndarray, bins: int = PSI_BINS) -> float:
    """Population stability index of `recent` vs. `reference`, on the reference's quantile bins."""
    reference, recent = reference[np.isfinite(reference)], recent[np.isfinite(recent)]
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(reference, edges)[0] / len(reference)
    q = np.histogram(recent, edges)[0] / len(recent)
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def _band(storm: Optional[float]) -> Optional[str]:
    from .regime_service import mood_for
    return None if storm is None or np.isnan(storm) else mood_for(storm)["name"]


# ---------------------------------------------------------------- monitors (each uses data up to `t` only)
def _days(data: Data, t: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(data.df.loc[data.df["date"] <= t, "date"].unique()))


def market_checks(data: Data, t: pd.Timestamp, storm: pd.Series) -> List[Check]:
    s = storm.get(t, np.nan)
    if np.isnan(s):
        return [Check("market_stress", "market_stress", "market", "insufficient_data", None, STORMY, "share of earlier days",
                      "Not enough forecast history to rank the market's forecast against earlier days.", {})]
    fired = s >= STORMY
    shown = f"{np.floor(s * 1000) / 10:g}%"  # floor, so 99.5% never reads as 100%
    return [Check("market_stress", "market_stress", "market", "alert" if fired else "ok", round(float(s), 3), STORMY,
                  "share of earlier days",
                  f"The market-wide forecast is busier than {shown} of earlier days (Stormy from {STORMY:.0%}).",
                  {"market_mood": _band(s)}, severity="warning")]


def surprise_checks(data: Data, t: pd.Timestamp) -> List[Check]:
    day = data.df[data.df["date"] == t]
    ok = day[day["pred"].gt(0) & day["rv20"].notna()]
    ratio = (ok["rv20"] / ok["pred"]).round(3)
    hit = ok.assign(ratio=ratio)[ratio >= SURPRISE].sort_values("ratio", ascending=False)
    fired = len(hit) > 0
    return [Check("model_surprise", "market_stress", "assets", "alert" if fired else "ok",
                  float(ratio.max()) if len(ratio) else None, SURPRISE, "realized ÷ forecast",
                  (f"{len(hit)} asset{'s' if len(hit) > 1 else ''} moved at least {SURPRISE:g}× more over the last 20 days "
                   f"than the model forecasts: {', '.join(hit['symbol'].head(6))}." if fired
                   else f"No asset's last 20 days reached {SURPRISE:g}× its forecast."),
                  {"assets": [{"symbol": r.symbol, "ratio": float(r.ratio)} for r in hit.itertuples()][:20],
                   "assets_checked": int(len(ok))}, severity="info")]


def regime_checks(data: Data, t: pd.Timestamp, storm: pd.Series) -> List[Check]:
    days = _days(data, t)
    if len(days) < PERSIST_DAYS + 1:
        return [Check("regime_change", "regime_change", "market", "insufficient_data", None, PERSIST_DAYS, "trading days",
                      "Not enough days to tell a lasting change of mood.", {})]
    bands = [_band(storm.get(d, np.nan)) for d in days[-(LOOKBACK_DAYS + 2):]]
    now_band = bands[-1]
    if now_band is None:
        return [Check("regime_change", "regime_change", "market", "insufficient_data", None, PERSIST_DAYS, "trading days",
                      "The market mood is not defined yet (warming up).", {})]
    run = 0  # how many days the current band has held
    while run < len(bands) and bands[-1 - run] == now_band:
        run += 1
    before = bands[-1 - run] if run < len(bands) else None
    fired = before is not None and PERSIST_DAYS <= run <= LOOKBACK_DAYS
    since = days[-run] if run <= len(days) else days[0]
    return [Check("regime_change", "regime_change", "market", "alert" if fired else "ok", float(run), PERSIST_DAYS, "trading days held",
                  (f"The market mood changed from {before} to {now_band} on {since.date()} and has held for {run} trading days."
                   if fired else f"Market mood: {now_band}, unchanged for {run if run <= LOOKBACK_DAYS else f'more than {LOOKBACK_DAYS}'} trading days."),
                  {"from": before, "to": now_band, "since": str(since.date())},
                  severity="warning" if fired and now_band == "Stormy" else "info")]


def data_quality_checks(data: Data, t: pd.Timestamp, now: Optional[pd.Timestamp],
                        last_price: Optional[pd.Timestamp] = None) -> List[Check]:
    out: List[Check] = []
    hist = data.df[data.df["date"] <= t]
    prev = hist[hist["date"] < t]
    day = hist[hist["date"] == t]
    # Assets that had forecasts before and none on t (a trading day for the panel, so not a holiday).
    missing = sorted(set(prev.loc[prev["date"] == prev["date"].max(), "symbol"]) - set(day["symbol"])) if len(prev) else []
    out.append(Check("missing_forecasts", "data_quality", "assets", "alert" if missing else "ok", float(len(missing)), 0, "assets",
                     f"No forecast on {t.date()} for {', '.join(missing)}, which had one the trading day before." if missing
                     else f"Every asset with a forecast the previous trading day has one on {t.date()}.",
                     {"assets": missing}, severity="warning"))
    bad = day[~(np.isfinite(day["pred"]) & (day["pred"] > 0) & np.isfinite(day["rv20"]) & (day["rv20"] >= 0))]
    out.append(Check("invalid_values", "data_quality", "assets", "alert" if len(bad) else "ok", float(len(bad)), 0, "rows",
                     f"Missing or invalid forecast / rv20 for {', '.join(bad['symbol'])} on {t.date()}." if len(bad)
                     else "All forecasts and rv20 values on this day are finite and positive.",
                     {"assets": sorted(bad["symbol"])}, severity="critical"))
    no_ret = sorted(day.loc[day["ret"].isna(), "symbol"]) if "ret" in day else []
    out.append(Check("missing_returns", "data_quality", "assets", "alert" if no_ret else "ok", float(len(no_ret)), 0, "assets",
                     f"No price return on {t.date()} for {', '.join(no_ret)}." if no_ret
                     else "Every asset has a price return on this trading day.", {"assets": no_ret}, severity="warning"))
    if now is not None and last_price is not None:  # only when evaluating the latest day against the clock
        last = last_price
        lag = int(np.busday_count(last.date(), now.date()))
        out.append(Check("stale_prices", "data_quality", "pipeline", "alert" if lag > STALE_BUSINESS_DAYS else "ok", float(lag),
                         STALE_BUSINESS_DAYS, "business days",
                         f"The newest price is from {last.date()}, {lag} business days before {now.date()}.",
                         {"last_price_date": str(last.date())}, severity="warning"))
    return out


def drift_checks(data: Data, t: pd.Timestamp) -> List[Check]:
    days = _days(data, t)
    out: List[Check] = []
    need = RECENT_DAYS + REFERENCE_DAYS
    windows = None
    if len(days) >= need:
        recent, ref = days[-RECENT_DAYS:], days[-need:-RECENT_DAYS]
        windows = {"recent": [str(recent[0].date()), str(recent[-1].date())], "reference": [str(ref[0].date()), str(ref[-1].date())]}
    for monitor, typ, col, what in (("input_drift", "data_drift", "rv20", "inputs (last 20 days' realized volatility, log scale)"),
                                    ("prediction_drift", "prediction_drift", "pred", "forecasts (log scale)")):
        if windows is None:
            out.append(Check(monitor, typ, "market", "insufficient_data", None, PSI_MAJOR, "PSI",
                             f"Needs {need} trading days of history; {len(days)} available.", {}))
            continue
        x = data.df[["date", col]].copy()
        x[col] = np.log(x[col].where(x[col] > 0))
        a = x.loc[x["date"].isin(ref), col].to_numpy()
        b = x.loc[x["date"].isin(recent), col].to_numpy()
        v = psi(a, b)
        status = "alert" if v >= PSI_MAJOR else "ok"
        level = "major" if v >= PSI_MAJOR else "moderate" if v >= PSI_MODERATE else "stable"
        out.append(Check(monitor, typ, "market", status, round(v, 3), PSI_MAJOR, "PSI",
                         f"Distribution of {what}: {level} shift over the last {RECENT_DAYS} days vs. the {REFERENCE_DAYS} before (PSI {v:.2f}).",
                         {**windows, "recent_median": round(float(np.exp(np.nanmedian(b))), 4),
                          "reference_median": round(float(np.exp(np.nanmedian(a))), 4)},
                         severity="warning" if typ == "prediction_drift" else "info"))
    # Performance: only forecasts whose 5-day result is complete at t (dated <= t - HORIZON trading days).
    cal = data.calendar[data.calendar <= t]
    if len(cal) <= HORIZON:
        done = days[:0]
    else:
        cutoff = cal[-HORIZON - 1]
        done = days[days <= cutoff]
    if len(done) < RECENT_DAYS:
        out.append(Check("performance_drift", "performance_drift", "model", "insufficient_data", None, NO_SKILL, "QLIKE ratio",
                         f"Needs {RECENT_DAYS} days whose {HORIZON}-day result is known; {len(done)} available.", {}))
        return out
    rec = data.df[data.df["date"].isin(done[-RECENT_DAYS:])].dropna(subset=["y", "pred", "pred_naive"])
    if rec.empty:
        out.append(Check("performance_drift", "performance_drift", "model", "insufficient_data", None, NO_SKILL, "QLIKE ratio",
                         "No realized results in the window.", {}))
        return out
    model, simple = qlike(rec["y"].to_numpy(), rec["pred"].to_numpy()), qlike(rec["y"].to_numpy(), rec["pred_naive"].clip(lower=0.03).to_numpy())
    ratio = model / simple if simple > 0 else np.nan
    fired = bool(np.isfinite(ratio) and ratio > NO_SKILL)
    out.append(Check("performance_drift", "performance_drift", "model", "alert" if fired else "ok", round(float(ratio), 3), NO_SKILL,
                     "QLIKE ratio (model ÷ simple)",
                     (f"Over the last {RECENT_DAYS} days with known results, the model's QLIKE was {ratio:.2f}× the simple "
                      f"estimate's: {'worse than' if fired else 'better than'} the simple estimate."),
                     {"window": [str(done[-RECENT_DAYS].date()), str(done[-1].date())], "qlike_model": round(model, 4),
                      "qlike_simple": round(simple, 4), "forecasts": int(len(rec)),
                      "note": f"Forecasts after {done[-1].date()} are excluded: their {HORIZON}-day result was not known on {t.date()}."},
                     severity="warning"))
    return out


def _view(data: Data, t: pd.Timestamp) -> Data:
    """The data as it was at the close of t: nothing dated after t."""
    return Data(df=data.df[data.df["date"] <= t], calendar=data.calendar[data.calendar <= t], rets=data.rets[data.rets.index <= t])


def checks_at(data: Data, t: pd.Timestamp, now: Optional[pd.Timestamp] = None, storm: Optional[pd.Series] = None,
              last_price: Optional[pd.Timestamp] = None) -> List[Check]:
    from .decision_service import RULE, market_storminess
    view = _view(data, t)
    if storm is None:
        storm = market_storminess(view.df, RULE)
    return (market_checks(view, t, storm) + surprise_checks(view, t) + regime_checks(view, t, storm)
            + data_quality_checks(view, t, now, last_price) + drift_checks(view, t))


# Which check function produces each monitor, so an episode's start can be traced back by
# re-running only that function. stale_prices compares with the clock: it exists only on the
# evaluation day, so its episode is that day.
GROUP_OF = {"market_stress": "market", "model_surprise": "surprise", "regime_change": "regime",
            "missing_forecasts": "quality", "invalid_values": "quality", "missing_returns": "quality",
            "input_drift": "drift", "prediction_drift": "drift", "performance_drift": "drift"}


def group_checks(data: Data, t: pd.Timestamp, group: str, storm: pd.Series) -> List[Check]:
    view = _view(data, t)
    if group == "market":
        return market_checks(view, t, storm)
    if group == "surprise":
        return surprise_checks(view, t)
    if group == "regime":
        return regime_checks(view, t, storm)
    if group == "quality":
        return data_quality_checks(view, t, None)
    return drift_checks(view, t)


# ---------------------------------------------------------------- episodes, ids, lifecycle
TITLES = {
    "market_stress": "Market stress: Stormy mood", "model_surprise": "Assets moving far more than forecast",
    "regime_change": "Market regime change", "missing_forecasts": "Missing forecasts", "invalid_values": "Invalid values",
    "missing_returns": "Missing price returns", "stale_prices": "Stale prices", "input_drift": "Input drift",
    "prediction_drift": "Prediction drift", "performance_drift": "Performance drift",
}
SOURCES = {
    "market_stress": "walk-forward forecasts (models/serving_paper_log.csv.gz), market average over stocks",
    "model_surprise": "walk-forward forecasts and rv20", "regime_change": "market mood bands (backend/regime_service.py)",
    "input_drift": "rv20, all assets", "prediction_drift": "walk-forward forecasts, all assets",
    "performance_drift": "forecasts vs. realized 5-day volatility (y), completed horizons only",
}


def evaluate(data: Data, t: pd.Timestamp, now: Optional[pd.Timestamp] = None) -> dict:
    """Checks on day t, and alerts as episodes over the trading days up to t."""
    from .decision_service import RULE, market_storminess
    days = _days(data, t)
    if not len(days) or days[-1] != t:
        raise ValueError(f"{t.date()} is not a day with forecasts")
    hist = data.df[data.df["date"] <= t]
    storm = market_storminess(hist, RULE)
    window = days[-(LOOKBACK_DAYS + 1):]
    last_price = data.calendar.max() if now is not None else None  # staleness: newest price we have now
    per_day = {d: checks_at(data, d, now if d == t else None, storm, last_price if d == t else None) for d in window}
    today = per_day[t]
    pos = {d: i for i, d in enumerate(days)}
    older: Dict[tuple, List[Check]] = {}  # (group, day) -> checks, for days before the window

    def fired(monitor: str, d: pd.Timestamp) -> bool:
        if d in per_day:
            checks = per_day[d]
        else:
            group = GROUP_OF.get(monitor)
            if group is None:
                return False
            if (group, d) not in older:
                older[(group, d)] = group_checks(data, d, group, storm)
            checks = older[(group, d)]
        return any(x.monitor == monitor and x.status == "alert" for x in checks)

    def episode_start(monitor: str, end: pd.Timestamp) -> tuple:
        """First day of the run of consecutive alert days ending on `end`, looking back as far as
        the condition holds (not just inside the window). Second value: the run reaches the first
        day with data, so the true start may be earlier."""
        i = pos[end]
        while i > 0 and fired(monitor, days[i - 1]):
            i -= 1
        return days[i], i == 0

    alerts = []
    for c in today:  # active episodes
        if c.status == "alert":
            start, at_data_start = episode_start(c.monitor, t)
            alerts.append(_alert(c, start, t, "active", at_data_start=at_data_start))
    for i, d in enumerate(window[:-1]):  # episodes that ended inside the window
        nxt = window[i + 1]
        for c in per_day[d]:
            if c.status == "alert" and not fired(c.monitor, nxt):
                start, at_data_start = episode_start(c.monitor, d)
                alerts.append(_alert(c, start, d, "resolved", resolved_on=nxt, at_data_start=at_data_start))
    order = {"critical": 0, "warning": 1, "info": 2}
    alerts.sort(key=lambda a: (a["status"] == "resolved", order[a["severity"]], a["last_seen"]), reverse=False)
    return {"as_of": str(t.date()), "alerts": alerts, "checks": [c.as_dict() for c in today], "config": CONFIG,
            "note": "Alerts describe measured conditions; they do not predict a crash or a loss."}


def _alert(c: Check, start: pd.Timestamp, last: pd.Timestamp, status: str, resolved_on: Optional[pd.Timestamp] = None,
           at_data_start: bool = False) -> dict:
    out = {
        "id": f"{c.monitor}:{c.scope}:{start.date()}", "type": c.type, "monitor": c.monitor, "severity": c.severity,
        "title": TITLES.get(c.monitor, c.monitor), "description": c.text, "observed_value": c.value, "threshold": c.threshold,
        "unit": c.unit, "scope": c.scope, "source": SOURCES.get(c.monitor, "walk-forward forecasts and prices"),
        "first_seen": str(start.date()), "last_seen": str(last.date()), "status": status, "evidence": c.evidence,
        # False when the episode runs back to the first day with data: it may have started earlier.
        "first_seen_exact": not at_data_start,
    }
    if resolved_on is not None:
        out["resolved_on"] = str(resolved_on.date())
    return out


def with_acks(result: dict) -> dict:
    """Copy of `result` where active alerts the user acknowledged are marked so. Resolved alerts
    stay resolved: the condition is gone, whatever the user did."""
    out = {**result, "alerts": [dict(a) for a in result["alerts"]]}
    with _lock:
        for a in out["alerts"]:
            if a["status"] == "active" and a["id"] in _acks:
                a["status"], a["acknowledged_at"] = "acknowledged", _acks[a["id"]]
    return out


def acknowledge(alert_id: str, when: Optional[str] = None) -> dict:
    with _lock:
        _acks[alert_id] = when or pd.Timestamp.now().isoformat(timespec="seconds")
        return {"id": alert_id, "acknowledged_at": _acks[alert_id]}


_results: Dict[str, tuple] = {}  # "slot" -> (data object, {date: (time, result)}), for the current data only
TTL = 600


def latest(date: Optional[str] = None) -> dict:
    """Alerts on the newest forecast day (checked against the clock for stale prices), or on a past
    day (evaluated with only the data available then). Cached; acknowledgements applied fresh."""
    import time as _time
    from .time_machine import load
    data = load()
    with _lock:
        slot = _results.get("slot")
        if slot is None or slot[0] is not data:  # tied to the data object, not its (reusable) id
            slot = (data, {})
            _results["slot"] = slot
        hit = slot[1].get(date)
    if hit and _time.time() - hit[0] < TTL:
        return with_acks(hit[1])
    days = pd.DatetimeIndex(sorted(data.df["date"].unique()))
    if date is None:
        t, now = days[-1], pd.Timestamp.now().normalize()
    else:
        req = pd.Timestamp(date)
        if req < days[0] or req > days[-1]:
            raise ValueError(f"Alerts can be evaluated from {days[0].date()} to {days[-1].date()}.")
        t, now = days[days <= req][-1], None
    result = evaluate(data, t, now)
    with _lock:
        slot = _results.get("slot")
        if slot is not None and slot[0] is data:
            if len(slot[1]) > 100:
                slot[1].clear()
            slot[1][date] = (_time.time(), result)
    return with_acks(result)
