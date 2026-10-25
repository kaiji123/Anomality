"""Historical Time Machine: what the system knew on a past day, and what happened next.

For an asset and a date t the view has two strictly separated parts:

- known: built ONLY from rows dated <= t. The walk-forward forecast issued at the close of t
  (made by a model that never saw t or later), the last 20 days' realized volatility, the
  market mood, and the decision the rule gives on that information. Decisions were never
  recorded: they are reconstructed by re-running the rule (backend/decision.py) on the data
  truncated at t, and checked against the backtest's own decision for that day.
- outcome: what was observed after t, shown separately: realized volatility over the
  forecast's own horizon (t+1 .. t+5, the same definition as the forecast), the price move over
  those days, and whether the horizon is complete in the price data.

Timing: forecast and decision at the close of t; trade at the close of t+1; first return
earned on t+2 (backend/decision.py EXEC_LAG). Dates are U.S. trading days for every asset
(crypto is sampled on the same calendar). A non-trading date uses the latest trading day
before it ("as of"), never a later one.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from regime.config import MARKET_PERIODS, TrainConfig

from .decision import EXEC_LAG, LEVELS, DecisionConfig, positions

HORIZON = TrainConfig().horizon  # the forecast's horizon in trading days (y = vol over t+1..t+h)
SURPRISE = 2.0  # realized >= 2x forecast = a surprise, as in the anomaly report
WINDOW_BEFORE, WINDOW_AFTER = 120, 30  # trading days shown around t in the chart
TTL = 600

_lock = threading.Lock()
_cache: dict = {}


class TimeMachineError(Exception):
    """The Time Machine cannot answer. Missing data for a valid request is not an error."""


class UnknownAsset(TimeMachineError):
    pass


class BadDate(TimeMachineError):
    pass


@dataclass
class Data:
    df: pd.DataFrame        # one row per forecast: symbol, date, pred, rv20, is_crypto, ret, y, pred_naive
    calendar: pd.DatetimeIndex  # trading days in the price data (may extend past the last forecast)
    rets: pd.DataFrame      # daily simple returns, one column per symbol, on `calendar`


def load() -> Data:
    """The backtest's own frame (same forecasts and prices), plus the realized target `y`."""
    with _lock:
        hit = _cache.get("data")
        if hit and time.time() - hit[0] < TTL:
            return hit[1]
    from regime.dataset import daily_returns

    from . import regime_service
    from .decision_service import frame
    preds = regime_service.recommended_forecasts()
    if preds is None:
        raise TimeMachineError("No walk-forward forecasts found. Run `python -m regime.train` first.")
    df = frame()
    extra = preds[["y", "pred_naive"]].reset_index()
    df = df.merge(extra, on=["symbol", "date"], how="left").sort_values(["symbol", "date"], ignore_index=True)
    rets = daily_returns(TrainConfig())
    data = Data(df=df, calendar=pd.DatetimeIndex(rets.index), rets=rets)
    with _lock:
        _cache["data"] = (time.time(), data)
    return data


def _num(v) -> Optional[float]:
    return None if v is None or pd.isna(v) else round(float(v), 4)


def _weekly(v: Optional[float]) -> Optional[float]:
    from .regime_service import weekly
    return None if v is None else weekly(v)


# ---------------------------------------------------------------- known at t
def known_at(df: pd.DataFrame, symbol: str, as_of: pd.Timestamp) -> dict:
    """Everything the system knew at the close of `as_of`. Uses ONLY rows dated <= as_of."""
    from .decision_service import RULE, market_storminess, simulate
    from .regime_service import mood_for
    hist = df[df["date"] <= as_of]
    g = hist[hist["symbol"] == symbol]
    row = g.iloc[-1]
    pred, rv20 = _num(row["pred"]), _num(row["rv20"])
    flags: List[str] = []

    # The app's rule (market mood): needs the whole market's forecasts up to t.
    mood_row = simulate(hist, RULE)
    mood_row = mood_row[(mood_row["symbol"] == symbol) & (mood_row["date"] == as_of)].iloc[0]
    storm = _num(mood_row["storm"])
    if storm is None:
        flags.append(f"Fewer than {RULE.min_history} days of forecasts before this date: the rule is warming up.")
    mood = {"exposure": float(mood_row["exposure"]), "level": LEVELS[float(mood_row["exposure"])],
            "reason": mood_row["reason"], "storminess": storm,
            "market_mood": mood_for(storm)["name"] if storm is not None else None,
            "rule": {"stormy_pct": RULE.stormy_pct, "cut_to": RULE.cut_to, "min_history": RULE.min_history}}

    # The previous rule (surprise), for comparison: one asset's own history up to t.
    prev = None
    if g["pred"].notna().all() and g["rv20"].notna().all():
        cfg = DecisionConfig(variant="tiered")
        expo, reasons = positions(g["pred"].to_numpy(), g["rv20"].to_numpy(), cfg)
        prev = {"exposure": float(expo[-1]), "level": LEVELS[float(expo[-1])], "reason": reasons[-1],
                "rule": {"choppy_surprise": cfg.choppy_surprise, "stress_surprise": cfg.stress_surprise,
                         "min_hold_days": cfg.min_hold_days}}
    else:
        flags.append("Realized volatility (rv20) is missing on or before this date for this asset: "
                     "the previous (surprise) rule cannot be reconstructed.")

    own = g["pred"].iloc[:-1]
    own_pct = float((own < row["pred"]).mean()) if len(own) >= 60 else None
    mkt = market_storminess(hist, RULE)  # equals `storm`; kept for the chart
    return {
        "date": str(as_of.date()),
        "forecast": {"vol": pred, "weekly_move_pct": _weekly(pred), "horizon_days": HORIZON,
                     "simple_estimate_vol": _num(row.get("pred_naive"))},
        "rv20": rv20,
        "surprise_ratio": round(rv20 / pred, 3) if pred and rv20 is not None else None,
        "asset_percentile": None if own_pct is None else round(own_pct, 3),
        "decision": {"mood_rule": mood, "previous_rule": prev,
                     "provenance": "Reconstructed: the rule re-run on data up to this date. Decisions were not "
                                   "recorded at the time, and nothing was traded live."},
        "history_days": int(len(g)),
        "flags": flags,
        "_market_storm": mkt,
    }


# ---------------------------------------------------------------- after t
def outcome_after(data: Data, symbol: str, as_of: pd.Timestamp, forecast_vol: Optional[float]) -> dict:
    """What was observed after `as_of`: the forecast horizon t+1..t+HORIZON. Never used above."""
    after = data.calendar[data.calendar > as_of]
    days = after[:HORIZON]
    base = {"horizon_days": HORIZON, "start": str(days[0].date()) if len(days) else None,
            "end": str(days[-1].date()) if len(days) else None, "days_available": int(len(days))}
    if len(days) < HORIZON:
        return {**base, "status": "incomplete",
                "text": f"Result not available yet: only {len(days)} of the {HORIZON} trading days after this date are in the price data."}
    r = data.rets[symbol].reindex(days) if symbol in data.rets else pd.Series(np.nan, index=days)
    move = None if r.isna().any() else float((1 + r).prod() - 1)
    y = data.df.loc[(data.df["symbol"] == symbol) & (data.df["date"] == as_of), "y"]
    realized = _num(y.iloc[0]) if len(y) else None
    if realized is None:
        return {**base, "status": "unavailable", "move_pct": None if move is None else round(move * 100, 2),
                "text": "The realized volatility for this horizon is not in the stored results."}
    expected = _weekly(forecast_vol)
    ratio = round(realized / forecast_vol, 3) if forecast_vol else None
    return {
        **base, "status": "complete",
        "realized_vol": realized, "realized_weekly_move_pct": _weekly(realized),
        "forecast_vol": forecast_vol, "ratio": ratio,
        "surprise": bool(ratio is not None and ratio >= SURPRISE),
        "move_pct": None if move is None else round(move * 100, 2),
        "expected_move_pct": expected,
        "inside_expected_move": None if move is None or expected is None else bool(abs(move * 100) <= expected),
    }


# ---------------------------------------------------------------- the view
def resolve_date(data: Data, symbol: str, requested: pd.Timestamp) -> dict:
    g = data.df[data.df["symbol"] == symbol]
    first, last = g["date"].min(), g["date"].max()
    if requested < first:
        return {"as_of": None, "status": "before_history",
                "text": f"No forecast exists for {symbol} before {first.date()} (the first walk-forward test day)."}
    if requested > last:
        return {"as_of": None, "status": "after_history",
                "text": f"The stored walk-forward forecasts for {symbol} end on {last.date()}."}
    as_of = g.loc[g["date"] <= requested, "date"].max()
    note = None
    if as_of != requested:
        note = (f"{requested.date()} is not a trading day with a forecast, so this shows the latest information "
                f"available on that date: the close of {as_of.date()}.")
    return {"as_of": as_of, "status": "ok", "text": note}


def view(symbol: str, date: str, data: Optional[Data] = None) -> dict:
    data = data or load()
    symbol = symbol.strip().upper()
    if symbol not in set(data.df["symbol"]):
        raise UnknownAsset(f"{symbol} isn't in the model's {data.df['symbol'].nunique()}-asset universe.")
    try:
        requested = pd.Timestamp(date)
    except (ValueError, TypeError):
        raise BadDate(f"'{date}' is not a date (use YYYY-MM-DD).")
    res = resolve_date(data, symbol, requested)
    out = {"symbol": symbol, "requested_date": str(requested.date()), "status": res["status"],
           "note": res["text"], "timing": None, "known": None, "outcome": None, "series": None}
    if res["as_of"] is None:
        return out
    as_of = res["as_of"]
    known = known_at(data.df, symbol, as_of)
    storm = known.pop("_market_storm")
    after = data.calendar[data.calendar > as_of]
    out["timing"] = {
        "forecast_and_decision": str(as_of.date()),
        "traded": str(after[EXEC_LAG - 1].date()) if len(after) >= EXEC_LAG else None,
        "first_return": str(after[EXEC_LAG].date()) if len(after) > EXEC_LAG else None,
        "convention": "Forecast and decision at the close of the selected day; traded at the next day's close; "
                      "the position first earns the return of the day after that.",
    }
    known["decision"]["mood_rule"]["matches_backtest"] = _matches_backtest(data, symbol, as_of, known)
    out["known"] = known
    out["outcome"] = outcome_after(data, symbol, as_of, known["forecast"]["vol"])
    out["series"] = _series(data, symbol, as_of, storm)
    return out


def _full_simulation(data: Data) -> pd.DataFrame:
    """The backtest's decisions on all of `data`. One slot, tied to the data object itself (not
    its id, which Python reuses): new data replaces the old result instead of piling up."""
    from .decision_service import RULE, simulate
    with _lock:
        slot = _cache.get("full")
    if slot is not None and slot[0] is data:
        return slot[1]
    full = simulate(data.df, RULE)
    with _lock:
        _cache["full"] = (data, full)
    return full


def _matches_backtest(data: Data, symbol: str, as_of: pd.Timestamp, known: dict) -> Optional[bool]:
    """Is the decision reconstructed from data up to t the one the full backtest used that day?"""
    full = _full_simulation(data)
    hit = full[(full["symbol"] == symbol) & (full["date"] == as_of)]
    return None if hit.empty else bool(float(hit["exposure"].iloc[0]) == known["decision"]["mood_rule"]["exposure"])


def _series(data: Data, symbol: str, as_of: pd.Timestamp, storm: pd.Series) -> dict:
    """Chart data. `known` points are dated <= t; `after` points are dated > t and drawn apart."""
    g = data.df[data.df["symbol"] == symbol].set_index("date")
    known = g[g.index <= as_of].iloc[-WINDOW_BEFORE:]
    later = g[g.index > as_of].iloc[:WINDOW_AFTER]
    st = storm.reindex(known.index)
    return {
        "known": {"date": [str(d.date()) for d in known.index],
                  "forecast": [_num(v) for v in known["pred"]], "rv20": [_num(v) for v in known["rv20"]],
                  "storminess": [_num(v) for v in st]},
        "after": {"date": [str(d.date()) for d in later.index], "rv20": [_num(v) for v in later["rv20"]]},
    }


def meta(data: Optional[Data] = None) -> dict:
    data = data or load()
    g = data.df.groupby("symbol")["date"].agg(["min", "max"])
    return {
        "assets": [{"symbol": s, "first": str(r["min"].date()), "last": str(r["max"].date()),
                    "crypto": bool(data.df.loc[data.df["symbol"] == s, "is_crypto"].iloc[0])} for s, r in g.iterrows()],
        "first": str(data.df["date"].min().date()), "last": str(data.df["date"].max().date()),
        "prices_through": str(data.calendar.max().date()),
        "horizon_days": HORIZON, "exec_lag": EXEC_LAG,
        "presets": [{"label": label, "date": start} for label, start, _ in MARKET_PERIODS.values()],
    }


MAX_VIEWS = 200


def cached_view(symbol: str, date: str, data: Optional[Data] = None) -> dict:
    """`view`, cached per (symbol, date) for the current data object only, at most MAX_VIEWS."""
    data = data or load()
    key = (symbol.strip().upper(), date)
    with _lock:
        slot = _cache.get("views")
        if slot is None or slot[0] is not data:  # new data: drop every view built from the old one
            slot = (data, OrderedDict())
            _cache["views"] = slot
        views = slot[1]
        if key in views:
            views.move_to_end(key)
            return views[key]
    out = view(symbol, date, data)
    with _lock:
        slot = _cache.get("views")
        if slot is not None and slot[0] is data:
            slot[1][key] = out
            while len(slot[1]) > MAX_VIEWS:
                slot[1].popitem(last=False)
    return out
