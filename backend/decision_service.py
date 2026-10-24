"""Serves the decision layer: today's position for every asset, and the backtest of acting on
the rule vs. buy-and-hold.

The headline rule follows the regime model's market mood: half invested when it's Stormy,
fully invested otherwise. The backtest only uses walk-forward test forecasts (out-of-sample).
Today's position also runs the rule through the days after those forecasts end, using the
final model's forecasts.
"""
from __future__ import annotations

import threading
import time
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from regime.config import MARKET_PERIODS, TrainConfig
from regime.dataset import daily_returns

from . import regime_service
from .backtest import hold, portfolio_returns, stats
from .decision import COST_BPS, EXEC_LAG, LEVELS, DecisionConfig, MoodConfig, mood_positions, positions, storminess

Rule = Optional[Union[MoodConfig, DecisionConfig]]
SYSTEMS: Dict[str, tuple] = {  # name -> (label, config; None = always fully invested)
    "mood_rule": ("Mood rule: half when Stormy", MoodConfig()),
    "buy_and_hold": ("Buy and hold", None),
    "surprise_rule": ("Previous rule: surprise", DecisionConfig(variant="tiered")),
}
RULE: MoodConfig = SYSTEMS["mood_rule"][1]  # the rule shown in the app
TTL = 600  # seconds, same as the regime model

_lock = threading.Lock()
_state: dict = {}


class DecisionUnavailable(Exception):
    pass


# ---------------------------------------------------------------- data and simulation
def frame(use_cache: bool = True) -> pd.DataFrame:
    """Walk-forward forecasts joined with each day's return.
    Columns: symbol, date, pred, rv20, is_crypto, ret; sorted by symbol, date."""
    preds = regime_service.recommended_forecasts()
    if preds is None:
        raise DecisionUnavailable("No walk-forward forecasts found. Run `python -m regime.train` first.")
    try:
        rets = daily_returns(TrainConfig(), use_cache)
    except RuntimeError as e:  # no network and no price cache
        raise DecisionUnavailable(str(e)) from e
    ret = rets.stack(future_stack=True).rename("ret").reorder_levels([1, 0])
    df = preds[["pred", "rv20", "is_crypto"]].join(ret.rename_axis(["symbol", "date"]), how="left")
    return df.dropna(subset=["ret"]).reset_index().sort_values(["symbol", "date"], ignore_index=True)


def market_storminess(df: pd.DataFrame, cfg: MoodConfig = RULE) -> pd.Series:
    """Per date: share of earlier days whose market forecast (average over stocks) was lower.
    This is the regime model's market mood as a number; >= cfg.stormy_pct means Stormy."""
    stocks = df[~df["is_crypto"].astype(bool)] if "is_crypto" in df else df
    mkt = stocks.groupby("date")["pred"].mean().sort_index()
    return pd.Series(storminess(mkt.to_numpy(), cfg.min_history), index=mkt.index, name="storm")


def simulate(df: pd.DataFrame, cfg: Rule) -> pd.DataFrame:
    """Add `exposure` and `reason` per row (and `storm` for the mood rule). `df` is sorted by symbol, date."""
    if cfg is None:
        return df.assign(exposure=1.0, reason="always invested")
    if isinstance(cfg, MoodConfig):
        storm = market_storminess(df, cfg)
        pct = df["date"].map(storm).to_numpy(dtype=float)
        exposure, reasons = mood_positions(pct, cfg)
        return df.assign(exposure=exposure, reason=reasons, storm=pct)
    parts = []
    for _, g in df.groupby("symbol", sort=False):
        exposure, reasons = positions(g["pred"].to_numpy(), g["rv20"].to_numpy(), cfg)
        parts.append(g.assign(exposure=exposure, reason=reasons))
    return pd.concat(parts, ignore_index=True)


def _rounded(s: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}


def _sliced(daily: pd.DataFrame) -> dict:
    out = {"all": _rounded(stats(daily["net"], daily["turnover"]))}
    for label, start, end in MARKET_PERIODS.values():
        m = (daily.index >= start) & (daily.index <= end)
        out[label] = _rounded(stats(daily["net"][m], daily["turnover"][m]))
    return out


def held_by_system(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Each system's decisions on `df`, with the exposure actually held each day."""
    return {name: hold(simulate(df, cfg)) for name, (_, cfg) in SYSTEMS.items()}


def summarize(df: pd.DataFrame, cost_bps: float = COST_BPS) -> dict:
    """Backtest every system on the equal-weight portfolio of all assets in `df`."""
    held = held_by_system(df)
    daily = {name: portfolio_returns(h, cost_bps) for name, h in held.items()}
    equity = pd.DataFrame({n: (1 + d["net"]).cumprod() * 100 for n, d in daily.items()}).resample("W-FRI").last()
    return {
        "systems": {name: {"label": SYSTEMS[name][0], **_sliced(d)} for name, d in daily.items()},
        "equity": {"date": [str(d.date()) for d in equity.index],
                   **{n: equity[n].round(2).tolist() for n in equity.columns}},
        "since": str(df["date"].min().date()),
        "through": str(df["date"].max().date()),
        "assets": int(df["symbol"].nunique()),
        "cost_bps": cost_bps,
        "periods": {label: {"start": a, "end": b} for label, a, b in MARKET_PERIODS.values()},
        "_held": held,
    }


# ---------------------------------------------------------------- cached state for the API
def _today_tape() -> pd.DataFrame:
    """The rule run day by day through today for every asset (walk-forward, then live forecasts)."""
    wf = regime_service.recommended_forecasts()[["pred", "rv20", "is_crypto"]]
    last = wf.index.get_level_values("date").max()
    live = regime_service.live_forecasts(last).reorder_levels(["symbol", "date"])
    crypto = set(wf.index.get_level_values("symbol")[wf["is_crypto"].astype(bool).to_numpy()])
    live = live.assign(is_crypto=live.index.get_level_values("symbol").isin(crypto))
    tape = pd.concat([wf, live]).sort_index().reset_index()
    return simulate(tape, RULE)


def _ensure() -> dict:
    with _lock:
        if _state.get("at") and time.time() - _state["at"] < TTL:
            return _state
        try:
            df = frame()
            tape = _today_tape()
        except regime_service.RegimeUnavailable as e:
            raise DecisionUnavailable(str(e)) from e
        _state.update(summary=summarize(df), tape=tape, at=time.time())
        return _state


def explain(reason: str, pct: Optional[float]) -> str:
    """The rule's reason in plain words."""
    if reason.startswith("stormy"):
        return (f"The market mood is Stormy: this week's forecast is busier than {pct:.0%} of past days, "
                f"so the rule holds every asset at half.")
    if reason.startswith("warming"):
        return "Not enough forecast history yet to judge the mood, so the rule stays fully invested."
    return (f"The market mood isn't Stormy (this week's forecast is busier than {pct:.0%} of past days; "
            f"Stormy starts at {RULE.stormy_pct:.0%}), so the rule stays fully invested.")


def _rule() -> dict:
    return {"stormy_pct": RULE.stormy_pct, "cut_to": RULE.cut_to, "min_history": RULE.min_history,
            "cost_bps": COST_BPS, "exec_lag": EXEC_LAG}


def _market(tape: pd.DataFrame) -> pd.DataFrame:
    """One row per date: storminess and the rule's exposure (the same for every asset)."""
    return tape.groupby("date")[["storm", "exposure"]].first().reset_index()


def _since(m: pd.DataFrame) -> pd.Timestamp:
    changed = m["exposure"].ne(m["exposure"].shift())
    return m["date"][changed].iloc[-1]


def plain_portfolio() -> dict:
    st = _ensure()
    tape = st["tape"]
    m = _market(tape)
    last = m.iloc[-1]
    since = _since(m)
    changes = m[m["exposure"].ne(m["exposure"].shift())].iloc[1:]  # skip the starting position
    prev = m["exposure"].shift()[changes.index]
    today = tape.groupby("symbol").tail(1)
    summary = {k: v for k, v in st["summary"].items() if not k.startswith("_")}
    storm = None if np.isnan(last["storm"]) else round(float(last["storm"]), 3)
    return {**summary, "rule": _rule(),
            "today": {"date": str(last["date"].date()),
                      "level": LEVELS[last["exposure"]], "exposure": float(last["exposure"]),
                      "storm": storm, "since": str(since.date()),
                      "days_in_position": int((m["date"] > since).sum()),
                      "text": explain(tape[tape["date"] == last["date"]]["reason"].iloc[0], storm),
                      "counts": {name: int((today["exposure"] == lvl).sum()) for lvl, name in LEVELS.items()},
                      "avg_exposure": round(float(today["exposure"].mean()), 3),
                      "assets": [{"symbol": r.symbol, "level": LEVELS[r.exposure], "exposure": r.exposure}
                                 for r in today.itertuples()]},
            "market": {"date": [str(d.date()) for d in m["date"]],
                       "storm": [None if np.isnan(v) else round(float(v), 3) for v in m["storm"]],
                       "exposure": m["exposure"].tolist(),
                       "time_half": round(float((m["exposure"] < 1).mean()), 3),
                       "switches": [{"date": str(r.date.date()), "from": LEVELS[p], "to": LEVELS[r.exposure],
                                     "storm": None if np.isnan(r.storm) else round(float(r.storm), 3)}
                                    for r, p in zip(changes.itertuples(), prev)][::-1]}}


def plain_asset(symbol: str) -> dict:
    st = _ensure()
    symbol = symbol.upper()
    tape = st["tape"]
    g = tape[tape["symbol"] == symbol]
    if g.empty:
        raise DecisionUnavailable(f"{symbol} isn't in the model's 56-asset universe, so the decision rule doesn't cover it.")
    last = g.iloc[-1]
    m = _market(tape)
    since = _since(m)
    held = st["summary"]["_held"]
    record = {}
    for name in ("mood_rule", "buy_and_hold"):
        h = held[name][held[name]["symbol"] == symbol]
        record[name] = _sliced(portfolio_returns(h))
    storm = None if np.isnan(last["storm"]) else round(float(last["storm"]), 3)
    return {
        "symbol": symbol,
        "date": str(last["date"].date()),
        "level": LEVELS[last["exposure"]],
        "exposure": float(last["exposure"]),
        "storm": storm,
        "text": explain(last["reason"], storm),
        "since": str(since.date()),
        "days_in_position": int((m["date"] > since).sum()),
        "forecast_move": regime_service.weekly(last["pred"]),
        "rule": _rule(),
        "time_in": {name: round(float((g["exposure"] == lvl).mean()), 3) for lvl, name in LEVELS.items()},
        "record": record,
    }
