"""Robustness of the decision layer: trading costs and forecast quality, for the dashboard.

- Costs: the same held positions as the backtest (backend.decision_service.held_by_system),
  re-priced at other one-way costs. Nothing is extrapolated: each row is a full backtest.
- Forecast quality: IC, RMSE, R² and QLIKE with the definitions in regime.metrics, for the
  recommended model, the original research model and the simple "last month" estimate, over
  all out-of-sample forecasts and in each market phase. These score the *forecast*, not the
  portfolio.

Run: python -m backend.robustness
"""
from __future__ import annotations

import threading
from typing import Dict, Optional

import numpy as np
import pandas as pd

from regime.config import MARKET_PERIODS, VOL_FLOOR
from regime.metrics import qlike

COSTS_BPS = (0, 5, 10, 20, 30, 50)
SYSTEMS = ("mood_rule", "buy_and_hold", "surprise_rule")

_lock = threading.Lock()
_cache: Dict[str, dict] = {}


# ---------------------------------------------------------------- costs
def cost_table(held: Dict[str, pd.DataFrame], costs=COSTS_BPS) -> dict:
    """Full-period stats of each system at each one-way cost (bps)."""
    from .backtest import portfolio_returns, stats
    rows = {name: [] for name in SYSTEMS if name in held}
    for c in costs:
        for name in rows:
            d = portfolio_returns(held[name], cost_bps=c)
            s = stats(d["net"], d["turnover"])
            rows[name].append({k: round(s[k], 4) for k in ("net_return", "ann_return", "sharpe", "max_drawdown", "turnover")})
    return {"bps": list(costs), "systems": rows}


def growth_breakeven(held: Dict[str, pd.DataFrame], a: str = "mood_rule", b: str = "buy_and_hold",
                     hi: float = 100.0, tol: float = 0.05) -> Optional[float]:
    """One-way cost (bps) above which `a` stops out-growing `b` (compounded). None if `a` never
    out-grows `b` at zero cost, or still does at `hi` bps. Bisection on full backtests."""
    from .backtest import portfolio_returns

    def gap(c: float) -> float:
        return _growth(portfolio_returns(held[a], cost_bps=c)) - _growth(portfolio_returns(held[b], cost_bps=c))

    lo_gap, hi_gap = gap(0.0), gap(hi)
    if lo_gap <= 0 or hi_gap > 0:
        return None
    lo = 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if gap(mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 1)


def rebalance_trades(R: np.ndarray) -> np.ndarray:
    """Share of the portfolio traded each day to bring an equal-weight portfolio back to 1/N
    after that day's returns (rows = days, columns = assets; no missing values)."""
    n = R.shape[1]
    w_end = (1 + R) / n / (1 + R.mean(axis=1, keepdims=True))
    return np.abs(w_end - 1 / n).sum(axis=1)


def drift_basket(R: np.ndarray) -> np.ndarray:
    """Daily return of a basket bought at 1/N on day one and never rebalanced."""
    w, out = np.full(R.shape[1], 1 / R.shape[1]), np.empty(len(R))
    for i, r in enumerate(R):
        out[i] = w @ r
        w = w * (1 + r) / (1 + out[i])
    return out


def portfolio_conventions(df: pd.DataFrame, held: Dict[str, pd.DataFrame], cost_bps: float = 5.0) -> dict:
    """The mood rule and buy-and-hold under four portfolio conventions, same positions and data.

    A: the published backtest (equal weight over the assets present that day, rebalanced daily,
       rebalancing trades not charged). B: only assets present from the first day. C: B with
       weights left to drift (a true buy-and-hold basket). D: B with daily rebalancing charged.
    The mood rule holds every asset at the same exposure, so in C and D it is that exposure
    times the basket."""
    from .backtest import portfolio_returns, stats
    c = cost_bps / 1e4

    def s(net: pd.Series, turnover: pd.Series) -> dict:
        r = stats(net, turnover)
        return {k: round(r[k], 4) for k in ("net_return", "ann_return", "ann_vol", "sharpe", "max_drawdown", "turnover")}

    start = df["date"].min()
    full = sorted(sym for sym, g in df.groupby("symbol") if g["date"].min() == start)
    out = {"A": {}, "B": {}, "C": {}, "D": {}, "assets": {"A": int(df["symbol"].nunique()), "fixed": len(full)}}
    for name in ("mood_rule", "buy_and_hold"):
        d = portfolio_returns(held[name], cost_bps)
        out["A"][name] = s(d["net"], d["turnover"])
        d = portfolio_returns(held[name][held[name]["symbol"].isin(full)], cost_bps)
        out["B"][name] = s(d["net"], d["turnover"])
    R = df[df["symbol"].isin(full)].pivot(index="date", columns="symbol", values="ret").sort_index()
    if R.isna().any().any():  # C and D need every fixed-universe asset on every day
        out["C"] = out["D"] = None
        return out
    m = held["mood_rule"]
    expo = m[m["symbol"] == full[0]].set_index("date")["held"].reindex(R.index)  # same for every asset
    traded = expo.diff().abs().fillna(expo)
    entry = pd.Series(np.r_[1.0, np.zeros(len(R) - 1)], index=R.index)  # buy-and-hold buys once
    basket = pd.Series(drift_basket(R.to_numpy()), index=R.index)
    out["C"]["buy_and_hold"] = s(basket - c * entry, entry)
    out["C"]["mood_rule"] = s(expo * basket - c * traded, traded)
    eq, rb = R.mean(axis=1), pd.Series(rebalance_trades(R.to_numpy()), index=R.index)
    out["D"]["buy_and_hold"] = s(eq - c * (entry + rb), entry + rb)
    out["D"]["mood_rule"] = s(expo * eq - c * (traded + expo * rb), traded + expo * rb)
    return out


def _growth(daily: pd.DataFrame) -> float:
    net = daily["net"].dropna()
    return float((1 + net).prod() - 1)


# ---------------------------------------------------------------- forecast quality
def daily_ic_fast(df: pd.DataFrame, pred: str, actual: str = "y", min_assets: int = 5) -> pd.Series:
    """regime.metrics.daily_ic (Spearman across assets, per date) without groupby.apply."""
    n = df.groupby("date")[pred].transform("size")
    d = df[n >= min_assets]
    ra, rb = d.groupby("date")[pred].rank(), d.groupby("date")[actual].rank()
    da = ra - ra.groupby(d["date"]).transform("mean")
    db = rb - rb.groupby(d["date"]).transform("mean")
    num = (da * db).groupby(d["date"]).sum()
    den = np.sqrt((da ** 2).groupby(d["date"]).sum() * (db ** 2).groupby(d["date"]).sum())
    return (num / den.replace(0, np.nan)).dropna()


def forecast_metrics(df: pd.DataFrame, pred: str, actual: str = "y") -> Optional[dict]:
    """Same numbers as regime.metrics.evaluate. `df` has columns date, `pred`, `actual`."""
    if df.empty:
        return None
    y, f = df[actual].to_numpy(), df[pred].to_numpy()
    sse, sst = float(np.sum((y - f) ** 2)), float(np.sum((y - y.mean()) ** 2))
    ic = daily_ic_fast(df, pred, actual)
    return {"ic": round(float(ic.mean()), 4) if len(ic) else None, "rmse": round(float(np.sqrt(sse / len(y))), 4),
            "r2": round(1 - sse / sst, 4) if sst > 0 else None, "qlike": round(qlike(y, f), 4), "n": int(len(y))}


def forecast_frame() -> Optional[pd.DataFrame]:
    """One row per out-of-sample forecast: y (realized next-week vol), ours, original, simple."""
    from .regime_service import _preds
    p = _preds()
    ours, orig = p.get("log"), p.get("raw")
    if ours is None:
        return None
    out = ours[["y", "pred", "pred_naive"]].rename(columns={"pred": "ours", "pred_naive": "simple"})
    out["simple"] = out["simple"].clip(lower=VOL_FLOOR)
    if orig is not None:
        out = out.join(orig["pred"].rename("original"), how="left")
    return out.reset_index()


def forecast_quality(fc: pd.DataFrame) -> dict:
    models = [m for m in ("ours", "original", "simple") if m in fc]
    if "original" in fc:
        fc = fc.dropna(subset=["original"])  # score every model on the same rows
    out = {"all": {m: forecast_metrics(fc, m) for m in models}}
    for label, start, end in MARKET_PERIODS.values():
        w = fc[(fc["date"] >= start) & (fc["date"] <= end)]
        out[label] = {m: forecast_metrics(w, m) for m in models}
    return {"models": models, "periods": out}


# ---------------------------------------------------------------- API
def report() -> dict:
    with _lock:
        if "report" in _cache:
            return _cache["report"]
    from .decision_service import frame, held_by_system
    df = frame()
    held = held_by_system(df)
    fc = forecast_frame()
    out = {"costs": cost_table(held), "growth_breakeven_bps": growth_breakeven(held),
           "conventions": portfolio_conventions(df, held),
           "forecast": forecast_quality(fc) if fc is not None else None}
    with _lock:
        _cache["report"] = out
    return out


def main() -> None:
    r = report()
    c = r["costs"]
    print("One-way cost   " + "".join(f"{b:>9} bps" for b in c["bps"]))
    for name, rows in c["systems"].items():
        print(f"{name:<15}" + "".join(f"{x['net_return'] * 100:>+12.0f}%" for x in rows) + "   growth")
        print(f"{'':<15}" + "".join(f"{x['sharpe']:>13.2f}" for x in rows) + "   Sharpe")
    print(f"\nMood rule out-grows buy and hold up to {r['growth_breakeven_bps']} bps one-way.")
    cv = r["conventions"]
    print(f"\nPortfolio conventions ({cv['assets']['A']} assets in A, {cv['assets']['fixed']} in B-D):")
    for key, label in (("A", "published: assets present, daily rebalance free"), ("B", "fixed universe"),
                       ("C", "fixed universe, weights drift (true buy and hold)"), ("D", "fixed universe, rebalancing charged")):
        if not cv[key]:
            print(f"  {key} {label}: not computable (missing days)")
            continue
        for n, v in cv[key].items():
            print(f"  {key} {label:<50} {n:<13} growth {v['net_return'] * 100:+5.0f}%  Sharpe {v['sharpe']:.2f}  "
                  f"vol {v['ann_vol'] * 100:4.1f}%  maxDD {v['max_drawdown'] * 100:5.1f}%")
    if r["forecast"]:
        print("\nForecast quality, all out-of-sample forecasts:")
        for m, v in r["forecast"]["periods"]["all"].items():
            print(f"  {m:<9} IC {v['ic']:.4f}  RMSE {v['rmse']:.4f}  R2 {v['r2']:.4f}  QLIKE {v['qlike']:.4f}  n={v['n']:,}")


if __name__ == "__main__":
    main()
