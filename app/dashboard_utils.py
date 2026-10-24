from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

SYSTEM_COLORS = {"A": "#8a8a8a", "B": "#2f6de1", "C": "#7b3fe4", "BH": "#c7c7c7"}
REGIME_COLORS = {"trending": "rgba(76, 175, 80, 0.18)", "sideways": "rgba(120, 120, 120, 0.15)", "high_vol": "rgba(255, 179, 71, 0.20)"}


def truncate_at_date(df: pd.DataFrame | None, date_col: str, cutoff: pd.Timestamp) -> pd.DataFrame | None:
    if df is None:
        return None
    return df[pd.to_datetime(df[date_col]) <= pd.Timestamp(cutoff)].copy()


def regime_intervals(df: pd.DataFrame) -> list[dict[str, object]]:
    if df.empty:
        return []
    d = df.sort_values("date")[["date", "regime"]].reset_index(drop=True)
    res = []
    start, reg, prev = d.loc[0, "date"], d.loc[0, "regime"], d.loc[0, "date"]
    for _, row in d.iloc[1:].iterrows():
        if row["regime"] != reg:
            res.append({"start": start, "end": prev, "regime": reg})
            start, reg = row["date"], row["regime"]
        prev = row["date"]
    res.append({"start": start, "end": prev, "regime": reg})
    return res


def verdict(ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return "supported"
    if ci_high < 0:
        return "contradicted"
    return "inconclusive"


def demo_mode_from_data(data: dict[str, pd.DataFrame | None], required: Iterable[str]) -> bool:
    return any(data.get(k) is None for k in required)


def compute_metrics(backtest_df: pd.DataFrame, decisions_df: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for s, p in backtest_df.groupby("system"):
        p = p.sort_values("date")
        n = pd.to_numeric(p["net_return"], errors="coerce").fillna(0)
        e = pd.to_numeric(p["equity"], errors="coerce")
        years = len(e) / 252 if len(e) else 0
        annual = float((e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1) if len(e) > 1 and years > 0 else np.nan
        vol = float(n.std(ddof=0) * math.sqrt(252))
        sharpe = float(n.mean() / (n.std(ddof=0) + 1e-12) * math.sqrt(252))
        mdd = float(pd.to_numeric(p["drawdown"], errors="coerce").min())
        turnover = float(pd.to_numeric(p["turnover"], errors="coerce").mean())
        switches = np.nan
        if decisions_df is not None and not decisions_df.empty:
            d = decisions_df[decisions_df["system"] == s]
            if not d.empty:
                years2 = max(len(d["date"].unique()) / 252, 1 / 252)
                switches = float((d["action"] == "switch").sum() / years2)
        rows.append({"system": s, "net_annual_return": annual, "volatility": vol, "sharpe": sharpe, "max_drawdown": mdd, "turnover": turnover, "switches_per_year": switches})
    return pd.DataFrame(rows)


def has_run_system() -> bool:
    try:
        from adaptivetrader import run_system  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False
