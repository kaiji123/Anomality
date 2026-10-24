"""Decision layer: turns the RG-ResMoE forecast into a position (exposure) for every asset.

Two rules, both rule-based by design (no training, fully explainable, logged per day):

- Mood rule (the one the app uses): hold every asset at half when the regime model's market
  mood is Stormy, i.e. the market-wide forecast is busier than 90% of all earlier days.
  Fully invested otherwise.
- Surprise rule (the previous rule, kept for comparison): compares the last 20 days' realized
  volatility with the forecast and cuts exposure when markets are wilder than expected.

See docs/DECISION_LAYER.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

COST_BPS = 5.0  # one-way trading cost per unit of exposure changed
EXEC_LAG = 1  # a decision at close t is traded at close t+1, so it first earns day t+2's return
LEVELS = {1.0: "Full", 0.5: "Half", 0.0: "Cash"}


# ---------------------------------------------------------------- mood rule (headline)
@dataclass
class MoodConfig:
    stormy_pct: float = 0.90  # where the app's Stormy mood starts (backend/regime_service.py MOODS)
    cut_to: float = 0.5       # exposure in Stormy weeks
    min_history: int = 60     # trading days of forecasts needed before the rule can call a storm


def storminess(market_forecast: np.ndarray, min_history: int) -> np.ndarray:
    """For each day, the share of *earlier* days with a lower market forecast (0 to 1).

    Uses only the past, so it can be acted on that day. NaN until `min_history` days exist."""
    v = np.asarray(market_forecast, dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(min_history, len(v)):
        out[i] = float((v[:i] < v[i]).mean())
    return out


def mood_positions(pct: np.ndarray, cfg: MoodConfig) -> Tuple[np.ndarray, List[str]]:
    """Exposure and reason for each day, from the market's storminess."""
    exposure, reasons = np.ones(len(pct)), []
    for i, p in enumerate(pct):
        if np.isnan(p):
            reasons.append("warming up")
        elif p >= cfg.stormy_pct:
            exposure[i] = cfg.cut_to
            reasons.append(f"stormy ({p:.0%})")
        else:
            reasons.append(f"not stormy ({p:.0%})")
    return exposure, reasons


# ---------------------------------------------------------------- surprise rule (previous, comparison)
@dataclass
class DecisionConfig:
    stress_surprise: float = 2.0    # realized/forecast >= this -> cash
    choppy_surprise: float = 1.3    # realized/forecast >= this -> half exposure
    min_hold_days: int = 5          # hysteresis: no new switch for this many days after one
    variant: str = "tiered"         # "binary" (0/1) or "tiered" (0/0.5/1)


def target_for(surprise: float, cfg: DecisionConfig) -> Tuple[float, str]:
    if surprise >= cfg.stress_surprise:
        return 0.0, f"stress ({surprise:.2f}x)"
    if cfg.variant == "tiered" and surprise >= cfg.choppy_surprise:
        return 0.5, f"choppy ({surprise:.2f}x)"
    return 1.0, "calm/normal"


def decide(forecast: float, realized_20d: float, days_since_switch: int,
           active_exposure: float, cfg: DecisionConfig) -> Tuple[float, str]:
    """Returns (exposure, reason). Hysteresis prevents flip-flopping."""
    surprise = realized_20d / max(forecast, 1e-6)
    target, reason = target_for(surprise, cfg)
    if target != active_exposure and days_since_switch < cfg.min_hold_days:
        return active_exposure, f"hold ({days_since_switch}d < {cfg.min_hold_days})"
    return target, reason


def positions(forecast: np.ndarray, realized_20d: np.ndarray,
              cfg: DecisionConfig) -> Tuple[np.ndarray, List[str]]:
    """Run the surprise rule over one asset's days in date order, starting fully invested."""
    active, days = 1.0, 0
    exposure, reasons = np.empty(len(forecast)), []
    for i, (f, rv) in enumerate(zip(forecast, realized_20d)):
        target, reason = decide(float(f), float(rv), days, active, cfg)
        if target != active:
            active, days = target, 0
        else:
            days += 1
        exposure[i] = active
        reasons.append(reason)
    return exposure, reasons
