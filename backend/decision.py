"""Brain 3: risk-targeting decision rule with hysteresis.

Maps RG-ResMoE's volatility forecast + realized vol to an exposure in {0, 0.5, 1}.
Rule-based by design: no training, fully explainable, logged per day.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DecisionConfig:
    stress_surprise: float = 2.0    # realized/forecast >= this -> cash
    choppy_surprise: float = 1.3    # realized/forecast >= this -> half exposure
    min_hold_days: int = 5          # hysteresis: hold new exposure >= N days
    variant: str = "tiered"         # "binary" (0/1) or "tiered" (0/0.5/1)


def target_for(surprise: float, cfg: DecisionConfig) -> tuple[float, str]:
    if surprise >= cfg.stress_surprise:
        return 0.0, f"stress ({surprise:.2f}x)"
    if cfg.variant == "tiered" and surprise >= cfg.choppy_surprise:
        return 0.5, f"choppy ({surprise:.2f}x)"
    return 1.0, "calm/normal"


def decide(forecast: float, realized_20d: float, days_since_switch: int,
           active_exposure: float, cfg: DecisionConfig) -> tuple[float, str]:
    """Returns (exposure, reason). Hysteresis prevents flip-flopping."""
    surprise = realized_20d / max(forecast, 1e-6)
    target, reason = target_for(surprise, cfg)
    if target != active_exposure and days_since_switch < cfg.min_hold_days:
        return active_exposure, f"hold ({days_since_switch}d < {cfg.min_hold_days})"
    return target, reason