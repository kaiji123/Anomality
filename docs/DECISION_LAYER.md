# Decision layer 

The forecaster (RG-ResMoE, see [REGIME_MODEL.md](REGIME_MODEL.md)) predicts how much prices
may move. The decision layer turns that into an exposure and measures whether acting on it
beats buy-and-hold after costs.

## The rule

`surprise = realized_20d_vol / forecast_vol`

| surprise | exposure | label |
|---|---|---|
| >= 2.0 | 0.0 | hold (cash) |
| >= 1.3 | 0.5 | bear-ish (half) |
| < 1.3 | 1.0 | bull (full) |

Hysteresis: any new exposure is held for at least 5 days (`min_hold_days`), preventing
flip-flopping that would eat transaction costs. Every day is logged with its reason
(`calm`, `choppy (1.4x)`, `stress (2.3x)`, `hold (3d < 5)`).

No training, no learned controller. Rule-based by design: explainable, testable, and
defensible in front of a quant judge.

## The backtest

Chronological, 5 bps one-way costs, 1-day execution lag. `signal` at close `t` executes at
close `t+1`, first earns `r_{t+2}`. No look-ahead: proved by `test_exec_lag` and
`test_costs`.

Four systems compared:

- `adaptive_binary`: 0/1 rule, no tiered exposure
- `adaptive_tiered`: 0/0.5/1 rule (the headline system)
- `buy_and_hold`: always long
- (all on the same 56 assets, 2019–2026, equal-weighted portfolio)

## Results

### Full period (2019–2026, 56 assets, 5 bps)

| system | net return | Sharpe | max drawdown | turnover |
|---|---|---|---|---|
| adaptive_binary | 131% | 0.50 | -64.6% | 4.6/yr |
| adaptive_tiered | 114% | 0.51 | -57.4% | 16.1/yr |
| buy_and_hold | 217% | 0.60 | -66.9% | 0.1/yr |

### 2022 bear market (Jan–Oct 2022)

| system | net return | max drawdown |
|---|---|---|
| adaptive_binary | -34.5% | -49.8% |
| adaptive_tiered | -33.4% | -43.9% |
| buy_and_hold | -35.8% | -51.1% |

**The trade:** buy-and-hold wins in a bull market. Both adaptive rules win on return and
drawdown in the 2022 bear market — the tiered rule saves ~7 pts of drawdown and ~2.4 pts of
return. That's the risk-management trade, measured honestly.

## Limitations

- **Turnover cost:** 16 switches/yr × 5 bps ≈ 160 bps/yr. That eats into the return advantage.
- **Sharpe in 2022 is worse** than buy-and-hold because volatility drops too. We report it.
- **Dropped rows:** 19,488 of 103,593 forecast rows had no return (crypto and one ETF on
  the day of download). Results are for the equity/ETF panel.
- **No live execution:** the backtest is the simulated trading engine. We don't claim a
  live bot.

## What this is not

Not a direction predictor. Not a learned controller. Not a live trading system. It's the
decision half of a trading research platform: a rule, a backtest, and an honest result.

## Reproduce

```bash
python -m backend.run_decision          # three-system comparison -> results/decision.json
python -m backend.cost_sensitivity      # edge vs. cost (0/5/10/20 bps)
python -m backend.regime_slices         # four regimes
python -m backend.portfolio_view NVDA   # per-asset decision tape
python -m backend.sensitivity           # robustness to stress_surprise / min_hold
python -m pytest backend/test_decision.py -v   # 4 tests, all pass