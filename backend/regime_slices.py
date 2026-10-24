"""Three systems across four market regimes. Shows when the rule helps and when it hurts.

Run: python -m backend.regime_slices
"""
import pandas as pd
from backend.backtest import backtest
from backend.decision import DecisionConfig
from backend.run_decision import load_predictions, attach_returns, build_positions, portfolio

REGIMES = {
    "COVID crash (Feb–Apr 2020)": ("2020-02-19", "2020-04-30"),
    "Recovery (May 2020 – Dec 2021)": ("2020-05-01", "2021-12-31"),
    "2022 bear market": ("2022-01-01", "2022-10-31"),
    "Calm (2023 – 2024)": ("2023-01-01", "2024-12-31"),
}

def main():
    df = attach_returns(load_predictions())
    cfgs = {"adaptive_binary": DecisionConfig(variant="binary"),
            "adaptive_tiered": DecisionConfig(variant="tiered")}
    pers = {name: build_positions(df, cfg) for name, cfg in cfgs.items()}
    bh = df.copy(); bh["exposure"] = 1.0; pers["buy_and_hold"] = bh

    for regime, (start, end) in REGIMES.items():
        print(f"\n{regime} ({start} -> {end}), 5 bps costs")
        print(f"{'system':<20}{'net_return':>12}{'max_drawdown':>14}{'sharpe':>10}{'turnover':>10}")
        print("-" * 66)
        for name, per in pers.items():
            sub = per[(per["date"] >= start) & (per["date"] <= end)]
            if sub.empty:
                print(f"{name:<20}  (no data)"); continue
            pos, ret = portfolio(sub)
            b = backtest(pos, ret, cost_bps=5.0)
            print(f"{name:<20}{b['net_return']:>12.4f}{b['max_drawdown']:>14.4f}{b['sharpe']:>10.4f}{b['turnover']:>10.2f}")

if __name__ == "__main__":
    main()