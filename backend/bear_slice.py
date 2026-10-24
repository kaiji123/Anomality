"""Print the 3 systems on the 2022 bear-market window (Jan–Oct 2022)."""
import pandas as pd
from backend.backtest import backtest
from backend.decision import DecisionConfig
from backend.run_decision import load_predictions, attach_returns, build_positions, portfolio

df = attach_returns(load_predictions())
cfgs = {"adaptive_binary": DecisionConfig(variant="binary"),
        "adaptive_tiered": DecisionConfig(variant="tiered")}

per_all = {}
for name, cfg in cfgs.items():
    per_all[name] = build_positions(df, cfg)
bh = df.copy(); bh["exposure"] = 1.0
per_all["buy_and_hold"] = bh

print("\n2022 bear market (2022-01-01 → 2022-10-31), 5 bps costs")
print(f"{'system':<20}{'net_return':>12}{'max_drawdown':>14}{'sharpe':>10}{'turnover':>10}")
print("-" * 66)
for name, per in per_all.items():
    per = per[(per["date"] >= "2022-01-01") & (per["date"] <= "2022-10-31")]
    pos, ret = portfolio(per)
    b = backtest(pos, ret, cost_bps=5.0)
    print(f"{name:<20}{b['net_return']:>12.4f}{b['max_drawdown']:>14.4f}{b['sharpe']:>10.4f}{b['turnover']:>10.2f}")