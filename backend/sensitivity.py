"""Is the tiered rule robust to its two parameters? Report the RANGE, not the best config.

Run: python -m backend.sensitivity
"""
import pandas as pd
from backend.backtest import backtest
from backend.decision import DecisionConfig
from backend.run_decision import load_predictions, attach_returns, build_positions, portfolio

def main():
    df = attach_returns(load_predictions())
    rows = []
    for stress in [1.8, 2.0, 2.2]:
        for hold in [3, 5, 7]:
            cfg = DecisionConfig(variant="tiered", stress_surprise=stress, min_hold_days=hold)
            per = build_positions(df, cfg)
            pos, ret = portfolio(per)
            b = backtest(pos, ret, cost_bps=5.0)
            rows.append({"stress_surprise": stress, "min_hold_days": hold,
                         "net_return": round(b["net_return"], 4),
                         "max_drawdown": round(b["max_drawdown"], 4),
                         "sharpe": round(b["sharpe"], 4),
                         "turnover": round(b["turnover"], 2)})
    out = pd.DataFrame(rows)
    print("\n" + out.to_string(index=False))
    print(f"\nRange across 9 configs:")
    for k in ["net_return", "max_drawdown", "sharpe", "turnover"]:
        print(f"  {k:<14} min {out[k].min():+.4f}  max {out[k].max():+.4f}  "
              f"(headline config 2.0/5: {out[(out.stress_surprise==2.0) & (out.min_hold_days==5)][k].iloc[0]:+.4f})")

if __name__ == "__main__":
    main()