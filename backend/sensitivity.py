"""Is the mood rule robust to its two settings? Report the RANGE, not the best config.

Run: python -m backend.sensitivity
"""
import pandas as pd

from .backtest import hold, portfolio_returns, stats
from .decision import MoodConfig
from .decision_service import frame, simulate


def main():
    df = frame()
    rows = []
    for stormy in [0.80, 0.85, 0.90, 0.95]:
        for cut in [0.5, 0.0]:
            d = portfolio_returns(hold(simulate(df, MoodConfig(stormy_pct=stormy, cut_to=cut))))
            b = stats(d["net"], d["turnover"])
            rows.append({"stormy_pct": stormy, "cut_to": cut,
                         "net_return": round(b["net_return"], 4),
                         "max_drawdown": round(b["max_drawdown"], 4),
                         "sharpe": round(b["sharpe"], 4),
                         "turnover": round(b["turnover"], 2)})
    out = pd.DataFrame(rows)
    print("\n" + out.to_string(index=False))
    bh = portfolio_returns(hold(simulate(df, None)))
    b = stats(bh["net"], bh["turnover"])
    print(f"\nBuy and hold: net_return {b['net_return']:+.4f}  max_drawdown {b['max_drawdown']:+.4f}  sharpe {b['sharpe']:.4f}")
    headline = out[(out.stormy_pct == 0.90) & (out.cut_to == 0.5)].iloc[0]
    print("\nRange across 8 configs:")
    for k in ["net_return", "max_drawdown", "sharpe", "turnover"]:
        print(f"  {k:<14} min {out[k].min():+.4f}  max {out[k].max():+.4f}  (headline 0.90 / half: {headline[k]:+.4f})")


if __name__ == "__main__":
    main()
