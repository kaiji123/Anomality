"""Edge vs. cost: net return, drawdown and Sharpe of the 3 systems at 0/5/10/20 bps.

Answers the judge's question: "what if costs are higher than 5 bps?"
Run: python -m backend.cost_sensitivity
"""
import pandas as pd

from .backtest import portfolio_returns, stats
from .decision_service import frame, held_by_system


def main():
    held = held_by_system(frame())
    rows = []
    for c in [0, 5, 10, 20]:
        for name, h in held.items():
            d = portfolio_returns(h, cost_bps=c)
            b = stats(d["net"], d["turnover"])
            rows.append({"cost_bps": c, "system": name,
                         "net_return": round(b["net_return"], 4),
                         "max_drawdown": round(b["max_drawdown"], 4),
                         "sharpe": round(b["sharpe"], 4),
                         "turnover": round(b["turnover"], 2)})
    out = pd.DataFrame(rows)
    print("\n" + out.to_string(index=False))

    print("\nDelta vs buy-and-hold:")
    piv = out.pivot(index="cost_bps", columns="system")
    for name in [n for n in held if n != "buy_and_hold"]:
        print(f"\n  {name}:")
        for k in ["net_return", "max_drawdown", "sharpe"]:
            delta = (piv[k][name] - piv[k]["buy_and_hold"]).round(4).to_dict()
            print(f"    {k:<14} " + "  ".join(f"{c:>2}bps: {v:+.4f}" for c, v in delta.items()))


if __name__ == "__main__":
    main()
