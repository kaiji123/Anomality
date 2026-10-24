"""Edge vs. cost: net return, drawdown, Sharpe of the 3 systems at 0/5/10/20 bps.

Answers the judge's question: "what if costs are higher than 5 bps?"
Run: python -m backend.cost_sensitivity
"""
import pandas as pd
from backend.backtest import backtest
from backend.decision import DecisionConfig
from backend.run_decision import load_predictions, attach_returns, build_positions, portfolio

def main():
    df = attach_returns(load_predictions())
    cfgs = {"adaptive_binary": DecisionConfig(variant="binary"),
            "adaptive_tiered": DecisionConfig(variant="tiered")}
    pers = {name: build_positions(df, cfg) for name, cfg in cfgs.items()}
    bh = df.copy(); bh["exposure"] = 1.0; pers["buy_and_hold"] = bh

    rows = []
    for c in [0, 5, 10, 20]:
        for name, per in pers.items():
            pos, ret = portfolio(per)
            b = backtest(pos, ret, cost_bps=c)
            rows.append({"cost_bps": c, "system": name,
                         "net_return": round(b["net_return"], 4),
                         "max_drawdown": round(b["max_drawdown"], 4),
                         "sharpe": round(b["sharpe"], 4),
                         "turnover": round(b["turnover"], 2)})
    out = pd.DataFrame(rows)
    print("\n" + out.to_string(index=False))

    # edge vs buy-and-hold at each cost level
    print("\nDelta vs buy-and-hold:")
    piv = out.pivot(index="cost_bps", columns="system")
    for name in ["adaptive_binary", "adaptive_tiered"]:
        print(f"\n  {name}:")
        for k in ["net_return", "max_drawdown", "sharpe"]:
            delta = (piv[k][name] - piv[k]["buy_and_hold"]).round(4).to_dict()
            print(f"    {k:<14} " + "  ".join(f"{c:>2}bps: {v:+.4f}" for c, v in delta.items()))

if __name__ == "__main__":
    main()