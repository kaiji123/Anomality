"""Per-asset decision tape: what the rule would have done, day by day.

Run: python -m backend.portfolio_view NVDA
"""
import sys
import pandas as pd
from backend.decision import DecisionConfig
from backend.run_decision import load_predictions, attach_returns, build_positions

def main(symbol: str = "NVDA"):
    df = attach_returns(load_predictions())
    if symbol not in df["symbol"].unique():
        raise SystemExit(f"{symbol} not in the forecast file. Available: {sorted(df['symbol'].unique())[:10]}...")
    per = build_positions(df, DecisionConfig(variant="tiered"))
    g = per[per["symbol"] == symbol].sort_values("date").copy()
    g["surprise"] = g["rv20"] / g["pred"]
    g["date"] = g["date"].dt.strftime("%Y-%m-%d")

    print(f"\n{symbol}: decision tape (tiered rule, 5 bps costs)")
    print(f"{'date':<12}{'forecast':>10}{'realized':>10}{'surprise':>10}{'exposure':>10}  reason")
    print("-" * 78)
    for _, r in g.tail(30).iterrows():
        print(f"{r['date']:<12}{r['pred']:>10.4f}{r['rv20']:>10.4f}{r['surprise']:>10.2f}{r['exposure']:>10.1f}  {r['reason']}")

    last = g.iloc[-1]
    print(f"\nToday ({last['date']}): position = {last['exposure']:.1f}, reason = {last['reason']}")
    print(f"Full history: {len(g)} days, {(g['exposure'] == 0).sum()} in cash, "
          f"{(g['exposure'] == 0.5).sum()} half, {(g['exposure'] == 1).sum()} full")

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    main(symbol)