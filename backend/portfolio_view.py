"""The rule's decision tape: what it would have done, day by day, through today.

Run: python -m backend.portfolio_view [SYMBOL]
"""
import sys

from .decision_service import DecisionUnavailable, plain_asset, plain_portfolio


def main(symbol: str = "NVDA"):
    try:
        p = plain_portfolio()
        a = plain_asset(symbol)
    except DecisionUnavailable as e:
        raise SystemExit(str(e))
    m = p["market"]
    print("\nMood rule: market storminess and position, last 30 days")
    print(f"{'date':<12}{'storminess':>12}{'position':>10}")
    print("-" * 34)
    for d, s, e in list(zip(m["date"], m["storm"], m["exposure"]))[-30:]:
        print(f"{d:<12}{(s if s is not None else float('nan')):>12.0%}{e:>10.1f}")
    print(f"\nToday ({a['date']}): {a['level']} for every asset, since {a['since']}. {a['text']}")
    print(f"Half invested on {m['time_half']:.0%} of days; {len(m['switches'])} switches since {p['since']}.")
    r, b = a["record"]["mood_rule"]["all"], a["record"]["buy_and_hold"]["all"]
    print(f"{a['symbol']}: rule {r['net_return']:+.0%} (worst drop {r['max_drawdown']:.0%}) "
          f"vs. buy and hold {b['net_return']:+.0%} ({b['max_drawdown']:.0%})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "NVDA")
