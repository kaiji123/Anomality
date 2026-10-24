"""Three systems across four market phases. Shows when the rule helps and when it hurts.

Run: python -m backend.regime_slices
"""
from regime.config import MARKET_PERIODS

from .decision_service import frame, summarize


def main():
    s = summarize(frame())
    for label, start, end in MARKET_PERIODS.values():
        print(f"\n{label} ({start} -> {end}), {s['cost_bps']:g} bps costs")
        print(f"{'system':<20}{'net_return':>12}{'max_drawdown':>14}{'sharpe':>10}{'turnover':>10}")
        print("-" * 66)
        for name, sys in s["systems"].items():
            b = sys[label]
            print(f"{name:<20}{b['net_return']:>12.4f}{b['max_drawdown']:>14.4f}{b['sharpe']:>10.4f}{b['turnover']:>10.2f}")


if __name__ == "__main__":
    main()
