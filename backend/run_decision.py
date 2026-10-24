"""Decision layer end-to-end: forecasts -> rule -> backtest -> comparison (mood rule, buy-and-hold, previous rule).

Run: python -m backend.run_decision
Outputs: results/decision.json and a printed table.
"""
from __future__ import annotations

import json

from regime.config import ROOT
from regime.fileio import atomic_path

from .decision_service import RULE, frame, simulate, summarize

RESULTS = ROOT / "results"
KEYS = ["net_return", "ann_return", "ann_vol", "sharpe", "max_drawdown", "turnover", "n_days"]


def print_table(systems: dict, period: str = "all") -> None:
    print(f"\n{'system':<20}" + "".join(f"{k:>13}" for k in KEYS))
    print("-" * (20 + 13 * len(KEYS)))
    for name, s in systems.items():
        print(f"{name:<20}" + "".join(f"{s[period][k]:>13.4f}" for k in KEYS))


def main() -> None:
    df = frame()
    print(f"{len(df):,} rows, {df['symbol'].nunique()} assets, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    s = summarize(df)
    print_table(s["systems"])

    out = {name: sys["all"] for name, sys in s["systems"].items()}  # top-level keys kept for older readers
    out.update({k: v for k, v in s.items() if not k.startswith("_")})
    sample_sym = df["symbol"].iloc[0]
    sim = simulate(df, RULE)  # the mood rule needs the whole market, so simulate everything first
    tape = sim[sim["symbol"] == sample_sym].copy()
    tape["date"] = tape["date"].dt.strftime("%Y-%m-%d")
    out["decision_log_sample"] = {
        "symbol": sample_sym,
        "rows": tape[["date", "pred", "storm", "exposure", "reason"]].round(4)
        .rename(columns={"pred": "forecast", "storm": "market_storminess"}).to_dict("records"),
    }
    RESULTS.mkdir(exist_ok=True)
    with atomic_path(RESULTS / "decision.json") as tmp:
        tmp.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS / 'decision.json'}")


if __name__ == "__main__":
    main()
