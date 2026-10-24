"""Brain 3 end-to-end: forecasts -> rule -> backtest -> three-system comparison.

Run: python -m backend.run_decision
Outputs: results/decision.json  +  a printed table.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from backend.backtest import backtest
from backend.decision import DecisionConfig, decide

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
RESULTS = ROOT / "results"; RESULTS.mkdir(exist_ok=True)
COST_BPS = 5.0


def load_predictions() -> pd.DataFrame:
    full = MODEL_DIR / "walkforward_predictions_paper_log.csv"
    compact = MODEL_DIR / "serving_paper_log.csv.gz"
    if full.exists():
        df = pd.read_csv(full, parse_dates=["date"])
        seeds = [c for c in df.columns if c.startswith("pred_moe_s")]
        if "pred" not in df.columns:
            df["pred"] = df[seeds].mean(axis=1)
    elif compact.exists():
        df = pd.read_csv(compact, parse_dates=["date"])
    else:
        raise SystemExit(f"No forecasts in {MODEL_DIR}")
    df["pred"] = df["pred"].clip(lower=0.03)
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def attach_returns(df: pd.DataFrame) -> pd.DataFrame:
    cache = ROOT / "data" / "processed" / "prices.parquet"
    if not cache.exists():
        raise SystemExit("Missing data/processed/prices.parquet — run data/download_returns.py first.")
    prices = pd.read_parquet(cache)
    rets = prices.pct_change().stack().rename("ret").reset_index()
    rets.columns = ["date", "symbol", "ret"]
    rets["date"] = pd.to_datetime(rets["date"])
    out = df.merge(rets, on=["date", "symbol"], how="left")
    missing = out["ret"].isna().sum()
    if missing:
        print(f"WARNING: {missing:,} rows have no return (asset not in cache). Dropping them.")
        out = out.dropna(subset=["ret"])
    return out


def build_positions(df: pd.DataFrame, cfg: DecisionConfig) -> pd.DataFrame:
    parts = []
    for sym, g in df.groupby("symbol", sort=False):
        g = g.sort_values("date").copy()
        active, days = 1.0, 0
        expo, reasons = [], []
        for _, r in g.iterrows():
            t, reason = decide(r["pred"], r["rv20"], days, active, cfg)
            if t != active: days = 0; active = t
            else: days += 1
            expo.append(active); reasons.append(reason)
        g["exposure"] = expo; g["reason"] = reasons
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def portfolio(per: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Equal-weight across assets: mean exposure and mean (exposure * ret) per date."""
    pos = per.groupby("date")["exposure"].mean()
    ret = (per["exposure"] * per["ret"]).groupby(per["date"]).mean()
    return pos.sort_index(), ret.sort_index()


def main() -> None:
    df = attach_returns(load_predictions())
    print(f"{len(df):,} rows, {df['symbol'].nunique()} assets, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")

    cfgs = {"adaptive_binary": DecisionConfig(variant="binary"),
            "adaptive_tiered": DecisionConfig(variant="tiered")}
    results = {}
    for name, cfg in cfgs.items():
        per = build_positions(df, cfg)
        pos, ret = portfolio(per)
        results[name] = backtest(pos, ret, cost_bps=COST_BPS)
        results[name]["_per"] = per

    # buy-and-hold: exposure 1.0 always
    bh = df.copy(); bh["exposure"] = 1.0
    pos_bh, ret_bh = portfolio(bh)
    results["buy_and_hold"] = backtest(pos_bh, ret_bh, cost_bps=COST_BPS)

    keys = ["net_return", "ann_return", "ann_vol", "sharpe", "max_drawdown", "turnover", "n_days"]
    print("\n" + f"{'system':<20}" + "".join(f"{k:>12}" for k in keys))
    print("-" * (20 + 12 * len(keys)))
    for name, r in results.items():
        print(f"{name:<20}" + "".join(f"{r.get(k, 0):>12.4f}" for k in keys))

    out = {k: {kk: vv for kk, vv in v.items() if kk != "_per"} for k, v in results.items()}
    # per-day decision log for one sample asset (C uses this for the panel)
    sample_sym = df["symbol"].iloc[0]
    sample = results["adaptive_tiered"]["_per"]
    sample = sample[sample["symbol"] == sample_sym].copy()
    sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
    out["decision_log_sample"] = {
        "symbol": sample_sym,
        "rows": sample[["date", "pred", "rv20", "exposure", "reason"]]
                       .round(4).rename(columns={"pred": "forecast", "rv20": "realized"})
                       .to_dict("records"),
    }
    (RESULTS / "decision.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS / 'decision.json'}")


if __name__ == "__main__":
    main()