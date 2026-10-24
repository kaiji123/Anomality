from __future__ import annotations

import numpy as np
import pandas as pd


def generate_demo_data(seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", "2020-04-30")
    assets = ["SPY", "QQQ"]
    systems = ["A", "B", "C", "BH"]
    market, regimes, backtest, decisions, events = [], [], [], [], []
    for asset in assets:
        px = 300.0
        eq = {(s, c): 1.0 for s in systems for c in [0, 10]}
        for d in dates:
            crash = pd.Timestamp("2020-02-20") <= d <= pd.Timestamp("2020-03-20")
            ret = (-0.012 if crash else 0.0015) + rng.normal(0, 0.007)
            op = px
            cl = max(1.0, px * (1 + ret))
            hi = max(op, cl) * 1.002
            lo = min(op, cl) * 0.998
            market.append({"date": d, "asset": asset, "open": op, "high": hi, "low": lo, "close": cl, "volume": float(abs(rng.normal(3e7, 4e6)))})
            reg = "high_vol" if crash else ("sideways" if d < pd.Timestamp("2020-02-20") else "trending")
            regimes.append({"date": d, "asset": asset, "regime": reg})
            for s in systems:
                pos = {"A": 0.8, "B": 1.0, "C": 1.1, "BH": 1.0}[s]
                if crash and s in {"A", "C"}:
                    pos -= 0.4
                gross = pos * ret
                for cost_bps in [0, 10]:
                    net = gross - (cost_bps / 10000.0) * 0.2
                    eq[(s, cost_bps)] *= 1 + net
                    backtest.append({"date": d, "asset": asset, "system": s, "cost_bps": cost_bps, "position": pos, "gross_return": gross, "net_return": net, "equity": eq[(s, cost_bps)], "drawdown": eq[(s, cost_bps)] / max(eq[(s, cost_bps)], 1.0) - 1, "turnover": abs(net) * 8})
            decisions.append({"date": d, "asset": asset, "system": "C", "action": "reduce" if crash else "keep", "position": 0.6 if crash else 1.1, "active_strategy_or_model": "C_online" if crash else "C_full", "reason": f"Regime={reg}", "prob_up": float(np.clip(0.55 + rng.normal(0, 0.07), 0, 1))})
            px = cl
        events.extend([
            {"event_id": f"{asset}-1", "date": pd.Timestamp("2020-02-24"), "asset": asset, "system": "C", "event_type": "change_alarm", "version_before": "v1", "version_after": "v1", "metric_name": "sharpe", "metric_before": 0.6, "metric_after": 0.5, "reason": "vol spike"},
            {"event_id": f"{asset}-2", "date": pd.Timestamp("2020-03-10"), "asset": asset, "system": "C", "event_type": "update_accepted", "version_before": "v1", "version_after": "v2", "metric_name": "sharpe", "metric_before": 0.5, "metric_after": 0.8, "reason": "candidate approved"},
        ])

    summary = pd.DataFrame([
        {"asset": "SPY", "system_vs": "C-A", "cost_bps": 0, "delta_sharpe": 0.11, "ci_low": 0.04, "ci_high": 0.17, "n_assets_favouring_c": 2},
        {"asset": "QQQ", "system_vs": "C-A", "cost_bps": 0, "delta_sharpe": -0.01, "ci_low": -0.07, "ci_high": 0.05, "n_assets_favouring_c": 2},
        {"asset": "POOLED", "system_vs": "C-A", "cost_bps": 0, "delta_sharpe": 0.07, "ci_low": 0.02, "ci_high": 0.10, "n_assets_favouring_c": 2},
        {"asset": "SPY", "system_vs": "C-A", "cost_bps": 10, "delta_sharpe": 0.08, "ci_low": 0.02, "ci_high": 0.13, "n_assets_favouring_c": 2},
        {"asset": "QQQ", "system_vs": "C-A", "cost_bps": 10, "delta_sharpe": -0.03, "ci_low": -0.09, "ci_high": 0.02, "n_assets_favouring_c": 2},
        {"asset": "POOLED", "system_vs": "C-A", "cost_bps": 10, "delta_sharpe": 0.05, "ci_low": 0.01, "ci_high": 0.08, "n_assets_favouring_c": 2},
    ])
    lab = pd.DataFrame([
        {"path_id": 1, "system": "A", "break_day": pd.Timestamp("2020-03-02"), "detection_delay_days": 12.0, "false_switches_per_year": 1.6},
        {"path_id": 1, "system": "B", "break_day": pd.Timestamp("2020-03-02"), "detection_delay_days": 16.0, "false_switches_per_year": 1.2},
        {"path_id": 1, "system": "C", "break_day": pd.Timestamp("2020-03-02"), "detection_delay_days": 7.0, "false_switches_per_year": 0.9},
    ])
    runs = pd.DataFrame([{"run_id": "demo-001", "timestamp": pd.Timestamp("2026-10-24"), "git_commit": "demo000", "config_hash": "demo", "notes": "demo"}])

    return {
        "market": pd.DataFrame(market).sort_values(["date", "asset"]).reset_index(drop=True),
        "regimes": pd.DataFrame(regimes).sort_values(["date", "asset"]).reset_index(drop=True),
        "backtest": pd.DataFrame(backtest).sort_values(["date", "asset", "system", "cost_bps"]).reset_index(drop=True),
        "events": pd.DataFrame(events).sort_values(["date", "event_id"]).reset_index(drop=True),
        "decisions": pd.DataFrame(decisions).sort_values(["date", "asset", "system"]).reset_index(drop=True),
        "summary": summary.sort_values(["asset", "system_vs", "cost_bps"]).reset_index(drop=True),
        "lab": lab.sort_values(["path_id", "system"]).reset_index(drop=True),
        "runs": runs,
    }
