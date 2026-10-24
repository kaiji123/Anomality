# AdaptiveTrader Dashboard Plan

Follow the agreed contracts in `results/` and render precomputed outputs with demo fallback when missing.

Schemas validated in `src/adaptivetrader/dashboard_io/validators.py`:
- market.parquet: date, asset, open, high, low, close, volume
- regimes.parquet: date, asset, regime in {trending, sideways, high_vol}
- backtest.parquet: date, asset, system, cost_bps, position, gross_return, net_return, equity, drawdown, turnover
- events.parquet: event_id, date, asset, system, event_type in {change_alarm, candidate_evaluated, update_accepted, update_rejected, strategy_switch, drift_alarm}, version_before, version_after, metric_name, metric_before, metric_after, reason
- decisions.parquet: date, asset, system, action in {keep, switch, reduce, cash}, position, active_strategy_or_model, reason, optional prob_up and score_*
- summary.parquet: asset, system_vs, cost_bps, delta_sharpe, ci_low, ci_high, n_assets_favouring_c
- lab.parquet: path_id, system, break_day, detection_delay_days, false_switches_per_year
- runs.csv: run_id, timestamp, git_commit, config_hash, notes

Validation includes required columns, numeric typing, sorted unique keys, no future dates, and enum checks.
