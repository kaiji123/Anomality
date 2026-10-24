from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import pandas as pd


@dataclass
class DashboardDataError(ValueError):
    file_name: str
    message: str

    def __str__(self) -> str:
        return f"{self.file_name}: {self.message}"


def _require_columns(df: pd.DataFrame, file_name: str, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DashboardDataError(file_name, f"missing required columns: {', '.join(missing)}")


def _parse_date_column(df: pd.DataFrame, file_name: str, col: str, nullable: bool = False) -> pd.DataFrame:
    parsed = pd.to_datetime(df[col], errors="coerce")
    if (parsed.isna() & ~df[col].isna()).any() or ((~nullable) and parsed.isna().any()):
        raise DashboardDataError(file_name, f"column '{col}' has invalid date values")
    if parsed.notna().any() and parsed.dropna().max() > pd.Timestamp(datetime.utcnow().date()):
        raise DashboardDataError(file_name, f"column '{col}' contains future dates")
    out = df.copy()
    out[col] = parsed.dt.normalize()
    return out


def _require_numeric(df: pd.DataFrame, file_name: str, cols: Iterable[str], nullable: bool = False) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        num = pd.to_numeric(df[c], errors="coerce")
        bad = (num.isna() & ~df[c].isna()) if nullable else num.isna()
        if bad.any():
            raise DashboardDataError(file_name, f"column '{c}' must be numeric")


def _require_enum(df: pd.DataFrame, file_name: str, col: str, allowed: set[str]) -> None:
    vals = set(df[col].dropna().astype(str).unique())
    invalid = sorted(vals - allowed)
    if invalid:
        raise DashboardDataError(file_name, f"column '{col}' has invalid values: {', '.join(invalid)}")


def _require_sorted_unique(df: pd.DataFrame, file_name: str, keys: list[str], sort_cols: list[str] | None = None) -> None:
    if df.duplicated(keys).any():
        raise DashboardDataError(file_name, f"duplicate key rows for {keys}")
    check = df.sort_values(sort_cols or keys).reset_index(drop=True)
    if not df.reset_index(drop=True).equals(check):
        raise DashboardDataError(file_name, f"rows must be sorted by {sort_cols or keys}")


def validate_market(df: pd.DataFrame) -> pd.DataFrame:
    fn = "market.parquet"
    _require_columns(df, fn, ["date", "asset", "open", "high", "low", "close", "volume"])
    out = _parse_date_column(df, fn, "date")
    _require_numeric(out, fn, ["open", "high", "low", "close", "volume"])
    _require_sorted_unique(out, fn, ["date", "asset"])
    return out


def validate_regimes(df: pd.DataFrame) -> pd.DataFrame:
    fn = "regimes.parquet"
    _require_columns(df, fn, ["date", "asset", "regime"])
    out = _parse_date_column(df, fn, "date")
    _require_enum(out, fn, "regime", {"trending", "sideways", "high_vol"})
    _require_sorted_unique(out, fn, ["date", "asset"])
    return out


def validate_backtest(df: pd.DataFrame) -> pd.DataFrame:
    fn = "backtest.parquet"
    _require_columns(df, fn, ["date", "asset", "system", "cost_bps", "position", "gross_return", "net_return", "equity", "drawdown", "turnover"])
    out = _parse_date_column(df, fn, "date")
    _require_numeric(out, fn, ["cost_bps", "position", "gross_return", "net_return", "equity", "drawdown", "turnover"])
    _require_sorted_unique(out, fn, ["date", "asset", "system", "cost_bps"])
    return out


def validate_events(df: pd.DataFrame) -> pd.DataFrame:
    fn = "events.parquet"
    _require_columns(df, fn, ["event_id", "date", "asset", "system", "event_type", "version_before", "version_after", "metric_name", "metric_before", "metric_after", "reason"])
    out = _parse_date_column(df, fn, "date")
    _require_enum(out, fn, "event_type", {"change_alarm", "candidate_evaluated", "update_accepted", "update_rejected", "strategy_switch", "drift_alarm"})
    _require_numeric(out, fn, ["metric_before", "metric_after"], nullable=True)
    if out["event_id"].duplicated().any():
        raise DashboardDataError(fn, "column 'event_id' must be unique")
    _require_sorted_unique(out, fn, ["event_id"], ["date", "event_id"])
    return out


def validate_decisions(df: pd.DataFrame) -> pd.DataFrame:
    fn = "decisions.parquet"
    _require_columns(df, fn, ["date", "asset", "system", "action", "position", "active_strategy_or_model", "reason"])
    out = _parse_date_column(df, fn, "date")
    _require_enum(out, fn, "action", {"keep", "switch", "reduce", "cash"})
    _require_numeric(out, fn, ["position"])
    if "prob_up" in out.columns:
        _require_numeric(out, fn, ["prob_up"], nullable=True)
        p = pd.to_numeric(out["prob_up"], errors="coerce")
        if ((p < 0) | (p > 1)).fillna(False).any():
            raise DashboardDataError(fn, "column 'prob_up' must be within [0, 1]")
    score_cols = [c for c in out.columns if c.startswith("score_")]
    _require_numeric(out, fn, score_cols, nullable=True)
    _require_sorted_unique(out, fn, ["date", "asset", "system"])
    return out


def validate_summary(df: pd.DataFrame) -> pd.DataFrame:
    fn = "summary.parquet"
    _require_columns(df, fn, ["asset", "system_vs", "cost_bps", "delta_sharpe", "ci_low", "ci_high", "n_assets_favouring_c"])
    _require_numeric(df, fn, ["cost_bps", "delta_sharpe", "ci_low", "ci_high", "n_assets_favouring_c"])
    _require_sorted_unique(df, fn, ["asset", "system_vs", "cost_bps"])
    return df


def validate_lab(df: pd.DataFrame) -> pd.DataFrame:
    fn = "lab.parquet"
    _require_columns(df, fn, ["path_id", "system", "break_day", "detection_delay_days", "false_switches_per_year"])
    out = _parse_date_column(df, fn, "break_day", nullable=True)
    _require_numeric(out, fn, ["detection_delay_days", "false_switches_per_year"])
    _require_sorted_unique(out, fn, ["path_id", "system"])
    return out


def validate_runs(df: pd.DataFrame) -> pd.DataFrame:
    fn = "runs.csv"
    _require_columns(df, fn, ["run_id", "timestamp", "git_commit", "config_hash", "notes"])
    out = _parse_date_column(df, fn, "timestamp")
    _require_sorted_unique(out, fn, ["run_id"], ["timestamp", "run_id"])
    return out
