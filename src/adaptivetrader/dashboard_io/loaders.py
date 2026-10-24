from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import pandas as pd

from .validators import (
    DashboardDataError,
    validate_backtest,
    validate_decisions,
    validate_events,
    validate_lab,
    validate_market,
    validate_regimes,
    validate_runs,
    validate_summary,
)

try:
    import streamlit as st
    cache_data = st.cache_data
except Exception:  # pragma: no cover
    def cache_data(func: Callable):
        return lru_cache(maxsize=16)(func)


def _results(results_dir: str | Path | None = None) -> Path:
    return Path(results_dir) if results_dir is not None else Path(__file__).resolve().parents[3] / "results"


def _load_parquet(path: Path, validator: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return validator(pd.read_parquet(path))
    except DashboardDataError:
        raise
    except Exception as exc:
        raise DashboardDataError(path.name, f"cannot read file ({exc})") from exc


def _load_csv(path: Path, validator: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return validator(pd.read_csv(path))
    except DashboardDataError:
        raise
    except Exception as exc:
        raise DashboardDataError(path.name, f"cannot read file ({exc})") from exc


@cache_data
def load_market(results_dir: str | Path | None = None):
    return _load_parquet(_results(results_dir) / "market.parquet", validate_market)


@cache_data
def load_regimes(results_dir: str | Path | None = None):
    return _load_parquet(_results(results_dir) / "regimes.parquet", validate_regimes)


@cache_data
def load_backtest(results_dir: str | Path | None = None):
    return _load_parquet(_results(results_dir) / "backtest.parquet", validate_backtest)


@cache_data
def load_events(results_dir: str | Path | None = None):
    return _load_parquet(_results(results_dir) / "events.parquet", validate_events)


@cache_data
def load_decisions(results_dir: str | Path | None = None):
    return _load_parquet(_results(results_dir) / "decisions.parquet", validate_decisions)


@cache_data
def load_summary(results_dir: str | Path | None = None):
    return _load_parquet(_results(results_dir) / "summary.parquet", validate_summary)


@cache_data
def load_lab(results_dir: str | Path | None = None):
    return _load_parquet(_results(results_dir) / "lab.parquet", validate_lab)


@cache_data
def load_runs(results_dir: str | Path | None = None):
    return _load_csv(_results(results_dir) / "runs.csv", validate_runs)
