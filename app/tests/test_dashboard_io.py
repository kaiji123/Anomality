from pathlib import Path

import pandas as pd
import pytest

from app.dashboard_utils import demo_mode_from_data
from src.adaptivetrader.dashboard_io import DashboardDataError
from src.adaptivetrader.dashboard_io import loaders


@pytest.mark.parametrize("loader_name,file_name,is_csv", [
    ("load_market", "market.parquet", False),
    ("load_regimes", "regimes.parquet", False),
    ("load_backtest", "backtest.parquet", False),
    ("load_events", "events.parquet", False),
    ("load_decisions", "decisions.parquet", False),
    ("load_summary", "summary.parquet", False),
    ("load_lab", "lab.parquet", False),
    ("load_runs", "runs.csv", True),
])
def test_malformed_loader_rejected(monkeypatch, tmp_path: Path, loader_name: str, file_name: str, is_csv: bool):
    (tmp_path / file_name).write_text("x")

    monkeypatch.setattr(pd, "read_parquet", lambda _p: pd.DataFrame({"bad": [1]}))
    monkeypatch.setattr(pd, "read_csv", lambda _p: pd.DataFrame({"bad": [1]}))

    loader = getattr(loaders, loader_name)
    if hasattr(loader, "clear"):
        loader.clear()
    with pytest.raises(DashboardDataError):
        loader(tmp_path)


def test_missing_files_demo_mode(tmp_path: Path):
    if hasattr(loaders.load_market, "clear"):
        loaders.load_market.clear()
    if hasattr(loaders.load_regimes, "clear"):
        loaders.load_regimes.clear()
    data = {"market": loaders.load_market(tmp_path), "regimes": loaders.load_regimes(tmp_path)}
    assert data["market"] is None and data["regimes"] is None
    assert demo_mode_from_data(data, ["market", "regimes"]) is True
