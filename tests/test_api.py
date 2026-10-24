"""End-to-end API checks against live market data. Run with: pytest --run-network"""
import pytest
from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)
pytestmark = pytest.mark.network


def test_anomaly_report_has_the_numbers_the_banner_shows():
    r = client.get("/api/regime/market")
    assert r.status_code == 200
    body = r.json()
    assert body["mood"]["name"] in {"Calm", "Normal", "Choppy", "Stormy"}
    assert len(body["mood"]["scale"]["cuts"]) == 3 and len(body["mood"]["levels"]) == 4
    assert body["stock_move"] > 0 and body["track"]["periods"] == 30


def test_asset_forecast_and_unknown_asset():
    r = client.get("/api/regime/NVDA")
    assert r.status_code == 200 and r.json()["move"] > 0 and r.json()["history"]["events"]
    assert client.get("/api/regime/GME").status_code == 404


def test_classic_analysis_still_works():
    r = client.get("/api/analyze/AAPL")
    assert r.status_code == 200 and r.json()["summary"]["verdict"] in {"HIGH", "ELEVATED", "NORMAL"}
