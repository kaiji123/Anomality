import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")


def pytest_configure(config):
    config.addinivalue_line("markers", "network: needs live market data from Yahoo Finance")


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m") and "network" in config.getoption("-m"):
        return
    if not config.getoption("--run-network"):
        skip = pytest.mark.skip(reason="needs network; run with --run-network")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip)


def pytest_addoption(parser):
    parser.addoption("--run-network", action="store_true", help="also run tests that fetch live market data")
