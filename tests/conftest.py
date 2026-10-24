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


@pytest.fixture(autouse=True)
def _block_network(request, monkeypatch):
    """Offline tests must really be offline (CI servers have no price cache and Yahoo may block
    them). Any connection attempt fails loudly unless the test is marked `network`."""
    if "network" in request.keywords:
        return
    import socket

    def guard(*args, **kwargs):
        raise RuntimeError(f"{request.node.name} tried to use the network; mark it @pytest.mark.network")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
