"""Settings for the RG-ResMoE regime model (see docs/REGIME_MODEL.md)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
CACHE_DIR = MODEL_DIR / "cache"

# Large, liquid U.S. stocks across sectors, plus broad ETFs.
STOCKS: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "NFLX", "INTC",
    "ORCL", "CRM", "ADBE", "CSCO", "QCOM", "JPM", "BAC", "GS", "MS", "WFC",
    "V", "MA", "BRK-B", "UNH", "JNJ", "PFE", "MRK", "LLY", "ABBV", "XOM",
    "CVX", "COP", "WMT", "COST", "HD", "MCD", "NKE", "KO", "PEP", "DIS",
    "BA", "CAT", "GE", "UPS", "T", "SPY", "QQQ", "IWM",
]
CRYPTO: List[str] = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "AVAX-USD", "LINK-USD"]

# Paper inputs: 16 stock features (x) and 2 regime variables (z).
X_COLS: List[str] = ["rv5", "rv20", "rv60", "cum5", "cum20", "rsi14"] + [f"lag{k}" for k in range(10)]
Z_COLS: List[str] = ["mkt_vol", "idio_vol"]
TARGET = "y"
# Volatility inputs that can be log-transformed before scaling (keeps crisis spikes in range).
LOG_COLS: List[str] = ["rv5", "rv20", "rv60", "mkt_vol", "idio_vol"]


@dataclass
class TrainConfig:
    # Data
    period: str = "10y"
    include_crypto: bool = True
    horizon: int = 5  # target = realized vol over the next `horizon` days
    # Walk-forward windows (paper: 504-day development window, 85/15 split, 63-day test, step 63)
    dev_days: int = 504
    val_frac: float = 0.15
    test_days: int = 63
    step_days: int = 63
    purge_days: int = 5  # gap so training targets never overlap the next split
    # Model. Our defaults; `paper_config()` below has the authors' published values.
    hidden: int = 64  # expert width
    base_hidden: int = 0  # base-network width (0 = same as `hidden`)
    gate_style: str = "block"  # "block" (2 hidden layers of `hidden`) or "paper" (1 hidden layer of 16, no dropout)
    mlp_hidden: int = 0  # plain-MLP baseline width (0 = auto size-match)
    experts: int = 4
    dropout: float = 0.1
    log_vol_inputs: bool = True  # our addition, not in the paper (see docs/REGIME_MODEL.md)
    alpha: float = 0.05  # shrinks expert corrections toward the base forecast
    lambda_lb: float = 0.1  # load-balancing penalty on the gate
    # Optimisation (full-batch Adam with early stopping, as in the paper)
    lr: float = 3e-3  # stage 1 (base) and plain-MLP learning rate
    lr_experts: float = 0.0  # stage 2 learning rate (0 = same as `lr`)
    max_epochs: int = 3000
    check_every: int = 1  # validate every N epochs
    patience: int = 150  # stop after this many validations without improvement
    scale_target: bool = True  # standardise y during training (the authors train on raw y)
    vol_floor: float = 0.03  # forecasts below 3% annualized vol are clamped (vol can't be negative)
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2])
    device: str = "cpu"
    max_windows: int = 0  # 0 = all windows; >0 = only the most recent N (quick runs)

    def to_dict(self) -> dict:
        return asdict(self)


def paper_config(**overrides) -> TrainConfig:
    """The authors' settings, from github.com/JunyiYe/rg-resmoe (src/train_rg_resmoe.py, walk_forward.py)."""
    cfg = TrainConfig(
        hidden=16, base_hidden=16, gate_style="paper", mlp_hidden=44, dropout=0.1,
        alpha=1.0, lambda_lb=0.5, lr=0.03, lr_experts=0.01,
        max_epochs=2000, check_every=5, patience=10, scale_target=False, log_vol_inputs=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
