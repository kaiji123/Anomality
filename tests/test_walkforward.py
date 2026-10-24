"""Walk-forward splits never leak future targets into training (regime/train.py)."""
import pandas as pd

from regime.config import TrainConfig
from regime.train import windows


def test_splits_are_ordered_with_purge_gaps_and_slide_by_the_step():
    cfg = TrainConfig()
    dates = pd.bdate_range("2017-01-02", periods=900)
    wins = windows(dates, cfg)
    assert len(wins) >= 2
    pos = {d: i for i, d in enumerate(dates)}
    for w in wins:
        train_end, val_start = pos[w["train"][-1]], pos[w["val"][0]]
        val_end, test_start = pos[w["val"][-1]], pos[w["test"][0]]
        # 5-day targets of the last training/validation rows must not reach the next split
        assert val_start - train_end > cfg.purge_days
        assert test_start - val_end > cfg.purge_days
        assert len(w["test"]) <= cfg.test_days
    assert pos[wins[1]["test"][0]] - pos[wins[0]["test"][0]] == cfg.step_days
