"""Log-scaling touches only the volatility inputs (regime/train.py: Scaler)."""
import numpy as np
import pandas as pd

from regime.config import LOG_COLS, X_COLS, Z_COLS
from regime.train import Scaler


def _frame(n=50, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(0.05, 0.6, size=(n, len(X_COLS) + len(Z_COLS))), columns=X_COLS + Z_COLS)
    df["y"] = rng.uniform(0.05, 0.6, n)
    return df


def test_log_scaling_changes_only_volatility_columns():
    df = _frame()
    x_raw, z_raw = Scaler._inputs(df, log_vol=False)
    x_log, z_log = Scaler._inputs(df, log_vol=True)
    for i, col in enumerate(X_COLS):
        expected = np.log(df[col]) if col in LOG_COLS else df[col]
        assert np.allclose(x_log[:, i], expected), col
        if col not in LOG_COLS:
            assert np.allclose(x_log[:, i], x_raw[:, i]), col
    assert np.allclose(z_log, np.log(df[Z_COLS].to_numpy()))


def test_standardisation_uses_training_statistics():
    train, test = _frame(seed=1), _frame(seed=2)
    sc = Scaler.fit(train, log_vol=True)
    x, _, _ = sc.tensors(train, "cpu")
    assert np.allclose(x.numpy().mean(0), 0, atol=1e-5)
    assert np.allclose(sc.unscale(np.array([0.0])), train["y"].mean())
    x_test, _, _ = sc.tensors(test, "cpu")  # test data is scaled with the training stats, not its own
    assert not np.allclose(x_test.numpy().mean(0), 0, atol=1e-3)
