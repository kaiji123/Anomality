"""Is the mood rule's edge over buy-and-hold more than luck? A paired block bootstrap.

The rule was adopted after the first rule lagged buy-and-hold on the same period, so its
backtest is not a clean hold-out test. This module asks a narrower, honest question: given
the daily returns we observed, how sure can we be that the differences are not noise?

Method: resample the two systems' daily net returns *together* (same days, so market moves
cancel) in circular blocks of consecutive days (keeps volatility clustering), recompute the
Sharpe and annual-return differences, repeat. Fixed seed, so the numbers are reproducible.

Run: python -m backend.significance
"""
from __future__ import annotations

import threading
from typing import Dict

import numpy as np
import pandas as pd

BLOCK_DAYS = 20      # about a trading month; results are reported for 5 and 60 too
N_BOOT = 5000
SEED = 0
ANNUAL = 252

_lock = threading.Lock()
_cache: Dict[str, dict] = {}


def _sharpe(x: np.ndarray) -> np.ndarray:
    """Annualized Sharpe (no risk-free rate) along the last axis, as in backend.backtest.stats."""
    sd = x.std(axis=-1, ddof=1)
    return np.where(sd > 0, x.mean(axis=-1) / np.where(sd > 0, sd, 1) * np.sqrt(ANNUAL), 0.0)


def block_indices(n: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, n) day indices: circular blocks of `block` consecutive days."""
    if n < 2 or block < 1:
        raise ValueError("need at least 2 days and a block of at least 1 day")
    nb = -(-n // block)
    starts = rng.integers(0, n, size=(n_boot, nb))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(n_boot, -1)[:, :n]


def paired_bootstrap(a: pd.Series, b: pd.Series, block: int = BLOCK_DAYS,
                     n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Differences a - b in Sharpe and annual return, with 95% intervals.

    `p_not_better` is the share of resamples where a is not better than b (a one-sided
    bootstrap p-value): small means the edge is unlikely to be luck."""
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    if len(both) < 2 * block:
        raise ValueError(f"only {len(both)} common days; need at least {2 * block}")
    x, y = both["a"].to_numpy(), both["b"].to_numpy()
    idx = block_indices(len(x), block, n_boot, np.random.default_rng(seed))
    xs, ys = x[idx], y[idx]
    d_sharpe = _sharpe(xs) - _sharpe(ys)
    d_ret = (xs.mean(axis=1) - ys.mean(axis=1)) * ANNUAL

    def summary(observed: float, draws: np.ndarray) -> dict:
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return {"observed": round(float(observed), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "p_not_better": round(float((draws <= 0).mean()), 4)}

    return {
        "days": int(len(both)), "block_days": block, "n_boot": n_boot, "seed": seed,
        "sharpe_diff": summary(_sharpe(x) - _sharpe(y), d_sharpe),
        "ann_return_diff": summary((x.mean() - y.mean()) * ANNUAL, d_ret),
    }


def verdict(r: dict) -> str:
    """Plain words, from the numbers only."""
    s, a = r["sharpe_diff"], r["ann_return_diff"]
    sig = s["ci95"][0] > 0
    return (f"Sharpe {s['observed']:+.2f} vs. buy-and-hold (95% interval {s['ci95'][0]:+.2f} to {s['ci95'][1]:+.2f}): "
            + ("statistically clear." if sig else "not statistically clear; the interval includes zero.")
            + f" Annual return {a['observed'] * 100:+.1f} pts (interval {a['ci95'][0] * 100:+.1f} to {a['ci95'][1] * 100:+.1f}).")


def mood_vs_buy_and_hold() -> dict:
    """The app's rule vs. buy-and-hold on the walk-forward backtest, for 5/20/60-day blocks."""
    with _lock:
        if "result" in _cache:
            return _cache["result"]
    from .backtest import portfolio_returns
    from .decision_service import frame, held_by_system

    held = held_by_system(frame())
    net = {name: portfolio_returns(h)["net"] for name, h in held.items()}
    by_block = {L: paired_bootstrap(net["mood_rule"], net["buy_and_hold"], block=L) for L in (5, BLOCK_DAYS, 60)}
    main = by_block[BLOCK_DAYS]
    out = {**main, "systems": ["mood_rule", "buy_and_hold"], "text": verdict(main),
           "other_blocks": {str(L): {"sharpe_diff": r["sharpe_diff"], "ann_return_diff": r["ann_return_diff"]}
                            for L, r in by_block.items() if L != BLOCK_DAYS}}
    with _lock:
        _cache["result"] = out
    return out


def main() -> None:
    r = mood_vs_buy_and_hold()
    print(f"Mood rule vs. buy-and-hold, {r['days']} days, {r['n_boot']} paired block resamples (seed {r['seed']})\n")
    for L, rr in [(str(r["block_days"]), r), *r["other_blocks"].items()]:
        s, a = rr["sharpe_diff"], rr["ann_return_diff"]
        print(f"  {L:>3}-day blocks  Sharpe {s['observed']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] p={s['p_not_better']:.3f}"
              f"   annual return {a['observed']:+.4f} [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}] p={a['p_not_better']:.3f}")
    print("\n" + r["text"])


if __name__ == "__main__":
    main()
