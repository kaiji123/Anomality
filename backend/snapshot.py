"""Save 10 years of daily prices for every asset the app ships with, into data/snapshot/.

The snapshot is committed to git, so a fresh clone can run the whole app (watchlist, anomaly
report, decision layer) even when Yahoo Finance is rate-limiting or unreachable.

Run: python -m backend.snapshot
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from regime.config import CRYPTO, STOCKS

from .app import WATCHLIST
from .data import SNAPSHOT_DIR, _fetch_yahoo, _safe, _write

PERIOD = "10y"


def universe() -> list:
    return sorted(set(STOCKS) | set(CRYPTO) | set(WATCHLIST["stocks"]) | set(WATCHLIST["crypto"]))


def main() -> None:
    syms = universe()
    saved_at = time.time()

    def one(sym: str):
        try:
            df, info = _fetch_yahoo(sym, PERIOD)
        except Exception as e:  # report and keep going; the old snapshot file (if any) stays
            return sym, f"skipped ({type(e).__name__}: {e})"
        _write(df, info, SNAPSHOT_DIR / f"{_safe(sym)}.csv.gz", {"saved_at": saved_at, "period": PERIOD})
        return sym, f"{len(df)} days to {df.index[-1].date()}"

    with ThreadPoolExecutor(max_workers=4) as pool:  # gentle on Yahoo
        for sym, msg in pool.map(one, syms):
            print(f"  {sym:<10} {msg}")
    size = sum(f.stat().st_size for f in SNAPSHOT_DIR.glob("*")) / 1e6
    print(f"\n{len(syms)} assets -> {SNAPSHOT_DIR} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
