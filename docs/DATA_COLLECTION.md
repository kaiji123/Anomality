# Data collection

How anomality gets its market data, what it does to it before analysis, and the
known limitations. The code lives in [`backend/data.py`](../backend/data.py); feature
preparation is in [`backend/analysis.py`](../backend/analysis.py).

## Source

| | |
|---|---|
| Provider | **Yahoo Finance**, public chart endpoint (unofficial, no API key or account) |
| Endpoint | `GET https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}` |
| Parameters | `range` = `1y` \| `2y` \| `5y` \| `10y` (the History control), `interval=1d`, `includePrePost=false` |
| Granularity | One candle per trading day (crypto: one per UTC calendar day, 7 days a week) |
| Fields used | `timestamp`, `quote.open/high/low/close/volume`, `adjclose`, and `meta` (name, currency, exchange, time zone, current trading period) |
| Coverage | Anything Yahoo lists: US and international stocks (`AAPL`, `BHP.AX`), ETFs (`SPY`), indices (`^GSPC`), crypto pairs (`BTC-USD`) |

Requests send a browser-style `User-Agent` header (Yahoo rejects requests without one) and
time out after 15 seconds.

The default watchlist (12 US stocks/ETFs and 8 crypto pairs) is defined in
[`backend/app.py`](../backend/app.py) as `WATCHLIST`. Tickers you add in the app are
saved only in your browser (`localStorage`).

## When data is fetched

- Data is requested **on demand**: when the watchlist is scanned, when you open an asset,
  or when you change a setting.
- Each `(symbol, history range)` result is **cached in memory for 10 minutes**, so
  changing the sensitivity, horizon, model or tree depth reuses the same prices.
- **Nothing is written to disk.** Restarting the server clears the cache.
- A scan fetches up to 8 symbols in parallel.

## Cleaning steps

Applied in this order to every series:

1. **Trading dates in the exchange's time zone.** Yahoo timestamps are in UTC and mark the
   session open, so a Sydney session starting 10:00 local time is stamped 23:00 UTC the
   *previous* day. Timestamps are converted to the exchange's time zone
   (`meta.exchangeTimezoneName`) before taking the date.
2. **Split and dividend adjustment.** Yahoo's raw prices and volumes are already
   **split-adjusted** (verified on Apple's 4:1 split in August 2020: there is no 4× price jump).
   They are *not* dividend-adjusted, so open/high/low/close are multiplied by
   `adjclose / close`. Otherwise every ex-dividend day would look like a small artificial drop.
3. **Missing values.** Days with no close are dropped. A missing open/high/low is filled
   with that day's close; missing volume is set to 0.
4. **Duplicates.** During a live session Yahoo can return an extra row with the latest
   price. If two rows share a date, the later one is kept.
5. **Unfinished sessions are excluded.** If the current trading session (from
   `meta.currentTradingPeriod`) has not closed yet, its candle is dropped, because partial
   volume and range would look artificially quiet. For crypto, a day closes at 00:00 UTC.
   The app notes when this happened.
6. **Minimum history.** At least 150 days are required, otherwise the request fails with
   "Not enough history".
7. **Warm-up.** The first 60 days are used only to compute rolling indicators (the longest
   window is 60 days) and are then removed from the analysis.

## Derived data

All indicators use only information available at the end of each day; nothing looks into the
future. z-scores compare a day against the **previous** 20 days, not including the day itself.

| Feature | Definition |
|---|---|
| Return z-score | log return ÷ standard deviation of the previous 20 daily log returns |
| Volume z-score | log(1 + volume) vs. mean and SD of the previous 20 days |
| Intraday range z-score | (high − low) / close vs. the previous 20 days |
| Opening gap z-score | log(open / previous close) ÷ 20-day return SD |
| Volatility regime | SD of returns over 5 days ÷ SD over 60 days |
| 5-day momentum | 5-day log return ÷ (20-day SD × √5) |
| Distance from 50-day avg | close ÷ 50-day moving average − 1 |
| RSI (14) | Wilder's relative strength index |
| Bollinger %B | position of the close within the 20-day ±2 SD bands |
| Shocks in last 10 days | count of days with \|return z\| above the sensitivity threshold |
| Volume trend | 5-day average volume ÷ 60-day average volume |

**Model target (label).** A day is labelled 1 if any of the next *H* days (the Horizon
setting) has \|return z\| above the sensitivity threshold. The last *H* days have no
known outcome yet, so they are left out of training and only scored.

## Offline fallback (synthetic data)

If Yahoo cannot be reached (no network, timeout, server error or rate limit), the app
generates **synthetic demo data** instead of failing: a random walk with fat-tailed returns,
shifting volatility and occasional jumps, seeded by the ticker name. Synthetic data is
always flagged (`synthetic: true` in the API, a yellow banner and "(synthetic demo data)" in
the name). An unknown ticker never falls back; it returns "Symbol not found".

## Known limitations

- **Unofficial endpoint.** Yahoo does not publish or guarantee this API. It can change, be
  rate-limited or break without notice. Yahoo's terms allow personal, non-commercial use;
  check them before using the data for anything else.
- **Daily data only.** Intraday anomalies that reverse before the close are invisible.
- **Survivorship.** Only currently listed tickers can be fetched, so delisted companies
  are missing from any historical comparison.
- **Volume quality.** Crypto volume is aggregated by Yahoo from a subset of exchanges, and
  some indices report no volume. When volume is all zero, volume features are neutralised.
- **Data errors.** Yahoo occasionally has bad prints (spikes, missing days). These can
  show up as anomalies, so check a surprising flag against a second source.
