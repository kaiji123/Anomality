# Intelligent alerts and drift detection

Deterministic monitors on the forecasts, their inputs and the prices. Code: `backend/alerts.py`;
tests: `tests/test_alerts.py`; API: `GET /api/alerts[?date=YYYY-MM-DD]`,
`POST /api/alerts/{id}/ack`.

Alerts describe conditions that were measured. They do not predict a crash or a loss.

## How it works

On an evaluation day t, every monitor reports `ok`, `alert` or `insufficient_data`, with the
measured value, the threshold, the unit and the evidence it used. **Only data available at the
close of t is used**: rows dated after t are never read, and a forecast's realized result is
used only once its 5-day horizon has ended (see performance drift).

The newest forecast day is the default. Any past day can be evaluated as it was then; the
"Use the Time Machine day" button lines the two features up.

An **alert** is a monitor that fired, grouped into an episode of consecutive trading days. Its
id is `monitor:scope:first day`. The first day is traced back as far as the condition actually
held, however long ago (re-running only that monitor on each earlier day, with the data
available then), so an ongoing episode keeps the same id and first-seen date on every refresh,
even when it has lasted longer than the 20-day lookback. If the episode runs back to the first
day with data, `first_seen_exact` is `false`: it may have started earlier. `stale_prices`
compares with the clock, so it only exists on the evaluation day. Statuses:

| Status | Meaning |
|---|---|
| Active | the condition holds on t |
| Seen | active, and acknowledged by a user (kept in server memory until restart; no database exists in the app, and the alert itself is recomputed from data, so nothing is lost) |
| Resolved | the condition stopped within the last 20 trading days; shows when (its first day may be older) |

## Monitors and thresholds

| Monitor | Type | Fires when | Why this threshold |
|---|---|---|---|
| market_stress | Market stress | market-wide forecast busier than **≥ 90%** of earlier days | the app's own Stormy band, and the mood rule's trigger |
| model_surprise | Market stress | an asset's last 20 days moved **≥ 2×** its forecast | the anomaly report's surprise mark |
| regime_change | Regime change | the market mood band changed and has held **3–20** trading days | 3 days filters one-day flickers; after 20 days it's no longer a change |
| missing_forecasts | Data quality | an asset with a forecast the previous trading day has none on t | a market holiday removes every asset, so it never fires on one |
| invalid_values | Data quality | forecast or rv20 missing, non-finite, or not positive on t | |
| missing_returns | Data quality | no price return on a trading day | |
| stale_prices | Data quality | newest price more than **3 business days** old (latest day only, against the clock) | a weekend plus a holiday |
| input_drift | Input drift | PSI of log(rv20), all assets, last **21** days vs. the **252** before, **≥ 0.25** | PSI rule of thumb: < 0.10 stable, 0.10–0.25 moderate, ≥ 0.25 major |
| prediction_drift | Prediction drift | same PSI on log(forecast) | |
| performance_drift | Performance drift | model QLIKE ÷ simple-estimate QLIKE **> 1** over the last 21 days whose result is known | above 1 the model has no skill over "last month's volatility" |

### Three kinds of drift, kept apart

- **Input drift**: the inputs' distribution moved (the market got wilder or calmer). Not a model
  failure by itself; it happens in every crisis.
- **Prediction drift**: the forecasts' distribution moved.
- **Performance drift**: the forecasts got *worse relative to the simple estimate*. A wilder
  market makes both worse, so it does not fire on volatility alone; it fires when the model
  loses its skill. It only uses forecasts dated up to t − 5 trading days, whose 5-day result
  was known at the close of t.

The tests build each kind separately (shift rv20 only, shift the forecasts only, shuffle the
forecasts across assets so their distribution is unchanged but their skill is gone) and check
that only the matching monitor fires.

## Insufficient data

Instead of an alert, a monitor reports `insufficient_data` when:

- the market mood needs 60 days of forecast history (warming up);
- drift needs 21 + 252 = 273 trading days of history (so it cannot run before about mid-2020);
- performance drift needs 21 days with known results.

## No look-ahead: how it is tested

- `test_alerts_on_a_past_day_ignore_later_data_*`: rewriting or deleting every row after t (on
  the shipped data and on synthetic panels) leaves the alerts at t identical.
- `test_performance_drift_uses_only_completed_horizons`: wrecking the results of the last 5 days
  changes nothing. A mutation that uses those results makes it fail. This needs its own test:
  a forecast row dated t carries its result over t+1 … t+5, so rows dated on or before t can
  hold future information, which the "rewrite rows after t" test cannot see.

## Limits

- Thresholds are conventions (the app's bands, a PSI rule of thumb, "no skill"), not
  significance tests. PSI on overlapping 20-day volatilities is smooth and autocorrelated.
- Monitors run on the walk-forward forecasts (to 2026-10-16). Stale prices compares the newest
  saved price with the server's clock.
- Acknowledgements live in server memory and reset on restart.
