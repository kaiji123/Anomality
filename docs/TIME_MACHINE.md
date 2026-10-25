# Historical Time Machine

Pick an asset and a past day. The app shows what the system knew at that day's close, and,
in a separate panel, what happened over the following days. Code: `backend/time_machine.py`;
tests: `tests/test_time_machine.py`.

## What it shows

| Panel | Contents | Built from |
|---|---|---|
| **Known at the close of t** | the RG-ResMoE forecast issued at t, last 20 days' realized volatility (`rv20`), the market mood, the asset's forecast vs. its own past, and the position each rule gives | rows dated **≤ t** only |
| **What happened next** | realized volatility over the forecast's horizon (t+1 … t+5), the forecast for the same days, their ratio, the price move and whether it stayed inside the expected move | rows and prices dated **> t** |
| Chart | forecast and `rv20` up to t; `rv20` after t dashed on a shaded background; the realized volatility over the horizon | the same split |

## Timing conventions

- **Forecast at t**: walk-forward test forecast made from data up to the close of t, by a model
  trained only on earlier windows (with a 5-day purge so training targets never reach the test
  window; `tests/test_walkforward.py`).
- **What the forecast predicts**: `y_t`, annualized realized volatility of daily log returns over
  the next 5 trading days (t+1 … t+5) (`regime/dataset.py`). The outcome panel compares the
  forecast with exactly this quantity, so the ratio is like-for-like.
- **Decision** at the close of t, **traded** at the close of t+1, **first return** earned on t+2
  (`backend/decision.py` `EXEC_LAG`).
- **Calendar**: U.S. trading days for every asset; crypto is sampled on the same days (a Monday
  crypto return includes the weekend), as in the model's training data.
- **A day without a forecast** (weekend, holiday) shows the latest trading day *before* it, with
  a note. Never a later one.
- **Expected move** = forecast × √(5/252), one standard deviation of the 5-day return (about 2
  weeks in 3 stay inside it). A **surprise** is realized ≥ 2× forecast, as in the anomaly report.

## Decisions are reconstructed, not recorded

No decision was logged or traded live. The Time Machine re-runs the rules on the data truncated
at t and labels the result "Reconstructed". It also checks the result against the backtest's
own position for that day ("✓ Same position as the backtest").

- **Mood rule** (the app's rule): half invested when the market forecast is busier than 90% of
  earlier days. Needs 60 days of history ("warming up" before that, fully invested).
- **Previous rule** (surprise): needs `pred` and `rv20` for every day of the asset's history up to
  t; if one is missing, it is reported as not reconstructable instead of being guessed.

## No look-ahead: how it is tested

`test_known_part_ignores_the_future_on_the_real_data` runs the real code on the shipped data,
rewrites or deletes every row after t for all 56 assets (forecasts, inputs, results, prices),
and checks that everything in the "known" panel and the known part of the chart is identical,
while the outcome does change. A mutation check (making one input use the full history) makes
these tests fail.

## Missing data

| Situation | What the app says |
|---|---|
| Asset outside the 56-asset universe | 404, "isn't in the model's universe" |
| Invalid date | 422 |
| Before the first forecast (2019-05-17; later for SOL-USD and AVAX-USD) or after the last | "No forecast exists…" (no numbers shown) |
| Fewer than 5 price days after t | "Result not available yet: only N of the 5 trading days…" |
| Realized volatility missing from the stored results | "not in the stored results" (the price move is still shown if available) |
| `rv20` missing | the previous rule is not reconstructed; the mood rule still is |

## Limits

- Covers the walk-forward test period only (2019-05-17 to 2026-10-16). Days after that use the
  final model's live forecasts elsewhere in the app, not here.
- Market mood here is the *market's* storminess vs. earlier days, the input of the mood rule.
  The asset's "vs. its own past" also uses only earlier days, so it can differ from the asset
  mood shown elsewhere in the app, which ranks this week against the whole history.
- Prices come from the same source as the backtest (Yahoo, or the saved snapshot offline).
