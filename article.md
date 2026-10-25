# Most Volatility Models Break in a Crash. We Found Out Why, Fixed It, and Built an App That Shows Its Work.

*How we built Anomality, a weekly anomaly report for stocks and crypto, and why we spent as
much effort on showing the evidence as on the model itself.*

---

Most forecasting models learn from normal markets. Then a crash arrives, conditions look like
nothing in the training data, and the model fails just when people need it.

We saw this happen. We took RG-ResMoE, a 2026 research model for forecasting volatility
(Ye & Borde, arXiv:2608.12251), and replayed it through the COVID crash. It broke in a
specific way that we could measure. A small change fixed it, and we tested the fix on seven
years of data the model never saw.

Then we built Anomality around the result: an app that tells you how much prices could move
this week, and shows you how far to trust that number.

## Headline numbers

Every number below is **out-of-sample**. The model is retrained on a rolling window and only
scored on weeks it never saw: May 2019 to October 2026, 56 stocks, ETFs and crypto assets,
103,593 forecasts.

| | |
|---|---|
| **±1.96%** | average miss on the weekly move, vs. ±2.24% for the simple "last month" estimate |
| **29 of 30** | test periods where it beat the simple estimate |
| **23% smaller** | misses than the simple estimate during the 2020 crash |
| **25 of 30** | test periods where our fix beat the original research model (p < 0.001) |
| **3.6×** | how much more the assets it flags as most volatile actually moved than the ones it flags as calm |
| **3,177** | weights in the whole model. Retraining the full 7-year backtest takes 12 minutes on a laptop |

## The failure: the model's gate collapsed in the crash

RG-ResMoE has a base network that forecasts next-week volatility, plus four "expert" networks
that add corrections. A **regime gate** decides how much to trust each expert, based on what
the market looks like right now.

That gate is where it broke. In March 2020, market volatility sat **11.7 standard deviations**
outside anything in the original model's training data. With inputs that far outside, the gate
stopped judging stocks one by one and pushed **93% of them into a single expert**. It treated
almost every asset the same just when they were behaving most differently from each other.

## The fix: put volatility on a log scale

Volatility is heavy-tailed. A crash doesn't move it a little; it can multiply it several times.
On a raw scale that shows up as an extreme outlier. On a log scale it looks like a large but
reasonable move.

So we fed the model log-scaled volatility. With that change, the COVID crash sits **4.3**
standard deviations out instead of 11.7, and the gate keeps judging stocks individually. The
fixed model beat the original in 25 of 30 walk-forward test periods (p < 0.001), and its misses
during the 2020 crash were 23% smaller than the simple estimate's.

The lesson goes beyond finance: **before adding model complexity, check what your inputs look
like when the world gets strange.** One rescaled feature did more than a bigger network would
have.

## Making the app show its work

A forecast on its own doesn't tell you whether to believe it. Most of the product work went
into making every number traceable and every limitation visible.

### Moves in plain units

The app shows "±3.9% this week" or "about ±$10.29 on today's $210.96 price", not "annualized
volatility". The market mood (Calm, Normal, Choppy, Stormy) comes with a scale showing what
each level meant historically and when the market was last there. Each asset lists its
**surprise weeks**: weeks when it moved at least twice as much as forecast.

It also says plainly what it doesn't do: it forecasts *how much* prices may move, not *which
direction*.

### The Historical Time Machine: what was known vs. what happened next

Pick an asset and a past day, such as the start of the COVID crash. The app shows two panels:

- **Known at the close of that day:** the forecast, the inputs and the market mood, built only
  from data dated on or before that day.
- **What happened next:** the volatility and price move over the following five trading days.

Look-ahead bias quietly ruins many backtests, so we test against it directly. One test runs the
real code on the shipped data, rewrites or deletes **every row after the chosen day** for all 56
assets, and checks that the "known" panel doesn't change while the outcome does. We also check
the test itself: if one input is changed to use the full history, the test fails.

Decisions shown in the Time Machine are labelled **"Reconstructed"**. Nothing was traded live,
and the app doesn't pretend otherwise.

### Alerts that show their measurements

An alert that's just a red dot gives you nothing to investigate. Each Anomality alert reports
the measured value, the threshold and the time windows it used. The monitors are:

- **Market stress:** the market forecast is busier than 90% of earlier days.
- **Model surprise:** an asset moved at least 2× its forecast over the last 20 days.
- **Regime change:** the market mood changed and stayed changed for 3 trading days.
- **Data quality:** missing forecasts, invalid values, stale prices.
- **Drift:** the inputs and the forecasts are each compared, last 21 days vs. the 252 before,
  using PSI (a standard score for how much a distribution has shifted). The model's accuracy
  is compared with the simple estimate's; if the model stops beating it, that's an alert.

Alerts use only data available on the day they're evaluated, so you can replay them through
history with the Time Machine.

### The AI analyst: the backend owns the numbers, Gemini explains them

We added a Google Gemini analyst, but with a firm rule: **it never computes anything.**

1. The backend collects verified numbers from the app's own calculations as numbered facts
   (F1, F2, …), each with its source.
2. Gemini gets only those facts and must answer in four sections: measured facts,
   interpretation, uncertainties, and missing information. It cites fact ids as it goes.
3. A validator **rejects any answer that cites an unknown fact or contains a number that isn't
   in the facts.** If the answer is rejected, the call fails, or no key is configured, the app
   shows a plain summary built from the same facts.

So the page always has an explanation with correct numbers, whatever the model does. Number
checks don't catch everything (a correct number can still come with a misleading sentence),
which is why the facts are always shown next to the explanation.

Voice reports (via ElevenLabs) follow the same rule. The endpoint takes a report id, not free
text, so the audio can only read back the report that was already shown. If the voice service
is down or the account can't use the chosen voice, the written report still works.

## Being honest about the trading rule

The app also has a simple decision layer: **when the market mood is Stormy, hold every asset at
half; otherwise stay fully invested.** In the backtest (after 0.05% trading costs, with trades
placed the next day):

| | Growth | Worst drop | COVID crash drop | Return per unit of risk |
|---|---|---|---|---|
| **Mood rule** | +579% | −30.3% | −21.3% | 1.37 |
| Buy and hold | +568% | −34.2% | −33.8% | 1.22 |

That looks good, so we checked whether it would hold up:

- **It isn't statistically clear.** A paired block bootstrap (5,000 resamples) gives a 95%
  interval for the Sharpe gap that includes zero. The fair reading is "similar return with
  smaller drops", not "beats the market".
- **Most of the edge comes from two storms**, 2020 and 2022. Two crashes is a small sample.
- **It isn't a clean hold-out test.** We adopted this rule after our first rule lagged
  buy-and-hold over the same period. The 90% threshold is the app's own Stormy band, not a
  tuned value (95% actually scored better), but some selection bias remains. The fair test is
  to run the rule unchanged on data after October 2026.

The app shows all of this alongside the chart.

## How we tested it

The test suite covers walk-forward splits (including a 5-day purge so training targets never
reach the test window), data cleaning, metrics, the decision layer, the Time Machine's
no-look-ahead guarantee, alert logic, the analyst's validator, and provider failures for Gemini
and ElevenLabs. On the current code it runs **205 passed, 3 skipped**. The skipped tests call
Yahoo Finance live and run separately.

Mocked tests and live tests prove different things. A mocked test shows we handle a provider
error correctly. It can't show that a real account is allowed to use a particular voice, and we
found that out the hard way.

## Five takeaways

1. **Check your inputs under stress.** Our biggest gain came from rescaling one feature, not
   from a bigger model.
2. **Keep the numbers outside the language model.** Compute in code, pass the model numbered
   facts, and reject answers that invent numbers.
3. **Make the timeline explicit.** Show what was known at prediction time separately from what
   happened later, and write a test that deletes the future.
4. **Show why an alert fired.** A measured value, a threshold and a time window give people
   something to investigate.
5. **Run significance tests on your own good results.** A +579% backtest that isn't
   statistically clear should be presented that way.

## What's next

- Run the mood rule unchanged on post-October 2026 data, as a true hold-out test.
- Evaluate the AI explanations more broadly: missing caveats and unsupported causal claims, not
  just numbers.
- Tighten drift alerts so a one-off unusual week is told apart from a lasting change in
  behaviour.

---

Anomality is open source: [github.com/kaiji123/Anomality](https://github.com/kaiji123/Anomality).
It's a research tool, not financial advice.

We'd like to hear from other developers. How do you validate AI-generated explanations? How do
you guard against look-ahead in time-series evaluation? What has worked for you when a
third-party AI service sits in a system people rely on?
