# daytrader

mma@Q-HQ7H1KQV64-mma daytrader % source .venv/bin/activate
(.venv) mma@Q-HQ7H1KQV64-mma daytrader % pip install -r requirements.txt

A day-trading app built on the Charles Schwab Trader API. Planned in three
sections:

1. **Pulling data** (this section — done)
2. **ML model** for buy/sell/hold signals (framework in place, see below — training/backtesting a specific model is next)
3. **Trade execution** (TODO)

## Setup

1. Register a developer app at https://developer.schwab.com/dashboard/apps
   and request the "Accounts and Trading Production" + "Market Data
   Production" API products. Note the App Key, App Secret, and set a
   callback URL with a port (e.g. `https://127.0.0.1:8182`) — Schwab's SDK
   defaults to port 443 if you omit one, which needs root to bind.
2. `pip install -r requirements.txt`
3. `cp env.example .env` and fill in `SCHWAB_API_KEY` / `SCHWAB_APP_SECRET` /
   `SCHWAB_CALLBACK_URL` — the callback URL **must exactly match** what you
   registered on the app (scheme, host, port, trailing slash).
4. Run the one-time interactive login:
   ```bash
   python -m schwab_client.auth
   ```
   This prints a Schwab login URL — open it, log in, approve access. Schwab
   redirects your browser to the callback URL; **copy that full URL from the
   address bar and paste it back into the terminal** when prompted
   (`Redirect URL>`). Do this quickly — Schwab's authorization codes are
   short-lived, and we deliberately don't auto-capture the redirect with a
   local server (that approach makes the browser show a "connection is not
   private" warning for the self-signed local listener, and the delay while
   reading/clicking through it is often enough to expire the code — see
   `schwab_client/auth.py` for details). This writes a token cache to
   `schwab_token.json` (path configurable via `SCHWAB_TOKEN_PATH`). The
   access token silently refreshes after that; the refresh token itself
   expires after 7 days and requires repeating this step.

## Running

```bash
python main.py --mode trading   # streams live data from market open to close
python main.py --mode train --symbol AAPL --strategy random_forest \
    --days-back 45 --validation-days 5 --out models/random_forest_aapl.pkl
python main.py --mode testing   # backtesting — not implemented yet (TODO)
```

`--mode trading` can be started at any time — it figures out whether the
market is open today (accounting for weekends/holidays/early closes) and
either starts streaming immediately or sleeps until the open. It then runs
continuously until the close, disconnects, and waits for the next session.
On every closed 1-minute bar it logs the active strategy's signal(s) and
probabilities for the current feature vector (see "The multi-horizon LSTM"
below for what that looks like with 6 horizons at once) — nothing gets
traded yet (section 3), it's purely observational for now.

`--mode train` fetches historical bars for one symbol, fits the chosen
strategy on all but the most recent N trading days, and reports accuracy on
those held-out days. See `modes/train_mode.py`'s docstring (or
`python -m modes.train_mode --help`) for the full flag reference — it also
works as a standalone command, not just through `main.py`.

Pass `--out models/some_name.pkl` to save the trained model, and the
validation results (the per-horizon accuracy/recall lines) are
automatically saved alongside it as `models/some_name.txt` — or use
`--report-out some/other/path.txt` to save the report somewhere else, even
without `--out`. Useful for comparing runs later without scrolling back
through terminal history. Report `.txt` files aren't gitignored (unlike `models/`'s binary model
files) since they're small and worth keeping as an experiment log — add a
`.gitignore` entry yourself if you'd rather not track them.

Pass `--plot-out models/some_name.png` (or let it default from `--out` the
same way `--report-out` does) for a chart per horizon —
`models/some_name_1min.png`, `models/some_name_5min.png`, etc. — of actual
close price over the validation window, with a marker at every bar the
strategy predicted BUY (▲) or SELL (▼), filled if that call turned out
correct, hollow if it didn't. The x-axis is bar sequence, not real elapsed
time — otherwise every overnight/weekend gap (no trading = no bars, but
still real hours) stretches the axis and squeezes the actual trading data
into unreadable clumps; tick labels still show the real date/time. This is
often more informative than the accuracy number alone: a model can have
mediocre accuracy but visibly sane behavior (buying dips, selling peaks) or
good accuracy that's really just class imbalance (see the "naive baseline"
flag in the text report) — looking at the chart makes that distinction
obvious in a way a single percentage can't. Needs `matplotlib` (only
imported if you actually request
a plot — not required for training/evaluation on its own).

## Project layout

```
config.py                  - env-var based configuration
schwab_client/
  auth.py                  - OAuth login / token management
  market_data.py           - REST market data (quotes, price history, option chains, movers, market hours)
  streaming.py              - WebSocket live data (level-one quotes, minute bars)
modes/
  schedule.py               - "is the market open, and when does it close today"
  live_features.py           - rolling per-symbol bar buffer -> feature vector, for trading_mode.py
  trading_mode.py             - the always-on live loop
  train_mode.py                - fetch history, fit a strategy, evaluate on held-out days
  testing_mode.py               - backtesting stub (TODO)
strategies/
  base.py                    - Signal/Prediction/StrategyBase — the interface every strategy implements
  registry.py                 - name -> class lookup + auto-discovery (get_strategy, available_strategies)
  features.py                  - shared technical-indicator feature engineering
  rule_based.py                 - SMA crossover, RSI mean-reversion (no training required)
  sklearn_models.py              - logistic regression, random forest, gradient boosting
  deep_learning.py                - lstm_three_classifier (multi-horizon LSTM); future lstm_regression will live here too
  reinforcement_learning.py        - PPO/DQN agent — stub, TODO
main.py                     - CLI entry point
tests/
  test_strategies.py         - registry + strategy-swapping + per-strategy tests
  test_live_features.py       - CHART_EQUITY message parsing -> feature vectors
  test_train_mode.py           - train/validation day-splitting, strategy construction
  (run all: python -m unittest discover tests)
```

## What data is available, and what's useful for day trading

The Schwab Market Data API exposes several kinds of data. Roughly in order
of how useful each is for an intraday model:

| Data | Where | Why it matters for day trading |
|---|---|---|
| **Level-one quotes** (bid/ask/last price & size, volume, day high/low, net change) | `streaming.LiveQuoteStream` (`LEVEL_ONE_EQUITY`), or `MarketDataClient.get_quote(s)` for snapshots | The core signal. Bid/ask spread and size tell you liquidity and short-term pressure; last price + volume drive most technical indicators. |
| **Minute (or 5/10/15/30-min) OHLCV bars** | `streaming.LiveQuoteStream` (`CHART_EQUITY`) live, `MarketDataClient.get_intraday_bars` for history | What most technical-indicator features (moving averages, RSI, MACD, VWAP, momentum, volatility) are computed from. You need history to warm these up before the model can produce signals — that's what `get_intraday_bars` is for at startup. |
| **Level-two order book depth** (`NASDAQ_BOOK`/`NYSE_BOOK`) | Not yet wired up; add via `StreamClient.nasdaq_book_subs` similar to `streaming.py` | Shows resting orders beyond best bid/ask — useful for order-flow/imbalance features in more advanced models, but heavier to process and not essential to start. |
| **Option chains** | `MarketDataClient.get_option_chain` | Only relevant if trading options directly, or if you want implied volatility as a sentiment/feature input for the underlying. |
| **Market movers/screeners** | `MarketDataClient.get_movers` | Useful for *symbol selection* (which stocks are moving today) rather than modeling a single symbol you already picked. |
| **Instrument/fundamental data** | `MarketDataClient.get_instruments` (not wired up yet) | Mostly useful for filtering your trading universe (sector, market cap), not for intraday signals. |
| **Market hours** | `MarketDataClient.get_market_hours` | Operational, not a model input — used by `modes/schedule.py` to know when to start/stop. |

**Recommendation for a first model:** start with live level-one quotes +
minute bars for a small, liquid symbol list (the two feeds already wired up
in `streaming.py`), backfilled with `get_intraday_bars` on startup so your
indicators aren't computing on an empty window. Add order-book depth later
if/when you want finer-grained order-flow features.

## ML model options

Every strategy — rule-based or ML — implements one interface
(`strategies.StrategyBase.predict(features) -> Prediction`, see
`strategies/base.py`) and is looked up by name from a registry
(`strategies/registry.py`). Swapping models is a one-line change:

```python
from strategies import get_strategy

strategy = get_strategy("random_forest")   # or "sma_crossover", "logistic_regression", ...
prediction = strategy.predict(feature_row)  # Prediction(signal=Signal.BUY, confidence=0.71)
```

or, for `trading_mode.py`, just set `DAYTRADER_STRATEGY=random_forest` in `.env` — no code changes.

**To add your own:** drop a new file in `strategies/`, subclass `StrategyBase`,
implement `predict()` (and `fit()` if it's trainable), decorate the class
with `@register("your_name")`. New files are auto-discovered — nothing else
needs to change, and one strategy's broken/missing dependency (e.g. `torch`
not installed) won't break the others (see `registry._discover`).

### Input features (`strategies/features.py::FEATURE_COLUMNS`)

Every strategy sees the same 16 engineered features, computed from raw
OHLCV bars:

| Feature | What it measures | Flavor |
|---|---|---|
| `return_1`, `momentum_5` | 1-bar / 5-bar % change | momentum |
| `sma_5`, `sma_20`, `sma_50` | moving averages at 3 timescales | trend context |
| `sma_50_slope` | rate of change of the *long* average itself | trend acceleration |
| `ema_12`, `ema_26`, `macd_pct` | exponential MAs and their gap, as a % of price | momentum/trend |
| `rsi_14` | 14-bar relative strength index | **mean-reversion** |
| `volatility_20` | 20-bar rolling std of returns | risk/regime |
| `adx_14` | Average Directional Index — trend *strength*, direction-agnostic | trend strength |
| `trend_slope_20`, `trend_r2_20` | rolling linear-regression slope and fit quality over 20 bars | trend strength/cleanliness |
| `trend_streak` | signed count of consecutive same-direction bars | persistence |
| `vwap_deviation` | deviation from the *daily-resetting* session VWAP | **mean-reversion** |

Two of these (`rsi_14`, `vwap_deviation`) are classically contrarian — high
RSI / large VWAP deviation are textbook "overextended, expect reversal"
signals, which is the wrong read during a genuine sustained trend. The
`adx_14`/`trend_slope_20`/`trend_r2_20`/`trend_streak`/`sma_50_slope` group
exists specifically to counterbalance that: they measure trend *strength
and persistence* directly, which nothing in the original 10-feature set
did. If a model still can't hold a directional call through a real trend,
that's now a "the model isn't weighting these features usefully" question
you can actually investigate (e.g. check feature importances on
`random_forest`/`random_forest_multi_horizon` — scikit-learn exposes
`.feature_importances_` on the underlying model) rather than "the
information was never there."

**Changing `FEATURE_COLUMNS` invalidates previously-saved models** — a
`.pkl`/`.pt` trained against the old 10-feature vector will silently
mismatch a 16-feature input. Retrain (`--out` the same path to overwrite)
after picking up a features.py change.

### What's implemented vs. stubbed

| Category | Strategies | Status |
|---|---|---|
| Rule-based baselines | `sma_crossover` (trend-following), `rsi_mean_reversion` | Implemented — no training needed, use these to sanity-check the pipeline and as a bar any ML model has to clear |
| Classical ML (single-horizon) | `logistic_regression`, `random_forest`, `gradient_boosting` | Implemented (scikit-learn) — predict one horizon (default: next bar). Need `fit()` on historical data before they'll do anything but `HOLD`. |
| Classical ML (multi-horizon) | `random_forest_multi_horizon` — 6 *independent* RandomForestClassifiers, one per horizon (1min/5min/10min/30min/1h/2h), same interface and same default horizons as the LSTM | Implemented (`strategies/sklearn_models.py`) — built specifically to compare against `lstm_three_classifier` head-to-head. Costs ~6x a single-horizon model to train (no shared encoder). |
| Deep learning | `lstm_three_classifier` — **multi-horizon**: one shared LSTM + 6 heads predicting 1min/5min/10min/30min/1h/2h ahead independently | Implemented (`strategies/deep_learning.py`, PyTorch) — see below. A future `lstm_regression` sibling (same encoder, regression heads) is planned. |
| Reinforcement learning | PPO/DQN agent | Stubbed (`strategies/reinforcement_learning.py`) — needs a backtest environment (testing_mode.py) to train against first |

### The multi-horizon LSTM

`get_strategy("lstm_three_classifier")` behaves exactly like every other strategy —
`predict(features) -> Prediction` — but internally buffers a rolling
window of feature rows and runs them through a shared LSTM encoder with
one small classifier head per horizon. `predict()` returns the primary
horizon's signal (default `"1min"`); the full 6-way breakdown is always
available:

```python
from strategies import get_strategy

strategy = get_strategy("lstm_three_classifier")          # untrained: everything HOLDs until fit() + a full window
prediction = strategy.predict(feature_row)
prediction.metadata["horizons"]           # {"1min": Prediction(...), "5min": Prediction(...), ..., "2h": Prediction(...)}
# or, equivalently:
strategy.predict_all_horizons(feature_row)
```

**Training:** it needs raw OHLCV history, not pre-computed features/labels
(it derives 6 sets of labels from `close` internally — one label array per
horizon can't be expressed as a single external `y`). Every trainable
strategy shares this same `fit(bars)` contract now (see
strategies/base.py's `StrategyBase.fit` docstring) — including
`logistic_regression`/`random_forest`/`gradient_boosting`, which used to
take pre-built features + external labels — specifically so `modes/train_mode.py`
can train any of them the same way:

```bash
pip install -r requirements.txt   # now includes torch — see the CPU-only install note in requirements.txt
python -m modes.train_mode --symbol AAPL --strategy lstm_three_classifier \
    --days-back 45 --validation-days 5 --out models/lstm_three_classifier_aapl.pt
```

Then, to reuse the trained model instead of retraining:

```python
from strategies.deep_learning import LstmMultiHorizonStrategy
strategy = LstmMultiHorizonStrategy.load("models/lstm_three_classifier_aapl.pt")
```

Caveats specific to this model, beyond the general ones below: it's trained
per-symbol on whatever history Schwab returns for that `--minutes-per-bar`
(as little as ~48 days for 1-minute bars — see `schwab_client/market_data.py`),
which is thin for a 6-head model; and the `sma_*`/`ema_*` features are
absolute price levels, not returns, so a model trained on one symbol won't
transfer to another without retraining. (The VWAP-not-reset-daily issue
flagged in an earlier version of this doc is fixed — see the feature list
below.)

**Comparing against random forest:** `random_forest_multi_horizon` (see the
table above) uses the identical 6 horizons, feature set, and label
threshold, so it's a fair head-to-head baseline for the LSTM:

```bash
python main.py --mode train --symbol AAPL --strategy lstm_three_classifier \
    --days-back 45 --validation-days 5
python main.py --mode train --symbol AAPL --strategy random_forest_multi_horizon \
    --days-back 45 --validation-days 5
```

Compare the per-horizon accuracy/recall lines each prints. Per Ghosh, Neufeld
& Sahoo (2021) below, expect the LSTM to edge out random forest by a real
but modest margin — if it doesn't, or random forest wins on some horizons,
that's a legitimate result, not a sign something's broken; it just means
the extra complexity isn't earning its keep for this symbol/horizon.

### Optimizing for a specific horizon ("strong signal over 30min-1h", not noisy 1-minute wiggles)

`train_mode` lets you reshape what's being trained/evaluated from the CLI —
no code changes — via `--horizon-minutes` and `--strategy-kwargs`:

```bash
python main.py --mode train --symbol AAPL --strategy random_forest_multi_horizon \
    --minutes-per-bar 5 --days-back 270 --validation-days 10 \
    --horizon-minutes 30,60 --label-threshold 0.004 \
    --out models/random_forest_multi_horizon_aapl_30_60.pkl
```

What each change is doing, and why, if you're chasing "strong signal, 30min
to 1h, don't care about minute-to-minute noise":

- **`--minutes-per-bar 5` + `--days-back 270`**: Schwab caps 1-minute bars
  at ~48 days but allows ~9 months at 5-minute resolution — 5-6x more
  training data for free, and 5-minute bars are naturally less noisy than
  1-minute ones for a 30-60 minute question.
- **`--horizon-minutes 30,60`**: trains/evaluates only the two horizons you
  actually want, instead of all six. For the LSTM specifically, this also
  stops the noisy short-horizon heads from sharing (and potentially
  degrading) the same encoder the 30min/1h heads use — see
  `strategies/deep_learning.py`'s multi-task-learning note.
- **`--label-threshold 0.004`**: bigger than the 0.001 used for the
  1-minute head — "strong" over 30-60 minutes should mean a bigger move
  than "strong" over 1 minute. 0.004 (0.4%) is a starting point, not a
  researched number — tune it against what the recall/predicted-vs-actual
  breakdown and the price chart (`--plot-out`) actually show for this
  symbol; too high and everything collapses back to HOLD-only, too low and
  you're back to noise.
- **`--strategy-kwargs '{"sequence_length": 24}'`** (LSTM only, not shown
  above): with 5-minute bars, the default `sequence_length=60` means 5
  hours of lookback for a 30-60 minute question — possibly more context
  than needed. Worth comparing against the default.

Judge results the same way as any other run: read the recall-by-class line
(not just accuracy — a 90% accuracy that's really 0% BUY/SELL recall is the
same trap as before), check it clears the naive-baseline flag, and look at
the chart. None of these values are guaranteed improvements — they're the
right things to *experiment with* for this specific goal, not a formula.

### Research backing this list (and its caveats)

- **Ghosh, Neufeld & Sahoo (2021)**, ["Forecasting directional movements of stock prices for intraday trading using LSTM and random forests"](https://arxiv.org/abs/2004.10178), *Finance Research Letters*. Backtested on all S&P 500 constituents, 1993–2018, on a multi-feature intraday-return classification task (the same shape of problem `strategies/sklearn_models.py` targets). Reported daily returns: **LSTM 0.64%/day, random forest 0.54%/day before transaction costs; 0.444%/day and 0.339%/day respectively after a 0.2%/day slippage assumption.** Two useful takeaways: (1) LSTM beat random forest, but by a real-not-huge margin — don't expect an order-of-magnitude jump from `random_forest` to a future `lstm` strategy; (2) transaction costs roughly cut returns by a third to a half — always evaluate a strategy after modeling costs, not before.
- **Théate & Ernst (2021)**, ["An Application of Deep Reinforcement Learning to Algorithmic Trading"](https://arxiv.org/abs/2004.06627), *Expert Systems with Applications*. Introduces TDQN, a DQN variant trained to directly maximize Sharpe ratio. The authors' own contribution is partly a *critique* of how prior RL-trading papers evaluated themselves too loosely — worth reading before trusting any single reported Sharpe ratio in this space, including their own.
- **Yang, Liu, Chen & Wang (2020)**, ["Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy"](https://openfin.engineering.columbia.edu/sites/default/files/content/publications/ensemble.pdf) (Columbia/FinRL). Trains PPO, A2C, and DDPG agents and dynamically switches to whichever had the best recent Sharpe ratio; the ensemble beat any single algorithm and buy-and-hold in their backtest. This "keep several models, pick/compare, don't marry one" idea is exactly what the registry pattern here is for.
- **Bailey & López de Prado (2014)**, ["The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551). The load-bearing caveat for this entire section: backtested performance metrics get inflated by selection bias (trying many models/seeds/hyperparameters and reporting the best) and non-normal returns. Any strategy shortlisted here — ML or RL especially — should be judged on a deflated/adjusted Sharpe ratio in backtesting (section 3 territory), not a raw one, and not on the single best run out of many.
- For contrast: **Fama (1970)**, "Efficient Capital Markets: A Review of Theory and Empirical Work," *Journal of Finance* — the classical argument that consistently beating the market via public information (which is all any of the above uses) shouldn't be possible. Doesn't mean don't try; means hold every backtest result here to a high bar of skepticism, especially pre-cost, in-sample ones.

**Bottom line:** start with `sma_crossover`/`rsi_mean_reversion` to validate the plumbing end-to-end, then `random_forest` as the best-evidenced classical-ML starting point, and treat `lstm_three_classifier`/RL as later upgrades with realistically modest expected gains over it — not the other way around.
