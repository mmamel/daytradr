"""
Train mode: fetch historical bars for one symbol, train a strategy on all
but the most recent N trading days, then evaluate it on those held-out
days — strategy-agnostic, thanks to every trainable strategy sharing the
same `fit(bars)` contract (see strategies/base.py).

Usage:
    python -m modes.train_mode --symbol AAPL --strategy random_forest \\
        --days-back 45 --validation-days 5 --out models/random_forest_aapl.pkl

    # or, equivalently, through the main CLI:
    python main.py --mode train --symbol AAPL --strategy random_forest \\
        --days-back 45 --validation-days 5 --out models/random_forest_aapl.pkl

Parameters, and why each one is there:
    --symbol            Which ticker to train on. One symbol per run —
                         see README.md's cross-sectional-ranking note if you
                         eventually want to train across a whole watchlist.
    --minutes-per-bar   Bar resolution (default 1). Interacts with
                         --days-back: Schwab only keeps ~48 days of 1-minute
                         bars vs. ~9 months for 5/10/15/30-minute bars (see
                         schwab_client/market_data.py). Asking for more
                         --days-back than that at 1-minute resolution won't
                         error — Schwab just returns what it has — but
                         you'll silently get less data than you asked for,
                         so this is validated and warned about up front.
    --days-back         How much history to fetch, ending now.
    --validation-days   The most recent N *trading* days (not a fraction of
                         windows) are held out for evaluation; the rest are
                         used for training. Splitting by calendar day rather
                         than by row count avoids slicing a day in half, and
                         "held out the last week" is a more useful mental
                         model than "held out the last 15% of rows."
    --strategy          Any name from strategies.available_strategies().
                         Rule-based strategies (sma_crossover,
                         rsi_mean_reversion) work too — fit() is a no-op for
                         them, but running them through the same evaluation
                         gives you the baseline every ML model here is
                         supposed to beat (see README.md#ml-model-options).
                         For a head-to-head comparison against
                         lstm_three_classifier, use random_forest_multi_horizon
                         instead of random_forest — it predicts the same 6
                         horizons on the same feature set, just with N
                         independent trees-based models instead of one
                         shared encoder.
    --label-threshold   What counts as BUY/SELL vs. HOLD noise (see
                         strategies.features.make_labels). Forwarded to the
                         strategy if it accepts it; ignored (with a log
                         message) otherwise.
    --horizon-minutes   Comma-separated minutes-ahead to predict, e.g.
                         "30,60" for just 30min/1h instead of the default
                         six. Converted to bars using --minutes-per-bar (so
                         "30" at --minutes-per-bar 5 becomes 6 bars ahead) —
                         errors clearly if a value isn't a whole number of
                         bars. Ignored (with a log message) for strategies
                         that don't accept it (i.e. it needs a multi-horizon
                         strategy, or exactly one value for a single-horizon
                         one like random_forest, which takes horizon_bars).
    --strategy-kwargs   A JSON object of any other constructor kwargs to
                         forward, e.g. '{"sequence_length": 24}' for the
                         LSTM. Anything a given strategy doesn't declare in
                         its constructor is dropped with a log message
                         rather than erroring, same as --label-threshold.
                         --label-threshold/--horizon-minutes take priority
                         over the same keys here if both are given.
    --out               Optional path to save the trained strategy
                         (.save()) for reuse by trading_mode.py without
                         retraining. Skipped, with a log message, for
                         strategies that don't implement save() (i.e. the
                         rule-based ones — they have no state to save).
    --report-out        Optional path to save the validation results (the
                         per-horizon accuracy/recall/baseline lines) as a
                         plain text file — handy for comparing runs later
                         without having to scroll back through terminal
                         history. Defaults to the same path as --out with a
                         .txt extension if --out is given and --report-out
                         isn't; otherwise not saved unless requested.
    --plot-out          Optional path to save a PNG chart: actual close
                         price over the validation window, with markers for
                         every BUY/SELL the strategy predicted (filled =
                         matched the actual outcome, hollow = didn't) — one
                         subplot per horizon. Needs matplotlib (not a hard
                         dependency of this file; only imported if you ask
                         for a plot). Defaults from --out the same way
                         --report-out does.
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from schwab_client.auth import get_client
from schwab_client.market_data import MarketDataClient
from strategies import StrategyBase, available_strategies, get_strategy, get_strategy_class
from strategies.base import LABEL_TO_SIGNAL
from strategies.features import compute_features, make_labels

logger = logging.getLogger(__name__)

# Approximate — see schwab_client/market_data.py. Not a hard cutoff (Schwab
# just returns less than asked for beyond this), so we warn, not error.
_MAX_DAYS_BACK = {1: 48, 5: 270, 10: 270, 15: 270, 30: 270}


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.strategy not in available_strategies():
        raise SystemExit(f"Unknown strategy '{args.strategy}'. Available: {available_strategies()}")

    if args.days_back > _MAX_DAYS_BACK[args.minutes_per_bar]:
        logger.warning(
            "--days-back %d exceeds Schwab's ~%d-day limit for %d-minute bars — "
            "you'll get less history than requested.",
            args.days_back, _MAX_DAYS_BACK[args.minutes_per_bar], args.minutes_per_bar,
        )

    logger.info("Fetching %d day(s) of %d-minute bars for %s...", args.days_back, args.minutes_per_bar, args.symbol)
    t0 = time.monotonic()
    bars = _fetch_bars(args.symbol, args.minutes_per_bar, args.days_back)
    logger.info(
        "Fetched %d bars for %s spanning %d trading day(s) (%.1fs)",
        len(bars), args.symbol, bars["_date"].nunique(), time.monotonic() - t0,
    )

    train_bars, validation_dates = _split_by_trading_days(bars, args.validation_days)
    logger.info(
        "Training on %d bars (%d trading days), validating on %d trading day(s): %s",
        len(train_bars), train_bars["_date"].nunique(), len(validation_dates), sorted(validation_dates),
    )

    strategy = _build_strategy(
        args.strategy, args.label_threshold, args.horizon_minutes, args.minutes_per_bar, args.strategy_kwargs
    )
    logger.info("Fitting '%s' on %d bars...", args.strategy, len(train_bars))
    t0 = time.monotonic()
    strategy.fit(train_bars)
    logger.info("fit() complete for '%s' (%.1fs)", args.strategy, time.monotonic() - t0)

    t0 = time.monotonic()
    report_lines, plot_data = _evaluate_and_report(strategy, bars, validation_dates, args.label_threshold)
    eval_seconds = time.monotonic() - t0
    logger.info("Evaluation complete (%.1fs)", eval_seconds)

    report_out = args.report_out or (str(Path(args.out).with_suffix(".txt")) if args.out else None)
    if report_out:
        header = [
            f"Training report — {datetime.now(tz=timezone.utc).isoformat()}",
            f"symbol={args.symbol} strategy={args.strategy} minutes_per_bar={args.minutes_per_bar}",
            f"days_back={args.days_back} validation_days={args.validation_days} label_threshold={args.label_threshold}",
            f"horizon_minutes={args.horizon_minutes} strategy_kwargs={args.strategy_kwargs}",
            "",
        ]
        footer = [f"\nEvaluation complete ({eval_seconds:.1f}s)"]
        Path(report_out).parent.mkdir(parents=True, exist_ok=True)
        Path(report_out).write_text("\n".join(header + report_lines + footer) + "\n")
        logger.info("Saved validation report to %s", report_out)

    plot_out = args.plot_out or (str(Path(args.out).with_suffix(".png")) if args.out else None)
    if plot_out:
        _plot_validation(plot_data, args.symbol, plot_out)

    if args.out:
        if hasattr(strategy, "save"):
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            strategy.save(args.out)
            logger.info("Saved trained model to %s", args.out)
        else:
            logger.info("'%s' has no state to save (rule-based strategy) — skipping --out.", args.strategy)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", required=True, help="e.g. AAPL")
    parser.add_argument("--minutes-per-bar", type=int, default=1, choices=[1, 5, 10, 15, 30])
    parser.add_argument("--days-back", type=int, default=45, help="How much history to fetch, ending now")
    parser.add_argument(
        "--validation-days", type=int, default=5,
        help="Most recent N trading days held out for evaluation",
    )
    parser.add_argument("--strategy", required=True, help="Name from strategies.available_strategies()")
    parser.add_argument("--label-threshold", type=float, default=0.001, help="BUY/SELL vs HOLD return threshold")
    parser.add_argument(
        "--horizon-minutes", type=_parse_int_list, default=None,
        help='Comma-separated minutes-ahead to predict, e.g. "30,60". Converted to bars via --minutes-per-bar.',
    )
    parser.add_argument(
        "--strategy-kwargs", type=json.loads, default={},
        help='JSON object of extra constructor kwargs, e.g. \'{"sequence_length": 24}\'.',
    )
    parser.add_argument("--out", default=None, help="Where to save the trained model, e.g. models/random_forest_aapl.pkl")
    parser.add_argument(
        "--report-out", default=None,
        help="Where to save the validation report as text. Defaults to --out with a .txt extension if --out is given.",
    )
    parser.add_argument(
        "--plot-out", default=None,
        help="Where to save a price-vs-predictions PNG chart (needs matplotlib). "
        "Defaults to --out with a .png extension if --out is given.",
    )
    return parser.parse_args(argv)


def _fetch_bars(symbol: str, minutes_per_bar: int, days_back: int) -> pd.DataFrame:
    market_data = MarketDataClient(client=get_client())
    start = datetime.now(tz=timezone.utc) - timedelta(days=days_back)
    candles = market_data.get_intraday_bars(symbol, minutes_per_bar=minutes_per_bar, start_datetime=start)
    if not candles:
        raise SystemExit(f"Schwab returned no bars for {symbol} — check the symbol and date range.")

    bars = pd.DataFrame(candles).sort_values("datetime").reset_index(drop=True)
    bars["_date"] = pd.to_datetime(bars["datetime"], unit="ms").dt.date
    return bars


def _split_by_trading_days(bars: pd.DataFrame, validation_days: int) -> tuple[pd.DataFrame, set]:
    unique_days = sorted(bars["_date"].unique())
    if validation_days >= len(unique_days):
        raise SystemExit(
            f"--validation-days ({validation_days}) must be less than the number of trading days "
            f"fetched ({len(unique_days)}). Increase --days-back or lower --validation-days."
        )
    validation_dates = set(unique_days[-validation_days:])
    train_bars = bars[~bars["_date"].isin(validation_dates)].reset_index(drop=True)
    return train_bars, validation_dates


def _parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",")]


def _horizon_kwargs(horizon_minutes: list[int] | None, minutes_per_bar: int) -> dict:
    """
    Converts --horizon-minutes into whatever a strategy's constructor might
    call it. Multi-horizon strategies (LstmMultiHorizonStrategy,
    RandomForestMultiHorizonStrategy) declare `horizons: dict[str, int]`;
    single-horizon ones (RandomForestStrategy, etc.) declare `horizon_bars:
    int`. Both candidate keys are produced here; _build_strategy's generic
    filtering picks whichever the target strategy actually accepts.

    Also sets `primary_horizon` to the first horizon given — both
    multi-horizon classes default that to "1min", which no longer exists
    once `horizons` is overridden to something that doesn't include it
    (their own constructors correctly reject that combination), so this
    has to be kept in sync whenever `horizons` is.
    """
    if not horizon_minutes:
        return {}

    bars_ahead = []
    for minutes in horizon_minutes:
        if minutes % minutes_per_bar != 0:
            raise SystemExit(
                f"--horizon-minutes {minutes} isn't a whole number of {minutes_per_bar}-minute bars "
                f"({minutes} / {minutes_per_bar} = {minutes / minutes_per_bar})."
            )
        bars_ahead.append(minutes // minutes_per_bar)

    kwargs = {
        "horizons": {f"{m}min": b for m, b in zip(horizon_minutes, bars_ahead)},
        "primary_horizon": f"{horizon_minutes[0]}min",
    }
    if len(horizon_minutes) == 1:
        kwargs["horizon_bars"] = bars_ahead[0]
    return kwargs


def _build_strategy(
    name: str,
    label_threshold: float,
    horizon_minutes: list[int] | None = None,
    minutes_per_bar: int = 1,
    strategy_kwargs: dict | None = None,
) -> StrategyBase:
    """
    Builds the requested strategy, forwarding only the constructor kwargs
    it actually declares (checked via inspect.signature rather than a
    try/except per kwarg, so this doesn't need updating every time a new
    strategy hyperparameter gets added). Anything not accepted is dropped
    with a log message instead of raising — e.g. --horizon-minutes with
    sma_crossover, which has no concept of a horizon at all.
    """
    candidate_kwargs = {
        **(strategy_kwargs or {}),
        "label_threshold": label_threshold,
        **_horizon_kwargs(horizon_minutes, minutes_per_bar),
    }

    cls = get_strategy_class(name)
    accepted_params = inspect.signature(cls.__init__).parameters
    supported = {k: v for k, v in candidate_kwargs.items() if k in accepted_params}
    dropped = set(candidate_kwargs) - set(supported)
    if dropped:
        logger.info("'%s' doesn't accept %s — using its own defaults for those.", name, sorted(dropped))

    if horizon_minutes and "horizons" not in accepted_params and "horizon_bars" not in accepted_params:
        logger.warning("--horizon-minutes was given but '%s' has no horizon concept at all — ignoring it.", name)
    elif horizon_minutes and "horizons" not in supported and "horizon_bars" not in supported:
        logger.warning(
            "'%s' only supports a single horizon (horizon_bars), but --horizon-minutes gave %d values "
            "— using its own default horizon instead.", name, len(horizon_minutes),
        )

    return get_strategy(name, **supported)


def _strategy_horizons(strategy: StrategyBase) -> dict[str, int]:
    """{horizon_name: bars_ahead} this strategy predicts, for evaluation."""
    if hasattr(strategy, "horizons"):  # multi-horizon, e.g. lstm_three_classifier
        return dict(strategy.horizons)
    horizon_bars = getattr(strategy, "horizon_bars", 1)  # sklearn default; 1 for rule-based baselines
    return {f"{horizon_bars}bar": horizon_bars}


def _evaluate_and_report(
    strategy: StrategyBase, bars: pd.DataFrame, validation_dates: set, label_threshold: float
) -> tuple[list[str], dict]:
    """
    Feeds every bar (train period first, then validation) through the
    strategy in chronological order — sequence-based strategies like the
    LSTM need that to warm their internal buffer — but only scores
    predictions made during the validation period.

    Returns:
      - the "Validation results" lines (logged as they're produced, same as
        before) so `main()` can also save them to a text file via
        --report-out without duplicating the formatting logic.
      - per-horizon time series (datetime/close/predicted/actual signal) of
        every validation-window row, for --plot-out (see _plot_validation).
    """
    horizons = _strategy_horizons(strategy)
    feature_rows = compute_features(bars).to_dict("records")
    close = bars["close"].to_numpy(dtype=float)
    true_labels = {name: make_labels(close, bars_ahead, label_threshold) for name, bars_ahead in horizons.items()}
    predict_all_horizons = getattr(strategy, "predict_all_horizons", None)

    stats = {
        name: {"correct": 0, "total": 0, "predicted": Counter(), "actual": Counter(), "correct_by_class": Counter()}
        for name in horizons
    }
    plot_data = {name: {"datetime_ms": [], "close": [], "predicted": [], "actual": []} for name in horizons}

    # Only sequence-based strategies (the LSTM) need to see bars before the
    # validation window at all, to warm their internal rolling buffer —
    # everything else (sklearn, rule-based) is stateless per-row, so
    # feeding it the entire training history here just to throw the
    # predictions away was pure wasted work (and, with a few hundred trees
    # and n_jobs=-1, was the actual cause of this looking "hung": tens of
    # thousands of single-row predict() calls, each paying multiprocessing
    # coordination overhead for almost no actual computation).
    validation_start_index = next(i for i, row in enumerate(feature_rows) if row["_date"] in validation_dates)
    warmup_rows = getattr(strategy, "sequence_length", 0)
    start_index = max(0, validation_start_index - warmup_rows)
    total_rows = len(feature_rows) - start_index

    logger.info(
        "Evaluating %d row(s) (%d warm-up + %d validation)...",
        total_rows, validation_start_index - start_index, len(feature_rows) - validation_start_index,
    )
    last_logged_pct = -1

    for i in range(start_index, len(feature_rows)):
        row = feature_rows[i]
        predictions = predict_all_horizons(row) if predict_all_horizons else {n: strategy.predict(row) for n in horizons}

        pct = (i - start_index + 1) * 100 // total_rows
        if pct >= last_logged_pct + 10:
            logger.info("  ...%d%% (%d/%d rows)", pct, i - start_index + 1, total_rows)
            last_logged_pct = pct

        if row["_date"] not in validation_dates:
            continue  # only here to warm the buffer (see comment above)

        for name, bars_ahead in horizons.items():
            if i >= len(close) - bars_ahead:
                continue  # future return not known this close to the end of the fetched data
            actual_signal = LABEL_TO_SIGNAL[int(true_labels[name][i])]
            predicted_signal = predictions[name].signal

            is_correct = predicted_signal == actual_signal
            s = stats[name]
            s["total"] += 1
            s["correct"] += int(is_correct)
            s["predicted"][predicted_signal.value] += 1
            s["actual"][actual_signal.value] += 1
            s["correct_by_class"][actual_signal.value] += int(is_correct)

            p = plot_data[name]
            p["datetime_ms"].append(row["datetime"])
            p["close"].append(row["close"])
            p["predicted"].append(predicted_signal.value)
            p["actual"].append(actual_signal.value)

    report_lines: list[str] = []

    def emit(line: str) -> None:
        logger.info(line)
        report_lines.append(line)

    emit(f"Validation results ({len(validation_dates)} day(s) held out): {sorted(validation_dates)}")
    for name, s in stats.items():
        if not s["total"]:
            emit(f"  {name:<8} no scoreable rows (validation window too short for this horizon)")
            continue

        accuracy = s["correct"] / s["total"]
        # The naive baseline: always predicting whichever class was most
        # common in the actual validation labels. Accuracy at or near this
        # number — regardless of how high it looks — means the model
        # learned nothing beyond the label imbalance (see the class-weighting
        # fix in strategies/sklearn_models.py and strategies/deep_learning.py
        # for *why* this happens and how it's addressed).
        baseline_class, baseline_count = s["actual"].most_common(1)[0]
        baseline_accuracy = baseline_count / s["total"]
        recall = {
            cls: s["correct_by_class"][cls] / count if count else float("nan") for cls, count in s["actual"].items()
        }

        flag = "  <-- barely beats always-predicting-the-majority-class!" if accuracy <= baseline_accuracy + 0.01 else ""
        emit(
            f"  {name:<8} accuracy={accuracy * 100:.1f}% "
            f"(naive '{baseline_class} always' baseline: {baseline_accuracy * 100:.1f}%){flag}"
        )
        recall_str = {k: f"{v:.0%}" for k, v in recall.items()}
        emit(f"  {'':<8} recall by class={recall_str}  predicted={dict(s['predicted'])}  actual={dict(s['actual'])}")

    return report_lines, plot_data


def _plot_validation(plot_data: dict, symbol: str, path: str) -> list[str]:
    """
    One PNG per horizon (not one crowded multi-subplot figure) — the actual
    close price over the validation window, with a marker at every bar the
    strategy predicted BUY (^) or SELL (v) — filled if that call matched
    the actual outcome, hollow if it didn't. HOLD isn't marked (it's the
    majority of bars and would just be visual noise).

    The x-axis is plotted against *bar sequence*, not real elapsed time:
    with real timestamps, every overnight/weekend gap (no trading = no
    bars, but still real elapsed hours) stretches the axis and squeezes the
    actual trading data into unreadable clumps. Plotting bar-by-bar instead
    keeps every bar equally spaced regardless of the real-time gap between
    it and the previous one, and the x-tick labels still show the real
    date/time so you don't lose that information.

    Returns the list of saved file paths.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe — no display needed to save a PNG
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib isn't installed — skipping --plot-out. Run `pip install matplotlib` to enable it.")
        return []

    base = Path(path)
    saved_paths = []

    for name, data in plot_data.items():
        n = len(data["close"])
        # Wide enough to scroll, not so wide it hits the Agg backend's
        # ~65536px-per-dimension limit: 0.08in/bar, capped at 350in — at the
        # 120 dpi used below that's a max of 350*120 = 42000px, comfortably
        # under the ceiling even for a long validation window.
        width_inches = min(max(16, n * 0.08), 350)
        fig, ax = plt.subplots(figsize=(width_inches, 6))

        if n == 0:
            ax.set_title(f"{symbol} — {name} (no scoreable rows)")
        else:
            times = pd.to_datetime(data["datetime_ms"], unit="ms")
            x = range(n)
            ax.plot(x, data["close"], color="steelblue", linewidth=1, zorder=1, label="close")

            for signal, marker, color in (("BUY", "^", "green"), ("SELL", "v", "red")):
                for xi, price, predicted, actual in zip(x, data["close"], data["predicted"], data["actual"]):
                    if predicted != signal:
                        continue
                    correct = predicted == actual
                    ax.scatter(
                        xi, price, marker=marker, s=70, zorder=2,
                        facecolors=color if correct else "none", edgecolors=color, linewidths=1.5,
                    )

            # More ticks than before now that there's much more horizontal
            # room to fit them legibly — roughly one every ~50 bars, capped
            # so it doesn't get silly for very short validation windows.
            tick_count = min(max(10, n // 50), n)
            step = (n - 1) / (tick_count - 1) if tick_count > 1 else 0
            tick_positions = sorted({round(i * step) for i in range(tick_count)})
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([times[i].strftime("%m/%d %H:%M") for i in tick_positions], rotation=45, ha="right")
            ax.set_title(f"{symbol} — {name} (filled marker = correct call, hollow = incorrect)")
            ax.set_ylabel("price")

        fig.tight_layout()
        horizon_path = base.with_name(f"{base.stem}_{name}{base.suffix}")
        horizon_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(horizon_path, dpi=120)
        plt.close(fig)
        saved_paths.append(str(horizon_path))

    logger.info("Saved %d validation plot(s): %s", len(saved_paths), ", ".join(saved_paths))
    return saved_paths


if __name__ == "__main__":
    main()
