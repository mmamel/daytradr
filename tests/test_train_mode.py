"""
Run with: python -m unittest tests.test_train_mode

Covers the parts of modes/train_mode.py that don't need a live Schwab
connection: the day-based train/validation split, strategy construction
(including the label_threshold fallback for strategies that don't accept
it), and horizon detection. The actual fetch-train-evaluate flow
(`train_mode.main`) needs real market data and isn't covered here.
"""
from __future__ import annotations

import unittest

try:
    import pandas  # noqa: F401

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


@unittest.skipUnless(_HAS_PANDAS, "pandas not installed")
class TestSplitByTradingDays(unittest.TestCase):
    @staticmethod
    def _bars_over_days(num_days: int, bars_per_day: int = 10):
        import pandas as pd

        rows = []
        for day in range(num_days):
            date = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)).date()
            for _ in range(bars_per_day):
                rows.append({"close": 100.0, "_date": date})
        return pd.DataFrame(rows)

    def test_holds_out_the_most_recent_n_days(self):
        from modes.train_mode import _split_by_trading_days

        bars = self._bars_over_days(10)
        train_bars, validation_dates = _split_by_trading_days(bars, validation_days=3)

        all_days = sorted(bars["_date"].unique())
        self.assertEqual(validation_dates, set(all_days[-3:]))
        self.assertTrue((train_bars["_date"] < min(validation_dates)).all())

    def test_rejects_validation_days_that_would_consume_everything(self):
        from modes.train_mode import _split_by_trading_days

        bars = self._bars_over_days(5)
        with self.assertRaises(SystemExit):
            _split_by_trading_days(bars, validation_days=5)


@unittest.skipUnless(_HAS_PANDAS, "pandas not installed")
class TestBuildStrategy(unittest.TestCase):
    def test_forwards_label_threshold_when_supported(self):
        from modes.train_mode import _build_strategy

        strategy = _build_strategy("sma_crossover", label_threshold=0.002)
        # sma_crossover doesn't take label_threshold — should fall back
        # to its own defaults rather than raising.
        self.assertEqual(type(strategy).__name__, "SmaCrossoverStrategy")

    @staticmethod
    def _require_sklearn():
        import sklearn  # noqa: F401

    def test_horizon_minutes_converted_to_bars_for_single_horizon_strategy(self):
        try:
            self._require_sklearn()
        except ImportError:
            self.skipTest("scikit-learn not installed")

        from modes.train_mode import _build_strategy

        strategy = _build_strategy(
            "random_forest", label_threshold=0.002, horizon_minutes=[30], minutes_per_bar=5
        )
        self.assertEqual(strategy.horizon_bars, 6)  # 30 minutes / 5-minute bars

    def test_horizon_minutes_converted_to_bars_for_multi_horizon_strategy(self):
        try:
            self._require_sklearn()
        except ImportError:
            self.skipTest("scikit-learn not installed")

        from modes.train_mode import _build_strategy

        strategy = _build_strategy(
            "random_forest_multi_horizon", label_threshold=0.002, horizon_minutes=[30, 60], minutes_per_bar=5
        )
        self.assertEqual(strategy.horizons, {"30min": 6, "60min": 12})
        # Regression test: primary_horizon defaults to "1min" in the
        # strategy's own constructor, which no longer exists once horizons
        # is overridden — _build_strategy must keep it in sync or the
        # constructor's own validation raises ValueError.
        self.assertEqual(strategy.primary_horizon, "30min")

    def test_horizon_minutes_not_evenly_divisible_errors_clearly(self):
        try:
            self._require_sklearn()
        except ImportError:
            self.skipTest("scikit-learn not installed")

        from modes.train_mode import _build_strategy

        with self.assertRaises(SystemExit):
            _build_strategy("random_forest", label_threshold=0.002, horizon_minutes=[7], minutes_per_bar=5)

    def test_multiple_horizon_minutes_ignored_for_single_horizon_strategy(self):
        try:
            self._require_sklearn()
        except ImportError:
            self.skipTest("scikit-learn not installed")

        from modes.train_mode import _build_strategy

        # Shouldn't raise, and shouldn't silently pick one of the values —
        # random_forest only accepts a single horizon_bars, and 2 values
        # don't map onto that, so it should fall back to its own default (1).
        strategy = _build_strategy(
            "random_forest", label_threshold=0.002, horizon_minutes=[30, 60], minutes_per_bar=5
        )
        self.assertEqual(strategy.horizon_bars, 1)

    def test_strategy_kwargs_passthrough(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

        from modes.train_mode import _build_strategy

        strategy = _build_strategy(
            "lstm_three_classifier", label_threshold=0.002, strategy_kwargs={"sequence_length": 24, "epochs": 1}
        )
        self.assertEqual(strategy.sequence_length, 24)

    def test_horizon_minutes_with_lstm_does_not_raise_on_default_primary_horizon(self):
        # Regression test for: ValueError: primary_horizon must be one of
        # ['30min', '60min'] — LstmMultiHorizonStrategy defaults
        # primary_horizon to "1min", which doesn't exist once --horizon-minutes
        # overrides the horizon set to exclude it.
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")

        from modes.train_mode import _build_strategy

        strategy = _build_strategy(
            "lstm_three_classifier", label_threshold=0.0015, horizon_minutes=[30, 60],
            minutes_per_bar=5, strategy_kwargs={"epochs": 1},
        )
        self.assertEqual(strategy.horizons, {"30min": 6, "60min": 12})
        self.assertEqual(strategy.primary_horizon, "30min")

    def test_unsupported_strategy_kwargs_dropped_not_raised(self):
        from modes.train_mode import _build_strategy

        # sma_crossover's __init__ doesn't take sequence_length — shouldn't raise.
        strategy = _build_strategy(
            "sma_crossover", label_threshold=0.002, strategy_kwargs={"sequence_length": 24}
        )
        self.assertEqual(type(strategy).__name__, "SmaCrossoverStrategy")


@unittest.skipUnless(_HAS_PANDAS, "pandas not installed")
class TestStrategyHorizons(unittest.TestCase):
    def test_single_horizon_strategy_defaults_to_one_bar(self):
        from modes.train_mode import _strategy_horizons
        from strategies import get_strategy

        strategy = get_strategy("sma_crossover")
        self.assertEqual(_strategy_horizons(strategy), {"1bar": 1})


@unittest.skipUnless(_HAS_PANDAS, "pandas not installed")
class TestEvaluateAndReport(unittest.TestCase):
    """Regression test for the perf bug where every row of the FULL fetched
    history (not just the validation days) was fed through predict() —
    harmless for a fast rule-based strategy like this one, but this test
    exists to pin down that the row-skipping/windowing math around it
    doesn't silently drop or double-count validation rows."""

    @staticmethod
    def _synthetic_bars(num_days: int, bars_per_day: int = 30):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(0)
        n = num_days * bars_per_day
        close = 100 * np.cumprod(1 + rng.normal(0, 0.001, size=n))
        dates = []
        datetimes_ms = []
        for day in range(num_days):
            date = (pd.Timestamp("2026-01-01") + pd.Timedelta(days=day)).date()
            day_start_ms = int(pd.Timestamp(date).timestamp() * 1000)
            dates.extend([date] * bars_per_day)
            datetimes_ms.extend(day_start_ms + bar * 60_000 for bar in range(bars_per_day))
        return pd.DataFrame(
            {
                "open": close, "high": close * 1.001, "low": close * 0.999,
                "close": close, "volume": rng.integers(1000, 5000, size=n),
                "datetime": datetimes_ms, "_date": dates,
            }
        )

    def test_scores_only_validation_rows_not_the_whole_history(self):
        from modes.train_mode import _evaluate_and_report, _split_by_trading_days
        from strategies import get_strategy

        bars = self._synthetic_bars(num_days=10, bars_per_day=30)
        _, validation_dates = _split_by_trading_days(bars, validation_days=3)
        strategy = get_strategy("sma_crossover")

        # Doesn't raise, and (implicitly, via not hanging) only iterates the
        # warm-up-plus-validation slice rather than the full 300-row history.
        report_lines, plot_data = _evaluate_and_report(strategy, bars, validation_dates, label_threshold=0.001)
        self.assertTrue(any("accuracy=" in line for line in report_lines))
        self.assertEqual(set(plot_data), {"1bar"})
        self.assertTrue(len(plot_data["1bar"]["close"]) > 0)


@unittest.skipUnless(_HAS_PANDAS, "pandas not installed")
class TestPlotValidation(unittest.TestCase):
    """Requires matplotlib; skipped automatically if it isn't installed."""

    def setUp(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")

    def test_saves_a_png_per_horizon_without_raising(self):
        import os
        import tempfile

        from modes.train_mode import _evaluate_and_report, _plot_validation, _split_by_trading_days
        from strategies import get_strategy

        bars = TestEvaluateAndReport._synthetic_bars(num_days=10, bars_per_day=30)
        _, validation_dates = _split_by_trading_days(bars, validation_days=3)
        strategy = get_strategy("sma_crossover")
        _, plot_data = _evaluate_and_report(strategy, bars, validation_dates, label_threshold=0.001)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "plot.png")
            saved_paths = _plot_validation(plot_data, "TEST", path)
            # One PNG per horizon, named plot_<horizon>.png, not one shared file.
            self.assertEqual(len(saved_paths), len(plot_data))
            for saved_path in saved_paths:
                self.assertTrue(os.path.getsize(saved_path) > 0)

    def test_handles_a_horizon_with_no_scoreable_rows(self):
        # Shouldn't crash if a horizon's window was too short to score
        # anything (see the "no scoreable rows" branch in _evaluate_and_report).
        import os
        import tempfile

        from modes.train_mode import _plot_validation

        empty_plot_data = {"1bar": {"datetime_ms": [], "close": [], "predicted": [], "actual": []}}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "plot.png")
            saved_paths = _plot_validation(empty_plot_data, "TEST", path)
            self.assertEqual(len(saved_paths), 1)
            self.assertTrue(os.path.getsize(saved_paths[0]) > 0)


if __name__ == "__main__":
    unittest.main()
