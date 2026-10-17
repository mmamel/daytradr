"""
Run with: python -m unittest tests.test_strategies

These tests double as documentation of the swap-a-strategy workflow: every
strategy is looked up by name from the registry and driven through the same
`predict(features) -> Prediction` call, regardless of whether it's a
three-line rule or a trained classifier.
"""
from __future__ import annotations

import unittest

from strategies import Signal, available_strategies, get_strategy


_HORIZON_NAMES = {"1min", "5min", "10min", "30min", "1h", "2h"}


def _synthetic_bars(n: int = 300):
    """A synthetic random-walk OHLCV DataFrame — enough for any trainable
    strategy's fit() to run against without needing a real Schwab pull."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    close = 100 * np.cumprod(1 + rng.normal(loc=0.0, scale=0.001, size=n))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.0005, size=n)),
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.integers(1000, 5000, size=n),
        }
    )


class TestRegistry(unittest.TestCase):
    def test_baseline_strategies_are_registered(self):
        names = available_strategies()
        for expected in ("sma_crossover", "rsi_mean_reversion"):
            self.assertIn(expected, names)

    def test_unknown_strategy_raises_clear_error(self):
        with self.assertRaises(KeyError):
            get_strategy("not_a_real_strategy")


class TestSmaCrossover(unittest.TestCase):
    def setUp(self):
        self.strategy = get_strategy("sma_crossover")

    def test_uptrend_signals_buy(self):
        prediction = self.strategy.predict({"sma_5": 105.0, "sma_20": 100.0})
        self.assertEqual(prediction.signal, Signal.BUY)

    def test_downtrend_signals_sell(self):
        prediction = self.strategy.predict({"sma_5": 95.0, "sma_20": 100.0})
        self.assertEqual(prediction.signal, Signal.SELL)

    def test_missing_features_hold_instead_of_crashing(self):
        prediction = self.strategy.predict({})
        self.assertEqual(prediction.signal, Signal.HOLD)
        self.assertEqual(prediction.confidence, 0.0)

    def test_nan_features_hold_instead_of_crashing(self):
        prediction = self.strategy.predict({"sma_5": float("nan"), "sma_20": 100.0})
        self.assertEqual(prediction.signal, Signal.HOLD)


class TestRsiMeanReversion(unittest.TestCase):
    def setUp(self):
        self.strategy = get_strategy("rsi_mean_reversion")

    def test_oversold_signals_buy(self):
        prediction = self.strategy.predict({"rsi_14": 20.0})
        self.assertEqual(prediction.signal, Signal.BUY)

    def test_overbought_signals_sell(self):
        prediction = self.strategy.predict({"rsi_14": 80.0})
        self.assertEqual(prediction.signal, Signal.SELL)

    def test_neutral_holds(self):
        prediction = self.strategy.predict({"rsi_14": 50.0})
        self.assertEqual(prediction.signal, Signal.HOLD)


class TestSwappingStrategiesIsUniform(unittest.TestCase):
    """The whole point of the registry: every strategy, ML or not, is driven
    through the exact same call — this is what makes swapping them a
    one-line config change instead of a code change."""

    def test_every_rule_based_strategy_handles_the_same_call_shape(self):
        features = {"sma_5": 101.0, "sma_20": 100.0, "rsi_14": 55.0}
        for name in ("sma_crossover", "rsi_mean_reversion"):
            strategy = get_strategy(name)
            prediction = strategy.predict(features)
            self.assertIn(prediction.signal, Signal)
            self.assertTrue(0.0 <= prediction.confidence <= 1.0)


class TestSklearnStrategies(unittest.TestCase):
    """Requires scikit-learn (and pandas); skipped automatically if either isn't installed."""

    def setUp(self):
        try:
            import sklearn  # noqa: F401
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn/pandas not installed")

    def test_unfitted_model_holds_instead_of_guessing(self):
        strategy = get_strategy("random_forest")
        prediction = strategy.predict({"sma_5": 101.0})
        self.assertEqual(prediction.signal, Signal.HOLD)
        self.assertIn("reason", prediction.metadata)

    def test_fit_then_predict_roundtrip(self):
        from strategies.features import compute_features

        strategy = get_strategy("logistic_regression")
        bars = _synthetic_bars(300)
        strategy.fit(bars)  # same fit(bars) contract as every other trainable strategy

        latest_row = compute_features(bars).iloc[-1].to_dict()
        prediction = strategy.predict(latest_row)
        self.assertIn(prediction.signal, Signal)
        self.assertIn("BUY", prediction.metadata["probabilities"])

    def test_save_and_load_roundtrip(self):
        import os
        import tempfile

        from strategies.sklearn_models import RandomForestStrategy

        strategy = get_strategy("random_forest")
        strategy.fit(_synthetic_bars(300))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pkl")
            strategy.save(path)
            loaded = RandomForestStrategy.load(path)
        self.assertTrue(loaded._is_fitted)


class TestRandomForestMultiHorizon(unittest.TestCase):
    """The sklearn counterpart to TestLstmStrategy below — same interface,
    same horizons, so the two are directly comparable via modes/train_mode.py."""

    def setUp(self):
        try:
            import sklearn  # noqa: F401
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn/pandas not installed")

    def test_predict_before_fit_holds_every_horizon(self):
        from strategies.features import FEATURE_COLUMNS

        strategy = get_strategy("random_forest_multi_horizon")
        prediction = strategy.predict({col: 1.0 for col in FEATURE_COLUMNS})
        self.assertEqual(prediction.signal, Signal.HOLD)
        self.assertEqual(set(prediction.metadata["horizons"]), _HORIZON_NAMES)

    def test_fit_then_predict_produces_a_signal_per_horizon(self):
        from strategies.features import compute_features

        strategy = get_strategy("random_forest_multi_horizon")
        bars = _synthetic_bars(300)
        strategy.fit(bars)

        latest_row = compute_features(bars).iloc[-1].to_dict()
        prediction = strategy.predict(latest_row)
        self.assertIn(prediction.signal, Signal)

        all_horizons = strategy.predict_all_horizons(latest_row)
        self.assertEqual(set(all_horizons), _HORIZON_NAMES)

    def test_uses_same_default_horizons_as_the_lstm(self):
        from strategies.deep_learning import HORIZONS as LSTM_HORIZONS

        strategy = get_strategy("random_forest_multi_horizon")
        self.assertEqual(strategy.horizons, LSTM_HORIZONS)

    def test_save_and_load_roundtrip(self):
        import os
        import tempfile

        from strategies.sklearn_models import RandomForestMultiHorizonStrategy

        strategy = get_strategy("random_forest_multi_horizon")
        strategy.fit(_synthetic_bars(300))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pkl")
            strategy.save(path)
            loaded = RandomForestMultiHorizonStrategy.load(path)
        self.assertTrue(loaded._is_fitted)
        self.assertEqual(set(loaded.horizons), _HORIZON_NAMES)


class TestLstmStrategy(unittest.TestCase):
    """Requires torch (and pandas); skipped automatically if either isn't installed."""

    HORIZON_NAMES = _HORIZON_NAMES

    def setUp(self):
        try:
            import torch  # noqa: F401
            import pandas  # noqa: F401
        except ImportError:
            self.skipTest("torch/pandas not installed")

    def test_predict_before_fit_holds_every_horizon(self):
        from strategies.features import FEATURE_COLUMNS

        strategy = get_strategy("lstm_three_classifier", sequence_length=5, epochs=1)
        prediction = strategy.predict({col: 1.0 for col in FEATURE_COLUMNS})
        self.assertEqual(prediction.signal, Signal.HOLD)
        self.assertEqual(set(prediction.metadata["horizons"]), self.HORIZON_NAMES)

    def test_fit_then_predict_produces_a_signal_per_horizon(self):
        from strategies.features import compute_features

        strategy = get_strategy("lstm_three_classifier", sequence_length=10, epochs=1)
        strategy.fit(_synthetic_bars(300))

        feature_rows = compute_features(_synthetic_bars(300)).to_dict("records")
        for row in feature_rows[-15:]:  # fill the rolling buffer past sequence_length
            prediction = strategy.predict(row)

        self.assertIn(prediction.signal, Signal)
        all_horizons = strategy.predict_all_horizons(feature_rows[-1])
        self.assertEqual(set(all_horizons), self.HORIZON_NAMES)

    def test_save_and_load_roundtrip(self):
        import os
        import tempfile

        from strategies.deep_learning import LstmMultiHorizonStrategy

        strategy = get_strategy("lstm_three_classifier", sequence_length=10, epochs=1)
        strategy.fit(_synthetic_bars(300))

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.pt")
            strategy.save(path)
            loaded = LstmMultiHorizonStrategy.load(path)
        self.assertTrue(loaded._is_fitted)


if __name__ == "__main__":
    unittest.main()
