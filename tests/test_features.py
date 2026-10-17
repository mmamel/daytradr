"""
Run with: python -m unittest tests.test_features

Verifies the trend-strength features added to strategies/features.py
actually measure what they claim to, using synthetic price paths with a
known shape (a clean straight-line trend vs. flat noise) rather than real
market data, so the expected answer is unambiguous.
"""
from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd

    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False


def _bars(close, datetime_ms=None):
    import pandas as pd

    n = len(close)
    df = pd.DataFrame(
        {
            "open": close, "high": [c * 1.001 for c in close], "low": [c * 0.999 for c in close],
            "close": close, "volume": [1000] * n,
        }
    )
    if datetime_ms is not None:
        df["datetime"] = datetime_ms
    return df


@unittest.skipUnless(_HAS_DEPS, "pandas/numpy not installed")
class TestTrendFeatures(unittest.TestCase):
    def test_clean_uptrend_has_high_r2_and_positive_slope(self):
        from strategies.features import compute_features

        # Perfectly linear rise — a straight line should fit it almost exactly.
        close = list(np.linspace(100, 110, 60))
        features = compute_features(_bars(close))
        last = features.iloc[-1]

        self.assertGreater(last["trend_r2_20"], 0.99)
        self.assertGreater(last["trend_slope_20"], 0)
        self.assertGreater(last["trend_streak"], 0)  # should be on a long up-streak

    def test_clean_downtrend_has_high_r2_and_negative_slope(self):
        from strategies.features import compute_features

        close = list(np.linspace(110, 100, 60))
        features = compute_features(_bars(close))
        last = features.iloc[-1]

        self.assertGreater(last["trend_r2_20"], 0.99)
        self.assertLess(last["trend_slope_20"], 0)
        self.assertLess(last["trend_streak"], 0)

    def test_choppy_flat_series_has_low_r2(self):
        from strategies.features import compute_features

        # Oscillates around a flat mean with no net direction — a straight
        # line should fit this badly.
        rng = np.random.default_rng(0)
        close = 100 + np.sin(np.arange(60) * 1.5) * 2 + rng.normal(0, 0.05, 60)
        features = compute_features(_bars(list(close)))
        last = features.iloc[-1]

        self.assertLess(last["trend_r2_20"], 0.5)

    def test_adx_higher_for_clean_trend_than_for_choppy_series(self):
        from strategies.features import compute_features

        trending = list(np.linspace(100, 130, 60))
        rng = np.random.default_rng(0)
        choppy = list(100 + rng.normal(0, 0.3, 60))

        trend_adx = compute_features(_bars(trending)).iloc[-1]["adx_14"]
        choppy_adx = compute_features(_bars(choppy)).iloc[-1]["adx_14"]

        self.assertGreater(trend_adx, choppy_adx)

    def test_signed_streak_direction_and_magnitude(self):
        from strategies.features import compute_features

        # Up, up, up, down, down — streak should read +1, +2, +3, -1, -2
        close = [100, 101, 102, 103, 102, 101]
        features = compute_features(_bars(close))
        streaks = features["trend_streak"].tolist()

        self.assertEqual(streaks[1:], [1, 2, 3, -1, -2])

    def test_all_feature_columns_present_after_warmup(self):
        from strategies.features import FEATURE_COLUMNS, compute_features

        close = list(np.linspace(100, 105, 60))
        features = compute_features(_bars(close))
        for col in FEATURE_COLUMNS:
            self.assertIn(col, features.columns)
            self.assertFalse(np.isnan(features.iloc[-1][col]), f"{col} is NaN after 60 bars of warm-up")


@unittest.skipUnless(_HAS_DEPS, "pandas/numpy not installed")
class TestVwapDailyReset(unittest.TestCase):
    def test_vwap_resets_at_day_boundary(self):
        from strategies.features import compute_features

        day_ms = 24 * 60 * 60 * 1000
        # Day 1: flat at 100. Day 2: flat at 200. Without a daily reset,
        # day 2's cumulative VWAP would still be dragged toward ~100+ by
        # day 1's history, so close (200) would show a huge deviation.
        # With a reset, day 2's VWAP should track ~200 almost immediately.
        close = [100.0] * 30 + [200.0] * 5
        datetime_ms = [i * 60_000 for i in range(30)] + [day_ms + i * 60_000 for i in range(5)]
        features = compute_features(_bars(close, datetime_ms=datetime_ms))

        last_deviation = features.iloc[-1]["vwap_deviation"]
        self.assertLess(abs(last_deviation), 0.01, "VWAP should have reset for day 2, not stayed anchored near day 1's price")

    def test_falls_back_to_cumulative_without_a_datetime_column(self):
        from strategies.features import compute_features

        close = [100.0] * 30 + [200.0] * 5
        features = compute_features(_bars(close))  # no datetime column
        # Just shouldn't raise, and should still produce a value (the old behavior).
        self.assertFalse(pd.isna(features.iloc[-1]["vwap_deviation"]))


if __name__ == "__main__":
    unittest.main()
