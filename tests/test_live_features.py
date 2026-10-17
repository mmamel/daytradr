"""
Run with: python -m unittest tests.test_live_features

Pins down the CHART_EQUITY message shape (per schwab-py's docs: a top-level
"content" list of per-symbol dicts keyed by "key", with OPEN_PRICE/
HIGH_PRICE/LOW_PRICE/CLOSE_PRICE/VOLUME/CHART_TIME_MILLIS fields) since
modes/live_features.py's parsing was written against that documentation,
not a live stream — this is the regression test for if that shape ever
turns out to be wrong or changes.
"""
from __future__ import annotations

import unittest

try:
    import pandas  # noqa: F401

    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def _bar_message(symbol: str, close: float, t_millis: int) -> dict:
    """Builds one CHART_EQUITY stream message with a single symbol's bar,
    matching the shape documented at
    https://schwab-py.readthedocs.io/en/latest/streaming.html"""
    return {
        "service": "CHART_EQUITY",
        "command": "SUBS",
        "content": [
            {
                "seq": 0,
                "key": symbol,
                "OPEN_PRICE": close - 0.05,
                "HIGH_PRICE": close + 0.1,
                "LOW_PRICE": close - 0.1,
                "CLOSE_PRICE": close,
                "VOLUME": 1000.0,
                "CHART_TIME_MILLIS": t_millis,
            }
        ],
    }


@unittest.skipUnless(_HAS_PANDAS, "pandas not installed")
class TestLiveFeatureTracker(unittest.TestCase):
    def setUp(self):
        from modes.live_features import LiveFeatureTracker

        self.tracker = LiveFeatureTracker(["AAPL", "TSLA"])

    def test_ignores_symbols_it_isnt_watching(self):
        msg = _bar_message("MSFT", close=300.0, t_millis=0)
        ready = self.tracker.handle_chart_equity_message(msg)
        self.assertEqual(ready, {})

    def test_holds_until_enough_bars_accumulate(self):
        # Fewer than 20 bars: strategies.features needs a 20-bar warm-up
        # window, so no feature row should come back yet.
        for i in range(19):
            ready = self.tracker.handle_chart_equity_message(_bar_message("AAPL", 100.0 + i, t_millis=i * 60_000))
        self.assertEqual(ready, {})

    def test_produces_a_feature_row_once_warmed_up(self):
        from strategies.features import FEATURE_COLUMNS

        ready = {}
        for i in range(25):
            ready = self.tracker.handle_chart_equity_message(_bar_message("AAPL", 100.0 + i * 0.1, t_millis=i * 60_000))

        self.assertIn("AAPL", ready)
        for col in FEATURE_COLUMNS:
            self.assertIn(col, ready["AAPL"])

    def test_tracks_multiple_symbols_independently(self):
        msg = {
            "service": "CHART_EQUITY",
            "content": [
                {"key": "AAPL", "OPEN_PRICE": 100, "HIGH_PRICE": 101, "LOW_PRICE": 99, "CLOSE_PRICE": 100.5, "VOLUME": 500, "CHART_TIME_MILLIS": 0},
                {"key": "TSLA", "OPEN_PRICE": 200, "HIGH_PRICE": 201, "LOW_PRICE": 199, "CLOSE_PRICE": 200.5, "VOLUME": 700, "CHART_TIME_MILLIS": 0},
            ],
        }
        self.tracker.handle_chart_equity_message(msg)
        # Both symbols' buffers should have exactly one bar each, independently.
        self.assertEqual(len(self.tracker._buffers["AAPL"]._bars), 1)
        self.assertEqual(len(self.tracker._buffers["TSLA"]._bars), 1)


if __name__ == "__main__":
    unittest.main()
