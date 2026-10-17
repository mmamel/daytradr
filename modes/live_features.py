"""
Bridges the raw CHART_EQUITY stream messages from schwab_client.streaming
into the rolling per-symbol bar history that strategies.features needs, so
trading_mode.py can compute a fresh feature vector every time a new 1-minute
bar closes and feed it straight into strategy.predict().

This is deliberately keyed off bar closes (CHART_EQUITY), not every raw
quote tick (LEVEL_ONE_EQUITY) — the feature set in strategies/features.py
(SMA/EMA/RSI/etc.) is bar-based, and a strategy re-evaluating on every tick
would just recompute the same feature row dozens of times between bars for
no benefit. "Each cycle" for strategy.predict() purposes means each closed
bar.
"""
from __future__ import annotations

from collections import deque

import pandas as pd

from strategies.features import latest_feature_vector

# Comfortably more than any strategy's sequence_length (60 for the LSTM) +
# the longest indicator warm-up window (50, for sma_50/sma_50_slope) +
# margin. Older bars are dropped automatically once a symbol's buffer is full.
_MAX_BUFFERED_BARS = 300


class SymbolBarBuffer:
    """Rolling OHLCV history for one symbol, fed one CHART_EQUITY bar at a time."""

    def __init__(self, maxlen: int = _MAX_BUFFERED_BARS):
        self._bars: deque[dict] = deque(maxlen=maxlen)

    def add_bar(self, chart_equity_entry: dict) -> None:
        """`chart_equity_entry` is one element of a CHART_EQUITY message's
        `content` list, e.g. {"key": "AAPL", "OPEN_PRICE": 421.65, ...}."""
        self._bars.append(
            {
                "datetime": chart_equity_entry["CHART_TIME_MILLIS"],
                "open": chart_equity_entry["OPEN_PRICE"],
                "high": chart_equity_entry["HIGH_PRICE"],
                "low": chart_equity_entry["LOW_PRICE"],
                "close": chart_equity_entry["CLOSE_PRICE"],
                "volume": chart_equity_entry["VOLUME"],
            }
        )

    def latest_features(self) -> dict[str, float] | None:
        """The newest feature row, or None if there isn't enough history yet
        (see strategies.features.latest_feature_vector — needs >= 20 bars)."""
        if not self._bars:
            return None
        return latest_feature_vector(pd.DataFrame(self._bars))


class LiveFeatureTracker:
    """One SymbolBarBuffer per watched symbol."""

    def __init__(self, symbols: list[str]):
        self._buffers = {symbol.upper(): SymbolBarBuffer() for symbol in symbols}

    def handle_chart_equity_message(self, msg: dict) -> dict[str, dict[str, float]]:
        """
        Feed one raw CHART_EQUITY stream message in; get back
        {symbol: feature_row} for every symbol in the message that (a) we're
        watching and (b) now has enough history for a feature vector.
        """
        ready = {}
        for entry in msg.get("content", []):
            symbol = entry.get("key")
            buffer = self._buffers.get(symbol)
            if buffer is None:
                continue  # a symbol we're not tracking — ignore
            buffer.add_bar(entry)
            features = buffer.latest_features()
            if features is not None:
                ready[symbol] = features
        return ready
