"""
Simple, un-trained baselines. No `fit()` needed — these exist so you always
have a cheap, transparent strategy to sanity-check the pipeline against and
to compare any ML model's backtest results to (a model that can't beat these
isn't earning its complexity).
"""
from __future__ import annotations

import math
from typing import Mapping

from strategies.base import Prediction, Signal, StrategyBase
from strategies.registry import register


@register("sma_crossover")
class SmaCrossoverStrategy(StrategyBase):
    """Classic trend-following rule: fast SMA above slow SMA => uptrend => BUY."""

    def __init__(self, fast_col: str = "sma_5", slow_col: str = "sma_20"):
        self.fast_col = fast_col
        self.slow_col = slow_col

    def predict(self, features: Mapping[str, float]) -> Prediction:
        fast = features.get(self.fast_col)
        slow = features.get(self.slow_col)
        if fast is None or slow is None or math.isnan(fast) or math.isnan(slow):
            return Prediction(Signal.HOLD, confidence=0.0)

        spread = (fast - slow) / slow if slow else 0.0
        if fast > slow:
            return Prediction(Signal.BUY, confidence=min(abs(spread) * 10, 1.0))
        if fast < slow:
            return Prediction(Signal.SELL, confidence=min(abs(spread) * 10, 1.0))
        return Prediction(Signal.HOLD, confidence=0.0)


@register("rsi_mean_reversion")
class RsiMeanReversionStrategy(StrategyBase):
    """Classic mean-reversion rule: RSI below `oversold` => BUY, above `overbought` => SELL."""

    def __init__(self, oversold: float = 30.0, overbought: float = 70.0, rsi_col: str = "rsi_14"):
        self.oversold = oversold
        self.overbought = overbought
        self.rsi_col = rsi_col

    def predict(self, features: Mapping[str, float]) -> Prediction:
        rsi = features.get(self.rsi_col)
        if rsi is None or math.isnan(rsi):
            return Prediction(Signal.HOLD, confidence=0.0)

        if rsi <= self.oversold:
            return Prediction(Signal.BUY, confidence=min((self.oversold - rsi) / self.oversold, 1.0))
        if rsi >= self.overbought:
            return Prediction(Signal.SELL, confidence=min((rsi - self.overbought) / (100 - self.overbought), 1.0))
        return Prediction(Signal.HOLD, confidence=0.0)
