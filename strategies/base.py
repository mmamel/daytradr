"""
The interface every trading strategy/model implements — this is the seam
that makes strategies swappable. If you write a class that satisfies
`StrategyBase`, the rest of the app (trading_mode, and later, backtesting
and order execution) doesn't need to know or care whether it's a 3-line
if-statement or a trained neural network.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


# Shared 3-class label convention for every trainable strategy (see
# strategies.features.make_labels, which all of them use to derive these
# labels from raw price history the same way). Keeping this map in one
# place means "what an integer label 0/1/2 means" can't drift between
# strategies/sklearn_models.py and strategies/deep_learning.py.
LABEL_TO_SIGNAL = {0: Signal.SELL, 1: Signal.HOLD, 2: Signal.BUY}
SIGNAL_TO_LABEL = {signal: label for label, signal in LABEL_TO_SIGNAL.items()}


@dataclass
class Prediction:
    """What every strategy hands back for a single symbol at a single point in time."""

    signal: Signal
    confidence: float = 0.0  # 0..1, model's own certainty; rule-based strategies can leave this at 0
    metadata: dict[str, Any] = field(default_factory=dict)  # e.g. {"predicted_return": 0.004}


class StrategyBase(ABC):
    """
    Subclass this, implement `predict`, and register it (see
    strategies/registry.py) to plug a new strategy into the app.

    `fit` is intentionally not abstract: rule-based strategies (moving
    average crossovers, RSI thresholds, etc.) don't train on anything and
    can just leave the default no-op in place. ML strategies override it.

    Every trainable strategy's `fit` takes the SAME thing — a raw OHLCV
    DataFrame (oldest-first, ['open','high','low','close','volume']) — not
    pre-computed features or externally-supplied labels. Each strategy
    derives its own features (strategies.features.compute_features) and
    labels (strategies.features.make_labels) internally. This is what lets
    modes/train_mode.py call `strategy.fit(bars)` the same way regardless
    of which strategy is selected.
    """

    def fit(self, bars: Any, labels: Any = None) -> None:
        """
        Train on raw OHLCV bars. `labels` exists only so a caller doesn't
        need a special case for strategies that don't use it (i.e. most of
        them — see the class docstring); it's ignored by every current
        implementation. No-op by default (rule-based strategies).
        """
        return None

    @abstractmethod
    def predict(self, features: Mapping[str, float]) -> Prediction:
        """
        `features` is one row of engineered indicators for one symbol at one
        point in time (see strategies/features.py) — e.g.
        {"sma_5": 101.2, "sma_20": 100.1, "rsi_14": 62.3, ...}.

        Must return a Prediction. Should not raise on missing/NaN features;
        return Signal.HOLD with confidence 0.0 if there isn't enough data
        yet to say anything useful.
        """
        raise NotImplementedError
