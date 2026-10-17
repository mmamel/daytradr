"""
Classical supervised-learning strategies, all sharing one generic wrapper.
This is the pattern to copy for any new scikit-learn-compatible model: pick
a classifier, register a name for it, done — no other file changes.

Label convention: 3-class BUY/HOLD/SELL, derived internally from raw price
history via strategies.features.make_labels — see strategies/base.py's
LABEL_TO_SIGNAL for the exact mapping. `horizon_bars` (default 1, i.e.
next-bar direction) and `label_threshold` control what "BUY" means; both
are constructor kwargs, e.g. get_strategy("random_forest", horizon_bars=5).

Backing research (see README.md#ml-model-options for the full list/caveats):
  Ghosh, Neufeld & Sahoo (2021), "Forecasting directional movements of stock
  prices for intraday trading using LSTM and random forests" (arXiv:2004.10178)
  found a random-forest classifier on a multi-feature intraday setup returned
  0.54%/day pre-cost, 0.339%/day after a 0.2%/day slippage assumption, on
  S&P 500 constituents 1993-2018 — i.e. positive but roughly halved once
  costs are modeled. Gradient boosting variants (XGBoost/LightGBM) report
  similar or better fit quality in several papers (e.g. Yang & Wu, 2021,
  E3S Conf.) but that literature mostly targets price-level regression
  rather than the intraday direction-classification task this app needs —
  treat it as weaker evidence than the FRL paper above.
"""
from __future__ import annotations

import pickle
from typing import Any, Mapping

import numpy as np

from strategies.base import LABEL_TO_SIGNAL, Prediction, Signal, StrategyBase
from strategies.registry import register


class SklearnClassifierStrategy(StrategyBase):
    """
    Generic adapter: wraps any scikit-learn classifier that implements
    `.fit(X, y)` and `.predict_proba(X)`/`.predict(X)`.

    `feature_order` fixes the column order fed to the model — it MUST match
    the order used when the model was trained. Defaults to
    strategies.features.FEATURE_COLUMNS.
    """

    def __init__(
        self,
        model: Any,
        feature_order: list[str] | None = None,
        horizon_bars: int = 1,
        label_threshold: float = 0.001,
    ):
        from strategies.features import FEATURE_COLUMNS

        self.model = model
        self.feature_order = feature_order or FEATURE_COLUMNS
        self.horizon_bars = horizon_bars
        self.label_threshold = label_threshold
        self._is_fitted = False

    def fit(self, bars, labels=None) -> None:
        """
        Train on raw OHLCV bars (oldest-first). `labels` is accepted for
        interface compatibility with StrategyBase but ignored — labels are
        derived internally so every strategy defines BUY/HOLD/SELL the same
        way (see strategies.features.make_labels).
        """
        from sklearn.utils.class_weight import compute_sample_weight

        from strategies.features import compute_features, make_labels

        feature_df = compute_features(bars)
        X_all = feature_df[self.feature_order].to_numpy(dtype=float)
        close = feature_df["close"].to_numpy(dtype=float)
        y_all = make_labels(close, self.horizon_bars, self.label_threshold)

        # Drop rows with NaN features (indicator warm-up) and the tail rows
        # whose future return isn't known yet (see make_labels docstring).
        valid = ~np.isnan(X_all).any(axis=1)
        valid[len(valid) - self.horizon_bars :] = False

        if valid.sum() < 10:
            raise ValueError(
                f"Not enough clean rows to train on: only {int(valid.sum())} out of {len(bars)} bars. "
                f"Need more history, a shorter horizon_bars, or both."
            )

        # With a label_threshold that only BUY/SELL rarely clears (typical
        # for short horizons — a 1-minute move big enough to matter is
        # uncommon), HOLD can be 95%+ of the data. Unweighted, minimizing
        # training loss is trivially satisfied by predicting HOLD every
        # time, which is a real local optimum, not a bug in the model — the
        # fix is to weight rare classes up so that's no longer "free."
        # sample_weight (rather than each model's own class_weight
        # constructor arg, which GradientBoostingClassifier doesn't even
        # have) works identically across every sklearn classifier here.
        y_train = y_all[valid]
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        self.model.fit(X_all[valid], y_train, sample_weight=sample_weight)
        self._is_fitted = True

    def predict(self, features: Mapping[str, float]) -> Prediction:
        if not self._is_fitted:
            return Prediction(Signal.HOLD, confidence=0.0, metadata={"reason": "model not fitted yet"})

        row = np.array([[features.get(col, np.nan) for col in self.feature_order]])
        if np.isnan(row).any():
            return Prediction(Signal.HOLD, confidence=0.0, metadata={"reason": "missing features"})

        label = int(self.model.predict(row)[0])
        confidence = 0.0
        probabilities = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(row)[0]
            confidence = float(max(proba))
            # self.model.classes_ gives the label-order predict_proba uses.
            probabilities = {
                LABEL_TO_SIGNAL[int(cls)].value: float(p) for cls, p in zip(self.model.classes_, proba)
            }

        signal = LABEL_TO_SIGNAL.get(label, Signal.HOLD)
        metadata = {"probabilities": probabilities} if probabilities else {}
        return Prediction(signal, confidence=confidence, metadata=metadata)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_order": self.feature_order,
                    "horizon_bars": self.horizon_bars,
                    "label_threshold": self.label_threshold,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "SklearnClassifierStrategy":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        # Bypass subclass __init__ (which would build a fresh, untrained
        # sklearn model) and restore the trained state directly instead.
        strategy = cls.__new__(cls)
        strategy.model = payload["model"]
        strategy.feature_order = payload["feature_order"]
        strategy.horizon_bars = payload["horizon_bars"]
        strategy.label_threshold = payload["label_threshold"]
        strategy._is_fitted = True
        return strategy


@register("logistic_regression")
class LogisticRegressionStrategy(SklearnClassifierStrategy):
    """Cheap, interpretable linear baseline for direction classification."""

    def __init__(self, horizon_bars: int = 1, label_threshold: float = 0.001, **model_kwargs):
        from sklearn.linear_model import LogisticRegression

        super().__init__(
            LogisticRegression(max_iter=1000, **model_kwargs),
            horizon_bars=horizon_bars,
            label_threshold=label_threshold,
        )


@register("random_forest")
class RandomForestStrategy(SklearnClassifierStrategy):
    """
    The model Ghosh, Neufeld & Sahoo (2021) benchmarked at 0.54%/day
    pre-cost intraday return on S&P 500 constituents — see module docstring.
    """

    def __init__(self, horizon_bars: int = 1, label_threshold: float = 0.001, **model_kwargs):
        from sklearn.ensemble import RandomForestClassifier

        defaults = dict(n_estimators=300, max_depth=6, n_jobs=-1, random_state=0)
        defaults.update(model_kwargs)
        super().__init__(RandomForestClassifier(**defaults), horizon_bars=horizon_bars, label_threshold=label_threshold)


@register("gradient_boosting")
class GradientBoostingStrategy(SklearnClassifierStrategy):
    """scikit-learn's built-in gradient boosting; swap in xgboost/lightgbm the same way if you install them."""

    def __init__(self, horizon_bars: int = 1, label_threshold: float = 0.001, **model_kwargs):
        from sklearn.ensemble import GradientBoostingClassifier

        super().__init__(
            GradientBoostingClassifier(**model_kwargs), horizon_bars=horizon_bars, label_threshold=label_threshold
        )


class MultiHorizonSklearnStrategy(StrategyBase):
    """
    N independent sklearn classifiers, one per horizon, behind the exact
    same predict()/predict_all_horizons() interface as
    strategies.deep_learning.LstmMultiHorizonStrategy — so the two can be
    compared head-to-head through modes/train_mode.py, which detects
    multi-horizon strategies via `hasattr(strategy, "horizons")` and
    doesn't care which architecture is behind it.

    Unlike the LSTM's shared encoder, there's no cheap way to share
    representations across horizons in a plain sklearn model — this trains
    `len(horizons)` completely independent models, so it costs roughly
    `len(horizons)` times what a single-horizon SklearnClassifierStrategy
    costs to train. No sequence buffering is needed, though (no
    `sequence_length`, no warm-up beyond the usual 20-bar feature warm-up)
    since every prediction is a single-row lookup, same as the
    single-horizon wrapper.

    `horizons` defaults to the exact same set as LstmMultiHorizonStrategy
    (strategies.deep_learning.HORIZONS) specifically so a random-forest run
    and an LSTM run are comparable on identical horizons out of the box.
    """

    def __init__(
        self,
        model_factory,
        horizons: dict[str, int] | None = None,
        feature_order: list[str] | None = None,
        label_threshold: float = 0.001,
        primary_horizon: str = "1min",
    ):
        from strategies.deep_learning import HORIZONS as DEFAULT_HORIZONS
        from strategies.features import FEATURE_COLUMNS

        self.horizons = dict(horizons or DEFAULT_HORIZONS)
        if primary_horizon not in self.horizons:
            raise ValueError(f"primary_horizon must be one of {list(self.horizons)}")
        self.primary_horizon = primary_horizon
        self.feature_order = feature_order or FEATURE_COLUMNS
        self.label_threshold = label_threshold
        # A fresh, independent model instance per horizon — reusing one
        # instance across horizons would mean the second .fit() call
        # overwrites the first horizon's trained model.
        self.models = {name: model_factory() for name in self.horizons}
        self._is_fitted = False

    def fit(self, bars, labels=None) -> None:
        """Same raw-OHLCV-bars contract as every other trainable strategy;
        trains each horizon's model independently on its own labels."""
        from sklearn.utils.class_weight import compute_sample_weight

        from strategies.features import compute_features, make_labels

        feature_df = compute_features(bars)
        X_all = feature_df[self.feature_order].to_numpy(dtype=float)
        close = feature_df["close"].to_numpy(dtype=float)
        valid_features = ~np.isnan(X_all).any(axis=1)

        for name, horizon_bars in self.horizons.items():
            y_all = make_labels(close, horizon_bars, self.label_threshold)
            valid = valid_features.copy()
            valid[len(valid) - horizon_bars :] = False

            if valid.sum() < 10:
                raise ValueError(
                    f"Not enough clean rows to train horizon '{name}': only {int(valid.sum())} out of {len(bars)} bars."
                )

            y_train = y_all[valid]
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
            self.models[name].fit(X_all[valid], y_train, sample_weight=sample_weight)

        self._is_fitted = True

    def predict(self, features: Mapping[str, float]) -> Prediction:
        all_horizons = self.predict_all_horizons(features)
        primary = all_horizons[self.primary_horizon]
        return Prediction(
            signal=primary.signal, confidence=primary.confidence, metadata={**primary.metadata, "horizons": all_horizons}
        )

    def predict_all_horizons(self, features: Mapping[str, float]) -> dict[str, Prediction]:
        if not self._is_fitted:
            return {name: Prediction(Signal.HOLD, 0.0, {"reason": "model not fitted yet"}) for name in self.horizons}

        row = np.array([[features.get(col, np.nan) for col in self.feature_order]])
        if np.isnan(row).any():
            return {name: Prediction(Signal.HOLD, 0.0, {"reason": "missing features"}) for name in self.horizons}

        predictions = {}
        for name, model in self.models.items():
            label = int(model.predict(row)[0])
            confidence = 0.0
            probabilities = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(row)[0]
                confidence = float(max(proba))
                probabilities = {LABEL_TO_SIGNAL[int(cls)].value: float(p) for cls, p in zip(model.classes_, proba)}
            predictions[name] = Prediction(
                signal=LABEL_TO_SIGNAL.get(label, Signal.HOLD),
                confidence=confidence,
                metadata={"probabilities": probabilities} if probabilities else {},
            )
        return predictions

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "models": self.models,
                    "horizons": self.horizons,
                    "feature_order": self.feature_order,
                    "label_threshold": self.label_threshold,
                    "primary_horizon": self.primary_horizon,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "MultiHorizonSklearnStrategy":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        strategy = cls.__new__(cls)
        strategy.models = payload["models"]
        strategy.horizons = payload["horizons"]
        strategy.feature_order = payload["feature_order"]
        strategy.label_threshold = payload["label_threshold"]
        strategy.primary_horizon = payload["primary_horizon"]
        strategy._is_fitted = True
        return strategy


@register("random_forest_multi_horizon")
class RandomForestMultiHorizonStrategy(MultiHorizonSklearnStrategy):
    """
    The multi-horizon counterpart to `random_forest`, for direct comparison
    against `lstm_three_classifier` on identical horizons — see module
    docstring for the Ghosh, Neufeld & Sahoo (2021) benchmark this
    architecture is based on.

    n_jobs=1 here (unlike `random_forest`'s n_jobs=-1): this class calls
    predict()/predict_proba() once per horizon per row during evaluation
    (6x the calls of the single-horizon version), and parallelizing a
    forest across processes for a single-row prediction is pure
    coordination overhead with no real work to divide up — it's the same
    anti-pattern that caused the "hanging" training run a few steps back,
    just waiting to happen again at a larger multiplier. Training
    (`fit()`, called once per horizon on the full batch) is somewhat slower
    as a result, which is the right side of that trade-off to be on.
    """

    def __init__(self, horizons: dict[str, int] | None = None, label_threshold: float = 0.001, **model_kwargs):
        from sklearn.ensemble import RandomForestClassifier

        defaults = dict(n_estimators=300, max_depth=6, n_jobs=1, random_state=0)
        defaults.update(model_kwargs)
        super().__init__(
            model_factory=lambda: RandomForestClassifier(**defaults),
            horizons=horizons,
            label_threshold=label_threshold,
        )
