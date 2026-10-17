"""
Multi-horizon LSTM strategy: one shared LSTM encoder + six small
classifier heads, one per look-ahead horizon (1/5/10/30/60/120 minutes),
each predicting BUY/HOLD/SELL for that horizon independently.

Registered as "lstm_three_classifier" (not just "lstm") because the label
framing — 3-class BUY/HOLD/SELL via a return threshold — is a deliberate
choice, not the only option (see README.md#ml-model-options). A future
regression version (predict the raw return, threshold at inference time
instead of training time) will live alongside this one as
"lstm_regression", sharing the same LSTM encoder architecture but a
different head/loss/label setup — hence baking the framing into the name
now, before that sibling exists.

Why one shared model instead of six separate ones: the six horizons all
describe the same underlying price path, so a shared encoder lets the model
learn one representation of "what's happening right now" and specialize
only the last layer per horizon — standard multi-task learning, and cheaper
to train/run than six independent LSTMs.

Interface note: every other strategy's `predict()` takes one row of latest
features (a single point in time). An LSTM needs a *sequence* of rows. To
keep the exact same call signature `predict(features: dict) -> Prediction`
that trading_mode.py, the registry, and the tests all use uniformly, this
class keeps its own rolling buffer of the last `sequence_length` feature
rows internally (see `_buffer`) and returns Signal.HOLD/confidence=0 until
that buffer fills up. Nothing outside this file needs to know that's
happening — you still call `strategy.predict(latest_feature_row)` on every
new bar, exactly like sma_crossover or random_forest.

Backing research: Ghosh, Neufeld & Sahoo (2021), "Forecasting directional
movements of stock prices for intraday trading using LSTM and random
forests" (arXiv:2004.10178, Finance Research Letters) — LSTM outperformed
random forest on the same style of intraday direction-classification task
this model performs (0.64%/day vs 0.54%/day pre-cost; 0.444%/day vs
0.339%/day after a 0.2%/day slippage assumption), by a real but modest
margin. Treat this model as a modest upgrade over strategies/sklearn_models.py's
RandomForestStrategy, not a leap — and always compare backtested results
against that baseline before trusting the extra complexity.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from strategies.base import LABEL_TO_SIGNAL, Prediction, Signal, StrategyBase
from strategies.features import FEATURE_COLUMNS, make_labels
from strategies.registry import register

logger = logging.getLogger(__name__)

# horizon name -> how many 1-minute bars ahead it corresponds to.
HORIZONS: dict[str, int] = {
    "1min": 1,
    "5min": 5,
    "10min": 10,
    "30min": 30,
    "1h": 60,
    "2h": 120,
}


class _MultiHorizonLSTM(nn.Module):
    """Shared LSTM encoder + one linear 3-way classifier head per horizon."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, horizon_names: list[str]):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_size, 3) for name in horizon_names})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (batch, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]  # final time step's hidden state, (batch, hidden_size)
        return {name: head(last_step) for name, head in self.heads.items()}


@register("lstm_three_classifier")
class LstmMultiHorizonStrategy(StrategyBase):
    def __init__(
        self,
        sequence_length: int = 60,
        hidden_size: int = 64,
        num_layers: int = 1,
        horizons: Mapping[str, int] | None = None,
        primary_horizon: str = "1min",
        label_threshold: float = 0.001,
        learning_rate: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 64,
        val_fraction: float = 0.15,
        device: str | None = None,
    ):
        self.sequence_length = sequence_length
        self.horizons = dict(horizons or HORIZONS)
        if primary_horizon not in self.horizons:
            raise ValueError(f"primary_horizon must be one of {list(self.horizons)}")
        self.primary_horizon = primary_horizon
        self.label_threshold = label_threshold
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_fraction = val_fraction

        self._device = torch.device(device or _pick_device())
        self.model = _MultiHorizonLSTM(
            input_size=len(FEATURE_COLUMNS),
            hidden_size=hidden_size,
            num_layers=num_layers,
            horizon_names=list(self.horizons),
        ).to(self._device)
        self.scaler = StandardScaler()
        self._is_fitted = False
        self._buffer: deque[list[float]] = deque(maxlen=sequence_length)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, bars, labels=None) -> None:
        """
        Train on one symbol's raw OHLCV history.

        `bars` must be a pandas DataFrame, oldest-first, with at least
        ['open', 'high', 'low', 'close', 'volume'].  `labels` is accepted
        for interface compatibility with StrategyBase but ignored — a
        single label array can't represent six horizons at once, so labels
        for all six are derived internally from `bars['close']`.
        """
        from strategies.features import compute_features

        feature_df = compute_features(bars)
        X_all = feature_df[FEATURE_COLUMNS].to_numpy(dtype=float)
        close = feature_df["close"].to_numpy(dtype=float)
        n = len(feature_df)

        labels_by_horizon = {name: make_labels(close, h, self.label_threshold) for name, h in self.horizons.items()}

        # A training window ending at index i needs: (a) sequence_length
        # valid (non-NaN) feature rows ending at i, and (b) a known future
        # close price for every horizon, i.e. i < n - max(horizon).
        row_valid = ~np.isnan(X_all).any(axis=1)
        max_horizon = max(self.horizons.values())
        last_valid_end = n - max_horizon - 1
        window_ends = [
            i
            for i in range(self.sequence_length - 1, last_valid_end + 1)
            if row_valid[i - self.sequence_length + 1 : i + 1].all()
        ]
        if len(window_ends) < 10:
            raise ValueError(
                f"Not enough history to train: only {len(window_ends)} valid windows from {n} bars. "
                f"Need at least sequence_length ({self.sequence_length}) + the longest horizon "
                f"({max_horizon}) bars of clean data, ideally hundreds more than that."
            )

        X_windows = np.stack([X_all[i - self.sequence_length + 1 : i + 1] for i in window_ends])
        y_windows = {name: labels_by_horizon[name][window_ends].astype(np.int64) for name in self.horizons}

        # Chronological split (NOT shuffled) — validating on data from
        # before the training period would leak future information back
        # into model selection. The one training/validation window pair
        # that straddles the split boundary shares a few bars; that's a
        # standard, accepted simplification for sliding-window setups.
        split = int(len(window_ends) * (1 - self.val_fraction))
        logger.info(
            "Training LSTM on %d windows (%d train / %d val) across horizons %s",
            len(window_ends), split, len(window_ends) - split, list(self.horizons),
        )

        self.scaler.fit(X_windows[:split].reshape(-1, X_windows.shape[-1]))
        X_scaled = self.scaler.transform(X_windows.reshape(-1, X_windows.shape[-1])).reshape(X_windows.shape)

        X_train = torch.tensor(X_scaled[:split], dtype=torch.float32, device=self._device)
        X_val = torch.tensor(X_scaled[split:], dtype=torch.float32, device=self._device)
        y_train = {name: torch.tensor(arr[:split], device=self._device) for name, arr in y_windows.items()}
        y_val = {name: torch.tensor(arr[split:], device=self._device) for name, arr in y_windows.items()}

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # A label_threshold that only BUY/SELL rarely clears (typical for
        # short horizons) can leave HOLD as 95%+ of the data. Unweighted,
        # minimizing cross-entropy is trivially satisfied by predicting
        # HOLD every time — a real local optimum, not a training bug. Each
        # horizon gets its own weights since class balance can differ per
        # horizon (a 2-hour move clears the threshold far more often than a
        # 1-minute one).
        loss_fns = {name: nn.CrossEntropyLoss(weight=_balanced_class_weights(y_train[name], self._device)) for name in self.horizons}

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            permutation = torch.randperm(len(X_train))  # shuffle windows, not time order, within the train split
            total_loss = 0.0
            for start in range(0, len(X_train), self.batch_size):
                idx = permutation[start : start + self.batch_size]
                optimizer.zero_grad()
                outputs = self.model(X_train[idx])
                loss = sum(loss_fns[name](outputs[name], y_train[name][idx]) for name in self.horizons)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * len(idx)

            avg_loss = total_loss / len(X_train)
            if epoch == self.epochs or epoch % max(1, self.epochs // 5) == 0:
                val_acc = self._evaluate(X_val, y_val)
                logger.info(
                    "epoch %d/%d — train loss %.4f — val accuracy: %s",
                    epoch, self.epochs, avg_loss,
                    {name: f"{acc:.1%}" for name, acc in val_acc.items()},
                )

        self.model.eval()
        self._is_fitted = True

    def _evaluate(self, X_val: torch.Tensor, y_val: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_val)
            acc = {
                name: float((outputs[name].argmax(dim=-1) == y_val[name]).float().mean())
                for name in self.horizons
            }
        self.model.train()
        return acc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: Mapping[str, float]) -> Prediction:
        """Same call shape as every other strategy — returns the primary
        horizon's Prediction, with the full 6-horizon breakdown attached in
        metadata["horizons"] for anything that wants more than one signal."""
        all_horizons = self.predict_all_horizons(features)
        primary = all_horizons[self.primary_horizon]
        return Prediction(
            signal=primary.signal,
            confidence=primary.confidence,
            metadata={**primary.metadata, "horizons": all_horizons},
        )

    def predict_all_horizons(self, features: Mapping[str, float]) -> dict[str, Prediction]:
        """Full breakdown: one Prediction per horizon (1min, 5min, 10min, 30min, 1h, 2h)."""
        row = [features.get(col, np.nan) for col in FEATURE_COLUMNS]
        self._buffer.append(row)

        if not self._is_fitted:
            return self._hold_all("model not fitted yet")
        if len(self._buffer) < self.sequence_length:
            return self._hold_all(f"buffering history ({len(self._buffer)}/{self.sequence_length} bars)")

        sequence = np.array(self._buffer, dtype=float)
        if np.isnan(sequence).any():
            return self._hold_all("missing/NaN features in the current window")

        scaled = self.scaler.transform(sequence)
        x = torch.tensor(scaled, dtype=torch.float32, device=self._device).unsqueeze(0)  # (1, seq_len, features)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)

        predictions = {}
        for name, logits in outputs.items():
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            label = int(probs.argmax())
            predictions[name] = Prediction(
                signal=LABEL_TO_SIGNAL[label],
                confidence=float(probs[label]),
                metadata={"probabilities": {"SELL": float(probs[0]), "HOLD": float(probs[1]), "BUY": float(probs[2])}},
            )
        return predictions

    def _hold_all(self, reason: str) -> dict[str, Prediction]:
        return {name: Prediction(Signal.HOLD, confidence=0.0, metadata={"reason": reason}) for name in self.horizons}

    # ------------------------------------------------------------------
    # Persistence — train once (see modes/train_mode.py), reuse across
    # trading_mode.py runs instead of retraining on every startup.
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "scaler_mean": self.scaler.mean_,
                "scaler_scale": self.scaler.scale_,
                "sequence_length": self.sequence_length,
                "horizons": self.horizons,
                "primary_horizon": self.primary_horizon,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, **kwargs) -> "LstmMultiHorizonStrategy":
        checkpoint = torch.load(path, map_location="cpu")
        strategy = cls(
            sequence_length=checkpoint["sequence_length"],
            horizons=checkpoint["horizons"],
            primary_horizon=checkpoint["primary_horizon"],
            **kwargs,
        )
        strategy.model.load_state_dict(checkpoint["model_state"])
        strategy.model.eval()
        strategy.scaler.mean_ = checkpoint["scaler_mean"]
        strategy.scaler.scale_ = checkpoint["scaler_scale"]
        strategy._is_fitted = True
        return strategy


def _balanced_class_weights(labels: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Per-class weight = n_samples / (n_classes * count_of_that_class) — the
    same "balanced" formula sklearn's class_weight="balanced" uses, so a
    rare class contributes as much to the average loss as the common one
    instead of being drowned out. Classes absent from this particular
    training split (e.g. a horizon that never saw a SELL) get clamped to a
    count of 1 to avoid dividing by zero; they'll just never get a gradient
    signal for that class, which is a data problem, not something a loss
    weight can fix.
    """
    counts = torch.bincount(labels, minlength=3).float().clamp(min=1)
    weights = len(labels) / (3 * counts)
    return weights.to(device)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
