"""
Turns raw OHLCV bars into the technical-indicator features strategies
actually consume. Every strategy (rule-based or ML) reads from the same
feature set, so swapping strategies never means re-deriving indicators —
only `strategies/*.py` (the decision logic) changes.

Input: a pandas DataFrame of bars, oldest-first, with at least
['open', 'high', 'low', 'close', 'volume'] columns — exactly what
schwab_client.market_data.get_intraday_bars / the CHART_EQUITY stream give
you once wrapped in a DataFrame. A 'datetime' column (epoch millis — also
present on both those code paths) enables the daily-reset VWAP; without it,
VWAP falls back to cumulative-since-the-start-of-whatever-was-passed-in.

NOTE ON TREND DETECTION: the original 10-feature set (return_1, momentum_5,
sma_5/20, ema_12/26, macd, rsi_14, volatility_20, vwap_deviation) skews
mean-reversion — rsi_14 and vwap_deviation specifically read "far from
average" as a sell signal, which is the wrong read during a genuine
sustained trend. adx_14, trend_slope_20/trend_r2_20, sma_50/sma_50_slope,
and trend_streak below were added specifically to give trend-following
behavior a chance: they measure trend *strength and persistence*, which
nothing in the original set did. Changing FEATURE_COLUMNS invalidates any
previously-saved strategy (.pkl/.pt) trained against the old feature
vector shape — retrain after picking this up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with indicator columns appended to `bars`."""
    df = bars.copy()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    df["return_1"] = close.pct_change(1)
    df["momentum_5"] = close.pct_change(5)

    df["sma_5"] = close.rolling(window=5).mean()
    df["sma_20"] = close.rolling(window=20).mean()
    sma_50 = close.rolling(window=50).mean()
    df["sma_50"] = sma_50
    # Rate of change of the *long* average itself, not just its level — is
    # the medium-term trend accelerating, flattening, or reversing.
    df["sma_50_slope"] = sma_50.pct_change(5)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["ema_12"] = ema_12
    df["ema_26"] = ema_26
    # As a fraction of price, not a raw dollar amount — a $0.50 MACD means
    # very different things for a $10 stock and a $500 one, and even for
    # one symbol its own price level drifts over a long training window.
    df["macd_pct"] = (ema_12 - ema_26) / close

    df["rsi_14"] = _rsi(close, window=14)
    df["volatility_20"] = df["return_1"].rolling(window=20).std()

    df["adx_14"] = _adx(high, low, close, window=14)

    slope, r2 = _rolling_trend(close, window=20)
    df["trend_slope_20"] = slope / close  # normalized to %/bar, same reasoning as macd_pct
    df["trend_r2_20"] = r2  # 0..1: how well a straight line fits the last 20 bars — "how clean is this move"

    df["trend_streak"] = _signed_streak(df["return_1"])

    if "datetime" in df.columns:
        day = pd.to_datetime(df["datetime"], unit="ms").dt.date
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).groupby(day).cumsum() / volume.groupby(day).cumsum()
    else:
        # No day boundary available (e.g. bars built without a 'datetime'
        # column) — cumulative since the start of whatever was passed in.
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
    df["vwap_deviation"] = (close - vwap) / vwap

    return df


def latest_feature_vector(bars: pd.DataFrame) -> dict[str, float] | None:
    """
    Convenience for live trading: compute features and return just the most
    recent row as a plain dict, ready to hand to `Strategy.predict(...)`.
    Returns None if there isn't enough history yet to compute anything
    meaningful (e.g. fewer than 50 bars seen so far — sma_50 is now the
    longest warm-up window any feature needs).
    """
    if len(bars) < 50:
        return None
    latest = compute_features(bars).iloc[-1]
    return latest.to_dict()


FEATURE_COLUMNS = [
    "return_1",
    "momentum_5",
    "sma_5",
    "sma_20",
    "sma_50",
    "sma_50_slope",
    "ema_12",
    "ema_26",
    "macd_pct",
    "rsi_14",
    "volatility_20",
    "adx_14",
    "trend_slope_20",
    "trend_r2_20",
    "trend_streak",
    "vwap_deviation",
]


def make_labels(close: np.ndarray, horizon_bars: int, threshold: float) -> np.ndarray:
    """
    Shared 3-class (BUY/HOLD/SELL) labeling used by every trainable
    strategy — both strategies/deep_learning.py (once per horizon) and
    strategies/sklearn_models.py, plus modes/train_mode.py's evaluation
    step. Keeping this in one place means "what counts as BUY vs. noise" is
    defined identically everywhere it's used.

    Integer labels {0=SELL, 1=HOLD, 2=BUY} for every index, based on the
    return from that bar to `horizon_bars` bars later. The last
    `horizon_bars` entries are meaningless (there's no future bar to look
    at yet) — callers must restrict training/evaluation to indices where
    the future is actually known.
    """
    n = len(close)
    future_return = np.zeros(n)
    future_return[: n - horizon_bars] = close[horizon_bars:] / close[: n - horizon_bars] - 1
    return np.where(future_return > threshold, 2, np.where(future_return < -threshold, 0, 1))


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average Directional Index (Wilder, 1978) — the standard purpose-built
    trend-*strength* indicator, deliberately direction-agnostic: high ADX
    means "this is trending, hard, one way or the other," independent of
    which way. Ranges roughly 0-100; conventionally >25 = trending,
    <20 = ranging/choppy. Nothing else in this feature set measures
    strength/persistence this directly — see the module docstring.

    Uses Wilder's smoothing, approximated via an EWM with alpha=1/window
    (the standard, widely-used approximation — true Wilder smoothing
    differs only in how the first `window` values are seeded).
    """
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    atr = true_range.ewm(alpha=1 / window, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / window, adjust=False).mean()


def _rolling_trend(close: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """
    Rolling ordinary-least-squares fit of price against bar position over
    the last `window` bars. Returns (slope, r_squared):
      - slope: direction and rate of the move (still in price units here —
        the caller normalizes by price to get %/bar, same as macd_pct).
      - r_squared: 0..1, how well a straight line explains the last
        `window` bars — this is the actual "how sustained/clean is this
        move" measurement nothing else in the feature set provides. A big
        net move with low r_squared is a noisy/choppy path; the same net
        move with high r_squared is a real sustained trend.
    """
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    ss_xx = float((x_centered**2).sum())

    def slope_fn(y: np.ndarray) -> float:
        return float((x_centered * (y - y.mean())).sum() / ss_xx)

    def r2_fn(y: np.ndarray) -> float:
        y_centered = y - y.mean()
        ss_yy = float((y_centered**2).sum())
        if ss_yy == 0:
            return 0.0
        ss_xy = float((x_centered * y_centered).sum())
        return (ss_xy**2) / (ss_xx * ss_yy)

    slope = close.rolling(window=window).apply(slope_fn, raw=True)
    r_squared = close.rolling(window=window).apply(r2_fn, raw=True)
    return slope, r_squared


def _signed_streak(returns: pd.Series) -> pd.Series:
    """
    How many consecutive bars have moved in the same direction, signed
    (+3 = 3 up-bars in a row, -5 = 5 down-bars in a row). A crude but cheap
    persistence signal distinct from adx_14/trend_r2_20 — those measure
    "is price net moving in one direction," this measures "hasn't reversed
    bar-to-bar recently," which a choppy-but-net-trending move can still
    score low on.
    """
    sign = np.sign(returns).fillna(0)
    group = (sign != sign.shift()).cumsum()
    return sign.groupby(group).cumcount().add(1).mul(sign)
