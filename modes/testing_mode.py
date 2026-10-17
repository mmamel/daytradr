"""
Testing mode (a.k.a. backtesting) — NOT YET IMPLEMENTED.

TODO: build this out once the ML model (section 2) exists. Rough plan for
when we get there:

  1. Pull historical bars for the target symbols/date-range via
     `schwab_client.market_data.MarketDataClient.get_intraday_bars` /
     `get_daily_bars` (Schwab gives ~48 days of 1-minute bars, ~9 months of
     5/10/15/30-minute bars, and years of daily bars — plan resolution
     around that limit, or persist bars over time to build a longer local
     history).
  2. Replay those bars in chronological order through the same
     `on_market_data`-shaped interface trading_mode uses, so the ML model
     and strategy code don't know whether they're live or backtesting.
  3. Simulate order fills (with slippage/commission assumptions) instead of
     calling the real trade-execution client from section 3.
  4. Track simulated P&L, win rate, drawdown, etc. and report a summary.
  5. Support parameter sweeps so different model configs/thresholds can be
     compared before ever risking real money.

For now this module intentionally does nothing.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run() -> None:
    raise NotImplementedError(
        "Testing/backtesting mode isn't built yet — see the TODO at the top of "
        "modes/testing_mode.py. Use `--mode trading` for now."
    )
