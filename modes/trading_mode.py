"""
Trading mode: the "always on" mode.

You can start this process at any time of day (or leave it running
permanently, e.g. under systemd/supervisord/a screen session). It:

  1. Figures out whether the market is open today, and if so, when it
     opens/closes (modes/schedule.py — accounts for weekends, holidays,
     and early closes).
  2. If the market isn't open yet, sleeps until the open.
  3. From open to close, streams live level-one quotes (and minute bars)
     for the configured symbols and hands each message to `on_market_data`.
  4. At the close, disconnects and goes back to step 1 for the next
     trading day.

Which strategy runs is picked at startup by `DAYTRADER_STRATEGY` (see
config.py and strategies/) and resolved via strategies.get_strategy — fails
fast at startup with a clear error if the name is unknown, before any
market data has to flow. Every time a bar closes (CHART_EQUITY), the
rolling per-symbol feature vector (modes/live_features.py) is fed into
`strategy.predict(...)`/`predict_all_horizons(...)` and the resulting
signal + probability is logged — that's today's "cycle." Raw quote ticks
(LEVEL_ONE_EQUITY) are logged at DEBUG only; the strategy isn't
re-evaluated on every tick, just on each new bar (see live_features.py's
docstring for why).

TODO(section 3): once a signal is produced, hand BUY/SELL off to the
trade-execution client instead of just logging it.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from config import AppConfig, SchwabConfig
from modes.live_features import LiveFeatureTracker
from modes.schedule import format_pacific, get_today_session
from schwab_client.auth import get_client
from schwab_client.market_data import MarketDataClient
from schwab_client.streaming import LiveQuoteStream
from strategies import StrategyBase, get_strategy

logger = logging.getLogger(__name__)


def on_quote(msg: dict) -> None:
    """Raw LEVEL_ONE_EQUITY tick — logged at DEBUG only; see module docstring
    for why the strategy doesn't re-run on every tick."""
    logger.debug("quote tick: %s", msg)


def on_minute_bar(msg: dict, strategy: StrategyBase, tracker: LiveFeatureTracker) -> None:
    """
    Handler for each closed 1-minute bar (CHART_EQUITY). Updates the rolling
    per-symbol feature history and, once there's enough of it, runs the
    strategy and logs the resulting signal(s) and probabilities.

    TODO(section 3): once a signal is produced, hand BUY/SELL off to the
    trade-execution client instead of just logging it.
    """
    for symbol, features in tracker.handle_chart_equity_message(msg).items():
        predict_all_horizons = getattr(strategy, "predict_all_horizons", None)
        horizons = predict_all_horizons(features) if predict_all_horizons else {"default": strategy.predict(features)}

        summary = ", ".join(f"{name}={pred.signal.value}({pred.confidence:.0%})" for name, pred in horizons.items())
        logger.info("%s: %s", symbol, summary)


async def _sleep_until(target_utc: datetime) -> None:
    now = datetime.now(tz=timezone.utc)
    delay = max((target_utc - now).total_seconds(), 0)
    if delay:
        logger.info("Sleeping %.0f minutes until %s", delay / 60, format_pacific(target_utc))
        await asyncio.sleep(delay)


async def _run_one_session(
    market_data: MarketDataClient, symbols: list[str], close_time: datetime, strategy: StrategyBase
) -> None:
    stream = LiveQuoteStream(market_data.client, symbols)
    # Fresh per session (i.e. per trading day) — indicators like the
    # cumulative VWAP in strategies/features.py are meant to reset daily,
    # and starting clean also means yesterday's overnight gap never leaks
    # into today's rolling window.
    tracker = LiveFeatureTracker(symbols)
    stream.on_quote(on_quote)
    stream.on_minute_bar(lambda msg: on_minute_bar(msg, strategy, tracker))

    async def _stop_at_close() -> None:
        await _sleep_until(close_time)
        logger.info("Market close reached, disconnecting stream")
        stream.stop()

    await asyncio.gather(stream.run_forever(), _stop_at_close())


async def run(cfg: AppConfig | None = None, schwab_cfg: SchwabConfig | None = None) -> None:
    cfg = cfg or AppConfig.from_env()
    client = get_client(schwab_cfg)
    market_data = MarketDataClient(client=client)

    # Fails fast here (before ever touching the market) if DAYTRADER_STRATEGY
    # names a strategy that isn't registered under strategies/.
    strategy = get_strategy(cfg.strategy_name)
    logger.info("Using strategy '%s' (%s)", cfg.strategy_name, type(strategy).__name__)

    while True:
        session = get_today_session(market_data)
        now = datetime.now(tz=timezone.utc)

        if not session.is_open_today or (session.close_time and now >= session.close_time):
            # Market's done for today (or closed entirely, e.g. weekend/holiday).
            # Check again early tomorrow.
            tomorrow = (now + timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)
            await _sleep_until(tomorrow)
            continue

        if now < session.open_time:
            await _sleep_until(session.open_time)

        logger.info("Market open — streaming live data for %s until %s", cfg.symbols, format_pacific(session.close_time))
        await _run_one_session(market_data, cfg.symbols, session.close_time, strategy)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
