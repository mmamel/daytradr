"""
Real-time market data via the Schwab streaming (WebSocket) API.

This is what "trading mode" actually pulls from once the market is open —
REST polling (market_data.py) is fine for snapshots, but a day-trading
strategy wants every tick without hammering the REST rate limit.

Docs: https://developer.schwab.com/products/trader-api--individual under
"Streamer Guide". Two services are wired up here:

  * LEVEL_ONE_EQUITY - best bid/ask/last, sizes, volume, etc. Fires on
    every quote change. This is the main feed a day-trading model wants.
  * CHART_EQUITY     - pushes a new OHLCV minute bar as each one closes.
    Useful for feeding bar-based indicators without re-aggregating ticks
    yourself.

Both are optional independently. Add more `add_*_handler`/`*_subs` calls
here later if the model wants level-two book depth (NASDAQ_BOOK/NYSE_BOOK)
or option quotes (LEVEL_ONE_OPTION).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Iterable

from schwab.client import Client
from schwab.streaming import StreamClient

logger = logging.getLogger(__name__)

QuoteHandler = Callable[[dict], Awaitable[None] | None]


class LiveQuoteStream:
    """
    Subscribes to live level-one quotes (and, optionally, minute bars) for a
    set of symbols and hands each message to your callback(s) as it arrives.

    Usage:
        stream = LiveQuoteStream(schwab_client, symbols=["AAPL", "TSLA"])
        stream.on_quote(my_callback)
        await stream.run_forever()   # blocks until stream.stop() is called
    """

    def __init__(self, client: Client, symbols: Iterable[str]):
        self._symbols = [s.upper() for s in symbols]
        self._stream = StreamClient(client)
        self._quote_handlers: list[QuoteHandler] = []
        self._bar_handlers: list[QuoteHandler] = []
        self._stop_event = asyncio.Event()

    def on_quote(self, handler: QuoteHandler) -> None:
        """Register a callback invoked with each raw LEVEL_ONE_EQUITY message."""
        self._quote_handlers.append(handler)

    def on_minute_bar(self, handler: QuoteHandler) -> None:
        """Register a callback invoked with each raw CHART_EQUITY (1-min bar) message."""
        self._bar_handlers.append(handler)

    def stop(self) -> None:
        self._stop_event.set()

    async def _dispatch(self, handlers: list[QuoteHandler], msg: dict) -> None:
        for handler in handlers:
            try:
                result = handler(msg)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Unhandled exception in stream handler")

    async def run_forever(self, *, include_minute_bars: bool = True) -> None:
        """
        Connect, subscribe, and process messages until `stop()` is called or
        the process is killed. Reconnects automatically with backoff if the
        socket drops (Schwab's stream disconnects periodically; this is
        expected, not exceptional).
        """
        self._stop_event.clear()
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                await self._run_once(include_minute_bars=include_minute_bars)
                backoff = 1.0  # clean run; reset backoff
            except Exception:
                logger.exception("Stream connection dropped, reconnecting in %.0fs", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _run_once(self, *, include_minute_bars: bool) -> None:
        await self._stream.login()

        self._stream.add_level_one_equity_handler(
            lambda msg: asyncio.create_task(self._dispatch(self._quote_handlers, msg))
        )
        await self._stream.level_one_equity_subs(self._symbols)

        if include_minute_bars:
            self._stream.add_chart_equity_handler(
                lambda msg: asyncio.create_task(self._dispatch(self._bar_handlers, msg))
            )
            await self._stream.chart_equity_subs(self._symbols)

        logger.info("Streaming live data for %s", self._symbols)

        while not self._stop_event.is_set():
            await self._stream.handle_message()
