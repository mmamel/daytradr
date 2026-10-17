"""
Thin, typed wrapper around the Schwab Trader API's Market Data endpoints.

This is the "pull data" half of the app. It intentionally does NOT place
trades (see the trading section, to be built later). Every method returns
plain dicts/lists (already-parsed JSON) so the rest of the app never has to
know it's talking to Schwab specifically.

Docs: https://developer.schwab.com/products/trader-api--individual
Base URL: https://api.schwabapi.com/marketdata/v1
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Iterable

import schwab
from schwab.client import Client

from schwab_client.auth import get_client
from config import SchwabConfig

logger = logging.getLogger(__name__)


def _unwrap(response) -> Any:
    """Raise on HTTP errors, otherwise return the parsed JSON body."""
    response.raise_for_status()
    return response.json()


class MarketDataClient:
    """
    High-level, synchronous REST client for on-demand / polled market data.

    For a continuously-updating feed (what you actually want while the
    market is open) see schwab_client.streaming.LiveQuoteStream instead —
    this class is best for one-off snapshots, historical backfill, and
    anything you'd call every few seconds rather than every tick.
    """

    def __init__(self, cfg: SchwabConfig | None = None, client: Client | None = None):
        self._client = client or get_client(cfg)

    @property
    def client(self) -> Client:
        """The underlying authenticated schwab-py Client, e.g. for streaming.py."""
        return self._client

    # ------------------------------------------------------------------
    # Real-time-ish snapshots (REST; a few hundred ms of staleness)
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str) -> dict:
        """Single-symbol snapshot quote: last/bid/ask/volume/day range/etc."""
        return _unwrap(self._client.get_quote(symbol))[symbol]

    def get_quotes(self, symbols: Iterable[str]) -> dict[str, dict]:
        """Snapshot quotes for multiple symbols in one round trip."""
        return _unwrap(self._client.get_quotes(list(symbols)))

    # ------------------------------------------------------------------
    # Historical bars (for indicator warm-up / feature engineering)
    # ------------------------------------------------------------------

    def get_intraday_bars(
        self,
        symbol: str,
        *,
        minutes_per_bar: int = 1,
        start_datetime: datetime | None = None,
        end_datetime: datetime | None = None,
        extended_hours: bool = False,
    ) -> list[dict]:
        """
        Minute-resolution OHLCV candles, e.g. to warm-start technical
        indicators (moving averages, RSI, etc.) before the live stream has
        accumulated enough history on its own.

        minutes_per_bar must be one of {1, 5, 10, 15, 30}.
        """
        method_by_interval = {
            1: self._client.get_price_history_every_minute,
            5: self._client.get_price_history_every_five_minutes,
            10: self._client.get_price_history_every_ten_minutes,
            15: self._client.get_price_history_every_fifteen_minutes,
            30: self._client.get_price_history_every_thirty_minutes,
        }
        if minutes_per_bar not in method_by_interval:
            raise ValueError(f"minutes_per_bar must be one of {sorted(method_by_interval)}")

        response = method_by_interval[minutes_per_bar](
            symbol,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            need_extended_hours_data=extended_hours,
        )
        return _unwrap(response).get("candles", [])

    def get_daily_bars(self, symbol: str, *, weekly: bool = False) -> list[dict]:
        """Daily (or weekly) OHLCV candles, back to ~1985 for most symbols."""
        method = self._client.get_price_history_every_week if weekly else self._client.get_price_history_every_day
        return _unwrap(method(symbol)).get("candles", [])

    # ------------------------------------------------------------------
    # Context / screening data
    # ------------------------------------------------------------------

    def get_option_chain(self, symbol: str, **kwargs) -> dict:
        """Full option chain (see Client.get_option_chain for kwargs like strike_count, contract_type)."""
        return _unwrap(self._client.get_option_chain(symbol, **kwargs))

    def get_movers(
        self,
        index: "Client.Movers.Index" = None,
        sort_order: "Client.Movers.SortOrder | None" = None,
    ) -> list[dict]:
        """Top-10 movers for an index (e.g. $SPX, $DJI, $COMPX, NASDAQ, NYSE)."""
        index = index or schwab.client.Client.Movers.Index.SPX
        return _unwrap(self._client.get_movers(index, sort_order=sort_order)).get("screeners", [])

    def get_market_hours(self, market: "Client.MarketHours.Market", date: datetime | None = None) -> dict:
        """Open/close times (and early-close/holiday info) for a market."""
        return _unwrap(self._client.get_market_hours([market], date=date))
