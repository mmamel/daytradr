"""
Figures out today's equity market session (open/close, or "closed today" for
weekends/holidays/half-days) so trading_mode knows when to start and stop
pulling data.

We ask Schwab's own /marketdata/v1/markets endpoint rather than hardcoding
9:30-16:00 ET, because it correctly reflects holidays and early-close days
(e.g. the day after Thanksgiving, 1pm ET close) without us maintaining a
holiday calendar by hand.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from schwab.client import Client

from schwab_client.market_data import MarketDataClient

logger = logging.getLogger(__name__)

# The exchange itself only knows Eastern time — NYSE opens/closes on the ET
# clock no matter where you are, so all the actual date/hours math below is
# anchored to EASTERN. PACIFIC is used purely for displaying times in the
# logs, since that's the timezone that's actually meaningful to a reader in
# Washington State.
EASTERN = ZoneInfo("America/New_York")
PACIFIC = ZoneInfo("America/Los_Angeles")


def format_pacific(dt: datetime) -> str:
    return dt.astimezone(PACIFIC).strftime("%Y-%m-%d %I:%M %p %Z")


@dataclass(frozen=True)
class MarketSession:
    is_open_today: bool
    open_time: datetime | None  # tz-aware, UTC
    close_time: datetime | None  # tz-aware, UTC


def get_today_session(market_data: MarketDataClient, *, when: datetime | None = None) -> MarketSession:
    """Query Schwab for today's regular equity session. Falls back to a plain
    9:30-16:00 America/New_York window (Mon-Fri) if the API call fails, so a
    transient outage doesn't stop trading_mode from ever starting."""
    when = when or datetime.now(tz=EASTERN)

    try:
        payload = market_data.get_market_hours(Client.MarketHours.Market.EQUITY, date=when)
        session = _parse_market_hours(payload)
    except Exception:
        logger.exception("Failed to fetch market hours from Schwab; falling back to static schedule")
        session = _static_fallback_session(when)

    _log_outcome(session, when)
    return session


def _log_outcome(session: MarketSession, when: datetime) -> None:
    """One INFO log for whichever of the three outcomes applies, with times
    shown in Pacific."""
    if not session.is_open_today or when >= session.close_time:
        logger.info("Market is closed today (weekend, holiday, or already past today's close).")
    elif when < session.open_time:
        logger.info("Market has not opened yet today; opens at %s.", format_pacific(session.open_time))
    else:
        logger.info("Market is currently open; closes at %s.", format_pacific(session.close_time))


def _parse_market_hours(payload: dict) -> MarketSession:
    # Response shape: {"equity": {"<product>": {"isOpen": bool, "sessionHours":
    # {"regularMarket": [{"start": iso8601, "end": iso8601}]}}}}
    equity = payload.get("equity", {})
    for product in equity.values():
        if not product.get("isOpen"):
            continue
        regular = product.get("sessionHours", {}).get("regularMarket", [])
        if not regular:
            continue
        start = datetime.fromisoformat(regular[0]["start"]).astimezone(timezone.utc)
        end = datetime.fromisoformat(regular[0]["end"]).astimezone(timezone.utc)
        return MarketSession(is_open_today=True, open_time=start, close_time=end)

    return MarketSession(is_open_today=False, open_time=None, close_time=None)


def _static_fallback_session(when: datetime) -> MarketSession:
    if when.weekday() >= 5:  # Sat/Sun
        return MarketSession(is_open_today=False, open_time=None, close_time=None)
    open_time = when.replace(hour=9, minute=30, second=0, microsecond=0).astimezone(timezone.utc)
    close_time = when.replace(hour=16, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    return MarketSession(is_open_today=True, open_time=open_time, close_time=close_time)
