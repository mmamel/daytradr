"""
Centralized configuration for the daytrader app.

All secrets are read from environment variables (loaded from a local .env file
via python-dotenv). Never hardcode API keys/secrets in source.

Required env vars (see .env.example):
    SCHWAB_API_KEY       - "App Key" from your app on developer.schwab.com
    SCHWAB_APP_SECRET    - "Secret" from your app on developer.schwab.com
    SCHWAB_CALLBACK_URL  - Callback URL registered on your app (e.g. https://127.0.0.1:8182)
    SCHWAB_TOKEN_PATH    - Where the OAuth token is cached on disk (created on first login)

Optional:
    DAYTRADER_SYMBOLS         - comma separated list of tickers to watch, e.g. "AAPL,TSLA,SPY"
    DAYTRADER_POLL_INTERVAL   - seconds between REST polls when not streaming (default 5)
    DAYTRADER_STRATEGY        - name of the strategy to run, from strategies.available_strategies()
                                (default "sma_crossover"). See strategies/ to add your own.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load a .env file from the project root if present. Does not overwrite
# variables already set in the real environment.
load_dotenv(Path(__file__).resolve().parent / ".env")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{name}'. "
            f"Copy env.example to .env and fill it in."
        )
    return value


@dataclass(frozen=True)
class SchwabConfig:
    api_key: str
    app_secret: str
    callback_url: str
    token_path: str

    @classmethod
    def from_env(cls) -> "SchwabConfig":
        return cls(
            api_key=_require_env("SCHWAB_API_KEY"),
            app_secret=_require_env("SCHWAB_APP_SECRET"),
            callback_url=_require_env("SCHWAB_CALLBACK_URL"),
            token_path=os.environ.get(
                "SCHWAB_TOKEN_PATH", str(Path(__file__).resolve().parent / "schwab_token.json")
            ),
        )


@dataclass(frozen=True)
class AppConfig:
    symbols: list[str] = field(default_factory=lambda: ["AAPL"])
    poll_interval_seconds: float = 5.0
    strategy_name: str = "sma_crossover"

    @classmethod
    def from_env(cls) -> "AppConfig":
        raw_symbols = os.environ.get("DAYTRADER_SYMBOLS", "AAPL")
        symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
        interval = float(os.environ.get("DAYTRADER_POLL_INTERVAL", "5"))
        strategy_name = os.environ.get("DAYTRADER_STRATEGY", "sma_crossover")
        return cls(symbols=symbols, poll_interval_seconds=interval, strategy_name=strategy_name)
