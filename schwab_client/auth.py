"""
OAuth setup for the Charles Schwab Trader API.

Schwab's OAuth flow (authorization-code + refresh token) is finicky:
  - The access token expires every 30 minutes.
  - The refresh token expires every 7 days and MUST be renewed by re-doing the
    interactive browser login before it does (there's no way to refresh past
    that window).
  - The initial login requires a real browser redirect back to your
    registered callback URL.

Rather than re-implement that dance by hand, we lean on `schwab-py`
(https://github.com/alexgolec/schwab-py), an actively maintained, MIT
licensed wrapper around Schwab's *official* documented REST/streaming
endpoints. It handles the OAuth handshake, token storage, and silent
access-token refresh for us. Everything else in this project (market_data.py,
streaming.py) is code we own and can swap to a different transport later
without touching the rest of the app.

We deliberately use schwab-py's *manual* login flow
(`client_from_manual_flow`) instead of its "easy"/automatic one. The
automatic flow spins up a temporary local HTTPS server to catch the OAuth
redirect, which forces your browser to show a "connection is not private"
warning (self-signed cert, unavoidable for a https://127.0.0.1 listener) —
and the delay while you read/click through that warning is often enough
for Schwab's short-lived authorization code to expire, which then fails
with `invalid_grant: Authorization code is invalid, expired or revoked`.
The manual flow skips the local server entirely: you copy the redirected
URL straight out of the address bar and paste it back here immediately, so
there's no interstitial and nothing to race against.

First-time setup:
    1. Register an app at https://developer.schwab.com/dashboard/apps and
       request the "Accounts and Trading Production" + "Market Data
       Production" products.
    2. Fill in .env with SCHWAB_API_KEY / SCHWAB_APP_SECRET / SCHWAB_CALLBACK_URL.
       SCHWAB_CALLBACK_URL must exactly match what's registered on the app
       (scheme, host, port, trailing slash) — Schwab validates it even
       though nothing actually listens on that port anymore.
    3. Run `python -m schwab_client.auth` once. It prints a Schwab login
       URL — open it, log in, approve access. Schwab redirects your browser
       to the (dead) callback URL; copy that full URL straight from the
       address bar and paste it back into the terminal when prompted
       (`Redirect URL>`). This writes the token file to SCHWAB_TOKEN_PATH.
    4. Every subsequent run reuses/refreshes that token file automatically
       — you won't see this flow again until the refresh token expires
       (7 days) or the token file is deleted.
"""
from __future__ import annotations

import logging
import os

import schwab

from config import SchwabConfig

logger = logging.getLogger(__name__)


def get_client(cfg: SchwabConfig | None = None) -> "schwab.client.Client":
    """
    Return an authenticated Schwab client: loads the cached token if one
    exists (silently refreshing the access token as needed), otherwise runs
    the interactive manual-copy-paste OAuth login flow once and caches the
    result.
    """
    cfg = cfg or SchwabConfig.from_env()

    if os.path.isfile(cfg.token_path):
        client = schwab.auth.client_from_token_file(cfg.token_path, cfg.api_key, cfg.app_secret)
        logger.info("Loaded cached Schwab token from %s", cfg.token_path)
        return client

    logger.info("No cached token found at %s — starting the interactive login flow.", cfg.token_path)
    client = schwab.auth.client_from_manual_flow(
        api_key=cfg.api_key,
        app_secret=cfg.app_secret,
        callback_url=cfg.callback_url,
        token_path=cfg.token_path,
    )
    logger.info("Authenticated Schwab client ready (token cached at %s)", cfg.token_path)
    return client


if __name__ == "__main__":
    # `python -m schwab_client.auth` — force the interactive login flow so
    # you can do it once, up front, from a real terminal/browser.
    logging.basicConfig(level=logging.INFO)
    get_client()
    print("Login succeeded and token was cached.")
