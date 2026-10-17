"""
Entry point for the day trading app.

    python main.py --mode trading    # run continuously, streaming live data
                                       # from market open to close (safe to
                                       # start at any time of day)
    python main.py --mode train --symbol AAPL --strategy random_forest \\
        --days-back 45 --validation-days 5    # train + evaluate a strategy;
                                                # see modes/train_mode.py for
                                                # every train-mode flag
    python main.py --mode testing    # backtesting — not implemented yet
"""
from __future__ import annotations

import argparse
import asyncio
import logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Day trading app")
    parser.add_argument(
        "--mode",
        choices=["trading", "train", "testing"],
        required=True,
        help="'trading' streams live data during market hours; 'train' fits+evaluates a strategy "
        "(see modes/train_mode.py --help); 'testing' backtests (TODO).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    # parse_known_args because --mode train takes its own flags (--symbol,
    # --strategy, etc., see modes/train_mode.py) that this parser doesn't
    # know about — they're forwarded as-is instead of rejected here.
    args, remaining = parser.parse_known_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.mode == "trading":
        from modes import trading_mode

        asyncio.run(trading_mode.run())
    elif args.mode == "train":
        from modes import train_mode

        train_mode.main(remaining)
    else:
        from modes import testing_mode

        testing_mode.run()


if __name__ == "__main__":
    main()
