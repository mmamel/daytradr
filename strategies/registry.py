"""
A name -> class registry so a strategy can be selected by a plain string
(from an env var, a CLI flag, or a backtest sweep) instead of an import.

To add a new strategy:
    1. Create strategies/my_strategy.py
    2. Subclass StrategyBase, implement predict()
    3. Decorate the class with @register("my_strategy")

That's it — nothing else needs to change. New files under strategies/ are
auto-discovered (see `_discover` below) the first time any strategy is
requested, so `strategies/__init__.py` never needs manual updates when you
add one.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Type

from strategies.base import StrategyBase

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Type[StrategyBase]] = {}
_NON_STRATEGY_MODULES = {"base", "registry", "features"}
_discovered = False


def register(name: str):
    """Class decorator: register a StrategyBase subclass under `name`."""

    def _decorator(cls: Type[StrategyBase]) -> Type[StrategyBase]:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"Strategy name '{name}' is already registered to {existing.__qualname__}; "
                f"choose a different name for {cls.__qualname__}."
            )
        _REGISTRY[name] = cls
        return cls

    return _decorator


def _discover() -> None:
    """Import every module under strategies/ once, so their @register(...)
    decorators run and populate _REGISTRY.

    One module failing to import (e.g. an ML strategy whose optional
    dependency, like torch or scikit-learn, isn't installed yet) shouldn't
    take down every other strategy with it — we log and skip instead of
    raising, so `available_strategies()` always reflects whatever actually
    works in the current environment."""
    global _discovered
    if _discovered:
        return
    import strategies as _pkg

    for _, module_name, _ in pkgutil.iter_modules(_pkg.__path__):
        if module_name in _NON_STRATEGY_MODULES:
            continue
        try:
            importlib.import_module(f"strategies.{module_name}")
        except ImportError:
            logger.warning(
                "Skipping strategies.%s — it failed to import (likely a missing optional "
                "dependency). It won't show up in available_strategies() until that's fixed.",
                module_name,
                exc_info=True,
            )
    _discovered = True


def get_strategy(name: str, **kwargs) -> StrategyBase:
    """Instantiate a registered strategy by name, e.g. get_strategy("random_forest", n_estimators=300)."""
    return get_strategy_class(name)(**kwargs)


def get_strategy_class(name: str) -> Type[StrategyBase]:
    """
    The registered class itself, not an instance — for callers that need to
    introspect a strategy's constructor (e.g. modes/train_mode.py checking
    which hyperparameters a given strategy actually accepts) before
    deciding what to pass it.
    """
    _discover()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown strategy '{name}'. Available: {available_strategies()}") from None


def available_strategies() -> list[str]:
    _discover()
    return sorted(_REGISTRY)
