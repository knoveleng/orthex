from __future__ import annotations

from typing import Callable

DIRECTION_STRATEGY_REGISTRY: dict[str, type] = {}

# TODO v1.1: geometric_median.py -- steadier variant, robust to outlier
# prompts. Not needed for v1 (mean_difference matches the Arditi et al.
# baseline used by the reference implementation this toolkit follows).


def register(name: str) -> Callable[[type], type]:
    def _wrap(cls: type) -> type:
        DIRECTION_STRATEGY_REGISTRY[name] = cls
        cls.name = name
        return cls

    return _wrap


def build_direction_strategy(strategy: str):
    if strategy not in DIRECTION_STRATEGY_REGISTRY:
        raise ValueError(f"Unknown direction strategy {strategy!r}; available: {list(DIRECTION_STRATEGY_REGISTRY)}")
    return DIRECTION_STRATEGY_REGISTRY[strategy]()
