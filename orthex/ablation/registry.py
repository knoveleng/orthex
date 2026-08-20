from __future__ import annotations

from typing import Callable

ABLATION_STRATEGY_REGISTRY: dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    def _wrap(cls: type) -> type:
        ABLATION_STRATEGY_REGISTRY[name] = cls
        cls.name = name
        return cls

    return _wrap


def build_ablation_strategy(strategy: str):
    if strategy not in ABLATION_STRATEGY_REGISTRY:
        raise ValueError(f"Unknown ablation strategy {strategy!r}; available: {list(ABLATION_STRATEGY_REGISTRY)}")
    return ABLATION_STRATEGY_REGISTRY[strategy]()
