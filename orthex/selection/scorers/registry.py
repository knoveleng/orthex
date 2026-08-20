from __future__ import annotations

from typing import Callable

SELECTION_SCORER_REGISTRY: dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    def _wrap(cls: type) -> type:
        SELECTION_SCORER_REGISTRY[name] = cls
        cls.name = name
        return cls

    return _wrap


def build_scorer(strategy: str, blacklist: list[str]):
    if strategy not in SELECTION_SCORER_REGISTRY:
        raise ValueError(f"Unknown selection strategy {strategy!r}; available: {list(SELECTION_SCORER_REGISTRY)}")
    return SELECTION_SCORER_REGISTRY[strategy](blacklist)
