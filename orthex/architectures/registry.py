from __future__ import annotations

from typing import Callable

ARCHITECTURE_ADAPTER_REGISTRY: dict[str, type] = {}


def register(name: str) -> Callable[[type], type]:
    def _wrap(cls: type) -> type:
        ARCHITECTURE_ADAPTER_REGISTRY[name] = cls
        cls.name = name
        return cls

    return _wrap


def build_architecture_adapter(name: str):
    if name not in ARCHITECTURE_ADAPTER_REGISTRY:
        raise ValueError(f"Unknown architecture adapter {name!r}; available: {list(ARCHITECTURE_ADAPTER_REGISTRY)}")
    return ARCHITECTURE_ADAPTER_REGISTRY[name]()
