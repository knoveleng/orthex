from __future__ import annotations

from typing import Callable

from orthex.backends.base import Backend

BACKEND_REGISTRY: dict[str, type[Backend]] = {}


def register(name: str) -> Callable[[type[Backend]], type[Backend]]:
    def _wrap(cls: type[Backend]) -> type[Backend]:
        BACKEND_REGISTRY[name] = cls
        cls.name = name
        return cls

    return _wrap


def build_backend(backend: str, model_id: str, dtype: str = "bfloat16") -> Backend:
    if backend not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend {backend!r}; available: {list(BACKEND_REGISTRY)}")
    return BACKEND_REGISTRY[backend](model_id, dtype)
