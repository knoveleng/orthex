from __future__ import annotations

import inspect
from typing import Callable

from orthex.data.base import DataProvider

DATA_PROVIDER_REGISTRY: dict[str, type[DataProvider]] = {}


def register(name: str) -> Callable[[type[DataProvider]], type[DataProvider]]:
    def _wrap(cls: type[DataProvider]) -> type[DataProvider]:
        DATA_PROVIDER_REGISTRY[name] = cls
        cls.name = name
        return cls

    return _wrap


def build_provider(provider: str, params: dict) -> DataProvider:
    """`params` is passed as **kwargs to the provider's constructor. Most
    providers (local_json, local_jsonl) list every param explicitly, so a
    typo'd param name (e.g. n_smaples) fails fast right here via TypeError
    instead of silently falling back to a default -- on that TypeError this
    re-raises naming the provider's actual accepted params, since the raw
    Python TypeError alone doesn't say which provider or which params are
    valid. hf_dataset is a deliberate exception: it accepts a **kwargs
    catch-all itself to pass arbitrary datasets.load_dataset() options
    through (dataset config `name`, `trust_remote_code`, `revision`, ...),
    trading that same fail-fast guarantee for flexibility on the params
    that ARE forwarded through it."""
    if provider not in DATA_PROVIDER_REGISTRY:
        raise ValueError(f"Unknown data provider {provider!r}; available: {list(DATA_PROVIDER_REGISTRY)}")
    cls = DATA_PROVIDER_REGISTRY[provider]
    try:
        return cls(**params)
    except TypeError as e:
        accepted = [p for p in inspect.signature(cls.__init__).parameters if p != "self"]
        raise TypeError(
            f"Invalid params for data provider {provider!r} ({e}); accepted params: {accepted}, got: {list(params)}"
        ) from e
