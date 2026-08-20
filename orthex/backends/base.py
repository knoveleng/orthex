from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    name: str
    model: Any
    tokenizer: Any

    @abstractmethod
    def __init__(self, model_id: str, dtype: str = "bfloat16"): ...
