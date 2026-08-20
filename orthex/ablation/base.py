from __future__ import annotations

from typing import Protocol

import torch


class AblationStrategy(Protocol):
    def apply(self, model, adapter, direction: torch.Tensor, targets: list[str]) -> None: ...
