from __future__ import annotations

from typing import Protocol

import torch

from orthex.capture.types import LayerSite


class DirectionStrategy(Protocol):
    def compute(
        self,
        positive_acts: dict[LayerSite, torch.Tensor],
        negative_acts: dict[LayerSite, torch.Tensor],
    ) -> dict[LayerSite, torch.Tensor]: ...
