from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerSite:
    layer: int
    site: str  # "resid_pre" | "resid_post"

    def __str__(self) -> str:
        return f"L{self.layer}/{self.site}"
