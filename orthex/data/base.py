from __future__ import annotations

from abc import ABC, abstractmethod


class DataProvider(ABC):
    """Split, sample count, and every other provider-specific knob are
    fixed at construction time (from the config's `params` dict) -- load()
    takes no arguments."""

    name: str

    @abstractmethod
    def load(self) -> list[str]:
        """Returns the configured prompt strings."""
