from __future__ import annotations

from typing import Protocol


class SelectionScorer(Protocol):
    def score(self, responses: list[str]) -> float:
        """Returns the fraction of responses exhibiting the target
        behavior (e.g. refusal rate)."""
        ...
