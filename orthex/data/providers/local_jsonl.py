from __future__ import annotations

import json

from orthex.data.base import DataProvider
from orthex.data.registry import register

# Only used when `text_field` isn't explicitly set -- see local_json.py's
# note on auto-detection being a convenience, not a substitute for pinning
# the field down on a file with an unfamiliar schema.
_FIELD_CANDIDATES = ("text", "prompt", "instruction", "input", "goal")


@register("local_jsonl")
class LocalJsonlProvider(DataProvider):
    """Reads one prompt string per line from a JSONL file (`{"text": "..."}`
    or any of `_FIELD_CANDIDATES`). `id` is the file path -- these files are
    typically already scoped to one split, so there's no `split` param."""

    def __init__(self, id: str, n_samples: int = 256, text_field: str | None = None):
        self.path = id
        self.n_samples = n_samples
        self.text_field = text_field

    def load(self) -> list[str]:
        prompts: list[str] = []
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompts.append(self._extract(row))
                if len(prompts) >= self.n_samples:
                    break
        return prompts

    def _extract(self, row: dict) -> str:
        if self.text_field is not None:
            if self.text_field not in row:
                raise KeyError(f"text_field {self.text_field!r} not found in row: {row}")
            return row[self.text_field]
        for key in _FIELD_CANDIDATES:
            if row.get(key):
                return row[key]
        raise KeyError(f"No usable text field ({_FIELD_CANDIDATES}) found in row: {row}; set text_field explicitly")
