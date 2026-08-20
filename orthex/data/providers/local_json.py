from __future__ import annotations

import json

from orthex.data.base import DataProvider
from orthex.data.registry import register

# "prompt" checked first: local eval sets (e.g. HarmBench/AdvBench-style
# files) consistently expose a raw `prompt` field, and some also carry a
# `text` field holding a full instruction-template block (not the bare
# prompt) that would be the wrong thing to pick by default. Only used when
# `text_field` isn't explicitly set -- auto-detection is a convenience, not
# a substitute for pinning the field down on a file with an unfamiliar or
# ambiguous schema.
_FIELD_CANDIDATES = ("prompt", "text", "instruction", "goal", "input")


@register("local_json")
class LocalJsonProvider(DataProvider):
    """Reads a single JSON file containing an array of objects (e.g.
    HarmBench/AdvBench-style eval sets), each with a usable text field. `id`
    is the file path -- these files are typically already scoped to one
    split (e.g. a `*_test.json` held-out set), so there's no `split` param."""

    def __init__(self, id: str, n_samples: int = 32, text_field: str | None = None):
        self.path = id
        self.n_samples = n_samples
        self.text_field = text_field

    def load(self) -> list[str]:
        with open(self.path) as fh:
            rows = json.load(fh)
        return [self._extract(row) for row in rows[: self.n_samples]]

    def _extract(self, row: dict) -> str:
        if self.text_field is not None:
            if self.text_field not in row:
                raise KeyError(f"text_field {self.text_field!r} not found in row: {row}")
            return row[self.text_field]
        for key in _FIELD_CANDIDATES:
            if row.get(key):
                return row[key]
        raise KeyError(f"No usable text field ({_FIELD_CANDIDATES}) found in row: {row}; set text_field explicitly")
