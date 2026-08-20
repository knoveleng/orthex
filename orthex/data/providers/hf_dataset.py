from __future__ import annotations

from orthex.data.base import DataProvider
from orthex.data.registry import register


@register("hf_dataset")
class HFDatasetProvider(DataProvider):
    """Role-agnostic HF Hub dataset loader. Both `mlabonne/harmful_behaviors`
    and `mlabonne/harmless_alpaca` expose a single `text` column with
    pre-existing train/test splits, so text_field's default covers the v1
    datasets without per-dataset special-casing.

    Unlike local_json/local_jsonl, this provider accepts **kwargs and
    forwards them straight to datasets.load_dataset() -- that call has a lot
    of legitimate optional params (`name` for a dataset's sub-config,
    `trust_remote_code`, `revision`, `streaming`, `data_dir`, ...) that
    aren't worth hand-enumerating here. The trade-off: a typo among these
    passthrough kwargs isn't caught at provider-construction time (params
    that ARE explicit -- id, split, text_field, n_samples -- still fail
    fast via the usual TypeError), it surfaces later as whatever error
    load_dataset() itself raises for an unrecognized kwarg."""

    def __init__(self, id: str, split: str = "train", text_field: str = "text", n_samples: int = 256, **kwargs):
        self.id = id
        self.split = split
        self.text_field = text_field
        self.n_samples = n_samples
        self.load_kwargs = kwargs

    def load(self) -> list[str]:
        from datasets import load_dataset

        ds = load_dataset(self.id, split=self.split, **self.load_kwargs)
        n = min(self.n_samples, len(ds))
        return list(ds[self.text_field][:n])
