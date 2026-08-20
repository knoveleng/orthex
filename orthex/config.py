from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    id: str
    architecture_adapter: str  # which family this model belongs to; used
    # throughout the pipeline (capture, selection, ablation), not just at
    # export -- lives with id/backend/dtype rather than under `export`
    backend: str = "native_hf"
    dtype: str = "bfloat16"


@dataclass
class DataSourceSpec:
    """One provider + its arbitrary constructor kwargs. `params` is passed
    straight through as **kwargs to the registered provider class -- each
    provider declares its own params (id, split, n_samples, text_field,
    ...), nothing generalizes across provider shapes at the config layer."""

    provider: str
    params: dict = field(default_factory=dict)
    name: str | None = None  # optional human-readable label, for reports/logs


@dataclass
class DataSplitConfig:
    positive: DataSourceSpec
    negative: DataSourceSpec


@dataclass
class DataConfig:
    train: DataSplitConfig  # positive+negative needed to compute the direction
    test: DataSourceSpec  # positive only -- refusal_rate and perplexity are
    # both measured on the same held-out behavior-eliciting prompts; there's
    # no use for a separate negative/harmless test set


@dataclass
class CaptureConfig:
    sites: list[str] = field(default_factory=lambda: ["resid_pre", "resid_post"])
    position: int = -1


@dataclass
class DirectionsConfig:
    strategy: str = "mean_difference"
    layer_range: list[int] = field(default_factory=lambda: [1, -1])


@dataclass
class SelectionConfig:
    strategy: str = "refusal_phrase_heuristic"
    eval_top_n: int = 20
    blacklist: list[str] = field(default_factory=list)
    sites: list[str] = field(default_factory=lambda: ["resid_pre"])
    max_new_tokens: int = 32
    generation_batch_size: int = 16
    # Bypasses the generation-based auto-pick entirely and uses this exact
    # (layer, site) candidate's direction -- for a human reviewing
    # selection_report.ranked from a prior run and picking one directly
    # instead of trusting the heuristic tie-break.
    force_layer: int | None = None
    force_site: str | None = None


@dataclass
class AblationConfig:
    strategy: str = "weight_orthogonalization"
    targets: list[str] = field(default_factory=lambda: ["embed", "attn_out", "mlp_out"])


@dataclass
class ExportConfig:
    push_to_hub: bool = False
    repo_id: str | None = None
    output_dir: str = "./out"
    # Forwarded as **kwargs to both model.push_to_hub() and tokenizer.
    # push_to_hub() (private, commit_message, revision, create_pr, ...) --
    # same rationale as HFDatasetProvider's **kwargs: too many legitimate
    # push_to_hub options to hand-enumerate here.
    push_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.push_to_hub and not self.repo_id:
            raise ValueError("export.repo_id must be set when export.push_to_hub is true")


@dataclass
class EvaluationConfig:
    refusal_rate: bool = True
    perplexity_delta: bool = True
    benchmark_subset: str | None = None


@dataclass
class OrthexConfig:
    model: ModelConfig
    data: DataConfig
    capture: CaptureConfig
    directions: DirectionsConfig
    selection: SelectionConfig
    ablation: AblationConfig
    export: ExportConfig
    evaluation: EvaluationConfig


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build(cls, data: dict):
    """Constructs `cls` (a flat dataclass) from a dict, ignoring unknown
    keys. DataConfig is nested and built separately via _build_data_split."""
    valid = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in valid}
    return cls(**kwargs)


def load_config_dict(*paths: str | Path) -> dict:
    """Loads one or more YAML files and deep-merges them in order (later
    paths override earlier ones), returning the raw merged dict."""
    merged: dict = {}
    for path in paths:
        with open(path) as fh:
            merged = _deep_merge(merged, yaml.safe_load(fh) or {})
    return merged


def _build_data_source(data: dict) -> DataSourceSpec:
    return DataSourceSpec(provider=data["provider"], params=data.get("params", {}), name=data.get("name"))


def _build_data_split(data: dict) -> DataSplitConfig:
    return DataSplitConfig(
        positive=_build_data_source(data["positive"]),
        negative=_build_data_source(data["negative"]),
    )


def build_config(data: dict) -> OrthexConfig:
    data = dict(data)
    return OrthexConfig(
        model=_build(ModelConfig, data["model"]),
        data=DataConfig(
            train=_build_data_split(data["data"]["train"]),
            test=_build_data_source(data["data"]["test"]),
        ),
        capture=_build(CaptureConfig, data.get("capture", {})),
        directions=_build(DirectionsConfig, data.get("directions", {})),
        selection=_build(SelectionConfig, data.get("selection", {})),
        ablation=_build(AblationConfig, data.get("ablation", {})),
        export=_build(ExportConfig, data["export"]),
        evaluation=_build(EvaluationConfig, data.get("evaluation", {})),
    )


def load_config(*paths: str | Path) -> OrthexConfig:
    return build_config(load_config_dict(*paths))


def apply_dotted_overrides(data: dict, overrides: dict[str, str]) -> dict:
    """Applies {"model.id": "foo", "model.architecture_adapter": "bar"}
    onto a raw config dict (pre-dataclass-construction), YAML-parsing each
    override value so booleans/ints/null come through typed."""
    data = dict(data)
    for dotted_key, raw_value in overrides.items():
        value = yaml.safe_load(raw_value)
        parts = dotted_key.split(".")
        node = data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return data
