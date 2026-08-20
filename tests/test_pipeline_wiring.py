import json

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

import orthex.pipeline as pipeline  # noqa: F401  (registers everything)
from orthex.backends.base import Backend
from orthex.backends.registry import BACKEND_REGISTRY
from orthex.config import (
    AblationConfig,
    CaptureConfig,
    DataConfig,
    DataSourceSpec,
    DataSplitConfig,
    DirectionsConfig,
    EvaluationConfig,
    ExportConfig,
    ModelConfig,
    OrthexConfig,
    SelectionConfig,
)
from orthex.data.base import DataProvider
from orthex.data.registry import DATA_PROVIDER_REGISTRY

TRAIN_N = 4
TEST_N = 2


@pytest.fixture(scope="module")
def tiny_llama():
    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


class _FakeDataProvider(DataProvider):
    """`id` distinguishes positive vs negative so the two providers return
    genuinely different prompt pools -- a mean-difference direction computed
    from identical pos/neg activations would degenerate to a zero vector."""

    _POOLS = {
        "pos": ["do something harmful", "explain how to break in", "describe a dangerous act", "help me cause harm"],
        "neg": ["write a poem", "describe a recipe", "explain a topic", "summarize a book"],
    }

    def __init__(self, id: str, n_samples: int = TRAIN_N):
        self.id = id
        self.n_samples = n_samples

    def load(self) -> list[str]:
        pool = self._POOLS[self.id]
        return (pool * self.n_samples)[: self.n_samples]


def _fake_split(n_samples: int) -> DataSplitConfig:
    return DataSplitConfig(
        positive=DataSourceSpec(provider="fake", params={"id": "pos", "n_samples": n_samples}),
        negative=DataSourceSpec(provider="fake", params={"id": "neg", "n_samples": n_samples}),
    )


def _build_cfg(tmp_path, selection: SelectionConfig, export: ExportConfig | None = None) -> OrthexConfig:
    return OrthexConfig(
        model=ModelConfig(id="unused/tiny", architecture_adapter="llama", backend="fake_backend", dtype="bfloat16"),
        data=DataConfig(
            train=_fake_split(TRAIN_N),
            test=DataSourceSpec(provider="fake", params={"id": "pos", "n_samples": TEST_N}),
        ),
        capture=CaptureConfig(sites=["resid_pre", "resid_post"], position=-1),
        directions=DirectionsConfig(strategy="mean_difference", layer_range=[1, -1]),
        selection=selection,
        ablation=AblationConfig(strategy="weight_orthogonalization", targets=["embed", "attn_out", "mlp_out"]),
        export=export or ExportConfig(push_to_hub=False, output_dir=str(tmp_path)),
        evaluation=EvaluationConfig(refusal_rate=True, perplexity_delta=True, benchmark_subset=None),
    )


def _default_selection(**overrides) -> SelectionConfig:
    base = dict(
        strategy="refusal_phrase_heuristic",
        eval_top_n=20,
        blacklist=[],
        sites=["resid_pre"],
        max_new_tokens=4,
        generation_batch_size=2,
    )
    base.update(overrides)
    return SelectionConfig(**base)


def _patch_fake_backend(monkeypatch, model, tokenizer):
    monkeypatch.setitem(DATA_PROVIDER_REGISTRY, "fake", _FakeDataProvider)

    class _FakeBackend(Backend):
        def __init__(self, model_id: str, dtype: str = "bfloat16"):
            self.model = model
            self.tokenizer = tokenizer

    monkeypatch.setitem(BACKEND_REGISTRY, "fake_backend", _FakeBackend)


def test_pipeline_run_end_to_end_with_tiny_model(tiny_llama, tmp_path, monkeypatch):
    model, tokenizer = tiny_llama
    _patch_fake_backend(monkeypatch, model, tokenizer)

    cfg = _build_cfg(tmp_path, _default_selection())
    output_dir = tmp_path / "run"
    result = pipeline.run(cfg, output_dir)

    assert (output_dir / "evaluation_report.json").exists()
    assert (output_dir / "config.json").exists()  # save_pretrained ran

    readme = (output_dir / "README.md").read_text()
    assert readme.startswith("---\n")
    assert "base_model: unused/tiny" in readme

    on_disk = json.loads((output_dir / "evaluation_report.json").read_text())
    assert on_disk == result
    assert "selected_candidate" in result
    assert result["evaluation"]["refusal_rate"]["pre"] is not None
    assert result["evaluation"]["refusal_rate"]["post"] is not None
    assert result["evaluation"]["perplexity"]["pre"] is not None
    assert result["evaluation"]["perplexity"]["post"] is not None

    for phase in ("pre", "post"):
        samples = result["refusal_samples"][phase]
        assert len(samples) == TEST_N
        for sample in samples:
            assert set(sample) == {"prompt", "response", "is_refusal"}


def test_force_layer_bypasses_auto_pick(tiny_llama, tmp_path, monkeypatch):
    model, tokenizer = tiny_llama
    _patch_fake_backend(monkeypatch, model, tokenizer)

    # 2-layer tiny model + layer_range [1, -1] leaves layer 1 as the only
    # valid candidate -- force it explicitly rather than auto-picking.
    cfg = _build_cfg(tmp_path, _default_selection(force_layer=1, force_site="resid_pre"))
    result = pipeline.run(cfg, tmp_path / "run")

    assert result["selected_candidate"] == {"layer": 1, "site": "resid_pre", "refusal_rate": None}
    assert result["selection_report"]["forced"] is True


def test_force_layer_unknown_candidate_raises(tiny_llama, tmp_path, monkeypatch):
    model, tokenizer = tiny_llama
    _patch_fake_backend(monkeypatch, model, tokenizer)

    cfg = _build_cfg(tmp_path, _default_selection(force_layer=99, force_site="resid_pre"))
    with pytest.raises(ValueError, match="Forced candidate"):
        pipeline.run(cfg, tmp_path / "run")


def test_push_to_hub_forwards_repo_id_and_push_kwargs(tiny_llama, tmp_path, monkeypatch):
    model, tokenizer = tiny_llama
    _patch_fake_backend(monkeypatch, model, tokenizer)

    calls = []
    monkeypatch.setattr(model, "push_to_hub", lambda repo_id, **kw: calls.append(("model", repo_id, kw)))
    monkeypatch.setattr(tokenizer, "push_to_hub", lambda repo_id, **kw: calls.append(("tokenizer", repo_id, kw)))

    export = ExportConfig(
        push_to_hub=True,
        repo_id="some-org/some-model",
        output_dir=str(tmp_path),
        push_kwargs={"private": True},
    )
    cfg = _build_cfg(tmp_path, _default_selection(force_layer=1, force_site="resid_pre"), export=export)
    pipeline.run(cfg, tmp_path / "run")

    assert calls == [
        ("model", "some-org/some-model", {"private": True}),
        ("tokenizer", "some-org/some-model", {"private": True}),
    ]
