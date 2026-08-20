import pytest

from orthex.ablation.registry import ABLATION_STRATEGY_REGISTRY, build_ablation_strategy
from orthex.architectures.registry import ARCHITECTURE_ADAPTER_REGISTRY, build_architecture_adapter
from orthex.backends.registry import BACKEND_REGISTRY, build_backend
from orthex.data.registry import DATA_PROVIDER_REGISTRY, build_provider
from orthex.directions.registry import DIRECTION_STRATEGY_REGISTRY, build_direction_strategy
from orthex.selection.scorers.registry import SELECTION_SCORER_REGISTRY, build_scorer

# Importing orthex.pipeline registers every provider/backend/strategy/adapter
# via its side-effect imports -- import it once so these registries are
# populated regardless of test order.
import orthex.pipeline  # noqa: F401,E402


def test_registry_has_all_data_providers():
    assert set(DATA_PROVIDER_REGISTRY) == {"hf_dataset", "local_jsonl", "local_json"}


def test_registry_has_all_backends():
    assert set(BACKEND_REGISTRY) == {"native_hf"}


def test_registry_has_all_direction_strategies():
    assert set(DIRECTION_STRATEGY_REGISTRY) == {"mean_difference"}


def test_registry_has_all_selection_scorers():
    assert set(SELECTION_SCORER_REGISTRY) == {"refusal_phrase_heuristic"}


def test_registry_has_all_ablation_strategies():
    assert set(ABLATION_STRATEGY_REGISTRY) == {"weight_orthogonalization"}


def test_registry_has_all_architecture_adapters():
    assert set(ARCHITECTURE_ADAPTER_REGISTRY) == {"llama", "qwen2", "qwen3", "gemma2", "gemma3", "qwen3_5"}


def test_build_provider_dispatches_to_registered_class_with_params(monkeypatch):
    class FakeProvider:
        def __init__(self, id, n_samples=2):
            self.id = id
            self.n_samples = n_samples

        def load(self):
            return ["fake"] * self.n_samples

    monkeypatch.setitem(DATA_PROVIDER_REGISTRY, "fake", FakeProvider)
    provider = build_provider("fake", {"id": "some-id", "n_samples": 3})
    assert isinstance(provider, FakeProvider)
    assert provider.load() == ["fake", "fake", "fake"]


def test_build_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown data provider"):
        build_provider("nonexistent", {"id": "id"})


def test_build_backend_unknown_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        build_backend("nonexistent", "id")


def test_build_direction_strategy_unknown_raises():
    with pytest.raises(ValueError, match="Unknown direction strategy"):
        build_direction_strategy("nonexistent")


def test_build_scorer_unknown_raises():
    with pytest.raises(ValueError, match="Unknown selection strategy"):
        build_scorer("nonexistent", [])


def test_build_ablation_strategy_unknown_raises():
    with pytest.raises(ValueError, match="Unknown ablation strategy"):
        build_ablation_strategy("nonexistent")


def test_build_architecture_adapter_unknown_raises():
    with pytest.raises(ValueError, match="Unknown architecture adapter"):
        build_architecture_adapter("nonexistent")
