import json
from unittest.mock import patch

from orthex.data.providers.hf_dataset import HFDatasetProvider
from orthex.data.providers.local_json import LocalJsonProvider
from orthex.data.providers.local_jsonl import LocalJsonlProvider


class _FakeHFDataset(dict):
    """Minimal stand-in for a datasets.Dataset: dict-like column access
    (ds["text"]) plus len()."""

    def __len__(self):
        return len(next(iter(self.values())))


def test_hf_dataset_provider_forwards_unknown_kwargs_to_load_dataset():
    fake_ds = _FakeHFDataset({"text": ["a", "b", "c"]})
    with patch("datasets.load_dataset", return_value=fake_ds) as mock_load:
        provider = HFDatasetProvider(
            id="some/dataset", split="train", n_samples=2, name="clean", trust_remote_code=True
        )
        result = provider.load()

    assert result == ["a", "b"]
    mock_load.assert_called_once_with("some/dataset", split="train", name="clean", trust_remote_code=True)


def test_hf_dataset_provider_explicit_params_still_typecheck():
    try:
        HFDatasetProvider()  # missing required `id`
        raise AssertionError("expected TypeError for missing required id")
    except TypeError:
        pass


def test_local_json_provider_reads_array_and_prefers_prompt_field(tmp_path):
    path = tmp_path / "harmbench_test.json"
    path.write_text(
        json.dumps(
            [
                {"prompt": "first prompt", "id": "a", "category": "standard"},
                {"prompt": "second prompt", "text": "a longer template block"},
                {"prompt": "third prompt"},
            ]
        )
    )
    provider = LocalJsonProvider(id=str(path), n_samples=3)
    assert provider.load() == ["first prompt", "second prompt", "third prompt"]


def test_local_json_provider_respects_n_samples(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([{"prompt": f"p{i}"} for i in range(10)]))
    provider = LocalJsonProvider(id=str(path), n_samples=3)
    assert provider.load() == ["p0", "p1", "p2"]


def test_local_jsonl_provider_still_reads_line_delimited(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "line one"}\n{"text": "line two"}\n')
    provider = LocalJsonlProvider(id=str(path), n_samples=2)
    assert provider.load() == ["line one", "line two"]


def test_local_json_provider_explicit_text_field_overrides_auto_detect(tmp_path):
    path = tmp_path / "alpaca.json"
    path.write_text(
        json.dumps(
            [{"prompt": "sniped by auto-detect", "instruction": "the field we actually want"}],
        )
    )
    provider = LocalJsonProvider(id=str(path), n_samples=1, text_field="instruction")
    assert provider.load() == ["the field we actually want"]


def test_local_json_provider_explicit_text_field_missing_raises(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps([{"prompt": "hi"}]))
    provider = LocalJsonProvider(id=str(path), n_samples=1, text_field="nope")
    try:
        provider.load()
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_local_jsonl_provider_explicit_text_field_overrides_auto_detect(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "sniped by auto-detect", "goal": "the field we actually want"}\n')
    provider = LocalJsonlProvider(id=str(path), n_samples=1, text_field="goal")
    assert provider.load() == ["the field we actually want"]
