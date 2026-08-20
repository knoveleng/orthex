import pytest

from orthex.cli import main


def test_cli_requires_config(monkeypatch):
    with pytest.raises(SystemExit):
        main([])


def test_cli_runs_pipeline_with_parsed_config(monkeypatch, tmp_path):
    calls = {}

    def fake_run(cfg, output_dir):
        calls["cfg"] = cfg
        calls["output_dir"] = output_dir
        return {"evaluation": {"refusal_rate": {}, "perplexity": {}}}

    monkeypatch.setattr("orthex.cli.pipeline.run", fake_run)

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        """
model: {id: some/model, architecture_adapter: llama}
data:
  train:
    positive: {provider: hf_dataset, params: {id: pos, n_samples: 4}}
    negative: {provider: hf_dataset, params: {id: neg, n_samples: 4}}
  test: {provider: hf_dataset, params: {id: pos, n_samples: 2}}
export: {}
"""
    )

    rc = main(["--config", str(config_path), "--set", "data.train.positive.params.n_samples=8"])

    assert rc == 0
    assert calls["cfg"].model.id == "some/model"
    assert calls["cfg"].data.train.positive.params["n_samples"] == 8
    assert calls["output_dir"].endswith("some/model".split("/")[-1])


def test_cli_set_overrides_model_id_and_adapter(monkeypatch, tmp_path):
    # There are no dedicated --model-id/--architecture-adapter flags:
    # --set is the one mechanism for overriding any field, including these.
    calls = {}

    def fake_run(cfg, output_dir):
        calls["cfg"] = cfg
        return {"evaluation": {"refusal_rate": {}, "perplexity": {}}}

    monkeypatch.setattr("orthex.cli.pipeline.run", fake_run)

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        """
model: {id: some/model, architecture_adapter: llama, dtype: bfloat16}
data:
  train:
    positive: {provider: hf_dataset, params: {id: pos, n_samples: 4}}
    negative: {provider: hf_dataset, params: {id: neg, n_samples: 4}}
  test: {provider: hf_dataset, params: {id: pos, n_samples: 2}}
export: {}
"""
    )

    rc = main(
        [
            "--config",
            str(config_path),
            "--set",
            "model.id=google/gemma-2-2b-it",
            "--set",
            "model.architecture_adapter=gemma2",
            "--set",
            "model.dtype=float16",
        ]
    )

    assert rc == 0
    assert calls["cfg"].model.id == "google/gemma-2-2b-it"
    assert calls["cfg"].model.architecture_adapter == "gemma2"
    assert calls["cfg"].model.dtype == "float16"


def test_cli_repeated_set_for_same_key_last_one_wins(monkeypatch, tmp_path):
    calls = {}

    def fake_run(cfg, output_dir):
        calls["cfg"] = cfg
        return {"evaluation": {"refusal_rate": {}, "perplexity": {}}}

    monkeypatch.setattr("orthex.cli.pipeline.run", fake_run)

    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        """
model: {id: some/model, architecture_adapter: llama}
data:
  train:
    positive: {provider: hf_dataset, params: {id: pos, n_samples: 4}}
    negative: {provider: hf_dataset, params: {id: neg, n_samples: 4}}
  test: {provider: hf_dataset, params: {id: pos, n_samples: 2}}
export: {}
"""
    )

    rc = main(["--config", str(config_path), "--set", "model.id=first", "--set", "model.id=second"])

    assert rc == 0
    assert calls["cfg"].model.id == "second"
