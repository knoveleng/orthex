from orthex.config import apply_dotted_overrides, build_config, load_config_dict


def test_default_config_round_trips():
    raw = load_config_dict("configs/default.yaml")
    cfg = build_config(raw)
    assert cfg.model.id == "meta-llama/Llama-3.2-3B-Instruct"
    assert cfg.data.train.positive.provider == "local_json"
    assert cfg.data.train.positive.params["id"] == "data/harmful_behaviors_train.json"
    assert cfg.data.train.negative.params["id"] == "data/harmless_alpaca_train.json"
    assert cfg.capture.sites == ["resid_pre", "resid_post"]
    assert cfg.directions.layer_range == [1, -1]
    assert cfg.selection.eval_top_n == 20
    assert cfg.ablation.targets == ["embed", "attn_out", "mlp_out"]
    assert cfg.model.architecture_adapter == "llama"
    assert cfg.evaluation.benchmark_subset is None


def test_default_config_test_source_is_positive_only():
    raw = load_config_dict("configs/default.yaml")
    cfg = build_config(raw)
    assert cfg.data.train.positive.params["id"] == "data/harmful_behaviors_train.json"
    # data.test is a single DataSourceSpec (no .negative) -- refusal_rate
    # and perplexity are both measured on this one held-out prompt set
    assert cfg.data.test.provider == "local_json"
    assert cfg.data.test.params["id"].endswith("harmbench_test.json")
    assert not hasattr(cfg.data.test, "negative")


def test_data_source_params_are_free_form_per_provider():
    raw = load_config_dict("configs/default.yaml")
    cfg = build_config(raw)
    assert cfg.data.train.positive.params["n_samples"] == 256
    assert cfg.data.test.params["n_samples"] == 32
    # positive/negative may have independent sample counts -- no shared
    # global n_train/n_test forcing them to match
    assert cfg.data.train.positive.params["n_samples"] == cfg.data.train.negative.params["n_samples"]


def test_model_and_adapter_overrides_apply_while_rest_of_config_is_inherited():
    # This is the mechanism orthex.cli's --set flag uses in place of the
    # old configs/models/*.yaml override files: configs/ now holds a
    # single default.yaml, and per-run fields (including model.id and
    # model.architecture_adapter -- there are no dedicated named flags for
    # these) are overridden from the command line via --set.
    raw = load_config_dict("configs/default.yaml")
    overridden = apply_dotted_overrides(
        raw, {"model.id": "Qwen/Qwen2.5-3B-Instruct", "model.architecture_adapter": "qwen2"}
    )
    cfg = build_config(overridden)
    assert cfg.model.id == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.model.architecture_adapter == "qwen2"
    # everything else inherited from default.yaml, unchanged
    assert cfg.data.train.positive.params["n_samples"] == 256
    assert cfg.selection.eval_top_n == 20


def test_apply_dotted_overrides_sets_nested_path_and_parses_yaml_scalars():
    raw = load_config_dict("configs/default.yaml")
    overridden = apply_dotted_overrides(
        raw,
        {
            "model.id": "foo/bar",
            "data.train.positive.params.n_samples": "10",
            "export.push_to_hub": "true",
            "export.repo_id": "some-org/some-model",
        },
    )
    cfg = build_config(overridden)
    assert cfg.model.id == "foo/bar"
    assert cfg.data.train.positive.params["n_samples"] == 10
    assert cfg.export.push_to_hub is True


def test_missing_test_block_raises():
    raw = load_config_dict("configs/default.yaml")
    del raw["data"]["test"]
    try:
        build_config(raw)
        raise AssertionError("expected a KeyError for missing data.test")
    except KeyError:
        pass


def test_push_to_hub_without_repo_id_raises():
    raw = load_config_dict("configs/default.yaml")
    raw["export"]["push_to_hub"] = True
    try:
        build_config(raw)
        raise AssertionError("expected a ValueError for push_to_hub without repo_id")
    except ValueError as e:
        assert "repo_id" in str(e)


def test_push_to_hub_with_repo_id_and_push_kwargs_succeeds():
    raw = load_config_dict("configs/default.yaml")
    raw["export"]["push_to_hub"] = True
    raw["export"]["repo_id"] = "some-org/some-model"
    raw["export"]["push_kwargs"] = {"private": True, "commit_message": "abliterated"}
    cfg = build_config(raw)
    assert cfg.export.repo_id == "some-org/some-model"
    assert cfg.export.push_kwargs == {"private": True, "commit_message": "abliterated"}


def test_push_to_hub_false_does_not_require_repo_id():
    raw = load_config_dict("configs/default.yaml")
    cfg = build_config(raw)  # default.yaml has push_to_hub: false, no repo_id
    assert cfg.export.push_to_hub is False
    assert cfg.export.repo_id is None
