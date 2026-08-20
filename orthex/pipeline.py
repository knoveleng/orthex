from __future__ import annotations

import json
from pathlib import Path

# Side-effect imports: registers every provider/backend/strategy/adapter.
from orthex.ablation import weight_orthogonalizer  # noqa: F401
from orthex.ablation.registry import build_ablation_strategy
from orthex.architectures import gemma3_adapter, llama_adapter, qwen3_5_adapter  # noqa: F401
from orthex.architectures.registry import build_architecture_adapter
from orthex.backends import native_hf_backend  # noqa: F401
from orthex.backends.registry import build_backend
from orthex.capture.activation_cache import capture_activations
from orthex.capture.types import LayerSite
from orthex.config import OrthexConfig
from orthex.data.providers import hf_dataset, local_json, local_jsonl  # noqa: F401
from orthex.data.registry import build_provider
from orthex.directions import mean_difference  # noqa: F401
from orthex.directions.mean_difference import raw_diff_magnitude
from orthex.directions.registry import build_direction_strategy
from orthex.evaluation import benchmark_runner
from orthex.model_card import render as render_model_card
from orthex.selection import candidate_scorer
from orthex.selection.scorers import refusal_phrase_heuristic  # noqa: F401
from orthex.selection.scorers.registry import build_scorer


def _resolve_layer_range(num_layers: int, layer_range: list[int]) -> set[int]:
    start, end = layer_range
    if end < 0:
        end = num_layers + end + 1  # doc's [1, -1] means "layer 1 through the last layer inclusive"
    return set(range(start, end))


def run(config: OrthexConfig, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backend = build_backend(config.model.backend, config.model.id, config.model.dtype)
    adapter = build_architecture_adapter(config.model.architecture_adapter)
    layers = adapter.decoder_layers(backend.model)

    pos_train = build_provider(config.data.train.positive.provider, config.data.train.positive.params).load()
    neg_train = build_provider(config.data.train.negative.provider, config.data.train.negative.params).load()
    # Test data is positive-only: refusal_rate and perplexity are both
    # measured on the same held-out behavior-eliciting prompts, pre vs.
    # post ablation -- there's no use for a separate negative test set.
    pos_test = build_provider(config.data.test.provider, config.data.test.params).load()

    pos_acts = capture_activations(
        backend.model, backend.tokenizer, layers, pos_train, config.capture.sites, config.capture.position
    )
    neg_acts = capture_activations(
        backend.model, backend.tokenizer, layers, neg_train, config.capture.sites, config.capture.position
    )

    valid_layers = _resolve_layer_range(len(layers), config.directions.layer_range)
    pos_acts = {
        site: t for site, t in pos_acts.items() if site.layer in valid_layers and site.site in config.selection.sites
    }
    neg_acts = {
        site: t for site, t in neg_acts.items() if site.layer in valid_layers and site.site in config.selection.sites
    }

    direction_strategy = build_direction_strategy(config.directions.strategy)
    candidates = direction_strategy.compute(pos_acts, neg_acts)
    magnitudes = raw_diff_magnitude(pos_acts, neg_acts)

    scorer = build_scorer(config.selection.strategy, config.selection.blacklist)

    if config.selection.force_layer is not None:
        forced_site = LayerSite(config.selection.force_layer, config.selection.force_site or "resid_pre")
        if forced_site not in candidates:
            available = sorted(candidates, key=lambda s: (s.layer, s.site))
            raise ValueError(f"Forced candidate {forced_site} not among computed candidates: {available}")
        best_site, best_direction = forced_site, candidates[forced_site]
        selection_report = {
            "forced": True,
            "selected": {"layer": best_site.layer, "site": best_site.site, "refusal_rate": None},
        }
    else:
        best_site, best_direction, selection_report = candidate_scorer.select_best(
            backend,
            layers,
            candidates,
            magnitudes,
            pos_test,
            scorer,
            config.selection.eval_top_n,
            config.selection.max_new_tokens,
            config.selection.generation_batch_size,
        )

    pre_report = benchmark_runner.evaluate(backend.model, backend.tokenizer, pos_test, scorer, config.evaluation)

    ablation_strategy = build_ablation_strategy(config.ablation.strategy)
    ablation_strategy.apply(backend.model, adapter, best_direction, config.ablation.targets)

    post_report = benchmark_runner.evaluate(backend.model, backend.tokenizer, pos_test, scorer, config.evaluation)

    backend.model.save_pretrained(output_dir)
    backend.tokenizer.save_pretrained(output_dir)

    def _delta(key: str):
        if key in pre_report and key in post_report:
            return post_report[key] - pre_report[key]
        return None

    result = {
        "model_id": config.model.id,
        "architecture_adapter": config.model.architecture_adapter,
        "selected_candidate": selection_report["selected"],
        "selection_report": selection_report,
        "evaluation": {
            "refusal_rate": {
                "pre": pre_report.get("refusal_rate"),
                "post": post_report.get("refusal_rate"),
                "delta": _delta("refusal_rate"),
            },
            "perplexity": {
                "pre": pre_report.get("perplexity"),
                "post": post_report.get("perplexity"),
                "delta": _delta("perplexity"),
            },
        },
        # One {prompt, response, is_refusal} record per held-out positive
        # test prompt, pre- and post-ablation -- for spot-checking what
        # actually changed, not just the aggregate refusal_rate.
        "refusal_samples": {
            "pre": pre_report.get("refusal_samples"),
            "post": post_report.get("refusal_samples"),
        },
    }

    with open(output_dir / "evaluation_report.json", "w") as fh:
        json.dump(result, fh, indent=2)

    (output_dir / "README.md").write_text(render_model_card(result))

    if config.export.push_to_hub:
        backend.model.push_to_hub(config.export.repo_id, **config.export.push_kwargs)
        backend.tokenizer.push_to_hub(config.export.repo_id, **config.export.push_kwargs)

    return result
