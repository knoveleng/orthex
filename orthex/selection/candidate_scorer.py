from __future__ import annotations

import torch

from orthex.capture.types import LayerSite
from orthex.generation import generate_completions
from orthex.selection.activation_projection import project_direction


def rank_candidates(
    backend,
    layers,
    candidates: dict[LayerSite, torch.Tensor],
    magnitudes: dict[LayerSite, float],
    test_prompts: list[str],
    scorer,
    eval_top_n: int,
    max_new_tokens: int,
    generation_batch_size: int,
) -> list[dict]:
    """Pre-filters the full candidate universe to the `eval_top_n` sites
    with the largest pre-normalization diff-of-means magnitude (a cheap,
    forward-pass-free proxy for how much a site separates positive/negative
    prompts -- a deliberate, documented interpretation of the doc's
    unspecified pre-filter mechanism), then scores each surviving candidate
    by generating short completions under a temporary activation-projection
    hook and measuring refusal rate. Returns all scored candidates sorted
    by refusal rate ascending (lowest = most refusal removed)."""
    ranked_sites = sorted(candidates.keys(), key=lambda s: magnitudes.get(s, 0.0), reverse=True)[:eval_top_n]

    results = []
    for site in ranked_sites:
        direction = candidates[site]
        layer = layers[site.layer]
        with project_direction(layer, site.site, direction):
            responses = generate_completions(
                backend.model, backend.tokenizer, test_prompts, max_new_tokens, generation_batch_size
            )
        results.append({"site": site, "refusal_rate": scorer.score(responses)})

    results.sort(key=lambda r: r["refusal_rate"])
    return results


def select_best(
    backend,
    layers,
    candidates: dict[LayerSite, torch.Tensor],
    magnitudes: dict[LayerSite, float],
    test_prompts: list[str],
    scorer,
    eval_top_n: int,
    max_new_tokens: int,
    generation_batch_size: int,
) -> tuple[LayerSite, torch.Tensor, dict]:
    """Non-interactive auto-pick for the live/automated run: among all
    ranked candidates tied at the lowest refusal rate, prefers the one
    closest to the model's middle layer (ablation literature generally
    finds mid-layer directions generalize better than very-early/late
    ones). `rank_candidates`' full breakdown is still returned in the
    report for a human to review after the fact."""
    ranked = rank_candidates(
        backend, layers, candidates, magnitudes, test_prompts, scorer, eval_top_n, max_new_tokens, generation_batch_size
    )
    best_rate = ranked[0]["refusal_rate"]
    tied = [r for r in ranked if r["refusal_rate"] == best_rate]
    mid_layer = len(layers) // 2
    best = min(tied, key=lambda r: abs(r["site"].layer - mid_layer))

    report = {
        "ranked": [
            {"layer": r["site"].layer, "site": r["site"].site, "refusal_rate": r["refusal_rate"]} for r in ranked
        ],
        "selected": {"layer": best["site"].layer, "site": best["site"].site, "refusal_rate": best["refusal_rate"]},
    }
    return best["site"], candidates[best["site"]], report
