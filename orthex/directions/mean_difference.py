from __future__ import annotations

import torch

from orthex.capture.types import LayerSite
from orthex.directions.registry import register


@register("mean_difference")
class MeanDifferenceStrategy:
    """Arditi et al. baseline: positive_mean - negative_mean, unit-normalized.
    Skips any site present on only one side (mirrors the reference
    implementation this is ported from)."""

    def compute(
        self,
        positive_acts: dict[LayerSite, torch.Tensor],
        negative_acts: dict[LayerSite, torch.Tensor],
    ) -> dict[LayerSite, torch.Tensor]:
        candidates: dict[LayerSite, torch.Tensor] = {}
        for site, pos in positive_acts.items():
            if site not in negative_acts:
                continue
            neg = negative_acts[site]
            direction = pos.mean(dim=0) - neg.mean(dim=0)
            direction = direction / (direction.norm() + 1e-8)
            candidates[site] = direction
        return candidates


def raw_diff_magnitude(
    positive_acts: dict[LayerSite, torch.Tensor],
    negative_acts: dict[LayerSite, torch.Tensor],
) -> dict[LayerSite, float]:
    """Pre-normalization diff-of-means magnitude per site -- a cheap,
    forward-pass-free proxy for "how much this site separates positive vs
    negative prompts," used by selection/candidate_scorer.py to pre-filter
    the candidate universe down to `eval_top_n` before the expensive
    generate-and-score step."""
    magnitudes: dict[LayerSite, float] = {}
    for site, pos in positive_acts.items():
        if site not in negative_acts:
            continue
        diff = pos.mean(dim=0) - negative_acts[site].mean(dim=0)
        magnitudes[site] = diff.norm().item()
    return magnitudes
