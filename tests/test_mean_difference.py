import torch

from orthex.capture.types import LayerSite
from orthex.directions.mean_difference import MeanDifferenceStrategy, raw_diff_magnitude


def test_mean_difference_matches_analytic_diff_of_means():
    site = LayerSite(1, "resid_pre")
    pos = torch.tensor([[1.0, 0.0], [3.0, 0.0]])  # mean [2, 0]
    neg = torch.tensor([[0.0, 1.0], [0.0, 3.0]])  # mean [0, 2]
    strategy = MeanDifferenceStrategy()
    result = strategy.compute({site: pos}, {site: neg})

    expected_raw = torch.tensor([2.0, -2.0])
    expected = expected_raw / expected_raw.norm()
    assert torch.allclose(result[site], expected, atol=1e-6)
    assert torch.isclose(result[site].norm(), torch.tensor(1.0), atol=1e-6)


def test_mean_difference_skips_site_missing_from_one_side():
    only_positive = LayerSite(0, "resid_pre")
    both_sides = LayerSite(1, "resid_pre")
    pos = {only_positive: torch.randn(4, 3), both_sides: torch.randn(4, 3)}
    neg = {both_sides: torch.randn(4, 3)}
    result = MeanDifferenceStrategy().compute(pos, neg)
    assert set(result) == {both_sides}


def test_raw_diff_magnitude_is_unnormalized():
    site = LayerSite(1, "resid_pre")
    pos = torch.tensor([[4.0, 0.0]])
    neg = torch.tensor([[0.0, 0.0]])
    magnitudes = raw_diff_magnitude({site: pos}, {site: neg})
    assert magnitudes[site] == 4.0
