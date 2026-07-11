"""Tests for prop probability variance fix and absolute ceiling/floor."""

from __future__ import annotations

from baseball_props.analysis.edge_sheets import (
    clamp_prop_probability,
    prob_over_continuous,
    scaled_prop_sigma,
)
from baseball_props.config import EDGE_HITS_SIGMA, MAX_PROP_PROB, MIN_PROP_PROB


def test_scaled_sigma_moves_with_mean() -> None:
    boosted = scaled_prop_sigma(EDGE_HITS_SIGMA, 1.5, 1.8)
    cut = scaled_prop_sigma(EDGE_HITS_SIGMA, 1.5, 1.2)
    assert boosted == EDGE_HITS_SIGMA * (1.8 / 1.5)
    assert cut == EDGE_HITS_SIGMA * (1.2 / 1.5)


def test_clamp_prop_probability_ceiling_and_floor() -> None:
    assert clamp_prop_probability(0.999) == MAX_PROP_PROB
    assert clamp_prop_probability(0.01) == MIN_PROP_PROB
    assert clamp_prop_probability(0.62) == 0.62


def test_prob_over_continuous_never_exceeds_ceiling() -> None:
    # Extreme mean vs low line with tiny sigma would otherwise approach 1.0
    prob = prob_over_continuous(3.5, 0.05, 0.5)
    assert prob is not None
    assert prob <= MAX_PROP_PROB
    assert prob >= MIN_PROP_PROB


def test_prob_over_continuous_never_below_floor() -> None:
    prob = prob_over_continuous(0.1, 0.05, 1.5)
    assert prob is not None
    assert prob >= MIN_PROP_PROB
    assert prob <= MAX_PROP_PROB
