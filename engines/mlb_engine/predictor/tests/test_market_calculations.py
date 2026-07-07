import math

from market.calculations import (
    american_to_decimal,
    compute_wager_metrics,
    confidence_from_edge_and_prob,
    confidence_label_from_score,
    confidence_tier_from_edge,
    full_kelly_fraction,
)


def test_american_to_decimal_minus_110() -> None:
    assert math.isclose(american_to_decimal(-110), 1.909090909, rel_tol=1e-4)


def test_compute_wager_metrics_positive_ev() -> None:
    metrics = compute_wager_metrics(0.55, -110)
    assert metrics.ev_per_unit is not None
    assert metrics.ev_per_unit > 0
    assert metrics.edge_pct is not None
    assert metrics.fractional_kelly_pct is not None


def test_confidence_tier_boundaries() -> None:
    assert confidence_tier_from_edge(5.0) == "Tier-1 High Conviction"
    assert confidence_tier_from_edge(3.0) == "Tier-2 Standard"
    assert confidence_tier_from_edge(1.0) == "Tier-3 Marginal"
    assert confidence_tier_from_edge(0.0) == "No Bet"


def test_full_kelly_non_negative() -> None:
    assert full_kelly_fraction(0.55, -110) >= 0.0


def test_fractional_kelly_capped_at_max_bet_pct() -> None:
    metrics = compute_wager_metrics(0.75, +300)
    assert metrics.fractional_kelly_pct is not None
    assert metrics.fractional_kelly_pct <= 5.0
    manual_uncapped = 0.25 * full_kelly_fraction(0.75, +300) * 100.0
    assert manual_uncapped > 5.0


def test_confidence_label_boundaries() -> None:
    assert confidence_label_from_score(85) == "Elite"
    assert confidence_label_from_score(84) == "High"
    assert confidence_label_from_score(70) == "High"
    assert confidence_label_from_score(69) == "Medium"
    assert confidence_label_from_score(50) == "Medium"
    assert confidence_label_from_score(49) == "Low"


def test_confidence_from_edge_and_prob() -> None:
    score, label = confidence_from_edge_and_prob(12.0, 0.58)
    assert label == confidence_label_from_score(score)
    assert label in {"Elite", "High", "Medium", "Low"}
