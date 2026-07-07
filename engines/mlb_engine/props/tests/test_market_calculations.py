import math

from baseball_props.market.calculations import (
    american_to_decimal,
    compute_wager_metrics,
    confidence_tier_from_edge,
)


def test_props_market_calculations_ev() -> None:
    metrics = compute_wager_metrics(0.55, -110)
    assert metrics.ev_per_unit is not None
    assert metrics.ev_per_unit > 0
    assert math.isclose(american_to_decimal(-110), 1.909090909, rel_tol=1e-4)


def test_props_confidence_tiers() -> None:
    assert confidence_tier_from_edge(6.0) == "Tier-1 High Conviction"
    assert confidence_tier_from_edge(4.0) == "Tier-2 Moderate"
    assert confidence_tier_from_edge(2.0) == "Tier-3 Speculative"
    assert confidence_tier_from_edge(0.5) == "Below Threshold"


def test_tier_scaled_kelly() -> None:
    from baseball_props.config import KELLY_MAX_STAKE_PCT, TIER_KELLY_MULTIPLIERS
    from baseball_props.market.calculations import tier_kelly_fraction

    metrics_high = compute_wager_metrics(0.60, -110, edge_pct=6.0)
    assert metrics_high.confidence_tier == "Tier-1 High Conviction"
    assert metrics_high.kelly_fraction is not None
    assert metrics_high.kelly_fraction <= KELLY_MAX_STAKE_PCT
    full = metrics_high.full_kelly
    assert full is not None
    expected = min(full * TIER_KELLY_MULTIPLIERS["Tier-1 High Conviction"], KELLY_MAX_STAKE_PCT)
    assert math.isclose(metrics_high.kelly_fraction, expected, rel_tol=1e-4)

    tier_frac = tier_kelly_fraction(full, "Tier-2 Moderate")
    assert tier_frac is not None
    assert tier_frac <= KELLY_MAX_STAKE_PCT
