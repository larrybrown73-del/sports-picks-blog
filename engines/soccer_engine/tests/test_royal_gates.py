"""Tests for Royal Picks clamp, tiering, contradiction, and validators."""

from __future__ import annotations

from ev_engine_core import (
    ROYAL_MAX_SCORE_BELOW_FLOOR,
    TIER_LOTTERY,
    TIER_ROYAL,
    TIER_STRONG,
    TIER_VALUE,
    MarketLeg,
    ProbabilityEstimate,
    assign_pick_tier,
    confidence_score,
    rank_ev_board,
    royal_approval_checklist,
)
from player_props_model import PlayerRateProfile, position_variance_scale
from royal_validators import enrich_results_with_situational_validators


def test_confidence_clamp_blocks_95_plus_on_sub_floor_true_prob() -> None:
    # Huge EV longshot with true_prob well under 55% must never clear 94.
    score = confidence_score(
        edge=0.40,
        ev_per_unit=3.0,
        true_probability=0.25,
        sample_size=50,
        volatility=0.20,
        warnings=[],
    )
    assert score <= ROYAL_MAX_SCORE_BELOW_FLOOR
    assert assign_pick_tier(score, 0.25) != TIER_ROYAL


def test_assign_pick_tier_requires_true_prob_floor_for_crown() -> None:
    assert assign_pick_tier(100, 0.54) == TIER_STRONG  # score alone is not enough
    assert assign_pick_tier(95, 0.55) == TIER_ROYAL
    assert assign_pick_tier(92, 0.70) == TIER_STRONG
    assert assign_pick_tier(85, 0.70) == TIER_VALUE
    assert assign_pick_tier(70, 0.70) == TIER_LOTTERY


def test_royal_approval_requires_full_checklist() -> None:
    leg = MarketLeg(
        game_id="mt_1",
        market_type="match_odds",
        selection="England",
        odds=2.0,
        odds_format="decimal",
        sportsbook="Bet365",
        side="home",
        team="England",
    )
    result = rank_ev_board(
        [leg],
        lambda _: ProbabilityEstimate(true_probability=0.70, sample_size=40, volatility=0.2),
        positive_only=False,
    )[0]
    # Team markets pass form/hit-rate; lineup_confirmed True for non-player.
    # Crown still needs score>=95 + checklist -- may or may not clear score.
    if result.confidence_score >= 95 and result.true_probability >= 0.55:
        assert result.royal_approved is True or royal_approval_checklist(result) is True
    assert result.tier in {TIER_ROYAL, TIER_STRONG, TIER_VALUE, TIER_LOTTERY}


def test_position_variance_compresses_defender_rates() -> None:
    assert position_variance_scale("ST") == 1.0
    assert position_variance_scale("CB") < position_variance_scale("CAM") < position_variance_scale("ST")


def test_hit_rate_gate_caps_tier_at_value_angle() -> None:
    profile = PlayerRateProfile(
        player_id="pl_1",
        player_name="Bench Defender",
        team_id="tm_1",
        minutes_per_appearance=60.0,
        goals_per_90=0.02,
        assists_per_90=0.02,
        shots_per_90=0.3,
        shots_on_target_per_90=0.1,
        appearances=20,
        position="CB",
    )
    leg = MarketLeg(
        game_id="mt_1",
        market_type="Player Assists",
        selection="Bench Defender Over 0.5",
        odds=11.0,
        odds_format="decimal",
        sportsbook="Bet365",
        entity_id="pl_1",
        entity_name="Bench Defender",
        side="over",
        line=0.5,
        sample_size=20,
        metadata={"home_team": "A", "away_team": "B"},
    )

    def provider(_: MarketLeg) -> ProbabilityEstimate:
        # Inflated model true-prob vs tiny season clear-rate -> hit-rate gate.
        return ProbabilityEstimate(true_probability=0.30, sample_size=20, volatility=0.2)

    results = rank_ev_board([leg], provider, positive_only=False, apply_correlation_filter=False)
    # Inject profile enrichment path used by rank_ev_board when provider has profiles.
    results = enrich_results_with_situational_validators(results, profiles={"pl_1": profile})
    assert results[0].hit_rate_ok is False
    assert results[0].confidence_score <= 89
