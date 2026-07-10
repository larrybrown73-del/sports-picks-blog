"""Tests for team defense tiers and contact-starter cross-reference."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from config import GOLD_GLOVE_BOOST, POOR_DEFENSE_PENALTY
from starter_rest_hierarchy import AceDominanceProfile, StarterRestAndHierarchyTracker
from team_defense import contact_defense_scalar, team_defense_tier


def _dominance(
    *,
    k_bb: float | None = 11.0,
    whip: float | None = 1.08,
    true_ace: bool = False,
    innings_eater: bool = True,
) -> AceDominanceProfile:
    return AceDominanceProfile(
        pitcher_id=1,
        k_bb_pct=k_bb,
        whip=whip,
        is_true_ace=true_ace,
        is_innings_eater=innings_eater,
    )


@patch("team_defense._defense_rank_sets", return_value=(frozenset({112}), frozenset({136})))
def test_team_defense_tier_elite_and_poor(_ranks) -> None:
    assert team_defense_tier(112, season=2026) == "elite"
    assert team_defense_tier(136, season=2026) == "poor"
    assert team_defense_tier(147, season=2026) == "neutral"


@patch("team_defense.team_defense_tier", return_value="elite")
def test_contact_defense_scalar_gold_glove(_tier) -> None:
    scalar, tag = contact_defense_scalar(112, season=2026)
    assert scalar == GOLD_GLOVE_BOOST
    assert tag == "gold_glove_boost"


@patch("team_defense.team_defense_tier", return_value="poor")
def test_contact_defense_scalar_brick_hands(_tier) -> None:
    scalar, tag = contact_defense_scalar(136, season=2026)
    assert scalar == POOR_DEFENSE_PENALTY
    assert tag == "poor_defense_penalty"


@patch("starter_rest_hierarchy._last_start_date", return_value=date(2026, 7, 1))
@patch("starter_rest_hierarchy._rotation_slot", return_value=1)
@patch("starter_rest_hierarchy._is_il_return", return_value=False)
@patch(
    "starter_rest_hierarchy._fetch_ace_dominance_profile",
    return_value=_dominance(k_bb=11.0, whip=1.08, innings_eater=True),
)
@patch("starter_rest_hierarchy.contact_defense_scalar", return_value=(GOLD_GLOVE_BOOST, "gold_glove_boost"))
def test_innings_eater_gets_gold_glove_boost(mock_defense, *_mocks) -> None:
    tracker = StarterRestAndHierarchyTracker()
    evaluation = tracker._evaluate(88, 112, game_date=date(2026, 7, 9), season=2026)
    assert evaluation.defense_scalar == GOLD_GLOVE_BOOST
    assert "gold_glove_boost" in evaluation.tags


@patch("starter_rest_hierarchy._last_start_date", return_value=date(2026, 7, 1))
@patch("starter_rest_hierarchy._rotation_slot", return_value=1)
@patch("starter_rest_hierarchy._is_il_return", return_value=False)
@patch(
    "starter_rest_hierarchy._fetch_ace_dominance_profile",
    return_value=_dominance(k_bb=24.0, whip=0.98, true_ace=True, innings_eater=False),
)
@patch("starter_rest_hierarchy.contact_defense_scalar", return_value=(POOR_DEFENSE_PENALTY, "poor_defense_penalty"))
def test_true_ace_bypasses_defense_logic(mock_defense, *_mocks) -> None:
    tracker = StarterRestAndHierarchyTracker()
    evaluation = tracker._evaluate(1, 136, game_date=date(2026, 7, 9), season=2026)
    assert evaluation.defense_scalar == 1.0
    assert "poor_defense_penalty" not in evaluation.tags
    mock_defense.assert_not_called()
