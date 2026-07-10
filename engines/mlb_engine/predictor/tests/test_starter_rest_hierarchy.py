"""Tests for starter rest and ace dominance hierarchy guardrails."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from config import (
    ACE_RUN_SUPPRESSION_FACTOR,
    BACK_END_STARTER_PENALTY,
    CONTACT_STARTER_MAX_BONUS,
    ELITE_ACE_SCALAR,
    OPTIMAL_REST_BONUS,
    RUST_PENALTY,
    SHORT_REST_PENALTY,
    TOP_OF_ROTATION_SCALAR,
)
from starter_rest_hierarchy import (
    AceDominanceProfile,
    StarterRestAndHierarchyTracker,
    _fetch_ace_dominance_profile,
    _hierarchy_scalar,
    _rest_scalar,
    _tier_from_slot,
    apply_starter_context_to_runs,
)


def _dominance(
    *,
    k_bb: float | None = 16.0,
    whip: float | None = 1.15,
    true_ace: bool = False,
    innings_eater: bool = False,
) -> AceDominanceProfile:
    return AceDominanceProfile(
        pitcher_id=1,
        k_bb_pct=k_bb,
        whip=whip,
        is_true_ace=true_ace,
        is_innings_eater=innings_eater,
    )


def test_short_rest_penalty() -> None:
    scalar, tag = _rest_scalar(3, il_return=False)
    assert tag == "short_rest"
    assert scalar == SHORT_REST_PENALTY


def test_true_ace_gets_elite_scalar() -> None:
    scalar, tag = _hierarchy_scalar(1, _dominance(k_bb=22.0, whip=0.98, true_ace=True))
    assert tag == "true_ace"
    assert scalar == ELITE_ACE_SCALAR


def test_depth_chart_starter_without_dominance() -> None:
    scalar, tag = _hierarchy_scalar(
        1,
        _dominance(k_bb=16.0, whip=1.12, true_ace=False, innings_eater=False),
    )
    assert tag == "depth_chart_starter"
    assert scalar == TOP_OF_ROTATION_SCALAR


def test_innings_eater_caps_depth_chart_suppression() -> None:
    scalar, tag = _hierarchy_scalar(
        1,
        _dominance(k_bb=11.0, whip=1.08, true_ace=False, innings_eater=True),
    )
    assert tag == "innings_eater_cap"
    assert scalar == CONTACT_STARTER_MAX_BONUS


def test_innings_eater_does_not_cap_back_end_penalty() -> None:
    scalar, tag = _hierarchy_scalar(
        3,
        _dominance(k_bb=10.0, whip=1.30, true_ace=False, innings_eater=True),
    )
    assert tag == "tier3_back_end"
    assert scalar == BACK_END_STARTER_PENALTY


@patch("starter_rest_hierarchy._fetch_pitcher_season_stat")
def test_fetch_ace_dominance_profile_true_ace(mock_stat) -> None:
    mock_stat.return_value = {
        "battersFaced": 250,
        "strikeOuts": 70,
        "baseOnBalls": 12,
        "whip": 0.98,
    }
    profile = _fetch_ace_dominance_profile(1, season=2026)
    assert profile.k_bb_pct == 23.2
    assert profile.is_true_ace
    assert not profile.is_innings_eater


@patch("starter_rest_hierarchy._fetch_pitcher_season_stat")
def test_fetch_ace_dominance_profile_innings_eater(mock_stat) -> None:
    mock_stat.return_value = {
        "battersFaced": 220,
        "strikeOuts": 30,
        "baseOnBalls": 18,
        "whip": 1.18,
    }
    profile = _fetch_ace_dominance_profile(2, season=2026)
    assert profile.k_bb_pct == pytest.approx(5.45, abs=0.01)
    assert profile.is_innings_eater
    assert not profile.is_true_ace


@patch("starter_rest_hierarchy.StarterRestAndHierarchyTracker.evaluate")
def test_true_ace_on_optimal_rest_gets_max_suppression(mock_evaluate) -> None:
    from starter_rest_hierarchy import StarterEvaluation

    combined = ELITE_ACE_SCALAR * OPTIMAL_REST_BONUS * ACE_RUN_SUPPRESSION_FACTOR
    mock_evaluate.return_value = StarterEvaluation(
        pitcher_id=1,
        team_id=147,
        days_rest=6,
        rotation_slot=1,
        tier=1,
        k_bb_pct=24.0,
        rest_scalar=OPTIMAL_REST_BONUS,
        hierarchy_scalar=ELITE_ACE_SCALAR,
        defense_scalar=1.0,
        combined_scalar=combined,
        tags=("optimal_rest", "true_ace", f"ace_synergy:{ACE_RUN_SUPPRESSION_FACTOR:.2f}"),
    )
    runs, tags = apply_starter_context_to_runs(
        5.0,
        pitcher_id=1,
        defending_team_id=147,
        game_date=date(2026, 7, 9),
        season=2026,
        label="Schlittler",
    )
    assert runs == 5.0 * combined
    assert any("ace_synergy" in tag for tag in tags)


@patch("starter_rest_hierarchy._last_start_date", return_value=date(2026, 7, 6))
@patch("starter_rest_hierarchy._rotation_slot", return_value=1)
@patch("starter_rest_hierarchy._is_il_return", return_value=False)
@patch("starter_rest_hierarchy.contact_defense_scalar", return_value=(1.0, None))
@patch(
    "starter_rest_hierarchy._fetch_ace_dominance_profile",
    return_value=_dominance(k_bb=11.0, whip=1.08, innings_eater=True),
)
def test_contact_sp1_capped_not_true_ace(mock_dom, mock_defense, mock_il, mock_slot, mock_last_start) -> None:
    tracker = StarterRestAndHierarchyTracker()
    evaluation = tracker._evaluate(88, 138, game_date=date(2026, 7, 9), season=2026)
    expected = SHORT_REST_PENALTY * CONTACT_STARTER_MAX_BONUS
    assert evaluation.combined_scalar == expected
    assert evaluation.tier == 1
    assert "innings_eater_cap" in evaluation.tags


def test_rotation_tier_mapping() -> None:
    assert _tier_from_slot(1) == 1
    assert _tier_from_slot(5) == 3


def test_rust_penalty_excludes_il_return() -> None:
    scalar, tag = _rest_scalar(12, il_return=True)
    assert tag == "neutral_rest"
    assert scalar == 1.0

    rusty, rusty_tag = _rest_scalar(12, il_return=False)
    assert rusty_tag == "rust_penalty"
    assert rusty == RUST_PENALTY
