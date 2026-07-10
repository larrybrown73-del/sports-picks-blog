"""Tests for starter baseline injection, secondary caps, and pitching veto."""

from __future__ import annotations

from unittest.mock import patch

from config import (
    LEAGUE_AVG_ERA,
    LEAGUE_AVG_RUNS,
    PITCHING_MISMATCH_OPP_ERA_MAX,
    PITCHING_MISMATCH_OUR_ERA_MIN,
    SECONDARY_MODIFIER_MAX_PCT,
    SP_BASELINE_RF_WEIGHT,
)
from pitcher_matchup import PitcherSeasonProfile
from starter_baseline import (
    apply_starter_baseline_injection,
    clamp_secondary_run_adjustments,
    expected_runs_vs_starter,
    pitching_mismatch_veto,
)


def _profile(name: str, era: float, whip: float) -> PitcherSeasonProfile:
    return PitcherSeasonProfile(
        pitcher_id=1,
        pitcher_name=name,
        season_era=era,
        season_whip=whip,
        season_babip=0.300,
        avg_fastball_velo=93.0,
        ground_ball_pct=42.0,
    )


def test_expected_runs_vs_elite_starter_is_low() -> None:
    runs = expected_runs_vs_starter(2.80, 0.95)
    assert runs < LEAGUE_AVG_RUNS


def test_expected_runs_vs_bad_starter_is_high() -> None:
    runs = expected_runs_vs_starter(5.80, 1.45)
    assert runs > LEAGUE_AVG_RUNS


@patch("starter_baseline.fetch_pitcher_season_profile")
@patch("starter_baseline.get_starting_pitcher_info")
def test_starter_baseline_heavily_weights_sp_metrics(mock_info, mock_fetch) -> None:
    mock_info.return_value = {
        "home_pitcher_id": 10,
        "away_pitcher_id": 20,
        "home_pitcher_name": "Elite Ace",
        "away_pitcher_name": "Bad Arm",
    }
    mock_fetch.side_effect = [
        _profile("Bad Arm", 5.50, 1.40),
        _profile("Elite Ace", 2.90, 0.98),
    ]

    result = apply_starter_baseline_injection(6.20, 5.90, game_id=1, season=2026)
    sp_weight = 1.0 - SP_BASELINE_RF_WEIGHT
    expected_home = SP_BASELINE_RF_WEIGHT * 6.20 + sp_weight * expected_runs_vs_starter(5.50, 1.40)
    expected_away = SP_BASELINE_RF_WEIGHT * 5.90 + sp_weight * expected_runs_vs_starter(2.90, 0.98)

    assert result.home_runs == expected_home
    assert result.away_runs == expected_away
    assert result.home_runs != 6.20
    assert result.away_runs != 5.90


def test_secondary_modifier_cap_limits_boost() -> None:
    home, away, tags = clamp_secondary_run_adjustments(
        5.00,
        4.00,
        5.60,
        4.00,
        cap=SECONDARY_MODIFIER_MAX_PCT,
    )
    assert home == 5.40
    assert away == 4.00
    assert any("secondary_cap:home" in tag for tag in tags)


def test_secondary_modifier_cap_limits_penalty() -> None:
    home, away, tags = clamp_secondary_run_adjustments(
        5.00,
        4.00,
        4.60,
        3.20,
        cap=SECONDARY_MODIFIER_MAX_PCT,
    )
    assert home == 4.60
    assert away == 3.68
    assert any("secondary_cap:away" in tag for tag in tags)


def test_pitching_mismatch_veto_triggers() -> None:
    assert pitching_mismatch_veto(our_sp_era=5.40, opponent_sp_era=3.80)
    assert not pitching_mismatch_veto(our_sp_era=4.80, opponent_sp_era=3.80)
    assert not pitching_mismatch_veto(our_sp_era=5.40, opponent_sp_era=4.10)


def test_pitching_mismatch_threshold_constants() -> None:
    assert PITCHING_MISMATCH_OUR_ERA_MIN == 5.00
    assert PITCHING_MISMATCH_OPP_ERA_MAX == 4.00
    assert pitching_mismatch_veto(
        our_sp_era=PITCHING_MISMATCH_OUR_ERA_MIN + 0.01,
        opponent_sp_era=PITCHING_MISMATCH_OPP_ERA_MAX - 0.01,
    )
