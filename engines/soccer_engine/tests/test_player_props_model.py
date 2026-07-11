from __future__ import annotations

import math

from historical_data import PlayerSeasonStats
from lineups import NOT_IN_SQUAD, STARTING, SUBSTITUTE
from player_props_model import (
    SUBSTITUTE_BASELINE_MINUTES,
    PlayerRateProfile,
    build_player_rate_profile,
    expected_minutes_factor,
    opponent_adjustment_factor,
    player_prop_probability,
)
from team_strength import TeamRatings


def test_build_player_rate_profile_computes_per_90_rates() -> None:
    stats = PlayerSeasonStats(
        player_id="pl_1",
        player_name="Test Player",
        team_id="tm_1",
        position="F",
        season_id="sn_1",
        minutes_played=1800,
        appearances=20,
        goals=9,
        assists=4,
        total_shots=45,
        shots_on_target=20,
    )
    profile = build_player_rate_profile(stats)
    assert profile is not None
    assert math.isclose(profile.goals_per_90, 9 * 90 / 1800)
    assert math.isclose(profile.assists_per_90, 4 * 90 / 1800)
    assert math.isclose(profile.shots_per_90, 45 * 90 / 1800)
    assert math.isclose(profile.shots_on_target_per_90, 20 * 90 / 1800)
    assert math.isclose(profile.minutes_per_appearance, 1800 / 20)


def test_build_player_rate_profile_drops_zero_minutes() -> None:
    stats = PlayerSeasonStats(
        player_id="pl_2",
        player_name="Unused Player",
        team_id="tm_1",
        position="F",
        season_id="sn_1",
        minutes_played=0,
        appearances=0,
        goals=0,
        assists=0,
        total_shots=0,
        shots_on_target=0,
    )
    assert build_player_rate_profile(stats) is None


def test_opponent_adjustment_factor_neutral_without_data() -> None:
    ratings = TeamRatings(
        attack={}, defense={}, home_advantage=0.2, rho=-0.05, league_avg_home_goals=1.4, league_avg_away_goals=1.1
    )
    assert opponent_adjustment_factor(ratings, "tm_unknown") == 1.0
    assert opponent_adjustment_factor(ratings, None) == 1.0


def test_opponent_adjustment_factor_scales_with_defense_strength() -> None:
    ratings = TeamRatings(
        attack={},
        defense={"weak_defense": 0.5, "strong_defense": -0.5, "avg": 0.0},
        home_advantage=0.2,
        rho=-0.05,
        league_avg_home_goals=1.4,
        league_avg_away_goals=1.1,
    )
    weak = opponent_adjustment_factor(ratings, "weak_defense")
    strong = opponent_adjustment_factor(ratings, "strong_defense")
    assert weak > 1.0 > strong


def test_expected_minutes_factor_caps_at_one_and_scales_down_for_rotation_players() -> None:
    full_time = PlayerRateProfile(
        player_id="pl_1",
        player_name="Starter",
        team_id="tm_1",
        minutes_per_appearance=95.0,
        goals_per_90=0.3,
        assists_per_90=0.2,
        shots_per_90=2.0,
        shots_on_target_per_90=1.0,
        appearances=20,
    )
    assert expected_minutes_factor(full_time) == 1.0

    rotation = PlayerRateProfile(
        player_id="pl_2",
        player_name="Rotation",
        team_id="tm_1",
        minutes_per_appearance=30.0,
        goals_per_90=0.3,
        assists_per_90=0.2,
        shots_per_90=2.0,
        shots_on_target_per_90=1.0,
        appearances=20,
    )
    assert math.isclose(expected_minutes_factor(rotation), 30.0 / 90.0)


def test_expected_minutes_factor_confirmed_not_in_squad_is_zero() -> None:
    starter_profile = PlayerRateProfile(
        player_id="pl_1",
        player_name="Starter",
        team_id="tm_1",
        minutes_per_appearance=90.0,
        goals_per_90=0.3,
        assists_per_90=0.2,
        shots_per_90=2.0,
        shots_on_target_per_90=1.0,
        appearances=20,
    )
    assert expected_minutes_factor(starter_profile, lineup_status=NOT_IN_SQUAD) == 0.0


def test_expected_minutes_factor_confirmed_substitute_uses_baseline_not_historical_average() -> None:
    high_minutes_profile = PlayerRateProfile(
        player_id="pl_1",
        player_name="Usually A Starter",
        team_id="tm_1",
        minutes_per_appearance=90.0,
        goals_per_90=0.3,
        assists_per_90=0.2,
        shots_per_90=2.0,
        shots_on_target_per_90=1.0,
        appearances=20,
    )
    assert math.isclose(
        expected_minutes_factor(high_minutes_profile, lineup_status=SUBSTITUTE),
        SUBSTITUTE_BASELINE_MINUTES / 90.0,
    )


def test_expected_minutes_factor_confirmed_starting_still_uses_historical_average() -> None:
    rotation_profile = PlayerRateProfile(
        player_id="pl_2",
        player_name="Rotation",
        team_id="tm_1",
        minutes_per_appearance=30.0,
        goals_per_90=0.3,
        assists_per_90=0.2,
        shots_per_90=2.0,
        shots_on_target_per_90=1.0,
        appearances=20,
    )
    # A starter who's normally subbed off at 30 minutes shouldn't suddenly
    # be modeled as playing a full 90 just because they're in the XI.
    assert math.isclose(expected_minutes_factor(rotation_profile, lineup_status=STARTING), 30.0 / 90.0)


def test_expected_minutes_factor_no_lineup_falls_back_to_historical_average() -> None:
    rotation_profile = PlayerRateProfile(
        player_id="pl_2",
        player_name="Rotation",
        team_id="tm_1",
        minutes_per_appearance=30.0,
        goals_per_90=0.3,
        assists_per_90=0.2,
        shots_per_90=2.0,
        shots_on_target_per_90=1.0,
        appearances=20,
    )
    assert math.isclose(expected_minutes_factor(rotation_profile, lineup_status=None), 30.0 / 90.0)


def test_player_prop_probability_confirmed_not_in_squad_is_zero() -> None:
    profile = PlayerRateProfile(
        player_id="pl_1",
        player_name="Forward",
        team_id="tm_1",
        minutes_per_appearance=90.0,
        goals_per_90=0.5,
        assists_per_90=0.2,
        shots_per_90=2.5,
        shots_on_target_per_90=1.2,
        appearances=20,
    )
    assert player_prop_probability(profile, "goals", None, lineup_status=NOT_IN_SQUAD) == 0.0


def test_player_prop_probability_no_line_matches_poisson_zero_complement() -> None:
    profile = PlayerRateProfile(
        player_id="pl_1",
        player_name="Forward",
        team_id="tm_1",
        minutes_per_appearance=90.0,
        goals_per_90=0.5,
        assists_per_90=0.2,
        shots_per_90=2.5,
        shots_on_target_per_90=1.2,
        appearances=20,
    )
    prob = player_prop_probability(profile, "goals", None)
    expected = 1.0 - math.exp(-0.5)
    assert math.isclose(prob, expected, rel_tol=1e-9)


def test_player_prop_probability_decreases_as_line_increases() -> None:
    profile = PlayerRateProfile(
        player_id="pl_1",
        player_name="Forward",
        team_id="tm_1",
        minutes_per_appearance=90.0,
        goals_per_90=0.3,
        assists_per_90=0.2,
        shots_per_90=3.0,
        shots_on_target_per_90=1.5,
        appearances=20,
    )
    over_0_5 = player_prop_probability(profile, "total_shots", 0.5)
    over_2_5 = player_prop_probability(profile, "total_shots", 2.5)
    assert over_0_5 > over_2_5


def test_player_prop_probability_zero_lambda_is_zero() -> None:
    idle_profile = PlayerRateProfile(
        player_id="pl_3",
        player_name="Unused",
        team_id="tm_1",
        minutes_per_appearance=0.0,
        goals_per_90=0.0,
        assists_per_90=0.0,
        shots_per_90=0.0,
        shots_on_target_per_90=0.0,
        appearances=0,
    )
    assert player_prop_probability(idle_profile, "goals", None) == 0.0
