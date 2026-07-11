from __future__ import annotations

import math

from ev_engine_core import MarketLeg, rank_ev_board
from lineups import MatchLineup
from player_props_model import PlayerRateProfile
from projection_model import SoccerProjectionModel, normalize_probability_scale
from team_strength import TeamRatings, match_result_probabilities, scoreline_matrix


def _fixture_ratings() -> TeamRatings:
    return TeamRatings(
        attack={"tm_home": 0.3, "tm_away": -0.2},
        defense={"tm_home": -0.1, "tm_away": 0.2},
        home_advantage=0.25,
        rho=-0.05,
        league_avg_home_goals=1.4,
        league_avg_away_goals=1.1,
        matches_played={"tm_home": 30, "tm_away": 28},
    )


def test_normalize_probability_scale_passes_decimals_and_rescales_percentages() -> None:
    assert normalize_probability_scale(0.65) == 0.65
    assert math.isclose(normalize_probability_scale(65.0), 0.65)


def test_grade_moneyline_matches_direct_team_strength_calculation() -> None:
    ratings = _fixture_ratings()
    model = SoccerProjectionModel(ratings)

    leg = MarketLeg(
        game_id="mt_1",
        market_type="match_odds",
        selection="Home Team",
        odds=1.9,
        odds_format="decimal",
        sportsbook="Bet365",
        side="home",
        team="Home Team",
        metadata={"home_team_id": "tm_home", "away_team_id": "tm_away"},
    )
    estimate = model(leg)

    lambda_home, lambda_away = ratings.expected_goals("tm_home", "tm_away")
    matrix = scoreline_matrix(lambda_home, lambda_away, ratings.rho)
    home_win, _, _ = match_result_probabilities(matrix)

    assert math.isclose(estimate.true_probability, home_win, rel_tol=1e-9)
    assert estimate.sample_size == 28  # min(30, 28)
    assert not estimate.warnings


def test_grade_missing_team_ids_returns_warned_zero_not_crash() -> None:
    model = SoccerProjectionModel(_fixture_ratings())
    leg = MarketLeg(
        game_id="mt_2",
        market_type="match_odds",
        selection="Home Team",
        odds=1.9,
        odds_format="decimal",
        sportsbook="Bet365",
        side="home",
    )
    estimate = model(leg)
    assert estimate.true_probability == 0.0
    assert any("home_team_id" in w for w in estimate.warnings)


def test_grade_asian_handicap_whole_line_excludes_push_from_win_plus_loss() -> None:
    ratings = _fixture_ratings()
    model = SoccerProjectionModel(ratings)
    leg = MarketLeg(
        game_id="mt_3",
        market_type="asian_handicap",
        selection="Home -1.0 AH",
        odds=1.95,
        odds_format="decimal",
        sportsbook="Bet365",
        side="home",
        team="Home Team",
        line=-1.0,
        metadata={"home_team_id": "tm_home", "away_team_id": "tm_away"},
    )
    estimate = model(leg)
    assert estimate.loss_probability is not None
    assert estimate.true_probability + estimate.loss_probability < 1.0
    assert any("push probability" in w for w in estimate.warnings)


def test_grade_player_market_uses_rate_profile() -> None:
    ratings = _fixture_ratings()
    profile = PlayerRateProfile(
        player_id="pl_1",
        player_name="Test Forward",
        team_id="tm_home",
        minutes_per_appearance=85.0,
        goals_per_90=0.5,
        assists_per_90=0.1,
        shots_per_90=3.0,
        shots_on_target_per_90=1.4,
        appearances=25,
    )
    model = SoccerProjectionModel(ratings, {"pl_1": profile})

    leg = MarketLeg(
        game_id="mt_4",
        market_type="player_shots",
        selection="Test Forward Over 2.5",
        odds=2.1,
        odds_format="decimal",
        sportsbook="DraftKings",
        entity_id="pl_1",
        entity_name="Test Forward",
        line=2.5,
        metadata={"home_team_id": "tm_home", "away_team_id": "tm_away"},
    )
    estimate = model(leg)
    assert 0.0 < estimate.true_probability < 1.0
    assert estimate.sample_size == 25


def _player_leg(*, entity_id: str, line: float, game_id: str = "mt_4") -> MarketLeg:
    return MarketLeg(
        game_id=game_id,
        market_type="player_shots",
        selection="Test Forward Over 2.5",
        odds=2.1,
        odds_format="decimal",
        sportsbook="DraftKings",
        entity_id=entity_id,
        entity_name="Test Forward",
        line=line,
        metadata={"home_team_id": "tm_home", "away_team_id": "tm_away"},
    )


def test_grade_player_market_without_lineup_falls_back_to_historical_estimate() -> None:
    ratings = _fixture_ratings()
    profile = PlayerRateProfile(
        player_id="pl_1",
        player_name="Test Forward",
        team_id="tm_home",
        minutes_per_appearance=85.0,
        goals_per_90=0.5,
        assists_per_90=0.1,
        shots_per_90=3.0,
        shots_on_target_per_90=1.4,
        appearances=25,
    )
    model = SoccerProjectionModel(ratings, {"pl_1": profile})

    estimate = model(_player_leg(entity_id="pl_1", line=2.5))

    assert 0.0 < estimate.true_probability < 1.0
    assert any("no confirmed lineup yet" in w for w in estimate.warnings)


def test_grade_player_market_confirmed_not_in_squad_forces_zero_probability() -> None:
    ratings = _fixture_ratings()
    profile = PlayerRateProfile(
        player_id="pl_1",
        player_name="Test Forward",
        team_id="tm_home",
        minutes_per_appearance=85.0,
        goals_per_90=0.5,
        assists_per_90=0.1,
        shots_per_90=3.0,
        shots_on_target_per_90=1.4,
        appearances=25,
    )
    model = SoccerProjectionModel(ratings, {"pl_1": profile})
    model.add_lineup(
        MatchLineup(
            match_id="mt_4",
            confirmed=True,
            starting_player_ids=frozenset(),
            substitute_player_ids=frozenset(),
            home_team_id="tm_home",
            away_team_id="tm_away",
        )
    )

    estimate = model(_player_leg(entity_id="pl_1", line=2.5))

    assert estimate.true_probability == 0.0
    assert any("excludes this player from the matchday squad" in w for w in estimate.warnings)


def test_grade_player_market_confirmed_starter_still_uses_historical_rate() -> None:
    ratings = _fixture_ratings()
    profile = PlayerRateProfile(
        player_id="pl_1",
        player_name="Test Forward",
        team_id="tm_home",
        minutes_per_appearance=85.0,
        goals_per_90=0.5,
        assists_per_90=0.1,
        shots_per_90=3.0,
        shots_on_target_per_90=1.4,
        appearances=25,
    )
    model = SoccerProjectionModel(ratings, {"pl_1": profile})
    model.add_lineup(
        MatchLineup(
            match_id="mt_4",
            confirmed=True,
            starting_player_ids=frozenset({"pl_1"}),
            substitute_player_ids=frozenset(),
            home_team_id="tm_home",
            away_team_id="tm_away",
        )
    )

    without_lineup_estimate = model(_player_leg(entity_id="pl_1", line=2.5, game_id="mt_unconfirmed"))
    with_lineup_estimate = model(_player_leg(entity_id="pl_1", line=2.5))

    # STARTING behaves the same as "unknown" for this profile -- both use
    # the historical minutes-per-appearance heuristic.
    assert math.isclose(with_lineup_estimate.true_probability, without_lineup_estimate.true_probability)
    assert not any("excludes this player" in w for w in with_lineup_estimate.warnings)


def test_grade_player_market_unconfirmed_lineup_is_treated_as_no_lineup() -> None:
    ratings = _fixture_ratings()
    profile = PlayerRateProfile(
        player_id="pl_1",
        player_name="Test Forward",
        team_id="tm_home",
        minutes_per_appearance=85.0,
        goals_per_90=0.5,
        assists_per_90=0.1,
        shots_per_90=3.0,
        shots_on_target_per_90=1.4,
        appearances=25,
    )
    model = SoccerProjectionModel(ratings, {"pl_1": profile})
    # An unconfirmed lineup should never be trusted to zero out minutes.
    model.add_lineup(
        MatchLineup(
            match_id="mt_4",
            confirmed=False,
            starting_player_ids=frozenset(),
            substitute_player_ids=frozenset(),
            home_team_id="tm_home",
            away_team_id="tm_away",
        )
    )

    estimate = model(_player_leg(entity_id="pl_1", line=2.5))

    assert estimate.true_probability > 0.0
    assert any("no confirmed lineup yet" in w for w in estimate.warnings)


def test_grade_player_market_missing_profile_returns_warned_zero() -> None:
    model = SoccerProjectionModel(_fixture_ratings(), {})
    leg = MarketLeg(
        game_id="mt_5",
        market_type="player_shots",
        selection="Unknown Player Over 1.5",
        odds=2.0,
        odds_format="decimal",
        sportsbook="DraftKings",
        entity_id="pl_unknown",
        entity_name="Unknown Player",
        line=1.5,
    )
    estimate = model(leg)
    assert estimate.true_probability == 0.0
    assert any("No player rate profile" in w for w in estimate.warnings)


def test_grade_unsupported_market_family_returns_warned_zero() -> None:
    model = SoccerProjectionModel(_fixture_ratings())
    leg = MarketLeg(
        game_id="mt_6",
        market_type="match_corners",
        selection="Over 9.5 corners",
        odds=1.9,
        odds_format="decimal",
        sportsbook="Bet365",
        side="over",
        line=9.5,
    )
    estimate = model(leg)
    assert estimate.true_probability == 0.0
    assert "no projection model coverage" in estimate.warnings[0]


def test_first_goalscorer_market_unsupported_not_approximated() -> None:
    ratings = _fixture_ratings()
    profile = PlayerRateProfile(
        player_id="pl_1",
        player_name="Test Forward",
        team_id="tm_home",
        minutes_per_appearance=85.0,
        goals_per_90=0.5,
        assists_per_90=0.1,
        shots_per_90=3.0,
        shots_on_target_per_90=1.4,
        appearances=25,
    )
    model = SoccerProjectionModel(ratings, {"pl_1": profile})
    leg = MarketLeg(
        game_id="mt_7",
        market_type="first_goalscorer",
        selection="Test Forward",
        odds=6.0,
        odds_format="decimal",
        sportsbook="Bet365",
        entity_id="pl_1",
        entity_name="Test Forward",
    )
    estimate = model(leg)
    assert estimate.true_probability == 0.0
    assert any("first_goalscorer" in w for w in estimate.warnings)


def test_end_to_end_rank_ev_board_with_mixed_supported_and_unsupported_legs() -> None:
    ratings = _fixture_ratings()
    model = SoccerProjectionModel(ratings)

    legs = [
        MarketLeg(
            game_id="mt_8",
            market_type="match_odds",
            selection="Home Team",
            odds=2.2,
            odds_format="decimal",
            sportsbook="Bet365",
            side="home",
            team="Home Team",
            metadata={"home_team_id": "tm_home", "away_team_id": "tm_away"},
        ),
        MarketLeg(
            game_id="mt_8",
            market_type="match_corners",
            selection="Over 9.5 corners",
            odds=1.9,
            odds_format="decimal",
            sportsbook="Bet365",
            side="over",
            line=9.5,
        ),
    ]
    results = rank_ev_board(legs, model, positive_only=False, apply_correlation_filter=False)
    assert len(results) == 2
    corners_result = next(r for r in results if r.leg.market_type == "match_corners")
    assert corners_result.true_probability == 0.0
    assert corners_result.positive_ev is False
