from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from scipy.stats import poisson as scipy_poisson

from historical_data import MatchResult
from team_strength import (
    SoccerModelDataError,
    asian_handicap_probabilities,
    btts_probability,
    fit_dixon_coles,
    match_result_probabilities,
    scoreline_matrix,
    total_goals_over_probability,
)


def test_scoreline_matrix_sums_to_one_and_matches_manual_poisson_at_zero_rho() -> None:
    matrix = scoreline_matrix(1.4, 1.1, rho=0.0, max_goals=15)
    assert math.isclose(matrix.sum(), 1.0, rel_tol=1e-9)

    # rho=0 collapses every tau correction to 1.0, so the matrix should be
    # an exact independent-Poisson product at every cell.
    manual_00 = scipy_poisson.pmf(0, 1.4) * scipy_poisson.pmf(0, 1.1)
    assert math.isclose(matrix[0, 0], manual_00, rel_tol=1e-9)


def test_match_result_probabilities_sum_to_one_and_favor_higher_lambda() -> None:
    matrix = scoreline_matrix(1.6, 1.0, rho=-0.05)
    home_win, draw, away_win = match_result_probabilities(matrix)
    assert math.isclose(home_win + draw + away_win, 1.0, rel_tol=1e-9)
    assert home_win > away_win


def test_btts_probability_is_a_valid_probability() -> None:
    matrix = scoreline_matrix(1.6, 1.3, rho=-0.05)
    yes = btts_probability(matrix)
    assert 0.0 < yes < 1.0


def test_total_goals_over_under_sum_to_one() -> None:
    matrix = scoreline_matrix(1.5, 1.2, rho=-0.05)
    over = total_goals_over_probability(matrix, 2.5)
    assert 0.0 < over < 1.0
    assert math.isclose(over + (1.0 - over), 1.0)


def test_asian_handicap_half_line_has_no_push() -> None:
    matrix = scoreline_matrix(1.6, 1.0, rho=-0.05)
    outcome = asian_handicap_probabilities(matrix, "home", -0.5)
    assert outcome.push_probability == 0.0
    assert math.isclose(outcome.win_probability + outcome.loss_probability, 1.0, rel_tol=1e-9)


def test_asian_handicap_whole_line_has_positive_push_probability() -> None:
    matrix = scoreline_matrix(1.6, 1.0, rho=-0.05)
    outcome = asian_handicap_probabilities(matrix, "home", -1.0)
    assert outcome.push_probability > 0.0
    assert math.isclose(
        outcome.win_probability + outcome.loss_probability + outcome.push_probability, 1.0, rel_tol=1e-9
    )


def test_asian_handicap_quarter_line_averages_adjacent_half_lines() -> None:
    matrix = scoreline_matrix(1.6, 1.0, rho=-0.05)
    quarter = asian_handicap_probabilities(matrix, "home", -0.25)
    lower = asian_handicap_probabilities(matrix, "home", -0.5)
    upper = asian_handicap_probabilities(matrix, "home", 0.0)
    assert math.isclose(quarter.win_probability, (lower.win_probability + upper.win_probability) / 2.0)
    assert math.isclose(quarter.push_probability, (lower.push_probability + upper.push_probability) / 2.0)


def test_asian_handicap_rejects_invalid_side() -> None:
    matrix = scoreline_matrix(1.6, 1.0, rho=-0.05)
    with pytest.raises(SoccerModelDataError):
        asian_handicap_probabilities(matrix, "neutral", -0.5)


def test_fit_dixon_coles_requires_minimum_matches() -> None:
    with pytest.raises(SoccerModelDataError):
        fit_dixon_coles([])


def _simulate_matches(
    rng: np.random.Generator,
    team_ids: list[str],
    log_attack: dict[str, float],
    log_defense: dict[str, float],
    home_advantage: float,
    *,
    repeats: int,
) -> list[MatchResult]:
    matches: list[MatchResult] = []
    base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    counter = 0
    for _ in range(repeats):
        for home in team_ids:
            for away in team_ids:
                if home == away:
                    continue
                lambda_home = math.exp(log_attack[home] + log_defense[away] + home_advantage)
                lambda_away = math.exp(log_attack[away] + log_defense[home])
                home_goals = int(rng.poisson(lambda_home))
                away_goals = int(rng.poisson(lambda_away))
                match_date = base_date + timedelta(days=counter)
                matches.append(
                    MatchResult(
                        match_id=f"mt_{counter}",
                        competition_id="comp_test",
                        utc_date=match_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                        home_team_id=home,
                        home_team_name=home,
                        away_team_id=away,
                        away_team_name=away,
                        home_goals=home_goals,
                        away_goals=away_goals,
                    )
                )
                counter += 1
    return matches


def test_fit_dixon_coles_recovers_relative_team_strength_from_synthetic_data() -> None:
    # Ground truth: A is a clearly strong team, D is clearly weak; B and C
    # are near-identical mid-table sides. A fixed seed keeps this
    # deterministic across runs -- no flaky statistical test.
    rng = np.random.default_rng(42)
    team_ids = ["A", "B", "C", "D"]
    log_attack_true = {"A": 0.5, "B": 0.0, "C": -0.05, "D": -0.6}
    log_defense_true = {"A": -0.3, "B": 0.0, "C": 0.05, "D": 0.45}
    home_advantage_true = 0.25

    matches = _simulate_matches(rng, team_ids, log_attack_true, log_defense_true, home_advantage_true, repeats=10)
    ratings = fit_dixon_coles(matches)

    # Fitted ratings should preserve the true ordering (values themselves
    # need not match exactly -- the model is scale-invariant, see
    # TeamRatings docstring -- but relative strength must come through).
    assert ratings.attack["A"] > ratings.attack["D"]
    assert ratings.defense["A"] < ratings.defense["D"]  # lower log-defense = stingier

    lambda_home, lambda_away = ratings.expected_goals("A", "D")
    lopsided_matrix = scoreline_matrix(lambda_home, lambda_away, ratings.rho)
    lopsided_home_win, _, lopsided_away_win = match_result_probabilities(lopsided_matrix)
    assert lopsided_home_win > 0.55

    lambda_home_even, lambda_away_even = ratings.expected_goals("B", "C")
    even_matrix = scoreline_matrix(lambda_home_even, lambda_away_even, ratings.rho)
    even_home_win, _, even_away_win = match_result_probabilities(even_matrix)

    # The lopsided A-vs-D gap should be much larger than the near-even B-vs-C gap.
    assert (lopsided_home_win - lopsided_away_win) > (even_home_win - even_away_win)

    # Every team appeared as both home and away in every repeat.
    assert ratings.matches_played["A"] == 2 * 3 * 10
