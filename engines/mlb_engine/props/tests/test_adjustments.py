from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baseball_props.core.adjustments import (
    calculate_projected_probability,
    calculate_projected_rate,
    log_odds_to_rate,
    rate_to_log_odds,
)


class TestCalculateProjectedRate:
    def test_neutral_player_and_opponent_returns_league_avg(self) -> None:
        league = 0.313
        result = calculate_projected_rate(league, league, league)
        assert result == pytest.approx(league)

    def test_scalar_boost_when_opponent_weak(self) -> None:
        result = calculate_projected_rate(0.350, 0.350, 0.313)
        assert result == pytest.approx(0.350 * 0.350 / 0.313)

    def test_vectorized_series(self) -> None:
        player = pd.Series([0.320, 0.300])
        opponent = pd.Series([0.330, 0.290])
        league = 0.313
        result = calculate_projected_rate(player, opponent, league)
        expected = player * opponent / league
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_zero_league_avg_fallback(self) -> None:
        result = calculate_projected_rate(0.320, 0.330, 0.0)
        assert result == pytest.approx(0.320)

    def test_clipping(self) -> None:
        result = calculate_projected_rate(0.50, 0.50, 0.10, clip_max=0.40)
        assert result == pytest.approx(0.40)


class TestLogOdds:
    def test_round_trip(self) -> None:
        p = np.array([0.1, 0.5, 0.9])
        recovered = log_odds_to_rate(rate_to_log_odds(p))
        np.testing.assert_allclose(recovered, p, rtol=1e-5)

    def test_log_odds_equivalent_to_multiplicative(self) -> None:
        league_p = 0.223
        player_p = 0.250
        opponent_p = 0.240

        multiplicative = calculate_projected_rate(player_p, opponent_p, league_p)
        log_odds = calculate_projected_probability(player_p, opponent_p, league_p)
        assert multiplicative == pytest.approx(log_odds, rel=0.01)

    def test_neutral_probability_returns_league(self) -> None:
        league_p = 0.085
        result = calculate_projected_probability(league_p, league_p, league_p)
        assert result == pytest.approx(league_p, rel=1e-4)
