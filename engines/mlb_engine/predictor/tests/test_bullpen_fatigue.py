"""Tests for late-inning bullpen fatigue and lockdown bonuses."""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import patch

from bullpen_fatigue import (
    BullpenStatus,
    RelieverWorkload,
    _is_overworked,
    _late_inning_game_multiplier,
    apply_bullpen_to_runs,
    compute_bullpen_fatigue,
)
from config import (
    LATE_INNING_RUN_SHARE,
    OVERWORKED_BULLPEN_PENALTY,
    RESTED_ELITE_BONUS,
)
from model import implied_win_probabilities


def test_overworked_reliever_detection() -> None:
    workload = RelieverWorkload(
        person_id=1,
        name="Closer",
        pitches_by_date={
            date(2026, 7, 7): 20,
            date(2026, 7, 8): 18,
        },
        total_pitches=38,
    )
    assert _is_overworked(workload)

    fresh = RelieverWorkload(
        person_id=2,
        name="Setup",
        pitches_by_date={date(2026, 7, 8): 18},
        total_pitches=18,
    )
    assert not _is_overworked(fresh)


def test_dead_arm_boosts_opponent_late_inning_runs() -> None:
    status = BullpenStatus(
        home_status="Dead Arm",
        away_status="Fresh",
        home_opponent_late_scalar=OVERWORKED_BULLPEN_PENALTY,
        away_opponent_late_scalar=1.0,
    )
    home_runs, away_runs, tags = apply_bullpen_to_runs(4.0, 4.5, status)
    expected_away_mult = _late_inning_game_multiplier(OVERWORKED_BULLPEN_PENALTY)
    assert away_runs == 4.5 * expected_away_mult
    assert home_runs == 4.0
    assert any("away_runs_late_inning" in tag for tag in tags)


def test_lockdown_suppresses_opponent_late_inning_runs() -> None:
    status = BullpenStatus(
        home_status="Lockdown",
        away_status="Fresh",
        home_opponent_late_scalar=RESTED_ELITE_BONUS,
        away_opponent_late_scalar=1.0,
    )
    home_runs, away_runs, _tags = apply_bullpen_to_runs(5.0, 5.0, status)
    expected_away_mult = _late_inning_game_multiplier(RESTED_ELITE_BONUS)
    assert away_runs == 5.0 * expected_away_mult
    assert home_runs == 5.0


def test_bad_starter_and_dead_arm_tanks_win_probability() -> None:
    base_home, base_away = 3.2, 5.4
    status = BullpenStatus(
        home_status="Dead Arm",
        away_status="Fresh",
        home_opponent_late_scalar=OVERWORKED_BULLPEN_PENALTY,
        away_opponent_late_scalar=1.0,
    )
    adjusted_home, adjusted_away, _ = apply_bullpen_to_runs(base_home, base_away, status)
    base_home_prob, _ = implied_win_probabilities(base_home, base_away)
    adjusted_home_prob, _ = implied_win_probabilities(adjusted_home, adjusted_away)
    assert adjusted_home_prob < base_home_prob


@patch("bullpen_fatigue._late_inning_scalar_for_team")
def test_compute_bullpen_fatigue_status_labels(mock_scalar) -> None:
    mock_scalar.side_effect = [
        (OVERWORKED_BULLPEN_PENALTY, "Dead Arm", ["dead_arm:Closer:46p"]),
        (1.0, "Fresh", []),
    ]
    status = compute_bullpen_fatigue(147, 121, datetime(2026, 7, 9), season=2026)
    assert status.home_status == "Dead Arm"
    assert status.away_status == "Fresh"
    assert status.home_opponent_late_scalar == OVERWORKED_BULLPEN_PENALTY


def test_late_inning_share_only_adjusts_fraction_of_runs() -> None:
    mult = _late_inning_game_multiplier(1.18)
    assert mult == (1.0 - LATE_INNING_RUN_SHARE) + LATE_INNING_RUN_SHARE * 1.18
