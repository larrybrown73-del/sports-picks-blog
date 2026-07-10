"""Tests for full-season pitcher matchup adjustments."""

from __future__ import annotations

from unittest.mock import patch

from pitcher_matchup import (
    PitcherSeasonProfile,
    TeamOffenseProfile,
    _apply_offense_adjustments,
    is_ground_ball_pitcher,
    is_patient_lineup,
    is_power_pitcher,
    pitcher_runs_allowed_scalar,
)


def _pitcher(
    *,
    era: float | None = 4.00,
    whip: float | None = 1.20,
    babip: float | None = 0.300,
    velo: float | None = 93.0,
    gb_pct: float | None = 42.0,
) -> PitcherSeasonProfile:
    return PitcherSeasonProfile(
        pitcher_id=1,
        pitcher_name="Test SP",
        season_era=era,
        season_whip=whip,
        season_babip=babip,
        avg_fastball_velo=velo,
        ground_ball_pct=gb_pct,
    )


def test_babip_luck_bonus_reduces_runs_allowed() -> None:
    scalar, tag = pitcher_runs_allowed_scalar(
        _pitcher(era=3.40, babip=0.335, whip=1.18)
    )
    assert tag == "babip_luck_bonus"
    assert scalar == 0.95


def test_regression_penalty_increases_runs_allowed() -> None:
    scalar, tag = pitcher_runs_allowed_scalar(
        _pitcher(era=3.20, whip=1.40, babip=0.255)
    )
    assert tag == "regression_penalty"
    assert scalar == 1.15


def test_power_pitcher_and_gb_flags() -> None:
    assert is_power_pitcher(_pitcher(velo=96.2))
    assert not is_power_pitcher(_pitcher(velo=94.0))
    assert is_ground_ball_pitcher(_pitcher(gb_pct=52.0))
    assert not is_ground_ball_pitcher(_pitcher(gb_pct=48.0))


def test_patient_lineup_detection() -> None:
    assert is_patient_lineup(TeamOffenseProfile(1, walk_pct=10.1, pitches_per_pa=3.80))
    assert is_patient_lineup(TeamOffenseProfile(1, walk_pct=8.0, pitches_per_pa=4.05))
    assert not is_patient_lineup(TeamOffenseProfile(1, walk_pct=8.0, pitches_per_pa=3.80))


def test_offense_adjustments_stack_matchup_scalars() -> None:
    pitcher = _pitcher(era=3.20, whip=1.40, babip=0.255, velo=96.5, gb_pct=54.0)
    offense = TeamOffenseProfile(team_id=99, walk_pct=10.0, pitches_per_pa=4.10)

    with patch("pitcher_matchup.is_velo_struggler", return_value=False):
        runs, tags = _apply_offense_adjustments(
            5.0,
            pitcher=pitcher,
            offense=offense,
            season=2026,
            label="SP",
        )

    assert runs == 5.0 * 1.15 * 1.12
    assert any("regression_penalty" in tag for tag in tags)
    assert any("patient_lineup" in tag for tag in tags)


def test_velo_dominance_penalizes_offense() -> None:
    pitcher = _pitcher(velo=96.5)
    offense = TeamOffenseProfile(team_id=99, walk_pct=7.0, pitches_per_pa=3.80)

    with patch("pitcher_matchup.is_velo_struggler", return_value=True):
        runs, tags = _apply_offense_adjustments(
            4.0,
            pitcher=pitcher,
            offense=offense,
            season=2026,
            label="SP",
        )

    assert runs == 4.0 * 0.88
    assert any("velo_dominance" in tag for tag in tags)
