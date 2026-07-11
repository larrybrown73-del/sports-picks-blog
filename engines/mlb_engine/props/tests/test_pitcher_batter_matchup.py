"""Tests for pitcher-hitter style interaction bridge."""

from __future__ import annotations

from baseball_props.analysis.hitter_discipline import BatterDisciplineProfile
from baseball_props.analysis.pitcher_batter_matchup import apply_pitcher_hitter_matchup


class _FakePitcher:
    pass


def _install_bridge(monkeypatch, *, gb: bool = False, power: bool = False, stability=(1.0, None)):
    pitcher = _FakePitcher()

    def _fetch(_pid, season=None, pitcher_name=""):
        return pitcher

    monkeypatch.setattr(
        "baseball_props.analysis.pitcher_batter_matchup._load_predictor_pitcher_matchup",
        lambda: {
            "fetch_pitcher_season_profile": _fetch,
            "is_ground_ball_pitcher": lambda _p: gb,
            "is_power_pitcher": lambda _p: power,
            "is_velo_struggler": lambda _team_id, _season: False,
            "pitcher_runs_allowed_scalar": lambda _p: stability,
            "PATIENT_LINEUP_ADVANTAGE": 1.12,
            "VELO_DOMINANCE_SCALAR": 0.88,
        },
    )
    return pitcher


def test_gb_pitcher_synergy_with_elite_discipline(monkeypatch) -> None:
    _install_bridge(monkeypatch, gb=True)
    discipline = BatterDisciplineProfile("1", k_pct=15.0, bb_pct=13.0)
    adjustments: dict[str, float] = {}
    proj = apply_pitcher_hitter_matchup(
        2.0,
        opponent_pitcher_id="123",
        discipline=discipline,
        batting_mlb_team_id=110,
        season=2025,
        adjustments=adjustments,
        warnings=[],
    )
    assert proj == 2.0 * 1.12
    assert adjustments["gb_pitcher_discipline_synergy"] == 1.12


def test_velo_dominance_compounds_erratic_swinger(monkeypatch) -> None:
    _install_bridge(monkeypatch, power=True)
    discipline = BatterDisciplineProfile("2", k_pct=30.0, bb_pct=5.0)
    adjustments: dict[str, float] = {}
    proj = apply_pitcher_hitter_matchup(
        2.0,
        opponent_pitcher_id="456",
        discipline=discipline,
        batting_mlb_team_id=110,
        season=2025,
        adjustments=adjustments,
        warnings=[],
    )
    assert proj == 2.0 * 0.88
    assert adjustments["velo_erratic_synergy"] == 0.88


def test_pitcher_stability_scalar_applies_before_synergy(monkeypatch) -> None:
    _install_bridge(monkeypatch, gb=True, stability=(1.15, "regression_penalty"))
    discipline = BatterDisciplineProfile("3", k_pct=15.0, bb_pct=13.0)
    adjustments: dict[str, float] = {}
    proj = apply_pitcher_hitter_matchup(
        2.0,
        opponent_pitcher_id="789",
        discipline=discipline,
        batting_mlb_team_id=110,
        season=2025,
        adjustments=adjustments,
        warnings=[],
    )
    assert proj == 2.0 * 1.15 * 1.12
    assert adjustments["pitcher_regression_penalty"] == 1.15
