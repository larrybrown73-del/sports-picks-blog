from __future__ import annotations

import pandas as pd
import pytest

from baseball_props.analysis.pitcher_projection import (
    _apply_hook_floor,
    _compute_proj_outs,
    _project_pitch_count,
    project_pitcher_outs_and_pitches,
)
from baseball_props.config import (
    FALLBACK_RELIEF_OUTS,
    FALLBACK_STARTER_OUTS,
    HOOK_FLOOR_MAX_PITCHES,
    HOOK_FLOOR_MIN_PITCHES,
    LEAGUE_PITCHES_PER_OUT,
    LEAGUE_STARTER_OUTS,
    MAX_PROJ_OUTS,
    PITCHES_PER_STRIKEOUT,
    PITCHES_PER_WALK,
)
from baseball_props.data.mock_slate import build_mock_slate
from baseball_props.opportunity.batters import project_batter_pa
from baseball_props.opportunity.pitchers import project_pitcher_outs


def _single_game() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "G001",
                "game_date": "2025-06-27",
                "home_team_id": "BOS",
                "away_team_id": "NYY",
                "park_id": "FEN",
                "sp_home_id": "P101",
                "sp_away_id": "P102",
                "sp_home_hand": "L",
                "sp_away_hand": "R",
            }
        ]
    )


def _tendency_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "pitcher_name": "Test Pitcher",
        "avg_outs_last5": 17.0,
        "pitch_efficiency": 5.2,
        "gs": 15,
        "is_true_starter": True,
        "sp_k_pct": 0.230,
        "sp_bb_pct": 0.080,
        "avg_bf_per_start": 25.0,
    }
    base.update(overrides)
    return base


def _marquee_tendency(avg_outs: float, efficiency: float = 5.2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _tendency_row(
                pitcher_id="P104",
                pitcher_name="Yoshinobu Yamamoto",
                avg_outs_last5=avg_outs,
                pitch_efficiency=efficiency,
                gs=17,
            )
        ]
    )


def test_pitch_count_not_canceled_efficiency_constant() -> None:
    """New formula must not collapse to avg_outs * 3.80."""
    games = _single_game()
    games.loc[0, "sp_home_id"] = "P999"
    games.loc[0, "sp_away_id"] = "P999"
    tendencies = pd.DataFrame(
        [
            _tendency_row(
                pitcher_id="P999",
                pitcher_name="Test Ace",
                avg_outs_last5=18.0,
                pitch_efficiency=5.0,
                gs=20,
            )
        ]
    )
    team_pitching = pd.DataFrame(
        [
            {
                "team_id": "NYY",
                "role": "sp",
                "woba_allowed": 0.31,
                "iso_allowed": 0.14,
                "k_pct": 0.25,
                "bb_pct": 0.08,
            }
        ]
    )
    projected = pd.DataFrame(
        [{"game_id": "G001", "team_id": "BOS", "player_id": "H010", "proj_pa": 36.0}]
    )

    result = project_pitcher_outs_and_pitches(
        games, tendencies, projected, team_pitching
    )
    old_buggy = 18.0 * 3.80
    assert result.iloc[0]["proj_pitch_count"] > old_buggy + 15


def test_dynamic_pitch_formula_exclusive_buckets() -> None:
    proj_outs = 18.0
    proj_bf = 25.0
    sp_k_pct = 0.25
    sp_bb_pct = 0.08

    scaled_bf, proj_k, proj_bb, contact_outs, proj_pitches = _project_pitch_count(
        proj_outs, proj_bf, sp_k_pct, sp_bb_pct, avg_outs_last5=15.0, gs=0
    )

    expected_k = scaled_bf * sp_k_pct
    expected_bb = scaled_bf * sp_bb_pct
    expected_contact = max(proj_outs - expected_k, 0.0)
    expected_pitches = (
        expected_contact * LEAGUE_PITCHES_PER_OUT
        + expected_bb * PITCHES_PER_WALK
        + expected_k * PITCHES_PER_STRIKEOUT
    )

    assert scaled_bf == pytest.approx(proj_bf)
    assert proj_k == pytest.approx(expected_k)
    assert proj_bb == pytest.approx(expected_bb)
    assert contact_outs == pytest.approx(expected_contact)
    assert proj_pitches == pytest.approx(expected_pitches)


def test_starter_mask_gs_zero_uses_league_defaults() -> None:
    """Relief-only profile should not inflate starter depth."""
    games = pd.DataFrame(
        [
            {
                "game_id": "G001",
                "home_team_id": "BOS",
                "away_team_id": "NYY",
                "sp_home_id": "999001",
                "sp_away_id": "999002",
            }
        ]
    )
    # Without pybaseball lookup, defaults apply — gs=0 path is tested via mock tendency
    tendencies = pd.DataFrame(
        [
            _tendency_row(
                pitcher_id="P101",
                pitcher_name="Opener",
                avg_outs_last5=LEAGUE_STARTER_OUTS,
                pitch_efficiency=LEAGUE_PITCHES_PER_OUT,
                gs=0,
                is_true_starter=False,
            )
        ]
    )
    games.loc[0, "sp_home_id"] = "P101"
    result = project_pitcher_outs_and_pitches(games, tendencies)
    assert result.iloc[0]["proj_outs"] == pytest.approx(FALLBACK_RELIEF_OUTS)


def test_corrupted_efficiency_capped_below_max_outs() -> None:
    """FanGraphs failure paths with tiny efficiency must not explode outs."""
    games = _single_game()
    games.loc[0, "sp_home_id"] = "BAD1"
    games.loc[0, "sp_away_id"] = "BAD1"
    tendencies = pd.DataFrame(
        [
            _tendency_row(
                pitcher_id="BAD1",
                avg_outs_last5=17.0,
                pitch_efficiency=0.5,
                gs=15,
            )
        ]
    )
    result = project_pitcher_outs_and_pitches(games, tendencies)
    assert result.iloc[0]["proj_outs"] <= MAX_PROJ_OUTS


def test_implausible_avg_outs_uses_starter_fallback() -> None:
    games = _single_game()
    games.loc[0, "sp_home_id"] = "BAD2"
    games.loc[0, "sp_away_id"] = "BAD2"
    tendencies = pd.DataFrame(
        [
            _tendency_row(
                pitcher_id="BAD2",
                avg_outs_last5=91.0,
                pitch_efficiency=0.97,
                gs=10,
            )
        ]
    )
    result = project_pitcher_outs_and_pitches(games, tendencies)
    assert result.iloc[0]["proj_outs"] == pytest.approx(FALLBACK_STARTER_OUTS)


def test_compute_proj_outs_never_exceeds_ceiling() -> None:
    assert _compute_proj_outs(95.0, LEAGUE_PITCHES_PER_OUT, gs=20) <= MAX_PROJ_OUTS
    assert _compute_proj_outs(17.0, 0.1, gs=20) <= MAX_PROJ_OUTS


def test_all_projected_outs_capped_at_max() -> None:
    games = _single_game()
    games.loc[0, "sp_home_id"] = "MAX1"
    games.loc[0, "sp_away_id"] = "MAX2"
    tendencies = pd.DataFrame(
        [
            _tendency_row(pitcher_id="MAX1", avg_outs_last5=91.0, pitch_efficiency=0.5, gs=12),
            _tendency_row(pitcher_id="MAX2", avg_outs_last5=17.0, pitch_efficiency=0.1, gs=15),
        ]
    )
    result = project_pitcher_outs_and_pitches(games, tendencies)
    assert (result["proj_outs"] <= MAX_PROJ_OUTS).all()


def test_hook_floor_marquee_starter_18_outs() -> None:
    raw = 70.0
    floored = _apply_hook_floor(raw, proj_outs=18.0, avg_outs_last5=18.1, gs=17)
    assert HOOK_FLOOR_MIN_PITCHES <= floored <= HOOK_FLOOR_MAX_PITCHES
    assert floored > raw


def test_hook_floor_not_applied_sub_five_ip() -> None:
    raw = 72.0
    floored = _apply_hook_floor(raw, proj_outs=14.0, avg_outs_last5=18.1, gs=17)
    assert floored == raw


def test_sub_five_ip_still_above_old_cap() -> None:
    games = _single_game()
    games.loc[0, "sp_home_id"] = "P103"
    games.loc[0, "sp_away_id"] = "P103"
    tendencies = pd.DataFrame(
        [
            _tendency_row(
                pitcher_id="P103",
                pitcher_name="Logan Webb",
                avg_outs_last5=14.0,
                pitch_efficiency=5.25,
            )
        ]
    )
    projected = pd.DataFrame(
        [{"game_id": "G001", "team_id": "NYY", "player_id": "H001", "proj_pa": 36.0}]
    )
    team_pitching = pd.DataFrame(
        [
            {
                "team_id": "SF",
                "role": "sp",
                "woba_allowed": 0.31,
                "iso_allowed": 0.14,
                "k_pct": 0.225,
                "bb_pct": 0.082,
            }
        ]
    )
    result = project_pitcher_outs_and_pitches(
        games, tendencies, projected, team_pitching
    )
    assert result.iloc[0]["proj_outs"] < 15.0
    assert result.iloc[0]["proj_pitch_count"] > 53.0


def test_mock_slate_pitch_counts_realistic() -> None:
    frames = build_mock_slate()
    projected = project_batter_pa(
        frames["lineups"], frames["vegas_totals"], frames["slate_games"]
    )

    result = project_pitcher_outs_and_pitches(
        frames["slate_games"],
        frames["pitcher_tendencies"],
        projected,
        frames["team_pitching"],
    )

    yama = result[result["pitcher_id"] == "P104"].iloc[0]
    assert yama["proj_pitch_count"] >= 85.0
    assert yama["proj_pitch_count"] <= 105.0

    webb = result[result["pitcher_id"] == "P103"].iloc[0]
    assert webb["proj_pitch_count"] > 59.0


def test_pitcher_specific_k_rates_change_pitch_count() -> None:
    games = _single_game()
    games.loc[0, "sp_home_id"] = "PA"
    games.loc[0, "sp_away_id"] = "PB"
    low_k = pd.DataFrame([_tendency_row(pitcher_id="PA", sp_k_pct=0.180, avg_bf_per_start=24.0)])
    high_k = pd.DataFrame([_tendency_row(pitcher_id="PB", sp_k_pct=0.320, avg_bf_per_start=27.0)])
    tendencies = pd.concat([low_k, high_k], ignore_index=True)
    projected = pd.DataFrame(
        [{"game_id": "G001", "team_id": "NYY", "player_id": "H001", "proj_pa": 36.0}]
    )

    result = project_pitcher_outs_and_pitches(games, tendencies, projected)
    pa_row = result[result["pitcher_id"] == "PA"].iloc[0]
    pb_row = result[result["pitcher_id"] == "PB"].iloc[0]
    assert pa_row["proj_k"] < pb_row["proj_k"]
    assert pa_row["proj_pitch_count"] != pb_row["proj_pitch_count"]


def test_opportunity_wrapper_delegates() -> None:
    frames = build_mock_slate()
    direct = project_pitcher_outs_and_pitches(
        frames["slate_games"], frames["pitcher_tendencies"]
    )
    wrapped = project_pitcher_outs(frames["slate_games"], frames["pitcher_tendencies"])
    pd.testing.assert_frame_equal(direct, wrapped)
