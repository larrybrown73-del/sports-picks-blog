"""Tests for Tough Out, innings-eater luck, and pre-All-Star motivation."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from config import (
    BREAK_PUSH_BONUS,
    GRITTY_OFFENSE_SCALAR,
    LOOK_AHEAD_TRAP_PENALTY,
    LUCK_REGRESSION_PENALTY,
    MISSING_STAR_BAT_PENALTY,
    VACATION_MODE_PENALTY,
)
from hitter_discipline import LineupBatter
from tough_out import (
    InningsEaterProfile,
    TeamContactProfile,
    apply_system_guardrails,
    apply_tough_out_run_scalars,
    break_push_bonus,
    compute_fip,
    is_pre_all_star_window,
    look_ahead_trap_scalar,
    luck_regression_scalar,
    missing_star_bat_lineup_scalar,
    tough_out_team_ids,
    vacation_mode_scalar,
)


def test_compute_fip_value() -> None:
    fip = compute_fip(
        home_runs=10,
        walks=20,
        hit_by_pitch=2,
        strikeouts=60,
        innings_pitched=60.0,
        constant=3.10,
    )
    assert abs(fip - ((130 + 66 - 120) / 60 + 3.10)) < 1e-9


def test_pre_all_star_window_2026() -> None:
    assert is_pre_all_star_window(date(2026, 7, 9), season=2026)
    assert is_pre_all_star_window(date(2026, 7, 12), season=2026)
    assert not is_pre_all_star_window(date(2026, 7, 13), season=2026)
    assert not is_pre_all_star_window(date(2026, 7, 1), season=2026)


def test_luck_regression_when_era_beats_fip() -> None:
    profile = InningsEaterProfile(
        pitcher_id=1,
        k_bb_pct=11.0,
        era=3.50,
        fip=4.40,
        innings_pitched=80.0,
        is_innings_eater=True,
    )
    scalar, tag = luck_regression_scalar(profile)
    assert tag == "luck_regression"
    assert scalar == LUCK_REGRESSION_PENALTY


def test_luck_regression_skips_non_eaters() -> None:
    profile = InningsEaterProfile(
        pitcher_id=1,
        k_bb_pct=18.0,
        era=3.20,
        fip=4.50,
        innings_pitched=80.0,
        is_innings_eater=False,
    )
    assert luck_regression_scalar(profile) == (1.0, None)


def test_tough_out_classification_mid_power_contact() -> None:
    profiles = []
    for team_id in range(1, 31):
        contact = 80.0 - (team_id * 0.2)
        whiff = 15.0 + (team_id * 0.2)
        slugging = 0.480 - (team_id * 0.004)
        profiles.append(
            TeamContactProfile(
                team_id=team_id,
                contact_pct=contact,
                whiff_pct=whiff,
                slugging=slugging,
                plate_appearances=2000,
            )
        )

    with patch("tough_out._league_team_contact_profiles", return_value=tuple(profiles)):
        tough_out_team_ids.cache_clear()
        ids = tough_out_team_ids(2026)

    assert 9 in ids and 10 in ids
    assert 1 not in ids
    assert 30 not in ids


def test_gritty_offense_vs_innings_eater() -> None:
    eater = InningsEaterProfile(
        pitcher_id=55,
        k_bb_pct=11.0,
        era=4.10,
        fip=4.20,
        innings_pitched=90.0,
        is_innings_eater=True,
    )
    with (
        patch("tough_out.fetch_innings_eater_profile", return_value=eater),
        patch("tough_out.is_tough_out", return_value=True),
        patch("tough_out.is_pre_all_star_window", return_value=False),
        patch("tough_out.team_top_hr_batter_ids", return_value=tuple()),
        patch("tough_out.team_l10_win_pct", return_value=0.600),
        patch("tough_out.team_l14_wrc_plus_as_of", return_value=100.0),
    ):
        runs, tags = apply_tough_out_run_scalars(
            5.0,
            offense_team_id=147,
            pitcher_id=55,
            pitcher_era=4.10,
            is_home_offense=True,
            game_date=date(2026, 6, 1),
            season=2026,
            label="SP",
        )
    assert runs == 5.0 * GRITTY_OFFENSE_SCALAR
    assert any("gritty_offense" in tag for tag in tags)


def test_vacation_mode_road_dead_team() -> None:
    with (
        patch("tough_out.is_pre_all_star_window", return_value=True),
        patch("tough_out.team_win_pct", return_value=0.350),
    ):
        scalar, tag = vacation_mode_scalar(
            team_id=136,
            is_home=False,
            game_date=date(2026, 7, 10),
            season=2026,
        )
    assert scalar == VACATION_MODE_PENALTY
    assert tag == "vacation_mode"


def test_break_push_bonus_for_winning_tough_out() -> None:
    with (
        patch("tough_out.is_pre_all_star_window", return_value=True),
        patch("tough_out.is_tough_out", return_value=True),
        patch("tough_out.team_win_pct", return_value=0.560),
        patch("tough_out.team_l10_win_pct", return_value=0.600),
    ):
        boosted = break_push_bonus(
            0.50,
            team_id=147,
            game_date=date(2026, 7, 10),
            season=2026,
        )
    assert boosted == 0.50 * BREAK_PUSH_BONUS


def test_look_ahead_trap_against_bottom_feeder() -> None:
    top = frozenset({147})
    bottom = frozenset({136})
    next_top = frozenset({119})
    with (
        patch("tough_out.win_pct_rank_sets", return_value=(top, bottom, next_top)),
        patch("tough_out.next_series_opponent", return_value=119),
        patch("tough_out.are_division_rivals", return_value=False),
        patch("tough_out.team_l14_wrc_plus_as_of", return_value=105.0),
    ):
        scalar, tag = look_ahead_trap_scalar(
            team_id=147,
            opponent_id=136,
            game_date=date(2026, 6, 15),
            season=2026,
        )
    assert scalar == LOOK_AHEAD_TRAP_PENALTY
    assert tag == "look_ahead_trap"


def test_look_ahead_trap_via_division_rival_next() -> None:
    top = frozenset({147})
    bottom = frozenset({136})
    next_top = frozenset({119})
    with (
        patch("tough_out.win_pct_rank_sets", return_value=(top, bottom, next_top)),
        patch("tough_out.next_series_opponent", return_value=110),
        patch("tough_out.are_division_rivals", return_value=True),
        patch("tough_out.team_l14_wrc_plus_as_of", return_value=100.0),
    ):
        scalar, tag = look_ahead_trap_scalar(
            team_id=147,
            opponent_id=136,
            game_date=date(2026, 6, 15),
            season=2026,
        )
    assert scalar == LOOK_AHEAD_TRAP_PENALTY
    assert tag == "look_ahead_trap"


def test_look_ahead_trap_skips_without_tough_next() -> None:
    top = frozenset({147})
    bottom = frozenset({136})
    next_top = frozenset({119})
    with (
        patch("tough_out.win_pct_rank_sets", return_value=(top, bottom, next_top)),
        patch("tough_out.next_series_opponent", return_value=140),
        patch("tough_out.are_division_rivals", return_value=False),
    ):
        assert look_ahead_trap_scalar(
            team_id=147,
            opponent_id=136,
            game_date=date(2026, 6, 15),
            season=2026,
        ) == (1.0, None)


def test_firepower_veto_nullifies_look_ahead_trap() -> None:
    top = frozenset({147})
    bottom = frozenset({136})
    next_top = frozenset({119})
    with (
        patch("tough_out.win_pct_rank_sets", return_value=(top, bottom, next_top)),
        patch("tough_out.next_series_opponent", return_value=119),
        patch("tough_out.are_division_rivals", return_value=False),
        # Opponent L14 firepower > 115 -> Tampa Bay fix
        patch("tough_out.team_l14_wrc_plus_as_of", return_value=120.0),
    ):
        scalar, tag = look_ahead_trap_scalar(
            team_id=147,
            opponent_id=136,
            game_date=date(2026, 6, 15),
            season=2026,
        )
    assert scalar == 1.0
    assert tag == "elite_firepower_veto"


def test_apply_system_guardrails_oakland_atlanta_tampa() -> None:
    context = {
        133: {
            "missing_top_hr_count": 2,
            "l10_win_pct": 0.400,
            "l14_wrcplus": 95.0,
            "break_push_bonus_active": True,
            "look_ahead_trap_active": True,
        },
        139: {
            "missing_top_hr_count": 0,
            "l10_win_pct": 0.600,
            "l14_wrcplus": 122.0,
            "break_push_bonus_active": True,
            "look_ahead_trap_active": True,
        },
    }
    projection = apply_system_guardrails(133, 139, 5.0, context)
    expected = 5.0 * MISSING_STAR_BAT_PENALTY * MISSING_STAR_BAT_PENALTY
    assert abs(projection - expected) < 1e-9
    assert context[133]["break_push_bonus_active"] is False
    assert context[133]["look_ahead_trap_active"] is False


def test_slump_veto_nullifies_break_push() -> None:
    with (
        patch("tough_out.is_pre_all_star_window", return_value=True),
        patch("tough_out.is_tough_out", return_value=True),
        patch("tough_out.team_win_pct", return_value=0.560),
        patch("tough_out.team_l10_win_pct", return_value=0.400),
    ):
        boosted = break_push_bonus(
            0.50,
            team_id=147,
            game_date=date(2026, 7, 10),
            season=2026,
        )
    assert boosted == 0.50


def test_missing_star_bat_from_starting_lineup() -> None:
    lineup = [
        LineupBatter(player_id=i, lineup_slot=slot)
        for slot, i in enumerate(range(1, 10), start=1)
    ]
    with patch("tough_out.team_top_hr_batter_ids", return_value=(99, 2, 3)):
        scalar, tags = missing_star_bat_lineup_scalar(
            team_id=147,
            season=2026,
            lineup=lineup,
            label="home_offense",
        )
    assert scalar == MISSING_STAR_BAT_PENALTY
    assert any("missing_star_bat:99" in tag for tag in tags)


def test_missing_star_bat_stacks_per_absent_hr_leader() -> None:
    lineup = [
        LineupBatter(player_id=i, lineup_slot=slot)
        for slot, i in enumerate(range(10, 19), start=1)
    ]
    with (
        patch("tough_out.team_top_hr_batter_ids", return_value=(1, 2, 3)),
        patch("tough_out.is_pre_all_star_window", return_value=False),
        patch("tough_out.team_l10_win_pct", return_value=0.600),
        patch("tough_out.team_l14_wrc_plus_as_of", return_value=100.0),
    ):
        runs, tags = apply_tough_out_run_scalars(
            5.0,
            offense_team_id=147,
            pitcher_id=None,
            pitcher_era=None,
            is_home_offense=True,
            game_date=date(2026, 6, 1),
            season=2026,
            label="home_offense",
            lineup=lineup,
        )
    expected = 5.0 * MISSING_STAR_BAT_PENALTY * MISSING_STAR_BAT_PENALTY * MISSING_STAR_BAT_PENALTY
    assert abs(runs - expected) < 1e-9
    assert any("missing_star_bat_scalar" in tag for tag in tags)
