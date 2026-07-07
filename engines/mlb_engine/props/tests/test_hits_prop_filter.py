from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd

from baseball_props.analysis.edge_sheets import (
    PASS_NO_DATA,
    aggregate_top_conviction,
    build_batter_hits_edge_sheet,
)
from baseball_props.analysis.guardrails import GameContext, evaluate_hits_prop
from baseball_props.analysis.parlay_builder import build_diversified_tickets, legs_from_batter_sheet
from baseball_props.config import (
    HITS_BULLPEN_FATIGUE_BONUS,
    HITS_CONTACT_BONUS_MULTIPLIER,
    HITS_CONTACT_K_PCT_MAX,
    HITS_LINEUP_SLOT_PENALTY,
    HITS_PROP_PRIMARY_LINE,
    HITS_PROP_TARGET_LINES,
    HITS_WEATHER_BONUS_MULTIPLIER,
)


def _base_context(**overrides: object) -> GameContext:
    defaults = {
        "game_id": "G001",
        "player_id": "592450",
        "player_name": "Aaron Judge",
        "opponent_pitcher_id": "P101",
        "opponent_team_id": "147",
        "batting_team_id": "147",
        "lineup_slot": 3,
        "venue_id": 3313,
        "park_tb_factor": 1.0,
        "temp_f": 72.0,
        "wind_mph": 5.0,
        "wind_dir": "calm",
        "game_date": date(2026, 7, 2),
        "opp_bullpen_status": "Moderate",
    }
    defaults.update(overrides)
    return GameContext(**defaults)  # type: ignore[arg-type]


def test_hits_target_lines_include_half_and_one_half() -> None:
    assert HITS_PROP_TARGET_LINES == (0.5, 1.5)
    assert HITS_PROP_PRIMARY_LINE == 1.5


def test_lineup_slot_penalty_for_bottom_order() -> None:
    ctx = _base_context(lineup_slot=7)
    result = evaluate_hits_prop(
        "592450",
        "P101",
        ctx,
        proj_hits=1.8,
        market_line=1.5,
        over_odds=-110,
        under_odds=-110,
        contact_profile={"k_pct": 0.22, "contact_pct": 0.80, "babip": 0.29, "pa": 20.0},
    )
    assert result.verdict == "Play"
    assert any("Lineup slot 7" in warning for warning in result.warnings)
    assert result.adjustments.get("lineup_penalty") == HITS_LINEUP_SLOT_PENALTY


def test_contact_hitter_receives_bonus_not_pass() -> None:
    ctx = _base_context()
    contact_profile = {
        "k_pct": 0.12,
        "contact_pct": 0.85,
        "babip": 0.32,
        "pa": 20.0,
    }
    result = evaluate_hits_prop(
        "592450",
        "P101",
        ctx,
        proj_hits=1.8,
        market_line=1.5,
        over_odds=-110,
        under_odds=-110,
        contact_profile=contact_profile,
    )
    assert contact_profile["k_pct"] < HITS_CONTACT_K_PCT_MAX
    assert result.adjustments.get("contact_bonus") == HITS_CONTACT_BONUS_MULTIPLIER
    assert result.verdict == "Play"


def test_environment_bonus_on_hot_windy_park() -> None:
    ctx = _base_context(temp_f=82.0, wind_dir="out_to_cf", park_tb_factor=1.08)
    result = evaluate_hits_prop(
        "592450",
        "P101",
        ctx,
        proj_hits=1.8,
        market_line=1.5,
        over_odds=-110,
        under_odds=-110,
        contact_profile={"k_pct": 0.22, "contact_pct": 0.80, "babip": 0.29, "pa": 30.0},
    )
    assert result.verdict == "Play"
    assert result.adjustments.get("park_hit_bonus") == HITS_WEATHER_BONUS_MULTIPLIER


def test_bullpen_fatigue_bonus_from_status() -> None:
    ctx = _base_context(opp_bullpen_status="Fatigued")
    with patch(
        "baseball_props.analysis.guardrails._resolve_bullpen_fatigued",
        return_value=(True, HITS_BULLPEN_FATIGUE_BONUS),
    ):
        result = evaluate_hits_prop(
            "592450",
            "P101",
            ctx,
            proj_hits=1.8,
            market_line=1.5,
            over_odds=-110,
            under_odds=-110,
            contact_profile={"k_pct": 0.22, "contact_pct": 0.80, "babip": 0.29, "pa": 30.0},
        )
    assert result.adjustments.get("bullpen_bonus") == HITS_BULLPEN_FATIGUE_BONUS


def test_edge_sheet_uses_hits_market() -> None:
    game_id = "abc123event456789012345678901234"
    prop_lines = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "player_name": "Aaron Judge",
                "market": "batter_hits",
                "side": "Over",
                "line": 1.5,
                "odds": -110,
                "bookmaker": "draftkings",
            },
            {
                "game_id": game_id,
                "player_name": "Aaron Judge",
                "market": "batter_hits",
                "side": "Under",
                "line": 1.5,
                "odds": -110,
                "bookmaker": "draftkings",
            },
        ]
    )
    projected = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "player_id": "592450",
                "player_name": "Aaron Judge",
                "team_id": "NYY",
                "lineup_slot": 3,
                "opp_sp_id": "P101",
                "opp_team_id": "BOS",
                "park_factor_runs": 1.0,
                "temp_f": 72.0,
                "wind_mph": 5.0,
                "wind_dir": "calm",
                "game_date": "2026-07-02",
                "proj_hits": 1.9,
            }
        ]
    )
    with patch(
        "baseball_props.analysis.guardrails.compute_contact_profile",
        return_value={"k_pct": 0.22, "contact_pct": 0.80, "babip": 0.29, "pa": 20.0},
    ):
        sheet = build_batter_hits_edge_sheet(projected, prop_lines)
    assert sheet.iloc[0]["market"] == "batter_hits"
    assert sheet.iloc[0]["market_line"] == 1.5
    assert sheet.iloc[0]["proj_hits"] == 1.9


def test_evaluate_hits_prop_pass_no_data_for_nan_projection() -> None:
    ctx = _base_context()
    result = evaluate_hits_prop(
        "592450",
        "P101",
        ctx,
        proj_hits=float("nan"),
        market_line=1.5,
        over_odds=-110,
        under_odds=-110,
    )
    assert result.verdict == "Pass"
    assert result.recommendation == PASS_NO_DATA
    assert result.edge_pct is None
    assert result.adjusted_prob_over is None


def test_parlay_builder_excludes_pass_no_data_rows() -> None:
    sheet = pd.DataFrame(
        [
            {
                "player_id": "P0",
                "player_name": "No Data",
                "game_id": "G0",
                "market": "batter_hits",
                "market_line": 1.5,
                "recommendation": PASS_NO_DATA,
                "edge_pct": None,
                "verdict": "Pass",
            },
            {
                "player_id": "P1",
                "player_name": "Player One",
                "game_id": "G1",
                "market": "batter_hits",
                "market_line": 1.5,
                "recommendation": "Over",
                "edge_pct": 10.0,
                "verdict": "Play",
            },
        ]
    )
    legs = legs_from_batter_sheet(sheet)
    assert len(legs) == 1
    assert legs[0].player_id == "P1"


def test_parlay_diversification_max_two_exposure() -> None:
    sheet = pd.DataFrame(
        [
            {
                "player_id": "P1",
                "player_name": "Player One",
                "game_id": "G1",
                "market": "batter_hits",
                "market_line": 1.5,
                "recommendation": "Over",
                "edge_pct": 10.0,
                "verdict": "Play",
            },
            {
                "player_id": "P2",
                "player_name": "Player Two",
                "game_id": "G2",
                "market": "batter_hits",
                "market_line": 1.5,
                "recommendation": "Over",
                "edge_pct": 9.0,
                "verdict": "Play",
            },
            {
                "player_id": "P3",
                "player_name": "Player Three",
                "game_id": "G3",
                "market": "batter_hits",
                "market_line": 1.5,
                "recommendation": "Over",
                "edge_pct": 8.0,
                "verdict": "Play",
            },
            {
                "player_id": "P4",
                "player_name": "Player Four",
                "game_id": "G4",
                "market": "batter_hits",
                "market_line": 1.5,
                "recommendation": "Over",
                "edge_pct": 7.0,
                "verdict": "Play",
            },
        ]
    )
    tickets = build_diversified_tickets(
        sheet, ticket_count=5, legs_per_ticket=1, max_player_exposure=2
    )
    p1_count = sum(
        1 for ticket in tickets for leg in ticket.legs if leg.player_id == "P1"
    )
    assert p1_count <= 2


def test_compute_contact_profile_from_statcast_frame() -> None:
    from baseball_props.data.statcast_feed import compute_contact_profile

    sc = pd.DataFrame(
        {
            "game_date": ["2026-06-01", "2026-06-01", "2026-06-02", "2026-06-02"],
            "events": ["single", "home_run", "double", "strikeout"],
            "description": ["hit_into_play", "hit_into_play", "hit_into_play", "swinging_strike"],
        }
    )
    profile = compute_contact_profile("592450", games=2, statcast_frame=sc)
    assert profile is not None
    assert profile["k_pct"] == 0.25
    assert profile["babip"] == 1.0
