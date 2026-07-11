from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from baseball_props.analysis.hitter_discipline import BatterDisciplineProfile

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
    HITS_MAX_ODDS_CAP,
    HITS_MIN_PROBABILITY_FLOOR,
    HITS_PROP_PRIMARY_LINE,
    HITS_PROP_TARGET_LINES,
    HITS_WEATHER_BONUS_MULTIPLIER,
    PLAYER_HIT_STREAK_BONUS,
    RECENT_CONTACT_BONUS,
)


@pytest.fixture(autouse=True)
def _neutral_batter_discipline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "baseball_props.analysis.guardrails.fetch_batter_discipline_profile",
        lambda player_id, season: BatterDisciplineProfile(player_id, k_pct=20.0, bb_pct=8.0),
    )
    monkeypatch.setattr(
        "baseball_props.analysis.guardrails.apply_pitcher_hitter_matchup",
        lambda adjusted_proj, **kwargs: adjusted_proj,
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


def test_bottom_order_penalty_for_slots_eight_nine() -> None:
    ctx = _base_context(lineup_slot=9)
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
    assert result.adjustments.get("bottom_order_penalty") == HITS_LINEUP_SLOT_PENALTY
    assert result.adjustments.get("adjusted_proj_hits") == pytest.approx(1.8 * HITS_LINEUP_SLOT_PENALTY)
    assert any("bottom order" in warning for warning in result.warnings)


def test_middle_order_slot_is_neutral() -> None:
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
    assert "bottom_order_penalty" not in result.adjustments
    assert "premium_slot_bonus" not in result.adjustments


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


def test_hits_probability_floor_drops_longshot() -> None:
    ctx = _base_context()
    with patch(
        "baseball_props.analysis.guardrails.apply_hits_momentum_multipliers",
        side_effect=lambda proj, *_args, **_kwargs: proj,
    ), patch(
        "baseball_props.analysis.guardrails._resolve_bullpen_fatigued",
        return_value=(False, 0.0),
    ), patch(
        "baseball_props.analysis.guardrails.prob_over_continuous",
        return_value=0.50,
    ), patch(
        "baseball_props.analysis.guardrails.best_side_edge",
        return_value=("Over", 0.50, 5.0),
    ):
        result = evaluate_hits_prop(
            "592450",
            "P101",
            ctx,
            proj_hits=1.5,
            market_line=1.5,
            over_odds=-110,
            under_odds=-110,
            contact_profile={"k_pct": 0.22, "contact_pct": 0.70, "babip": 0.25, "pa": 30.0},
        )
    assert result.verdict == "Pass"
    assert any("55% floor" in warning for warning in result.warnings)


def test_hits_odds_cap_drops_plus_money_ladder() -> None:
    ctx = _base_context()
    with patch(
        "baseball_props.analysis.guardrails.apply_hits_momentum_multipliers",
        side_effect=lambda proj, *_args, **_kwargs: proj,
    ), patch(
        "baseball_props.analysis.guardrails._resolve_bullpen_fatigued",
        return_value=(False, 0.0),
    ), patch(
        "baseball_props.analysis.guardrails.prob_over_continuous",
        return_value=0.65,
    ), patch(
        "baseball_props.analysis.guardrails.best_side_edge",
        return_value=("Over", 0.65, 8.0),
    ):
        result = evaluate_hits_prop(
            "592450",
            "P101",
            ctx,
            proj_hits=2.0,
            market_line=1.5,
            over_odds=HITS_MAX_ODDS_CAP + 25,
            under_odds=-180,
            contact_profile={"k_pct": 0.22, "contact_pct": 0.70, "babip": 0.25, "pa": 30.0},
        )
    assert result.verdict == "Pass"
    assert any("exceed +130" in warning for warning in result.warnings)


def test_hit_streak_and_recent_contact_momentum() -> None:
    ctx = _base_context()
    with patch(
        "baseball_props.data.statcast_feed.consecutive_hit_games",
        return_value=4,
    ), patch(
        "baseball_props.data.statcast_feed.compute_contact_profile",
        return_value={"contact_pct": 0.84, "k_pct": 0.10, "babip": 0.31, "pa": 12.0},
    ):
        result = evaluate_hits_prop(
            "592450",
            "P101",
            ctx,
            proj_hits=1.6,
            market_line=1.5,
            over_odds=-110,
            under_odds=-110,
            contact_profile={"k_pct": 0.12, "contact_pct": 0.85, "babip": 0.32, "pa": 30.0},
        )
    assert result.adjustments.get("hit_streak_bonus") == PLAYER_HIT_STREAK_BONUS
    assert result.adjustments.get("recent_contact_bonus") == RECENT_CONTACT_BONUS
    assert result.adjusted_prob_over is not None
    assert result.adjusted_prob_over >= HITS_MIN_PROBABILITY_FLOOR


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
