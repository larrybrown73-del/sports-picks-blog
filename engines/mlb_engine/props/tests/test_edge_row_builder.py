import math

import pandas as pd

from baseball_props.analysis.edge_row_builder import (
    build_batter_edge_row,
    build_pitcher_edge_row,
)
from baseball_props.analysis.edge_sheets import PASS_NO_DATA
from baseball_props.config import HITS_PROP_PRIMARY_LINE, HITS_PROP_TARGET_LINES


def test_batter_row_no_quote_pass_no_data() -> None:
    row = pd.Series(
        {
            "game_id": "g1",
            "player_id": "p1",
            "player_name": "Test Player",
            "team_id": "NYY",
            "proj_hits": 1.2,
            "lineup_slot": 3,
        }
    )
    out = build_batter_edge_row(row, None, pd.DataFrame())
    assert out["recommendation"] == PASS_NO_DATA
    assert out["ev_per_unit"] is None
    assert out["confidence_tier"] is None
    assert out["kelly_fraction"] is None


def test_batter_row_no_odds_null_market_metrics() -> None:
    row = pd.Series(
        {
            "game_id": "g1",
            "player_id": "p1",
            "player_name": "Test Player",
            "team_id": "NYY",
            "proj_hits": 1.2,
            "lineup_slot": 3,
        }
    )
    quote = pd.Series(
        {
            "game_id": "g1",
            "player_name": "Test Player",
            "market": "batter_hits",
            "market_line": 1.5,
            "over_odds": None,
            "under_odds": None,
        }
    )
    out = build_batter_edge_row(row, quote, pd.DataFrame())
    assert out["market_line"] == 1.5
    assert out["recommendation"] in {PASS_NO_DATA, "Pass", "Pass (insufficient edge)"}
    assert out["edge_pct"] is None
    assert out["ev_per_unit"] is None


def test_batter_row_continuous_line_with_odds() -> None:
    row = pd.Series(
        {
            "game_id": "g1",
            "player_id": "p1",
            "player_name": "Test Player",
            "team_id": "NYY",
            "proj_hits": 1.4,
            "lineup_slot": 3,
        }
    )
    quote = pd.Series(
        {
            "game_id": "g1",
            "player_name": "Test Player",
            "market": "batter_hits",
            "market_line": 1.5,
            "over_odds": -110,
            "under_odds": -110,
        }
    )
    out = build_batter_edge_row(row, quote, pd.DataFrame())
    assert out["recommendation"] in {"Over", "Pass", PASS_NO_DATA, "Pass (insufficient edge)"}
    if out["recommendation"] == "Over":
        assert out["edge_pct"] is not None
        assert out["edge_pct"] > 0
        assert out["ev_per_unit"] is not None
        assert out["confidence_tier"] is not None


def test_pitcher_row_no_quote() -> None:
    row = pd.Series(
        {
            "game_id": "g1",
            "pitcher_name": "Ace Pitcher",
            "team_id": "NYY",
            "proj_outs": 17.0,
            "proj_pitch_count": 95.0,
        }
    )
    out = build_pitcher_edge_row(row, None)
    assert out["recommendation"] == PASS_NO_DATA
    assert out["kelly_fraction"] is None


def test_hits_target_lines_use_guardrail_path() -> None:
    """Over 0.5 / Over 1.5 hits lines use guardrails.evaluate_hits_prop."""
    assert HITS_PROP_PRIMARY_LINE in HITS_PROP_TARGET_LINES
    assert 0.5 in HITS_PROP_TARGET_LINES
