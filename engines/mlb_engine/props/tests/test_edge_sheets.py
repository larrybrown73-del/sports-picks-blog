import math

import pandas as pd

from baseball_props.analysis.conviction import compute_conviction_plays
from baseball_props.analysis.edge_sheets import (
    PASS_NO_DATA,
    aggregate_top_conviction,
    american_to_implied,
    best_side_edge,
    build_batter_tb_edge_sheet,
    build_pitcher_outs_edge_sheet,
    prob_over_continuous,
)
from baseball_props.data.odds_props import consolidated_prop_quotes


def test_american_to_implied_minus_110() -> None:
    assert math.isclose(american_to_implied(-110), 110 / 210, rel_tol=1e-6)


def test_batter_sheet_recommends_over_with_positive_edge() -> None:
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
                "player_name": "Aaron Judge",
                "team_id": "NYY",
                "lineup_slot": 3,
                "proj_hits": 1.9,
            }
        ]
    )
    sheet = build_batter_tb_edge_sheet(projected, prop_lines)

    assert len(sheet) == 1
    row = sheet.iloc[0]
    assert row["market_line"] == 1.5
    assert row["recommendation"] == "Over"
    assert row["edge_pct"] is not None
    assert row["edge_pct"] > 0
    assert row["probability_pct"] > 50


def test_pitcher_sheet_recommends_under() -> None:
    game_id = "abc123event456789012345678901234"
    prop_lines = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "player_name": "Gerrit Cole",
                "market": "pitcher_outs",
                "side": "Over",
                "line": 17.5,
                "odds": -110,
                "bookmaker": "draftkings",
            },
            {
                "game_id": game_id,
                "player_name": "Gerrit Cole",
                "market": "pitcher_outs",
                "side": "Under",
                "line": 17.5,
                "odds": -110,
                "bookmaker": "draftkings",
            },
        ]
    )
    pitcher_outs = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "team_id": "NYY",
                "pitcher_name": "Gerrit Cole",
                "proj_outs": 16.0,
                "proj_pitch_count": 92.0,
            }
        ]
    )
    sheet = build_pitcher_outs_edge_sheet(pitcher_outs, prop_lines)

    assert len(sheet) == 1
    row = sheet.iloc[0]
    assert row["recommendation"] == "Under"
    assert row["edge_pct"] is not None
    assert row["edge_pct"] > 0


def test_aggregator_sorts_by_absolute_edge_pct() -> None:
    batter_sheet = pd.DataFrame(
        [
            {
                "player_name": "Player A",
                "proj_hits": 2.0,
                "market_line": 1.5,
                "edge_pct": 2.0,
                "probability_pct": 55.0,
                "recommendation": "Over",
                "market": "batter_hits",
            }
        ]
    )
    pitcher_sheet = pd.DataFrame(
        [
            {
                "pitcher_name": "Pitcher B",
                "proj_outs": 18.0,
                "market_line": 16.5,
                "edge_pct": -8.5,
                "recommendation": "Over",
                "market": "pitcher_outs",
            }
        ]
    )
    result = aggregate_top_conviction(batter_sheet, pitcher_sheet, top_n=2)

    assert len(result) == 2
    assert result.iloc[0]["player_name"] == "Pitcher B"
    assert result.iloc[0]["edge_pct"] == -8.5
    assert result.iloc[1]["player_name"] == "Player A"


def test_batter_sheet_filters_implausible_tb_line() -> None:
    game_id = "abc123event456789012345678901234"
    prop_lines = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "player_name": "Aaron Judge",
                "market": "batter_hits",
                "side": "Over",
                "line": 4.5,
                "odds": -110,
                "bookmaker": "draftkings",
            },
            {
                "game_id": game_id,
                "player_name": "Aaron Judge",
                "market": "batter_hits",
                "side": "Under",
                "line": 4.5,
                "odds": -110,
                "bookmaker": "draftkings",
            },
        ]
    )
    projected = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "player_name": "Aaron Judge",
                "team_id": "NYY",
                "lineup_slot": 3,
                "proj_hits": 2.0,
            }
        ]
    )
    sheet = build_batter_tb_edge_sheet(projected, prop_lines)
    assert sheet.iloc[0]["recommendation"] == PASS_NO_DATA
    assert pd.isna(sheet.iloc[0]["market_line"])
    assert sheet.iloc[0]["edge_pct"] is None or pd.isna(sheet.iloc[0]["edge_pct"])


def test_consolidated_prop_quotes_median_odds() -> None:
    props = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "player_name": "Aaron Judge",
                "market": "batter_hits",
                "side": "Over",
                "line": 1.5,
                "odds": -120,
                "bookmaker": "draftkings",
            },
            {
                "game_id": "g1",
                "player_name": "Aaron Judge",
                "market": "batter_hits",
                "side": "Under",
                "line": 1.5,
                "odds": -100,
                "bookmaker": "draftkings",
            },
            {
                "game_id": "g1",
                "player_name": "Aaron Judge",
                "market": "batter_hits",
                "side": "Over",
                "line": 1.5,
                "odds": -100,
                "bookmaker": "fanduel",
            },
            {
                "game_id": "g1",
                "player_name": "Aaron Judge",
                "market": "batter_hits",
                "side": "Under",
                "line": 1.5,
                "odds": -120,
                "bookmaker": "fanduel",
            },
        ]
    )
    quotes = consolidated_prop_quotes(props)
    assert len(quotes) == 1
    assert quotes.iloc[0]["market_line"] == 1.5
    assert quotes.iloc[0]["over_odds"] == -110
    assert quotes.iloc[0]["under_odds"] == -110


def test_best_side_edge_picks_higher_positive_edge() -> None:
    p_over = prob_over_continuous(2.1, 0.55, 1.5)
    rec, prob, edge = best_side_edge(p_over, -110, -110)
    assert rec == "Over"
    assert edge > 0
    assert prob == p_over


def test_batter_sheet_pass_no_data_when_projection_missing() -> None:
    game_id = "abc123event456789012345678901234"
    projected = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "player_name": "Aaron Judge",
                "team_id": "NYY",
                "lineup_slot": 3,
                "proj_hits": float("nan"),
            }
        ]
    )
    sheet = build_batter_tb_edge_sheet(projected, pd.DataFrame())
    assert len(sheet) == 1
    row = sheet.iloc[0]
    assert row["recommendation"] == PASS_NO_DATA
    assert row["verdict"] == "Pass"
    assert row["edge_pct"] is None or pd.isna(row["edge_pct"])


def test_batter_sheet_pass_no_data_when_odds_missing() -> None:
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
        ]
    )
    projected = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "player_name": "Aaron Judge",
                "team_id": "NYY",
                "lineup_slot": 3,
                "proj_hits": 2.0,
            }
        ]
    )
    sheet = build_batter_tb_edge_sheet(projected, prop_lines)
    row = sheet.iloc[0]
    assert row["recommendation"] in {PASS_NO_DATA, "Pass", "Pass (insufficient edge)"}
    assert row["market_line"] == 1.5
    assert row["edge_pct"] is None or pd.isna(row["edge_pct"])
    assert not sheet["edge_pct"].apply(lambda x: isinstance(x, float) and math.isnan(x)).any()


def test_best_side_edge_returns_none_edge_when_odds_missing() -> None:
    p_over = prob_over_continuous(2.1, 0.55, 1.5)
    rec, prob, edge = best_side_edge(p_over, None, -110)
    assert rec in {"Over", "Under"}
    assert prob is not None
    assert edge is None


def test_aggregate_top_conviction_excludes_pass_no_data() -> None:
    batter_sheet = pd.DataFrame(
        [
            {
                "player_name": "No Data Player",
                "proj_hits": 2.0,
                "market_line": None,
                "edge_pct": None,
                "probability_pct": None,
                "recommendation": PASS_NO_DATA,
                "verdict": "Pass",
                "market": "batter_hits",
            },
            {
                "player_name": "Playable Player",
                "proj_hits": 2.0,
                "market_line": 1.5,
                "edge_pct": 3.5,
                "probability_pct": 58.0,
                "recommendation": "Over",
                "verdict": "Play",
                "market": "batter_hits",
            },
        ]
    )
    result = aggregate_top_conviction(batter_sheet, pd.DataFrame(), top_n=5)
    assert len(result) == 1
    assert result.iloc[0]["player_name"] == "Playable Player"


def test_conviction_delegates_to_edge_sheets() -> None:
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
                "player_name": "Aaron Judge",
                "team_id": "NYY",
                "proj_hits": 1.9,
            }
        ]
    )
    pitcher_outs = pd.DataFrame(columns=["game_id", "pitcher_name", "proj_outs"])

    result = compute_conviction_plays(projected, pitcher_outs, prop_lines, top_n=3)

    assert len(result) == 1
    assert "edge_pct" in result.columns
    assert result.iloc[0]["recommendation"] == "Over"
