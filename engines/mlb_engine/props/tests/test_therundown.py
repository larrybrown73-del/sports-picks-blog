from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from baseball_props.data.ingest import _games_to_vegas_rows
from baseball_props.data.odds_props import _parse_prop_payload
from baseball_props.data.therundown import (
    OFF_BOARD_SENTINEL,
    adapt_rundown_event_props,
    adapt_rundown_vegas_games,
    discover_mlb_prop_market_ids,
    fetch_rundown_batched_props,
)

FIXTURE = Path(__file__).parent / "fixtures" / "therundown_mlb_events_sample.json"
SAMPLE_DATE = date(2026, 6, 28)


def _load_fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_discover_mlb_prop_market_ids_maps_names() -> None:
    markets_payload = {
        "3": [
            {"id": 501, "name": "Player Total Bases", "proposition": True},
            {"id": 502, "name": "Pitcher Outs Recorded", "proposition": True},
            {"id": 99, "name": "Team Total Bases", "proposition": True},
        ]
    }
    with patch(
        "baseball_props.data.therundown.read_odds_cache",
        return_value=None,
    ), patch(
        "baseball_props.data.therundown._fetch_rundown_json",
        return_value=markets_payload,
    ), patch("baseball_props.data.therundown.write_odds_cache"):
        discovered = discover_mlb_prop_market_ids(SAMPLE_DATE)

    assert discovered["batter_total_bases"] == 501
    assert discovered["pitcher_outs"] == 502
    assert "team" not in str(discovered)


def test_discover_mlb_prop_market_ids_falls_back_to_catalog() -> None:
    """Date-scoped discovery often lists only core markets (1/2/3)."""
    core_only_payload = {
        "3": [
            {"id": 1, "name": "Moneyline", "proposition": False},
            {"id": 2, "name": "Handicap", "proposition": False},
            {"id": 3, "name": "Totals", "proposition": False},
        ]
    }
    with patch(
        "baseball_props.data.therundown.read_odds_cache",
        return_value=None,
    ), patch(
        "baseball_props.data.therundown._fetch_rundown_json",
        return_value=core_only_payload,
    ), patch("baseball_props.data.therundown.write_odds_cache"):
        discovered = discover_mlb_prop_market_ids(SAMPLE_DATE)

    assert discovered["batter_total_bases"] == 11
    assert discovered["pitcher_outs"] == 973


def test_adapt_rundown_vegas_games_produces_parseable_totals() -> None:
    games = adapt_rundown_vegas_games(_load_fixture())
    assert len(games) == 1
    assert games[0]["id"] == "9876543210123456"

    rows = _games_to_vegas_rows(games, sportsbook_keys="draftkings,fanduel,betmgm")
    assert len(rows) == 1
    assert rows[0]["game_total"] == pytest.approx(8.5)
    assert rows[0]["home_implied_runs"] == pytest.approx(5.0)
    assert rows[0]["away_implied_runs"] == pytest.approx(3.5)


def test_adapt_rundown_event_props_maps_player_lines() -> None:
    event = _load_fixture()["events"][0]
    market_id_map = {"batter_total_bases": 501, "pitcher_outs": 502}
    adapted = adapt_rundown_event_props(event, market_id_map)
    parsed = _parse_prop_payload(
        adapted,
        "9876543210123456",
        {"batter_total_bases", "pitcher_outs"},
    )

    assert not parsed.empty
    judge = parsed[
        (parsed["player_name"] == "Aaron Judge") & (parsed["market"] == "batter_total_bases")
    ]
    assert len(judge) >= 2
    assert set(judge["side"]) == {"Over", "Under"}
    assert float(judge.loc[judge["side"] == "Over", "line"].iloc[0]) == pytest.approx(1.5)

    rodon_over = parsed[
        (parsed["player_name"] == "Carlos Rodon")
        & (parsed["market"] == "pitcher_outs")
        & (parsed["side"] == "Over")
    ]
    assert len(rodon_over) == 1
    assert float(rodon_over.iloc[0]["line"]) == pytest.approx(15.5)

    off_board = parsed[
        (parsed["player_name"] == "Carlos Rodon")
        & (parsed["bookmaker"] == "fanduel")
    ]
    assert off_board.empty


def test_fetch_rundown_batched_props_filters_events() -> None:
    fixture = _load_fixture()
    market_map = {"batter_total_bases": 501, "pitcher_outs": 502}

    with patch(
        "baseball_props.data.therundown.discover_mlb_prop_market_ids",
        return_value=market_map,
    ), patch(
        "baseball_props.data.therundown.fetch_rundown_mlb_events",
        return_value=fixture,
    ):
        df = fetch_rundown_batched_props(
            ["9876543210123456"],
            markets="batter_total_bases,pitcher_outs",
        )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert set(df["market"]) <= {"batter_total_bases", "pitcher_outs"}
    assert OFF_BOARD_SENTINEL not in df["line"].astype(float).tolist()
