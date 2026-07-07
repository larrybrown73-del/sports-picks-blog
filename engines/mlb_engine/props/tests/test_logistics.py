from __future__ import annotations

from datetime import date

import pandas as pd

from baseball_props.analysis.logistics import (
    ScheduleGameSlice,
    build_travel_rest_matrix,
    compute_travel_rest_multiplier,
    parse_home_plate_official,
    parse_schedule_game,
)


def test_parse_schedule_game_skips_malformed() -> None:
    assert parse_schedule_game({}) is None
    assert parse_schedule_game({"gamePk": 0}) is None
    assert parse_schedule_game({"gamePk": 123, "teams": {}}) is None


def test_parse_schedule_game_valid() -> None:
    raw = {
        "gamePk": 777001,
        "officialDate": "2026-07-03",
        "status": {"abstractGameState": "Final", "codedGameState": "F"},
        "teams": {
            "home": {"team": {"id": 147, "abbreviation": "NYY"}},
            "away": {"team": {"id": 111, "abbreviation": "BOS"}},
        },
        "venue": {"id": 3313, "location": {"defaultCoordinates": {"latitude": 40.83, "longitude": -73.93}}},
    }
    parsed = parse_schedule_game(raw)
    assert parsed is not None
    assert parsed.game_pk == 777001
    assert parsed.home_abbrev == "NYY"
    assert parsed.is_final is True


def test_parse_home_plate_official() -> None:
    payload = {
        "officials": [
            {"officialType": "First Base", "official": {"fullName": "Ump A"}},
            {"officialType": "Home Plate", "official": {"fullName": "Marvin Hudson"}},
        ]
    }
    assert parse_home_plate_official(payload) == "Marvin Hudson"
    assert parse_home_plate_official({}) is None


def test_compute_travel_rest_multiplier_neutral_default() -> None:
    mult, tag = compute_travel_rest_multiplier(
        days_rest=1,
        is_back_to_back=False,
        travel_miles=200.0,
        travel_zone_delta=0,
    )
    assert mult == 1.0
    assert tag == "Neutral"


def test_compute_travel_rest_multiplier_b2b() -> None:
    mult, tag = compute_travel_rest_multiplier(
        days_rest=0,
        is_back_to_back=True,
        travel_miles=0.0,
        travel_zone_delta=0,
    )
    assert mult < 1.0
    assert "B2B" in tag


def test_build_travel_rest_matrix_empty_slate() -> None:
    result = build_travel_rest_matrix(pd.DataFrame())
    assert result.empty


def test_build_travel_rest_matrix_no_prior_games_neutral(monkeypatch) -> None:
    monkeypatch.setattr(
        "baseball_props.analysis.logistics.fetch_schedule_window",
        lambda _start, _end: [],
    )
    games = pd.DataFrame(
        [
            {
                "game_id": "777001",
                "game_date": "2026-07-03",
                "home_team_id": "NYY",
                "away_team_id": "BOS",
                "mlb_game_pk": 777001,
            }
        ]
    )
    table = build_travel_rest_matrix(games, date(2026, 7, 3))
    assert len(table) == 2
    assert (table["travel_rest_multiplier"] == 1.0).all()
    assert (table["travel_rest_tag"] == "Neutral").all()


def test_build_travel_rest_matrix_b2b_detection(monkeypatch) -> None:
    prior = ScheduleGameSlice(
        game_pk=777000,
        game_date=date(2026, 7, 3),
        home_abbrev="NYY",
        away_abbrev="BOS",
        home_team_id=147,
        away_team_id=111,
        venue_id="3313",
        venue_lat=40.8296,
        venue_lon=-73.9262,
        is_final=True,
    )
    current = ScheduleGameSlice(
        game_pk=777001,
        game_date=date(2026, 7, 3),
        home_abbrev="NYY",
        away_abbrev="TB",
        home_team_id=147,
        away_team_id=139,
        venue_id="3313",
        venue_lat=40.8296,
        venue_lon=-73.9262,
        is_final=False,
    )

    monkeypatch.setattr(
        "baseball_props.analysis.logistics.fetch_schedule_window",
        lambda _start, _end: [prior, current],
    )
    games = pd.DataFrame(
        [
            {
                "game_id": "777001",
                "game_date": "2026-07-03",
                "home_team_id": "NYY",
                "away_team_id": "TB",
                "mlb_game_pk": 777001,
            }
        ]
    )
    table = build_travel_rest_matrix(games, date(2026, 7, 3))
    nyy = table[table["team_id"] == "NYY"].iloc[0]
    assert bool(nyy["is_back_to_back"]) is True
    assert nyy["travel_rest_multiplier"] < 1.0
