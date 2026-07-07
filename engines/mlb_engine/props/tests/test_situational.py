from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from baseball_props.analysis.situational_adjustments import (
    apply_lineup_absence_penalty,
    apply_travel_rest_to_rates,
    apply_umpire_to_runs,
)
from baseball_props.data.game_context import (
    build_lineup_absence_penalties,
    build_travel_rest_table,
    build_umpire_table,
)


def test_travel_rest_empty_schedule_neutral_multiplier(monkeypatch) -> None:
    monkeypatch.setattr(
        "baseball_props.analysis.logistics.fetch_schedule_window",
        lambda _start, _end: [],
    )
    games = pd.DataFrame(
        [
            {
                "game_id": "G1",
                "game_date": "2026-07-03",
                "home_team_id": "NYY",
                "away_team_id": "BOS",
                "mlb_game_pk": 1,
            }
        ]
    )
    table = build_travel_rest_table(games, date(2026, 7, 3))
    assert not table.empty
    assert (table["travel_rest_multiplier"] == 1.0).all()
    assert (table["travel_rest_tag"] == "Neutral").all()


def test_umpire_table_defaults_when_fetch_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "baseball_props.data.game_context.fetch_game_officials_payload",
        lambda _pk: {},
    )
    games = pd.DataFrame(
        [
            {
                "game_id": "G1",
                "home_team_id": "NYY",
                "away_team_id": "BOS",
                "mlb_game_pk": 999,
            }
        ]
    )
    table = build_umpire_table(games)
    assert table.iloc[0]["run_environment_modifier"] == 1.0
    assert table.iloc[0]["umpire_name"] == "TBD"


def test_umpire_table_resolves_known_crew(monkeypatch) -> None:
    monkeypatch.setattr(
        "baseball_props.data.game_context.fetch_game_officials_payload",
        lambda _pk: {
            "officials": [
                {"officialType": "Home Plate", "official": {"id": 1, "fullName": "Marvin Hudson"}},
            ]
        },
    )
    games = pd.DataFrame(
        [
            {
                "game_id": "G1",
                "home_team_id": "NYY",
                "away_team_id": "BOS",
                "mlb_game_pk": 999,
            }
        ]
    )
    table = build_umpire_table(games)
    assert table.iloc[0]["umpire_name"] == "Marvin Hudson"
    assert table.iloc[0]["run_environment_modifier"] == pytest.approx(0.97)


def test_lineup_absence_stub_neutral_penalty() -> None:
    lineups = pd.DataFrame(
        [
            {"game_id": "G1", "team_id": "NYY", "player_id": "1", "lineup_slot": 1},
        ]
    )
    table = build_lineup_absence_penalties(lineups)
    assert table.iloc[0]["offensive_penalty"] == 1.0


def test_apply_travel_rest_multiplier_to_rates() -> None:
    df = pd.DataFrame([{"proj_woba": 0.32, "travel_rest_multiplier": 1.0}])
    out = apply_travel_rest_to_rates(df)
    assert out.iloc[0]["proj_woba"] == pytest.approx(0.32)


def test_apply_umpire_modifier_to_park_factor() -> None:
    df = pd.DataFrame([{"park_factor_runs": 1.05, "run_environment_modifier": 1.0}])
    out = apply_umpire_to_runs(df)
    assert out.iloc[0]["park_factor_runs"] == pytest.approx(1.05)


def test_apply_lineup_absence_penalty_to_projections() -> None:
    df = pd.DataFrame([{"proj_total_bases": 1.8, "offensive_penalty": 1.0}])
    out = apply_lineup_absence_penalty(df)
    assert out.iloc[0]["proj_total_bases"] == pytest.approx(1.8)
