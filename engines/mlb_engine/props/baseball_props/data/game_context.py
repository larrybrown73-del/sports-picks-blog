from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from baseball_props.analysis.edge_row_builder import lookup_umpire_modifiers
from baseball_props.analysis.logistics import (
    build_travel_rest_matrix,
    fetch_game_officials_payload,
    parse_home_plate_official,
)
from baseball_props.data.data_health import safe_feature_slice
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)


def build_travel_rest_table(
    slate_games: pd.DataFrame,
    slate_date: date | None = None,
) -> pd.DataFrame:
    """Travel/rest context per team-game via 48-hour schedule lookback."""
    return build_travel_rest_matrix(slate_games, slate_date)


def _neutral_umpire_row(game_id: str) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "umpire_id": "",
        "umpire_name": "TBD",
        "zone_size_modifier": 1.0,
        "run_environment_modifier": 1.0,
    }


def _umpire_row_for_game(game: pd.Series) -> dict[str, Any]:
    game_id = str(game["game_id"])
    game_pk = game.get("mlb_game_pk")
    try:
        pk = int(game_pk)
    except (TypeError, ValueError):
        return _neutral_umpire_row(game_id)

    if pk <= 0:
        return _neutral_umpire_row(game_id)

    def _fetch() -> dict[str, Any]:
        payload = fetch_game_officials_payload(pk)
        crew_name = parse_home_plate_official(payload)
        mods = lookup_umpire_modifiers(crew_name)
        umpire_id = ""
        if payload.get("officials"):
            for entry in payload["officials"]:
                if str(entry.get("officialType", "")).strip().lower() in {
                    "home plate",
                    "home_plate",
                    "homeplate",
                }:
                    official = entry.get("official") or {}
                    umpire_id = str(official.get("id", ""))
                    break
        return {
            "game_id": game_id,
            "umpire_id": umpire_id,
            "umpire_name": mods.umpire_name or "TBD",
            "zone_size_modifier": mods.zone_size_modifier,
            "run_environment_modifier": mods.run_environment_modifier,
        }

    return safe_feature_slice(
        f"umpire_game_{pk}",
        _fetch,
        default=_neutral_umpire_row(game_id),
    )


def build_umpire_table(slate_games: pd.DataFrame) -> pd.DataFrame:
    """Home-plate umpire context per game with strict 1.0 fallbacks."""
    if slate_games.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "umpire_id",
                "umpire_name",
                "zone_size_modifier",
                "run_environment_modifier",
            ]
        )

    rows: list[dict[str, Any]] = []
    for _, game in slate_games.iterrows():
        try:
            rows.append(_umpire_row_for_game(game))
        except Exception as exc:
            logger.debug("Umpire row failed for game %s: %s", game.get("game_id"), exc)
            rows.append(_neutral_umpire_row(str(game["game_id"])))
    return pd.DataFrame(rows)


def build_lineup_absence_penalties(
    lineups: pd.DataFrame,
    injury_lookup: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Placeholder lineup absence penalties keyed by game/team."""
    del injury_lookup
    if lineups.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "team_id",
                "key_starter_absent",
                "offensive_penalty",
                "defensive_penalty",
            ]
        )

    unique = lineups[["game_id", "team_id"]].drop_duplicates()
    rows: list[dict[str, Any]] = []
    for _, row in unique.iterrows():
        rows.append(
            {
                "game_id": str(row["game_id"]),
                "team_id": str(row["team_id"]),
                "key_starter_absent": False,
                "offensive_penalty": 1.0,
                "defensive_penalty": 1.0,
            }
        )
    return pd.DataFrame(rows)
