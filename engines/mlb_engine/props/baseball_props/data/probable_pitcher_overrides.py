from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProbablePitcherOverride:
    """Manual starter correction when MLB schedule probables lag announcements."""

    pitcher_id: str
    pitcher_name: str
    side: str  # "home" or "away"
    hand: str = "R"
    mlb_game_pk: int | None = None
    away_team: str | None = None
    home_team: str | None = None


# Keyed by slate date (ISO) for easy day-of overrides.
PROBABLE_PITCHER_OVERRIDES: dict[str, list[ProbablePitcherOverride]] = {
    "2026-07-04": [
        ProbablePitcherOverride(
            mlb_game_pk=822716,
            away_team="PIT",
            home_team="WSH",
            side="home",
            pitcher_id="641793",
            pitcher_name="Zack Littell",
            hand="R",
        ),
    ],
}


def _matches_game(row: pd.Series, override: ProbablePitcherOverride) -> bool:
    if override.mlb_game_pk is not None:
        try:
            if int(row.get("mlb_game_pk", 0)) == override.mlb_game_pk:
                return True
        except (TypeError, ValueError):
            pass
    away = str(row.get("away_team_id", "")).upper()
    home = str(row.get("home_team_id", "")).upper()
    if override.away_team and override.home_team:
        return away == override.away_team.upper() and home == override.home_team.upper()
    return False


def apply_probable_pitcher_overrides(
    slate_games: pd.DataFrame,
    pitcher_names: dict[str, str],
    *,
    slate_date: str | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Apply manual probable-pitcher overrides; returns updated games and name map."""
    if slate_games.empty:
        return slate_games, pitcher_names

    overrides: list[ProbablePitcherOverride] = []
    if slate_date and slate_date in PROBABLE_PITCHER_OVERRIDES:
        overrides.extend(PROBABLE_PITCHER_OVERRIDES[slate_date])
    if not overrides:
        return slate_games, pitcher_names

    games = slate_games.copy()
    names = dict(pitcher_names)

    for override in overrides:
        for idx, row in games.iterrows():
            if not _matches_game(row, override):
                continue
            id_col = f"sp_{override.side}_id"
            hand_col = f"sp_{override.side}_hand"
            old_id = str(row.get(id_col, ""))
            games.at[idx, id_col] = override.pitcher_id
            games.at[idx, hand_col] = override.hand.upper()
            names[override.pitcher_id] = override.pitcher_name
            logger.info(
                "Probable pitcher override: game %s (%s @ %s) %s SP %s → %s (%s)",
                row.get("mlb_game_pk", row.get("game_id")),
                row.get("away_team_id"),
                row.get("home_team_id"),
                override.side,
                old_id or "none",
                override.pitcher_name,
                override.pitcher_id,
            )
            break

    return games, names
