from __future__ import annotations

import pandas as pd

from baseball_props.config import LEAGUE_AVG_IMPLIED_RUNS, LEAGUE_PA_PER_RUN, SLOT_PA_WEIGHTS
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)


def _slot_weight_series(slots: pd.Series) -> pd.Series:
    """Map lineup slots to PA weights."""
    weights = slots.map(SLOT_PA_WEIGHTS)
    missing = weights.isna()
    if missing.any():
        logger.warning(
            "Unknown lineup slots for %d rows; defaulting weight to 1.0",
            int(missing.sum()),
        )
        weights = weights.fillna(1.0)
    return weights


def project_batter_pa(
    lineups: pd.DataFrame,
    vegas_totals: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Project plate appearances from batting order and Vegas implied team totals.

    Formula: team_implied_runs * slot_weight * league_pa_per_run / 9
    (normalized so a full lineup scales with team implied runs)
    """
    df = lineups.merge(games[["game_id", "home_team_id", "away_team_id"]], on="game_id")
    vegas = vegas_totals.set_index("game_id")

    def _team_implied(row: pd.Series) -> float:
        game_id = row["game_id"]
        if game_id not in vegas.index:
            logger.warning(
                "No Vegas totals for game_id %s; using league avg %.1f implied runs",
                game_id,
                LEAGUE_AVG_IMPLIED_RUNS,
            )
            return LEAGUE_AVG_IMPLIED_RUNS
        game = vegas.loc[game_id]
        if row["team_id"] == row["home_team_id"]:
            return float(game["home_implied_runs"])
        return float(game["away_implied_runs"])

    df["team_implied_runs"] = df.apply(_team_implied, axis=1)
    df["slot_weight"] = _slot_weight_series(df["lineup_slot"])
    weight_sum = sum(SLOT_PA_WEIGHTS.values())

    # Distribute team PA pool (implied_runs * league rate) by lineup slot weight
    df["proj_pa"] = (
        df["team_implied_runs"] * LEAGUE_PA_PER_RUN * df["slot_weight"] / weight_sum
    )

    logger.info(
        "Projected PA for %d batters (mean %.2f PA)",
        len(df),
        df["proj_pa"].mean(),
    )
    return df[["game_id", "team_id", "player_id", "lineup_slot", "proj_pa"]]
