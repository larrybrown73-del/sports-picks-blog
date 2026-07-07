from __future__ import annotations

import pandas as pd

from baseball_props.config import LEAGUE_AVG, PROBABILITY_METRICS, RATE_METRICS
from baseball_props.core.adjustments import (
    calculate_projected_probability,
    calculate_projected_rate,
)
from baseball_props.analysis.situational_adjustments import (
    apply_lineup_absence_penalty,
    apply_travel_rest_to_rates,
    apply_umpire_to_runs,
)
from baseball_props.environment.factors import apply_environment_to_rates
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)

_OPP_COL_MAP: dict[str, str] = {
    "woba": "opp_sp_woba_allowed",
    "iso": "opp_sp_iso_allowed",
    "k_pct": "opp_sp_k_pct",
    "bb_pct": "opp_sp_bb_pct",
}

_MATCHUP_COL_MAP: dict[str, str] = {
    "woba": "matchup_woba",
    "iso": "matchup_iso",
    "k_pct": "matchup_k_pct",
    "bb_pct": "matchup_bb_pct",
}


def project_player_rates(player_games: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized Base Rate + Matchup Adjustment for all rate metrics.

    Applies environment multiplier to wOBA and ISO projections.
    """
    df = player_games.copy()

    for metric in RATE_METRICS:
        matchup_col = _MATCHUP_COL_MAP[metric]
        opp_col = _OPP_COL_MAP[metric]
        proj_col = f"proj_{metric}"
        league_val = LEAGUE_AVG[metric]

        if metric in PROBABILITY_METRICS:
            df[proj_col] = calculate_projected_probability(
                df[matchup_col],
                df[opp_col],
                league_val,
            )
        else:
            df[proj_col] = calculate_projected_rate(
                df[matchup_col],
                df[opp_col],
                league_val,
            )

    env_cols = ["proj_woba", "proj_iso"]
    df = apply_environment_to_rates(df, env_cols)
    df = apply_travel_rest_to_rates(df)
    df = apply_umpire_to_runs(df)
    df = apply_lineup_absence_penalty(df)

    logger.info("Projected rates for %d player-games", len(df))
    return df
