from __future__ import annotations

import pandas as pd

from baseball_props.analysis.pitcher_projection import project_pitcher_outs_and_pitches

__all__ = ["project_pitcher_outs", "project_pitcher_outs_and_pitches"]


def project_pitcher_outs(
    games: pd.DataFrame,
    pitcher_tendencies: pd.DataFrame,
    projected_batters: pd.DataFrame | None = None,
    team_pitching: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Backward-compatible wrapper; delegates to analysis.pitcher_projection."""
    return project_pitcher_outs_and_pitches(
        games,
        pitcher_tendencies,
        projected_batters,
        team_pitching,
    )
