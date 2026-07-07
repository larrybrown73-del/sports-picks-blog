from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)

SCHEMAS: dict[str, list[str]] = {
    "slate_games": [
        "game_id",
        "game_date",
        "home_team_id",
        "away_team_id",
        "park_id",
        "sp_home_id",
        "sp_away_id",
        "sp_home_hand",
        "sp_away_hand",
        "mlb_game_pk",
    ],
    "player_baselines": [
        "player_id",
        "season_woba",
        "roll14_woba",
        "roll30_woba",
        "season_iso",
        "roll14_iso",
        "roll30_iso",
        "season_k_pct",
        "roll14_k_pct",
        "roll30_k_pct",
        "season_bb_pct",
        "roll14_bb_pct",
        "roll30_bb_pct",
        "season_wrc_plus",
        "roll14_wrc_plus",
        "roll30_wrc_plus",
        "season_hard_hit_pct",
        "roll14_hard_hit_pct",
        "roll30_hard_hit_pct",
        "season_pa",
    ],
    "matchup_splits": [
        "player_id",
        "split",
        "woba",
        "iso",
        "k_pct",
        "bb_pct",
        "wrc_plus",
        "hard_hit_pct",
    ],
    "pitcher_platoon_splits": [
        "pitcher_id",
        "split",
        "woba_allowed",
        "iso_allowed",
        "k_pct",
        "bb_pct",
        "bf",
    ],
    "team_pitching": [
        "team_id",
        "role",
        "woba_allowed",
        "iso_allowed",
        "k_pct",
        "bb_pct",
    ],
    "park_weather": [
        "park_id",
        "park_factor_runs",
        "park_factor_hr",
        "temp_f",
        "wind_mph",
        "wind_dir",
    ],
    "vegas_totals": [
        "game_id",
        "home_implied_runs",
        "away_implied_runs",
        "game_total",
    ],
    "lineups": [
        "game_id",
        "team_id",
        "lineup_slot",
        "player_id",
        "player_name",
        "bat_hand",
    ],
    "pitcher_tendencies": [
        "pitcher_id",
        "pitcher_name",
        "avg_outs_last5",
        "pitch_efficiency",
        "gs",
        "is_true_starter",
        "sp_k_pct",
        "sp_bb_pct",
        "avg_bf_per_start",
    ],
}


def validate_columns(df: pd.DataFrame, schema_name: str) -> None:
    """Raise ValueError if required columns are missing."""
    if schema_name not in SCHEMAS:
        raise KeyError(f"Unknown schema: {schema_name}")
    required = SCHEMAS[schema_name]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Schema '{schema_name}' missing columns: {missing}"
        )
    logger.debug("Validated schema '%s' (%d rows)", schema_name, len(df))


DISPLAY_NAME_FALLBACK = "Unknown"


def resolve_display_name(df: pd.DataFrame, id_col: str, name_col: str) -> pd.Series:
    """Return name column, falling back to id when name is missing (live feeds)."""
    if name_col in df.columns:
        names = df[name_col].astype(str)
        missing = df[name_col].isna() | names.str.strip().eq("")
        if missing.any():
            return names.where(~missing, df[id_col].astype(str))
        return names
    return df[id_col].astype(str)


@dataclass
class SlateContext:
    """Merged player-game frame ready for projection."""

    player_games: pd.DataFrame
    games: pd.DataFrame
    fallback_counts: dict[str, int]
    data_health: Any = None

    @property
    def n_players(self) -> int:
        return len(self.player_games)


TbPropVerdict = Literal["Play", "Pass", "Caution"]


@dataclass
class TbGameContext:
    game_id: str
    player_id: str
    player_name: str
    opponent_pitcher_id: str
    opponent_team_id: str
    lineup_slot: int | None
    mlb_game_pk: str | None
    park_id: str
    park_factor_runs: float
    park_factor_hr: float
    temp_f: float
    wind_mph: float
    wind_dir: str
    is_outdoor: bool
    opp_bullpen_fatigue_score: float
    opp_bullpen_fatigue_status: str
    proj_tb: float
    roll15_xbh_rate: float | None
    effective_woba: float | None
    effective_iso: float | None
    market_line: float


@dataclass
class TbPropEvaluation:
    verdict: TbPropVerdict
    model_mu_multiplier: float
    flags: list[str]
    alt_suggestion: str | None
    alt_market: str | None
    alt_line: float | None
    alt_odds: str | None
    notes: list[str]
