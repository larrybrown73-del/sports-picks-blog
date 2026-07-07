from __future__ import annotations

import numpy as np
import pandas as pd

from baseball_props.logging_utils import get_logger
from baseball_props.types import PitcherHand, SplitKey

logger = get_logger(__name__)

_HAND_TO_SPLIT: dict[str, SplitKey] = {"L": "vs_lhp", "R": "vs_rhp"}


def hand_to_split(hand: PitcherHand | str) -> SplitKey:
    """Map pitcher handedness to batter split key."""
    key = _HAND_TO_SPLIT.get(str(hand).upper())
    if key is None:
        raise ValueError(f"Invalid pitcher hand: {hand}")
    return key


def resolve_opposing_sp_hand(player_games: pd.DataFrame) -> pd.DataFrame:
    """
    Assign opposing SP handedness and split key per batter row.

    Home batters face away SP; away batters face home SP.
    """
    df = player_games.copy()
    is_home = df["team_id"] == df["home_team_id"]
    df["opp_sp_hand"] = np.where(is_home, df["sp_away_hand"], df["sp_home_hand"])
    df["split_key"] = df["opp_sp_hand"].map(_HAND_TO_SPLIT)
    return df


def resolve_opposing_sp_id(player_games: pd.DataFrame) -> pd.DataFrame:
    """Assign opposing probable starter id per batter row."""
    df = player_games.copy()
    is_home = df["team_id"] == df["home_team_id"]
    df["opp_sp_id"] = np.where(is_home, df["sp_away_id"], df["sp_home_id"])
    return df


def apply_split_rates(
    player_games: pd.DataFrame,
    splits: pd.DataFrame,
    metrics: list[str],
) -> tuple[pd.DataFrame, int]:
    """
    Merge handedness split rates; fall back to effective baseline when missing.

    Returns updated frame and count of split fallbacks used.
    """
    df = player_games.copy()
    split_cols = ["player_id", "split"] + metrics
    splits_narrow = splits[split_cols].rename(columns={"split": "split_key"})

    df = df.merge(splits_narrow, on=["player_id", "split_key"], how="left", suffixes=("", "_split"))

    fallback_count = 0
    for metric in metrics:
        split_col = metric
        effective_col = f"effective_{metric}"
        matchup_col = f"matchup_{metric}"

        missing = df[split_col].isna()
        if missing.any():
            fallback_count += int(missing.sum())
            logger.warning(
                "Split fallback for %s: %d rows missing %s split, using effective baseline",
                metric,
                int(missing.sum()),
                df.loc[missing, "split_key"].iloc[0] if missing.any() else "",
            )

        df[matchup_col] = df[split_col].fillna(df[effective_col])

    if fallback_count:
        logger.warning("Total split fallbacks across metrics: %d row-metric pairs", fallback_count)

    return df, fallback_count
