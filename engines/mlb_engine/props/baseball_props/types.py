from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

import pandas as pd

PitcherHand = Literal["L", "R"]
BatterHand = Literal["L", "R", "S"]
SplitKey = Literal["vs_lhp", "vs_rhp"]
PitcherSplitKey = Literal["vs_lhb", "vs_rhb"]
PitchingRole = Literal["sp", "bullpen"]

MetricName = Literal["woba", "iso", "k_pct", "bb_pct", "wrc_plus", "hard_hit_pct"]


class SlateFrames(TypedDict):
    slate_games: pd.DataFrame
    player_baselines: pd.DataFrame
    matchup_splits: pd.DataFrame
    pitcher_platoon_splits: pd.DataFrame
    team_pitching: pd.DataFrame
    park_weather: pd.DataFrame
    vegas_totals: pd.DataFrame
    lineups: pd.DataFrame
    pitcher_tendencies: pd.DataFrame
    odds_event_ids: NotRequired[list[str]]
    lineup_source_counts: NotRequired[dict[str, int]]
    game_context: NotRequired[pd.DataFrame]
    data_health: NotRequired[Any]
