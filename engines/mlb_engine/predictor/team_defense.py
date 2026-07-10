"""Team defensive quality rankings from Statcast Outs Above Average."""

from __future__ import annotations

import io
import logging
from functools import lru_cache
from typing import Literal

import pandas as pd
import requests
import statsapi

from config import (
    DEFENSE_OAA_MIN_ATTEMPTS,
    ELITE_DEFENSE_TOP_N,
    GOLD_GLOVE_BOOST,
    POOR_DEFENSE_BOTTOM_N,
    POOR_DEFENSE_PENALTY,
)
from data_health import safe_feature_fetch

logger = logging.getLogger(__name__)

DefenseTier = Literal["elite", "poor", "neutral"]

_SAVANT_OAA_URL = (
    "https://baseballsavant.mlb.com/leaderboard/outs_above_average"
    "?type=Fielder&startYear={season}&endYear={season}&range=year&min={min_attempts}&csv=true"
)


@lru_cache(maxsize=8)
def _team_name_to_id_map(season: int) -> dict[str, int]:
    try:
        payload = statsapi.get("teams", {"sportId": 1, "season": season})
    except Exception as exc:
        logger.warning("Team map fetch failed for defense lookup: %s", exc)
        return {}

    mapping: dict[str, int] = {}
    for team in payload.get("teams", []):
        team_name = str(team.get("teamName") or "").strip()
        team_id = team.get("id")
        if team_name and team_id is not None:
            mapping[team_name] = int(team_id)
    return mapping


def _fetch_team_oaa_totals(season: int) -> dict[int, float]:
    url = _SAVANT_OAA_URL.format(season=season, min_attempts=DEFENSE_OAA_MIN_ATTEMPTS)
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    frame = pd.read_csv(io.StringIO(response.text.lstrip("\ufeff")))
    if frame.empty or "display_team_name" not in frame.columns:
        return {}

    if "outs_above_average" not in frame.columns:
        return {}

    name_to_id = _team_name_to_id_map(season)
    grouped = (
        frame.groupby("display_team_name", as_index=False)["outs_above_average"]
        .sum()
        .sort_values("outs_above_average", ascending=False)
    )

    totals: dict[int, float] = {}
    for row in grouped.itertuples(index=False):
        team_name = str(row.display_team_name).strip()
        team_id = name_to_id.get(team_name)
        if team_id is None:
            continue
        totals[team_id] = float(row.outs_above_average)
    return totals


@lru_cache(maxsize=8)
def _defense_rank_sets(season: int) -> tuple[frozenset[int], frozenset[int]]:
    totals = _fetch_team_oaa_totals(season)
    if len(totals) < ELITE_DEFENSE_TOP_N + POOR_DEFENSE_BOTTOM_N:
        return frozenset(), frozenset()

    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    elite_ids = frozenset(team_id for team_id, _score in ranked[:ELITE_DEFENSE_TOP_N])
    poor_ids = frozenset(team_id for team_id, _score in ranked[-POOR_DEFENSE_BOTTOM_N:])
    return elite_ids, poor_ids


def team_defense_tier(team_id: int, *, season: int) -> DefenseTier:
    """Return elite/poor/neutral defensive tier for a team."""

    def _resolve() -> DefenseTier:
        elite_ids, poor_ids = _defense_rank_sets(season)
        if team_id in elite_ids:
            return "elite"
        if team_id in poor_ids:
            return "poor"
        return "neutral"

    return safe_feature_fetch(
        f"team_defense_tier_{team_id}_{season}",
        _resolve,
        fallback="neutral",
    )


def contact_defense_scalar(team_id: int, *, season: int) -> tuple[float, str | None]:
    """Return opponent-run scalar from team defense quality."""
    tier = team_defense_tier(team_id, season=season)
    if tier == "elite":
        return GOLD_GLOVE_BOOST, "gold_glove_boost"
    if tier == "poor":
        return POOR_DEFENSE_PENALTY, "poor_defense_penalty"
    return 1.0, None
