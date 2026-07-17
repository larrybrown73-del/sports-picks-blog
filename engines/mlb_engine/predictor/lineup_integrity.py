"""Lineup integrity guardrails for missing star power bats."""

from __future__ import annotations

import logging
from functools import lru_cache

import requests
import statsapi

from config import (
    ENABLE_LINEUP_INJURY_CHECK,
    MISSING_STAR_BAT_PENALTY,
    STAR_POWER_BAT_TOP_N,
    STAR_POWER_MIN_HR,
    STAR_POWER_MIN_PA,
)
from data_health import safe_feature_fetch
from hitter_discipline import LineupBatter

logger = logging.getLogger(__name__)

_IL_STATUS_PREFIXES = ("D",)  # D10, D15, D60, etc.


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=64)
def team_star_power_batter_ids(team_id: int, season: int) -> tuple[int, ...]:
    """Top power bats for a club by season HR (with PA/HR floors)."""

    def _fetch() -> tuple[int, ...]:
        try:
            response = requests.get(
                "https://statsapi.mlb.com/api/v1/stats",
                params={
                    "stats": "season",
                    "group": "hitting",
                    "season": season,
                    "sportIds": 1,
                    "teamId": team_id,
                    "playerPool": "all",
                    "limit": 40,
                    "order": "desc",
                    "sortStat": "homeRuns",
                },
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            logger.debug("Star-power batting fetch failed for %s: %s", team_id, exc)
            return tuple()

        stars: list[int] = []
        splits = (payload.get("stats") or [{}])[0].get("splits") or []
        for split in splits:
            player = split.get("player") or {}
            player_id = _safe_int(player.get("id"))
            stat = split.get("stat") or {}
            pa = _safe_float(stat.get("plateAppearances")) or 0.0
            hr = _safe_float(stat.get("homeRuns")) or 0.0
            if player_id is None:
                continue
            if pa < STAR_POWER_MIN_PA or hr < STAR_POWER_MIN_HR:
                continue
            stars.append(player_id)
            if len(stars) >= STAR_POWER_BAT_TOP_N:
                break
        return tuple(stars)

    return safe_feature_fetch(
        f"team_star_power_{team_id}_{season}",
        _fetch,
        fallback=tuple(),
    )


@lru_cache(maxsize=64)
def team_injured_player_ids(team_id: int, season: int) -> frozenset[int]:
    """Player IDs currently on an IL / disabled roster status."""

    def _fetch() -> frozenset[int]:
        injured: set[int] = set()
        try:
            payload = statsapi.get(
                "team_roster",
                {"teamId": team_id, "rosterType": "fullRoster", "season": season},
            )
        except Exception as exc:
            logger.debug("Full roster fetch failed for %s: %s", team_id, exc)
            return frozenset()

        for entry in payload.get("roster") or []:
            person = entry.get("person") or {}
            player_id = _safe_int(person.get("id"))
            status = entry.get("status") or {}
            code = str(status.get("code") or "").upper()
            description = str(status.get("description") or "").lower()
            if player_id is None:
                continue
            on_il = code.startswith(_IL_STATUS_PREFIXES) or "injur" in description
            if on_il:
                injured.add(player_id)
        return frozenset(injured)

    return safe_feature_fetch(
        f"team_injured_ids_{team_id}_{season}",
        _fetch,
        fallback=frozenset(),
    )


def missing_star_batter_ids(
    *,
    team_id: int,
    season: int,
    lineup: list[LineupBatter] | None,
) -> list[int]:
    """
    Stars considered missing from today's offense.

    A star counts as missing when:
    - they are on the IL / disabled list, or
    - a full lineup is posted and they are not in the batting order.
    """
    stars = team_star_power_batter_ids(team_id, season)
    if not stars:
        return []

    injured = team_injured_player_ids(team_id, season)
    lineup_ids = {batter.player_id for batter in (lineup or [])}
    lineup_posted = len(lineup_ids) >= 8

    missing: list[int] = []
    for star_id in stars:
        if star_id in injured:
            missing.append(star_id)
            continue
        if lineup_posted and star_id not in lineup_ids:
            missing.append(star_id)
    return missing


def missing_star_bat_scalar(
    *,
    team_id: int,
    season: int,
    lineup: list[LineupBatter] | None,
    label: str,
) -> tuple[float, list[str]]:
    """Return stacked run scalar for each missing star power bat."""
    if not ENABLE_LINEUP_INJURY_CHECK:
        return 1.0, []

    missing = missing_star_batter_ids(team_id=team_id, season=season, lineup=lineup)
    if not missing:
        return 1.0, []

    scalar = MISSING_STAR_BAT_PENALTY ** len(missing)
    tags = [
        f"{label}:missing_star_bat:{pid}:{MISSING_STAR_BAT_PENALTY:.2f}"
        for pid in missing
    ]
    tags.append(f"{label}:missing_star_bat_scalar:{scalar:.3f}")
    return scalar, tags


def apply_lineup_integrity_to_runs(
    offense_runs: float,
    *,
    team_id: int,
    season: int,
    lineup: list[LineupBatter] | None,
    label: str,
) -> tuple[float, list[str]]:
    scalar, tags = missing_star_bat_scalar(
        team_id=team_id,
        season=season,
        lineup=lineup,
        label=label,
    )
    if abs(scalar - 1.0) < 1e-9:
        return offense_runs, []
    return offense_runs * scalar, tags
