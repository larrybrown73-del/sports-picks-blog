"""Hitter plate-discipline scalars for predictor team run adjustments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import requests

from config import (
    BOTTOM_ORDER_PENALTY,
    DISCIPLINE_BONUS,
    DISCIPLINE_BONUS_ELITE_BB_PCT,
    ERRATIC_SWINGER_BB_PCT,
    ERRATIC_SWINGER_K_PCT,
    ERRATIC_SWINGER_PENALTY,
    PREMIUM_SLOT_MAX,
    PREMIUM_SLOT_SCALAR,
)
from data_health import safe_feature_fetch

logger = logging.getLogger(__name__)

_SLOT_PA_WEIGHTS: dict[int, float] = {
    1: 1.12,
    2: 1.08,
    3: 1.05,
    4: 1.03,
    5: 1.00,
    6: 0.97,
    7: 0.94,
    8: 0.91,
    9: 0.88,
}


@dataclass(frozen=True)
class LineupBatter:
    player_id: int
    lineup_slot: int


@dataclass(frozen=True)
class BatterDisciplineProfile:
    player_id: int
    k_pct: float | None
    bb_pct: float | None


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_percentage(rate: float | None) -> float | None:
    if rate is None:
        return None
    return rate * 100.0 if rate <= 1.0 else rate


def _slot_weight(lineup_slot: int) -> float:
    return _SLOT_PA_WEIGHTS.get(lineup_slot, 0.88)


def lineup_slot_scalar(lineup_slot: int) -> float:
    if lineup_slot <= PREMIUM_SLOT_MAX:
        return PREMIUM_SLOT_SCALAR
    if lineup_slot >= 8:
        return BOTTOM_ORDER_PENALTY
    return 1.0


@lru_cache(maxsize=256)
def fetch_batter_discipline_profile(player_id: int, season: int) -> BatterDisciplineProfile:
    try:
        response = requests.get(
            f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats",
            params={"stats": "season", "group": "hitting", "season": season},
            timeout=20,
        )
        response.raise_for_status()
        stats = response.json().get("stats") or []
        if not stats:
            return BatterDisciplineProfile(player_id=player_id, k_pct=None, bb_pct=None)
        splits = stats[0].get("splits") or []
        if not splits:
            return BatterDisciplineProfile(player_id=player_id, k_pct=None, bb_pct=None)
        stat: dict[str, Any] = splits[0].get("stat") or {}
        plate_appearances = _safe_float(stat.get("plateAppearances"))
        if not plate_appearances or plate_appearances <= 0:
            return BatterDisciplineProfile(player_id=player_id, k_pct=None, bb_pct=None)
        strike_outs = _safe_float(stat.get("strikeOuts")) or 0.0
        walks = _safe_float(stat.get("baseOnBalls")) or 0.0
        return BatterDisciplineProfile(
            player_id=player_id,
            k_pct=100.0 * strike_outs / plate_appearances,
            bb_pct=100.0 * walks / plate_appearances,
        )
    except Exception as exc:
        logger.debug("Batter discipline fetch failed for %s: %s", player_id, exc)
        return BatterDisciplineProfile(player_id=player_id, k_pct=None, bb_pct=None)


def _extract_lineup_side(box: dict[str, Any], side: str) -> list[LineupBatter]:
    team = box.get("teams", {}).get(side, {})
    batting_order = team.get("battingOrder") or []
    players = team.get("players") or {}
    batters: list[LineupBatter] = []
    for slot, player_key in enumerate(batting_order, start=1):
        player = players.get(player_key) or players.get(f"ID{player_key}")
        if not player:
            continue
        person = player.get("person") or {}
        pid = person.get("id")
        if pid is None:
            continue
        batters.append(LineupBatter(player_id=int(pid), lineup_slot=slot))
    return batters


def fetch_game_lineup(game_id: int) -> tuple[list[LineupBatter], list[LineupBatter]]:
    def _fetch() -> tuple[list[LineupBatter], list[LineupBatter]]:
        response = requests.get(
            f"https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore",
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        away = _extract_lineup_side(payload, "away")
        home = _extract_lineup_side(payload, "home")
        if away or home:
            return away, home
        live = payload.get("liveData", {}).get("boxscore", payload)
        return _extract_lineup_side(live, "away"), _extract_lineup_side(live, "home")

    return safe_feature_fetch(
        f"game_lineup_{game_id}",
        _fetch,
        fallback=([], []),
    )


def _batter_run_scalar(profile: BatterDisciplineProfile, lineup_slot: int) -> tuple[float, list[str]]:
    tags: list[str] = []
    scalar = lineup_slot_scalar(lineup_slot)
    if scalar != 1.0:
        if scalar > 1.0:
            tags.append(f"slot{lineup_slot}:premium:{scalar:.2f}")
        else:
            tags.append(f"slot{lineup_slot}:bottom:{scalar:.2f}")

    k_pct = _as_percentage(profile.k_pct)
    bb_pct = _as_percentage(profile.bb_pct)
    if bb_pct is not None and bb_pct > DISCIPLINE_BONUS_ELITE_BB_PCT:
        scalar *= DISCIPLINE_BONUS
        tags.append(f"p{profile.player_id}:discipline:{DISCIPLINE_BONUS:.2f}")
    if (
        k_pct is not None
        and bb_pct is not None
        and k_pct > ERRATIC_SWINGER_K_PCT
        and bb_pct < ERRATIC_SWINGER_BB_PCT
    ):
        scalar *= ERRATIC_SWINGER_PENALTY
        tags.append(f"p{profile.player_id}:erratic:{ERRATIC_SWINGER_PENALTY:.2f}")
    return scalar, tags


def lineup_offense_scalar(lineup: list[LineupBatter], season: int) -> tuple[float, list[str]]:
    """Weighted lineup scalar for team run generation."""
    if not lineup:
        return 1.0, []

    weighted = 0.0
    total_weight = 0.0
    tags: list[str] = []
    for batter in lineup:
        profile = fetch_batter_discipline_profile(batter.player_id, season)
        batter_scalar, batter_tags = _batter_run_scalar(profile, batter.lineup_slot)
        weight = _slot_weight(batter.lineup_slot)
        weighted += batter_scalar * weight
        total_weight += weight
        tags.extend(batter_tags)

    if total_weight <= 0:
        return 1.0, tags
    return weighted / total_weight, tags


def apply_lineup_discipline_to_runs(
    runs: float,
    lineup: list[LineupBatter],
    *,
    season: int,
    label: str,
) -> tuple[float, list[str]]:
    scalar, tags = lineup_offense_scalar(lineup, season)
    if abs(scalar - 1.0) < 1e-9:
        return runs, []
    adjusted_tags = [f"{label}:{tag}" for tag in tags]
    adjusted_tags.append(f"{label}:lineup_scalar:{scalar:.3f}")
    return runs * scalar, adjusted_tags
