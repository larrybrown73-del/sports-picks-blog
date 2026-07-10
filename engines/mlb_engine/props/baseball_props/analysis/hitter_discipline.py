"""Hitter plate-discipline profiles and prop-level adjustment scalars."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import requests

from baseball_props.config import (
    BOTTOM_ORDER_PENALTY,
    DISCIPLINE_BONUS,
    DISCIPLINE_BONUS_ELITE_BB_PCT,
    ERRATIC_SWINGER_BB_PCT,
    ERRATIC_SWINGER_K_PCT,
    ERRATIC_SWINGER_PENALTY,
    PREMIUM_SLOT_MAX,
    PREMIUM_SLOT_SCALAR,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatterDisciplineProfile:
    player_id: str
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
    """Normalize K%/BB% to 0–100 scale."""
    if rate is None:
        return None
    return rate * 100.0 if rate <= 1.0 else rate


def lineup_slot_prob_scalar(lineup_slot: int | None) -> tuple[float, str | None]:
    """Return probability scalar and label for batting-order slot."""
    if lineup_slot is None:
        return 1.0, None
    if lineup_slot <= PREMIUM_SLOT_MAX:
        return PREMIUM_SLOT_SCALAR, "premium_slot"
    if lineup_slot >= 8:
        return BOTTOM_ORDER_PENALTY, "bottom_order_penalty"
    return 1.0, None


@lru_cache(maxsize=256)
def fetch_batter_discipline_profile(player_id: str, season: int) -> BatterDisciplineProfile:
    """Season walk/strikeout rates for a batter."""
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
        k_pct = 100.0 * strike_outs / plate_appearances
        bb_pct = 100.0 * walks / plate_appearances
        return BatterDisciplineProfile(player_id=player_id, k_pct=k_pct, bb_pct=bb_pct)
    except Exception as exc:
        logger.debug("Batter discipline fetch failed for %s: %s", player_id, exc)
        return BatterDisciplineProfile(player_id=player_id, k_pct=None, bb_pct=None)


def apply_hitter_discipline_to_projection(
    adjusted_proj: float,
    profile: BatterDisciplineProfile,
    adjustments: dict[str, float],
) -> float:
    """Apply patient-eye bonus and hero-swing penalty to a hits projection."""
    proj = float(adjusted_proj)
    k_pct = _as_percentage(profile.k_pct)
    bb_pct = _as_percentage(profile.bb_pct)

    if bb_pct is not None and bb_pct > DISCIPLINE_BONUS_ELITE_BB_PCT:
        proj *= DISCIPLINE_BONUS
        adjustments["discipline_bonus"] = DISCIPLINE_BONUS

    if (
        k_pct is not None
        and bb_pct is not None
        and k_pct > ERRATIC_SWINGER_K_PCT
        and bb_pct < ERRATIC_SWINGER_BB_PCT
    ):
        proj *= ERRATIC_SWINGER_PENALTY
        adjustments["erratic_swinger_penalty"] = ERRATIC_SWINGER_PENALTY

    return proj


def is_elite_discipline(profile: BatterDisciplineProfile) -> bool:
    bb_pct = _as_percentage(profile.bb_pct)
    return bb_pct is not None and bb_pct > DISCIPLINE_BONUS_ELITE_BB_PCT


def is_erratic_swinger(profile: BatterDisciplineProfile) -> bool:
    k_pct = _as_percentage(profile.k_pct)
    bb_pct = _as_percentage(profile.bb_pct)
    return (
        k_pct is not None
        and bb_pct is not None
        and k_pct > ERRATIC_SWINGER_K_PCT
        and bb_pct < ERRATIC_SWINGER_BB_PCT
    )


def batter_team_run_scalar(profile: BatterDisciplineProfile) -> float:
    """Per-batter scalar contribution for team run generation."""
    scalar = 1.0
    if is_elite_discipline(profile):
        scalar *= DISCIPLINE_BONUS
    if is_erratic_swinger(profile):
        scalar *= ERRATIC_SWINGER_PENALTY
    return scalar
