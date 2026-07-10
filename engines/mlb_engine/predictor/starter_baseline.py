"""Starting-pitcher ERA/WHIP baseline injection and pitching mismatch veto."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from baseball_data import get_starting_pitcher_info
from config import (
    LEAGUE_AVG_ERA,
    LEAGUE_AVG_RUNS,
    LEAGUE_AVG_WHIP,
    PITCHING_MISMATCH_OPP_ERA_MAX,
    PITCHING_MISMATCH_OUR_ERA_MIN,
    SP_BASELINE_RF_WEIGHT,
)
from pitcher_matchup import fetch_pitcher_season_profile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StarterBaselineResult:
    home_runs: float
    away_runs: float
    tags: tuple[str, ...]


def _safe_era_whip(era: float | None, whip: float | None) -> tuple[float, float]:
    era_val = era if era is not None and era > 0 else LEAGUE_AVG_ERA
    whip_val = whip if whip is not None and whip > 0 else LEAGUE_AVG_WHIP
    return era_val, whip_val


def expected_runs_vs_starter(era: float | None, whip: float | None) -> float:
    """Project runs scored against a starter from season ERA and WHIP."""
    era_val, whip_val = _safe_era_whip(era, whip)
    era_factor = era_val / LEAGUE_AVG_ERA
    whip_factor = whip_val / LEAGUE_AVG_WHIP
    combined = era_factor * 0.65 + whip_factor * 0.35
    return max(1.5, LEAGUE_AVG_RUNS * combined)


def apply_starter_baseline_injection(
    rf_home_runs: float,
    rf_away_runs: float,
    *,
    game_id: int,
    season: int,
) -> StarterBaselineResult:
    """
    Blend RF output with SP ERA/WHIP expectations so team-offense bias does not dominate.

    home_runs = runs by home offense (vs away starter)
    away_runs = runs by away offense (vs home starter)
    """
    info = get_starting_pitcher_info(game_id)
    tags: list[str] = []

    away_sp = (
        fetch_pitcher_season_profile(
            int(info["away_pitcher_id"]),
            pitcher_name=str(info.get("away_pitcher_name") or ""),
            season=season,
        )
        if info.get("away_pitcher_id") is not None
        else None
    )
    home_sp = (
        fetch_pitcher_season_profile(
            int(info["home_pitcher_id"]),
            pitcher_name=str(info.get("home_pitcher_name") or ""),
            season=season,
        )
        if info.get("home_pitcher_id") is not None
        else None
    )

    sp_home_offense = expected_runs_vs_starter(
        away_sp.season_era if away_sp else None,
        away_sp.season_whip if away_sp else None,
    )
    sp_away_offense = expected_runs_vs_starter(
        home_sp.season_era if home_sp else None,
        home_sp.season_whip if home_sp else None,
    )

    rf_weight = max(0.0, min(1.0, SP_BASELINE_RF_WEIGHT))
    sp_weight = 1.0 - rf_weight

    home_runs = rf_weight * rf_home_runs + sp_weight * sp_home_offense
    away_runs = rf_weight * rf_away_runs + sp_weight * sp_away_offense

    if away_sp:
        tags.append(
            f"baseline:away_sp:{away_sp.pitcher_name}:era{away_sp.season_era}:whip{away_sp.season_whip}"
        )
    if home_sp:
        tags.append(
            f"baseline:home_sp:{home_sp.pitcher_name}:era{home_sp.season_era}:whip{home_sp.season_whip}"
        )
    tags.append(f"baseline:blend:rf{rf_weight:.2f}_sp{sp_weight:.2f}")
    tags.append(f"baseline:home_runs:{home_runs:.2f}")
    tags.append(f"baseline:away_runs:{away_runs:.2f}")

    return StarterBaselineResult(home_runs=home_runs, away_runs=away_runs, tags=tuple(tags))


def clamp_secondary_run_adjustments(
    baseline_home_runs: float,
    baseline_away_runs: float,
    adjusted_home_runs: float,
    adjusted_away_runs: float,
    *,
    cap: float,
) -> tuple[float, float, tuple[str, ...]]:
    """Hard-cap combined secondary modifier impact to +/- cap fraction of baseline."""

    def _clamp_side(baseline: float, adjusted: float, label: str) -> tuple[float, str | None]:
        if baseline <= 0:
            return adjusted, None
        low = baseline * (1.0 - cap)
        high = baseline * (1.0 + cap)
        clamped = max(low, min(high, adjusted))
        if abs(clamped - adjusted) < 1e-9:
            return adjusted, None
        return clamped, f"secondary_cap:{label}:{adjusted:.2f}->{clamped:.2f}"

    tags: list[str] = []
    home, home_tag = _clamp_side(baseline_home_runs, adjusted_home_runs, "home")
    away, away_tag = _clamp_side(baseline_away_runs, adjusted_away_runs, "away")
    if home_tag:
        tags.append(home_tag)
    if away_tag:
        tags.append(away_tag)
    if tags:
        tags.append(f"secondary_cap:max_pct:{cap:.2f}")
    return home, away, tuple(tags)


def fetch_starter_eras(game_id: int, *, season: int) -> tuple[float | None, float | None]:
    """Return (home_sp_era, away_sp_era) for veto checks."""
    info = get_starting_pitcher_info(game_id)
    home_era: float | None = None
    away_era: float | None = None

    if info.get("home_pitcher_id") is not None:
        profile = fetch_pitcher_season_profile(
            int(info["home_pitcher_id"]),
            pitcher_name=str(info.get("home_pitcher_name") or ""),
            season=season,
        )
        if profile:
            home_era = profile.season_era

    if info.get("away_pitcher_id") is not None:
        profile = fetch_pitcher_season_profile(
            int(info["away_pitcher_id"]),
            pitcher_name=str(info.get("away_pitcher_name") or ""),
            season=season,
        )
        if profile:
            away_era = profile.season_era

    return home_era, away_era


def pitching_mismatch_veto(
    *,
    our_sp_era: float | None,
    opponent_sp_era: float | None,
) -> bool:
    """
    Auto-drop when our starter is bad (ERA > 5.00) and opponent starter is elite (< 4.00).
    """
    if our_sp_era is None or opponent_sp_era is None:
        return False
    return (
        our_sp_era > PITCHING_MISMATCH_OUR_ERA_MIN
        and opponent_sp_era < PITCHING_MISMATCH_OPP_ERA_MAX
    )
