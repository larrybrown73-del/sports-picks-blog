from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from baseball_props.analysis.edge_sheets import (
    PASS_NO_DATA,
    _is_valid_number,
    best_side_edge,
    prob_over_continuous,
)
from baseball_props.config import (
    EDGE_HITS_SIGMA,
    HITS_BABIP_FLOOR,
    HITS_BULLPEN_FATIGUE_BONUS,
    HITS_CONTACT_BONUS_MULTIPLIER,
    HITS_CONTACT_K_PCT_MAX,
    HITS_CONTACT_PCT_FLOOR,
    HITS_CONTACT_ROLLING_GAMES,
    HITS_LINEUP_SLOT_PENALTY,
    HITS_LINEUP_TOP_SLOT,
    HITS_MAX_ODDS_CAP,
    HITS_MIN_ADJUSTED_EDGE_PCT,
    HITS_MIN_PROBABILITY_FLOOR,
    HITS_PARK_HIT_BONUS_THRESHOLD,
    HITS_PROP_TARGET_LINES,
    HITS_WEATHER_BONUS_MULTIPLIER,
    HITS_WEATHER_TEMP_BOOST_F,
    PREDICTOR_PATH,
)
from baseball_props.data.data_health import safe_feature_slice
from baseball_props.data.mlb_live import fetch_active_roster_hitters
from baseball_props.data.statcast_feed import (
    apply_hits_momentum_multipliers,
    compute_contact_profile,
)
from baseball_props.environment.factors import compute_environment_multiplier
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)

_OUT_WIND_DIRS = {"out", "out_to_lf", "out_to_cf", "out_to_rf"}
_WIND_OUT_RE = re.compile(r"\bout\b", re.IGNORECASE)


@dataclass(frozen=True)
class GameContext:
    game_id: str
    player_id: str
    player_name: str
    opponent_pitcher_id: str
    opponent_team_id: str
    batting_team_id: str
    lineup_slot: int | None
    venue_id: int | None
    park_tb_factor: float
    temp_f: float | None
    wind_mph: float | None
    wind_dir: str | None
    game_date: date
    opp_bullpen_status: str | None = None
    home_team_id: str | None = None
    away_team_id: str | None = None
    mlb_game_pk: int | None = None
    home_mlb_team_id: int | None = None
    away_mlb_team_id: int | None = None
    umpire_run_modifier: float = 1.0


@dataclass
class HitsPropEvaluation:
    verdict: Literal["Play", "Pass"]
    base_proj_hits: float
    adjusted_prob_over: float | None
    edge_pct: float | None
    recommendation: str
    warnings: list[str] = field(default_factory=list)
    adjustments: dict[str, float] = field(default_factory=dict)


def fetch_game_weather_mlb(game_id: str | int) -> tuple[float | None, float | None, str | None]:
    """Fetch temperature and wind from MLB game feed when slate env is missing."""
    try:
        response = requests.get(
            f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live",
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        weather = (payload.get("gameData") or {}).get("weather") or {}
        temp_raw = weather.get("temp")
        temp_f = float(temp_raw) if temp_raw not in (None, "") else None
        wind_raw = str(weather.get("wind") or "").strip()
        wind_mph: float | None = None
        speed_match = re.search(r"(\d+)\s*mph", wind_raw, re.IGNORECASE)
        if speed_match:
            wind_mph = float(speed_match.group(1))
        wind_dir = "out" if _WIND_OUT_RE.search(wind_raw) else "neutral"
        return temp_f, wind_mph, wind_dir
    except Exception as exc:
        logger.debug("MLB weather fetch failed for game %s: %s", game_id, exc)
        return None, None, None


def _is_wind_out(wind_dir: str | None) -> bool:
    if not wind_dir:
        return False
    normalized = str(wind_dir).strip().lower()
    return normalized in _OUT_WIND_DIRS or "out" in normalized


def _is_hits_target_line(market_line: float) -> bool:
    return any(abs(float(market_line) - float(target)) < 1e-6 for target in HITS_PROP_TARGET_LINES)


def _resolve_bullpen_fatigued(game_context: GameContext) -> tuple[bool, float]:
    status = game_context.opp_bullpen_status
    if status:
        if "fatigued" in status.lower():
            return True, HITS_BULLPEN_FATIGUE_BONUS

    home_id = game_context.home_mlb_team_id
    away_id = game_context.away_mlb_team_id
    if home_id is None or away_id is None:
        return False, 0.0

    predictor_root = Path(PREDICTOR_PATH)
    if not predictor_root.is_dir():
        return False, 0.0

    try:
        path_str = str(predictor_root)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        from bullpen_fatigue import compute_bullpen_fatigue

        as_of = datetime.combine(game_context.game_date, datetime.min.time())

        def _fetch_bullpen():
            return compute_bullpen_fatigue(home_id, away_id, as_of)

        result = safe_feature_slice(
            "predictor_bullpen_fatigue",
            _fetch_bullpen,
            default=None,
            report=None,
            empty_check=lambda value: value is None,
        )
        if result is None:
            return False, 0.0
        opp_is_home = str(game_context.opponent_team_id) == str(game_context.home_team_id)
        opp_status = result.home_status if opp_is_home else result.away_status
        if "fatigued" in opp_status.lower():
            return True, HITS_BULLPEN_FATIGUE_BONUS
    except Exception as exc:
        logger.debug("Predictor bullpen bridge failed: %s", exc)
    return False, 0.0


def _verify_roster_membership(
    player_id: str,
    team_id: str,
    warnings: list[str],
) -> bool:
    if not str(team_id).strip().isdigit():
        return True
    roster = safe_feature_slice(
        "mlb_active_roster",
        lambda: fetch_active_roster_hitters(int(team_id)),
        default=[],
        report=None,
        empty_check=lambda value: not value,
    )
    if not roster:
        warnings.append("Missing-Data Warning: mlb_active_roster — roster unavailable")
        return True
    on_roster = player_id in {h["player_id"] for h in roster}
    if not on_roster:
        warnings.append(f"Player {player_id} not on active roster for team {team_id}")
    return on_roster


def _apply_contact_bonus(
    profile: dict[str, float] | None,
    adjustments: dict[str, float],
    warnings: list[str],
) -> float:
    """Reward low-K contact hitters instead of penalizing singles-heavy profiles."""
    if not profile:
        warnings.append(
            "Missing-Data Warning: statcast_contact_profile — using league-average baseline"
        )
        return 1.0

    k_pct = profile.get("k_pct")
    contact_pct = profile.get("contact_pct")
    babip = profile.get("babip")
    adjustments["k_pct"] = k_pct if k_pct is not None else float("nan")
    adjustments["contact_pct"] = contact_pct if contact_pct is not None else float("nan")
    adjustments["babip"] = babip if babip is not None else float("nan")

    if k_pct is None or k_pct >= HITS_CONTACT_K_PCT_MAX:
        return 1.0

    elevated_contact = contact_pct is not None and contact_pct >= HITS_CONTACT_PCT_FLOOR
    high_babip = babip is not None and babip >= HITS_BABIP_FLOOR
    if elevated_contact or high_babip:
        adjustments["contact_bonus"] = HITS_CONTACT_BONUS_MULTIPLIER
        return HITS_CONTACT_BONUS_MULTIPLIER
    return 1.0


def evaluate_hits_prop(
    player_id: str,
    opponent_pitcher_id: str,
    game_context: GameContext,
    *,
    proj_hits: float,
    market_line: float,
    over_odds: float | None,
    under_odds: float | None,
    prop_lines: pd.DataFrame | None = None,
    contact_profile: dict[str, float] | None = None,
) -> HitsPropEvaluation:
    """Apply Over 0.5 / Over 1.5 hits checklist guardrails on top of base projection."""
    del opponent_pitcher_id, prop_lines  # reserved for future SP-specific filters

    if not _is_valid_number(proj_hits) or not _is_valid_number(market_line):
        return HitsPropEvaluation(
            verdict="Pass",
            base_proj_hits=float(proj_hits) if _is_valid_number(proj_hits) else 0.0,
            adjusted_prob_over=None,
            edge_pct=None,
            recommendation=PASS_NO_DATA,
            warnings=["Missing projection or market line"],
        )

    warnings: list[str] = []
    adjustments: dict[str, float] = {}
    adjusted_proj = float(proj_hits)
    prob_multiplier = 1.0
    bullpen_bonus = 0.0
    verdict: Literal["Play", "Pass"] = "Play"
    recommendation = "Over"

    if not _is_hits_target_line(market_line):
        model_prob_over = prob_over_continuous(adjusted_proj, EDGE_HITS_SIGMA, market_line)
        if model_prob_over is None:
            return HitsPropEvaluation(
                verdict="Pass",
                base_proj_hits=proj_hits,
                adjusted_prob_over=None,
                edge_pct=None,
                recommendation=PASS_NO_DATA,
                warnings=warnings,
            )
        rec, prob_side, edge_pct = best_side_edge(model_prob_over, over_odds, under_odds)
        if edge_pct is None or not _is_valid_number(edge_pct):
            return HitsPropEvaluation(
                verdict="Pass",
                base_proj_hits=proj_hits,
                adjusted_prob_over=prob_side,
                edge_pct=None,
                recommendation=PASS_NO_DATA,
                warnings=warnings,
            )
        return HitsPropEvaluation(
            verdict="Play",
            base_proj_hits=proj_hits,
            adjusted_prob_over=model_prob_over,
            edge_pct=edge_pct,
            recommendation=rec,
            warnings=warnings,
            adjustments=adjustments,
        )

    _verify_roster_membership(player_id, game_context.batting_team_id, warnings)

    lineup_slot = game_context.lineup_slot
    if lineup_slot is None or lineup_slot > HITS_LINEUP_TOP_SLOT:
        warnings.append(
            f"Lineup slot {lineup_slot} outside top {HITS_LINEUP_TOP_SLOT}; applying penalty"
        )
        prob_multiplier *= HITS_LINEUP_SLOT_PENALTY
        adjustments["lineup_penalty"] = HITS_LINEUP_SLOT_PENALTY

    profile = contact_profile
    if profile is None:
        profile = safe_feature_slice(
            "statcast_contact_profile",
            lambda: compute_contact_profile(player_id, HITS_CONTACT_ROLLING_GAMES),
            default={},
            report=None,
            empty_check=lambda value: not value,
        )
    contact_multiplier = _apply_contact_bonus(profile, adjustments, warnings)
    prob_multiplier *= contact_multiplier

    temp_f = game_context.temp_f
    wind_mph = game_context.wind_mph
    wind_dir = game_context.wind_dir
    if temp_f is None and game_context.mlb_game_pk is not None:
        fetched_temp, fetched_wind, fetched_dir = safe_feature_slice(
            "mlb_game_weather",
            lambda: fetch_game_weather_mlb(game_context.mlb_game_pk),
            default=(None, None, None),
            report=None,
            empty_check=lambda value: all(v is None for v in value),
        )
        if all(v is None for v in (fetched_temp, fetched_wind, fetched_dir)):
            warnings.append(
                "Missing-Data Warning: mlb_game_weather — using league-average baseline"
            )
        temp_f = temp_f if temp_f is not None else fetched_temp
        wind_mph = wind_mph if wind_mph is not None else fetched_wind
        wind_dir = wind_dir if wind_dir is not None else fetched_dir

    env_row = pd.DataFrame(
        [
            {
                "park_factor_runs": game_context.park_tb_factor,
                "temp_f": temp_f if temp_f is not None else 72.0,
                "wind_mph": wind_mph if wind_mph is not None else 0.0,
                "wind_dir": wind_dir or "",
            }
        ]
    )
    env_mult = float(compute_environment_multiplier(env_row).iloc[0])
    adjustments["env_multiplier"] = env_mult

    env_bonus_applied = False
    if game_context.park_tb_factor >= HITS_PARK_HIT_BONUS_THRESHOLD:
        adjusted_proj *= HITS_WEATHER_BONUS_MULTIPLIER
        env_bonus_applied = True
        adjustments["park_hit_bonus"] = HITS_WEATHER_BONUS_MULTIPLIER
    elif temp_f is not None and temp_f > HITS_WEATHER_TEMP_BOOST_F:
        adjusted_proj *= HITS_WEATHER_BONUS_MULTIPLIER
        env_bonus_applied = True
        adjustments["temp_bonus"] = HITS_WEATHER_BONUS_MULTIPLIER
    elif _is_wind_out(wind_dir):
        adjusted_proj *= HITS_WEATHER_BONUS_MULTIPLIER
        env_bonus_applied = True
        adjustments["wind_out_bonus"] = HITS_WEATHER_BONUS_MULTIPLIER

    if not env_bonus_applied and env_mult > 1.0:
        adjusted_proj *= min(env_mult, HITS_WEATHER_BONUS_MULTIPLIER)

    adjusted_proj = apply_hits_momentum_multipliers(
        adjusted_proj,
        player_id,
        adjustments,
        warnings,
    )

    fatigued, bullpen_bonus = _resolve_bullpen_fatigued(game_context)
    if fatigued:
        adjustments["bullpen_bonus"] = bullpen_bonus

    model_prob_over = prob_over_continuous(adjusted_proj, EDGE_HITS_SIGMA, market_line)
    if model_prob_over is None:
        return HitsPropEvaluation(
            verdict="Pass",
            base_proj_hits=proj_hits,
            adjusted_prob_over=None,
            edge_pct=None,
            recommendation=PASS_NO_DATA,
            warnings=warnings,
            adjustments=adjustments,
        )
    model_prob_over = min(0.99, model_prob_over * prob_multiplier + bullpen_bonus)
    adjustments["adjusted_prob_over"] = model_prob_over

    rec, prob_side, edge_pct = best_side_edge(model_prob_over, over_odds, under_odds)

    if rec != "Over" or edge_pct is None or not _is_valid_number(edge_pct) or edge_pct < HITS_MIN_ADJUSTED_EDGE_PCT:
        verdict = "Pass"
        recommendation = "Pass" if rec != "Over" else "Pass (insufficient edge)"
        if rec != "Over":
            warnings.append(f"Model prefers {rec} at {market_line}")
        elif _is_valid_number(edge_pct):
            warnings.append(f"Edge {edge_pct:.1f}% below {HITS_MIN_ADJUSTED_EDGE_PCT}% threshold")
    elif model_prob_over < HITS_MIN_PROBABILITY_FLOOR:
        verdict = "Pass"
        recommendation = "Pass (probability floor)"
        warnings.append(
            f"Model Over prob {model_prob_over:.1%} below {HITS_MIN_PROBABILITY_FLOOR:.0%} floor"
        )
    elif over_odds is not None and _is_valid_number(over_odds) and float(over_odds) > HITS_MAX_ODDS_CAP:
        verdict = "Pass"
        recommendation = "Pass (odds cap)"
        warnings.append(
            f"Over odds +{int(float(over_odds))} exceed +{HITS_MAX_ODDS_CAP} cap"
        )

    return HitsPropEvaluation(
        verdict=verdict,
        base_proj_hits=proj_hits,
        adjusted_prob_over=model_prob_over,
        edge_pct=edge_pct if verdict == "Play" and _is_valid_number(edge_pct) else None,
        recommendation="Over" if verdict == "Play" else recommendation,
        warnings=warnings,
        adjustments=adjustments,
    )
