"""Starter rest days and rotation hierarchy evaluation for run suppression."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache

import statsapi

from config import (
    ACE_RUN_SUPPRESSION_FACTOR,
    ACE_DOMINANCE_MIN_BATFERS_FACED,
    BACK_END_STARTER_PENALTY,
    CONTACT_STARTER_K_BB_PCT,
    CONTACT_STARTER_MAX_BONUS,
    ELITE_ACE_SCALAR,
    OPTIMAL_REST_BONUS,
    OPTIMAL_REST_MAX_DAYS,
    OPTIMAL_REST_MIN_DAYS,
    RUST_MIN_DAYS,
    RUST_PENALTY,
    SHORT_REST_PENALTY,
    STARTER_REST_LOOKBACK_DAYS,
    TOP_OF_ROTATION_SCALAR,
    TRUE_ACE_K_BB_PCT,
    TRUE_ACE_WHIP_MAX,
)
from data_health import safe_feature_fetch
from team_defense import contact_defense_scalar

logger = logging.getLogger(__name__)

_PITCHER_POSITION_CODE = "1"
_BOXSCORE_CACHE: dict[int, dict] = {}


@dataclass(frozen=True)
class AceDominanceProfile:
    pitcher_id: int
    k_bb_pct: float | None
    whip: float | None
    is_true_ace: bool
    is_innings_eater: bool


@dataclass(frozen=True)
class StarterEvaluation:
    pitcher_id: int
    team_id: int
    days_rest: int | None
    rotation_slot: int | None
    tier: int
    k_bb_pct: float | None
    rest_scalar: float
    hierarchy_scalar: float
    defense_scalar: float
    combined_scalar: float
    tags: tuple[str, ...]


class StarterRestAndHierarchyTracker:
    """Evaluate starter rest and rotation tier from live schedule and depth usage."""

    def evaluate(
        self,
        pitcher_id: int,
        team_id: int,
        *,
        game_date: date,
        season: int,
    ) -> StarterEvaluation:
        return safe_feature_fetch(
            f"starter_context_{pitcher_id}_{team_id}_{game_date.isoformat()}",
            lambda: self._evaluate(pitcher_id, team_id, game_date=game_date, season=season),
            fallback=StarterEvaluation(
                pitcher_id=pitcher_id,
                team_id=team_id,
                days_rest=None,
                rotation_slot=None,
                tier=2,
                k_bb_pct=None,
                rest_scalar=1.0,
                hierarchy_scalar=1.0,
                defense_scalar=1.0,
                combined_scalar=1.0,
                tags=("unknown",),
            ),
        )

    def _evaluate(
        self,
        pitcher_id: int,
        team_id: int,
        *,
        game_date: date,
        season: int,
    ) -> StarterEvaluation:
        last_start = _last_start_date(pitcher_id, team_id, before=game_date, season=season)
        days_rest = (game_date - last_start).days if last_start is not None else None

        rotation_slot = _rotation_slot(pitcher_id, team_id, season=season)
        tier = _tier_from_slot(rotation_slot)
        dominance = _fetch_ace_dominance_profile(pitcher_id, season=season)
        rest_scalar, rest_tag = _rest_scalar(
            days_rest,
            il_return=_is_il_return(pitcher_id, team_id, last_start, game_date),
        )
        hierarchy_scalar, tier_tag = _hierarchy_scalar(tier, dominance)

        combined = rest_scalar * hierarchy_scalar
        defense_scalar = 1.0
        tags = [rest_tag, tier_tag]
        if dominance.is_innings_eater and not dominance.is_true_ace:
            defense_scalar, defense_tag = contact_defense_scalar(team_id, season=season)
            if defense_tag:
                combined *= defense_scalar
                tags.append(defense_tag)
        if dominance.is_true_ace and rest_tag == "optimal_rest":
            combined *= ACE_RUN_SUPPRESSION_FACTOR
            tags.append(f"ace_synergy:{ACE_RUN_SUPPRESSION_FACTOR:.2f}")
        if dominance.k_bb_pct is not None:
            tags.append(f"k_bb_pct:{dominance.k_bb_pct:.1f}")

        return StarterEvaluation(
            pitcher_id=pitcher_id,
            team_id=team_id,
            days_rest=days_rest,
            rotation_slot=rotation_slot,
            tier=tier,
            k_bb_pct=dominance.k_bb_pct,
            rest_scalar=rest_scalar,
            hierarchy_scalar=hierarchy_scalar,
            defense_scalar=defense_scalar,
            combined_scalar=combined,
            tags=tuple(tags),
        )


def _tier_from_slot(rotation_slot: int | None) -> int:
    if rotation_slot is None:
        return 2
    if rotation_slot <= 2:
        return 1
    if rotation_slot <= 4:
        return 2
    return 3


def _rest_scalar(days_rest: int | None, *, il_return: bool) -> tuple[float, str]:
    if days_rest is None:
        return 1.0, "neutral_rest"
    if days_rest < 4:
        return SHORT_REST_PENALTY, "short_rest"
    if OPTIMAL_REST_MIN_DAYS <= days_rest <= OPTIMAL_REST_MAX_DAYS:
        return OPTIMAL_REST_BONUS, "optimal_rest"
    if days_rest >= RUST_MIN_DAYS and not il_return:
        return RUST_PENALTY, "rust_penalty"
    return 1.0, "neutral_rest"


def _hierarchy_scalar(tier: int, dominance: AceDominanceProfile) -> tuple[float, str]:
    if dominance.is_true_ace:
        return ELITE_ACE_SCALAR, "true_ace"

    if tier == 1:
        base_scalar = TOP_OF_ROTATION_SCALAR
        base_tag = "depth_chart_starter"
    elif tier == 3:
        base_scalar = BACK_END_STARTER_PENALTY
        base_tag = "tier3_back_end"
    else:
        base_scalar = 1.0
        base_tag = "tier2_mid_rotation"

    if dominance.is_innings_eater and base_scalar < 1.0:
        return max(base_scalar, CONTACT_STARTER_MAX_BONUS), "innings_eater_cap"
    return base_scalar, base_tag


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_pitcher_season_stat(person_id: int, season: int) -> dict:
    try:
        payload = statsapi.get(
            "people",
            {
                "personIds": person_id,
                "hydrate": f"stats(group=[pitching],type=[season],season={season})",
            },
        )
        people = payload.get("people", [])
        if not people:
            return {}
        stats = people[0].get("stats", [])
        if not stats:
            return {}
        splits = stats[0].get("splits", [])
        if not splits:
            return {}
        return splits[0].get("stat", {}) or {}
    except Exception as exc:
        logger.debug("Pitcher season stat fetch failed for %s: %s", person_id, exc)
        return {}


def _fetch_ace_dominance_profile(pitcher_id: int, *, season: int) -> AceDominanceProfile:
    stat = _fetch_pitcher_season_stat(pitcher_id, season)
    batters_faced = _safe_float(stat.get("battersFaced")) or 0.0
    strike_outs = _safe_float(stat.get("strikeOuts")) or 0.0
    walks = _safe_float(stat.get("baseOnBalls")) or 0.0
    whip = _safe_float(stat.get("whip"))
    if whip is None:
        hits = _safe_float(stat.get("hits")) or 0.0
        innings = _safe_float(stat.get("inningsPitched")) or 0.0
        if innings > 0:
            whip = (walks + hits) / innings

    k_bb_pct: float | None = None
    if batters_faced >= ACE_DOMINANCE_MIN_BATFERS_FACED:
        k_pct = 100.0 * strike_outs / batters_faced
        bb_pct = 100.0 * walks / batters_faced
        k_bb_pct = k_pct - bb_pct

    is_true_ace = (
        k_bb_pct is not None
        and whip is not None
        and k_bb_pct > TRUE_ACE_K_BB_PCT
        and whip < TRUE_ACE_WHIP_MAX
    )
    is_innings_eater = k_bb_pct is not None and k_bb_pct < CONTACT_STARTER_K_BB_PCT

    return AceDominanceProfile(
        pitcher_id=pitcher_id,
        k_bb_pct=k_bb_pct,
        whip=whip,
        is_true_ace=is_true_ace,
        is_innings_eater=is_innings_eater,
    )


def _parse_game_date(game: dict) -> date | None:
    raw = game.get("game_date")
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw), "%Y-%m-%d").date()
    except ValueError:
        return None


def _boxscore(game_id: int) -> dict:
    if game_id in _BOXSCORE_CACHE:
        return _BOXSCORE_CACHE[game_id]
    try:
        payload = statsapi.boxscore_data(game_id)
    except Exception as exc:
        logger.debug("Boxscore fetch failed for game %s: %s", game_id, exc)
        payload = {}
    _BOXSCORE_CACHE[game_id] = payload
    return payload


def _starter_id_for_team(boxscore: dict, team_id: int, game: dict) -> int | None:
    for side in ("home", "away"):
        if game.get(f"{side}_id") != team_id:
            continue
        pitchers = boxscore.get(f"{side}Pitchers", [])
        if not pitchers:
            return None
        person_id = pitchers[0].get("personId")
        return int(person_id) if person_id else None
    return None


def _last_start_date(
    pitcher_id: int,
    team_id: int,
    *,
    before: date,
    season: int,
) -> date | None:
    lookback_start = before - timedelta(days=STARTER_REST_LOOKBACK_DAYS)
    try:
        games = statsapi.schedule(
            start_date=lookback_start.strftime("%m/%d/%Y"),
            end_date=(before - timedelta(days=1)).strftime("%m/%d/%Y"),
            team=team_id,
            sportId=1,
        )
    except Exception as exc:
        logger.debug("Starter rest schedule fetch failed: %s", exc)
        return None

    last_start: date | None = None
    for game in games:
        if game.get("game_type") != "R" or game.get("status") != "Final":
            continue
        game_day = _parse_game_date(game)
        if game_day is None:
            continue
        boxscore = _boxscore(int(game["game_id"]))
        starter_id = _starter_id_for_team(boxscore, team_id, game)
        if starter_id == pitcher_id:
            if last_start is None or game_day > last_start:
                last_start = game_day
    return last_start


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _fetch_active_pitcher_ids(team_id: int) -> list[int]:
    try:
        payload = statsapi.get("team_roster", {"teamId": team_id, "rosterType": "active"})
    except Exception as exc:
        logger.debug("Rotation roster fetch failed for %s: %s", team_id, exc)
        return []

    pitcher_ids: list[int] = []
    for entry in payload.get("roster", []):
        position = entry.get("position", {}) or {}
        if str(position.get("code", "")) != _PITCHER_POSITION_CODE:
            continue
        pid = _safe_int((entry.get("person") or {}).get("id"))
        if pid is not None:
            pitcher_ids.append(pid)
    return pitcher_ids


def _pitcher_games_started(person_id: int, season: int) -> float:
    stat = _fetch_pitcher_season_stat(person_id, season)
    games_started = stat.get("gamesStarted")
    return float(games_started or 0.0)


@lru_cache(maxsize=64)
def _team_rotation_rank_map(team_id: int, season: int) -> dict[int, int]:
    usage: list[tuple[int, float]] = []
    for person_id in _fetch_active_pitcher_ids(team_id):
        games_started = _pitcher_games_started(person_id, season)
        if games_started > 0:
            usage.append((person_id, games_started))
    usage.sort(key=lambda row: row[1], reverse=True)
    return {person_id: index + 1 for index, (person_id, _gs) in enumerate(usage)}


def _rotation_slot(pitcher_id: int, team_id: int, *, season: int) -> int | None:
    ranks = _team_rotation_rank_map(team_id, season)
    return ranks.get(pitcher_id)


def _is_il_return(
    pitcher_id: int,
    team_id: int,
    last_start: date | None,
    game_date: date,
) -> bool:
    if last_start is None:
        return False
    gap_days = (game_date - last_start).days
    if gap_days < RUST_MIN_DAYS:
        return False

    try:
        payload = statsapi.get("injuries", {"sportId": 1, "teamId": team_id})
    except Exception as exc:
        logger.debug("Injury fetch failed for team %s: %s", team_id, exc)
        return gap_days > 45

    for entry in payload.get("injuries", []):
        person = entry.get("person", {}) or {}
        if _safe_int(person.get("id")) != pitcher_id:
            continue
        injury_date_raw = entry.get("date") or entry.get("effectiveDate")
        if not injury_date_raw:
            return True
        try:
            injury_date = datetime.strptime(str(injury_date_raw)[:10], "%Y-%m-%d").date()
        except ValueError:
            return True
        if last_start <= injury_date <= game_date:
            return True
    return False


def apply_starter_context_to_runs(
    offense_runs: float,
    *,
    pitcher_id: int | None,
    defending_team_id: int,
    game_date: date,
    season: int,
    label: str,
) -> tuple[float, list[str]]:
    """Apply starter rest and hierarchy scalars to projected offense runs."""
    if pitcher_id is None:
        return offense_runs, []

    evaluation = StarterRestAndHierarchyTracker().evaluate(
        int(pitcher_id),
        int(defending_team_id),
        game_date=game_date,
        season=season,
    )
    if abs(evaluation.combined_scalar - 1.0) < 1e-9:
        return offense_runs, []

    tags = [
        f"{label}:{tag}:{evaluation.combined_scalar:.2f}"
        if "synergy" in tag
        else f"{label}:{tag}"
        for tag in evaluation.tags
    ]
    if evaluation.days_rest is not None:
        tags.append(f"{label}:days_rest:{evaluation.days_rest}")
    if evaluation.rotation_slot is not None:
        tags.append(f"{label}:rotation_slot:SP{evaluation.rotation_slot}")
    tags.append(f"{label}:starter_scalar:{evaluation.combined_scalar:.3f}")
    return offense_runs * evaluation.combined_scalar, tags
