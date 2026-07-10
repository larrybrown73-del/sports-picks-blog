"""Bullpen fatigue and late-inning (7-9) run adjustments from StatsAPI workload."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache

import statsapi

from config import (
    BULLPEN_HIGH_LEVERAGE_ARMS,
    BULLPEN_LOOKBACK_DAYS,
    ELITE_BULLPEN_TOP_N,
    LATE_INNING_RUN_SHARE,
    OVERWORKED_BULLPEN_PENALTY,
    OVERWORKED_CONSECUTIVE_DAYS,
    OVERWORKED_PITCH_THRESHOLD_3D,
    RESTED_ELITE_BONUS,
)
from data_health import safe_feature_fetch

logger = logging.getLogger(__name__)

_BOXSCORE_CACHE: dict[tuple[int, str], list[dict]] = {}
_PITCHER_POSITION_CODE = "1"


@dataclass(frozen=True)
class BullpenStatus:
    home_status: str
    away_status: str
    home_opponent_late_scalar: float
    away_opponent_late_scalar: float
    home_penalty: float = 0.0
    away_penalty: float = 0.0
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def display(self) -> str:
        return f"H:{self.home_status} / A:{self.away_status}"


@dataclass(frozen=True)
class RelieverWorkload:
    person_id: int
    name: str
    pitches_by_date: dict[date, int]
    total_pitches: int


def _safe_pitch_count(value: object) -> int:
    if value in (None, ""):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_game_datetime(game: dict) -> datetime | None:
    for key in ("game_datetime", "gameDate"):
        raw = game.get(key)
        if not raw:
            continue
        try:
            if "T" in str(raw):
                return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            continue

    game_date = game.get("game_date")
    if game_date:
        try:
            return datetime.strptime(str(game_date), "%Y-%m-%d")
        except ValueError:
            pass
    return None


def _team_pitchers(game_id: int, team_side: str) -> list[dict]:
    cache_key = (game_id, team_side)
    if cache_key in _BOXSCORE_CACHE:
        return _BOXSCORE_CACHE[cache_key]

    try:
        boxscore = statsapi.boxscore_data(game_id)
        pitchers = boxscore.get(f"{team_side}Pitchers", [])
    except Exception as exc:
        logger.warning("Boxscore unavailable for game %s (%s): %s", game_id, team_side, exc)
        pitchers = []

    _BOXSCORE_CACHE[cache_key] = pitchers
    return pitchers


def _team_final_games(team_id: int, as_of: datetime, *, lookback_days: int) -> list[dict]:
    schedule_start = (as_of.date() - timedelta(days=lookback_days)).strftime("%m/%d/%Y")
    schedule_end = (as_of.date() - timedelta(days=1)).strftime("%m/%d/%Y")
    try:
        games = statsapi.schedule(start_date=schedule_start, end_date=schedule_end, sportId=1)
    except Exception as exc:
        logger.warning("Schedule fetch failed for bullpen fatigue: %s", exc)
        return []

    selected: list[dict] = []
    for game in games:
        if game.get("game_type") != "R" or game.get("status") != "Final":
            continue
        game_dt = _parse_game_datetime(game)
        if game_dt is None or game_dt >= as_of:
            continue
        if game.get("home_id") != team_id and game.get("away_id") != team_id:
            continue
        selected.append(game)
    return selected


def _reliever_workloads(
    team_id: int,
    as_of: datetime,
    *,
    lookback_days: int = BULLPEN_LOOKBACK_DAYS,
) -> dict[int, RelieverWorkload]:
    workloads: dict[int, RelieverWorkload] = {}
    for game in _team_final_games(team_id, as_of, lookback_days=lookback_days):
        side = "home" if game.get("home_id") == team_id else "away"
        pitchers = _team_pitchers(game["game_id"], side)
        if not pitchers:
            continue

        game_dt = _parse_game_datetime(game)
        if game_dt is None:
            continue
        game_day = game_dt.date()

        starter_seen = False
        for row in pitchers:
            person_id = row.get("personId") or 0
            if not person_id:
                continue
            if not starter_seen:
                starter_seen = True
                continue

            pitches = _safe_pitch_count(row.get("p"))
            if pitches <= 0:
                continue

            pid = int(person_id)
            name = str(row.get("name") or row.get("namefield") or f"ID{pid}")
            existing = workloads.get(pid)
            if existing is None:
                workloads[pid] = RelieverWorkload(
                    person_id=pid,
                    name=name,
                    pitches_by_date={game_day: pitches},
                    total_pitches=pitches,
                )
            else:
                by_date = dict(existing.pitches_by_date)
                by_date[game_day] = by_date.get(game_day, 0) + pitches
                workloads[pid] = RelieverWorkload(
                    person_id=pid,
                    name=name,
                    pitches_by_date=by_date,
                    total_pitches=existing.total_pitches + pitches,
                )
    return workloads


def _max_consecutive_active_days(active_dates: list[date]) -> int:
    if not active_dates:
        return 0
    ordered = sorted(active_dates)
    best = 1
    streak = 1
    for idx in range(1, len(ordered)):
        if (ordered[idx] - ordered[idx - 1]).days == 1:
            streak += 1
            best = max(best, streak)
        else:
            streak = 1
    return best


def _is_overworked(workload: RelieverWorkload) -> bool:
    if workload.total_pitches > OVERWORKED_PITCH_THRESHOLD_3D:
        return True
    return _max_consecutive_active_days(list(workload.pitches_by_date)) >= OVERWORKED_CONSECUTIVE_DAYS


def _pitches_on_date(workload: RelieverWorkload, day: date) -> int:
    return workload.pitches_by_date.get(day, 0)


def _fetch_active_pitcher_ids(team_id: int) -> list[int]:
    try:
        payload = statsapi.get("team_roster", {"teamId": team_id, "rosterType": "active"})
    except Exception as exc:
        logger.debug("Roster fetch failed for team %s: %s", team_id, exc)
        return []

    pitcher_ids: list[int] = []
    for entry in payload.get("roster", []):
        position = entry.get("position", {}) or {}
        if str(position.get("code", "")) != _PITCHER_POSITION_CODE:
            continue
        person = entry.get("person", {}) or {}
        pid = person.get("id")
        if pid is not None:
            pitcher_ids.append(int(pid))
    return pitcher_ids


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


def _leverage_score(stat: dict) -> float:
    saves = _safe_float(stat.get("saves")) or 0.0
    holds = _safe_float(stat.get("holds")) or 0.0
    games_finished = _safe_float(stat.get("gamesFinished")) or 0.0
    innings = _safe_float(stat.get("inningsPitched")) or 0.0
    games_started = _safe_float(stat.get("gamesStarted")) or 0.0
    if games_started >= 5:
        return -1.0
    return saves * 3.0 + holds * 2.0 + games_finished + min(innings, 40.0) * 0.05


def _high_leverage_arms(team_id: int, season: int) -> list[tuple[int, str]]:
    arms: list[tuple[int, str, float]] = []
    for person_id in _fetch_active_pitcher_ids(team_id):
        stat = _fetch_pitcher_season_stat(person_id, season)
        score = _leverage_score(stat)
        if score <= 0:
            continue
        name = str(stat.get("name") or f"ID{person_id}")
        arms.append((person_id, name, score))
    arms.sort(key=lambda row: row[2], reverse=True)
    return [(pid, name) for pid, name, _score in arms[:BULLPEN_HIGH_LEVERAGE_ARMS]]


def _team_bullpen_era_whip(team_id: int, season: int) -> tuple[float | None, float | None]:
    innings_total = 0.0
    earned_runs = 0.0
    walks = 0.0
    hits = 0.0

    for person_id in _fetch_active_pitcher_ids(team_id):
        stat = _fetch_pitcher_season_stat(person_id, season)
        games_started = _safe_float(stat.get("gamesStarted")) or 0.0
        games_played = _safe_float(stat.get("gamesPlayed")) or 0.0
        if games_played <= 0 or games_started >= 5:
            continue

        innings = _safe_float(stat.get("inningsPitched")) or 0.0
        if innings <= 0:
            continue

        innings_total += innings
        earned_runs += _safe_float(stat.get("earnedRuns")) or 0.0
        walks += _safe_float(stat.get("baseOnBalls")) or 0.0
        hits += _safe_float(stat.get("hits")) or 0.0

    if innings_total <= 0:
        return None, None

    era = 9.0 * earned_runs / innings_total
    whip = (walks + hits) / innings_total
    return era, whip


@lru_cache(maxsize=8)
def _elite_bullpen_team_ids(season: int) -> frozenset[int]:
    try:
        teams = statsapi.get("teams", {"sportId": 1, "season": season})
    except Exception as exc:
        logger.warning("Team list fetch failed for bullpen quality: %s", exc)
        return frozenset()

    ranked: list[tuple[int, float, float]] = []
    for team in teams.get("teams", []):
        team_id = int(team["id"])
        era, whip = _team_bullpen_era_whip(team_id, season)
        if era is None or whip is None:
            continue
        ranked.append((team_id, era, whip))

    ranked.sort(key=lambda row: (row[1], row[2]))
    return frozenset(team_id for team_id, _era, _whip in ranked[:ELITE_BULLPEN_TOP_N])


def _late_inning_scalar_for_team(
    team_id: int,
    as_of: datetime,
    *,
    season: int,
) -> tuple[float, str, list[str]]:
    workloads = _reliever_workloads(team_id, as_of)
    leverage_arms = _high_leverage_arms(team_id, season)
    tags: list[str] = []

    if leverage_arms:
        hl_ids = {pid for pid, _name in leverage_arms}
        for pid, name in leverage_arms:
            workload = workloads.get(pid)
            if workload and _is_overworked(workload):
                tags.append(f"dead_arm:{name}:{workload.total_pitches}p")
                return OVERWORKED_BULLPEN_PENALTY, "Dead Arm", tags

        yesterday = as_of.date() - timedelta(days=1)
        elite = team_id in _elite_bullpen_team_ids(season)
        if elite:
            rested = True
            for pid, name in leverage_arms:
                workload = workloads.get(pid)
                pitches_yesterday = _pitches_on_date(workload, yesterday) if workload else 0
                if pitches_yesterday > 0:
                    rested = False
                    break
            if rested:
                tags.append("lockdown:rested_elite")
                return RESTED_ELITE_BONUS, "Lockdown", tags

        if workloads:
            top_usage = max(
                (workloads[pid].total_pitches for pid in hl_ids if pid in workloads),
                default=0,
            )
            if top_usage > 0:
                return 1.0, "Fresh", tags

    return 1.0, "Unknown", tags


def _late_inning_game_multiplier(opponent_late_scalar: float) -> float:
    share = LATE_INNING_RUN_SHARE
    return (1.0 - share) + share * opponent_late_scalar


def compute_bullpen_fatigue(
    home_id: int,
    away_id: int,
    as_of: datetime,
    game_id: int | None = None,
    *,
    season: int | None = None,
) -> BullpenStatus:
    """Compute late-inning bullpen scalars for both teams."""
    del game_id
    season = season or as_of.year

    def _fetch() -> BullpenStatus:
        home_scalar, home_status, home_tags = _late_inning_scalar_for_team(
            home_id, as_of, season=season
        )
        away_scalar, away_status, away_tags = _late_inning_scalar_for_team(
            away_id, as_of, season=season
        )
        tags = [f"home:{tag}" for tag in home_tags] + [f"away:{tag}" for tag in away_tags]
        return BullpenStatus(
            home_status=home_status,
            away_status=away_status,
            home_opponent_late_scalar=home_scalar,
            away_opponent_late_scalar=away_scalar,
            tags=tuple(tags),
        )

    return safe_feature_fetch(
        "bullpen_fatigue",
        _fetch,
        fallback=BullpenStatus(
            home_status="Unknown",
            away_status="Unknown",
            home_opponent_late_scalar=1.0,
            away_opponent_late_scalar=1.0,
        ),
    )


def apply_bullpen_to_runs(
    home_runs: float,
    away_runs: float,
    status: BullpenStatus,
) -> tuple[float, float, list[str]]:
    """
    Layer late-inning bullpen scalars on projected runs before win probability.

    A tired home bullpen boosts away runs in innings 7-9; rested elite arms suppress them.
    """
    tags: list[str] = list(status.tags)
    home_mult = _late_inning_game_multiplier(status.away_opponent_late_scalar)
    away_mult = _late_inning_game_multiplier(status.home_opponent_late_scalar)

    adjusted_home = home_runs * home_mult
    adjusted_away = away_runs * away_mult

    if abs(home_mult - 1.0) > 1e-9:
        tags.append(f"home_runs_late_inning:{home_mult:.3f}")
    if abs(away_mult - 1.0) > 1e-9:
        tags.append(f"away_runs_late_inning:{away_mult:.3f}")

    return adjusted_home, adjusted_away, tags


def apply_bullpen_penalty(
    home_prob: float,
    away_prob: float,
    status: BullpenStatus,
) -> tuple[float, float]:
    """Backward-compatible shim; run adjustments are applied upstream."""
    del status
    return home_prob, away_prob
