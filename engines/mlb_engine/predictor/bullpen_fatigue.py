"""Bullpen fatigue tracking from recent reliever pitch counts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import statsapi

from config import (
    BULLPEN_FATIGUE_PITCH_THRESHOLD,
    BULLPEN_FATIGUE_WIN_PROB_PENALTY,
    BULLPEN_FATIGUE_WINDOW_HOURS,
    BULLPEN_LOOKBACK_DAYS,
)
from model import PROB_CLAMP_LOW, normalize_win_probabilities

logger = logging.getLogger(__name__)

_BOXSCORE_CACHE: dict[tuple[int, str], list[dict]] = {}


@dataclass(frozen=True)
class BullpenStatus:
    home_status: str
    away_status: str
    home_penalty: float
    away_penalty: float

    @property
    def display(self) -> str:
        return f"H:{self.home_status} / A:{self.away_status}"


def _safe_pitch_count(value: object) -> int:
    if value in (None, ""):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


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


def _reliever_pitch_totals(
    team_id: int,
    as_of: datetime,
    *,
    lookback_days: int = BULLPEN_LOOKBACK_DAYS,
) -> tuple[dict[str, int], bool]:
    """Return per-reliever pitch totals in the fatigue window and whether data was found."""
    window_start = as_of - timedelta(hours=BULLPEN_FATIGUE_WINDOW_HOURS)
    schedule_start = (as_of.date() - timedelta(days=lookback_days)).strftime("%m/%d/%Y")
    schedule_end = as_of.date().strftime("%m/%d/%Y")

    try:
        games = statsapi.schedule(start_date=schedule_start, end_date=schedule_end, sportId=1)
    except Exception as exc:
        logger.warning("Schedule fetch failed for bullpen fatigue: %s", exc)
        return {}, False

    if not games:
        return {}, False

    totals: dict[str, int] = {}
    found_any = False

    for game in games:
        if game.get("game_type") != "R" or game.get("status") != "Final":
            continue

        game_dt = _parse_game_datetime(game)
        if game_dt is None or game_dt >= as_of or game_dt < window_start:
            continue

        is_home = game.get("home_id") == team_id
        is_away = game.get("away_id") == team_id
        if not is_home and not is_away:
            continue

        side = "home" if is_home else "away"
        pitchers = _team_pitchers(game["game_id"], side)
        if not pitchers:
            continue

        found_any = True
        starter_seen = False
        for row in pitchers:
            person_id = row.get("personId") or 0
            if not person_id:
                continue

            name = str(row.get("name") or row.get("namefield") or f"ID{person_id}")
            pitches = _safe_pitch_count(row.get("p"))

            if not starter_seen:
                starter_seen = True
                continue

            totals[name] = totals.get(name, 0) + pitches

    return totals, found_any


def _team_fatigue_status(team_id: int, as_of: datetime) -> tuple[str, float]:
    totals, found = _reliever_pitch_totals(team_id, as_of)
    if not found:
        return "Unknown", 0.0

    for name, pitch_count in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        if pitch_count >= BULLPEN_FATIGUE_PITCH_THRESHOLD:
            return f"Fatigued ({name}: {pitch_count} pitches)", BULLPEN_FATIGUE_WIN_PROB_PENALTY

    if totals:
        return "Fresh", 0.0
    return "Unknown", 0.0


def compute_bullpen_fatigue(
    home_id: int,
    away_id: int,
    as_of: datetime,
    game_id: int | None = None,
) -> BullpenStatus:
    """Compute bullpen fatigue penalties for both teams."""
    del game_id  # reserved for future first-pitch timing
    home_status, home_penalty = _team_fatigue_status(home_id, as_of)
    away_status, away_penalty = _team_fatigue_status(away_id, as_of)
    return BullpenStatus(
        home_status=home_status,
        away_status=away_status,
        home_penalty=home_penalty,
        away_penalty=away_penalty,
    )


def apply_bullpen_penalty(
    home_prob: float,
    away_prob: float,
    status: BullpenStatus,
) -> tuple[float, float]:
    """Subtract fatigue penalties and re-normalize win probabilities."""
    home = max(home_prob - status.home_penalty, PROB_CLAMP_LOW)
    away = max(away_prob - status.away_penalty, PROB_CLAMP_LOW)
    return normalize_win_probabilities(home, away)
