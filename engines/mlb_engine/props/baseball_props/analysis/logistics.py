from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests

from baseball_props.config import (
    LOGISTICS_LOOKBACK_HOURS,
    TRAVEL_REST_B2B_PENALTY,
    TRAVEL_REST_LONG_MILES,
    TRAVEL_REST_LONG_MILES_PENALTY,
    TRAVEL_REST_TZ_DELTA_THRESHOLD,
    TRAVEL_REST_TZ_PENALTY,
)
from baseball_props.data.data_health import safe_feature_slice
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
DEFAULT_TIMEOUT = 25.0

TRAVEL_REST_COLUMNS = [
    "game_id",
    "team_id",
    "days_rest",
    "is_back_to_back",
    "travel_miles",
    "travel_zone_delta",
    "travel_rest_multiplier",
    "travel_rest_tag",
]

# MLB team abbrev → numeric id (Stats API)
TEAM_ABBREV_TO_ID: dict[str, int] = {
    "ARI": 109,
    "ATL": 144,
    "BAL": 110,
    "BOS": 111,
    "CHC": 112,
    "CWS": 145,
    "CIN": 113,
    "CLE": 114,
    "COL": 115,
    "DET": 116,
    "HOU": 117,
    "KC": 118,
    "LAA": 108,
    "LAD": 119,
    "MIA": 146,
    "MIL": 158,
    "MIN": 142,
    "NYM": 121,
    "NYY": 147,
    "ATH": 133,
    "OAK": 133,
    "PHI": 143,
    "PIT": 134,
    "SD": 135,
    "SF": 137,
    "SEA": 136,
    "STL": 138,
    "TB": 139,
    "TEX": 140,
    "TOR": 141,
    "WSH": 120,
}

# Home-market coordinates (lat, lon) and UTC offset hours for travel estimates
TEAM_HOME_GEO: dict[str, tuple[float, float, int]] = {
    "ARI": (33.4453, -112.0667, -7),
    "ATL": (33.8907, -84.4677, -5),
    "BAL": (39.2839, -76.6217, -5),
    "BOS": (42.3467, -71.0972, -5),
    "CHC": (41.9484, -87.6553, -6),
    "CWS": (41.8299, -87.6338, -6),
    "CIN": (39.0979, -84.5082, -5),
    "CLE": (41.4962, -81.6852, -5),
    "COL": (39.7559, -104.9942, -7),
    "DET": (42.3390, -83.0485, -5),
    "HOU": (29.7573, -95.3555, -6),
    "KC": (39.0517, -94.4803, -6),
    "LAA": (33.8003, -117.8827, -8),
    "LAD": (34.0739, -118.2400, -8),
    "MIA": (25.7781, -80.2197, -5),
    "MIL": (43.0280, -87.9712, -6),
    "MIN": (44.9817, -93.2776, -6),
    "NYM": (40.7571, -73.8458, -5),
    "NYY": (40.8296, -73.9262, -5),
    "ATH": (37.7516, -122.2005, -8),
    "OAK": (37.7516, -122.2005, -8),
    "PHI": (39.9061, -75.1665, -5),
    "PIT": (40.4469, -80.0057, -5),
    "SD": (32.7073, -117.1566, -8),
    "SF": (37.7786, -122.3893, -8),
    "SEA": (47.5914, -122.3325, -8),
    "STL": (38.6226, -90.1929, -6),
    "TB": (27.7682, -82.6534, -5),
    "TEX": (32.7512, -97.0832, -6),
    "TOR": (43.6414, -79.3894, -5),
    "WSH": (38.8730, -77.0074, -5),
}


@dataclass(frozen=True)
class ScheduleGameSlice:
    game_pk: int
    game_date: date
    home_abbrev: str
    away_abbrev: str
    home_team_id: int
    away_team_id: int
    venue_id: str
    venue_lat: float | None
    venue_lon: float | None
    is_final: bool


@dataclass(frozen=True)
class TravelRestRow:
    game_id: str
    team_id: str
    days_rest: int
    is_back_to_back: bool
    travel_miles: float
    travel_zone_delta: int
    travel_rest_multiplier: float
    travel_rest_tag: str


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_date(raw: Any) -> date | None:
    if raw is None:
        return None
    if isinstance(raw, date):
        return raw
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def _get_json(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{MLB_API_BASE}{path}"
    response = requests.get(url, params=params or {}, timeout=DEFAULT_TIMEOUT)
    if 400 <= response.status_code <= 499:
        return {}
    response.raise_for_status()
    return response.json()


def _is_final_game(raw: dict[str, Any]) -> bool:
    status = raw.get("status", {}) or {}
    if status.get("abstractGameState") == "Final":
        return True
    return status.get("codedGameState") == "F"


def _team_abbrev(team: dict[str, Any]) -> str:
    return str(team.get("abbreviation") or team.get("fileCode") or "").upper()


def _venue_coords(venue: dict[str, Any], home_abbrev: str) -> tuple[float | None, float | None]:
    location = venue.get("location") or {}
    lat = _safe_float(location.get("defaultCoordinates", {}).get("latitude"))
    lon = _safe_float(location.get("defaultCoordinates", {}).get("longitude"))
    if lat is not None and lon is not None:
        return lat, lon
    geo = TEAM_HOME_GEO.get(home_abbrev)
    if geo:
        return geo[0], geo[1]
    return None, None


def parse_schedule_game(raw: dict[str, Any]) -> ScheduleGameSlice | None:
    """Extract validated schedule fields; return None when required keys are missing."""
    game_pk = _safe_int(raw.get("gamePk"), default=0)
    if game_pk <= 0:
        return None

    game_date = _safe_date(raw.get("officialDate") or raw.get("gameDate"))
    if game_date is None:
        return None

    home = (raw.get("teams") or {}).get("home", {}).get("team", {}) or {}
    away = (raw.get("teams") or {}).get("away", {}).get("team", {}) or {}
    home_id = _safe_int(home.get("id"), default=0)
    away_id = _safe_int(away.get("id"), default=0)
    home_abbrev = _team_abbrev(home)
    away_abbrev = _team_abbrev(away)
    if home_id <= 0 or away_id <= 0 or not home_abbrev or not away_abbrev:
        return None

    venue = raw.get("venue") or {}
    venue_id = str(venue.get("id", "UNK"))
    lat, lon = _venue_coords(venue, home_abbrev)

    return ScheduleGameSlice(
        game_pk=game_pk,
        game_date=game_date,
        home_abbrev=home_abbrev,
        away_abbrev=away_abbrev,
        home_team_id=home_id,
        away_team_id=away_id,
        venue_id=venue_id,
        venue_lat=lat,
        venue_lon=lon,
        is_final=_is_final_game(raw),
    )


def parse_home_plate_official(payload: dict[str, Any]) -> str | None:
    """Return home-plate umpire full name from boxscore or live feed payload."""
    officials = payload.get("officials")
    if not officials:
        game_data = payload.get("gameData") or {}
        officials = game_data.get("officials")
    if not isinstance(officials, list):
        return None

    for entry in officials:
        if not isinstance(entry, dict):
            continue
        official_type = str(entry.get("officialType", "")).strip().lower()
        if official_type not in {"home plate", "home_plate", "homeplate"}:
            continue
        official = entry.get("official") or {}
        name = official.get("fullName") or official.get("name")
        if name and str(name).strip():
            return str(name).strip()
    return None


def fetch_schedule_window(start: date, end: date) -> list[ScheduleGameSlice]:
    """Fetch MLB schedule between start and end (inclusive); never raises."""
    params = {
        "sportId": 1,
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
        "hydrate": "team,venue(location)",
    }
    try:
        payload = _get_json("/schedule", params)
    except requests.RequestException as exc:
        logger.debug("Schedule window fetch failed: %s", exc)
        return []

    slices: list[ScheduleGameSlice] = []
    for day in payload.get("dates") or []:
        if not isinstance(day, dict):
            continue
        for game in day.get("games") or []:
            if not isinstance(game, dict):
                continue
            parsed = parse_schedule_game(game)
            if parsed is not None:
                slices.append(parsed)
    return slices


def fetch_game_officials_payload(game_pk: int) -> dict[str, Any]:
    """Fetch boxscore (or live feed) for umpire crew; returns {} on failure."""
    try:
        return _get_json(f"/game/{game_pk}/boxscore")
    except requests.RequestException:
        pass
    try:
        return _get_json(f"/game/{game_pk}/feed/live")
    except requests.RequestException as exc:
        logger.debug("Officials fetch failed for game %s: %s", game_pk, exc)
        return {}


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_miles = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_miles * math.asin(math.sqrt(a))


def _team_location_for_game(team_abbrev: str, game: ScheduleGameSlice) -> tuple[float | None, float | None, int]:
    """Return (lat, lon, tz_offset) for where team played in this game."""
    if team_abbrev == game.home_abbrev:
        if game.venue_lat is not None and game.venue_lon is not None:
            tz = TEAM_HOME_GEO.get(team_abbrev, (0.0, 0.0, 0))[2]
            return game.venue_lat, game.venue_lon, tz
        geo = TEAM_HOME_GEO.get(team_abbrev)
        if geo:
            return geo[0], geo[1], geo[2]
    elif team_abbrev == game.away_abbrev:
        home_geo = TEAM_HOME_GEO.get(game.home_abbrev)
        if game.venue_lat is not None and game.venue_lon is not None:
            tz = home_geo[2] if home_geo else 0
            return game.venue_lat, game.venue_lon, tz
        if home_geo:
            return home_geo[0], home_geo[1], home_geo[2]
    return None, None, 0


def compute_travel_rest_multiplier(
    *,
    days_rest: int,
    is_back_to_back: bool,
    travel_miles: float,
    travel_zone_delta: int,
) -> tuple[float, str]:
    """Compound travel/rest penalties; default neutral 1.0 when no stressors."""
    multiplier = 1.0
    tags: list[str] = []

    if is_back_to_back or days_rest == 0:
        multiplier *= TRAVEL_REST_B2B_PENALTY
        tags.append("B2B")
    if travel_miles >= TRAVEL_REST_LONG_MILES:
        multiplier *= TRAVEL_REST_LONG_MILES_PENALTY
        tags.append("LongTravel")
    if travel_zone_delta >= TRAVEL_REST_TZ_DELTA_THRESHOLD:
        multiplier *= TRAVEL_REST_TZ_PENALTY
        tags.append("Timezone")

    if not tags:
        return 1.0, "Neutral"
    return multiplier, "+".join(tags)


def _neutral_travel_row(game_id: str, team_id: str) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "team_id": team_id,
        "days_rest": 1,
        "is_back_to_back": False,
        "travel_miles": 0.0,
        "travel_zone_delta": 0,
        "travel_rest_multiplier": 1.0,
        "travel_rest_tag": "Neutral",
    }


def _build_team_history(
    schedule: list[ScheduleGameSlice],
) -> dict[str, list[ScheduleGameSlice]]:
    history: dict[str, list[ScheduleGameSlice]] = {}
    for game in schedule:
        if not game.is_final:
            continue
        for abbrev in (game.home_abbrev, game.away_abbrev):
            history.setdefault(abbrev, []).append(game)
    for abbrev in history:
        history[abbrev].sort(key=lambda g: (g.game_date, g.game_pk))
    return history


def _find_prior_game(
    team_abbrev: str,
    current_date: date,
    current_pk: int,
    history: list[ScheduleGameSlice],
) -> ScheduleGameSlice | None:
    prior: ScheduleGameSlice | None = None
    for game in history:
        if game.game_pk == current_pk:
            break
        if game.game_date <= current_date:
            prior = game
    return prior


def _compute_team_travel_row(
    *,
    game_id: str,
    team_abbrev: str,
    current_date: date,
    current_pk: int,
    current_game: ScheduleGameSlice | None,
    history: list[ScheduleGameSlice],
) -> dict[str, Any]:
    if not team_abbrev or team_abbrev not in TEAM_ABBREV_TO_ID:
        return _neutral_travel_row(game_id, team_abbrev)

    prior = _find_prior_game(team_abbrev, current_date, current_pk, history)
    if prior is None:
        return _neutral_travel_row(game_id, team_abbrev)

    days_rest = max(0, (current_date - prior.game_date).days)
    is_b2b = days_rest == 0

    prior_lat, prior_lon, prior_tz = _team_location_for_game(team_abbrev, prior)
    if current_game is not None:
        curr_lat, curr_lon, curr_tz = _team_location_for_game(team_abbrev, current_game)
    else:
        geo = TEAM_HOME_GEO.get(team_abbrev)
        curr_lat, curr_lon, curr_tz = (geo[0], geo[1], geo[2]) if geo else (None, None, 0)

    travel_miles = 0.0
    if (
        prior_lat is not None
        and prior_lon is not None
        and curr_lat is not None
        and curr_lon is not None
    ):
        travel_miles = _haversine_miles(prior_lat, prior_lon, curr_lat, curr_lon)

    travel_zone_delta = abs(curr_tz - prior_tz)

    multiplier, tag = compute_travel_rest_multiplier(
        days_rest=days_rest,
        is_back_to_back=is_b2b,
        travel_miles=travel_miles,
        travel_zone_delta=travel_zone_delta,
    )

    return {
        "game_id": game_id,
        "team_id": team_abbrev,
        "days_rest": days_rest,
        "is_back_to_back": is_b2b,
        "travel_miles": round(travel_miles, 1),
        "travel_zone_delta": travel_zone_delta,
        "travel_rest_multiplier": multiplier,
        "travel_rest_tag": tag,
    }


def _build_travel_rest_matrix_impl(
    slate_games: pd.DataFrame,
    slate_date: date | None,
    *,
    lookback_hours: int,
) -> pd.DataFrame:
    if slate_games.empty:
        return pd.DataFrame(columns=TRAVEL_REST_COLUMNS)

    slate_day = slate_date
    if slate_day is None and "game_date" in slate_games.columns:
        slate_day = _safe_date(slate_games.iloc[0]["game_date"])
    if slate_day is None:
        slate_day = date.today()

    lookback_days = max(1, math.ceil(lookback_hours / 24))
    window_start = slate_day - timedelta(days=lookback_days)
    schedule = fetch_schedule_window(window_start, slate_day)
    team_history = _build_team_history(schedule)

    schedule_by_pk = {g.game_pk: g for g in schedule}

    rows: list[dict[str, Any]] = []
    for _, game in slate_games.iterrows():
        game_id = str(game["game_id"])
        game_date = _safe_date(game.get("game_date")) or slate_day
        game_pk = _safe_int(game.get("mlb_game_pk"), default=0)
        current_slice = schedule_by_pk.get(game_pk) if game_pk > 0 else None

        for team_col in ("home_team_id", "away_team_id"):
            team_abbrev = str(game[team_col]).upper()
            history = team_history.get(team_abbrev, [])
            rows.append(
                _compute_team_travel_row(
                    game_id=game_id,
                    team_abbrev=team_abbrev,
                    current_date=game_date,
                    current_pk=game_pk,
                    current_game=current_slice,
                    history=history,
                )
            )

    return pd.DataFrame(rows, columns=TRAVEL_REST_COLUMNS)


def build_travel_rest_matrix(
    slate_games: pd.DataFrame,
    slate_date: date | None = None,
    *,
    lookback_hours: int = LOGISTICS_LOOKBACK_HOURS,
) -> pd.DataFrame:
    """Build per team-game travel/rest matrix with strict 1.0 fallbacks."""
    neutral = pd.DataFrame(columns=TRAVEL_REST_COLUMNS)
    if slate_games.empty:
        return neutral

    def _compute() -> pd.DataFrame:
        return _build_travel_rest_matrix_impl(
            slate_games,
            slate_date,
            lookback_hours=lookback_hours,
        )

    return safe_feature_slice(
        "travel_rest_matrix",
        _compute,
        default=pd.DataFrame(
            [
                _neutral_travel_row(str(g["game_id"]), str(g[team_col]))
                for _, g in slate_games.iterrows()
                for team_col in ("home_team_id", "away_team_id")
            ],
            columns=TRAVEL_REST_COLUMNS,
        ),
    )
