from __future__ import annotations

import hashlib
import logging
from datetime import date
from typing import Any

import requests

from baseball_props.logging_utils import get_logger, log_once

logger = get_logger(__name__)

SERP_API_KEY_ENV = "SERP_API_KEY"
SERPAPI_BASE_URL = "https://serpapi.com/search.json"

DEFAULT_MLB_QUERY = "MLB games"
NEUTRAL_GAME_TOTAL = 9.0


class SerpApiError(Exception):
    """SerpApi request failed or quota exhausted."""


def get_serp_api_key() -> str:
    import os

    from baseball_props.data.ingest import PROJECT_ROOT

    key = os.getenv(SERP_API_KEY_ENV)
    if not key:
        raise SerpApiError(
            f"Missing {SERP_API_KEY_ENV}. Set it in {PROJECT_ROOT / '.env'} (see .env.example)."
        )
    return key


def _is_serpapi_auth_or_quota_error(response: requests.Response) -> bool:
    if response.status_code in {401, 402, 429}:
        return True
    return False


def _serpapi_search(q: str, *, timeout: float) -> dict[str, Any]:
    params = {
        "engine": "google",
        "q": q,
        "api_key": get_serp_api_key(),
    }
    response = requests.get(SERPAPI_BASE_URL, params=params, timeout=timeout)
    if _is_serpapi_auth_or_quota_error(response):
        raise SerpApiError(f"SerpApi auth/quota error (HTTP {response.status_code})")
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise SerpApiError(str(exc)) from exc

    payload = response.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise SerpApiError(str(payload["error"]))
    if not isinstance(payload, dict):
        raise SerpApiError("SerpApi returned non-object JSON")
    return payload


def _synthetic_event_id(home_team: str, away_team: str, game_date: str) -> str:
    raw = f"{away_team}@{home_team}:{game_date}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _teams_from_game(game: dict[str, Any]) -> tuple[str, str, str] | None:
    teams = game.get("teams")
    if not isinstance(teams, list):
        return None

    names: list[str] = []
    for team in teams:
        if not isinstance(team, dict):
            continue
        name = str(team.get("name", "")).strip()
        if name:
            names.append(name)

    if len(names) < 2:
        return None

    game_date = str(game.get("date", game.get("status", date.today().isoformat())))
    return names[0], names[1], game_date


def _event_from_teams(
    home_team: str,
    away_team: str,
    *,
    game_date: str,
    commence_time: str | None = None,
) -> dict[str, Any]:
    return {
        "id": _synthetic_event_id(home_team, away_team, game_date),
        "sport_key": "baseball_mlb",
        "home_team": home_team,
        "away_team": away_team,
        "commence_time": commence_time or game_date,
    }


def _collect_serpapi_games(sports_results: dict[str, Any]) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []

    raw_games = sports_results.get("games")
    if isinstance(raw_games, list):
        games.extend(g for g in raw_games if isinstance(g, dict))

    spotlight = sports_results.get("game_spotlight")
    if isinstance(spotlight, dict):
        games.append(spotlight)

    return games


def adapt_serpapi_sports_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Map SerpApi sports_results to Odds API event list."""
    sports_results = payload.get("sports_results")
    if not isinstance(sports_results, dict):
        return []

    events: list[dict[str, Any]] = []
    seen: set[str] = set()

    for game in _collect_serpapi_games(sports_results):
        parsed = _teams_from_game(game)
        if parsed is None:
            spotlight_teams = game.get("teams")
            if isinstance(spotlight_teams, list) and len(spotlight_teams) >= 2:
                away = str(spotlight_teams[0].get("name", "")).strip()
                home = str(spotlight_teams[1].get("name", "")).strip()
                game_date = str(game.get("date", date.today().isoformat()))
                if away and home:
                    parsed = (away, home, game_date)
        if parsed is None:
            continue

        away_team, home_team, game_date = parsed
        event = _event_from_teams(
            home_team,
            away_team,
            game_date=game_date,
            commence_time=str(game.get("date")) if game.get("date") else None,
        )
        if event["id"] in seen:
            continue
        seen.add(event["id"])
        events.append(event)

    return events


def _neutral_bookmakers(home_team: str, away_team: str) -> list[dict[str, Any]]:
    return [
        {
            "key": "serpapi",
            "title": "SerpApi",
            "markets": [
                {
                    "key": "totals",
                    "outcomes": [
                        {"name": "Over", "point": NEUTRAL_GAME_TOTAL, "price": -110},
                        {"name": "Under", "point": NEUTRAL_GAME_TOTAL, "price": -110},
                    ],
                },
                {
                    "key": "spreads",
                    "outcomes": [
                        {"name": home_team, "point": 0.0, "price": -110},
                        {"name": away_team, "point": 0.0, "price": -110},
                    ],
                },
            ],
        }
    ]


def adapt_serpapi_vegas_games(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extend SerpApi events with neutral totals/spreads for Vegas parsing."""
    rows: list[dict[str, Any]] = []
    for event in events:
        home_team = str(event.get("home_team", ""))
        away_team = str(event.get("away_team", ""))
        if not home_team or not away_team:
            continue
        rows.append(
            {
                "id": event["id"],
                "home_team": home_team,
                "away_team": away_team,
                "commence_time": event.get("commence_time"),
                "bookmakers": _neutral_bookmakers(home_team, away_team),
            }
        )
    return rows


def _fetch_mlb_events_from_serpapi(*, timeout: float) -> list[dict[str, Any]]:
    today = date.today().isoformat()
    query = f"{DEFAULT_MLB_QUERY} {today}"
    payload = _serpapi_search(query, timeout=timeout)
    events = adapt_serpapi_sports_results(payload)
    if events:
        return events
    logger.warning("SerpApi returned no MLB events for query %r", query)
    return []


def _fetch_player_props_stale_or_empty(cache_key: str) -> dict[str, Any]:
    from baseball_props.data.ingest import read_odds_cache_stale

    stale = read_odds_cache_stale(cache_key)
    if stale is not None:
        return stale

    parts = cache_key.split(":")
    event_id = parts[2] if len(parts) > 2 else "offline-mock-event-id000000"
    return {
        "id": event_id,
        "sport_key": "baseball_mlb",
        "bookmakers": [],
    }


def fetch_from_serpapi(
    cache_key: str,
    url: str,
    params: dict[str, Any],
    *,
    timeout: float = 20.0,
) -> Any:
    """
    Secondary odds source when The Odds API returns 401.

    SerpApi supplies MLB game metadata via Google sports_results; player prop
    lines are served from stale player_props:v3 cache when available.
    """
    log_once(
        "serpapi_fallback_attempt",
        logger,
        logging.INFO,
        "Attempting SerpApi fallback for this slate.",
    )
    logger.debug("Attempting SerpApi fallback for %s", cache_key)

    if cache_key.startswith("player_props:v3:"):
        return _fetch_player_props_stale_or_empty(cache_key)

    if cache_key.startswith("live_vegas_totals"):
        events = _fetch_mlb_events_from_serpapi(timeout=timeout)
        vegas = adapt_serpapi_vegas_games(events)
        if not vegas:
            raise SerpApiError("No SerpApi Vegas games available")
        return vegas

    if cache_key == "mlb_events":
        events = _fetch_mlb_events_from_serpapi(timeout=timeout)
        if not events:
            raise SerpApiError("No SerpApi MLB events available")
        return events

    raise SerpApiError(f"No SerpApi route for cache_key={cache_key} url={url}")
