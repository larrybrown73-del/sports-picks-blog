from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from baseball_props.config import ADVANCED_SPLIT_METRICS, DEFAULT_SLATE_SOURCE, LEAGUE_AVG, RATE_METRICS
from baseball_props.core.baselines import build_effective_baselines
from baseball_props.data.data_health import DataHealthReport
from baseball_props.data.schemas import SCHEMAS, SlateContext, validate_columns
from baseball_props.logging_utils import get_logger, log_once
from baseball_props.matchups.handedness import (
    BATTER_TO_PITCHER_SPLIT,
    resolve_active_batter_hand_series,
)
from baseball_props.matchups.splits import (
    apply_split_rates,
    resolve_opposing_sp_hand,
    resolve_opposing_sp_id,
)
from baseball_props.types import SlateFrames

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

THE_ODDS_API_KEY_ENV = "THE_ODDS_API_KEY"
THE_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
DEFAULT_SPORTSBOOK_KEYS = "draftkings,fanduel,betmgm"
ODDS_CACHE_DIR = PROJECT_ROOT / "data" / "cache"
ODDS_CACHE_TTL = timedelta(minutes=30)

SlateSource = Literal["mock", "live"]

# MLB / Odds API abbreviation variants → canonical abbrev used for cross-API matching
TEAM_ABBREV_CANONICAL: dict[str, str] = {
    "AZ": "ARI",
    "ARI": "ARI",
    "CHW": "CWS",
    "CWS": "CWS",
    "TBR": "TB",
    "TB": "TB",
    "WSN": "WSH",
    "WSH": "WSH",
    "ATH": "OAK",
    "OAK": "OAK",
    "SFG": "SF",
    "SF": "SF",
    "KCR": "KC",
    "KC": "KC",
    "SDP": "SD",
    "SD": "SD",
}

# Canonical team abbreviations → substring match in The Odds API team names
TEAM_ABBREV_ALIASES: dict[str, str] = {
    "NYY": "yankees",
    "BOS": "red sox",
    "LAD": "dodgers",
    "SF": "giants",
    "ARI": "diamondbacks",
    "ATL": "braves",
    "BAL": "orioles",
    "CHC": "cubs",
    "CWS": "white sox",
    "CIN": "reds",
    "CLE": "guardians",
    "COL": "rockies",
    "DET": "tigers",
    "HOU": "astros",
    "KC": "royals",
    "LAA": "angels",
    "MIA": "marlins",
    "MIL": "brewers",
    "MIN": "twins",
    "NYM": "mets",
    "OAK": "athletics",
    "PHI": "phillies",
    "PIT": "pirates",
    "SD": "padres",
    "SEA": "mariners",
    "STL": "cardinals",
    "TB": "rays",
    "TEX": "rangers",
    "TOR": "blue jays",
    "WSH": "nationals",
}

# Slate abbrev → city label used by TheRundown (and partial Odds API names)
TEAM_ABBREV_CITIES: dict[str, str] = {
    "ARI": "arizona",
    "AZ": "arizona",
    "ATL": "atlanta",
    "BAL": "baltimore",
    "BOS": "boston",
    "CHC": "chicago",
    "CWS": "chicago",
    "CIN": "cincinnati",
    "CLE": "cleveland",
    "COL": "colorado",
    "DET": "detroit",
    "HOU": "houston",
    "KC": "kansas city",
    "LAA": "los angeles",
    "LAD": "los angeles",
    "MIA": "miami",
    "MIL": "milwaukee",
    "MIN": "minnesota",
    "NYM": "new york",
    "NYY": "new york",
    "OAK": "oakland",
    "ATH": "athletics",
    "PHI": "philadelphia",
    "PIT": "pittsburgh",
    "SD": "san diego",
    "SEA": "seattle",
    "SF": "san francisco",
    "STL": "st louis",
    "TB": "tampa bay",
    "TEX": "texas",
    "TOR": "toronto",
    "WSH": "washington",
}


def normalize_team_abbrev(abbrev: str) -> str:
    """Map MLB/Odds/local abbreviation variants to a single canonical abbrev."""
    return TEAM_ABBREV_CANONICAL.get(abbrev.upper(), abbrev.upper())


def get_odds_api_key(*, required: bool = True) -> str | None:
    """
    Return The Odds API key from environment variables.

    Loads `.env` from the project root on import. Set THE_ODDS_API_KEY in `.env`
    or export it in your shell before running live feeds.
    """
    key = os.getenv(THE_ODDS_API_KEY_ENV)
    if not key and required:
        raise EnvironmentError(
            f"Missing {THE_ODDS_API_KEY_ENV}. "
            f"Create {PROJECT_ROOT / '.env'} (see .env.example)."
        )
    if not key:
        logger.warning("No %s found; live odds feeds unavailable", THE_ODDS_API_KEY_ENV)
    return key


RUNDOWN_API_KEY_ENV = "RUNDOWN_API_KEY"
RUNDOWN_BASE_URL = "https://therundown.io/api/v2"


def get_rundown_api_key(*, required: bool = True) -> str | None:
    """Return TheRundown API key from environment variables."""
    key = os.getenv(RUNDOWN_API_KEY_ENV)
    if not key and required:
        raise EnvironmentError(
            f"Missing {RUNDOWN_API_KEY_ENV}. "
            f"Create {PROJECT_ROOT / '.env'} (see .env.example)."
        )
    return key


def _odds_cache_path(cache_key: str) -> Path:
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:24]
    return ODDS_CACHE_DIR / f"{digest}.json"


def read_odds_cache(cache_key: str) -> Any | None:
    """Return cached JSON payload when fetched within ODDS_CACHE_TTL, else None."""
    entry = _load_odds_cache_entry(cache_key)
    if entry is None:
        return None
    try:
        fetched_at = datetime.fromisoformat(entry["fetched_at"])
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - fetched_at
        if age > ODDS_CACHE_TTL:
            logger.debug("Odds cache expired for %s (age %s)", cache_key, age)
            return None
        logger.info("Using cached Odds API data for %s (age %s)", cache_key, age)
        return entry["data"]
    except (KeyError, TypeError, ValueError) as exc:
        logger.debug("Odds cache unreadable for %s: %s", cache_key, exc)
        return None


def _load_odds_cache_entry(cache_key: str) -> dict[str, Any] | None:
    path = _odds_cache_path(cache_key)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Odds cache unreadable for %s: %s", cache_key, exc)
        return None


def read_odds_cache_stale(cache_key: str) -> Any | None:
    """Return cached JSON payload regardless of TTL (offline bypass)."""
    entry = _load_odds_cache_entry(cache_key)
    if entry is None:
        return None
    try:
        log_once(
            "odds_stale_cache",
            logger,
            logging.WARNING,
            "Using stale Odds API cache entries (offline bypass) for this slate.",
        )
        logger.debug("Using stale Odds API cache for %s (offline bypass)", cache_key)
        return entry["data"]
    except KeyError:
        return None


def _is_odds_auth_or_quota_error(response: requests.Response) -> bool:
    """True when the API key is invalid or the account has no remaining quota."""
    if response.status_code in {402, 429}:
        return True
    remaining = response.headers.get("x-requests-remaining")
    if remaining is not None and remaining.isdigit() and int(remaining) == 0:
        return True
    try:
        body = response.json()
        if not isinstance(body, dict):
            return False
        message = str(body.get("message", "")).lower()
        return any(
            token in message
            for token in (
                "quota",
                "exhausted",
                "usage",
                "credit",
                "not authorized",
                "invalid api key",
                "out of requests",
            )
        )
    except (ValueError, AttributeError):
        return False


def _offline_odds_fallback(cache_key: str, url: str) -> Any:
    """Static mock payloads so downstream parsers can run without live API access."""
    log_once(
        "odds_offline_static",
        logger,
        logging.WARNING,
        "Odds API offline; using static fallback payloads for this slate.",
    )
    logger.debug("Odds API offline for %s (%s); using static fallback payload", cache_key, url)
    if cache_key.startswith("live_vegas_totals"):
        return _MOCK_LIVE_VEGAS_GAMES
    if cache_key == "mlb_events":
        return []
    if cache_key.startswith("player_props:v3:"):
        parts = cache_key.split(":")
        event_id = parts[2] if len(parts) > 2 else "offline-mock-event-id000000"
        return {
            "id": event_id,
            "sport_key": "baseball_mlb",
            "bookmakers": [],
        }
    return []


def _resolve_offline_odds_data(cache_key: str, url: str) -> Any:
    stale = read_odds_cache_stale(cache_key)
    if stale is not None:
        return stale
    return _offline_odds_fallback(cache_key, url)


_MOCK_LIVE_VEGAS_GAMES: list[dict[str, Any]] = []


def write_odds_cache(cache_key: str, data: Any) -> None:
    """Persist API JSON to data/cache/ with a UTC timestamp."""
    ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _odds_cache_path(cache_key)
    entry = {
        "cache_key": cache_key,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(entry, handle)


def fetch_rundown_json(
    cache_key: str,
    path: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 20.0,
) -> Any:
    """Fetch TheRundown V2 JSON with header auth and a 30-minute local file cache."""
    cached = read_odds_cache(cache_key)
    if cached is not None:
        logger.debug("[TheRundown] cache hit for %s (skipped HTTP)", cache_key)
        return cached

    url = f"{RUNDOWN_BASE_URL.rstrip('/')}/{path.lstrip('/')}"
    headers = {"X-TheRundown-Key": get_rundown_api_key(required=True) or ""}

    try:
        response = requests.get(url, params=params or {}, headers=headers, timeout=timeout)
        logger.debug(
            "[TheRundown] HTTP %s for %s (key in header: %s)",
            response.status_code,
            path,
            bool(headers.get("X-TheRundown-Key")),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        logger.debug("[TheRundown] request failed for %s: %s (status=%s)", path, exc, status)
        log_once(
            "rundown_request_failed",
            logger,
            logging.WARNING,
            "TheRundown request failed; will attempt Odds API fallback where configured.",
        )
        logger.debug("TheRundown request failed for %s: %s", cache_key, exc)
        raise

    data = response.json()
    write_odds_cache(cache_key, data)
    return data


def fetch_odds_api_json(
    cache_key: str,
    url: str,
    params: dict[str, Any],
    *,
    timeout: float = 20.0,
) -> Any:
    """Fetch Odds API JSON with a 30-minute local file cache."""
    cached = read_odds_cache(cache_key)
    if cached is not None:
        return cached

    try:
        response = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        log_once(
            "odds_request_failed",
            logger,
            logging.WARNING,
            "Odds API request failed; using fallback payloads for this slate.",
        )
        logger.debug("Odds API request failed for %s: %s", cache_key, exc)
        return _resolve_offline_odds_data(cache_key, url)

    if _is_odds_auth_or_quota_error(response) and response.status_code != 401:
        log_once(
            "odds_auth_quota",
            logger,
            logging.WARNING,
            "Odds API auth/quota error; using fallback chain for this slate.",
        )
        logger.debug(
            "Odds API auth/quota error for %s (HTTP %s)",
            cache_key,
            response.status_code,
        )
        return _resolve_offline_odds_data(cache_key, url)

    if response.status_code == 401:
        log_once(
            "odds_api_401_fallback",
            logger,
            logging.WARNING,
            "⚠️ Primary API unauthorized. Utilizing SerpApi and stale cache fallback chain for this slate.",
        )
        logger.debug("Odds API unauthorized (401) for %s; trying fallback chain", cache_key)
        from baseball_props.data.prop_odds import PropOddsError, fetch_from_prop_odds
        from baseball_props.data.serpapi import SerpApiError, fetch_from_serpapi

        for fetcher, label in (
            (fetch_from_serpapi, "SerpApi"),
            (fetch_from_prop_odds, "Prop-Odds"),
        ):
            try:
                data = fetcher(cache_key, url, params, timeout=timeout)
                if cache_key.startswith("player_props:v3:") and not (
                    isinstance(data, dict) and data.get("bookmakers")
                ):
                    continue
                write_odds_cache(cache_key, data)
                return data
            except (SerpApiError, PropOddsError) as exc:
                log_once(
                    f"odds_{label.lower()}_fallback_failed",
                    logger,
                    logging.WARNING,
                    "%s fallback unavailable for this slate; continuing with cache/offline fallbacks.",
                    label,
                )
                logger.debug("%s fallback failed for %s: %s", label, cache_key, exc)
        return _resolve_offline_odds_data(cache_key, url)

    try:
        response.raise_for_status()
    except requests.HTTPError:
        raise

    remaining = response.headers.get("x-requests-remaining")
    if remaining is not None:
        logger.info("The Odds API requests remaining: %s", remaining)
    data = response.json()
    if cache_key.startswith("player_props:v3:") and not (
        isinstance(data, dict) and data.get("bookmakers")
    ):
        stale = read_odds_cache_stale(cache_key)
        if stale is not None:
            return stale
        return _resolve_offline_odds_data(cache_key, url)
    write_odds_cache(cache_key, data)
    return data


def _validate_all(frames: SlateFrames) -> None:
    for name in SCHEMAS:
        if name in frames:
            validate_columns(frames[name], name)


def _normalize_team_name(team_name: str) -> str:
    """Normalize team names for cross-API matching."""
    replacements = {
        "d-backs": "diamondbacks",
        "diamond backs": "diamondbacks",
        "athletics": "athletics",
        "oakland athletics": "athletics",
    }
    normalized = (
        team_name.lower()
        .replace(".", "")
        .replace("-", " ")
        .replace("  ", " ")
        .strip()
    )
    return replacements.get(normalized, normalized)


RUNDOWN_LABEL_ALIASES: dict[str, str] = {
    "oakland": "athletics",
}


def _matches_rundown_team_label(rundown_label: str, odds_team_name: str) -> bool:
    """Match TheRundown city-only labels (e.g. 'Chicago') to Odds API full names."""
    label = _normalize_team_name(rundown_label)
    full = _normalize_team_name(odds_team_name)
    if label == full:
        return True
    if not label:
        return False
    alias = RUNDOWN_LABEL_ALIASES.get(label, label)
    if alias in full:
        return True
    return full.startswith(label + " ") or label in full.split()


def _matches_team_abbrev(team_abbrev: str, api_team_name: str) -> bool:
    """Return True when a slate team abbreviation matches an API team name."""
    canonical = normalize_team_abbrev(team_abbrev)
    normalized = _normalize_team_name(api_team_name)
    alias = TEAM_ABBREV_ALIASES.get(canonical, canonical.lower())
    if alias in normalized:
        return True
    city = TEAM_ABBREV_CITIES.get(canonical)
    if city:
        if normalized == city or normalized.startswith(city + " "):
            return True
        if city in normalized.split():
            return True
    if canonical in {"OAK", "ATH"} and (
        normalized == "oakland"
        or normalized.startswith("oakland ")
        or "athletics" in normalized
    ):
        return True
    return False


def _match_slate_teams_to_events(
    home_abbrev: str,
    away_abbrev: str,
    events_index: pd.DataFrame,
) -> pd.DataFrame:
    """Return event rows where API home/away labels match slate team abbreviations."""
    if events_index.empty:
        return events_index.iloc[0:0]
    mask = events_index.apply(
        lambda row: _matches_team_abbrev(home_abbrev, str(row["home_team"]))
        and _matches_team_abbrev(away_abbrev, str(row["away_team"])),
        axis=1,
    )
    return events_index[mask]


def is_odds_event_id(event_id: str) -> bool:
    """Return True when an ID is a valid external prop event id (not mock slate G###)."""
    value = str(event_id).strip()
    if not value:
        return False
    if value.startswith("G") and value[1:].isdigit():
        return False
    return True


def _build_slate_event_team_index(live_vegas: pd.DataFrame | None = None) -> pd.DataFrame:
    """game_id + team labels for slate→market matching (Vegas rows + full Rundown schedule)."""
    parts: list[pd.DataFrame] = []
    if live_vegas is not None and not live_vegas.empty and "home_team" in live_vegas.columns:
        parts.append(live_vegas[["game_id", "home_team", "away_team"]].copy())
    try:
        from baseball_props.data.therundown import fetch_rundown_event_list

        rundown = fetch_rundown_event_list()
        if not rundown.empty:
            parts.append(
                rundown.rename(columns={"event_id": "game_id"})[
                    ["game_id", "home_team", "away_team"]
                ]
            )
    except Exception as exc:
        logger.debug("TheRundown event list unavailable for slate matching: %s", exc)
    if not parts:
        return pd.DataFrame(columns=["game_id", "home_team", "away_team"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset=["game_id"], keep="first")


def resolve_odds_event_ids_for_slate(
    slate_games: pd.DataFrame,
    live_vegas: pd.DataFrame | None = None,
) -> list[str]:
    """
    Map slate games to real Odds API event hashes by team match.

    Used for player prop fetches — never pass mock IDs (G001) or MLB game_pk values.
    """
    if slate_games.empty:
        return []

    if live_vegas is None:
        live_vegas = fetch_live_vegas_totals()

    events_index = _build_slate_event_team_index(live_vegas)
    if events_index.empty:
        logger.warning("Cannot resolve Odds API event IDs: no event team metadata")
        return []

    event_ids: list[str] = []
    seen: set[str] = set()
    for _, game in slate_games.iterrows():
        home_abbrev = str(game["home_team_id"])
        away_abbrev = str(game["away_team_id"])
        matched = _match_slate_teams_to_events(home_abbrev, away_abbrev, events_index)
        if matched.empty:
            logger.warning(
                "No Odds API event for slate game %s (%s @ %s)",
                game.get("mlb_game_pk", game["game_id"]),
                away_abbrev,
                home_abbrev,
            )
            continue

        event_id = str(matched.iloc[0]["game_id"])
        if not is_odds_event_id(event_id):
            logger.warning("Skipping non-Odds event id for props: %s", event_id)
            continue
        if event_id not in seen:
            event_ids.append(event_id)
            seen.add(event_id)

    logger.info("Resolved %d Odds API event IDs for player props", len(event_ids))
    return event_ids


def _pick_bookmaker(
    bookmakers: list[dict[str, Any]],
    sportsbook_keys: str,
) -> dict[str, Any] | None:
    preferred = [k.strip() for k in sportsbook_keys.split(",") if k.strip()]
    by_key = {str(b.get("key", "")): b for b in bookmakers}
    for key in preferred:
        if key in by_key:
            return by_key[key]
    return bookmakers[0] if bookmakers else None


def _games_to_vegas_rows(
    games: list[dict[str, Any]],
    *,
    sportsbook_keys: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for game in games:
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        game_id = game.get("id", f"{away_team}@{home_team}")

        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            logger.warning("No bookmakers for %s @ %s; skipping", away_team, home_team)
            continue

        book = _pick_bookmaker(bookmakers, sportsbook_keys)
        if book is None:
            continue
        totals_market = next(
            (m for m in book.get("markets", []) if m.get("key") == "totals"),
            None,
        )
        spreads_market = next(
            (m for m in book.get("markets", []) if m.get("key") == "spreads"),
            None,
        )

        if totals_market is None:
            logger.warning("No totals market for %s @ %s; skipping", away_team, home_team)
            continue

        total_point = float(totals_market["outcomes"][0]["point"])
        home_spread = 0.0
        if spreads_market is not None:
            for outcome in spreads_market["outcomes"]:
                if _normalize_team_name(outcome.get("name", "")) == _normalize_team_name(
                    home_team
                ):
                    home_spread = float(outcome["point"])
                    break

        home_implied = (total_point - home_spread) / 2.0
        away_implied = (total_point + home_spread) / 2.0

        rows.append(
            {
                "game_id": game_id,
                "home_implied_runs": round(home_implied, 2),
                "away_implied_runs": round(away_implied, 2),
                "game_total": round(total_point, 2),
                "home_team": home_team,
                "away_team": away_team,
            }
        )
    return rows


def fetch_live_vegas_totals(
    *,
    sportsbook_keys: str = DEFAULT_SPORTSBOOK_KEYS,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """
    Fetch MLB game totals (TheRundown primary, Odds API fallback).

    Uses the first preferred bookmaker per game. Implied team runs are derived
    from the game total and spread when present; otherwise split evenly.
    """
    games: list[dict[str, Any]] | Any = []
    try:
        from baseball_props.data.therundown import (
            CORE_MARKET_IDS,
            adapt_rundown_vegas_games,
            fetch_rundown_mlb_events,
        )

        slate_date = date.today()
        cache_key = f"rundown_vegas:{slate_date.isoformat()}:{CORE_MARKET_IDS}:{sportsbook_keys}"
        payload = fetch_rundown_mlb_events(
            slate_date,
            market_ids=CORE_MARKET_IDS,
            cache_key=cache_key,
            timeout=timeout,
        )
        games = adapt_rundown_vegas_games(payload)
        event_count = len(payload.get("events", [])) if isinstance(payload, dict) else 0
        logger.debug(
            "[TheRundown] Vegas fetch: %d raw events -> %d adapted games with totals",
            event_count,
            len(games),
        )
        if games:
            logger.info("Fetched live Vegas totals from TheRundown for %d games", len(games))
    except Exception as exc:
        logger.warning("TheRundown vegas fetch failed (%s); falling back to Odds API", exc)
        games = []

    if not games:
        api_key = get_odds_api_key(required=False)
        if not api_key:
            games = _resolve_offline_odds_data(
                f"live_vegas_totals:{sportsbook_keys}",
                THE_ODDS_API_URL,
            )
        else:
            params = {
                "apiKey": api_key,
                "regions": "us",
                "markets": "spreads,totals",
                "oddsFormat": "american",
                "bookmakers": sportsbook_keys,
            }
            cache_key = f"live_vegas_totals:{sportsbook_keys}"
            games = fetch_odds_api_json(cache_key, THE_ODDS_API_URL, params, timeout=timeout)

    if not isinstance(games, list):
        games = []

    rows = _games_to_vegas_rows(games, sportsbook_keys=sportsbook_keys)

    if not rows:
        logger.warning("Live odds fetch returned no usable games")
        return pd.DataFrame(
            columns=["game_id", "home_implied_runs", "away_implied_runs", "game_total"]
        )

    logger.info("Fetched live Vegas totals for %d games", len(rows))
    return pd.DataFrame(rows)


def merge_live_odds_game_ids(
    slate_games: pd.DataFrame,
    lineups: pd.DataFrame,
    live_vegas: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Match slate rows to Odds API game_id hashes by team; update lineups in sync."""
    games = slate_games.copy()
    lu = lineups.copy()
    vegas_rows: list[dict[str, object]] = []
    events_index = _build_slate_event_team_index(live_vegas)
    vegas_by_id: dict[str, pd.Series] = {}
    if not live_vegas.empty:
        for _, row in live_vegas.iterrows():
            vegas_by_id[str(row["game_id"])] = row

    for idx, game in games.iterrows():
        home_abbrev = str(game["home_team_id"])
        away_abbrev = str(game["away_team_id"])
        matched = _match_slate_teams_to_events(home_abbrev, away_abbrev, events_index)
        if matched.empty:
            logger.warning(
                "No Odds API match for MLB game %s (%s @ %s)",
                game.get("mlb_game_pk", game["game_id"]),
                away_abbrev,
                home_abbrev,
            )
            continue

        live_row = matched.iloc[0]
        old_id = str(game["game_id"])
        new_id = str(live_row["game_id"])
        games.at[idx, "game_id"] = new_id
        lu.loc[lu["game_id"] == old_id, "game_id"] = new_id
        vegas_row = vegas_by_id.get(new_id)
        if vegas_row is not None:
            vegas_rows.append(
                {
                    "game_id": new_id,
                    "home_implied_runs": vegas_row["home_implied_runs"],
                    "away_implied_runs": vegas_row["away_implied_runs"],
                    "game_total": vegas_row["game_total"],
                }
            )
        else:
            vegas_rows.append(
                {
                    "game_id": new_id,
                    "home_implied_runs": 4.5,
                    "away_implied_runs": 4.5,
                    "game_total": 9.0,
                }
            )

    vegas = pd.DataFrame(vegas_rows)
    return games.reset_index(drop=True), lu.reset_index(drop=True), vegas


def build_live_slate_frames(slate_date: date | None = None) -> SlateFrames:
    """Build full slate from MLB schedule, pybaseball stats, and live Vegas odds."""
    from baseball_props.data.data_health import DataHealthReport
    from baseball_props.data.mlb_live import build_slate_from_schedule
    from baseball_props.data.statcast_feed import (
        build_park_weather_stub,
        build_pitcher_platoon_splits,
        build_pitcher_tendencies,
        build_player_baselines_and_splits,
        build_team_pitching_stub,
    )

    slate_games, lineups, pitcher_names, lineup_source_counts = build_slate_from_schedule(
        slate_date
    )
    if slate_games.empty:
        logger.error("No live slate games available for %s", slate_date or date.today())
        raise ValueError(
            "No live slate games available for the requested date. "
            "Check the schedule or try a different --date."
        )

    from baseball_props.data.probable_pitcher_overrides import apply_probable_pitcher_overrides

    slate_day = slate_date or date.today()
    slate_games, pitcher_names = apply_probable_pitcher_overrides(
        slate_games,
        pitcher_names,
        slate_date=slate_day.isoformat(),
    )

    health = DataHealthReport()
    player_ids = lineups["player_id"].astype(str).unique().tolist()
    player_baselines, matchup_splits = build_player_baselines_and_splits(
        player_ids, data_health=health
    )

    team_ids = pd.unique(
        pd.concat([slate_games["home_team_id"], slate_games["away_team_id"]])
    ).tolist()
    team_pitching = build_team_pitching_stub([str(t) for t in team_ids])
    park_weather = build_park_weather_stub(
        slate_games["park_id"].astype(str).unique().tolist()
    )

    live_vegas = fetch_live_vegas_totals()
    odds_event_ids = resolve_odds_event_ids_for_slate(slate_games, live_vegas)
    slate_games, lineups, vegas_totals = merge_live_odds_game_ids(
        slate_games, lineups, live_vegas
    )
    if vegas_totals.empty:
        logger.warning("Using MLB game_pk as game_id; no matched Vegas totals")
        vegas_totals = pd.DataFrame(
            [
                {
                    "game_id": r["game_id"],
                    "home_implied_runs": 4.5,
                    "away_implied_runs": 4.5,
                    "game_total": 9.0,
                }
                for _, r in slate_games.iterrows()
            ]
        )

    pitcher_tendencies = build_pitcher_tendencies(slate_games, pitcher_names)

    sp_ids: list[str] = []
    for col in ("sp_home_id", "sp_away_id"):
        for raw in slate_games[col].astype(str).tolist():
            norm = str(raw).strip()
            if norm and norm != "nan":
                sp_ids.append(norm)
    pitcher_platoon_splits = build_pitcher_platoon_splits(sp_ids)

    slate_games_out = slate_games[[c for c in SCHEMAS["slate_games"] if c in slate_games.columns]]

    return SlateFrames(
        slate_games=slate_games_out,
        player_baselines=player_baselines,
        matchup_splits=matchup_splits,
        pitcher_platoon_splits=pitcher_platoon_splits,
        team_pitching=team_pitching,
        park_weather=park_weather,
        vegas_totals=vegas_totals,
        lineups=lineups,
        pitcher_tendencies=pitcher_tendencies,
        odds_event_ids=odds_event_ids,
        lineup_source_counts=lineup_source_counts,
        data_health=health,
    )


def load_slate_frames(
    source: SlateSource = DEFAULT_SLATE_SOURCE,
    *,
    slate_date: date | None = None,
) -> SlateFrames:
    """
    Load slate input DataFrames from mock data or live feeds.

    `mock` — fully synthetic slate for development.
    `live` — MLB schedule lineups + pybaseball stats + Odds API Vegas totals.
    """
    from baseball_props.data.mock_slate import build_mock_slate

    if source == "mock":
        return build_mock_slate()

    if source == "live":
        return build_live_slate_frames(slate_date)

    raise ValueError(f"Unknown slate source: {source}")


def build_slate_context(
    frames: SlateFrames,
    *,
    injury_lookup: dict | None = None,
    slate_date: date | None = None,
    data_health: DataHealthReport | None = None,
) -> SlateContext:
    """
    Validate, merge, and return a player-game frame ready for projection.

    One row per (game_id, player_id) batter in the slate.
    """
    _validate_all(frames)

    games = frames["slate_games"]
    lineups = frames["lineups"]
    baselines = frames["player_baselines"]
    splits = frames["matchup_splits"]
    pitching = frames["team_pitching"]
    park_weather = frames["park_weather"]

    effective = build_effective_baselines(baselines, RATE_METRICS)
    effective_advanced = build_effective_baselines(baselines, ADVANCED_SPLIT_METRICS)

    player_games = lineups.merge(games, on="game_id", how="left")
    player_games = player_games.merge(effective, on="player_id", how="left")
    player_games = player_games.merge(
        effective_advanced,
        on="player_id",
        how="left",
        suffixes=("", "_adv"),
    )

    player_games = resolve_opposing_sp_hand(player_games)
    player_games = resolve_opposing_sp_id(player_games)
    if "bat_hand" not in player_games.columns:
        player_games["bat_hand"] = "R"
    player_games["batter_hand_active"] = resolve_active_batter_hand_series(
        player_games["bat_hand"],
        player_games["opp_sp_hand"],
    )
    player_games["pitcher_split_key"] = player_games["batter_hand_active"].map(
        BATTER_TO_PITCHER_SPLIT
    ).fillna("vs_rhb")

    player_games, split_fallbacks = apply_split_rates(player_games, splits, RATE_METRICS)
    player_games, advanced_fallbacks = apply_split_rates(
        player_games, splits, ADVANCED_SPLIT_METRICS
    )
    player_games = player_games.rename(columns={"matchup_wrc_plus": "wrc_plus_split"})
    split_fallbacks += advanced_fallbacks

    player_games["opp_team_id"] = np.where(
        player_games["team_id"] == player_games["away_team_id"],
        player_games["home_team_id"],
        player_games["away_team_id"],
    )

    sp_pitching = pitching[pitching["role"] == "sp"].rename(
        columns={
            "team_id": "opp_team_id",
            "woba_allowed": "opp_sp_woba_allowed",
            "iso_allowed": "opp_sp_iso_allowed",
            "k_pct": "opp_sp_k_pct",
            "bb_pct": "opp_sp_bb_pct",
        }
    )
    player_games = player_games.merge(
        sp_pitching[
            [
                "opp_team_id",
                "opp_sp_woba_allowed",
                "opp_sp_iso_allowed",
                "opp_sp_k_pct",
                "opp_sp_bb_pct",
            ]
        ],
        on="opp_team_id",
        how="left",
    )

    opp_col_map = {
        "woba": "opp_sp_woba_allowed",
        "iso": "opp_sp_iso_allowed",
        "k_pct": "opp_sp_k_pct",
        "bb_pct": "opp_sp_bb_pct",
    }
    missing_pitch = player_games["opp_sp_woba_allowed"].isna()
    if missing_pitch.any():
        logger.warning(
            "Pitching fallback: %d rows missing opponent SP metrics, using league average",
            int(missing_pitch.sum()),
        )
    for metric, opp_col in opp_col_map.items():
        player_games[f"league_{metric}"] = LEAGUE_AVG[metric]
        player_games[opp_col] = player_games[opp_col].fillna(LEAGUE_AVG[metric])

    pitcher_platoon = frames.get("pitcher_platoon_splits")
    platoon_pitch_fallbacks = 0
    if pitcher_platoon is not None and not pitcher_platoon.empty:
        pps = pitcher_platoon.rename(columns={"split": "pitcher_split_key"}).rename(
            columns={
                "woba_allowed": "opp_sp_platoon_woba_allowed",
                "iso_allowed": "opp_sp_platoon_iso_allowed",
                "k_pct": "opp_sp_platoon_k_pct",
                "bb_pct": "opp_sp_platoon_bb_pct",
                "bf": "opp_sp_platoon_bf",
            }
        )
        merge_cols = [
            "pitcher_id",
            "pitcher_split_key",
            "opp_sp_platoon_woba_allowed",
            "opp_sp_platoon_iso_allowed",
            "opp_sp_platoon_k_pct",
            "opp_sp_platoon_bb_pct",
            "opp_sp_platoon_bf",
        ]
        player_games = player_games.merge(
            pps[merge_cols],
            left_on=["opp_sp_id", "pitcher_split_key"],
            right_on=["pitcher_id", "pitcher_split_key"],
            how="left",
        )
        player_games = player_games.drop(columns=["pitcher_id"], errors="ignore")

        missing_platoon = player_games["opp_sp_platoon_woba_allowed"].isna()
        platoon_pitch_fallbacks = int(missing_platoon.sum())
        if platoon_pitch_fallbacks:
            logger.warning(
                "Pitcher platoon fallback: %d rows missing individual SP split, using team SP stub",
                platoon_pitch_fallbacks,
            )
        player_games["opp_sp_woba_allowed"] = player_games[
            "opp_sp_platoon_woba_allowed"
        ].fillna(player_games["opp_sp_woba_allowed"])
        player_games["opp_sp_iso_allowed"] = player_games[
            "opp_sp_platoon_iso_allowed"
        ].fillna(player_games["opp_sp_iso_allowed"])
        player_games["opp_sp_k_pct"] = player_games["opp_sp_platoon_k_pct"].fillna(
            player_games["opp_sp_k_pct"]
        )
        player_games["opp_sp_bb_pct"] = player_games["opp_sp_platoon_bb_pct"].fillna(
            player_games["opp_sp_bb_pct"]
        )

    if "season_pa" in baselines.columns:
        player_games = player_games.merge(
            baselines[["player_id", "season_pa"]],
            on="player_id",
            how="left",
        )

    player_games = player_games.merge(park_weather, on="park_id", how="left")

    from baseball_props.data.bullpen_fatigue import build_team_bullpen_fatigue_table
    from baseball_props.data.data_health import DataHealthReport, safe_feature_slice

    health = data_health or DataHealthReport()
    if frames.get("data_health") is not None:
        health.merge(frames["data_health"])

    opp_teams = player_games["opp_team_id"].astype(str).unique().tolist()
    fatigue_table = safe_feature_slice(
        "bullpen_fatigue",
        lambda: build_team_bullpen_fatigue_table(opp_teams),
        default=pd.DataFrame(
            columns=[
                "opp_team_id",
                "opp_bullpen_fatigue_score",
                "opp_bullpen_fatigue_status",
            ]
        ),
        report=health,
        empty_check=lambda df: df.empty,
    )
    if not fatigue_table.empty:
        player_games = player_games.merge(fatigue_table, on="opp_team_id", how="left")
    else:
        player_games["opp_bullpen_fatigue_score"] = 0.35
        player_games["opp_bullpen_fatigue_status"] = "Moderate"

    player_games["opp_bullpen_fatigue_score"] = player_games[
        "opp_bullpen_fatigue_score"
    ].fillna(0.35)
    player_games["opp_bullpen_fatigue_status"] = player_games[
        "opp_bullpen_fatigue_status"
    ].fillna("Moderate")

    from baseball_props.data.game_context import (
        build_lineup_absence_penalties,
        build_travel_rest_table,
        build_umpire_table,
    )

    slate_day = slate_date
    if slate_day is None and "game_date" in games.columns and not games.empty:
        try:
            slate_day = date.fromisoformat(str(games.iloc[0]["game_date"])[:10])
        except ValueError:
            slate_day = None

    travel_rest = build_travel_rest_table(games, slate_day)
    umpire_table = build_umpire_table(games)
    absence_table = build_lineup_absence_penalties(lineups, injury_lookup)

    if not travel_rest.empty:
        player_games = player_games.merge(
            travel_rest, on=["game_id", "team_id"], how="left", suffixes=("", "_tr")
        )
    if not umpire_table.empty:
        player_games = player_games.merge(umpire_table, on="game_id", how="left")
    if not absence_table.empty:
        player_games = player_games.merge(
            absence_table, on=["game_id", "team_id"], how="left", suffixes=("", "_abs")
        )

    if injury_lookup:
        from baseball_props.data.injuries import injury_rust_multiplier, lookup_injury

        if "player_name" in player_games.columns:
            injury_records = player_games["player_name"].map(
                lambda name: lookup_injury(str(name), injury_lookup)
            )
            player_games["injury_status"] = injury_records.map(
                lambda rec: rec.get("status") if rec else None
            )
            player_games["injury_multiplier"] = injury_records.map(injury_rust_multiplier)
        else:
            player_games["injury_status"] = None
            player_games["injury_multiplier"] = 1.0

    fallback_counts = {
        "split_fallbacks": split_fallbacks,
        "pitcher_platoon_fallbacks": platoon_pitch_fallbacks,
        **health.fallback_counts,
    }

    logger.info(
        "Built slate context: %d player-games across %d games",
        len(player_games),
        len(games),
    )

    return SlateContext(
        player_games=player_games,
        games=games,
        fallback_counts=fallback_counts,
        data_health=health,
    )
