from __future__ import annotations

import logging
from datetime import date
from typing import Any

import requests

from baseball_props.logging_utils import get_logger, log_once, log_once

logger = get_logger(__name__)

PROP_ODDS_API_KEY_ENV = "PROP_ODDS_API_KEY"
PROP_ODDS_BASE_URL = "https://api.prop-odds.com/v1/"

# The Odds API market key → Prop-Odds market key (MLB)
PROP_ODDS_MARKET_MAP: dict[str, str] = {
    "batter_total_bases": "batter_bases_over_under",
    "pitcher_outs": "pitcher_outs_over_under",
    "totals": "total_over_under",
    "spreads": "spread",
    "h2h": "moneyline",
}

ODDS_API_MARKET_FROM_PROP: dict[str, str] = {
    v: k for k, v in PROP_ODDS_MARKET_MAP.items()
}

_PREFERRED_BOOKS = frozenset({"draftkings", "fanduel", "betmgm"})


class PropOddsError(Exception):
    """Prop-Odds request failed or quota exhausted."""


def get_prop_odds_api_key() -> str:
    import os

    from baseball_props.data.ingest import PROJECT_ROOT

    key = os.getenv(PROP_ODDS_API_KEY_ENV)
    if not key:
        env_path = PROJECT_ROOT / ".env"
        raise PropOddsError(
            f"Missing {PROP_ODDS_API_KEY_ENV}. Set it in {env_path} (see .env.example)."
        )
    return key


def _prop_odds_url(path: str) -> str:
    return f"{PROP_ODDS_BASE_URL.rstrip('/')}/{path.lstrip('/')}"


def _prop_odds_headers() -> dict[str, str]:
    return {"X-Api-Key": get_prop_odds_api_key()}


def _is_prop_odds_auth_or_quota_error(response: requests.Response) -> bool:
    if response.status_code in {401, 402, 429}:
        return True
    remaining = response.headers.get("x-requests-remaining")
    if remaining is not None and remaining.isdigit() and int(remaining) == 0:
        return True
    return False


def _prop_odds_get(path: str, *, params: dict[str, Any] | None = None, timeout: float) -> Any:
    response = requests.get(
        _prop_odds_url(path),
        headers=_prop_odds_headers(),
        params=params,
        timeout=timeout,
    )
    if _is_prop_odds_auth_or_quota_error(response):
        raise PropOddsError(f"Prop-Odds auth/quota error (HTTP {response.status_code})")
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise PropOddsError(str(exc)) from exc
    return response.json()


def _parse_player_side_name(raw_name: str) -> tuple[str, str | None]:
    """Parse 'Over - Aaron Judge' / 'Under - Aaron Judge' into side + player."""
    text = str(raw_name).strip()
    for side in ("Over", "Under"):
        prefix = f"{side} - "
        if text.startswith(prefix):
            return side, text[len(prefix) :].strip()
        if text.startswith(side):
            rest = text[len(side) :].strip(" -")
            return side, rest or None
    return text, None


def adapt_prop_odds_market_payload(
    payload: dict[str, Any],
    *,
    odds_api_market_key: str,
) -> list[dict[str, Any]]:
    """Map one Prop-Odds market response into Odds API bookmaker dicts."""
    bookmakers: list[dict[str, Any]] = []
    for sportsbook in payload.get("sportsbooks", []):
        book_key = str(sportsbook.get("bookie_key", sportsbook.get("key", ""))).lower()
        if _PREFERRED_BOOKS and book_key not in _PREFERRED_BOOKS:
            continue
        market = sportsbook.get("market") or {}
        outcomes: list[dict[str, Any]] = []
        for outcome in market.get("outcomes", []):
            side, player = _parse_player_side_name(str(outcome.get("name", "")))
            if odds_api_market_key.startswith("batter_") or odds_api_market_key.startswith(
                "pitcher_"
            ):
                if player is None:
                    player = outcome.get("description") or outcome.get("participant")
                if not player or side not in {"Over", "Under"}:
                    continue
                row = {
                    "name": side,
                    "description": str(player).strip(),
                    "price": outcome.get("price", outcome.get("odds")),
                    "point": outcome.get("point", outcome.get("line")),
                }
            else:
                row = {
                    "name": side,
                    "price": outcome.get("price", outcome.get("odds")),
                    "point": outcome.get("point", outcome.get("line")),
                }
            if row.get("point") is not None or row.get("price") is not None:
                outcomes.append(row)
        if outcomes:
            bookmakers.append(
                {
                    "key": book_key,
                    "title": sportsbook.get("title", book_key),
                    "markets": [{"key": odds_api_market_key, "outcomes": outcomes}],
                }
            )
    return bookmakers


def _merge_bookmakers(bookmaker_groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in bookmaker_groups:
        for book in group:
            key = book["key"]
            if key not in merged:
                merged[key] = {"key": key, "title": book.get("title", key), "markets": []}
            merged[key]["markets"].extend(book.get("markets", []))
    return list(merged.values())


def adapt_prop_odds_games_list(payload: Any) -> list[dict[str, Any]]:
    """Map Prop-Odds games list into Odds API event list (no odds yet)."""
    games = payload.get("games", payload) if isinstance(payload, dict) else payload
    if not isinstance(games, list):
        return []
    rows: list[dict[str, Any]] = []
    for game in games:
        if not isinstance(game, dict):
            continue
        rows.append(
            {
                "id": game.get("game_id", game.get("id")),
                "sport_key": "baseball_mlb",
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "commence_time": game.get("start_timestamp", game.get("commence_time")),
            }
        )
    return rows


def adapt_prop_odds_event_payload(
    game_meta: dict[str, Any],
    market_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a single-event Odds API-shaped dict for player_props:v3 cache."""
    bookmaker_groups: list[list[dict[str, Any]]] = []
    for odds_api_key, payload in market_payloads.items():
        bookmaker_groups.append(
            adapt_prop_odds_market_payload(payload, odds_api_market_key=odds_api_key)
        )
    return {
        "id": game_meta.get("game_id", game_meta.get("id")),
        "sport_key": "baseball_mlb",
        "home_team": game_meta.get("home_team"),
        "away_team": game_meta.get("away_team"),
        "commence_time": game_meta.get("start_timestamp", game_meta.get("commence_time")),
        "bookmakers": _merge_bookmakers(bookmaker_groups),
    }


def adapt_prop_odds_vegas_game(
    game_meta: dict[str, Any],
    market_payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build one Odds API game object with totals/spreads for live_vegas_totals."""
    event = adapt_prop_odds_event_payload(game_meta, market_payloads)
    return {
        "id": event["id"],
        "home_team": event.get("home_team"),
        "away_team": event.get("away_team"),
        "commence_time": event.get("commence_time"),
        "bookmakers": event.get("bookmakers", []),
    }


def _fetch_mlb_games(*, timeout: float) -> list[dict[str, Any]]:
    today = date.today().isoformat()
    payload = _prop_odds_get(
        "games/mlb",
        params={"date": today, "tz": "America/New_York"},
        timeout=timeout,
    )
    games_raw = payload.get("games", [])
    if not isinstance(games_raw, list):
        return []
    return [g for g in games_raw if isinstance(g, dict)]


def _fetch_market_odds(game_id: str, prop_market: str, *, timeout: float) -> dict[str, Any]:
    return _prop_odds_get(f"odds/{game_id}/{prop_market}", timeout=timeout)


def _resolve_prop_odds_game_id(event_id: str, games: list[dict[str, Any]]) -> str | None:
    from baseball_props.data.ingest import DEFAULT_SPORTSBOOK_KEYS, read_odds_cache_stale

    for game in games:
        gid = str(game.get("game_id", game.get("id", "")))
        if gid == event_id:
            return gid

    vegas_cache = read_odds_cache_stale(f"live_vegas_totals:{DEFAULT_SPORTSBOOK_KEYS}")
    if isinstance(vegas_cache, list):
        for row in vegas_cache:
            if str(row.get("id")) != event_id:
                continue
            home = str(row.get("home_team", "")).lower()
            away = str(row.get("away_team", "")).lower()
            for game in games:
                gh = str(game.get("home_team", "")).lower()
                ga = str(game.get("away_team", "")).lower()
                if gh == home and ga == away:
                    return str(game.get("game_id", game.get("id")))

    if len(games) == 1:
        return str(games[0].get("game_id", games[0].get("id")))

    return None


def _fetch_events_from_prop_odds(*, timeout: float) -> list[dict[str, Any]]:
    games = _fetch_mlb_games(timeout=timeout)
    return adapt_prop_odds_games_list({"games": games})


def _fetch_vegas_from_prop_odds(*, timeout: float) -> list[dict[str, Any]]:
    games = _fetch_mlb_games(timeout=timeout)
    results: list[dict[str, Any]] = []
    for game in games:
        gid = str(game.get("game_id", game.get("id", "")))
        if not gid:
            continue
        market_payloads: dict[str, dict[str, Any]] = {}
        for odds_key, prop_key in (("totals", "total_over_under"), ("spreads", "spread")):
            try:
                market_payloads[odds_key] = _fetch_market_odds(gid, prop_key, timeout=timeout)
            except PropOddsError as exc:
                logger.debug("Prop-Odds %s unavailable for game %s: %s", prop_key, gid, exc)
        if not market_payloads:
            continue
        results.append(adapt_prop_odds_vegas_game(game, market_payloads))
    return results


def _fetch_event_props_from_prop_odds(
    event_id: str,
    markets_csv: str,
    _sportsbook_keys: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    games = _fetch_mlb_games(timeout=timeout)
    game_id = _resolve_prop_odds_game_id(event_id, games)
    if game_id is None:
        raise PropOddsError(f"No Prop-Odds game match for event id {event_id}")

    game_meta = next(
        (g for g in games if str(g.get("game_id", g.get("id"))) == game_id),
        {"game_id": game_id},
    )

    market_payloads: dict[str, dict[str, Any]] = {}
    for odds_market in markets_csv.split(","):
        odds_market = odds_market.strip()
        if not odds_market:
            continue
        prop_market = PROP_ODDS_MARKET_MAP.get(odds_market)
        if prop_market is None:
            logger.warning("No Prop-Odds mapping for market %s", odds_market)
            continue
        market_payloads[odds_market] = _fetch_market_odds(game_id, prop_market, timeout=timeout)

    if not market_payloads:
        raise PropOddsError(f"No Prop-Odds prop markets fetched for game {game_id}")

    return adapt_prop_odds_event_payload(game_meta, market_payloads)


def fetch_from_prop_odds(
    cache_key: str,
    url: str,
    params: dict[str, Any],
    *,
    timeout: float = 20.0,
) -> Any:
    """
    Secondary odds source when The Odds API returns 401.

    Returns payloads in the same shape expected by downstream parsers and cache keys.
    """
    log_once(
        "prop_odds_fallback_attempt",
        logger,
        logging.INFO,
        "Attempting Prop-Odds fallback for this slate.",
    )
    logger.debug("Attempting Prop-Odds fallback for %s", cache_key)

    if cache_key.startswith("live_vegas_totals"):
        return _fetch_vegas_from_prop_odds(timeout=timeout)

    if cache_key == "mlb_events":
        return _fetch_events_from_prop_odds(timeout=timeout)

    if cache_key.startswith("player_props:v3:"):
        parts = cache_key.split(":")
        if len(parts) < 5:
            raise PropOddsError(f"Unrecognized player props cache key: {cache_key}")
        event_id = parts[2]
        markets = parts[3]
        sportsbooks = parts[4]
        return _fetch_event_props_from_prop_odds(
            event_id,
            markets,
            sportsbooks,
            timeout=timeout,
        )

    raise PropOddsError(f"No Prop-Odds route for cache_key={cache_key} url={url}")
