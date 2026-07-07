from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from baseball_props.data.ingest import (
    DEFAULT_SPORTSBOOK_KEYS,
    read_odds_cache,
    write_odds_cache,
)
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)

RUNDOWN_API_KEY_ENV = "RUNDOWN_API_KEY"
RUNDOWN_BASE_URL = "https://therundown.io/api/v2"
MLB_SPORT_ID = 3
DEFAULT_AFFILIATE_IDS = "19,22,23"
CORE_MARKET_IDS = "1,2,3"
RUNDOWN_UTC_OFFSET = 300
OFF_BOARD_SENTINEL = 0.0001

AFFILIATE_TO_BOOK: dict[str, str] = {
    "19": "draftkings",
    "22": "betmgm",
    "23": "fanduel",
}

RUNDOWN_MARKET_TO_ODDS_KEY: dict[int, str] = {
    1: "h2h",
    2: "spreads",
    3: "totals",
}

INTERNAL_PROP_MARKET_PATTERNS: list[tuple[str, str]] = [
    ("pitching_outs", "pitcher_outs"),
    ("pitcher outs", "pitcher_outs"),
    ("outs recorded", "pitcher_outs"),
    ("total bases", "batter_total_bases"),
]

# Global /markets catalog IDs used when date-scoped discovery lists only core markets.
STATIC_MLB_PROP_MARKET_IDS: dict[str, int] = {
    "batter_total_bases": 11,
    "pitcher_outs": 973,
}


class RundownError(Exception):
    """TheRundown request or adaptation failed."""


def _fetch_rundown_json(
    cache_key: str,
    path: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: float = 20.0,
) -> Any:
    from baseball_props.data.ingest import fetch_rundown_json

    return fetch_rundown_json(cache_key, path, params, timeout=timeout)


def _is_off_board(value: Any) -> bool:
    try:
        return float(value) == OFF_BOARD_SENTINEL
    except (TypeError, ValueError):
        return False


def _rundown_url(path: str) -> str:
    return f"{RUNDOWN_BASE_URL.rstrip('/')}/{path.lstrip('/')}"


def _events_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        events = payload.get("events", [])
        return [e for e in events if isinstance(e, dict)]
    if isinstance(payload, list):
        return [e for e in payload if isinstance(e, dict)]
    return []


def _team_names(event: dict[str, Any]) -> tuple[str, str]:
    teams = event.get("teams") or []
    if len(teams) >= 2:
        away = str(teams[0].get("name", teams[0].get("team_name", "")))
        home = str(teams[1].get("name", teams[1].get("team_name", "")))
        return away, home
    return str(event.get("away_team", "")), str(event.get("home_team", ""))


def _event_id(event: dict[str, Any]) -> str:
    return str(event.get("event_id", event.get("id", "")))


def _participant_side(participant: dict[str, Any]) -> str | None:
    ptype = str(participant.get("type", "")).upper()
    name = str(participant.get("name", "")).strip()
    if ptype == "TYPE_OVER" or name.endswith(" Over") or name == "Over":
        return "Over"
    if ptype == "TYPE_UNDER" or name.endswith(" Under") or name == "Under":
        return "Under"
    return None


def _strip_player_suffix(name: str) -> str:
    for suffix in (" Over", " Under"):
        if name.endswith(suffix):
            return name[: -len(suffix)].strip()
    return name.strip()


def _map_prop_market_name(name: str) -> str | None:
    lowered = name.lower().strip()
    if lowered in {"bases", "player bases"}:
        return "batter_total_bases"
    for pattern, internal in INTERNAL_PROP_MARKET_PATTERNS:
        if pattern in lowered:
            if internal == "batter_total_bases" and "team" in lowered:
                continue
            return internal
    return None


def _validate_prop_market_cache(cached: Any) -> dict[str, int] | None:
    if not isinstance(cached, dict) or not cached:
        return None
    validated: dict[str, int] = {}
    for key, value in cached.items():
        try:
            validated[str(key)] = int(value)
        except (TypeError, ValueError):
            return None
    return validated or None


def discover_mlb_prop_market_ids(
    slate_date: date | None = None,
    *,
    timeout: float = 20.0,
) -> dict[str, int]:
    """Return internal market key -> TheRundown market_id for the slate date."""
    ref = slate_date or date.today()
    cache_key = f"rundown_mlb_markets:v2:{ref.isoformat()}"
    validated = _validate_prop_market_cache(read_odds_cache(cache_key))
    if validated:
        return validated

    path = f"/sports/{MLB_SPORT_ID}/markets/{ref.isoformat()}"
    params = {"offset": str(RUNDOWN_UTC_OFFSET)}
    payload = _fetch_rundown_json(cache_key, path, params, timeout=timeout)

    sport_markets = payload.get(str(MLB_SPORT_ID), payload) if isinstance(payload, dict) else []
    if isinstance(sport_markets, dict):
        sport_markets = sport_markets.get("markets", [])
    if not isinstance(sport_markets, list):
        sport_markets = []

    discovered: dict[str, int] = {}
    for market in sport_markets:
        if not isinstance(market, dict):
            continue
        if not market.get("proposition"):
            continue
        internal = _map_prop_market_name(str(market.get("name", "")))
        if internal and internal not in discovered:
            discovered[internal] = int(market["id"])

    if not discovered:
        discovered = dict(STATIC_MLB_PROP_MARKET_IDS)
        logger.warning(
            "TheRundown /sports/%s/markets/%s returned no player prop markets; "
            "using global catalog IDs: %s",
            MLB_SPORT_ID,
            ref.isoformat(),
            discovered,
        )

    write_odds_cache(cache_key, discovered)
    logger.info(
        "Discovered TheRundown MLB prop markets for %s: %s",
        ref.isoformat(),
        discovered,
    )
    return discovered


def fetch_rundown_mlb_events(
    slate_date: date | None,
    *,
    market_ids: str,
    cache_key: str,
    timeout: float = 20.0,
) -> dict[str, Any]:
    ref = slate_date or date.today()
    path = f"/sports/{MLB_SPORT_ID}/events/{ref.isoformat()}"
    params = {
        "market_ids": market_ids,
        "affiliate_ids": DEFAULT_AFFILIATE_IDS,
        "main_line": "true",
        "offset": str(RUNDOWN_UTC_OFFSET),
    }
    payload = _fetch_rundown_json(cache_key, path, params, timeout=timeout)
    if not isinstance(payload, dict):
        raise RundownError(f"Unexpected TheRundown events payload type: {type(payload)}")
    return payload


def _book_outcomes_from_market(
    market: dict[str, Any],
    *,
    home_team: str,
    away_team: str,
) -> dict[str, list[dict[str, Any]]]:
    """Map affiliate_id -> list of Odds-API-style outcomes for one Rundown market."""
    market_id = int(market.get("market_id", market.get("id", 0)))
    odds_key = RUNDOWN_MARKET_TO_ODDS_KEY.get(market_id)
    if odds_key is None:
        return {}

    by_book: dict[str, list[dict[str, Any]]] = {}
    for participant in market.get("participants", []):
        if not isinstance(participant, dict):
            continue
        side = _participant_side(participant)
        participant_name = _strip_player_suffix(str(participant.get("name", "")))

        for line in participant.get("lines", []):
            if not isinstance(line, dict):
                continue
            line_value = line.get("value")
            if _is_off_board(line_value):
                continue
            try:
                point = float(line_value)
            except (TypeError, ValueError):
                if odds_key == "spreads":
                    point = 0.0
                else:
                    continue

            for aff_id, price_obj in (line.get("prices") or {}).items():
                if not isinstance(price_obj, dict):
                    continue
                price = price_obj.get("price")
                if _is_off_board(price):
                    continue
                book_key = AFFILIATE_TO_BOOK.get(str(aff_id))
                if book_key is None:
                    continue

                if odds_key == "totals":
                    if side not in {"Over", "Under"}:
                        continue
                    outcome = {"name": side, "point": point, "price": price}
                elif odds_key == "spreads":
                    team_name = participant_name or str(participant.get("name", ""))
                    outcome = {"name": team_name, "point": point, "price": price}
                else:
                    team_name = participant_name or str(participant.get("name", ""))
                    outcome = {"name": team_name, "price": price}

                by_book.setdefault(book_key, []).append(outcome)

    return by_book


def _merge_bookmakers_for_event(
    event: dict[str, Any],
    *,
    market_filter: set[int] | None = None,
) -> list[dict[str, Any]]:
    home_team, away_team = _team_names(event)
    books: dict[str, dict[str, Any]] = {}

    for market in event.get("markets", []):
        if not isinstance(market, dict):
            continue
        market_id = int(market.get("market_id", market.get("id", 0)))
        if market_filter is not None and market_id not in market_filter:
            continue
        odds_key = RUNDOWN_MARKET_TO_ODDS_KEY.get(market_id)
        internal_prop = _map_prop_market_name(str(market.get("name", "")))
        market_key = internal_prop if internal_prop else odds_key
        if market_key is None:
            continue

        by_book = _book_outcomes_from_market(
            market,
            home_team=home_team,
            away_team=away_team,
        )
        for book_key, outcomes in by_book.items():
            if book_key not in books:
                books[book_key] = {"key": book_key, "title": book_key, "markets": []}
            books[book_key]["markets"].append({"key": market_key, "outcomes": outcomes})

    return list(books.values())


def adapt_rundown_vegas_games(events_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt TheRundown events payload into Odds-API-shaped game list for vegas parsing."""
    games: list[dict[str, Any]] = []
    for event in _events_from_payload(events_payload):
        away, home = _team_names(event)
        eid = _event_id(event)
        if not eid:
            continue
        bookmakers = _merge_bookmakers_for_event(
            event,
            market_filter={1, 2, 3},
        )
        if not bookmakers:
            continue
        games.append(
            {
                "id": eid,
                "home_team": home,
                "away_team": away,
                "commence_time": event.get("event_date", event.get("commence_time")),
                "bookmakers": bookmakers,
            }
        )
    return games


def adapt_rundown_event_props(
    event: dict[str, Any],
    market_id_map: dict[str, int],
) -> dict[str, Any]:
    """Adapt one TheRundown event into Odds-API-shaped player prop payload."""
    away, home = _team_names(event)
    eid = _event_id(event)
    allowed_ids = set(market_id_map.values())
    bookmakers: dict[str, dict[str, Any]] = {}

    for market in event.get("markets", []):
        if not isinstance(market, dict):
            continue
        market_id = int(market.get("market_id", market.get("id", 0)))
        if market_id not in allowed_ids:
            continue
        internal_key = next(
            (k for k, mid in market_id_map.items() if mid == market_id),
            None,
        )
        if internal_key is None:
            continue

        for participant in market.get("participants", []):
            if not isinstance(participant, dict):
                continue
            side = _participant_side(participant)
            if side not in {"Over", "Under"}:
                continue
            player_name = _strip_player_suffix(str(participant.get("name", "")))
            if not player_name:
                continue

            for line in participant.get("lines", []):
                if not isinstance(line, dict):
                    continue
                line_value = line.get("value")
                if _is_off_board(line_value):
                    continue
                try:
                    point = float(line_value)
                except (TypeError, ValueError):
                    continue

                for aff_id, price_obj in (line.get("prices") or {}).items():
                    if not isinstance(price_obj, dict):
                        continue
                    price = price_obj.get("price")
                    if _is_off_board(price):
                        continue
                    book_key = AFFILIATE_TO_BOOK.get(str(aff_id))
                    if book_key is None:
                        continue
                    outcome = {
                        "name": side,
                        "description": player_name,
                        "point": point,
                        "price": price,
                    }
                    if book_key not in bookmakers:
                        bookmakers[book_key] = {
                            "key": book_key,
                            "title": book_key,
                            "markets": [],
                        }
                    market_entry = next(
                        (
                            m
                            for m in bookmakers[book_key]["markets"]
                            if m.get("key") == internal_key
                        ),
                        None,
                    )
                    if market_entry is None:
                        market_entry = {"key": internal_key, "outcomes": []}
                        bookmakers[book_key]["markets"].append(market_entry)
                    market_entry["outcomes"].append(outcome)

    return {
        "id": eid,
        "home_team": home,
        "away_team": away,
        "commence_time": event.get("event_date", event.get("commence_time")),
        "bookmakers": list(bookmakers.values()),
    }


def adapt_rundown_events_list(events_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in _events_from_payload(events_payload):
        away, home = _team_names(event)
        eid = _event_id(event)
        if not eid:
            continue
        rows.append(
            {
                "id": eid,
                "sport_key": "baseball_mlb",
                "home_team": home,
                "away_team": away,
                "commence_time": event.get("event_date", event.get("commence_time")),
            }
        )
    return rows


def fetch_rundown_event_list(*, timeout: float = 20.0) -> pd.DataFrame:
    cache_key = f"rundown_mlb_events:{date.today().isoformat()}"
    payload = fetch_rundown_mlb_events(
        date.today(),
        market_ids="1",
        cache_key=cache_key,
        timeout=timeout,
    )
    rows = adapt_rundown_events_list(payload)
    return pd.DataFrame(
        [
            {
                "event_id": r["id"],
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "commence_time": r.get("commence_time"),
            }
            for r in rows
        ]
    )


def fetch_rundown_batched_props(
    event_ids: list[str],
    *,
    markets: str = "batter_total_bases,pitcher_outs",
    sportsbook_keys: str = DEFAULT_SPORTSBOOK_KEYS,
    slate_date: date | None = None,
    timeout: float = 20.0,
) -> pd.DataFrame:
    from baseball_props.data.odds_props import DEFAULT_PROP_MARKETS, _parse_prop_payload

    if not event_ids:
        return pd.DataFrame(
            columns=["game_id", "player_name", "market", "side", "line", "odds", "bookmaker"]
        )

    ref = slate_date or date.today()
    allowed = {m.strip() for m in markets.split(",") if m.strip()}
    market_id_map = discover_mlb_prop_market_ids(ref, timeout=timeout)
    market_id_map = {k: v for k, v in market_id_map.items() if k in allowed}
    logger.debug("[TheRundown] Discovered prop market IDs for %s: %s", ref.isoformat(), market_id_map)
    if not market_id_map:
        raise RundownError(f"No TheRundown prop markets discovered for {ref.isoformat()}")

    market_ids_csv = ",".join(str(v) for v in market_id_map.values())
    cache_key = f"rundown_player_props:{ref.isoformat()}:{market_ids_csv}:{sportsbook_keys}"
    payload = fetch_rundown_mlb_events(
        ref,
        market_ids=market_ids_csv,
        cache_key=cache_key,
        timeout=timeout,
    )

    wanted = set(event_ids)
    frames: list[pd.DataFrame] = []
    event_market_debug: list[dict[str, Any]] = []
    matched_events = 0
    for event in _events_from_payload(payload):
        eid = _event_id(event)
        if eid not in wanted:
            continue
        matched_events += 1
        market_ids_in_event = sorted(
            {
                int(m.get("market_id", m.get("id", 0)))
                for m in event.get("markets", [])
                if isinstance(m, dict)
            }
        )
        event_market_debug.append({"event_id": eid[:12], "market_ids": market_ids_in_event})
        adapted = adapt_rundown_event_props(event, market_id_map)
        parsed = _parse_prop_payload(adapted, eid, allowed)
        if not parsed.empty:
            frames.append(parsed)

    logger.debug(
        "[TheRundown] Props payload: %d matched slate events; market IDs sample: %s",
        matched_events,
        event_market_debug[:3],
    )

    if not frames:
        logger.debug("[TheRundown] Props fetch: 0 lines (no prop markets in event payloads)")
        return pd.DataFrame(
            columns=["game_id", "player_name", "market", "side", "line", "odds", "bookmaker"]
        )

    combined = pd.concat(frames, ignore_index=True)
    api_players = sorted(combined["player_name"].dropna().unique().tolist())[:3]
    logger.debug(
        "[TheRundown] Props fetch: %d lines across %d events (sample players: %s)",
        len(combined),
        len(frames),
        api_players,
    )
    logger.info(
        "Fetched %d TheRundown player prop rows across %d events",
        len(combined),
        len(frames),
    )
    return combined
