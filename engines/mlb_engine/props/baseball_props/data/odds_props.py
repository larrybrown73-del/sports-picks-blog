from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from baseball_props.data.ingest import (
    DEFAULT_SPORTSBOOK_KEYS,
    fetch_odds_api_json,
    get_odds_api_key,
    is_odds_event_id,
)
from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)

ODDS_EVENTS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
ODDS_EVENT_ODDS_URL = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"

DEFAULT_PROP_MARKETS = "batter_total_bases,batter_hits,pitcher_outs"

# Game-level markets that must never be treated as player props
_BLOCKED_MARKET_KEYS = frozenset(
    {
        "h2h",
        "spreads",
        "totals",
        "team_totals",
        "alternate_totals",
        "alternate_team_totals",
        "alternate_spreads",
    }
)

PROP_LINE_BOUNDS: dict[str, tuple[float, float]] = {
    "batter_total_bases": (0.5, 3.5),
    "batter_hits": (0.5, 2.5),
    "pitcher_outs": (10.0, 27.0),
}

_PROP_COLUMNS = [
    "game_id",
    "player_name",
    "market",
    "side",
    "line",
    "odds",
    "bookmaker",
]


def _markets_from_param(markets: str) -> set[str]:
    return {m.strip() for m in markets.split(",") if m.strip()}


def _is_allowed_player_prop_market(market_key: str, allowed_markets: set[str]) -> bool:
    if market_key in _BLOCKED_MARKET_KEYS:
        return False
    if market_key not in allowed_markets:
        return False
    return market_key.startswith("batter_") or market_key.startswith("pitcher_")


def is_plausible_prop_line(market: str, line: float) -> bool:
    """Return True when a parsed prop line is within expected bounds for the market."""
    bounds = PROP_LINE_BOUNDS.get(market)
    if bounds is None:
        return True
    low, high = bounds
    return low <= line <= high


def _parse_player_prop_outcome(
    outcome: dict,
    *,
    market_key: str,
    event_id: str,
    book_key: str,
) -> dict[str, object] | None:
    side = str(outcome.get("name", "")).strip()
    if side not in {"Over", "Under"}:
        return None

    player_name = str(outcome.get("description") or "").strip()
    if not player_name or player_name in {"Over", "Under"}:
        return None

    point = outcome.get("point")
    if point is None:
        return None

    line = float(point)
    if not is_plausible_prop_line(market_key, line):
        logger.debug(
            "Skipping implausible %s line %.1f for %s (event %s)",
            market_key,
            line,
            player_name,
            event_id,
        )
        return None

    return {
        "game_id": event_id,
        "player_name": player_name,
        "market": market_key,
        "side": side,
        "line": line,
        "odds": outcome.get("price"),
        "bookmaker": book_key,
    }


def _parse_prop_payload(
    payload: dict,
    event_id: str,
    allowed_markets: set[str] | None = None,
) -> pd.DataFrame:
    """Parse player prop outcomes; skip game/team totals and malformed rows."""
    if allowed_markets is None:
        allowed_markets = _markets_from_param(DEFAULT_PROP_MARKETS)

    rows: list[dict[str, object]] = []
    for book in payload.get("bookmakers", []):
        book_key = book.get("key", "")
        for market in book.get("markets", []):
            market_key = market.get("key", "")
            if not _is_allowed_player_prop_market(market_key, allowed_markets):
                continue
            for outcome in market.get("outcomes", []):
                parsed = _parse_player_prop_outcome(
                    outcome,
                    market_key=market_key,
                    event_id=event_id,
                    book_key=book_key,
                )
                if parsed is not None:
                    rows.append(parsed)

    if not rows:
        return pd.DataFrame(columns=_PROP_COLUMNS)
    return pd.DataFrame(rows)


def _event_props_cache_key(
    event_id: str,
    *,
    markets: str,
    sportsbook_keys: str,
) -> str:
    return f"player_props:v3:{event_id}:{markets}:{sportsbook_keys}"


def _payload_has_player_props(payload: Any) -> bool:
    return isinstance(payload, dict) and bool(payload.get("bookmakers"))


def _fetch_stale_props_by_team_labels(
    away_label: str,
    home_label: str,
    *,
    slate_game_id: str,
    markets: str,
) -> pd.DataFrame:
    """Find a non-empty historical player_props cache entry by team names."""
    import json

    from baseball_props.data.ingest import ODDS_CACHE_DIR, _matches_rundown_team_label

    allowed = _markets_from_param(markets)
    for path in ODDS_CACHE_DIR.glob("*.json"):
        try:
            with path.open(encoding="utf-8") as handle:
                entry = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        cache_key = str(entry.get("cache_key", ""))
        if not cache_key.startswith("player_props:v3:"):
            continue
        payload = entry.get("data")
        if not _payload_has_player_props(payload):
            continue
        if not _matches_rundown_team_label(
            home_label, str(payload.get("home_team", ""))
        ) or not _matches_rundown_team_label(
            away_label, str(payload.get("away_team", ""))
        ):
            continue
        parsed = _parse_prop_payload(payload, slate_game_id, allowed)
        if parsed.empty:
            continue
        logger.info(
            "Using stale props cache for %s @ %s (%d rows)",
            away_label,
            home_label,
            len(parsed),
        )
        return parsed
    return pd.DataFrame(columns=_PROP_COLUMNS)


def _fetch_odds_api_event_list(*, timeout: float = 20.0) -> pd.DataFrame:
    """List MLB events from The Odds API (never TheRundown)."""
    api_key = get_odds_api_key(required=False)
    if not api_key:
        return pd.DataFrame(columns=["event_id", "home_team", "away_team", "commence_time"])

    params = {"apiKey": api_key}
    events = fetch_odds_api_json("mlb_events_odds_api", ODDS_EVENTS_URL, params, timeout=timeout)
    rows = [
        {
            "event_id": e.get("id"),
            "home_team": e.get("home_team"),
            "away_team": e.get("away_team"),
            "commence_time": e.get("commence_time"),
        }
        for e in events
        if isinstance(e, dict)
    ]
    return pd.DataFrame(rows)


def _map_to_odds_api_event_ids(
    external_event_ids: list[str],
    *,
    timeout: float = 20.0,
) -> dict[str, str]:
    """
    Map external event IDs (e.g. TheRundown hashes) to The Odds API event IDs by team match.

    Player prop fetches require Odds API event hashes; vegas ingestion may supply Rundown IDs.
    """
    from baseball_props.data.ingest import _matches_rundown_team_label

    if not external_event_ids:
        return {}

    external_df = fetch_event_ids(timeout=timeout)
    odds_df = _fetch_odds_api_event_list(timeout=timeout)
    if external_df.empty or odds_df.empty:
        return {}

    mapping: dict[str, str] = {}
    for ext_id in external_event_ids:
        ext_rows = external_df[external_df["event_id"].astype(str) == str(ext_id)]
        if ext_rows.empty:
            continue
        ext_row = ext_rows.iloc[0]
        home_label = str(ext_row["home_team"])
        away_label = str(ext_row["away_team"])
        matched = odds_df[
            odds_df.apply(
                lambda row: _matches_rundown_team_label(home_label, str(row["home_team"]))
                and _matches_rundown_team_label(away_label, str(row["away_team"])),
                axis=1,
            )
        ]
        if matched.empty:
            logger.warning(
                "No Odds API event match for external id %s (%s @ %s)",
                ext_id,
                away_label,
                home_label,
            )
            continue
        mapping[str(ext_id)] = str(matched.iloc[0]["event_id"])

    if mapping:
        logger.info(
            "Mapped %d/%d external event IDs to Odds API IDs for prop fallback",
            len(mapping),
            len(external_event_ids),
        )
    return mapping


def fetch_event_ids(*, timeout: float = 20.0) -> pd.DataFrame:
    """List current MLB events (TheRundown primary, Odds API fallback)."""
    try:
        from baseball_props.data.therundown import fetch_rundown_event_list

        return fetch_rundown_event_list(timeout=timeout)
    except Exception as exc:
        logger.warning("TheRundown event list failed (%s); falling back to Odds API", exc)

    api_key = get_odds_api_key(required=True)
    params = {"apiKey": api_key}
    events = fetch_odds_api_json("mlb_events", ODDS_EVENTS_URL, params, timeout=timeout)
    rows = [
        {
            "event_id": e.get("id"),
            "home_team": e.get("home_team"),
            "away_team": e.get("away_team"),
            "commence_time": e.get("commence_time"),
        }
        for e in events
    ]
    return pd.DataFrame(rows)


def fetch_event_player_props(
    event_id: str,
    *,
    markets: str = DEFAULT_PROP_MARKETS,
    sportsbook_keys: str = DEFAULT_SPORTSBOOK_KEYS,
    slate_game_id: str | None = None,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """Fetch player prop lines for one event via /events/{eventId}/odds."""
    if not is_odds_event_id(event_id):
        return pd.DataFrame(columns=_PROP_COLUMNS)

    allowed_markets = _markets_from_param(markets)
    api_key = get_odds_api_key(required=True)
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": markets,
        "oddsFormat": "american",
        "bookmakers": sportsbook_keys,
    }
    url = ODDS_EVENT_ODDS_URL.format(event_id=event_id)
    cache_key = _event_props_cache_key(
        event_id, markets=markets, sportsbook_keys=sportsbook_keys
    )
    payload = fetch_odds_api_json(cache_key, url, params, timeout=timeout)
    parse_game_id = slate_game_id or str(event_id)
    return _parse_prop_payload(payload, parse_game_id, allowed_markets)


def fetch_batched_player_props(
    event_ids: list[str],
    *,
    markets: str = DEFAULT_PROP_MARKETS,
    sportsbook_keys: str = DEFAULT_SPORTSBOOK_KEYS,
    timeout: float = 20.0,
) -> pd.DataFrame:
    """
    Fetch player props for multiple events.

    TheRundown: one daily events call with prop market_ids (preferred).
    Odds API fallback: per-event /events/{eventId}/odds route.
    """
    valid_ids = sorted({eid for eid in event_ids if is_odds_event_id(eid)})
    if not valid_ids:
        return pd.DataFrame(columns=_PROP_COLUMNS)

    try:
        from baseball_props.data.therundown import fetch_rundown_batched_props

        rundown_df = fetch_rundown_batched_props(
            valid_ids,
            markets=markets,
            sportsbook_keys=sportsbook_keys,
            timeout=timeout,
        )
        if not rundown_df.empty:
            return rundown_df
        logger.warning(
            "TheRundown returned 0 prop rows for %d events; falling back to Odds API",
            len(valid_ids),
        )
    except Exception as exc:
        logger.warning("TheRundown props fetch failed (%s); falling back to Odds API", exc)

    odds_api_id_map = _map_to_odds_api_event_ids(valid_ids, timeout=timeout)
    logger.debug(
        "[Odds API fallback] Mapped %d/%d external event IDs for prop fetch",
        len(odds_api_id_map),
        len(valid_ids),
    )

    frames: list[pd.DataFrame] = []
    for external_id in valid_ids:
        odds_api_id = odds_api_id_map.get(external_id, external_id)
        try:
            props = fetch_event_player_props(
                odds_api_id,
                markets=markets,
                sportsbook_keys=sportsbook_keys,
                slate_game_id=external_id,
                timeout=timeout,
            )
            if props.empty:
                events_df = fetch_event_ids(timeout=timeout)
                ext_rows = events_df[events_df["event_id"].astype(str) == external_id]
                if not ext_rows.empty:
                    row = ext_rows.iloc[0]
                    props = _fetch_stale_props_by_team_labels(
                        str(row["away_team"]),
                        str(row["home_team"]),
                        slate_game_id=external_id,
                        markets=markets,
                    )
            if not props.empty:
                frames.append(props)
        except requests.RequestException as exc:
            logger.warning(
                "Failed props fetch for event %s (odds id %s): %s",
                external_id,
                odds_api_id,
                exc,
            )

    if not frames:
        logger.debug("[Props fallback] No live or cached prop rows matched for this slate")
        return pd.DataFrame(columns=_PROP_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    logger.debug(
        "[Props fallback] Combined %d prop rows across %d events",
        len(combined),
        len(frames),
    )
    logger.info(
        "Fetched %d player prop rows across %d events (%d API calls)",
        len(combined),
        len(valid_ids),
        len(valid_ids),
    )
    return combined


def fetch_all_player_props(
    event_ids: list[str] | None = None,
    *,
    markets: str = DEFAULT_PROP_MARKETS,
    sportsbook_keys: str = DEFAULT_SPORTSBOOK_KEYS,
) -> pd.DataFrame:
    """Fetch prop lines for supplied events via per-event Odds API calls."""
    if event_ids is None:
        events = fetch_event_ids()
        event_ids = events["event_id"].dropna().astype(str).tolist()

    skipped = len(event_ids) - len([eid for eid in event_ids if is_odds_event_id(eid)])
    if skipped:
        logger.warning("Skipped %d invalid event id(s) for prop fetch", skipped)

    try:
        return fetch_batched_player_props(
            event_ids,
            markets=markets,
            sportsbook_keys=sportsbook_keys,
        )
    except requests.RequestException as exc:
        logger.warning("Batched player props fetch failed: %s", exc)
        return pd.DataFrame(columns=_PROP_COLUMNS)


def market_lines(props: pd.DataFrame) -> pd.DataFrame:
    """Collapse over/under rows to one market line per player/market/book."""
    if props.empty:
        return pd.DataFrame(columns=["game_id", "player_name", "market", "bookmaker", "market_line"])
    lines = (
        props.dropna(subset=["line"])
        .groupby(["game_id", "player_name", "market", "bookmaker"], as_index=False)["line"]
        .first()
    )
    return lines.rename(columns={"line": "market_line"})


def consolidated_market_lines(props: pd.DataFrame) -> pd.DataFrame:
    """One line per player/market/game using median across bookmakers (Over side)."""
    if props.empty:
        return pd.DataFrame(columns=["game_id", "player_name", "market", "market_line"])
    over_rows = props[props["side"] == "Over"]
    source = over_rows if not over_rows.empty else props
    per_book = market_lines(source)
    if per_book.empty:
        return pd.DataFrame(columns=["game_id", "player_name", "market", "market_line"])
    return (
        per_book.groupby(["game_id", "player_name", "market"], as_index=False)["market_line"]
        .median()
    )


def consolidated_prop_quotes(props: pd.DataFrame) -> pd.DataFrame:
    """One quote per player/market/game: median line and median American odds per side."""
    empty_cols = [
        "game_id",
        "player_name",
        "market",
        "market_line",
        "over_odds",
        "under_odds",
    ]
    if props.empty:
        return pd.DataFrame(columns=empty_cols)

    per_book_lines = market_lines(props)
    if per_book_lines.empty:
        return pd.DataFrame(columns=empty_cols)

    line_medians = (
        per_book_lines.groupby(["game_id", "player_name", "market"], as_index=False)[
            "market_line"
        ].median()
    )

    over_odds = (
        props[props["side"] == "Over"]
        .dropna(subset=["odds"])
        .groupby(["game_id", "player_name", "market"], as_index=False)["odds"]
        .median()
        .rename(columns={"odds": "over_odds"})
    )
    under_odds = (
        props[props["side"] == "Under"]
        .dropna(subset=["odds"])
        .groupby(["game_id", "player_name", "market"], as_index=False)["odds"]
        .median()
        .rename(columns={"odds": "under_odds"})
    )

    quotes = line_medians.merge(
        over_odds, on=["game_id", "player_name", "market"], how="left"
    ).merge(under_odds, on=["game_id", "player_name", "market"], how="left")
    return quotes[empty_cols]
