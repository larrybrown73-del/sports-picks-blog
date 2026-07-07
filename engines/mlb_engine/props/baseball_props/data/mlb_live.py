from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Literal

import pandas as pd
import requests

from baseball_props.logging_utils import get_logger

logger = get_logger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
DEFAULT_TIMEOUT = 25.0

LineupSource = Literal["boxscore", "previous_game", "roster_depth", "roster_reconciled"]

# Minimum batters required after reconciling a previous-game lineup with the active roster.
_MIN_RECONCILED_LINEUP = 7

# Position codes for building a synthetic 9-man order from active roster.
_ROSTER_SLOT_POSITIONS: tuple[str, ...] = (
    "C",
    "1B",
    "2B",
    "3B",
    "SS",
    "LF",
    "CF",
    "RF",
    "DH",
)
_POSITION_CODE_TO_ABBR: dict[str, str] = {
    "2": "C",
    "3": "1B",
    "4": "2B",
    "5": "3B",
    "6": "SS",
    "7": "LF",
    "8": "CF",
    "9": "RF",
    "10": "DH",
    "O": "OF",
}


def _get_json(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{MLB_API_BASE}{path}"
    response = requests.get(url, params=params or {}, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_todays_schedule(slate_date: date | None = None) -> list[dict[str, Any]]:
    """Return scheduled MLB games for a date with probable pitchers."""
    d = slate_date or date.today()
    params = {
        "sportId": 1,
        "date": d.strftime("%Y-%m-%d"),
        "hydrate": "probablePitcher(note),team,venue",
    }
    payload = _get_json("/schedule", params)
    games: list[dict[str, Any]] = []
    for day in payload.get("dates", []):
        games.extend(day.get("games", []))
    logger.info("MLB schedule: %d games on %s", len(games), d.isoformat())
    return games


def _parse_bat_hand(person: dict[str, Any]) -> str:
    bat = person.get("batSide", {}) or {}
    code = str(bat.get("code", "R")).upper()
    return code if code in {"L", "R", "S"} else "R"


def _parse_pitcher(probable: dict[str, Any] | None) -> tuple[str, str, str]:
    if not probable:
        return "", "R", ""
    pid = str(probable.get("id", ""))
    hand = probable.get("pitchHand", {}) or {}
    code = str(hand.get("code", "R")).upper()
    full_name = str(probable.get("fullName") or probable.get("name") or "").strip()
    return pid, code if code in {"L", "R"} else "R", full_name


def _extract_lineup_side(
    boxscore: dict[str, Any], side: str
) -> list[dict[str, str | int]]:
    teams = boxscore.get("teams", {})
    team = teams.get(side, {})
    players = team.get("players", {}) or {}
    batting_order = team.get("battingOrder") or []
    rows: list[dict[str, str | int]] = []
    for slot, player_key in enumerate(batting_order, start=1):
        lookup_keys = [player_key, f"ID{player_key}"]
        player = None
        for key in lookup_keys:
            if key in players:
                player = players[key]
                break
        if player is None:
            continue
        person = player.get("person", {})
        pid = person.get("id")
        if pid is None:
            continue
        name = person.get("fullName") or f"Player {pid}"
        rows.append(
            {
                "player_id": str(pid),
                "player_name": name,
                "lineup_slot": slot,
                "bat_hand": _parse_bat_hand(person),
            }
        )
    return rows


def fetch_game_lineups(game_pk: int) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
    """
    Fetch home/away batting orders from boxscore, falling back to live feed.
    Returns (away_lineup, home_lineup).
    """
    for endpoint in (f"/game/{game_pk}/boxscore", f"/game/{game_pk}/feed/live"):
        try:
            payload = _get_json(endpoint)
            box = payload.get("liveData", {}).get("boxscore", payload)
            away = _extract_lineup_side(box, "away")
            home = _extract_lineup_side(box, "home")
            if away and home:
                return away, home
        except requests.RequestException as exc:
            logger.debug("Lineup fetch failed for %s on %s: %s", game_pk, endpoint, exc)
    return [], []


def _lineup_from_boxscore_side(boxscore: dict[str, Any], side: str) -> list[dict[str, str | int]]:
    box = boxscore.get("liveData", {}).get("boxscore", boxscore)
    return _extract_lineup_side(box, side)


def fetch_team_previous_lineup(
    team_id: int,
    *,
    before_date: date,
    lookback_days: int = 21,
) -> list[dict[str, str | int]]:
    """Return batting order from the team's most recent finalized game before slate date."""
    start = before_date - timedelta(days=lookback_days)
    params = {
        "sportId": 1,
        "teamId": team_id,
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": before_date.strftime("%Y-%m-%d"),
        "gameTypes": "R",
    }
    try:
        payload = _get_json("/schedule", params)
    except requests.RequestException as exc:
        logger.debug("Previous-game schedule fetch failed for team %s: %s", team_id, exc)
        return []

    candidates: list[tuple[str, int, str]] = []
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            status = game.get("status", {}) or {}
            if status.get("abstractGameState") != "Final" and status.get("codedGameState") != "F":
                continue
            game_pk = int(game["gamePk"])
            game_date = str(game.get("officialDate") or game.get("gameDate", "")[:10])
            if game_date > before_date.isoformat():
                continue
            home = game.get("teams", {}).get("home", {}).get("team", {})
            away = game.get("teams", {}).get("away", {}).get("team", {})
            if int(home.get("id", 0)) == team_id:
                candidates.append((game_date, game_pk, "home"))
            elif int(away.get("id", 0)) == team_id:
                candidates.append((game_date, game_pk, "away"))

    if not candidates:
        return []

    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    for game_date, game_pk, side in candidates:
        try:
            boxscore = _get_json(f"/game/{game_pk}/boxscore")
        except requests.RequestException as exc:
            logger.debug("Previous-game boxscore fetch failed for %s: %s", game_pk, exc)
            continue

        lineup = _lineup_from_boxscore_side(boxscore, side)
        if lineup:
            logger.info(
                "Using previous-game lineup for team %s from game %s on %s (%d batters)",
                team_id,
                game_pk,
                game_date,
                len(lineup),
            )
            return lineup

    return []



def fetch_active_roster_hitters(team_id: int) -> list[dict[str, str]]:
    """Return non-pitchers on the active roster with position metadata."""
    try:
        payload = _get_json(f"/teams/{team_id}/roster", {"rosterType": "active"})
    except requests.RequestException as exc:
        logger.debug("Roster fetch failed for team %s: %s", team_id, exc)
        return []

    hitters: list[dict[str, str]] = []
    for entry in payload.get("roster", []):
        position = entry.get("position", {}) or {}
        pos_code = str(position.get("code", ""))
        if pos_code == "1":
            continue
        person = entry.get("person", {}) or {}
        pid = person.get("id")
        if pid is None:
            continue
        abbr = _POSITION_CODE_TO_ABBR.get(pos_code, position.get("abbreviation", "UT"))
        hitters.append(
            {
                "player_id": str(pid),
                "player_name": person.get("fullName") or f"Player {pid}",
                "position": abbr,
                "bat_hand": _parse_bat_hand(person),
            }
        )
    return hitters


def resolve_confirmed_lineup_slot(
    player_id: str,
    team_id: str | int,
    mlb_game_pk: str | int | None,
    *,
    fallback_slot: int | None = None,
) -> int | None:
    """
    Confirm lineup slot via Stats API boxscore and active roster.

    Falls back to ingested lineup_slot when live data is unavailable.
    """
    pid = str(player_id).strip()
    try:
        tid = int(team_id)
    except (TypeError, ValueError):
        return fallback_slot

    roster = fetch_active_roster_hitters(tid)
    if roster and pid not in {h["player_id"] for h in roster}:
        logger.warning(
            "Player %s not on active roster for team %s; using fallback slot %s",
            pid,
            tid,
            fallback_slot,
        )
        return fallback_slot

    if mlb_game_pk is not None:
        try:
            away_lineup, home_lineup = fetch_game_lineups(int(mlb_game_pk))
            for row in (away_lineup or []) + (home_lineup or []):
                if str(row.get("player_id")) == pid:
                    return int(row["lineup_slot"])
        except (TypeError, ValueError) as exc:
            logger.debug("Lineup slot lookup failed for game %s: %s", mlb_game_pk, exc)

    return fallback_slot


def fetch_roster_depth_lineup(team_id: int) -> list[dict[str, str | int]]:
    """Build a projected 9-man order from the active roster when no lineup is posted."""
    hitters = fetch_active_roster_hitters(team_id)
    if not hitters:
        return []

    by_position: dict[str, list[dict[str, str]]] = {}
    for hitter in hitters:
        by_position.setdefault(hitter["position"], []).append(hitter)

    chosen: list[dict[str, str]] = []
    used_ids: set[str] = set()
    for slot_pos in _ROSTER_SLOT_POSITIONS:
        pool = by_position.get(slot_pos, [])
        if slot_pos == "DH" and not pool:
            pool = by_position.get("OF", [])
        for hitter in pool:
            if hitter["player_id"] not in used_ids:
                chosen.append(hitter)
                used_ids.add(hitter["player_id"])
                break

    for hitter in hitters:
        if len(chosen) >= 9:
            break
        if hitter["player_id"] not in used_ids:
            chosen.append(hitter)
            used_ids.add(hitter["player_id"])

    if len(chosen) < 9:
        logger.warning(
            "Roster depth chart for team %s only yielded %d hitters",
            team_id,
            len(chosen),
        )
        return []

    lineup = [
        {
            "player_id": row["player_id"],
            "player_name": row["player_name"],
            "lineup_slot": slot,
            "bat_hand": row.get("bat_hand", "R"),
        }
        for slot, row in enumerate(chosen[:9], start=1)
    ]
    logger.info(
        "Using roster depth-chart lineup for team %s (%d batters)",
        team_id,
        len(lineup),
    )
    return lineup


def reconcile_lineup_with_active_roster(
    lineup: list[dict[str, str | int]],
    team_id: int,
    *,
    roster_hitters: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, str | int]], int]:
    """
    Drop departed players from a stale lineup and backfill from the active roster.

    Returns (lineup, removed_count).
    """
    if not lineup:
        return [], 0

    roster = roster_hitters if roster_hitters is not None else fetch_active_roster_hitters(team_id)
    if not roster:
        return lineup, 0

    active_ids = {h["player_id"] for h in roster}
    kept = [row for row in lineup if str(row["player_id"]) in active_ids]
    removed = len(lineup) - len(kept)
    if removed == 0 and len(kept) >= 9:
        return kept[:9], 0

    used_ids = {str(row["player_id"]) for row in kept}
    for hitter in roster:
        if len(kept) >= 9:
            break
        if hitter["player_id"] not in used_ids:
            kept.append(
                {
                    "player_id": hitter["player_id"],
                    "player_name": hitter["player_name"],
                    "lineup_slot": len(kept) + 1,
                    "bat_hand": hitter.get("bat_hand", "R"),
                }
            )
            used_ids.add(hitter["player_id"])

    for slot, row in enumerate(kept[:9], start=1):
        row["lineup_slot"] = slot

    if removed:
        logger.info(
            "Reconciled lineup for team %s: removed %d inactive player(s), %d batters remain",
            team_id,
            removed,
            len(kept),
        )
    return kept[:9], removed


def filter_injured_from_lineups(
    lineups: pd.DataFrame,
    injury_lookup: dict[str, dict] | None,
) -> pd.DataFrame:
    """
    Log-only injury filter stub — flags IL players but does not drop from lineups yet.
    """
    if lineups.empty or not injury_lookup:
        return lineups

    from baseball_props.data.injuries import lookup_injury

    flagged = 0
    for _, row in lineups.iterrows():
        name = str(row.get("player_name", ""))
        record = lookup_injury(name, injury_lookup)
        if record and str(record.get("status", "")).upper().startswith("IL"):
            flagged += 1
    if flagged:
        logger.info(
            "Injury filter stub: %d lineup slot(s) match IL status (not removed)",
            flagged,
        )
    return lineups


def resolve_game_lineups(
    game_pk: int,
    home_team_id: int,
    away_team_id: int,
    home_abbrev: str,
    away_abbrev: str,
    slate_date: date,
    *,
    roster_cache: dict[int, list[dict[str, str]]] | None = None,
) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]], LineupSource]:
    """
    Resolve away/home lineups: posted boxscore → previous game (roster-filtered) → active roster depth.
    """
    cache = roster_cache if roster_cache is not None else {}

    def _roster_for(team_id: int) -> list[dict[str, str]]:
        if team_id not in cache:
            cache[team_id] = fetch_active_roster_hitters(team_id)
        return cache[team_id]

    away_lineup, home_lineup = fetch_game_lineups(game_pk)
    if away_lineup and home_lineup:
        return away_lineup, home_lineup, "boxscore"

    logger.warning(
        "Lineups not posted for game %s (%s @ %s); trying previous-game fallback",
        game_pk,
        away_abbrev,
        home_abbrev,
    )
    source: LineupSource = "previous_game"
    if not away_lineup:
        away_lineup = fetch_team_previous_lineup(away_team_id, before_date=slate_date)
    if not home_lineup:
        home_lineup = fetch_team_previous_lineup(home_team_id, before_date=slate_date)

    if away_lineup:
        away_lineup, away_removed = reconcile_lineup_with_active_roster(
            away_lineup, away_team_id, roster_hitters=_roster_for(away_team_id)
        )
        if away_removed:
            source = "roster_reconciled"
    if home_lineup:
        home_lineup, home_removed = reconcile_lineup_with_active_roster(
            home_lineup, home_team_id, roster_hitters=_roster_for(home_team_id)
        )
        if home_removed:
            source = "roster_reconciled"

    if (
        away_lineup
        and home_lineup
        and len(away_lineup) >= _MIN_RECONCILED_LINEUP
        and len(home_lineup) >= _MIN_RECONCILED_LINEUP
    ):
        return away_lineup, home_lineup, source

    logger.warning(
        "Previous-game lineups unavailable or too stale for game %s (%s @ %s); using roster depth chart",
        game_pk,
        away_abbrev,
        home_abbrev,
    )
    if not away_lineup or len(away_lineup) < _MIN_RECONCILED_LINEUP:
        away_lineup = fetch_roster_depth_lineup(away_team_id)
    if not home_lineup or len(home_lineup) < _MIN_RECONCILED_LINEUP:
        home_lineup = fetch_roster_depth_lineup(home_team_id)
    if away_lineup and home_lineup:
        return away_lineup, home_lineup, "roster_depth"

    return [], [], "roster_depth"


def build_slate_from_schedule(
    slate_date: date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], dict[str, int]]:
    """
    Build slate_games, lineups, probable-pitcher name map, and lineup-source counts from MLB schedule.
    """
    slate_day = slate_date or date.today()
    games_raw = fetch_todays_schedule(slate_date)
    if not games_raw:
        return (
            pd.DataFrame(columns=[
                "game_id", "game_date", "home_team_id", "away_team_id", "park_id",
                "sp_home_id", "sp_away_id", "sp_home_hand", "sp_away_hand", "mlb_game_pk",
            ]),
            pd.DataFrame(columns=["game_id", "team_id", "lineup_slot", "player_id", "player_name", "bat_hand"]),
            {},
            {},
        )

    slate_rows: list[dict[str, object]] = []
    lineup_rows: list[dict[str, object]] = []
    pitcher_names: dict[str, str] = {}
    lineup_source_counts: dict[str, int] = {}
    roster_cache: dict[int, list[dict[str, str]]] = {}

    for game in games_raw:
        game_pk = int(game["gamePk"])
        home = game.get("teams", {}).get("home", {}).get("team", {})
        away = game.get("teams", {}).get("away", {}).get("team", {})
        home_id = home.get("abbreviation", "HOM")
        away_id = away.get("abbreviation", "AWY")
        home_team_id = int(home.get("id", 0))
        away_team_id = int(away.get("id", 0))
        venue = game.get("venue", {}) or {}
        park_id = str(venue.get("id", "UNK"))

        sp_home_id, sp_home_hand, sp_home_name = _parse_pitcher(
            game.get("teams", {}).get("home", {}).get("probablePitcher")
        )
        sp_away_id, sp_away_hand, sp_away_name = _parse_pitcher(
            game.get("teams", {}).get("away", {}).get("probablePitcher")
        )
        if sp_home_id and sp_home_name:
            pitcher_names[sp_home_id] = sp_home_name
        if sp_away_id and sp_away_name:
            pitcher_names[sp_away_id] = sp_away_name

        away_lineup, home_lineup, lineup_source = resolve_game_lineups(
            game_pk,
            home_team_id,
            away_team_id,
            str(home_id),
            str(away_id),
            slate_day,
            roster_cache=roster_cache,
        )
        if not away_lineup or not home_lineup:
            logger.warning(
                "Skipping game %s (%s @ %s): no lineup via boxscore, previous game, or roster",
                game_pk,
                away_id,
                home_id,
            )
            continue

        lineup_source_counts[lineup_source] = lineup_source_counts.get(lineup_source, 0) + 1

        if lineup_source != "boxscore":
            logger.info(
                "Game %s (%s @ %s) using %s lineup source",
                game_pk,
                away_id,
                home_id,
                lineup_source,
            )

        game_date = str(game.get("officialDate") or game.get("gameDate", "")[:10])
        game_id = str(game_pk)
        slate_rows.append(
            {
                "game_id": game_id,
                "game_date": game_date,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "park_id": park_id,
                "sp_home_id": sp_home_id,
                "sp_away_id": sp_away_id,
                "sp_home_hand": sp_home_hand,
                "sp_away_hand": sp_away_hand,
                "mlb_game_pk": game_pk,
            }
        )

        for team_id, lineup in ((away_id, away_lineup), (home_id, home_lineup)):
            for row in lineup:
                lineup_rows.append(
                    {
                        "game_id": game_id,
                        "team_id": team_id,
                        "lineup_slot": row["lineup_slot"],
                        "player_id": row["player_id"],
                        "player_name": row["player_name"],
                        "bat_hand": row.get("bat_hand", "R"),
                    }
                )

    slate_games = pd.DataFrame(slate_rows)
    lineups = pd.DataFrame(lineup_rows)
    logger.info(
        "Built live slate: %d games, %d lineup slots (lineup sources: %s)",
        len(slate_games),
        len(lineups),
        lineup_source_counts or "none",
    )
    return slate_games, lineups, pitcher_names, lineup_source_counts
