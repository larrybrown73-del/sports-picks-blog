"""
Historical data ingestion for the soccer projection model.

Sourced entirely from TheStatsAPI's *results* and *player-stats* endpoints
(what already happened), never its *odds* endpoints (what a book is
offering right now). Keeping this module blind to sportsbook prices is what
guarantees the projection model it feeds can never accidentally anchor on
the market it's supposed to be independently evaluated against -- see the
architecture contract at the top of ev_engine_core.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ev_engine_core import is_thestatsapi_not_found, thestatsapi_get


@dataclass(frozen=True)
class MatchResult:
    """One finished match's final score, used to fit team-strength ratings."""

    match_id: str
    competition_id: str
    utc_date: str
    home_team_id: str
    home_team_name: str
    away_team_id: str
    away_team_name: str
    home_goals: int
    away_goals: int


@dataclass(frozen=True)
class PlayerSeasonStats:
    """
    One player's season-aggregate counting stats, used to derive per-90
    rates for the player-prop projection model. Aggregated (not per-match)
    on purpose: season totals are one API call per player instead of one
    per match the player appeared in, and a per-90 rate model only needs a
    stable long-run rate, not match-by-match granularity.
    """

    player_id: str
    player_name: str
    team_id: str | None
    position: str | None
    season_id: str
    minutes_played: int
    appearances: int
    goals: int
    assists: int
    total_shots: int
    shots_on_target: int


def fetch_historical_matches(
    api_key: str,
    competition_id: str,
    *,
    season_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    per_page: int = 100,
    max_pages: int = 50,
    session: Any | None = None,
) -> list[MatchResult]:
    """
    Page through GET /football/matches?status=finished for one competition
    and return clean MatchResult records for fitting.

    Ruthless missing-data policy: any fixture missing a final score (e.g. an
    abandoned match some feeds still tag "finished" without a score, or a
    walkover with no team ids) is dropped outright -- never imputed with a
    0-0 placeholder or backfilled from another source.
    """

    results: list[MatchResult] = []
    page = 1
    while page <= max_pages:
        payload = thestatsapi_get(
            "/football/matches",
            api_key,
            params={
                key: value
                for key, value in {
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "date_from": date_from,
                    "date_to": date_to,
                    "status": "finished",
                    "page": page,
                    "per_page": per_page,
                }.items()
                if value is not None
            },
            session=session,
        )
        rows = payload.get("data", []) or []
        for row in rows:
            match = _match_result_from_row(row)
            if match is not None:
                results.append(match)

        meta = payload.get("meta", {}) or {}
        total_pages = meta.get("total_pages", page)
        if page >= total_pages or not rows:
            break
        page += 1

    return results


def _match_result_from_row(row: dict[str, Any]) -> MatchResult | None:
    score = row.get("score") or {}
    home_goals = score.get("home")
    away_goals = score.get("away")
    # A "finished" fixture with no recorded score cannot contribute a goal
    # observation to the model -- drop it rather than guessing a scoreline.
    if home_goals is None or away_goals is None:
        return None

    home_team = row.get("home_team") or {}
    away_team = row.get("away_team") or {}
    if not home_team.get("id") or not away_team.get("id"):
        return None

    return MatchResult(
        match_id=str(row.get("id") or ""),
        competition_id=str(row.get("competition_id") or ""),
        utc_date=str(row.get("utc_date") or ""),
        home_team_id=str(home_team["id"]),
        home_team_name=str(home_team.get("name") or ""),
        away_team_id=str(away_team["id"]),
        away_team_name=str(away_team.get("name") or ""),
        home_goals=int(home_goals),
        away_goals=int(away_goals),
    )


def fetch_player_season_stats(
    api_key: str,
    player_id: str,
    season_id: str,
    *,
    player_name: str | None = None,
    competition_id: str | None = None,
    stage: str | None = None,
    session: Any | None = None,
) -> PlayerSeasonStats | None:
    """
    GET /football/players/{player_id}/stats.

    `player_name` is accepted as a caller-supplied override because this
    endpoint's payload does not itself include the player's name (only
    `/football/players` and `/football/players/{id}` do) -- pass it through
    from whichever roster/lookup call resolved this player_id.

    Returns None (dropped, not guessed) if the player has no recorded
    minutes for the season: a per-90 rate is mathematically undefined
    without a positive minutes denominator. Also returns None (rather than
    raising) on a 404 -- TheStatsAPI returns one for a player with no stats
    page for the given season/competition at all (e.g. a new signing who
    hasn't featured yet), which is exactly as "no data for this player"
    as an empty/zero-minutes payload, not a real failure. A single
    stats-less player must never be allowed to crash the whole
    roster-building loop in daily_model.py.
    """

    try:
        payload = thestatsapi_get(
            f"/football/players/{player_id}/stats",
            api_key,
            params={
                key: value
                for key, value in {
                    "season_id": season_id,
                    "competition_id": competition_id,
                    "stage": stage,
                }.items()
                if value is not None
            },
            session=session,
        )
    except Exception as exc:
        if is_thestatsapi_not_found(exc):
            return None
        raise
    data = payload.get("data", payload)
    minutes_played = data.get("minutes_played")
    if not minutes_played:
        return None

    scoring = data.get("scoring") or {}
    shooting = data.get("shooting") or {}

    return PlayerSeasonStats(
        player_id=str(data.get("player_id") or player_id),
        player_name=player_name or str(data.get("player_name") or ""),
        team_id=str(data["team_id"]) if data.get("team_id") is not None else None,
        position=str(data.get("position")) if data.get("position") is not None else None,
        season_id=str(data.get("season_id") or season_id),
        minutes_played=int(minutes_played),
        appearances=int(data.get("appearances") or 0),
        goals=int(scoring.get("goals") or 0),
        assists=int(scoring.get("assists") or 0),
        total_shots=int(shooting.get("total_shots") or 0),
        shots_on_target=int(shooting.get("shots_on_target") or 0),
    )


def fetch_team_players(
    api_key: str,
    team_id: str,
    *,
    per_page: int = 100,
    max_pages: int = 10,
    session: Any | None = None,
) -> list[dict[str, Any]]:
    """
    GET /football/teams/{team_id}/players -- enumerates a team's roster
    (id, name, position) so a caller can then pull each player's season
    stats via `fetch_player_season_stats` and build a rate profile.

    Deliberately NOT `/football/players?team_id=...`: that filters by a
    player's *current club* (`current_team`), which is a different field
    from the roster being asked for here -- it silently returns an empty
    roster (0 players, not an error) for any national/international team,
    since a player's `current_team` is their club, not their
    `national_team`. That bug meant every World Cup team's player-prop
    coverage silently built zero profiles despite the roster+stats fetch
    "succeeding" (empty result, not an exception) -- caught only by
    manually inspecting a live payload, not by any error path.
    """

    players: list[dict[str, Any]] = []
    page = 1
    while page <= max_pages:
        payload = thestatsapi_get(
            f"/football/teams/{team_id}/players",
            api_key,
            params={"page": page, "per_page": per_page},
            session=session,
        )
        rows = payload.get("data", []) or []
        players.extend(rows)

        meta = payload.get("meta", {}) or {}
        total_pages = meta.get("total_pages", page)
        if page >= total_pages or not rows:
            break
        page += 1

    return players
