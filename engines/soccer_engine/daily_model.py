"""
Daily model-fitting pipeline: turns "today's match schedule" into a
`{competition_id: SoccerProjectionModel}` map ready to grade every match in
that slate.

Fits Dixon-Coles team ratings and builds player rate profiles from
TheStatsAPI history. A competition with a full slate of teams can mean a
few hundred HTTP calls (one roster call per team, one season-stats call per
rostered player, one paginated results call per competition) to build
player-prop coverage -- most of which return the exact same answer if this
pipeline runs more than once in a day (a retry, grading a second slate,
manual reruns). To protect API quota, every heavy fetch below goes through
`cache_store.py`'s same-day, file-based cache first: see that module's
docstring for the on-disk format and invalidation rule (one JSON file per
category per calendar day; a new day means a new, empty file).
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from cache_store import CACHE_DIR, read_cache, write_cache
from ev_engine_core import thestatsapi_get
from historical_data import (
    MatchResult,
    PlayerSeasonStats,
    fetch_historical_matches,
    fetch_player_season_stats,
    fetch_team_players,
)
from player_props_model import build_player_rate_profile
from projection_model import SoccerProjectionModel
from royal_validators import (  # noqa: F401 -- re-exported for Royal Picks callers
    FORM_CONFLICT_SCORE_PENALTY,
    FORM_DEVIATION_THRESHOLD,
    HIT_RATE_FLOOR,
    build_downside_analysis,
    enrich_results_with_situational_validators,
    estimated_hit_rate,
    form_deviation_ratio,
    normalize_position,
    position_variance_scale,
)
from team_strength import SoccerModelDataError, fit_dixon_coles

# Long enough for Dixon-Coles' own exponential time-decay weighting (see
# team_strength.DEFAULT_TIME_DECAY_XI) to do the "recent form matters most"
# work itself, short enough to stay a handful of API pages per competition.
HISTORY_LOOKBACK_DAYS = 730
MIN_HISTORICAL_MATCHES_TO_FIT = 10

CACHE_CATEGORY_HISTORICAL_MATCHES = "historical_matches"
CACHE_CATEGORY_TEAM_PLAYERS = "team_players"
CACHE_CATEGORY_PLAYER_SEASON_STATS = "player_season_stats"

logger = logging.getLogger("soccer_engine.daily_model")


def _cached_fetch_historical_matches(
    api_key: str,
    competition_id: str,
    *,
    date_from: str,
    date_to: str,
    session: Any | None,
    today: date,
    cache_dir: Path,
) -> list[MatchResult]:
    """Read-through cache around historical_data.fetch_historical_matches, keyed by the exact query window."""

    cache_key = f"{competition_id}|{date_from}|{date_to}"
    cache = read_cache(CACHE_CATEGORY_HISTORICAL_MATCHES, today, cache_dir=cache_dir)
    if cache_key in cache:
        return [MatchResult(**row) for row in cache[cache_key]]

    matches = fetch_historical_matches(api_key, competition_id, date_from=date_from, date_to=date_to, session=session)
    cache[cache_key] = [asdict(match) for match in matches]
    write_cache(CACHE_CATEGORY_HISTORICAL_MATCHES, today, cache, cache_dir=cache_dir)
    return matches


def _cached_fetch_team_players(
    api_key: str,
    team_id: str,
    *,
    session: Any | None,
    today: date,
    cache_dir: Path,
) -> list[dict[str, Any]]:
    """Read-through cache around historical_data.fetch_team_players, keyed by team_id (rosters barely move intra-day)."""

    cache = read_cache(CACHE_CATEGORY_TEAM_PLAYERS, today, cache_dir=cache_dir)
    if team_id in cache:
        return cache[team_id]

    players = fetch_team_players(api_key, team_id, session=session)
    cache[team_id] = players
    write_cache(CACHE_CATEGORY_TEAM_PLAYERS, today, cache, cache_dir=cache_dir)
    return players


def _cached_fetch_player_season_stats(
    api_key: str,
    player_id: str,
    season_id: str,
    *,
    player_name: str | None,
    competition_id: str | None,
    session: Any | None,
    today: date,
    cache_dir: Path,
) -> PlayerSeasonStats | None:
    """
    Read-through cache around historical_data.fetch_player_season_stats,
    keyed by (player_id, season_id) -- this is the single biggest source of
    call volume (one request per rostered player), so it's the fetch this
    cache protects the most.

    A cached `None` (a player with no recorded season stats) is stored and
    honored just like a real record -- `cache_key in cache` distinguishes
    "already asked, API had nothing" from "haven't asked yet", so a
    stats-less player isn't re-requested every run for the rest of the day.
    """

    cache_key = f"{player_id}|{season_id}"
    cache = read_cache(CACHE_CATEGORY_PLAYER_SEASON_STATS, today, cache_dir=cache_dir)
    if cache_key in cache:
        cached_value = cache[cache_key]
        return PlayerSeasonStats(**cached_value) if cached_value is not None else None

    stats = fetch_player_season_stats(
        api_key,
        player_id,
        season_id,
        player_name=player_name,
        competition_id=competition_id,
        session=session,
    )
    cache[cache_key] = asdict(stats) if stats is not None else None
    write_cache(CACHE_CATEGORY_PLAYER_SEASON_STATS, today, cache, cache_dir=cache_dir)
    return stats


def fetch_daily_matches(
    api_key: str,
    target_date: date,
    *,
    session: Any | None = None,
    competition_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """
    GET /football/matches?date_from=...&date_to=...&status=scheduled for one
    calendar date (UTC). Returns raw match rows -- these are upcoming
    fixtures with no score yet, so they are NOT `historical_data.MatchResult`
    records (those are finished-match-only, used for fitting).

    This endpoint has no server-side competition filter, so on a given day
    it returns literally every scheduled match worldwide -- domestic
    leagues, qualifiers, friendlies, everything. Left unscoped, that means
    build_models_for_matches ends up fitting/fetching rosters+season-stats
    for every one of those competitions, which is exactly what triggered
    sustained TheStatsAPI rate limiting in production (hundreds of calls
    for competitions nobody was actually grading bets for that day).
    `competition_ids` (when given) filters the response down to just the
    competitions the caller actually cares about -- e.g. one tournament --
    client-side, cutting the downstream fan-out proportionally.
    """

    iso_date = target_date.isoformat()
    payload = thestatsapi_get(
        "/football/matches",
        api_key,
        params={"date_from": iso_date, "date_to": iso_date, "status": "scheduled", "per_page": 100},
        session=session,
    )
    matches = payload.get("data", []) or []
    if competition_ids is not None:
        allowed = {str(cid) for cid in competition_ids}
        matches = [match for match in matches if str(match.get("competition_id")) in allowed]
    return matches


def build_models_for_matches(
    api_key: str,
    matches: list[dict[str, Any]],
    *,
    session: Any | None = None,
    cache_dir: Path = CACHE_DIR,
    build_player_profiles: bool = True,
) -> dict[str, SoccerProjectionModel]:
    """
    One SoccerProjectionModel per competition_id represented in `matches`.

    Competitions with fewer than MIN_HISTORICAL_MATCHES_TO_FIT finished
    matches in the lookback window are skipped outright -- never force-fit
    on too little data. Callers should treat a missing competition_id in the
    returned dict the same way SoccerProjectionModel itself treats an
    unsupported market: "no coverage today", not a crash.

    `cache_dir` is exposed (rather than hardcoded to cache_store.CACHE_DIR)
    purely so tests can point it at a throwaway tmp_path instead of writing
    real files under engines/soccer_engine/cache/ on every test run.

    `build_player_profiles=False` skips the roster + season-stats fan-out
    entirely (one call per team, one per rostered player -- the bulk of
    this pipeline's request volume) and fits team ratings only. Set this
    when player props can't be graded anyway regardless of how good this
    data is -- e.g. TheStatsAPI's /odds/players endpoint 403ing with
    ADDON_REQUIRED because the account's plan doesn't include the
    player_odds add-on (see ev_engine_core.is_thestatsapi_addon_required) --
    so those calls aren't wasted quota/time on data nothing downstream can
    use.

    When `build_player_profiles=True`, profiles are only built for teams
    actually playing in `matches` (today's slate) -- NOT every team in the
    multi-year historical window used to fit that competition's Dixon-Coles
    ratings. A tournament with a long qualifying history (e.g. the World
    Cup) can have 10-25x more teams in its historical fit than are playing
    on any single day, and each extra team costs one roster fetch plus one
    season-stats fetch per rostered player -- against a rate-limited API,
    that difference is the gap between a few minutes and over an hour.
    """

    models: dict[str, SoccerProjectionModel] = {}
    # (competition_id, season_id) pairs, deduplicated -- season_id is
    # required by the player-season-stats endpoint below, and every match
    # row already carries its own season_id, so no extra lookup is needed.
    competitions = {
        (match.get("competition_id"), match.get("season_id")) for match in matches if match.get("competition_id")
    }

    # Team ids actually playing TODAY, grouped by competition -- deliberately
    # NOT ratings.matches_played.keys() (every team in the ~2-year historical
    # fit window), which on a tournament with a long qualifying history (e.g.
    # the World Cup) can be 10-25x more teams than are on today's slate. That
    # mismatch is what turned one day's player-prop build into an hour-plus
    # of rate-limited roster/season-stats fetches for ~50 teams when only 2
    # were actually needed to grade today's board.
    todays_team_ids_by_competition: dict[str, set[str]] = {}
    for match in matches:
        competition_id = match.get("competition_id")
        if not competition_id:
            continue
        team_ids = todays_team_ids_by_competition.setdefault(competition_id, set())
        for side in ("home_team", "away_team"):
            team_id = (match.get(side) or {}).get("id")
            if team_id:
                team_ids.add(str(team_id))

    today = datetime.now(timezone.utc).date()
    date_from = (today - timedelta(days=HISTORY_LOOKBACK_DAYS)).isoformat()
    date_to = today.isoformat()

    for competition_id, season_id in competitions:
        try:
            historical = _cached_fetch_historical_matches(
                api_key,
                competition_id,
                date_from=date_from,
                date_to=date_to,
                session=session,
                today=today,
                cache_dir=cache_dir,
            )
        except Exception:
            # One competition's historical-results fetch failing (rate
            # limit, network blip) must not cost every OTHER competition in
            # today's slate their model -- on a big multi-competition day
            # (e.g. a full round of world/continental fixtures) this is the
            # difference between "today's board is short one league" and
            # "today's board is empty and no alert goes out at all".
            logger.exception("Failed to fetch historical matches for competition %s; skipping.", competition_id)
            continue

        if len(historical) < MIN_HISTORICAL_MATCHES_TO_FIT:
            continue

        try:
            ratings = fit_dixon_coles(historical)
        except SoccerModelDataError:
            continue

        if not build_player_profiles:
            profiles: dict[str, Any] = {}
        else:
            try:
                profiles = _build_player_profiles_for_teams(
                    api_key,
                    todays_team_ids_by_competition.get(competition_id, set()),
                    season_id=season_id,
                    competition_id=competition_id,
                    session=session,
                    today=today,
                    cache_dir=cache_dir,
                )
            except Exception:
                # Team ratings already fit successfully -- grade this
                # competition's team-level markets with an empty player-prop
                # profile map (SoccerProjectionModel already handles "no
                # profile for this player" as a clean warning) rather than
                # losing the whole competition over player-prop coverage.
                logger.exception(
                    "Failed to build player profiles for competition %s; team markets only.", competition_id
                )
                profiles = {}

        models[competition_id] = SoccerProjectionModel(ratings, profiles)

    return models


def _build_player_profiles_for_teams(
    api_key: str,
    team_ids: Any,
    *,
    season_id: str | None,
    competition_id: str,
    session: Any | None,
    today: date,
    cache_dir: Path,
) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    if not season_id:
        # No season to query season-stats against -- return an empty profile
        # map rather than guessing a season_id; player props for this
        # competition will come back as a clean "no rate profile" warning
        # from SoccerProjectionModel instead of a fabricated number.
        return profiles

    for team_id in team_ids:
        try:
            team_players = _cached_fetch_team_players(
                api_key, team_id, session=session, today=today, cache_dir=cache_dir
            )
        except Exception:
            # One team's roster call failing (network blip, rate limit, a
            # transiently bad response) must not cost every OTHER team in
            # this competition their player-prop coverage for the day.
            logger.exception("Failed to fetch roster for team %s (competition %s); skipping.", team_id, competition_id)
            continue

        for player in team_players:
            player_id = player.get("id")
            if not player_id:
                continue
            try:
                stats = _cached_fetch_player_season_stats(
                    api_key,
                    player_id,
                    season_id,
                    player_name=player.get("name"),
                    competition_id=competition_id,
                    session=session,
                    today=today,
                    cache_dir=cache_dir,
                )
            except Exception:
                # Same reasoning as above, one level down: a single
                # player's stats call failing (e.g. a 429 mid-roster, a
                # flaky response) must not sink every other player's --
                # let alone every other TEAM's and COMPETITION's -- props
                # for the day. Dropped, not guessed, exactly like a missing
                # price is dropped elsewhere in this codebase.
                logger.exception(
                    "Failed to fetch season stats for player %s (team %s, competition %s); skipping.",
                    player_id,
                    team_id,
                    competition_id,
                )
                continue
            if stats is None:
                continue
            profile = build_player_rate_profile(stats)
            if profile is not None:
                profiles[profile.player_id] = profile

    return profiles
