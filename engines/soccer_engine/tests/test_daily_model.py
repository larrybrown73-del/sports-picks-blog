from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from cache_store import cache_path
from daily_model import (
    CACHE_CATEGORY_HISTORICAL_MATCHES,
    CACHE_CATEGORY_PLAYER_SEASON_STATS,
    CACHE_CATEGORY_TEAM_PLAYERS,
    build_models_for_matches,
    fetch_daily_matches,
)
from ev_engine_core import MarketLeg


class _FakeResponse:
    def __init__(self, json_data: dict) -> None:
        self._json_data = json_data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._json_data


class _FakeSession:
    """
    Routes GET calls by URL path so a single fake session can stand in for
    every fetch_* function daily_model.py calls (matches, players,
    player-stats) without hitting the network.
    """

    def __init__(self, responses_by_path: dict[str, dict | list[dict]]) -> None:
        self._responses_by_path = responses_by_path
        self.requests: list[tuple[str, dict]] = []

    def get(self, url: str, *, headers=None, params=None, timeout=None):
        path = url.split("thestatsapi.com/api", 1)[-1].split("?")[0]
        self.requests.append((path, params or {}))

        if path == "/football/matches":
            return _FakeResponse(self._paginated_matches(params or {}))
        if path.startswith("/football/teams/") and path.endswith("/players"):
            # Real endpoint: GET /football/teams/{team_id}/players -- kept
            # under the same "/football/players" fixture key across all
            # team_ids for simplicity, since these tests only ever exercise
            # one team's roster at a time.
            return _FakeResponse({"data": self._responses_by_path.get("/football/players", [])})
        if path.endswith("/stats"):
            player_id = path.split("/")[3]
            return _FakeResponse({"data": self._responses_by_path.get(f"/stats/{player_id}", {})})
        raise AssertionError(f"Unexpected path requested in test: {path}")

    def _paginated_matches(self, params: dict) -> dict:
        rows = self._responses_by_path.get("/football/matches", [])
        return {"data": rows, "meta": {"total_pages": 1}}


def test_fetch_daily_matches_uses_scheduled_status_and_date_range() -> None:
    session = _FakeSession({"/football/matches": [{"id": "mt_1"}]})
    matches = fetch_daily_matches("test_key", date(2026, 7, 10), session=session)

    assert matches == [{"id": "mt_1"}]
    path, params = session.requests[0]
    assert path == "/football/matches"
    assert params["date_from"] == "2026-07-10"
    assert params["date_to"] == "2026-07-10"
    assert params["status"] == "scheduled"


def test_fetch_daily_matches_filters_by_competition_ids() -> None:
    """
    /football/matches has no server-side competition filter -- it returns
    every match scheduled worldwide for the day. Left unfiltered, the
    downstream pipeline fans out to every one of those competitions'
    rosters/season-stats, which is what triggered sustained TheStatsAPI
    rate limiting in production. `competition_ids` filters this
    client-side to just the competitions the caller cares about.
    """

    session = _FakeSession(
        {
            "/football/matches": [
                {"id": "mt_1", "competition_id": "comp_world_cup"},
                {"id": "mt_2", "competition_id": "comp_random_domestic_league"},
                {"id": "mt_3", "competition_id": "comp_world_cup"},
            ]
        }
    )

    matches = fetch_daily_matches("test_key", date(2026, 7, 7), session=session, competition_ids={"comp_world_cup"})

    assert {m["id"] for m in matches} == {"mt_1", "mt_3"}


def test_fetch_daily_matches_returns_everything_when_competition_ids_not_given() -> None:
    session = _FakeSession(
        {
            "/football/matches": [
                {"id": "mt_1", "competition_id": "comp_a"},
                {"id": "mt_2", "competition_id": "comp_b"},
            ]
        }
    )

    matches = fetch_daily_matches("test_key", date(2026, 7, 7), session=session)

    assert {m["id"] for m in matches} == {"mt_1", "mt_2"}


def _finished_match_row(match_id: str, home_id: str, away_id: str, home_goals: int, away_goals: int, day: int) -> dict:
    return {
        "id": match_id,
        "competition_id": "comp_1",
        "utc_date": f"2025-01-{day:02d}T15:00:00.000Z",
        "home_team": {"id": home_id, "name": home_id},
        "away_team": {"id": away_id, "name": away_id},
        "score": {"home": home_goals, "away": away_goals},
    }


def _todays_match_row(
    match_id: str,
    competition_id: str,
    season_id: str | None,
    home_id: str = "tm_A",
    away_id: str = "tm_B",
) -> dict:
    """
    A "today's schedule" row shaped like the real /football/matches payload
    (nested home_team/away_team objects) -- build_models_for_matches scopes
    player-profile building to exactly the team ids found here, so a stub
    missing these fields would silently resolve to zero teams and zero
    profiles regardless of what the historical/roster fixtures provide.
    """
    return {
        "id": match_id,
        "competition_id": competition_id,
        "season_id": season_id,
        "home_team": {"id": home_id, "name": home_id},
        "away_team": {"id": away_id, "name": away_id},
    }


def test_build_models_for_matches_skips_competitions_with_too_little_history(tmp_path: Path) -> None:
    todays_matches = [_todays_match_row("mt_today", "comp_sparse", "sn_1")]
    # Only 3 finished matches on record -- below MIN_HISTORICAL_MATCHES_TO_FIT.
    session = _FakeSession(
        {
            "/football/matches": [
                _finished_match_row(f"mt_{i}", "tm_A", "tm_B", 1, 1, i + 1) for i in range(3)
            ],
        }
    )
    models = build_models_for_matches("test_key", todays_matches, session=session, cache_dir=tmp_path)
    assert models == {}


def test_build_models_for_matches_fits_ratings_and_player_profiles(tmp_path: Path) -> None:
    todays_matches = [_todays_match_row("mt_today", "comp_1", "sn_1")]

    teams = ["tm_A", "tm_B"]
    finished_rows = []
    counter = 0
    for _ in range(6):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                finished_rows.append(_finished_match_row(f"mt_{counter}", home, away, 1, 1, (counter % 27) + 1))
                counter += 1

    roster = [{"id": "pl_1", "name": "Test Player"}]
    player_stats = {
        "player_id": "pl_1",
        "team_id": "tm_A",
        "position": "F",
        "season_id": "sn_1",
        "minutes_played": 900,
        "appearances": 10,
        "scoring": {"goals": 5, "assists": 2},
        "shooting": {"total_shots": 30, "shots_on_target": 15},
    }

    session = _FakeSession(
        {
            "/football/matches": finished_rows,
            "/football/players": roster,
            "/stats/pl_1": player_stats,
        }
    )

    models = build_models_for_matches("test_key", todays_matches, session=session, cache_dir=tmp_path)

    assert "comp_1" in models
    model = models["comp_1"]

    home_leg = MarketLeg(
        game_id="mt_x",
        market_type="match_odds",
        selection="tm_A",
        odds=2.0,
        odds_format="decimal",
        sportsbook="Bet365",
        side="home",
        metadata={"home_team_id": "tm_A", "away_team_id": "tm_B"},
    )
    team_estimate = model(home_leg)
    assert 0.0 < team_estimate.true_probability < 1.0  # team ratings fit successfully for both teams

    player_leg = MarketLeg(
        game_id="mt_x",
        market_type="player_shots",
        selection="Test Player Over 1.5",
        odds=2.0,
        odds_format="decimal",
        sportsbook="Bet365",
        entity_id="pl_1",
        entity_name="Test Player",
        line=1.5,
        metadata={"home_team_id": "tm_A", "away_team_id": "tm_B"},
    )
    player_estimate = model(player_leg)
    assert not any("No player rate profile" in w for w in player_estimate.warnings)


def test_build_models_for_matches_scopes_player_profiles_to_todays_teams_only(tmp_path: Path) -> None:
    """
    Regression test: player profiles must be built only for teams playing
    TODAY, not every team in the multi-year historical window used to fit
    Dixon-Coles ratings. Production hit this exactly -- the World Cup's
    historical fit pulled in ~50 teams while only 2 played on a given day,
    turning a few minutes of roster/season-stats fetches into over an hour
    against a rate-limited API.

    tm_C here is a THIRD team that appears in the historical results (so it
    IS part of ratings.matches_played) but is NOT in today's match -- its
    roster must never be requested.
    """

    todays_matches = [_todays_match_row("mt_today", "comp_1", "sn_1")]

    teams = ["tm_A", "tm_B", "tm_C"]
    finished_rows = []
    counter = 0
    for _ in range(6):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                finished_rows.append(_finished_match_row(f"mt_{counter}", home, away, 1, 1, (counter % 27) + 1))
                counter += 1

    roster = [{"id": "pl_1", "name": "Test Player"}]
    player_stats = {
        "player_id": "pl_1",
        "team_id": "tm_A",
        "position": "F",
        "season_id": "sn_1",
        "minutes_played": 900,
        "appearances": 10,
        "scoring": {"goals": 5, "assists": 2},
        "shooting": {"total_shots": 30, "shots_on_target": 15},
    }
    session = _FakeSession(
        {"/football/matches": finished_rows, "/football/players": roster, "/stats/pl_1": player_stats}
    )

    models = build_models_for_matches("test_key", todays_matches, session=session, cache_dir=tmp_path)

    assert "comp_1" in models
    # tm_C DID play enough historical matches to be part of the Dixon-Coles
    # fit -- confirming the roster skip below is genuinely about today's
    # slate, not an accidental side effect of tm_C being unrated.
    assert "tm_C" in models["comp_1"]._team_ratings.matches_played

    roster_paths_requested = {path for path, _ in session.requests if path.endswith("/players") and "/teams/" in path}
    assert roster_paths_requested == {"/football/teams/tm_A/players", "/football/teams/tm_B/players"}


def test_build_models_for_matches_skips_roster_fetch_when_player_profiles_disabled(tmp_path: Path) -> None:
    """
    build_player_profiles=False must skip the roster/season-stats fan-out
    entirely -- not just discard its results -- since the whole point is
    to avoid spending that quota when player props can't be graded anyway
    (e.g. TheStatsAPI's player_odds add-on isn't on the account's plan).
    """

    todays_matches = [_todays_match_row("mt_today", "comp_1", "sn_1")]
    teams = ["tm_A", "tm_B"]
    finished_rows = []
    counter = 0
    for _ in range(6):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                finished_rows.append(_finished_match_row(f"mt_{counter}", home, away, 1, 1, (counter % 27) + 1))
                counter += 1

    session = _FakeSession({"/football/matches": finished_rows})

    models = build_models_for_matches(
        "test_key", todays_matches, session=session, cache_dir=tmp_path, build_player_profiles=False
    )

    requested_paths = {path for path, _ in session.requests}
    assert "/football/players" not in requested_paths
    assert not any(path.endswith("/stats") for path in requested_paths)

    assert "comp_1" in models
    model = models["comp_1"]
    home_leg = MarketLeg(
        game_id="mt_x",
        market_type="match_odds",
        selection="tm_A",
        odds=2.0,
        odds_format="decimal",
        sportsbook="Bet365",
        side="home",
        metadata={"home_team_id": "tm_A", "away_team_id": "tm_B"},
    )
    assert 0.0 < model(home_leg).true_probability < 1.0  # team markets still work fine


def test_build_models_for_matches_skips_player_profiles_without_season_id(tmp_path: Path) -> None:
    todays_matches = [_todays_match_row("mt_today", "comp_1", None)]
    teams = ["tm_A", "tm_B"]
    finished_rows = []
    counter = 0
    for _ in range(6):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                finished_rows.append(_finished_match_row(f"mt_{counter}", home, away, 1, 1, (counter % 27) + 1))
                counter += 1

    session = _FakeSession({"/football/matches": finished_rows})
    models = build_models_for_matches("test_key", todays_matches, session=session, cache_dir=tmp_path)

    assert "comp_1" in models
    model = models["comp_1"]

    player_leg = MarketLeg(
        game_id="mt_x",
        market_type="player_shots",
        selection="Test Player Over 1.5",
        odds=2.0,
        odds_format="decimal",
        sportsbook="Bet365",
        entity_id="pl_1",
        entity_name="Test Player",
        line=1.5,
        metadata={"home_team_id": "tm_A", "away_team_id": "tm_B"},
    )
    player_estimate = model(player_leg)
    assert any("No player rate profile" in w for w in player_estimate.warnings)


class _HTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = type("_Response", (), {"status_code": status_code})()


def test_build_models_for_matches_skips_player_when_stats_fetch_raises(tmp_path: Path) -> None:
    """
    One player's season-stats call blowing up (e.g. a 429 mid-roster) must
    not cost every other player -- let alone the whole competition -- their
    player-prop coverage for the day.
    """

    todays_matches = [_todays_match_row("mt_today", "comp_1", "sn_1")]
    teams = ["tm_A", "tm_B"]
    finished_rows = []
    counter = 0
    for _ in range(6):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                finished_rows.append(_finished_match_row(f"mt_{counter}", home, away, 1, 1, (counter % 27) + 1))
                counter += 1

    roster = [{"id": "pl_flaky", "name": "Flaky Player"}, {"id": "pl_ok", "name": "OK Player"}]
    player_stats_ok = {
        "player_id": "pl_ok",
        "team_id": "tm_A",
        "position": "F",
        "season_id": "sn_1",
        "minutes_played": 900,
        "appearances": 10,
        "scoring": {"goals": 5, "assists": 2},
        "shooting": {"total_shots": 30, "shots_on_target": 15},
    }

    class _FlakyStatsSession(_FakeSession):
        def get(self, url: str, *, headers=None, params=None, timeout=None):
            if url.endswith("/football/players/pl_flaky/stats"):
                raise _HTTPError(429)
            return super().get(url, headers=headers, params=params, timeout=timeout)

    session = _FlakyStatsSession(
        {
            "/football/matches": finished_rows,
            "/football/players": roster,
            "/stats/pl_ok": player_stats_ok,
        }
    )

    models = build_models_for_matches("test_key", todays_matches, session=session, cache_dir=tmp_path)

    assert "comp_1" in models  # the one flaky player didn't sink the whole competition's model
    model = models["comp_1"]

    ok_player_leg = MarketLeg(
        game_id="mt_x",
        market_type="player_shots",
        selection="OK Player Over 1.5",
        odds=2.0,
        odds_format="decimal",
        sportsbook="Bet365",
        entity_id="pl_ok",
        entity_name="OK Player",
        line=1.5,
        metadata={"home_team_id": "tm_A", "away_team_id": "tm_B"},
    )
    assert not any("No player rate profile" in w for w in model(ok_player_leg).warnings)


def test_build_models_for_matches_skips_team_when_roster_fetch_raises(tmp_path: Path) -> None:
    """One team's roster call blowing up must not sink every OTHER team in that competition's player coverage."""

    todays_matches = [_todays_match_row("mt_today", "comp_1", "sn_1")]
    teams = ["tm_A", "tm_B"]
    finished_rows = []
    counter = 0
    for _ in range(6):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                finished_rows.append(_finished_match_row(f"mt_{counter}", home, away, 1, 1, (counter % 27) + 1))
                counter += 1

    roster = [{"id": "pl_ok", "name": "OK Player"}]
    player_stats_ok = {
        "player_id": "pl_ok",
        "team_id": "tm_B",
        "position": "F",
        "season_id": "sn_1",
        "minutes_played": 900,
        "appearances": 10,
        "scoring": {"goals": 5, "assists": 2},
        "shooting": {"total_shots": 30, "shots_on_target": 15},
    }

    class _FlakyRosterSession(_FakeSession):
        def get(self, url: str, *, headers=None, params=None, timeout=None):
            if url.endswith("/football/players") and (params or {}).get("team_id") == "tm_A":
                raise _HTTPError(429)
            return super().get(url, headers=headers, params=params, timeout=timeout)

    session = _FlakyRosterSession(
        {
            "/football/matches": finished_rows,
            "/football/players": roster,
            "/stats/pl_ok": player_stats_ok,
        }
    )

    models = build_models_for_matches("test_key", todays_matches, session=session, cache_dir=tmp_path)

    assert "comp_1" in models  # tm_A's flaky roster call didn't sink the whole competition's model
    model = models["comp_1"]

    ok_player_leg = MarketLeg(
        game_id="mt_x",
        market_type="player_shots",
        selection="OK Player Over 1.5",
        odds=2.0,
        odds_format="decimal",
        sportsbook="Bet365",
        entity_id="pl_ok",
        entity_name="OK Player",
        line=1.5,
        metadata={"home_team_id": "tm_A", "away_team_id": "tm_B"},
    )
    assert not any("No player rate profile" in w for w in model(ok_player_leg).warnings)


def test_build_models_for_matches_skips_competition_when_historical_fetch_raises(tmp_path: Path) -> None:
    """
    A multi-competition slate (e.g. a full day of world/continental
    fixtures) where ONE competition's historical-results call blows up
    (rate limit, network blip) must still produce models for every OTHER
    competition -- not come back empty and silently cost the whole day's
    alert.
    """

    todays_matches = [
        _todays_match_row("mt_flaky", "comp_flaky", "sn_1"),
        _todays_match_row("mt_ok", "comp_ok", "sn_1"),
    ]
    teams = ["tm_A", "tm_B"]
    finished_rows = []
    counter = 0
    for _ in range(6):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                finished_rows.append(_finished_match_row(f"mt_{counter}", home, away, 1, 1, (counter % 27) + 1))
                counter += 1

    class _FlakyHistorySession(_FakeSession):
        def get(self, url: str, *, headers=None, params=None, timeout=None):
            if url.endswith("/football/matches") and (params or {}).get("competition_id") == "comp_flaky":
                raise _HTTPError(429)
            return super().get(url, headers=headers, params=params, timeout=timeout)

        def _paginated_matches(self, params: dict) -> dict:
            rows = [row for row in finished_rows if row["competition_id"] == params.get("competition_id")]
            return {"data": rows, "meta": {"total_pages": 1}}

    # Both competitions share the same finished_rows pool tagged "comp_1" by
    # _finished_match_row -- retag half of them "comp_ok" so the fake
    # session's per-competition filter actually has matches to return.
    for row in finished_rows:
        row["competition_id"] = "comp_ok"

    session = _FlakyHistorySession({"/football/matches": finished_rows, "/football/players": []})

    models = build_models_for_matches("test_key", todays_matches, session=session, cache_dir=tmp_path)

    assert "comp_flaky" not in models  # the flaky competition is cleanly dropped, not crashing the whole run
    assert "comp_ok" in models  # every other competition still gets graded


def _slate_with_full_history(competition_id: str = "comp_1", season_id: str = "sn_1") -> tuple[list[dict], list[dict]]:
    todays_matches = [_todays_match_row("mt_today", competition_id, season_id)]
    teams = ["tm_A", "tm_B"]
    finished_rows = []
    counter = 0
    for _ in range(6):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                finished_rows.append(_finished_match_row(f"mt_{counter}", home, away, 1, 1, (counter % 27) + 1))
                counter += 1
    return todays_matches, finished_rows


def test_build_models_for_matches_writes_cache_files_under_cache_dir(tmp_path: Path) -> None:
    todays_matches, finished_rows = _slate_with_full_history()
    roster = [{"id": "pl_1", "name": "Test Player"}]
    player_stats = {
        "player_id": "pl_1",
        "team_id": "tm_A",
        "position": "F",
        "season_id": "sn_1",
        "minutes_played": 900,
        "appearances": 10,
        "scoring": {"goals": 5, "assists": 2},
        "shooting": {"total_shots": 30, "shots_on_target": 15},
    }
    session = _FakeSession(
        {"/football/matches": finished_rows, "/football/players": roster, "/stats/pl_1": player_stats}
    )

    build_models_for_matches("test_key", todays_matches, session=session, cache_dir=tmp_path)

    today = date.today()
    assert cache_path(CACHE_CATEGORY_HISTORICAL_MATCHES, today, cache_dir=tmp_path).exists()
    assert cache_path(CACHE_CATEGORY_TEAM_PLAYERS, today, cache_dir=tmp_path).exists()
    assert cache_path(CACHE_CATEGORY_PLAYER_SEASON_STATS, today, cache_dir=tmp_path).exists()


def test_build_models_for_matches_second_run_hits_cache_not_the_api(tmp_path: Path) -> None:
    todays_matches, finished_rows = _slate_with_full_history()
    roster = [{"id": "pl_1", "name": "Test Player"}]
    player_stats = {
        "player_id": "pl_1",
        "team_id": "tm_A",
        "position": "F",
        "season_id": "sn_1",
        "minutes_played": 900,
        "appearances": 10,
        "scoring": {"goals": 5, "assists": 2},
        "shooting": {"total_shots": 30, "shots_on_target": 15},
    }
    session = _FakeSession(
        {"/football/matches": finished_rows, "/football/players": roster, "/stats/pl_1": player_stats}
    )

    first_models = build_models_for_matches("test_key", todays_matches, session=session, cache_dir=tmp_path)
    requests_after_first_run = len(session.requests)
    assert requests_after_first_run > 0

    # A second run against the SAME cache_dir/session must be served entirely
    # from disk -- this is the whole point of the cache (protecting quota
    # on retries/reruns within the same day).
    second_models = build_models_for_matches("test_key", todays_matches, session=session, cache_dir=tmp_path)
    assert len(session.requests) == requests_after_first_run  # no new HTTP calls at all

    assert set(first_models.keys()) == set(second_models.keys()) == {"comp_1"}


def test_build_models_for_matches_caches_players_with_no_season_stats(tmp_path: Path) -> None:
    todays_matches, finished_rows = _slate_with_full_history()
    # Roster player pl_2 has no stats -- fetch_player_season_stats returns
    # None for them (empty dict has no minutes_played).
    roster = [{"id": "pl_2", "name": "No Stats Player"}]
    session = _FakeSession({"/football/matches": finished_rows, "/football/players": roster, "/stats/pl_2": {}})

    build_models_for_matches("test_key", todays_matches, session=session, cache_dir=tmp_path)
    requests_after_first_run = len(session.requests)

    # Re-run: even a cached "no stats" answer for pl_2 must short-circuit
    # the API call, not be treated as "never fetched, try again".
    build_models_for_matches("test_key", todays_matches, session=session, cache_dir=tmp_path)
    assert len(session.requests) == requests_after_first_run

    cache = json.loads(cache_path(CACHE_CATEGORY_PLAYER_SEASON_STATS, date.today(), cache_dir=tmp_path).read_text())
    assert cache["pl_2|sn_1"] is None
