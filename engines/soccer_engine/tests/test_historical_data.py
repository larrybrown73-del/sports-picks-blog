from __future__ import annotations

from historical_data import fetch_player_season_stats, fetch_team_players


class _HTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = type("_Response", (), {"status_code": status_code})()


class _FakeResponse:
    def __init__(self, json_data: dict, *, status_code: int | None = None) -> None:
        self._json_data = json_data
        self._status_code = status_code

    def raise_for_status(self) -> None:
        if self._status_code is not None:
            raise _HTTPError(self._status_code)

    def json(self) -> dict:
        return self._json_data


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def get(self, url: str, *, headers=None, params=None, timeout=None):
        return self._response


def test_fetch_player_season_stats_returns_none_on_404() -> None:
    """
    A 404 means TheStatsAPI has no stats page at all for this
    player/season/competition (e.g. a new signing who hasn't featured yet)
    -- treated identically to a zero-minutes payload, never allowed to
    raise and crash the caller's roster-building loop.
    """

    session = _FakeSession(_FakeResponse({}, status_code=404))

    result = fetch_player_season_stats("test_key", "pl_1", "sn_1", session=session)

    assert result is None


def test_fetch_player_season_stats_reraises_non_404_errors() -> None:
    session = _FakeSession(_FakeResponse({}, status_code=500))

    try:
        fetch_player_season_stats("test_key", "pl_1", "sn_1", session=session)
        raise AssertionError("expected the 500 error to propagate")
    except Exception as exc:
        assert "500" in str(exc)


class _CapturingRosterSession:
    """
    Records the exact URL requested -- unlike _FakeSession above, which
    always returns the same fixed response regardless of URL.
    """

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.requested_urls: list[str] = []

    def get(self, url: str, *, headers=None, params=None, timeout=None):
        self.requested_urls.append(url)
        return self._response


def test_fetch_team_players_uses_teams_players_endpoint_not_players_team_id() -> None:
    """
    GET /football/players?team_id=... filters by a player's *current club*
    (current_team), which is a DIFFERENT field from the roster being asked
    for here -- it silently returns an empty list (0 players, no error)
    for any national/international team, since a player's current_team is
    their club, not their national_team. That bug meant every World Cup
    team's player-prop coverage silently built zero profiles. The correct
    endpoint is GET /football/teams/{team_id}/players.
    """

    session = _CapturingRosterSession(_FakeResponse({"data": [{"id": "pl_1", "name": "Test Player"}]}))

    players = fetch_team_players("test_key", "tm_70912", session=session)

    assert players == [{"id": "pl_1", "name": "Test Player"}]
    assert len(session.requested_urls) == 1
    assert "/football/teams/tm_70912/players" in session.requested_urls[0]
    assert "team_id=" not in session.requested_urls[0]


def test_fetch_player_season_stats_parses_a_real_payload() -> None:
    payload = {
        "data": {
            "player_id": "pl_1",
            "team_id": "tm_1",
            "position": "F",
            "season_id": "sn_1",
            "minutes_played": 900,
            "appearances": 10,
            "scoring": {"goals": 5, "assists": 2},
            "shooting": {"total_shots": 30, "shots_on_target": 15},
        }
    }
    session = _FakeSession(_FakeResponse(payload))

    result = fetch_player_season_stats("test_key", "pl_1", "sn_1", player_name="Test Player", session=session)

    assert result is not None
    assert result.player_name == "Test Player"
    assert result.minutes_played == 900
    assert result.goals == 5


def test_fetch_player_season_stats_returns_none_for_zero_minutes() -> None:
    payload = {"data": {"player_id": "pl_1", "minutes_played": 0}}
    session = _FakeSession(_FakeResponse(payload))

    assert fetch_player_season_stats("test_key", "pl_1", "sn_1", session=session) is None
