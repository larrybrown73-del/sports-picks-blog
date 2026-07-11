from __future__ import annotations

import pytest

from lineups import NOT_IN_SQUAD, STARTING, SUBSTITUTE, LineupNotAvailable, MatchLineup, fetch_match_lineup


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
        self.requests: list[str] = []

    def get(self, url: str, *, headers=None, params=None, timeout=None):
        self.requests.append(url)
        return self._response


def test_fetch_match_lineup_parses_starting_xi_and_substitutes_from_both_teams() -> None:
    payload = {
        "data": {
            "match_id": "mt_1",
            "confirmed": True,
            "home": {
                "id": "tm_home",
                "starting_xi": [{"id": "p_1"}, {"id": "p_2"}],
                "substitutes": [{"id": "p_3"}],
            },
            "away": {
                "id": "tm_away",
                "starting_xi": [{"id": "p_10"}],
                "substitutes": [{"id": "p_11"}, {"id": "p_12"}],
            },
        }
    }
    session = _FakeSession(_FakeResponse(payload))

    lineup = fetch_match_lineup("test_key", "mt_1", session=session)

    assert lineup.match_id == "mt_1"
    assert lineup.confirmed is True
    assert lineup.starting_player_ids == frozenset({"p_1", "p_2", "p_10"})
    assert lineup.substitute_player_ids == frozenset({"p_3", "p_11", "p_12"})
    assert lineup.home_team_id == "tm_home"
    assert lineup.away_team_id == "tm_away"
    assert session.requests == ["https://api.thestatsapi.com/api/football/matches/mt_1/lineups"]


def test_fetch_match_lineup_defaults_confirmed_to_false_when_absent() -> None:
    payload = {"data": {"match_id": "mt_1", "home": {}, "away": {}}}
    session = _FakeSession(_FakeResponse(payload))

    lineup = fetch_match_lineup("test_key", "mt_1", session=session)

    assert lineup.confirmed is False


def test_fetch_match_lineup_raises_lineup_not_available_on_404() -> None:
    session = _FakeSession(_FakeResponse({}, status_code=404))

    with pytest.raises(LineupNotAvailable):
        fetch_match_lineup("test_key", "mt_1", session=session)


def test_fetch_match_lineup_reraises_non_404_errors() -> None:
    session = _FakeSession(_FakeResponse({}, status_code=500))

    with pytest.raises(Exception):
        fetch_match_lineup("test_key", "mt_1", session=session)


def test_match_lineup_status_for_starting_substitute_and_not_in_squad() -> None:
    lineup = MatchLineup(
        match_id="mt_1",
        confirmed=True,
        starting_player_ids=frozenset({"p_1"}),
        substitute_player_ids=frozenset({"p_2"}),
        home_team_id="tm_home",
        away_team_id="tm_away",
    )

    assert lineup.status_for("p_1") == STARTING
    assert lineup.status_for("p_2") == SUBSTITUTE
    assert lineup.status_for("p_99") == NOT_IN_SQUAD
