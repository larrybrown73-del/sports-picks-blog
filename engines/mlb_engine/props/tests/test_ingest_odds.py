import pandas as pd
import pytest

from baseball_props.data.ingest import (
    _matches_rundown_team_label,
    _matches_team_abbrev,
    is_odds_event_id,
    resolve_odds_event_ids_for_slate,
)


def test_matches_rundown_team_label_city_to_full_name() -> None:
    assert _matches_rundown_team_label("Washington", "Washington Nationals")
    assert _matches_rundown_team_label("Chicago", "Chicago Cubs")
    assert _matches_rundown_team_label("Milwaukee", "Milwaukee Brewers")
    assert _matches_rundown_team_label("Oakland", "Athletics")


def test_matches_team_abbrev_rundown_city_labels() -> None:
    assert _matches_team_abbrev("WSH", "Washington")
    assert _matches_team_abbrev("BAL", "Baltimore")
    assert _matches_team_abbrev("CHC", "Chicago")
    assert _matches_team_abbrev("LAD", "Los Angeles")
    assert _matches_team_abbrev("ATH", "Oakland")


def test_is_odds_event_id_rejects_mock_ids() -> None:
    assert not is_odds_event_id("G001")
    assert is_odds_event_id("746123")
    assert is_odds_event_id("9876543210123456")
    assert is_odds_event_id("a1b2c3d4e5f6789012345678901234ab")


def test_resolve_odds_event_ids_matches_by_team() -> None:
    slate_games = pd.DataFrame(
        [
            {
                "game_id": "746123",
                "home_team_id": "NYY",
                "away_team_id": "BOS",
                "mlb_game_pk": 746123,
            }
        ]
    )
    live_vegas = pd.DataFrame(
        [
            {
                "game_id": "a1b2c3d4e5f6789012345678901234ab",
                "home_team": "New York Yankees",
                "away_team": "Boston Red Sox",
                "home_implied_runs": 4.5,
                "away_implied_runs": 4.0,
                "game_total": 8.5,
            }
        ]
    )
    ids = resolve_odds_event_ids_for_slate(slate_games, live_vegas)
    assert ids == ["a1b2c3d4e5f6789012345678901234ab"]


def test_resolve_odds_event_ids_matches_az_to_arizona() -> None:
    slate_games = pd.DataFrame(
        [
            {
                "game_id": "822960",
                "home_team_id": "TB",
                "away_team_id": "AZ",
                "mlb_game_pk": 822960,
            }
        ]
    )
    live_vegas = pd.DataFrame(
        [
            {
                "game_id": "c3d4e5f6789012345678901234abcdef01",
                "home_team": "Tampa Bay Rays",
                "away_team": "Arizona Diamondbacks",
                "home_implied_runs": 4.3,
                "away_implied_runs": 4.1,
                "game_total": 8.4,
            }
        ]
    )
    ids = resolve_odds_event_ids_for_slate(slate_games, live_vegas)
    assert ids == ["c3d4e5f6789012345678901234abcdef01"]


def test_resolve_odds_event_ids_uses_rundown_schedule_when_vegas_sparse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slate_games = pd.DataFrame(
        [
            {"game_id": "1", "home_team_id": "BAL", "away_team_id": "WSH", "mlb_game_pk": 1},
            {"game_id": "2", "home_team_id": "NYM", "away_team_id": "PHI", "mlb_game_pk": 2},
        ]
    )
    live_vegas = pd.DataFrame(
        [
            {
                "game_id": "only-one-game-hash",
                "home_team": "San Francisco",
                "away_team": "Atlanta",
                "home_implied_runs": 4.5,
                "away_implied_runs": 4.0,
                "game_total": 8.5,
            }
        ]
    )
    rundown = pd.DataFrame(
        [
            {"event_id": "bal-wsh-hash", "home_team": "Baltimore", "away_team": "Washington"},
            {"event_id": "nym-phi-hash", "home_team": "New York", "away_team": "Philadelphia"},
        ]
    )
    monkeypatch.setattr(
        "baseball_props.data.therundown.fetch_rundown_event_list",
        lambda **kwargs: rundown,
    )
    ids = resolve_odds_event_ids_for_slate(slate_games, live_vegas)
    assert ids == ["bal-wsh-hash", "nym-phi-hash"]
