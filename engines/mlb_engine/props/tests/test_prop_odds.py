from unittest.mock import MagicMock, patch

import pytest

from baseball_props.data import ingest
from baseball_props.data.prop_odds import (
    PropOddsError,
    adapt_prop_odds_event_payload,
    adapt_prop_odds_games_list,
    adapt_prop_odds_market_payload,
    fetch_from_prop_odds,
)
from baseball_props.data.serpapi import SerpApiError
from baseball_props.logging_utils import reset_log_once


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "ODDS_CACHE_DIR", tmp_path)
    reset_log_once()
    yield tmp_path
    reset_log_once()


def test_adapt_prop_odds_games_list() -> None:
    payload = {
        "games": [
            {
                "game_id": "GAME123456789012345678901234567",
                "home_team": "New York Yankees",
                "away_team": "Boston Red Sox",
                "start_timestamp": "2026-06-27T23:05:00Z",
            }
        ]
    }
    result = adapt_prop_odds_games_list(payload)
    assert len(result) == 1
    assert result[0]["id"] == "GAME123456789012345678901234567"
    assert result[0]["home_team"] == "New York Yankees"


def test_adapt_prop_odds_market_payload_player_props() -> None:
    payload = {
        "sportsbooks": [
            {
                "bookie_key": "draftkings",
                "market": {
                    "outcomes": [
                        {
                            "name": "Over - Aaron Judge",
                            "price": -110,
                            "line": 1.5,
                        },
                        {
                            "name": "Under - Aaron Judge",
                            "price": -110,
                            "line": 1.5,
                        },
                    ]
                },
            }
        ]
    }
    books = adapt_prop_odds_market_payload(
        payload,
        odds_api_market_key="batter_total_bases",
    )
    assert len(books) == 1
    assert books[0]["key"] == "draftkings"
    outcomes = books[0]["markets"][0]["outcomes"]
    assert outcomes[0]["name"] == "Over"
    assert outcomes[0]["description"] == "Aaron Judge"
    assert outcomes[0]["point"] == 1.5


def test_adapt_prop_odds_event_payload_merges_markets() -> None:
    game = {
        "game_id": "GAME123456789012345678901234567",
        "home_team": "New York Yankees",
        "away_team": "Boston Red Sox",
    }
    tb_payload = {
        "sportsbooks": [
            {
                "bookie_key": "draftkings",
                "market": {
                    "outcomes": [
                        {"name": "Over - Aaron Judge", "price": -110, "line": 1.5},
                        {"name": "Under - Aaron Judge", "price": -110, "line": 1.5},
                    ]
                },
            }
        ]
    }
    outs_payload = {
        "sportsbooks": [
            {
                "bookie_key": "draftkings",
                "market": {
                    "outcomes": [
                        {"name": "Over - Gerrit Cole", "price": -115, "line": 17.5},
                        {"name": "Under - Gerrit Cole", "price": -105, "line": 17.5},
                    ]
                },
            }
        ]
    }
    event = adapt_prop_odds_event_payload(
        game,
        {
            "batter_total_bases": tb_payload,
            "pitcher_outs": outs_payload,
        },
    )
    assert event["id"] == "GAME123456789012345678901234567"
    assert len(event["bookmakers"]) == 1
    market_keys = {m["key"] for m in event["bookmakers"][0]["markets"]}
    assert market_keys == {"batter_total_bases", "pitcher_outs"}


def test_fetch_odds_api_json_401_tries_prop_odds_after_serpapi_fails(cache_dir) -> None:
    cache_key = "test:prop_odds_first"
    prop_payload = [{"id": "from-prop-odds"}]

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {}

    with patch("baseball_props.data.ingest.requests.get", return_value=mock_response), patch(
        "baseball_props.data.serpapi.fetch_from_serpapi",
        side_effect=SerpApiError("unavailable"),
    ), patch(
        "baseball_props.data.prop_odds.fetch_from_prop_odds",
        return_value=prop_payload,
    ) as mock_prop:
        result = ingest.fetch_odds_api_json(
            cache_key,
            "https://example.com/odds",
            {"apiKey": "bad"},
        )

    mock_prop.assert_called_once()
    assert result == prop_payload
    assert ingest.read_odds_cache(cache_key) == prop_payload


def test_fetch_odds_api_json_401_falls_back_to_stale_when_prop_odds_fails(cache_dir) -> None:
    cache_key = "test:stale_after_prop_odds"
    stale_payload = [{"id": "stale-event"}]
    ingest.write_odds_cache(cache_key, stale_payload)
    path = ingest._odds_cache_path(cache_key)
    import json
    from datetime import datetime, timedelta, timezone

    stale_time = datetime.now(timezone.utc) - timedelta(minutes=31)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "cache_key": cache_key,
                "fetched_at": stale_time.isoformat(),
                "data": stale_payload,
            },
            handle,
        )

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {}

    with patch("baseball_props.data.ingest.requests.get", return_value=mock_response), patch(
        "baseball_props.data.serpapi.fetch_from_serpapi",
        side_effect=SerpApiError("quota exhausted"),
    ), patch(
        "baseball_props.data.prop_odds.fetch_from_prop_odds",
        side_effect=PropOddsError("quota exhausted"),
    ):
        result = ingest.fetch_odds_api_json(
            cache_key,
            "https://example.com/odds",
            {"apiKey": "bad"},
        )

    assert result == stale_payload


def test_fetch_from_prop_odds_routes_player_props_cache_key() -> None:
    event_id = "abc123event456789012345678901234"
    cache_key = f"player_props:v3:{event_id}:batter_total_bases,pitcher_outs:draftkings,fanduel,betmgm"
    expected = {"id": event_id, "bookmakers": []}

    with patch(
        "baseball_props.data.prop_odds._fetch_event_props_from_prop_odds",
        return_value=expected,
    ) as mock_fetch:
        result = fetch_from_prop_odds(
            cache_key,
            "https://example.com",
            {},
            timeout=5.0,
        )

    mock_fetch.assert_called_once()
    assert result == expected
