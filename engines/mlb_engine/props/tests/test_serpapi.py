import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from baseball_props.data import ingest
from baseball_props.data.serpapi import (
    SerpApiError,
    adapt_serpapi_sports_results,
    adapt_serpapi_vegas_games,
    fetch_from_serpapi,
)
from baseball_props.logging_utils import reset_log_once


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "ODDS_CACHE_DIR", tmp_path)
    reset_log_once()
    yield tmp_path
    reset_log_once()


def _sample_serpapi_payload() -> dict:
    return {
        "sports_results": {
            "title": "MLB",
            "games": [
                {
                    "teams": [
                        {"name": "Boston Red Sox", "score": "3"},
                        {"name": "New York Yankees", "score": "5"},
                    ],
                    "status": "FT",
                    "date": "Jun 27",
                }
            ],
        }
    }


def test_adapt_serpapi_sports_results_builds_events() -> None:
    events = adapt_serpapi_sports_results(_sample_serpapi_payload())
    assert len(events) == 1
    assert events[0]["home_team"] == "New York Yankees"
    assert events[0]["away_team"] == "Boston Red Sox"
    assert len(events[0]["id"]) >= 16


def test_adapt_serpapi_vegas_games_includes_neutral_markets() -> None:
    events = adapt_serpapi_sports_results(_sample_serpapi_payload())
    vegas = adapt_serpapi_vegas_games(events)
    assert len(vegas) == 1
    markets = {m["key"] for m in vegas[0]["bookmakers"][0]["markets"]}
    assert markets == {"totals", "spreads"}
    totals = next(m for m in vegas[0]["bookmakers"][0]["markets"] if m["key"] == "totals")
    assert totals["outcomes"][0]["point"] == 9.0


def test_fetch_from_serpapi_player_props_uses_stale_cache(cache_dir) -> None:
    cache_key = (
        "player_props:v3:abc123event456789012345678901234:"
        "batter_total_bases,pitcher_outs:draftkings,fanduel,betmgm"
    )
    stale_payload = {
        "id": "abc123event456789012345678901234",
        "bookmakers": [{"key": "draftkings", "markets": []}],
    }
    ingest.write_odds_cache(cache_key, stale_payload)
    path = ingest._odds_cache_path(cache_key)
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

    with patch("baseball_props.data.serpapi._serpapi_search") as mock_search:
        result = fetch_from_serpapi(cache_key, "https://example.com", {}, timeout=5.0)

    mock_search.assert_not_called()
    assert result == stale_payload


def test_fetch_from_serpapi_mlb_events_routes_search() -> None:
    with patch(
        "baseball_props.data.serpapi._fetch_mlb_events_from_serpapi",
        return_value=[{"id": "event1", "home_team": "A", "away_team": "B"}],
    ) as mock_fetch:
        result = fetch_from_serpapi("mlb_events", "https://example.com", {}, timeout=5.0)

    mock_fetch.assert_called_once()
    assert result[0]["id"] == "event1"


def test_fetch_odds_api_json_401_uses_serpapi_before_prop_odds(cache_dir) -> None:
    cache_key = "test:serpapi_first"
    serp_payload = [{"id": "from-serpapi", "home_team": "A", "away_team": "B"}]

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {}

    with patch("baseball_props.data.ingest.requests.get", return_value=mock_response), patch(
        "baseball_props.data.serpapi.fetch_from_serpapi",
        return_value=serp_payload,
    ) as mock_serp, patch(
        "baseball_props.data.prop_odds.fetch_from_prop_odds",
    ) as mock_prop:
        result = ingest.fetch_odds_api_json(
            cache_key,
            "https://example.com/odds",
            {"apiKey": "bad"},
        )

    mock_serp.assert_called_once()
    mock_prop.assert_not_called()
    assert result == serp_payload


def test_fetch_odds_api_json_401_falls_through_to_prop_odds(cache_dir) -> None:
    cache_key = "test:prop_odds_second"
    prop_payload = [{"id": "from-prop-odds"}]

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {}

    with patch("baseball_props.data.ingest.requests.get", return_value=mock_response), patch(
        "baseball_props.data.serpapi.fetch_from_serpapi",
        side_effect=SerpApiError("quota exhausted"),
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


def test_fetch_odds_api_json_401_uses_stale_after_both_fallbacks_fail(cache_dir) -> None:
    cache_key = "test:stale_after_both"
    stale_payload = [{"id": "stale-event"}]
    path = ingest._odds_cache_path(cache_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
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

    from baseball_props.data.prop_odds import PropOddsError

    with patch("baseball_props.data.ingest.requests.get", return_value=mock_response), patch(
        "baseball_props.data.serpapi.fetch_from_serpapi",
        side_effect=SerpApiError("no events"),
    ), patch(
        "baseball_props.data.prop_odds.fetch_from_prop_odds",
        side_effect=PropOddsError("unavailable"),
    ):
        result = ingest.fetch_odds_api_json(
            cache_key,
            "https://example.com/odds",
            {"apiKey": "bad"},
        )

    assert result == stale_payload
