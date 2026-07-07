import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from baseball_props.data import ingest
from baseball_props.data.odds_props import (
    ODDS_EVENT_ODDS_URL,
    fetch_batched_player_props,
)
from baseball_props.logging_utils import reset_log_once


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "ODDS_CACHE_DIR", tmp_path)
    reset_log_once()
    yield tmp_path
    reset_log_once()


def test_read_odds_cache_returns_fresh_data(cache_dir) -> None:
    cache_key = "test:live_vegas"
    payload = [{"id": "event1", "home_team": "A", "away_team": "B"}]
    ingest.write_odds_cache(cache_key, payload)

    assert ingest.read_odds_cache(cache_key) == payload


def test_read_odds_cache_expires_after_ttl(cache_dir, monkeypatch) -> None:
    cache_key = "test:expired"
    path = ingest._odds_cache_path(cache_key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stale_time = datetime.now(timezone.utc) - timedelta(minutes=31)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "cache_key": cache_key,
                "fetched_at": stale_time.isoformat(),
                "data": [{"id": "old"}],
            },
            handle,
        )

    assert ingest.read_odds_cache(cache_key) is None


def test_fetch_odds_api_json_uses_cache_without_second_request(cache_dir) -> None:
    cache_key = "test:fetch_once"
    ingest.write_odds_cache(cache_key, [{"id": "cached"}])

    with patch("baseball_props.data.ingest.requests.get") as mock_get:
        result = ingest.fetch_odds_api_json(
            cache_key,
            "https://example.com/odds",
            {"apiKey": "test"},
        )

    assert result == [{"id": "cached"}]
    mock_get.assert_not_called()


def test_fetch_batched_player_props_uses_per_event_endpoint() -> None:
    event_ids = [
        "abc123event456789012345678901234",
        "def456event789012345678901234567890",
    ]
    api_response = {
        "id": event_ids[0],
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "batter_total_bases",
                        "outcomes": [
                            {
                                "name": "Over",
                                "description": "Aaron Judge",
                                "price": -110,
                                "point": 1.5,
                            },
                            {
                                "name": "Under",
                                "description": "Aaron Judge",
                                "price": -110,
                                "point": 1.5,
                            },
                        ],
                    }
                ],
            }
        ],
    }

    with (
        patch(
            "baseball_props.data.therundown.fetch_rundown_batched_props",
            return_value=pd.DataFrame(
                columns=["game_id", "player_name", "market", "side", "line", "odds", "bookmaker"]
            ),
        ),
        patch(
            "baseball_props.data.odds_props.fetch_odds_api_json",
            return_value=api_response,
        ) as mock_fetch,
        patch(
            "baseball_props.data.odds_props._map_to_odds_api_event_ids",
            side_effect=lambda ids, **kwargs: {eid: eid for eid in ids},
        ),
    ):
        result = fetch_batched_player_props(event_ids)

    assert mock_fetch.call_count == 2
    for call in mock_fetch.call_args_list:
        _, url, params = call[0]
        assert "/events/" in url
        assert url.endswith("/odds")
        assert url != ingest.THE_ODDS_API_URL
        assert "eventIds" not in params
        assert params["markets"] == "batter_total_bases,batter_hits,pitcher_outs"
    assert ODDS_EVENT_ODDS_URL.format(event_id=event_ids[0]) in [
        call[0][1] for call in mock_fetch.call_args_list
    ]
    assert len(result) == 4
    assert (result["player_name"] == "Aaron Judge").all()


def test_fetch_live_vegas_totals_reads_cache(cache_dir, monkeypatch) -> None:
    monkeypatch.setattr(ingest, "get_odds_api_key", lambda **_: "test-key")
    games = [
        {
            "id": "abc123event456789012345678901234",
            "home_team": "New York Yankees",
            "away_team": "Boston Red Sox",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [{"name": "Over", "point": 8.5, "price": -110}],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "New York Yankees", "point": -1.5, "price": -110},
                                {"name": "Boston Red Sox", "point": 1.5, "price": -110},
                            ],
                        },
                    ],
                }
            ],
        }
    ]
    ingest.write_odds_cache(
        f"live_vegas_totals:{ingest.DEFAULT_SPORTSBOOK_KEYS}",
        games,
    )

    with (
        patch(
            "baseball_props.data.therundown.fetch_rundown_mlb_events",
            side_effect=RuntimeError("skip rundown in cache test"),
        ),
        patch("baseball_props.data.ingest.requests.get") as mock_get,
    ):
        df = ingest.fetch_live_vegas_totals()

    mock_get.assert_not_called()
    assert len(df) == 1
    assert df.iloc[0]["game_total"] == 8.5


def test_fetch_odds_api_json_401_uses_stale_cache(cache_dir) -> None:
    cache_key = "test:stale_on_401"
    stale_payload = [{"id": "stale-event", "home_team": "A", "away_team": "B"}]
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
    mock_response.json.return_value = {"message": "Invalid API key"}

    from baseball_props.data.prop_odds import PropOddsError
    from baseball_props.data.serpapi import SerpApiError

    with patch("baseball_props.data.ingest.requests.get", return_value=mock_response), patch(
        "baseball_props.data.serpapi.fetch_from_serpapi",
        side_effect=SerpApiError("unavailable"),
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


def test_fetch_odds_api_json_401_uses_offline_fallback_when_no_cache(cache_dir) -> None:
    cache_key = f"live_vegas_totals:{ingest.DEFAULT_SPORTSBOOK_KEYS}"

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {"x-requests-remaining": "0"}
    mock_response.json.return_value = {"message": "Usage quota has been reached"}

    from baseball_props.data.prop_odds import PropOddsError
    from baseball_props.data.serpapi import SerpApiError

    with patch("baseball_props.data.ingest.requests.get", return_value=mock_response), patch(
        "baseball_props.data.serpapi.fetch_from_serpapi",
        side_effect=SerpApiError("quota exhausted"),
    ), patch(
        "baseball_props.data.prop_odds.fetch_from_prop_odds",
        side_effect=PropOddsError("quota exhausted"),
    ):
        result = ingest.fetch_odds_api_json(
            cache_key,
            ingest.THE_ODDS_API_URL,
            {"apiKey": "exhausted"},
        )

    assert result == []


def test_fetch_live_vegas_totals_survives_401_offline(cache_dir, monkeypatch) -> None:
    monkeypatch.setattr(ingest, "get_odds_api_key", lambda **_: "test-key")

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {}
    mock_response.json.return_value = {"message": "Invalid API key"}

    from baseball_props.data.prop_odds import PropOddsError
    from baseball_props.data.serpapi import SerpApiError

    with patch("baseball_props.data.ingest.requests.get", return_value=mock_response), patch(
        "baseball_props.data.serpapi.fetch_from_serpapi",
        side_effect=SerpApiError("unavailable"),
    ), patch(
        "baseball_props.data.prop_odds.fetch_from_prop_odds",
        side_effect=PropOddsError("unavailable"),
    ):
        df = ingest.fetch_live_vegas_totals()

    assert df.empty


def test_fetch_odds_api_json_401_logs_fallback_warning_once(cache_dir, caplog) -> None:
    from baseball_props.data.prop_odds import PropOddsError
    from baseball_props.data.serpapi import SerpApiError

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {}
    mock_response.json.return_value = {"message": "Invalid API key"}

    cache_keys = [f"player_props:v3:event{i}:m:books" for i in range(5)]

    with caplog.at_level("WARNING"), patch(
        "baseball_props.data.ingest.requests.get",
        return_value=mock_response,
    ), patch(
        "baseball_props.data.serpapi.fetch_from_serpapi",
        side_effect=SerpApiError("unavailable"),
    ), patch(
        "baseball_props.data.prop_odds.fetch_from_prop_odds",
        side_effect=PropOddsError("unavailable"),
    ):
        for cache_key in cache_keys:
            ingest.fetch_odds_api_json(cache_key, "https://example.com/odds", {"apiKey": "bad"})

    primary_warnings = [
        record
        for record in caplog.records
        if record.levelname == "WARNING"
        and "Primary API unauthorized" in record.message
    ]
    serpapi_failed = [
        record
        for record in caplog.records
        if record.levelname == "WARNING" and "SerpApi fallback unavailable" in record.message
    ]
    prop_odds_failed = [
        record
        for record in caplog.records
        if record.levelname == "WARNING" and "Prop-Odds fallback unavailable" in record.message
    ]

    assert len(primary_warnings) == 1
    assert len(serpapi_failed) == 1
    assert len(prop_odds_failed) == 1
